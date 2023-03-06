# Modified implementation of Resnet2Vq from https://github.com/yccyenchicheng/AutoSDF 
# Includes baselines Resnet2VOX, Resnet2TSDF
import os
from collections import OrderedDict
from models.rand_tf_model import RandTransformerModel

import numpy as np
import einops
import mcubes
from omegaconf import OmegaConf
from termcolor import colored, cprint
from einops import rearrange, repeat
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn, optim

import torchvision.utils as vutils
import torchvision.transforms as transforms
from pytorch3d.structures import Meshes

from pytorch3d.loss import chamfer_distance

from pytorch3d.ops import sample_points_from_meshes
from utils.util import sample_pnts_from_obj, normalize_to_unit_square, write_obj, read_obj

from models.base_model import BaseModel
from models.networks.resnet2vq_net import ResNet2VQ
from models.networks.resnet2tsdf_net import ResNet2TSDF
from models.networks.resnet2vox_net import ResNet2Vox
from models.networks.resnet2mv import ResNet2MV
from models.networks.pvqvae_networks.auto_encoder import PVQVAE
from models.networks.transformer_networks.rand_transformer import RandTransformer
from models.base_model import create_model

import utils
from utils.pix3d_util import downsample_voxel
from utils.util_3d import init_mesh_renderer, render_mesh, render_sdf, render_voxel, dict_map

class Opt:
    def __init__(self):
        self.name = 'opt'

class ResNet2VQModel(BaseModel):
    def name(self):
        return 'ResNet2VQ-Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.note=opt.note
        self.isTrain = opt.isTrain
        self.model_name = self.name()

        # -------------------------------
        # Define Networks
        # -------------------------------

        assert opt.vq_cfg is not None
        assert opt.tf_cfg is not None

        resnet_conf = OmegaConf.load(opt.resnet_cfg)
        vq_conf = OmegaConf.load(opt.vq_cfg)
        tf_conf = OmegaConf.load(opt.tf_cfg)

        self.pe_conf = tf_conf.pe
        
        # init resnet2vq network
        if self.note=='resnet2vq':
            print('resnet2vq')
            self.net = ResNet2VQ(opt)
        elif self.note=='resnet2TSDF':
            print('ResNet2TSDF')
            self.net = ResNet2TSDF(opt)
        elif self.note=='resnet2VOX':
            print('ResNet2Vox')
            self.net = ResNet2Vox(opt)
        elif self.note=='resnet2MV':
            self.net = ResNet2MV(opt)

        


        self.net.to(opt.device)
        # init vqvae for decoding shapes
        mparam = vq_conf.model.params
        n_embed = mparam.n_embed
        embed_dim = mparam.embed_dim
        ddconfig = mparam.ddconfig
        self.vqvae = PVQVAE(ddconfig, n_embed, embed_dim)
        self.load_vqvae(opt.vq_ckpt)
        self.vqvae.to(opt.device)
        self.vqvae.eval()

        self.rand_tf=RandTransformerModel()
        self.rand_tf.initialize(opt)
        self.load_rand_tf(opt.ckpt)
        self.rand_tf.tf.to(opt.device)
        self.rand_tf.tf.eval()
        self.rand_tf.eval()

        if self.isTrain:
            # ----------------------------------
            # define loss functions
            # ----------------------------------
            self.criterion_nll = nn.NLLLoss()
            self.criterion_nce = nn.CrossEntropyLoss()
            self.criterion_nce.to(opt.device)

            self.criterion_l1= nn.L1Loss()
            self.criterion_l1.to(opt.device)

            self.criterion_bce=nn.BCELoss()
            self.criterion_bce.to(opt.device)

            # ---------------------------------
            # initialize optimizers
            # ---------------------------------
            self.optimizer = optim.Adam([p for p in self.net.parameters() if p.requires_grad == True], lr=opt.lr)

            if opt.debug == '1':
                self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 100 if opt.dataset_mode == 'imagenet' else 30, 1.)
            else:
                self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 100 if opt.dataset_mode == 'imagenet' else 5, 0.9)

            self.optimizers = [self.optimizer]
            self.schedulers = [self.scheduler]

            self.print_networks(verbose=False)

        # transforms
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        resolution = resnet_conf.data.resolution
        self.resolution = resolution

        # hyper-parameters for SDF
        if opt.dataset_mode in ['sdf', 'sdf_code', 'sdf_code_v0']:
            nC = resolution
            assert nC == 64, 'right now, only trained with sdf resolution = 64'
            self.down_size = 8   # x: res, x_cube: res//8
            self.cube_size = nC // self.down_size    # size per cube. nC=64, down_size=8 -> size=8 for each smaller cube
            self.stride = nC // self.down_size
            self.ncubes_per_dim = nC // self.cube_size

        # grid size
        if opt.vq_note == 'default':
            self.grid_size = 8
        elif opt.vq_note == '4x4x4':
            self.grid_size = 4
        elif opt.vq_note == '16x16x16':
            self.grid_size = 16


        # setup renderer
        dist, elev, azim = 1.7, 20, 20   
        self.renderer = init_mesh_renderer(image_size=256, dist=dist, elev=elev, azim=azim, device=self.opt.device)

        # for saving best ckpt
        self.best_iou = -1e12

    def load_vqvae(self, vq_ckpt):
        assert type(vq_ckpt) == str         
        state_dict = torch.load(vq_ckpt)

        self.vqvae.load_state_dict(state_dict)
        print(colored('[*] VQVAE: weight successfully load from: %s' % vq_ckpt, 'blue'))

    def load_rand_tf(self, ckpt):

        if type(ckpt) == str:
            state_dict = torch.load(ckpt)
        else:
            state_dict = ckpt

        self.vqvae.load_state_dict(state_dict['vqvae'])
        self.rand_tf.tf.load_state_dict(state_dict['tf'])
        print(colored('[*] weight successfully load from: %s' % ckpt, 'blue'))
    
    def get_rand_tf_model(self,opt_tf, opt):

            opt_tf.model='rand_tf'
            opt_tf.tf_cfg=opt.tf_cfg
            opt_tf.ckpt = opt.ckpt

            # load vq stuff
            opt_tf.vq_model='pvqvae'
            opt_tf.vq_cfg=opt.vq_cfg
            opt_tf.vq_ckpt=opt.vq_ckpt
            opt_tf.vq_note='default'
            
            ### opt.vq_dset='sdf_code' # original
            opt_tf.vq_dset= opt.vq_dset
            model = create_model(opt_tf)
            print(f'[*] "{opt_tf.model}" initialized.')
            model.load_ckpt(opt_tf.ckpt)
            return model

    def get_shape_comp_opt(self, opt):
        opt_tf = Opt()

        # args
        gpuid=[0]
        name='test_transformer'
        # default args
        opt_tf.serial_batches = False
        opt_tf.nThreads = 4

        # important args
        opt_tf.dataset_mode = 'shapenet_code'
        opt_tf.seed = 111
        opt_tf.isTrain = False
        opt_tf.gpu_ids = gpuid
        opt_tf.device = self.opt.device
        opt_tf.batch_size = opt.batch_size
        opt_tf.max_dataset_size = opt.max_dataset_size

        opt_tf.name = name

        utils.util.seed_everything(opt_tf.seed)
        opt_tf.phase = 'test'
        return opt_tf
    def get_gen_order(self, sz, device):
        # return torch.randperm(sz).to(device)
        return torch.randperm(sz, device=device)
    
    def set_input(self, input, gen_order=None):
        # x, y = input
        self.x = input['sdf']
        self.cat_str=input['cat_str']
        self.cat_id=list(map(int, input['cat_id']))

        ll=[torch.tensor(dict_map[self.cat_id[i]]) for i,j in enumerate(self.cat_id)]
        self.gt_cls=torch.stack(ll)
        #self.one_hot=torch.nn.functional.one_hot(ll, num_classes=9) #(bs, 9)
        self.x_idx = input['idx']
        self.z_q = input['z_q']
        bs, dz, hz, wz = self.x_idx.shape
        self.z_shape = self.z_q.shape

        if self.note=='resnet2MV':
            self.img = input['imgs']
        else:
            self.img = input['img']


        # only use in test_iou
        self.gt_vox = input['gt_vox']

        self.x_idx_seq = rearrange(self.x_idx, 'bs dz hz wz -> (dz hz wz) bs').contiguous() # to (T, B)
        self.x_idx = self.x_idx_seq.clone()
        vars_list = ['x_idx', 'z_q', 'x', 'img', 'gt_vox']

        self.tocuda(var_names=vars_list)

    def forward(self):          
        self.outp=self.net(self.img)


    def inference(self, data, recon_tf=False, should_render=False, verbose=False):
        self.net.eval()
        
        self.set_input(data)
        with torch.no_grad():
            self.x = self.x
            self.x_recon = self.vqvae.decode(self.z_q) # could extract this as well

            outp = self.net(self.img)
            outp_sdf= outp
            outp = F.softmax(outp, dim=1) # compute the prob. of next ele
            outp = outp.argmax(dim=1)
            outp = rearrange(outp, 'bs d h w -> (d h w) bs')
            
            if self.note== 'resnet2TSDF':
                self.x_recon_resnet = outp_sdf
            elif self.note== 'resnet2VOX':
                self.x_recon_resnet = outp_sdf #>= 0.5
            elif self.note=='resnet2vq' or self.note=='resnet2MV':
                self.x_recon_resnet = self.vqvae.decode_enc_idices(outp, z_spatial_dim=self.grid_size)
            if recon_tf==True:
                self.x_recon_rand_tf=torch.zeros(self.x.shape).to(self.opt.device)
                for i in range(self.img.shape[0]):
                    x_recon_rand_tf_i= self.rand_tf.single_view_recon(
                        img_tensor=self.img[i].unsqueeze(0), 
                        sdf_tensor=self.x[i].unsqueeze(0), 
                        idx_tensor=self.x_idx[i].unsqueeze(0),
                        zq_tensor= self.z_q[i].unsqueeze(0), 
                        vox_tensor=self.gt_vox[i].unsqueeze(0),
                        resnet2vq=self.net ,bs=9, topk=10, alpha=0.75)
                    best_iou_i= utils.util.iou(self.x[i].repeat(9,1,1,1,1), x_recon_rand_tf_i, 0.0)
                    x_recon_rand_tf_i=x_recon_rand_tf_i[torch.argmax(best_iou_i)]

                    self.x_recon_rand_tf[i]=x_recon_rand_tf_i #.squeeze(0)
            else:
                self.x_recon_rand_tf=None

                
            self.img = self.img
            
            if should_render:
                if self.note=='resnet2vq' or self.note== 'resnet2TSDF' or self.note=='resnet2VOX':
                    self.image, self.pix3d1 = render_sdf(self.renderer, self.x)
                    self.image_recon_resnet, self.pix3d2 = render_sdf(self.renderer, self.x_recon_resnet)




        self.net.train()

    def compute_cd(self, pix3d1, pix3d2):

        bs=len(self.pix3d1.faces_list())
        list_mesh_gt=[]
        list_mesh_recon=[]            
        for i in range(bs):
            try:
                    mesh_obj = {'v': pix3d1[i].verts_list()[0].cpu().numpy(),
                                'f': pix3d1[i].faces_list()[0].cpu().numpy()}
                    write_obj('pix3d1_gt.obj', mesh_obj)
                    obj_data = read_obj('pix3d1_gt.obj', ['v', 'f'])
                    os.remove('pix3d1_gt.obj')
                    sampled_points = sample_pnts_from_obj(obj_data, n_pnts=10000, mode='random')
                    sampled_points = normalize_to_unit_square(sampled_points)[0]
                    list_mesh_gt.append(sampled_points)
            except:
                    continue

            try:
                    mesh_obj = {'v': pix3d2[i].verts_list()[0].cpu().numpy(),
                                'f': pix3d2[i].faces_list()[0].cpu().numpy()}
                    write_obj('pix3d2_gt.obj', mesh_obj)
                    obj_data = read_obj('pix3d2_gt.obj',  flags=['v', 'f'])
                    os.remove('pix3d2_gt.obj')
                    sampled_points = sample_pnts_from_obj(obj_data, n_pnts=10000, mode='random')
                    sampled_points = normalize_to_unit_square(sampled_points)[0]
                    list_mesh_recon.append(sampled_points)
            except:
                    continue
            
        arr_pix3d1=np.array(list_mesh_gt)
        arr_pix3d2=np.array(list_mesh_recon)
        cd=torch.zeros((bs,))
        if arr_pix3d1.shape==arr_pix3d2.shape:
            cd=utils.util.cd(arr_pix3d1, arr_pix3d2)
            
        return cd

    
    def test_iou(self, data, thres=0.0):
        """
            thres: threshold to consider a voxel to be free space or occupied space.
        """

        # self.set_input(data)

        self.net.eval()
        self.inference(data, recon_tf=False, should_render=True) 
        self.net.train()

        if self.opt.note=='resnet2VOX':
            x=self.gt_vox
            cat=torch.tensor(self.cat_id)
            x_recon_resnet = self.x_recon_resnet
        else:
            x = self.x
            cat=torch.tensor(self.cat_id)
            x_recon_resnet = self.x_recon_resnet
        
        iou = utils.util.iou(x, x_recon_resnet, thres) 
        cd=self.compute_cd(self.pix3d1, self.pix3d2)

        return iou , cd


    def eval_metrics(self,dataloader, thres=0.0):
        self.eval()

        iou_list = []
        cd_list = []
        with torch.no_grad():
            for ix, test_data in tqdm(enumerate(dataloader), total=len(dataloader)):
                    iou, cd = self.test_iou_orig(test_data, thres=thres)
                    cd_list.append(cd.detach())
                    iou_list.append(iou.detach())

        iou = torch.cat(iou_list)
        iou_mean, iou_std = iou.mean(), iou.std()

        cd=torch.cat(cd_list)
        cd_mean, cd_std= cd.mean(), cd.std()
        
        ret = OrderedDict([
            ('iou', iou_mean.data),
            ('iou_std', iou_std.data),
            ('cd', cd_mean.data),
            ('cd_std ', cd_std.data),
        ])

        # check whether to save best epoch
        if ret['iou'] > self.best_iou:
            self.best_iou = ret['iou']
            save_name = f'epoch-best'
            self.save(save_name)
    
        self.train()
        return ret


    def backward(self):
        if self.note=='resnet2TSDF':
            target=self.x
            outp= self.outp
            outp = torch.clamp(outp,-0.2, 0.2)

            loss=self.criterion_l1(outp, target)
        
        elif self.note=='resnet2VOX':
            target=self.gt_vox

            outp= self.outp
            loss=self.criterion_bce(outp, target.unsqueeze(1))

        
        elif self.note=='resnet2vq' or self.note=='resnet2MV':
            target = self.x_idx 
            target = rearrange(target, '(d h w) bs -> bs d h w', d=self.grid_size, h=self.grid_size, w=self.grid_size)
            outp = self.outp 
            loss_nce = self.criterion_nce(outp, target)
            loss=loss_nce

        self.loss_nll=loss
        self.loss = loss
        self.loss.backward()

    def optimize_parameters(self, total_steps):
        # self.vqvae.train()

        self.set_requires_grad([self.net], requires_grad=True)

        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def get_current_errors(self):
        
        ret = OrderedDict([
            ('nll', self.loss_nll.data)
            #('cls_loss', self.loss_cls.data),

        ])

        return ret

    def get_current_visuals(self):

        with torch.no_grad():
            self.pix3d3 =None
            self.pix3d2=None
            # self.image = self.render(self.x)
            if self.note=='resnet2vq' or self.note== 'resnet2TSDF' or self.note=='resnet2MV':
                self.image_recon, self.pix3d1 = render_sdf(self.renderer, self.x_recon)
                self.image_recon_resnet, self.pix3d2 = render_sdf(self.renderer, self.x_recon_resnet)
            else:
                self.image_recon, self.pix3d1 = render_voxel(self.renderer, self.gt_vox)
                try:
                    self.image_recon_resnet, self.pix3d2 = render_voxel(self.renderer, self.x_recon_resnet.squeeze())
                except:
                    pass

            if not(self.x_recon_rand_tf is None):
                self.image_recon_rand_tf, self.pix3d3 = render_sdf(self.renderer, self.x_recon_rand_tf)
            # self.image_recon_gt = self.render(self.x_recon_gt)
            if self.note=='resnet2MV':
                self.image = self.img[:,0,:,:,:]
            else:
                self.image=self.img
            pass

        if self.x_recon_rand_tf is None:

            vis_tensor_names = [
                'image',
                'image_recon',
                'image_recon_resnet',
            ]
        else:
        
            vis_tensor_names = [
                'image',
                'image_recon',
                'image_recon_resnet',
                'image_recon_rand_tf'
            ]

        vis_ims = self.tnsrs2ims(vis_tensor_names)
        visuals = zip(vis_tensor_names, vis_ims)
                            
        return OrderedDict(visuals), self.pix3d1, self.pix3d2, self.pix3d3


    def save(self, label):

        state_dict = {
            'vqvae': self.vqvae.cpu().state_dict(),
            'resnet2vq': self.net.cpu().state_dict(),
        }

        save_filename = 'resnet2vq_%s.pth' % (label)
        save_path = os.path.join(self.save_dir, save_filename)

        torch.save(state_dict, save_path)
        self.vqvae.to(self.opt.device)
        self.net.to(self.opt.device)

    def load_ckpt(self, ckpt):
        if type(ckpt) == str:
            state_dict = torch.load(ckpt)
        else:
            state_dict = ckpt

        self.vqvae.load_state_dict(state_dict['vqvae'])
        self.net.load_state_dict(state_dict['resnet2vq'])
        print(colored('[*] weight successfully load from: %s' % ckpt, 'blue'))
