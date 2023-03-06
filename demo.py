# Demo script
# given demo input scene view from 3D-FRONT, output scene 3D predicted bounding boxes and the 3D scene reconstruction

# Modified code of https://github.com/yinyunie/Total3DUnderstanding

from net_utils.utils import load_device, load_model
from net_utils.utils import CheckpointIO
from configs.config_utils import mount_external_config
import numpy as np
import torch
from torchvision import transforms
import os
from time import time
from PIL import Image
import json
import math
from configs.data_config import Relation_Config, NYU40CLASSES, NYU37_TO_PIX3D_CLS_MAPPING
rel_cfg = Relation_Config()
d_model = int(rel_cfg.d_g/4)
from models.total3d.dataloader import collate_fn

HEIGHT_PATCH = 256
WIDTH_PATCH = 256

data_transforms = transforms.Compose([
    transforms.Resize((HEIGHT_PATCH, WIDTH_PATCH)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def parse_detections(detections):
    bdb2D_pos = []
    size_cls = []
    for det in detections:
        bdb2D_pos.append(det['bbox'])
        size_cls.append(NYU40CLASSES.index(det['class']))
    return bdb2D_pos, size_cls

def get_g_features(bdb2D_pos):
    n_objects = len(bdb2D_pos)
    g_feature = [[((loc2[0] + loc2[2]) / 2. - (loc1[0] + loc1[2]) / 2.) / (loc1[2] - loc1[0]),
                  ((loc2[1] + loc2[3]) / 2. - (loc1[1] + loc1[3]) / 2.) / (loc1[3] - loc1[1]),
                  math.log((loc2[2] - loc2[0]) / (loc1[2] - loc1[0])),
                  math.log((loc2[3] - loc2[1]) / (loc1[3] - loc1[1]))] \
                 for id1, loc1 in enumerate(bdb2D_pos)
                 for id2, loc2 in enumerate(bdb2D_pos)]

    locs = [num for loc in g_feature for num in loc]

    pe = torch.zeros(len(locs), d_model)
    position = torch.from_numpy(np.array(locs)).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe.view(n_objects * n_objects, rel_cfg.d_g)

def load_demo_data(demo_path, device):
    img_path = os.path.join(demo_path, 'img.jpg')
    assert os.path.exists(img_path)

    cam_K_path = os.path.join(demo_path, 'cam_K.txt')
    assert os.path.exists(cam_K_path)

    detection_path = os.path.join(demo_path, 'detections.json')
    assert detection_path

    '''preprocess'''
    image = Image.open(img_path).convert('RGB')
    cam_K = np.loadtxt(cam_K_path)
    with open(detection_path, 'r') as file:
        detections = json.load(file)
    boxes = dict()

    bdb2D_pos, size_cls = parse_detections(detections)

    # obtain geometric features
    boxes['g_feature'] = get_g_features(bdb2D_pos)

    # encode class
    cls_codes = torch.zeros([len(size_cls), len(NYU40CLASSES)])
    cls_codes[range(len(size_cls)), size_cls] = 1
    boxes['size_cls'] = cls_codes

    # get object images
    id=demo_path.replace('demo/inputs/', '')
    print('id is ', id)
    patch = []
    patch_jid=[]
    patch_mask_jid=[]
    for i,bdb in enumerate(bdb2D_pos):
        img_o = image.crop((bdb[0], bdb[1], bdb[2], bdb[3]))
        img = data_transforms(img_o)
        newsize = (256, 256)
        img_jid=img_o.resize(newsize)
        patch_jid.append(img_jid)
        path_to_mask=f'/mnt/hdd/ayadi/prepare_data/mask/rendertask{id}_{i}.png'
        patch.append(img)
        img_mask = Image.open(path_to_mask) #.convert('1') #.astype(np.uint8) * 255
        cropped_image = img_mask.crop((bdb[0], bdb[1], bdb[2], bdb[3]))
        cropped_image = cropped_image.resize(newsize)
        patch_mask_jid.append(cropped_image)

    boxes['patch'] = torch.stack(patch)
    boxes['patch_jid']= patch_jid
    boxes['patch_mask_jid']= patch_mask_jid #torch.stack(patch_mask_jid)

    image = data_transforms(image)
    camera = dict()
    camera['K'] = cam_K
    boxes['bdb2D_pos'] = np.array(bdb2D_pos)

    """assemble data"""
    data = collate_fn([{'image':image, 'boxes_batch':boxes, 'camera':camera}])
    image = data['image'].to(device)
    K = data['camera']['K'].float().to(device)
    patch = data['boxes_batch']['patch'].to(device)
    size_cls = data['boxes_batch']['size_cls'].float().to(device)
    g_features = data['boxes_batch']['g_feature'].float().to(device)
    split = data['obj_split']
    rel_pair_counts = torch.cat([torch.tensor([0]), torch.cumsum(
        torch.pow(data['obj_split'][:, 1] - data['obj_split'][:, 0], 2), 0)], 0)
    cls_codes = torch.zeros([size_cls.size(0), 9]).to(device)
    cls_codes[range(size_cls.size(0)), [NYU37_TO_PIX3D_CLS_MAPPING[cls.item()] for cls in
                                        torch.argmax(size_cls, dim=1)]] = 1
    bdb2D_pos = data['boxes_batch']['bdb2D_pos'].float().to(device)

    input_data = {'image':image, 'K':K, 'patch':patch, 'patch_jid':patch_jid , 'patch_mask_jid': patch_mask_jid, 'patch_for_mesh':patch, 'g_features':g_features,
                  'size_cls':size_cls, 'split':split, 'rel_pair_counts':rel_pair_counts,
                  'cls_codes':cls_codes, 'bdb2D_pos':bdb2D_pos}
    return input_data

def run(cfg):
    '''Begin to run network.'''
    checkpoint = CheckpointIO(cfg)

    '''Mount external config data'''
    cfg = mount_external_config(cfg)

    '''Load save path'''
    cfg.log_string('Data save path: %s' % (cfg.save_path))

    '''Load device'''
    cfg.log_string('Loading device settings.')
    device = load_device(cfg)

    '''Load net'''
    cfg.log_string('Loading model.')
    net = load_model(cfg, device=device)
    checkpoint.register_modules(net=net)
    cfg.log_string(net)

    '''Load existing checkpoint'''
    checkpoint.parse_checkpoint()
    cfg.log_string('-' * 100)

    '''Load data'''
    cfg.log_string('Loading data.')
    data = load_demo_data(cfg.config['demo_path'], device)

    '''Run demo'''
    net.train(cfg.config['mode'] == 'train')
    with torch.no_grad():
        start = time()
        est_data = net(data)
        end = time()

    print('Time elapsed: %s.' % (end-start))

    '''write and visualize outputs'''
    from net_utils.libs import get_layout_bdb_3dfront,get_bdb_evaluation , get_rotation_matix_result
    from utils.qual_util import load_resnet2vq_model, preprocess_img
    from PIL import Image
    import matplotlib.image
    import pytorch3d.io.obj_io
    from scipy.io import savemat
    from utils.demo_util import get_shape_comp_opt, get_shape_comp_model
    from utils.qual_util import load_resnet2vq_model
    from utils.util_3d import  sdf_to_mesh

    lo_bdb3D_out = get_layout_bdb_3dfront(cfg.bins_tensor, est_data['lo_ori_reg_result'],
                                          torch.argmax(est_data['lo_ori_cls_result'], 1),
                                          est_data['lo_centroid_result'],
                                          est_data['lo_coeffs_result'])
    # camera orientation for evaluation
    cam_R_out = get_rotation_matix_result(cfg.bins_tensor,
                                          torch.argmax(est_data['pitch_cls_result'], 1), est_data['pitch_reg_result'],
                                          torch.argmax(est_data['roll_cls_result'], 1), est_data['roll_reg_result'])

    # projected center
    P_result = torch.stack(((data['bdb2D_pos'][:, 0] + data['bdb2D_pos'][:, 2]) / 2 -
                            (data['bdb2D_pos'][:, 2] - data['bdb2D_pos'][:, 0]) * est_data['offset_2D_result'][:, 0],
                            (data['bdb2D_pos'][:, 1] + data['bdb2D_pos'][:, 3]) / 2 -
                            (data['bdb2D_pos'][:, 3] - data['bdb2D_pos'][:, 1]) * est_data['offset_2D_result'][:,1]), 1)

    bdb3D_out_form_cpu, bdb3D_out = get_bdb_evaluation(cfg.bins_tensor,
                                                       torch.argmax(est_data['ori_cls_result'], 1),
                                                       est_data['ori_reg_result'],
                                                       torch.argmax(est_data['centroid_cls_result'], 1),
                                                       est_data['centroid_reg_result'],
                                                       data['size_cls'], est_data['size_reg_result'], P_result,
                                                       data['K'], cam_R_out, data['split'], return_bdb=True)

    # save results
    nyu40class_ids = [int(evaluate_bdb['classid']) for evaluate_bdb in bdb3D_out_form_cpu]
    save_path = cfg.config['demo_path'].replace('inputs', 'outputs')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # save layout
    savemat(os.path.join(save_path, 'layout.mat'),
            mdict={'layout': lo_bdb3D_out[0, :, :].cpu().numpy()})
    # save bounding boxes and camera poses
    interval = data['split'][0].cpu().tolist()
    current_cls = nyu40class_ids[interval[0]:interval[1]]

    savemat(os.path.join(save_path, 'bdb_3d.mat'),
            mdict={'bdb': bdb3D_out_form_cpu[interval[0]:interval[1]], 'class_id': current_cls})
    savemat(os.path.join(save_path, 'r_ex.mat'),
            mdict={'cam_R': cam_R_out[0, :, :].cpu().numpy()})
    
    opt = get_shape_comp_opt(gpu_id=0)
    model = get_shape_comp_model(opt)    
    model.eval()
    resnet2vq = load_resnet2vq_model(opt)
    for obj_id, obj_cls in enumerate(current_cls):
        file_path = os.path.join(save_path, '%s_%s.obj' % (obj_id, obj_cls))
        img_input=data['patch_jid'][obj_id]
        img_mask=data['patch_mask_jid'][obj_id]
        img_input= preprocess_img(img_input, img_mask)
        single_view_recon = model.single_view_recon_dummy(img_input, resnet2vq, bs=1, topk=5, alpha=0.8)
        gen_mesh = sdf_to_mesh(single_view_recon)
        for i in range(1):
            mesh_path=os.path.join(file_path)
            faces=gen_mesh[i].faces_list()[0]
            vertices=gen_mesh[i].verts_list()[0]
            pytorch3d.io.obj_io.save_obj(mesh_path, vertices, faces)

    #########################################################################
    #
    #   Visualization
    #
    #########################################################################
    import scipy.io as sio
    from utils.visualize import format_bbox, format_layout, format_mesh, Box, draw_projected_bdb3d_GT
    from glob import glob

    pre_layout_data = sio.loadmat(os.path.join(save_path, 'layout.mat'))['layout']
    pre_box_data = sio.loadmat(os.path.join(save_path, 'bdb_3d.mat'))

    pre_boxes = format_bbox(pre_box_data, 'prediction')
    
    pre_layout = format_layout(pre_layout_data)
    pre_cam_R = sio.loadmat(os.path.join(save_path, 'r_ex.mat'))['cam_R']

    vtk_objects, pre_boxes = format_mesh(glob(os.path.join(save_path, '*.obj')), pre_boxes)

    image = np.array(Image.open(os.path.join(cfg.config['demo_path'], 'img.jpg')).convert('RGB'))
    cam_K = np.loadtxt(os.path.join(cfg.config['demo_path'], 'cam_K.txt'))

    scene_box = Box(image, None, cam_K, None, pre_cam_R, None, pre_layout, None, pre_boxes, 'prediction', output_mesh = vtk_objects, sample_data=data, save_path=save_path, sequence_id=3001)
    scene_box.draw3D(if_save=True, save_path = '%s/recon.png' % (save_path))
