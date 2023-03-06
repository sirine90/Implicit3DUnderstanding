# Modified implementation from https://github.com/yccyenchicheng/AutoSDF
# Functions used for autoregressive prior training 

from __future__ import print_function
import os
import random
import numpy as np
from PIL import Image
from einops import rearrange
import torch
import torchvision.utils as vutils

from torch.autograd import Variable
from torch.optim.lr_scheduler import _LRScheduler
#from external.pyTorchChamferDistance.chamfer_distance import ChamferDistance
import re

from chamferdist import ChamferDistance
dist_chamfer = ChamferDistance()

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    # image_numpy = image_tensor[0].cpu().float().numpy()
    # if image_numpy.shape[0] == 1:
    #     image_numpy = np.tile(image_numpy, (3, 1, 1))
    # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    # return image_numpy.astype(imtype)

    n_img = min(image_tensor.shape[0], 16)
    image_tensor = image_tensor[:n_img]

    if image_tensor.shape[1] == 1:
        image_tensor = image_tensor.repeat(1, 3, 1, 1)

    # if image_tensor.shape[1] == 4:
        # import pdb; pdb.set_trace()

    image_tensor = vutils.make_grid( image_tensor, nrow=4 )

    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = ( np.transpose( image_numpy, (1, 2, 0) ) + 1) / 2.0 * 255.
    return image_numpy.astype(imtype)

def to_variable(numpy_data, volatile=False):
    numpy_data = numpy_data.astype(np.float32)
    torch_data = torch.from_numpy(numpy_data).float()
    variable = Variable(torch_data, volatile=volatile)
    return variable

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def seed_everything(seed):
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    

def iou(x_gt, x, thres):
    thres_gt = 0.0
    # compute iou
    # > 0 free space, < 0 occupied
    x_gt_mask = x_gt.clone().detach()
    x_gt_mask[x_gt > thres_gt] = 0.
    x_gt_mask[x_gt <= thres_gt] = 1.

    x_mask = x.clone().detach()
    x_mask[x > thres] = 0.
    x_mask[x <= thres] = 1.

    inter = torch.logical_and(x_gt_mask, x_mask)
    union = torch.logical_or(x_gt_mask, x_mask)
    inter = rearrange(inter, 'b c d h w -> b (c d h w)')
    union = rearrange(union, 'b c d h w -> b (c d h w)')

    iou = inter.sum(1) / (union.sum(1) + 1e-12)
    return iou

def cd(gt_points, pred_points): 
    #chamfer distance
    gt_points=torch.from_numpy(gt_points).float()
    pred_points=torch.from_numpy(pred_points).float()
    chamfer_values=[]
    for index in range(gt_points.shape[0]):
        dist1 = dist_chamfer(gt_points[index].unsqueeze(0), pred_points[index].unsqueeze(0)) #[:2]
        chamfer_values.append(dist1)
    torch.Tensor(chamfer_values)
    return torch.Tensor(chamfer_values)

def sample_pnts_from_obj(data, n_pnts = 5000, mode = 'uniform'):
        # sample points on each object mesh.

        flags = data.keys()
        all_pnts = data['v'][:,:3]
        area_list = np.array(calculate_face_area(data))
        distribution = area_list/np.sum(area_list)

        # sample points the probability depends on the face area
        new_pnts = []
        if mode == 'random':

            random_face_ids = np.random.choice(len(data['f']), n_pnts, replace=True, p=distribution)
            random_face_ids, sample_counts = np.unique(random_face_ids, return_counts=True)

            for face_id, sample_count in zip(random_face_ids, sample_counts):

                face = data['f'][face_id]
                vid_in_face = [int(item.split('/')[0]) for item in face]
                weights = np.diff(np.sort(np.vstack(
                    [np.zeros((1, sample_count)), np.random.uniform(0, 1, size=(len(vid_in_face) - 1, sample_count)),
                    np.ones((1, sample_count))]), axis=0), axis=0)
                new_pnt = all_pnts[np.array(vid_in_face) - 1].T.dot(weights)

                if 'vn' in flags:
                    nid_in_face = [int(item.split('/')[2]) for item in face]
                    new_normal = data['vn'][np.array(nid_in_face)-1].T.dot(weights)
                    new_pnt = np.hstack([new_pnt, new_normal])


                new_pnts.append(new_pnt.T)

            random_pnts = np.vstack(new_pnts)

        else:

            for face_idx, face in enumerate(data['f']):
                vid_in_face = [int(item.split('/')[0]) for item in face]

                n_pnts_on_face = distribution[face_idx] * n_pnts

                if n_pnts_on_face < 1:
                    continue

                dim = len(vid_in_face)
                npnts_dim = (np.math.factorial(dim - 1)*n_pnts_on_face)**(1/(dim-1))
                npnts_dim = int(npnts_dim)

                weights = np.stack(np.meshgrid(*[np.linspace(0, 1, npnts_dim) for _ in range(dim - 1)]), 0)
                weights = weights.reshape(dim - 1, -1)
                last_column = 1 - weights.sum(0)
                weights = np.vstack([weights, last_column])
                weights = weights[:, last_column >= 0]
                new_pnt = (all_pnts[np.array(vid_in_face) - 1].T.dot(weights)).T
                if 'vn' in flags:
                    nid_in_face = [int(item.split('/')[2]) for item in face]
                    new_normal = data['vn'][np.array(nid_in_face) - 1].T.dot(weights)
                    new_pnt = np.hstack([new_pnt, new_normal])
                new_pnts.append(new_pnt)
            random_pnts = np.vstack(new_pnts)
        return random_pnts

def normalize_to_unit_square(points):
        centre = (points.max(0) + points.min(0))/2.
        point_shapenet = points - centre

        scale = point_shapenet.max()
        point_shapenet = point_shapenet / scale

        return point_shapenet, centre, scale
def write_obj(objfile, data):
    with open(objfile, 'w+') as file:
        for item in data['v']:
            file.write('v' + ' %f' * len(item) % tuple(item) + '\n')

        for item in data['f']:
            file.write('f' + ' %s' * len(item) % tuple(item) + '\n')


def read_obj(model_path,flags = ('v')):
    fid = open(model_path, 'r')

    data = {}

    for head in flags:
        data[head] = []

    for line in fid:
        # line = line.strip().split(' ')
        line = re.split('\s+', line.strip())
        if line[0] in flags:
            data[line[0]].append(line[1:])

    fid.close()

    if 'v' in data.keys():
        data['v'] = np.array(data['v']).astype(np.float)

    if 'vt' in data.keys():
        data['vt'] = np.array(data['vt']).astype(np.float)

    if 'vn' in data.keys():
        data['vn'] = np.array(data['vn']).astype(np.float)

    return data

def calculate_face_area(data):

    face_areas = []

    for face in data['f']:
        vid_in_face = [int(item.split('/')[0]) for item in face]
        face_area = get_area(data['v'][np.array(vid_in_face) - 1,:3].tolist())
        face_areas.append(face_area)

    return face_areas

def get_area(poly):
    if len(poly) < 3: # not a plane - no area
        return 0

    total = [0, 0, 0]
    for i in range(len(poly)):
        vi1 = poly[i]
        if i is len(poly)-1:
            vi2 = poly[0]
        else:
            vi2 = poly[i+1]
        prod = cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]

    result = dot(total, unit_normal(poly[0], poly[1], poly[2]))
    return abs(result/2)
def unit_normal(a, b, c):
    x = det([[1,a[1],a[2]],
             [1,b[1],b[2]],
             [1,c[1],c[2]]])
    y = det([[a[0],1,a[2]],
             [b[0],1,b[2]],
             [c[0],1,c[2]]])
    z = det([[a[0],a[1],1],
             [b[0],b[1],1],
             [c[0],c[1],1]])
    magnitude = (x**2 + y**2 + z**2)**.5
    if magnitude == 0.:
        return (0., 0., 0.)
    else:
        return (x/magnitude, y/magnitude, z/magnitude)

def det(a):
    return a[0][0]*a[1][1]*a[2][2] + a[0][1]*a[1][2]*a[2][0] + a[0][2]*a[1][0]*a[2][1] - a[0][2]*a[1][1]*a[2][0] - a[0][1]*a[1][0]*a[2][2] - a[0][0]*a[1][2]*a[2][1]

# unit normal vector of plane defined by points a, b, and c
#dot product of vectors a and b
def dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
def unit_normal(a, b, c):
    x = det([[1,a[1],a[2]],
             [1,b[1],b[2]],
             [1,c[1],c[2]]])
    y = det([[a[0],1,a[2]],
             [b[0],1,b[2]],
             [c[0],1,c[2]]])
    z = det([[a[0],a[1],1],
             [b[0],b[1],1],
             [c[0],c[1],1]])
    magnitude = (x**2 + y**2 + z**2)**.5
    if magnitude == 0.:
        return (0., 0., 0.)
    else:
        return (x/magnitude, y/magnitude, z/magnitude)

def cross(a, b):
    x = a[1] * b[2] - a[2] * b[1]
    y = a[2] * b[0] - a[0] * b[2]
    z = a[0] * b[1] - a[1] * b[0]
    return (x, y, z)


# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738

# import torch
# import torch.nn.functional as F
def DiffAugment(x, policy='', channels_first=True):
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x


def rand_brightness(x):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x


def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x


def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x


def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2).contiguous()
    return x


def rand_cutout(x, ratio=0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}

# Noam Learning rate schedule.
# From https://github.com/tugstugi/pytorch-saltnet/blob/master/utils/lr_scheduler.py
class NoamLR(_LRScheduler):
	
	def __init__(self, optimizer, warmup_steps):
		self.warmup_steps = warmup_steps
		super().__init__(optimizer)

	def get_lr(self):
		last_epoch = max(1, self.last_epoch)
		scale = self.warmup_steps ** 0.5 * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
		return [base_lr * scale for base_lr in self.base_lrs]