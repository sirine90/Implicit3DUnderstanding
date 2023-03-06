# Functions to visualize 3D reconstructed scene
# Script used for Demo and for Visualization of test samples
import sys
sys.path.append('.')
import argparse
from utils.front3d_config import FRONT3D_CONFIG
import os
import torch
import json
import pickle
from configs.data_config import Config
import numpy as np
from libs.tools import R_from_yaw_pitch_roll
import scipy.io as sio
from glob import glob
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from utils.vis_tools import Scene3D, nyu_color_palette
from configs.data_config import RECON_3D_CLS
from utils.front3d_utils import proj_from_point_to_2d, get_corners_of_bb3d_no_index
import re
from tqdm import tqdm
from net_utils.libs import get_bdb_evaluation


import numpy as np
from utils.vis_tools import Scene3D, nyu_color_palette

def get_bbox_corners(bbox_size,obj_matrix,wrd2cam_matrix):
    s_x, s_y, s_z = bbox_size[0], bbox_size[1], bbox_size[2]
    verts_cannonical = [[- s_x / 2, - s_y / 2, - s_z / 2],
    [- s_x / 2, - s_y / 2, s_z / 2],
    [- s_x / 2, s_y / 2, - s_z / 2],
    [- s_x / 2, s_y / 2, s_z / 2],
    [s_x / 2, - s_y / 2, - s_z / 2],
    [s_x / 2, - s_y / 2, s_z / 2],
    [s_x / 2, s_y / 2, - s_z / 2],
    [s_x / 2, s_y / 2, s_z / 2]]
    verts_cannonical=np.array(verts_cannonical)

    verts = np.dot(obj_matrix[0:3,0:3], verts_cannonical.T).T+obj_matrix[0:3,3]
    verts=np.dot(wrd2cam_matrix[0:3,0:3],verts.T).T+wrd2cam_matrix[0:3,3]
    verts[:,0:2]=-verts[:,0:2]
    return verts

def project_points2img(points,K):
    points=np.dot(points,K[0:3,0:3].T)
    x_coor=points[:,0]/points[:,2]
    y_coor=points[:,1]/points[:,2]
    z=points[:,2]
    img_coor=np.concatenate([x_coor[:,np.newaxis],y_coor[:,np.newaxis],z[:,np.newaxis]],axis=1)
    return img_coor[:,:2]


def draw_projected_bdb3d_GT(ex, if_save = True, save_path=''):
        mean_size_path='/mnt/hdd/ayadi/data/3dfront/preprocessed/new_size_avg_category.pkl'
        if os.path.exists(mean_size_path):
                with open(mean_size_path, 'rb') as file:
                    avg_size = pickle.load(file)
        avg_size= np.vstack([avg_size[key] for key in range(len(avg_size))])
        from PIL import Image, ImageDraw, ImageFont
        print('ex keys ', ex.keys())

        img_map = Image.fromarray(ex['rgb_img'][:])
        antialias = 1
        img_map = img_map.resize([s * antialias for s in img_map.size], Image.ANTIALIAS)

        draw = ImageDraw.Draw(img_map)

        width = 8 * antialias
        i=0
        for coeffs, tran_matrix, class_id in zip(ex['boxes']['scale'],ex['boxes']['tran_matrix'], ex['boxes']['size_cls']):
            if class_id == 0:
                continue
            # if class_id not in RECON_3D_CLS:
            #     continue
            #center_from_3D, invalid_ids = proj_from_point_to_2d(centroid, self.cam_K, cam_R)
            
            coeffs=(ex['boxes']['size_reg'][i]+1)*avg_size[class_id]                       
            bdb3d_corners = get_bbox_corners(coeffs, tran_matrix , ex['camera']['wrd2cam_matrix'])
            bdb2D_from_3D = project_points2img(bdb3d_corners, ex['camera']['K'])* antialias 
            bdb2D_from_3D = [tuple(item) for item in bdb2D_from_3D]

            color = [c * 8/9 for c in nyu_color_palette[class_id]]

            draw.line([bdb2D_from_3D[0], bdb2D_from_3D[1], bdb2D_from_3D[3], bdb2D_from_3D[2], bdb2D_from_3D[0]],
                        fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), width=width)
            draw.line([bdb2D_from_3D[4], bdb2D_from_3D[5], bdb2D_from_3D[7], bdb2D_from_3D[6], bdb2D_from_3D[4]],
                        fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), width=width)
            draw.line([bdb2D_from_3D[0], bdb2D_from_3D[4]],
                        fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), width=width)
            draw.line([bdb2D_from_3D[1], bdb2D_from_3D[5]],
                        fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), width=width)
            draw.line([bdb2D_from_3D[3], bdb2D_from_3D[7]],
                        fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), width=width)
            draw.line([bdb2D_from_3D[2], bdb2D_from_3D[6]],
                        fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), width=width)
            i=i+1
            # draw.text(tuple(center_from_3D), NYU40CLASSES[class_id],
            #           fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 20))

        img_map = img_map.resize([s // antialias for s in img_map.size], Image.ANTIALIAS)
        # img_map.show()
        if if_save:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            img_map.save(save_path)
        
        return bdb3d_corners

def R_from_yaw_pitch_roll_gt(yaw, pitch, roll):
    Rp = torch.zeros((1,3, 3)).cuda()
    Ry = torch.zeros((1,3,3)).cuda()
    Rr = torch.zeros((1,3,3)).cuda()
    cp=torch.cos(pitch)
    sp=torch.sin(pitch)
    cy=torch.cos(yaw)
    sy=torch.sin(yaw)
    cr=torch.cos(roll)
    sr=torch.sin(roll)
    Rp[:,0,0]=1
    Rp[:,1,1]=cp
    Rp[:,1,2]=-sp
    Rp[:,2,1]=sp
    Rp[:,2,2]=cp

    Ry[:,0,0]=cy
    Ry[:,0,2]=sy
    Ry[:,1,1]=1
    Ry[:,2,0]=-sy
    Ry[:,2,2]=cy

    Rr[:,0,0]=cr
    Rr[:,0,1]=-sr
    Rr[:,1,0]=sr
    Rr[:,1,1]=cr
    Rr[:,2,2]=1
    
    R=torch.bmm(torch.bmm(Rr,Rp),Ry)
    return R[0,:,:]


def num_from_bins(bins, cls, reg):
    """
    :param bins: b x 2 tensors
    :param cls: b long tensors
    :param reg: b tensors
    :return: bin_center: b tensors
    """
    bin_width = (bins[0][1] - bins[0][0])
    bin_center = (bins[cls, 0] + bins[cls, 1]) / 2
    return bin_center + reg * bin_width

def get_rotation_matrix(cam_data, bin):

    pitch = num_from_bins(np.array(bin['pitch_bin']), cam_data['pitch_cls'], cam_data['pitch_reg'])
    roll = num_from_bins(np.array(bin['roll_bin']), cam_data['roll_cls'], cam_data['roll_reg'])
    R = R_from_yaw_pitch_roll(0., pitch, roll)

    return R

def get_rotation_matrix_gt(cam_data, bin):
    '''
    get rotation matrix from predicted camera pitch, roll angles.
    '''
    pitch = num_from_bins(np.array(bin['pitch_bin']), cam_data['pitch_cls'],  cam_data['pitch_reg'])
    roll = num_from_bins(np.array(bin['roll_bin']), cam_data['roll_cls'], cam_data['roll_reg'])
    r_ex = R_from_yaw_pitch_roll_gt(torch.zeros_like(torch.tensor(pitch)), torch.tensor(pitch),torch.tensor(roll))
    cam_R_inv=R_from_yaw_pitch_roll_gt(torch.zeros_like(torch.tensor(pitch)), -torch.tensor(pitch), -torch.tensor(roll))

    return r_ex,cam_R_inv


def format_mesh(obj_files, bboxes):

    vtk_objects = {}

    for obj_file in obj_files:
        filename = '.'.join(os.path.basename(obj_file).split('.')[:-1])
        obj_idx = int(filename.split('_')[0])
        class_id = int(filename.split('_')[1].split(' ')[0])
        assert bboxes['class_id'][obj_idx] == class_id

        object = vtk.vtkOBJReader()
        object.SetFileName(obj_file)
        object.Update()

        # get points from object
        polydata = object.GetOutput()
        # read points using vtk_to_numpy
        points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(float)

        mesh_center = (points.max(0) + points.min(0)) / 2.
        points = points - mesh_center

        mesh_coef = (points.max(0) - points.min(0)) / 2.
        points = points.dot(np.diag(1./mesh_coef)).dot(np.diag(bboxes['coeffs'][obj_idx]))

        # set orientation
        points = points.dot(bboxes['basis'][obj_idx])

        # move to center
        points = points + bboxes['centroid'][obj_idx]
        points_array = numpy_to_vtk(points, deep=True)
        
        polydata.GetPoints().SetData(points_array)
        object.Update()

        vtk_objects[obj_idx] = object

    return vtk_objects, bboxes

def get_bdb_form_from_corners(corners, obj_matrix, wrd2cam_matrix, K):
    #vec_0 = (corners[:, 2, :] - corners[:, 1, :]) / 2.
    #vec_1 = (corners[:, 0, :] - corners[:, 4, :]) / 2.
    #vec_2 = (corners[:, 1, :] - corners[:, 0, :]) / 2.

    vec_0 = (corners[:, 4, :] - corners[:, 0, :]) / 2.
    vec_1 = (corners[:, 2, :] - corners[:, 0, :]) / 2.
    vec_2 = (corners[:, 1, :] - corners[:, 0, :]) / 2.

    coeffs_0 = np.linalg.norm(vec_0, axis=1)
    coeffs_1 = np.linalg.norm(vec_1, axis=1)
    coeffs_2 = np.linalg.norm(vec_2, axis=1)
    coeffs = np.stack([coeffs_0, coeffs_1, coeffs_2], axis=1)

    centroid = (corners[:, 0, :] + corners[:, 6, :]) / 2.

    basis_0 = np.dot(np.diag(1 / coeffs_0), vec_0)
    basis_1 = np.dot(np.diag(1 / coeffs_1), vec_1)
    basis_2 = np.dot(np.diag(1 / coeffs_2), vec_2)

    basis = np.stack([basis_0, basis_1, basis_2], axis=1)
    controller='gt'

    if controller=='gt':
        R=np.array([[1.00, 0.0, 0.0],
                            [0.00, -1.0, 0],
                            [0.00 ,0.0 , 1.]])
     
        for i in range(basis.shape[0]):
            basis[i,:,:]=R.dot(basis[i,:,:])


    else: 
        R=np.array([[0.00, 0.0, 1.0],
                            [0.00, -1.0, 0],
                            [1.00 ,0.0 ,0.]])
        for i in range(basis.shape[0]):
            basis[i,:,:]=R.dot(basis[i,:,:])
            xx=coeffs[i,0]
            coeffs[i,0]=coeffs[i,2]
            coeffs[i,2]=xx

    return {'basis': basis, 'coeffs': coeffs, 'centroid': centroid}

def get_bdb_form_from_corners_NEW(corners, obj_matrix, wrd2cam_matrix, K):
    vec_0 = (corners[:, 4, :] - corners[:, 0, :]) / 2.
    vec_1 = (corners[:, 2, :] - corners[:, 0, :]) / 2.
    vec_2 = (corners[:, 1, :] - corners[:, 0, :]) / 2.

    coeffs_0 = np.linalg.norm(vec_0, axis=1)
    coeffs_1 = np.linalg.norm(vec_1, axis=1)
    coeffs_2 = np.linalg.norm(vec_2, axis=1)
    coeffs = np.stack([coeffs_0, coeffs_1, coeffs_2], axis=1)

    centroid = (corners[:, 0, :] + corners[:, 6, :])/ 2. 

    verts=centroid*0
    for i in range(centroid.shape[0]):
        verts[i] = np.dot(obj_matrix[i][0:3,0:3], centroid[i].T).T+obj_matrix[i][0:3,3]
        verts[i]=np.dot(wrd2cam_matrix[0:3,0:3],centroid[i].T).T+wrd2cam_matrix[0:3,3]
    
        verts[i,:3]=-verts[i,:3]
    centroid=verts

    basis_0 = np.dot(np.diag(1 / coeffs_0), vec_0)
    basis_1 = np.dot(np.diag(1 / coeffs_1), vec_1)
    basis_2 = np.dot(np.diag(1 / coeffs_2), vec_2)

    basis = np.stack([basis_0, basis_1, basis_2], axis=1) 
    R=np.array([[0.00, 0.0, 1.0],
                        [0.00, -1.0, 0],
                        [1.00 ,0.0 ,0.]])
    for i in range(basis.shape[0]):
        basis[i,:,:]=R.dot(basis[i,:,:])
        xx=coeffs[i,0]
        coeffs[i,0]=coeffs[i,2]
        coeffs[i,2]=xx

    return {'basis': basis, 'coeffs': coeffs/2, 'centroid': centroid}

def format_bbox(box, type,  obj_matrix=None, wrd2cam_matrix=None, K=None):

    if type == 'prediction':
        boxes = {}
        basis_list = []
        centroid_list = []
        coeff_list = []

        # convert bounding boxes
        box_data = box['bdb'][0]

        for index in range(len(box_data)):
            basis = box_data[index]['basis'][0][0]

            R=np.array([[0.00, 0.0, 1.0],
                        [0.00, 1.0, 0],
                        [1.00 ,0.0 ,0.]])
            basis=R.dot(basis)
            
            centroid = box_data[index]['centroid'][0][0][0]
            coeffs = box_data[index]['coeffs'][0][0][0]

            xx=coeffs[0]
            coeffs[0]=coeffs[2]
            coeffs[2]=xx

            basis_list.append(basis)
            centroid_list.append(centroid)
            coeff_list.append(coeffs)

        boxes['basis'] = np.stack(basis_list, 0)
        boxes['centroid'] = np.stack(centroid_list, 0)
        boxes['coeffs'] = np.stack(coeff_list, 0) #/2
        boxes['class_id'] = box['class_id'][0] 

    else:   
        boxes = get_bdb_form_from_corners(box['bdb3D'], obj_matrix, wrd2cam_matrix, K)
        boxes['class_id'] = box['size_cls'] #.tolist()

    return boxes

def format_layout(layout_data):

    layout_bdb = {}

    centroid = (layout_data.max(0) + layout_data.min(0)) / 2. 

    vector_z = (layout_data[1] - layout_data[0]) / 2.
    coeff_z = np.linalg.norm(vector_z)
    basis_z = vector_z/coeff_z

    vector_x = (layout_data[2] - layout_data[1]) / 2.
    coeff_x = np.linalg.norm(vector_x)
    basis_x = vector_x/coeff_x

    vector_y = (layout_data[0] - layout_data[4]) / 2.
    coeff_y = np.linalg.norm(vector_y)
    basis_y = vector_y/coeff_y

    basis = np.array([basis_x, basis_y, basis_z])
    coeffs = np.array([coeff_x, coeff_y, coeff_z])


    layout_bdb['coeffs'] = coeffs
    layout_bdb['centroid'] = centroid
    layout_bdb['basis'] = basis

    return layout_bdb


def format_layout_NEW(layout_data,  obj_matrix, wrd2cam_matrix, wrd2cam_matrix2, cam_R):

    layout_bdb = {}

    centroid = (layout_data.max(0) + layout_data.min(0)) / 2.

    vector_z = (layout_data[1] - layout_data[0])/ 2.
    coeff_z = np.linalg.norm(vector_z)
    basis_z = vector_z/coeff_z

    vector_x = (layout_data[4] - layout_data[0])/ 2.
    coeff_x = np.linalg.norm(vector_x)
    basis_x = vector_x/coeff_x

    vector_y = (layout_data[2] - layout_data[0])/ 2.
    coeff_y = np.linalg.norm(vector_y)
    basis_y = vector_y/coeff_y

    basis = np.array([basis_x, basis_y, basis_z])
    coeffs = np.array([coeff_x, coeff_y, coeff_z]) 

    layout_bdb['coeffs'] = coeffs
    layout_bdb['centroid'] = centroid
    layout_bdb['basis'] = basis

    return layout_bdb


class Box(Scene3D):

    def __init__(self, img_map, depth_map, cam_K, gt_cam_R, pre_cam_R, gt_layout, pre_layout, gt_boxes, pre_boxes, type, output_mesh, sample_data=None, save_path=None, sequence_id=None):
        super(Scene3D, self).__init__()
        self.sample_data=sample_data
        self.sequence_id=sequence_id
        self.save_path=os.path.join(save_path, '%s_bbox_GT.png' % (self.sequence_id)) 
        self._cam_K = cam_K
        self.gt_cam_R = gt_cam_R
        self._cam_R = gt_cam_R
        self.pre_cam_R = pre_cam_R
        self.gt_layout = gt_layout
        self.pre_layout = pre_layout
        self.gt_boxes = gt_boxes
        self.pre_boxes = pre_boxes
        self.mode = type
        self._img_map = img_map
        self._depth_map = depth_map
        if self.mode == 'prediction' or self.mode=='gt':
            self.output_mesh = output_mesh

    @property
    def valid_objects_num(self):
        if type == 'prediction':
            boxes = self.pre_boxes
        else:
            boxes = self.gt_boxes
        return len([class_id for class_id in boxes['class_id'] if class_id in RECON_3D_CLS])

    def draw_projected_bdb3d(self, ex, type = 'prediction', if_save = True, save_path='', vis=False):
        from PIL import Image, ImageDraw, ImageFont

        img_map = Image.fromarray(self.img_map[:])
        antialias =1
        img_map = img_map.resize([s * antialias for s in img_map.size], Image.ANTIALIAS)

        draw = ImageDraw.Draw(img_map)

        width = 8 * antialias

        if type == 'prediction':
            boxes = self.pre_boxes
            cam_R = self.pre_cam_R
        else:
            boxes = self.gt_boxes
            cam_R = self.gt_cam_R
        i=0
        for coeffs, centroid, class_id, basis in zip(boxes['coeffs'], boxes['centroid'], boxes['class_id'], boxes['basis']):
            if class_id == 0:
                continue
            # if class_id not in RECON_3D_CLS:
            #     continue
            center_from_3D, invalid_ids = proj_from_point_to_2d(centroid, self.cam_K, cam_R)
            bdb3d_corners = get_corners_of_bb3d_no_index(basis, coeffs, centroid)
            if vis:
                T_cam = np.array([[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]])
                bdb3d_corners=np.dot(bdb3d_corners, T_cam.T)
            bdb2D_from_3D = proj_from_point_to_2d(bdb3d_corners, self.cam_K, cam_R)[0] * antialias

            if type=='gt':
                bdb2D_from_3D=bdb2D_from_3D            

            # bdb2D_from_3D = np.round(bdb2D_from_3D).astype('int32')
            bdb2D_from_3D = [tuple(item) for item in bdb2D_from_3D]

            color = [c * 8/9 for c in nyu_color_palette[class_id]]

            draw.line([bdb2D_from_3D[0], bdb2D_from_3D[1], bdb2D_from_3D[2], bdb2D_from_3D[3], bdb2D_from_3D[0]],
                      fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), width=width)
            draw.line([bdb2D_from_3D[4], bdb2D_from_3D[5], bdb2D_from_3D[6], bdb2D_from_3D[7], bdb2D_from_3D[4]],
                      fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), width=width)
            draw.line([bdb2D_from_3D[0], bdb2D_from_3D[4]],
                      fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), width=width)
            draw.line([bdb2D_from_3D[1], bdb2D_from_3D[5]],
                      fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), width=width)
            draw.line([bdb2D_from_3D[2], bdb2D_from_3D[6]],
                      fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), width=width)
            draw.line([bdb2D_from_3D[3], bdb2D_from_3D[7]],
                      fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), width=width)

            # draw.text(tuple(center_from_3D), NYU40CLASSES[class_id],
            #           fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 20))

        img_map = img_map.resize([s // antialias for s in img_map.size], Image.ANTIALIAS)
        # img_map.show()

        if if_save:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            img_map.save(save_path)

    def get_bbox_actor(self, box, color, opacity):

        #corners=draw_projected_bdb3d_GT(self.sample_data,if_save=True, save_path='demo_data/image.jpg')
        #corners=[corners[i] for i in range(corners.shape[0])]
        faces = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (0, 3, 7, 4)]
        vectors = [box['coeffs'][basis_id] * basis for basis_id, basis in enumerate(box['basis'])]
        corners, faces = self.get_box_corners(box['centroid'], vectors)
        #corners=draw_projected_bdb3d_GT(self.sample_data,if_save=True, save_path='demo_data/image.jpg')
        bbox_actor = self.set_actor(self.set_mapper(self.set_cube_prop(corners, faces, color), 'box'))
        bbox_actor.GetProperty().SetOpacity(opacity)
        return bbox_actor

    def get_bbox_line_actor(self, box, color, opacity, width=10):
        vectors = [box['coeffs'][basis_id] * basis for basis_id, basis in enumerate(box['basis'])]
        corners, faces = self.get_box_corners(box['centroid'], vectors)
        #corners=draw_projected_bdb3d_GT(self.sample_data,if_save=True, save_path='demo_data/image.jpg')
        #faces = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (0, 3, 7, 4)]
        bbox_actor = self.set_actor(self.set_mapper(self.set_bbox_line_actor(corners, faces, color), 'box'))
        bbox_actor.GetProperty().SetOpacity(opacity)
        bbox_actor.GetProperty().SetLineWidth(width)
        return bbox_actor

    def get_orientation_actor(self, centroid, vector, color):

        arrow_actor = self.set_arrow_actor(centroid, vector)
        arrow_actor.GetProperty().SetColor(color)

        return arrow_actor

    def get_voxel_actor(self, voxels, voxel_vector, color):
        # draw each voxel
        voxel_actors = []
        for point in voxels:
            corners, faces = self.get_box_corners(point, voxel_vector)
            voxel_actor = self.set_actor(self.set_mapper(self.set_cube_prop(corners, faces, color), 'box'))
            voxel_actors.append(voxel_actor)
        return voxel_actors

    def set_render(self):
        renderer = vtk.vtkRenderer()
        renderer.ResetCamera()

        # '''draw layout system'''
        # renderer.AddActor(self.set_axes_actor())

        '''draw gt camera orientation'''
        if self.mode == 'gt' or self.mode == 'both':
            color = [[1., 0., 0.], [1., 0., 0.], [1., 0., 0.]]
            center = [0, 0, 0]
            T_cam = np.array([[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]])
            #vectors=np.dot(vectors, T_cam.T)

            vectors = np.dot(self.gt_cam_R.T,T_cam)
            # for index in range(vectors.shape[0]):
            #     arrow_actor = self.set_arrow_actor(center, vectors[index])
            #     arrow_actor.GetProperty().SetColor(color[index])
            #     renderer.AddActor(arrow_actor)
            '''set camera property'''
            camera = self.set_camera(center, vectors, self.cam_K)
            renderer.SetActiveCamera(camera)

        '''draw predicted camera orientation'''
        if self.mode == 'prediction' or self.mode == 'both':
            color = [[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]]
            center = [0, 0, 0]
            vectors = self.pre_cam_R.T
            # for index in range(vectors.shape[0]):
            #     arrow_actor = self.set_arrow_actor(center, vectors[index])
            #     arrow_actor.GetProperty().SetColor(color[index])
            #     renderer.AddActor(arrow_actor)
            '''set camera property'''
            camera = self.set_camera(center, vectors, self.cam_K)
            renderer.SetActiveCamera(camera)

        '''draw gt layout'''
        if self.mode == 'gt' or self.mode == 'both':
            color = (75, 75, 75)
            opacity = 0.2
            layout_actor = self.get_bbox_actor(self.gt_layout, color, opacity)
            renderer.AddActor(layout_actor)
            layout_line_actor = self.get_bbox_line_actor(self.gt_layout, color, 1.)
            renderer.AddActor(layout_line_actor)

        
        '''draw predicted layout'''

        
        if self.mode == 'prediction' or self.mode == 'both':
            color = (75, 75, 75)
            opacity = 0.2
            layout_actor = self.get_bbox_actor(self.pre_layout, color, opacity)
            renderer.AddActor(layout_actor)
            layout_line_actor = self.get_bbox_line_actor(self.pre_layout, (75,75,75), 1.)
            renderer.AddActor(layout_line_actor)
        

        '''draw gt obj bounding boxes'''
        if self.mode == 'gt' or self.mode == 'both':
            for coeffs, centroid, class_id, basis in zip(self.gt_boxes['coeffs'],
                                                         self.gt_boxes['centroid'],
                                                         self.gt_boxes['class_id'],
                                                         self.gt_boxes['basis']):
                if class_id not in RECON_3D_CLS:
                    continue
                color = [1., 0., 0.]
                opacity = 0.2
                box = {'coeffs':coeffs, 'centroid':centroid, 'class_id':class_id, 'basis':basis}
                bbox_actor = self.get_bbox_actor(box, color, opacity)
                renderer.AddActor(bbox_actor)

                # draw orientations
                color = [[0.8, 0.8, 0.8],[0.8, 0.8, 0.8],[1., 0., 0.]]
                vectors = [box['coeffs'][v_id] * vector for v_id, vector in enumerate(box['basis'])]

                for index in range(3):
                    arrow_actor = self.get_orientation_actor(box['centroid'], vectors[index], color[index])
                    renderer.AddActor(arrow_actor)

        '''draw predicted obj bounding boxes'''
        if self.mode == 'prediction' or self.mode == 'both':
            for coeffs, centroid, class_id, basis in zip(self.pre_boxes['coeffs'],
                                                         self.pre_boxes['centroid'],
                                                         self.pre_boxes['class_id'],
                                                         self.pre_boxes['basis']):
                if class_id not in RECON_3D_CLS:
                    continue
                color = nyu_color_palette[class_id]
                opacity = 0.2
                box = {'coeffs':coeffs, 'centroid':centroid, 'class_id':class_id, 'basis':basis}
                bbox_actor = self.get_bbox_actor(box, color, opacity)
                renderer.AddActor(bbox_actor)

                # draw orientations
                color = [[0.8, 0.8, 0.8],[0.8, 0.8, 0.8],[1., 0., 0.]]
                vectors = [box['coeffs'][v_id] * vector for v_id, vector in enumerate(box['basis'])]

                for index in range(3):
                    arrow_actor = self.get_orientation_actor(box['centroid'], vectors[index], color[index])
                    renderer.AddActor(arrow_actor)

        # draw mesh
        if self.output_mesh:
            for obj_idx, class_id in enumerate(self.pre_boxes['class_id']):
                if class_id not in RECON_3D_CLS:
                    continue
                color = nyu_color_palette[class_id]

                object = self.output_mesh[obj_idx]

                object_actor = self.set_actor(self.set_mapper(object, 'model'))
                object_actor.GetProperty().SetColor(color)
                renderer.AddActor(object_actor)

        # '''draw point cloud'''
        # point_actor = self.set_actor(self.set_mapper(self.set_points_property(np.eye(3)), 'box'))
        # point_actor.GetProperty().SetPointSize(1)
        # renderer.AddActor(point_actor)

        renderer.SetBackground(1., 1., 1.)

        return renderer, None

    def draw3D(self, if_save, save_path):
        '''
        Visualize 3D models with their bounding boxes.
        '''
        render_window_interactor = vtk.vtkRenderWindowInteractor()
        render_window = self.set_render_window()
        render_window_interactor.SetRenderWindow(render_window)
        render_window.Render()
        if not if_save:
            render_window_interactor.Start()

        if if_save:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            im = vtk.vtkWindowToImageFilter()
            writer = vtk.vtkPNGWriter()
            im.SetInput(render_window)
            im.ReadFrontBufferOff()
            im.Update()
            writer.SetInputConnection(im.GetOutputPort())
            writer.SetFileName(save_path)
            writer.Write()

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser(description='3D visualization of Total3D results.')
    parser.add_argument('--result_path', type=str, default='out/total3d/joint_train_new/visualization',
                        help='Results exported from test.py.')
    parser.add_argument('--sequence_id', type=int, default=274,
                        help='Give the sequence id in test set you want to visualize.')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Save results folder.')
    parser.add_argument('--search_folder', type=str, default=None,
                        help='Selected image folder.')
    parser.add_argument('--offscreen', action='store_true', default=False,
                        help='Render offscreen.')
    parser.add_argument('--min_obj', type=int, default=0,
                        help='Min number of objects.')
    parser.add_argument('--max_obj', type=int, default=1000,
                        help='Max number of objects.')
    parser.add_argument('--gt', action='store_true', default=False,
                        help='Render offscreen.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    front3d_config = FRONT3D_CONFIG()

    test_split = '/mnt/hdd/ayadi/data/3dfront/splits/test.json'
    with open(test_split, 'r') as f:
        data_frame = json.load(f)

    def visualize(sequence_id, args):
        save_path = args.save_path
        if save_path is None:
            save_path = './demo/3dfront/'
            if_save = False
        else:
            if_save = True

        test_file = '/mnt/hdd/ayadi/data/3dfront/3dfront_train_test_data/' + str(sequence_id) + '.pkl'
        assert os.path.exists(test_file)

        with open(test_file, 'rb') as file:
            sample_data = pickle.load(file)

        # depth image
        depth_img = sample_data['depth_map']

        '''load metadata'''
        rgb_image = sample_data['rgb_img']
        cam_K = sample_data['camera']['K']
        cam_K[:2]=cam_K[:2]/2

        gt_config = Config('3dfront')
        bins = gt_config.bins
        gt_cam_R = get_rotation_matrix(sample_data['camera'], bins)

        # ========================================================================= #
        #                    Get gt and reconstructed data
        # ========================================================================= #
        '''load ground-truth data'''
        gt_layout_data = sample_data['layout']
        gt_box_data = sample_data['boxes']

        gt_boxes = format_bbox(gt_box_data, 'gt', obj_matrix= gt_box_data['tran_matrix'], wrd2cam_matrix=sample_data['camera']['wrd2cam_matrix'], K=cam_K)
        gt_layout = format_layout_NEW(gt_layout_data['bdb3D'], obj_matrix= gt_box_data['tran_matrix'], wrd2cam_matrix=sample_data['camera']['wrd2cam_matrix'],wrd2cam_matrix2=sample_data['camera']['wrd2cam_matrix2'], cam_R=gt_cam_R )

        '''load prediction data'''
        pre_path = os.path.join(args.result_path, str(sequence_id))
        pre_layout_data = sio.loadmat(os.path.join(pre_path, 'layout.mat'))['layout']

        pre_box_data = sio.loadmat(os.path.join(pre_path, 'bdb_3d.mat'))
        pre_boxes = format_bbox(pre_box_data, 'prediction')

        pre_layout = format_layout(pre_layout_data)
        pre_cam_R = sio.loadmat(os.path.join(pre_path, 'r_ex.mat'))['cam_R']

        pre_cam_R=gt_cam_R

        vtk_objects, pre_boxes = format_mesh(glob(os.path.join(pre_path, '*.obj')), pre_boxes)

        scene_box = Box(rgb_image, depth_img, cam_K, gt_cam_R, pre_cam_R, gt_layout, pre_layout, gt_boxes, pre_boxes,
                        'gt' if args.gt else 'prediction', output_mesh=vtk_objects, sample_data=sample_data, save_path=save_path, sequence_id=sequence_id)

        if args.min_obj <= scene_box.valid_objects_num <= args.max_obj:
            
            scene_box.draw_projected_bdb3d(sample_data, 'gt' if args.gt else 'prediction', if_save=if_save,
                                           save_path=os.path.join(save_path, '%s_bbox.png' % (sequence_id)), vis=True)
            
            draw_projected_bdb3d_GT(sample_data,if_save=if_save, save_path=os.path.join(save_path, '%s_bbox_GT.png' % (sequence_id)))
            # scene_box.draw_image()
            scene_box.draw3D(if_save=if_save, save_path=os.path.join(save_path, '%s_recon.png' % (sequence_id)))

    if args.offscreen and args.save_path is not None:
        args_str = ' '.join([f"--{k} {v if v is not True else ''}"
                             for k, v in args.__dict__.items()
                             if k not in ('offscreen') and v is not False])
        os.system(f'xvfb-run -a -s "-screen 0 800x600x24" {sys.executable} utils/visualize.py {args_str}')
    else:
        if args.search_folder not in (None, 'None') or args.sequence_id <= 0:
            if args.search_folder not in (None, 'None'):
                files = glob(os.path.join(args.search_folder, '*'))
            elif args.sequence_id <= 0:
                files = glob(os.path.join(args.result_path, '*'))
            sequence_ids = set([int(re.findall('\d+', os.path.splitext(os.path.basename(d))[0])[0]) for d in files])
            for i in tqdm(sequence_ids):
                visualize(i, args)
        else:
            visualize(args.sequence_id, args)




