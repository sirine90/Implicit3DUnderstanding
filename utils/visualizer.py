
# Adopted visualising functions from https://github.com/yccyenchicheng/AutoSDF

import pickle
import os
import ntpath
import time

from termcolor import colored
from . import util

import torch
import imageio
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pytorch3d.io.obj_io

def parse_line(line):
    info_d = {}

    l1, l2 = line.split(') ')
    l1 = l1.replace('(', '')
    l1 = l1.split(', ')

    l2 = l2.replace('(', '')
    l2 = l2.split(' ')

    info_d = {}
    for s in l1:
        
        k, v = s.split(': ')
        
        
        if k in ['epoch', 'iters']:
            info_d[k] = int(v)
        else:
            info_d[k] = float(v)

    l2_keys = l2[0::2]
    l2_vals = l2[1::2]
    
    for k, v in zip(l2_keys, l2_vals):
        k = k.replace(':','')
        info_d[k] = float(v)

    return info_d


class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.isTrain = opt.isTrain
        self.gif_fps = 4

        if self.isTrain:
            # self.log_dir = os.path.join(opt.checkpoints_dir, opt.name)
            self.log_dir = os.path.join(opt.logs_dir, opt.name)
        else:
            self.log_dir = os.path.join(opt.results_dir, opt.name)

        self.name = opt.name
        self.opt = opt

        self.img_dir = os.path.join(self.log_dir, 'images')
        self.mesh_dir = os.path.join(self.log_dir, 'meshes')
        print('[*] create image directory:\n%s...' % os.path.abspath(self.img_dir) )
        print('[*] create mesh directory:\n%s...' % os.path.abspath(self.mesh_dir) )
        util.mkdirs([self.img_dir])
        util.mkdirs([self.mesh_dir])
        # self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')

        if self.isTrain:
            self.log_name = os.path.join(self.log_dir, 'loss_log.txt')
            # with open(self.log_name, "a") as log_file:
            with open(self.log_name, "w") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    def print_current_errors(self, epoch, epoch_iters, total_iters, errors, t):
        message = '(GPU: %s, epoch: %d, iters: %d, time: %.3f) ' % (self.opt.gpu_ids_str, epoch, epoch_iters, t)
        for k, v in errors.items():
            message += '%s: %.6f ' % (k, v)

        print(colored(message, 'magenta'))
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

        self.log_tensorboard_errors(errors, total_iters)

    def print_current_metrics(self, epoch, metrics, phase):
        message = '([%s] GPU: %s, epoch: %d) ' % (phase, self.opt.gpu_ids_str, epoch)
        for k, v in metrics.items():
            message += '%s: %.3f ' % (k, v)

        print(colored(message, 'yellow'))
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

        self.log_tensorboard_metrics(metrics, epoch, phase)

    def display_current_results(self, visuals, epoch, pix3d1, pix3d2, pix3d3, im_name='', phase='train'):
        if epoch>100000:
            bs=len(pix3d1.faces_list())
            for label, image_numpy in visuals.items():
                if label=='image':
                        for i in range(bs):
                            mesh_path=os.path.join(self.mesh_dir, 'step%.3d_%s_%s_%s_%s.obj' % (epoch, phase, label, im_name, i))
                            try:
                                faces=pix3d1[i].faces_list()[0]
                                vertices=pix3d1[i].verts_list()[0]
                                pytorch3d.io.obj_io.save_obj(mesh_path, -1*vertices, faces)
                            except:
                                continue
                elif label=='image_recon':
                    for i in range(bs):
                            mesh_path=os.path.join(self.mesh_dir, 'step%.3d_%s_%s_%s_%s.obj' % (epoch, phase, label, im_name, i))
                            try:
                                faces=pix3d2[i].faces_list()[0]
                                vertices=pix3d2[i].verts_list()[0]
                                pytorch3d.io.obj_io.save_obj(mesh_path, -1*vertices, faces)
                            except:
                                continue
                elif (label=='image_recon_resnet' or label=='image_recon_tf' or label=='image_recon_rand_tf') and (pix3d3 is not None):
                        for i in range(bs):
                            mesh_path=os.path.join(self.mesh_dir, 'step%.3d_%s_%s_%s_%s.obj' % (epoch, phase, label, im_name, i))
                            try:
                                faces=pix3d3[i].faces_list()[0]
                                vertices=pix3d3[i].verts_list()[0]
                                pytorch3d.io.obj_io.save_obj(mesh_path, -1*vertices, faces)
                            except:
                                continue
           
                
        # write images to disk
        for label, image_numpy in visuals.items():
            img_path = os.path.join(self.img_dir, 'step%.3d_%s_%s_%s.png' % (epoch, phase, label, im_name))
            try:
                   util.save_image(image_numpy, img_path)
            except:
                import pdb; pdb.set_trace()
                    
        # log to tensorboard
        self.log_tensorboard_visuals(visuals, epoch, phase=phase)

    def log_tensorboard_visuals(self, visuals, cur_step, labels_while_list=None, phase='train'):
        writer = self.opt.writer

        if labels_while_list is None:
            labels_while_list = []

        for ix, (label, image_numpy) in enumerate(visuals.items()):
            if image_numpy.shape[2] == 4:
                image_numpy = image_numpy[:, :, :3]
            
            if label not in labels_while_list:
                # writer.add_image('vis/%d-%s' % (ix+1, label), image_numpy, global_step=cur_step, dataformats='HWC')
                writer.add_image('%s/%d-%s' % (phase, ix+1, label), image_numpy, global_step=cur_step, dataformats='HWC')
            else:
                pass
                # log the unwanted image just in case
                # writer.add_image('other/%s' % (label), image_numpy, global_step=cur_step, dataformats='HWC')

    def log_tensorboard_errors(self, errors, cur_step):
        writer = self.opt.writer

        for label, error in errors.items():
            writer.add_scalar('losses/%s' % label, error, cur_step)

    def log_tensorboard_metrics(self, metrics, cur_step, phase):
        writer = self.opt.writer

        for label, value in metrics.items():
            writer.add_scalar('metrics/%s-%s' % (phase, label), value, cur_step)