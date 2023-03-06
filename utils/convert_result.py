import numpy as np
import torch

from net_utils.bins import *

def convert_result(data_batch,est_data):
    split=data_batch["split"]
    save_dict_list=[]
    for idx,interval in enumerate(split):
        sequence_id=data_batch['sequence_id'][idx]
        length=interval[1]-interval[0]
        bbox_list=[]
        gt_bbox_list=[]
        for i in range(length):
            # object detection result
            centroid_cls_result=est_data['centroid_cls_result'][interval[0]+i]
            centroid_reg_result=est_data['centroid_reg_result'][interval[0]+i]
            size_reg_result=est_data['size_reg_result'][interval[0]+i].cpu().numpy()
            size_cls=data_batch['size_cls'][interval[0]+i]
            ori_cls_result=est_data['ori_cls_result'][interval[0]+i]
            ori_reg_result=est_data['ori_reg_result'][interval[0]+i]
            offest2D_result=est_data['offset_2D_result'][interval[0]+i].cpu().numpy()
            bdb2D=data_batch['bdb2D'][interval[0]+i].cpu().numpy()
            #print(bdb2D)

            max_centroid_ind=torch.argmax(centroid_cls_result)
            centroid_reg=centroid_reg_result[max_centroid_ind].cpu().numpy()
            centroid=np.mean(bin['centroid_bin'][max_centroid_ind])+centroid_reg*DEPTH_WIDTH
            max_ori_ind=torch.argmax(ori_cls_result)
            ori_reg=ori_reg_result[max_ori_ind].cpu().numpy()
            ori=np.mean(bin['ori_bin'][max_ori_ind])+ori_reg*ORI_BIN_WIDTH
            #print(max_ori_ind, ori_reg,ori)
            project_center=np.zeros([2])
            project_center[0]=(bdb2D[2]+bdb2D[0])/2-offest2D_result[0]*(bdb2D[2]-bdb2D[0])
            project_center[1]=(bdb2D[1]+bdb2D[3])/2-offest2D_result[1]*(bdb2D[3]-bdb2D[1])
            size_class_ind=torch.where(size_cls>0)[0]
            size=torch.tensor(bin['avg_size'][size_class_ind])*(1+size_reg_result)
            bbox_dict={
                "project_center":project_center,
                "centroid_depth":centroid,
                "size":size.cpu().numpy(),
                "yaw":ori,
                "bdb2D":bdb2D,
            }
            bbox_list.append(bbox_dict)


            #ground truth: object detection
            centroid_cls = data_batch['centroid_cls'][interval[0] + i].cpu().numpy().astype(np.int32)
            centroid_reg = data_batch['centroid_reg'][interval[0] + i]
            size_reg = data_batch['size_reg'][interval[0] + i].cpu().numpy()
            size_cls = data_batch['size_cls'][interval[0] + i]
            ori_cls = data_batch['ori_cls'][interval[0] + i].cpu().numpy().astype(np.int32)
            ori_reg = data_batch['ori_reg'][interval[0] + i]
            offest2D = data_batch['offset_2D'][interval[0] + i].cpu().numpy()
            bdb2D = data_batch['bdb2D'][interval[0] + i].cpu().numpy()

            gt_centroid_reg = centroid_reg.cpu().numpy()
            gt_centroid = np.mean(bin['centroid_bin'][centroid_cls]) + gt_centroid_reg * DEPTH_WIDTH
            #max_ori_ind = torch.argmax(ori_cls)
            gt_ori_reg = ori_reg.cpu().numpy()
            gt_ori = np.mean(bin['ori_bin'][ori_cls]) + gt_ori_reg * ORI_BIN_WIDTH
            #print(ori_cls,gt_ori_reg,gt_ori)
            project_center = np.zeros([2])
            project_center[0] = (bdb2D[2] + bdb2D[0]) / 2 - offest2D[0] * (bdb2D[2] - bdb2D[0])
            project_center[1] = (bdb2D[1] + bdb2D[3]) / 2 - offest2D[1] * (bdb2D[3] - bdb2D[1])
            size_class_ind = torch.where(size_cls > 0)[0]
            size = torch.tensor(bin['avg_size'][size_class_ind])*(1+ size_reg)
            gt_bbox_dict = {
                "project_center": project_center,
                "centroid_depth": gt_centroid,
                "size": size.cpu().numpy(),
                "yaw": gt_ori,
                "bdb2D": bdb2D,
            }
            gt_bbox_list.append(gt_bbox_dict)

        #layout estinamtion result
        pitch_cls_result=est_data['pitch_cls_result'][idx]
        pitch_reg_result=est_data['pitch_reg_result'][idx]
        max_pitch_ind=torch.argmax(pitch_cls_result)
        pitch_reg=pitch_reg_result[max_pitch_ind].cpu().numpy()
        pitch=np.mean(bin['pitch_bin'][max_pitch_ind])+pitch_reg*PITCH_WIDTH

        roll_cls_result = est_data['roll_cls_result'][idx]
        roll_reg_result = est_data['roll_reg_result'][idx]
        max_roll_ind = torch.argmax(roll_cls_result)
        roll_reg = roll_reg_result[max_roll_ind].cpu().numpy()
        roll = np.mean(bin['roll_bin'][max_roll_ind]) + roll_reg * ROLL_WIDTH

        #ground truth layout data
        gt_pitch=np.mean(bin['pitch_bin'][data_batch['pitch_cls'][idx].cpu().numpy()])+data_batch['pitch_reg'][idx].cpu().numpy()*PITCH_WIDTH
        gt_roll = np.mean(bin['roll_bin'][data_batch['roll_cls'][idx].cpu().numpy()]) + data_batch['roll_reg'][
            idx].cpu().numpy() * ROLL_WIDTH
        lo_centroid=avg_layout['avg_centroid']+est_data['lo_centroid_result'][idx].cpu().numpy()
        lo_size=avg_layout['avg_size']+est_data['lo_coeffs_result'][idx].cpu().numpy()
        layout_dict={
            "pitch":pitch,
            "roll":roll,
            "gt_pitch":gt_pitch,
            "gt_roll":gt_roll,
            "lo_centroid":lo_centroid,
            "lo_size":lo_size,
        }
        save_dict={
            "sequence_id":sequence_id,
            "layout":layout_dict,
            "bboxes":bbox_list,
            "gt_bboxes":gt_bbox_list
        }
        save_dict_list.append(save_dict)
    return save_dict_list