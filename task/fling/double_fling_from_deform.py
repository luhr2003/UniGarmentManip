'''fling pipeline'''


import os
import pickle 
import sys
from typing import List

import numpy as np

curpath=os.getcwd()
sys.path.append(curpath)
sys.path.append(os.path.join(curpath,'garmentgym'))
sys.path.append(curpath+'/train')
import torch
from train.model.basic_pn import basic_model
import argparse
from garmentgym.garmentgym.base.record import task_info
from garmentgym.garmentgym.env.fling import FlingEnv
import open3d as o3d
import torch.nn.functional as F
import pyflex
from typing import List
from garmentgym.garmentgym.base.config import Task_result
from task.fling.fling_info import Fling_Demo
from garmentgym.garmentgym.base.config import *



class Fling_process:
    def __init__(self,model,demo_path,env: FlingEnv):
        self.model=model
        self.demo_path=demo_path
        self.env=env
        self.load_demonstration()
        self.pick_point=None
        

    def load_demonstration(self):
        with open(self.demo_path,'rb') as f:
            demo:Fling_Demo=pickle.load(f)
    
        self.flat_pc_ready=demo.pc
        self.point=demo.point
        self.left_shoulder_id=self.point['left_shoulder']
        self.right_shoulder_id=self.point['right_shoulder']
        self.left_sleeve_id=self.point['left_sleeve']
        self.right_sleeve_id=self.point['right_sleeve']
        self.left_bottom_id=self.point['left_bottom']
        self.right_bottom_id=self.point['right_bottom']
        self.left_sleeve_middle_id=self.point['left_sleeve_middle']
        self.right_sleeve_middle_id=self.point['right_sleeve_middle']
        self.pc_ori=demo.pc_ori
    
    def start_fling(self):
        for i in range(3):
            env.update_camera(0)
            cur_info=env.get_cur_info()
            cur_pc_points=np.array(cur_info.cur_info.points)
            cur_pc_colors=np.array(cur_info.cur_info.colors)
            cur_pc=np.concatenate([cur_pc_points,cur_pc_colors],axis=1)
            cur_pc_index=np.arange(len(cur_pc))
            cur_pc=cur_pc[np.random.choice(cur_pc_index,10000,replace=True)]
            cur_pc_ready=torch.from_numpy(cur_pc).float()
            cur_pc_ready[:,2]=6-2*cur_pc_ready[:,2]
            cur_pc_ready=cur_pc_ready.unsqueeze(0)

            camera_intrinsics=cur_info.config.get_camera_matrix()[0]
            camera_extrinsics=cur_info.config.get_camera_matrix()[1]

            #---------------------pass network---------------------

            self.flat_pc_ready=self.flat_pc_ready.cuda()
            cur_pc_ready=cur_pc_ready.cuda()

            flat_feature=model(self.flat_pc_ready)
            cur_feature=model(cur_pc_ready)
            flat_feature=F.normalize(flat_feature,dim=-1)
            cur_feature=F.normalize(cur_feature,dim=-1)
            flat_feature=flat_feature[0]
            cur_feature=cur_feature[0]

            #---------------------calculate correspondence---------------------
            cur_left_shoulder_pcd_id=torch.argmax(torch.sum(flat_feature[self.left_shoulder_id]*cur_feature,dim=1,keepdim=True))
            cur_right_shoulder_pcd_id=torch.argmax(torch.sum(flat_feature[self.right_shoulder_id]*cur_feature,dim=1,keepdim=True))
            cur_left_sleeve_pcd_id=torch.argmax(torch.sum(flat_feature[self.left_sleeve_id]*cur_feature,dim=1,keepdim=True))
            cur_right_sleeve_pcd_id=torch.argmax(torch.sum(flat_feature[self.right_sleeve_id]*cur_feature,dim=1,keepdim=True))
            cur_left_bottom_pcd_id=torch.argmax(torch.sum(flat_feature[self.left_bottom_id]*cur_feature,dim=1,keepdim=True))
            cur_right_bottom_pcd_id=torch.argmax(torch.sum(flat_feature[self.right_bottom_id]*cur_feature,dim=1,keepdim=True))
            cur_left_sleeve_middle_pcd_id=torch.argmax(torch.sum(flat_feature[self.left_sleeve_middle_id]*cur_feature,dim=1,keepdim=True))
            cur_right_sleeve_middle_pcd_id=torch.argmax(torch.sum(flat_feature[self.right_sleeve_middle_id]*cur_feature,dim=1,keepdim=True))

            



            cur_left_shoulder_pixel=pcd_to_pixel(cur_pc[cur_left_shoulder_pcd_id],camera_intrinsics)
            cur_right_shoulder_pixel=pcd_to_pixel(cur_pc[cur_right_shoulder_pcd_id],camera_intrinsics)
            cur_left_sleeve_pixel=pcd_to_pixel(cur_pc[cur_left_sleeve_pcd_id],camera_intrinsics)
            cur_right_sleeve_pixel=pcd_to_pixel(cur_pc[cur_right_sleeve_pcd_id],camera_intrinsics)
            cur_left_bottom_pixel=pcd_to_pixel(cur_pc[cur_left_bottom_pcd_id],camera_intrinsics)
            cur_right_bottom_pixel=pcd_to_pixel(cur_pc[cur_right_bottom_pcd_id],camera_intrinsics)
            cur_left_sleeve_middle_pixel=pcd_to_pixel(cur_pc[cur_left_sleeve_middle_pcd_id],camera_intrinsics)
            cur_right_sleeve_middle_pixel=pcd_to_pixel(cur_pc[cur_right_sleeve_middle_pcd_id],camera_intrinsics)

            cur_left_shoulder_world=pixel_to_world(cur_left_shoulder_pixel,cur_pc[cur_left_shoulder_pcd_id,2],camera_intrinsics,camera_extrinsics)
            cur_right_shoulder_world=pixel_to_world(cur_right_shoulder_pixel,cur_pc[cur_right_shoulder_pcd_id,2],camera_intrinsics,camera_extrinsics)
            cur_left_sleeve_world=pixel_to_world(cur_left_sleeve_pixel,cur_pc[cur_left_sleeve_pcd_id,2],camera_intrinsics,camera_extrinsics)
            cur_right_sleeve_world=pixel_to_world(cur_right_sleeve_pixel,cur_pc[cur_right_sleeve_pcd_id,2],camera_intrinsics,camera_extrinsics)
            cur_left_bottom_world=pixel_to_world(cur_left_bottom_pixel,cur_pc[cur_left_bottom_pcd_id,2],camera_intrinsics,camera_extrinsics)
            cur_right_bottom_world=pixel_to_world(cur_right_bottom_pixel,cur_pc[cur_right_bottom_pcd_id,2],camera_intrinsics,camera_extrinsics)
            cur_left_sleeve_middle_world=pixel_to_world(cur_left_sleeve_middle_pixel,cur_pc[cur_left_sleeve_middle_pcd_id,2],camera_intrinsics,camera_extrinsics)
            cur_right_sleeve_middle_world=pixel_to_world(cur_right_sleeve_middle_pixel,cur_pc[cur_right_sleeve_middle_pcd_id,2],camera_intrinsics,camera_extrinsics)
            

            left_shoulder_score=torch.sum(flat_feature[self.left_shoulder_id]*cur_feature[cur_left_shoulder_pcd_id])
            right_shoulder_score=torch.sum(flat_feature[self.right_shoulder_id]*cur_feature[cur_right_shoulder_pcd_id])
            left_sleeve_score=torch.sum(flat_feature[self.left_sleeve_id]*cur_feature[cur_left_sleeve_pcd_id])
            right_sleeve_score=torch.sum(flat_feature[self.right_sleeve_id]*cur_feature[cur_right_sleeve_pcd_id])
            left_bottom_score=torch.sum(flat_feature[self.left_bottom_id]*cur_feature[cur_left_bottom_pcd_id])
            right_bottom_score=torch.sum(flat_feature[self.right_bottom_id]*cur_feature[cur_right_bottom_pcd_id])
            left_sleeve_middle_score=torch.sum(flat_feature[self.left_sleeve_middle_id]*cur_feature[cur_left_sleeve_middle_pcd_id])
            right_sleeve_middle_score=torch.sum(flat_feature[self.right_sleeve_middle_id]*cur_feature[cur_right_sleeve_middle_pcd_id])

            

            #---------------------ready_pair_for_fling--------------------------
            shoulder_pair_score=(left_shoulder_score+right_shoulder_score)/2
            sleeve_pair_score=(left_sleeve_score+right_sleeve_score)/2
            bottom_pair_score=(left_bottom_score+right_bottom_score)/2
            sleeve_middle_pair_score=(left_sleeve_middle_score+right_sleeve_middle_score)/2






            env.update_camera(3)
            if i==0:
                env.pick_and_fling_primitive_new(cur_left_shoulder_world,cur_right_shoulder_world)
                print("shoulder")
                self.pick_point="shoulder"
            elif i==1:
                env.pick_and_fling_primitive_new(cur_left_sleeve_world,cur_right_sleeve_world)
                print("sleeve")
                self.pick_point="sleeve"
            elif bottom_pair_score>shoulder_pair_score and bottom_pair_score>sleeve_pair_score and bottom_pair_score>sleeve_middle_pair_score:
                env.pick_and_fling_primitive_bottom(cur_left_bottom_world,cur_right_bottom_world)
                print("bottom")
                self.pick_point="bottom"
            else:
                env.pick_and_fling_primitive_new(cur_left_sleeve_middle_world,cur_right_sleeve_middle_world)
                print("middle")
                self.pick_point="middle"

            center_object()
            env.update_camera(0)
            cur_area=env.compute_coverage()

            if cur_area/init_area>0.8:
                print("pass")
                break

def center_object():
    pos = pyflex.get_positions().reshape(-1, 4)
    mid_x = (np.max(pos[:, 0]) + np.min(pos[:, 0]))/2
    mid_y = (np.max(pos[:, 2]) + np.min(pos[:, 2]))/2
    pos[:, [0, 2]] -= np.array([mid_x, mid_y])
    pyflex.set_positions(pos.flatten())
    env.step_sim_fn()






def pixel_to_world(pixel_coordinates, depth, camera_intrinsics, camera_extrinsics):
        # 将像素坐标点转换为相机坐标系
        camera_coordinates = np.dot(np.linalg.inv(camera_intrinsics), np.append(pixel_coordinates, 1.0))
        camera_coordinates *= depth

        # 将相机坐标系中的点转换为世界坐标系
        world_point = np.dot(np.linalg.inv(camera_extrinsics), np.append(camera_coordinates, 1.0))
        world_point[2]=-world_point[2]
        return world_point[:3]

def pcd_to_pixel(pcd,camera_intrinsics):
    x=pcd[0]*camera_intrinsics[0,0]/pcd[2]+camera_intrinsics[0,2]
    y=pcd[1]*camera_intrinsics[1,1]/pcd[2]+camera_intrinsics[1,2]
    return np.array([x,y])




if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--current_cloth',type=str,default='/home/isaac/correspondence/softgym_cloth/garmentgym/cloth3d/train')
    parser.add_argument('--model_path',type=str,default='./UniGarmentManip/checkpoint/tops.pth')
    parser.add_argument('--mesh_id',type=str,default="00037")
    parser.add_argument("--log_file",type=str,default="fling_test/log.json")
    parser.add_argument("--store_path",type=str,default="fling_test")
    parser.add_argument("--demonstration",type=str,default="./UniGarmentManip/demonstration/fling/00044.pkl")
    parser.add_argument("--device",type=str,default="cuda:0")
    parser.add_argument("--iter",type=int,default=0)


    args=parser.parse_args()
    current_cloth=args.current_cloth
    model_path=args.model_path
    mesh_id=args.mesh_id
    demonstration=args.demonstration
    log_file=args.log_file
    store_path=args.store_path
    device=args.device
    iter=args.iter
    config=Config()




    
    print("---------------load model----------------")
    # load model
    model=basic_model(512).cuda()
    model.load_state_dict(torch.load(model_path,map_location="cuda:0")["model_state_dict"])
    print('load model from {}'.format(model_path))
    model.eval()


    print("---------------load flat cloth----------------")
    # load flat cloth
    env=FlingEnv(current_cloth,store_path=store_path,id=mesh_id,config=config)
    for j in range(50):
        env.step_sim_fn()



    #calculate flat cloth area
    init_area = env.compute_coverage()
    flat_info:task_info=env.get_cur_info()
    print("---------------start deform----------------")
    env.throw_down()
    

    for j in range(50):
        env.step_sim_fn()

    print("---------------start fling----------------")
    # start fling
    process=Fling_process(model,demonstration,env)
    process.start_fling()
    center_object()
    for j in range(50):
        env.step_sim_fn()

    cur_area=env.compute_coverage()


    env.update_camera(1)

    for j in range(50):
        env.step_sim_fn()

    if(cur_area/init_area>0.85):
        fling_result=True
    else:
        fling_result=False
    print("fling result",fling_result)

    env.export_image()
       
    #-------------save result--------------
    log_file_path=os.path.join(store_path,log_file)
    with open(log_file_path,"rb") as f:
        task_result:Task_result=pickle.load(f)
        task_result.current_num+=1
        if fling_result:
            task_result.success_num+=1
        else:
            task_result.fail_num+=1
        task_result.success_rate=task_result.success_num/task_result.current_num
        task_result.result_dict[mesh_id]=fling_result
    with open(log_file_path,"wb") as f:
        pickle.dump(task_result,f)
    print(task_result)
    
    
    
    
        





    

