from copy import deepcopy
import os
import pickle 
import sys
from typing import List
import cv2
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
curpath=os.getcwd()
sys.path.append(curpath)
sys.path.append(os.path.join(curpath,'garmentgym'))
sys.path.append(curpath+'/train')
import torch
from train.model.basic_pn import basic_model
import argparse
from  garmentgym.garmentgym.base.record import task_info
from garmentgym.garmentgym.env.fling_fold import FlingFoldEnv
import open3d as o3d
import torch.nn.functional as F
import pyflex
from typing import List
from garmentgym.garmentgym.base.config import *
from task.fling.fling_info import Fling_Demo
from garmentgym.garmentgym.base.config import Task_result

device="cuda:0"



def pcd_to_pixel(pcd,camera_intrinsics):
    x=pcd[0]*camera_intrinsics[0,0]/pcd[2]+camera_intrinsics[0,2]
    y=pcd[1]*camera_intrinsics[1,1]/pcd[2]+camera_intrinsics[1,2]
    return np.array([x,y])


def pixel_to_world(pixel_coordinates, depth, camera_intrinsics, camera_extrinsics):
        # 将像素坐标点转换为相机坐标系
        camera_coordinates = np.dot(np.linalg.inv(camera_intrinsics), np.append(pixel_coordinates, 1.0))
        camera_coordinates *= depth

        # 将相机坐标系中的点转换为世界坐标系
        world_point = np.dot(np.linalg.inv(camera_extrinsics), np.append(camera_coordinates, 1.0))
        world_point[2]=-world_point[2]
        return world_point[:3]






def world_to_pixel(world_point, camera_intrinsics, camera_extrinsics):
        # 将世界坐标点转换为相机坐标系
        #u 是宽
        world_point[2]=-world_point[2]
        camera_point = np.dot(camera_extrinsics, np.append(world_point, 1.0))
        # 将相机坐标点转换为像素坐标系
        pixel_coordinates = np.dot(camera_intrinsics, camera_point[:3])
        pixel_coordinates /= pixel_coordinates[2]
        return pixel_coordinates[:2]

def world_to_pixel_valid(world_point,depth,camera_intrinsics,camera_extrinsics):
    # 将世界坐标点转换为相机坐标系
    #u 是宽
    world_point[2]=-world_point[2]
    camera_point = np.dot(camera_extrinsics, np.append(world_point, 1.0))
    # 将相机坐标点转换为像素坐标系
    pixel_coordinates = np.dot(camera_intrinsics, camera_point[:3])
    pixel_coordinates /= pixel_coordinates[2]


    x, y = pixel_coordinates[:2]
    depth=depth.reshape((depth.shape[0],depth.shape[1]))
    height, width = depth.shape

    # Generate coordinate matrices for all pixels in the depth map
    X, Y = np.meshgrid(np.arange(height), np.arange(width))

    # Calculate Euclidean distances from each pixel to the given coordinate
    distances = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
    # Mask depth map to exclude zero depth values
    nonzero_mask = depth != 0

    # Apply mask to distances and find the minimum distance
    min_distance = np.min(distances[nonzero_mask])

    # Generate a boolean mask for the nearest non-zero depth point
    nearest_mask = (distances == min_distance) & nonzero_mask

    # Get the coordinates of the nearest non-zero depth point
    nearest_coordinate = (X[nearest_mask][0], Y[nearest_mask][0])

    return np.array(nearest_coordinate)

class Fling_process:
    def __init__(self,model,demo_path,env: FlingFoldEnv):
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

            # env.update_camera(1)
            if shoulder_pair_score>sleeve_pair_score and shoulder_pair_score>bottom_pair_score and shoulder_pair_score>sleeve_middle_pair_score:
                env.pick_and_fling_primitive_new(cur_left_shoulder_world,cur_right_shoulder_world)
                print("shoulder")
                self.pick_point="shoulder"
            elif sleeve_pair_score>shoulder_pair_score and sleeve_pair_score>bottom_pair_score and sleeve_pair_score>sleeve_middle_pair_score:
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

            cur_area=env.compute_coverage()
            if cur_area/env.clothes.init_coverage>0.8:
                break
        
        



if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str,default="simple")
    parser.add_argument('--demonstration',type=str,default='/home/isaac/correspondence/UniGarmentManip/demonstration/fold/simple_fold2/00044')
    parser.add_argument("--fling_demonstration",type=str,default="/home/isaac/correspondence/UniGarmentManip/demonstration/fling/00044.pkl")
    parser.add_argument('--current_cloth',type=str,default='/home/isaac/correspondence/UniGarmentManip/garmentgym/cloth3d/train/')
    parser.add_argument('--id',type=str,default='00037')
    parser.add_argument('--model_path',type=str,default='/home/isaac/correspondence/UniGarmentManip/checkpoint/tops.pth')
    parser.add_argument('--store_path',type=str,default='./flingfold_tshirt')
    parser.add_argument('--log_file', type=str,default="fling_fold_simple.pkl")
    parser.add_argument("--device",type=str,default="cuda:0")
    parser.add_argument("--iter",type=int,default=0)
    args=parser.parse_args()
    demonstration=args.demonstration
    current_cloth=args.current_cloth
    model_path=args.model_path
    fling_demonstration=args.fling_demonstration
    id=args.id
    log_file=args.log_file
    task_name=args.task_name
    store_path=args.store_path
    config=Config()
    iter=args.iter
    
    os.makedirs(store_path,exist_ok=True)
    store_path=os.path.join(store_path,str(id))
    os.makedirs(store_path,exist_ok=True)
    store_path=os.path.join(store_path,str(iter))
    os.makedirs(store_path,exist_ok=True)



    print("---------------load model----------------")
    # load model
    model=basic_model(512).cuda()
    model.load_state_dict(torch.load(model_path,map_location="cuda:0")["model_state_dict"])
    print('load model from {}'.format(model_path))
    model.eval()

    print("---------------load demonstration sequence----------------")
    # load demonstration sequence
    info_sequence=list()
    for i in sorted(os.listdir(demonstration)):
        if i.endswith('.pkl'):
            with open(os.path.join(demonstration,i),'rb') as f:
                print("load {}".format(f))
                info_sequence.append(pickle.load(f))


    print("---------------load flat cloth----------------")
    # load flat cloth
    env=FlingFoldEnv(current_cloth,store_path=store_path,id=id,config=config)
    for j in range(50):
        pyflex.step()
        env.step_sim_fn()

    
    flat_info=env.get_cur_info()

    print("---------------start deform----------------")
    var=0.4
    env.move_sleeve(var)
    env.move_bottom(var)
    env.move_left_right(var)

    


    for j in range(50):
        pyflex.step()
        env.step_sim_fn()
    
    #env.update_camera(2)

    print("---------------start fling----------------")
    # start fling
    process=Fling_process(model,fling_demonstration,env)
    process.start_fling()
    for j in range(20):
        pyflex.step()
        env.step_sim_fn()

    cur_info=env.get_cur_info()

    print("---------------start fold----------------")
    print("len(info_sequence):",len(info_sequence))
    
    for i in range(len(info_sequence)-1):
        demo=info_sequence[i]
        cur_shape:task_info=env.get_cur_info()
        #-------------prepare pc--------------
        cur_pc_points=torch.tensor(cur_shape.cur_info.points).float()
        cur_pc_colors=torch.tensor(cur_shape.cur_info.colors).float()
        cur_pc=torch.cat([cur_pc_points,cur_pc_colors],dim=1)

        demo_pc=torch.tensor(demo.cur_info.points).float()
        demo_colors=torch.tensor(demo.cur_info.colors).float()
        demo_pc=torch.cat([demo_pc,demo_colors],dim=1)

        # down sample to 10000
        if len(demo_pc)>10000:
            demo_pc=demo_pc[torch.randperm(len(demo_pc))[:10000]]
        if len(cur_pc)>10000:
            cur_pc=cur_pc[torch.randperm(len(cur_pc))[:10000]]

        #-------------calculate query point--------------
        cur_action=info_sequence[i+1].action[-1]
        action_function=cur_action[0]
        action_points=cur_action[1]
        action_pcd=[]
        action_id=[]
        
        for point in action_points:
            point_pixel=world_to_pixel_valid(point,demo.cur_info.depth,demo.config.get_camera_matrix()[0],demo.config.get_camera_matrix()[1]).astype(np.int32)
            cam_matrix=demo.config.get_camera_matrix()[0]
            z=demo.cur_info.depth[point_pixel[1],point_pixel[0]]
            x=(point_pixel[0]-cam_matrix[0,2])*z/cam_matrix[0,0]
            y=(point_pixel[1]-cam_matrix[1,2])*z/cam_matrix[1,1]
            point_pcd=np.array([x,y,z])
            action_pcd.append(point_pcd)
            point_pcd=point_pcd.reshape(1,3)
            point_id=np.argmin(np.linalg.norm(demo_pc[:,:3]-point_pcd,axis=1))
            action_id.append(point_id)


        #-------------pass network--------------

        #通过网络
        demo_pc_ready=deepcopy(demo_pc)
        demo_pc_ready[:,2]=6-2*demo_pc_ready[:,2]
        cur_pc_ready=deepcopy(cur_pc)
        cur_pc_ready[:,2]=6-2*cur_pc_ready[:,2]

        demo_pc_ready=demo_pc_ready.cuda()
        cur_pc_ready=cur_pc_ready.cuda()

        demo_pc_ready=demo_pc_ready.unsqueeze(0)
        cur_pc_ready=cur_pc_ready.unsqueeze(0)
        demo_feature=model(demo_pc_ready)
        cur_feature=model(cur_pc_ready)
        demo_feature=F.normalize(demo_feature,dim=-1)
        cur_feature=F.normalize(cur_feature,dim=-1)
        demo_feature=demo_feature[0]
        cur_feature=cur_feature[0]

        #-------------find correspondence--------------
        cur_pc=cur_pc.numpy()
        action_world=[]
        cur_pcd=[]
        for id in action_id:
            cur_pcd_id=torch.argmax(torch.sum(demo_feature[id]*cur_feature,dim=1,keepdim=True))
            cur_pcd.append(cur_pcd_id)
            cur_matrix=cur_shape.config.get_camera_matrix()[0]
            action_rgbd=np.zeros((2))
            action_rgbd[0]=cur_pc[cur_pcd_id,0]*cur_matrix[0,0]/cur_pc[cur_pcd_id,2]+cur_matrix[0,2]
            action_rgbd[1]=cur_pc[cur_pcd_id,1]*cur_matrix[1,1]/cur_pc[cur_pcd_id,2]+cur_matrix[1,2]
            cur_world=pixel_to_world(action_rgbd,cur_pc[cur_pcd_id,2],cur_shape.config.get_camera_matrix()[0],cur_shape.config.get_camera_matrix()[1])
            action_world.append(cur_world)
        
        #-------------execute action--------------
        # env.update_camera(1)
        print("action:",action_function)
        env.execute_action([action_function,action_world])
        
        for j in range(50):
            pyflex.step()
            env.step_sim_fn()
        
        
        
    #-------------check success--------------
    result=env.check_success(type=task_name)
    print("fold result:",result)
    env.export_image()
    #-------------save result--------------
    log_file_path=os.path.join(store_path,log_file)
    # print("log_file_path:",log_file_path)
    with open(log_file_path,"rb") as f:
        task_result:Task_result=pickle.load(f)
        task_result.current_num+=1
        if result:
            task_result.success_num+=1
        else:
            task_result.fail_num+=1
        task_result.success_rate=task_result.success_num/task_result.current_num
        task_result.result_dict[id]=result
    print(task_result)
    with open(log_file_path,"wb") as f:
        pickle.dump(task_result,f)
        
        
        




    

