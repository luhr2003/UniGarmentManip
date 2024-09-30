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
from garmentgym.garmentgym.base.record import task_info
from garmentgym.garmentgym.env.fling import FlingEnv
import open3d as o3d
import torch.nn.functional as F
import pyflex
from typing import List

class visualize:
    def __init__(self,pc1,pc2):
        self.pc1=pc1.cpu().numpy()
        self.pc2=pc2.cpu().numpy()
        self.pcd1=o3d.geometry.PointCloud()
        self.pcd1_position=self.pc1[:,:3]
        self.pcd1_position=self.standardize_bbox(self.pcd1_position)
        self.pcd1_color=self.colormap(self.pcd1_position)
        self.pcd1_position[:,0]-=0.6
        self.pcd1.points=o3d.utility.Vector3dVector(self.pcd1_position)
        self.pcd1.colors=o3d.utility.Vector3dVector(self.pcd1_color)
        # o3d.visualization.draw_geometries([pcd1])
        self.pcd2=o3d.geometry.PointCloud()
        self.pcd2_position=self.pc2[:,:3]
        self.pcd2_position=self.standardize_bbox(self.pcd2_position)
        self.pcd2_position[:,0]+=0.6
        self.pcd2.points=o3d.utility.Vector3dVector(self.pcd2_position)


        self.query_id=None
    
    
    def show_match(self,query_id):
        query_id=query_id.cpu().numpy()
        query_id=query_id.astype(np.int32)
        self.query_id=query_id
        self.pcd2_color=self.pcd1_color[query_id]
        self.pcd2_color=self.pcd2_color.reshape(-1,3)
        self.pcd2.colors=o3d.utility.Vector3dVector(self.pcd2_color)
        o3d.visualization.draw_geometries([self.pcd1,self.pcd2])
    
    @staticmethod
    def standardize_bbox(pcl):
        mins = np.amin(pcl, axis=0)
        maxs = np.amax(pcl, axis=0)
        center = (mins + maxs) / 2.
        scale = np.amax(maxs-mins)
        print("Center: {}, Scale: {}".format(center, scale))
        result = ((pcl - center)/scale).astype(np.float32)  # [-0.5, 0.5]
        return result
    @staticmethod
    def colormap(pcl):
        # color_map = np.zeros_like(pcl)
        # for i in range(pcl.shape[0]):
        #     vec = np.array(
        #         [pcl[i, 0]+0.5, pcl[i, 1]+0.5, pcl[i, 2]+0.5-0.0125])
        #     vec = np.clip(vec, 0.001, 1.0)
        #     norm = np.sqrt(np.sum(vec**2))
        #     vec /= norm
        #     color_map[i][0] = vec[0]
        #     color_map[i][1] = vec[1]
        #     color_map[i][2] = vec[2]
        # return color_map
        color_map=deepcopy(pcl)
        color_map[:,0]+=0.5
        color_map[:,1]+=0.5
        color_map[:,2]+=(0.5-0.0125)
        color_map=np.clip(color_map,0.001,1.0)
        norm=np.sqrt(np.sum(color_map**2,axis=1))
        color_map=color_map/norm[:,np.newaxis]
        return color_map


def pixel_to_world(pixel_coordinates, depth, camera_intrinsics, camera_extrinsics):
        # 将像素坐标点转换为相机坐标系
        camera_coordinates = np.dot(np.linalg.inv(camera_intrinsics), np.append(pixel_coordinates, 1.0))
        camera_coordinates *= depth

        # 将相机坐标系中的点转换为世界坐标系
        world_point = np.dot(np.linalg.inv(camera_extrinsics), np.append(camera_coordinates, 1.0))
        world_point[2]=-world_point[2]
        return world_point[:3]



def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]



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

def pixel_to_pcd(pixel_coordinates,depth,camera_intrinsics):
    z=depth[pixel_coordinates[1],pixel_coordinates[0]]
    x=(pixel_coordinates[0]-camera_intrinsics[0,2])*z/camera_intrinsics[0,0]
    y=(pixel_coordinates[1]-camera_intrinsics[1,2])*z/camera_intrinsics[1,1]
    return np.array([x,y,z])

def pcd_to_pixel(pcd,camera_intrinsics):
    x=pcd[0]*camera_intrinsics[0,0]/pcd[2]+camera_intrinsics[0,2]
    y=pcd[1]*camera_intrinsics[1,1]/pcd[2]+camera_intrinsics[1,2]
    return np.array([x,y])

class Fling_Demo:
    def __init__(self,mesh_id,pc,point:dict,pc_ori):
        self.mesh_id=mesh_id
        self.pc=pc
        self.point=point
        self.pc_ori=pc_ori


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--current_cloth',type=str,default='/home/luhr/correspondence/softgym_cloth/garmentgym/tops')
    parser.add_argument('--mesh_id',type=str,default="00044")
    parser.add_argument('--store_path',type=str,default="/home/luhr/correspondence/softgym_cloth/task/fling")

    args=parser.parse_args()
    current_cloth=args.current_cloth
    mesh_id=args.mesh_id
    store_path=args.store_path




    print("---------------load flat cloth----------------")
    # load flat cloth
    env=FlingEnv(current_cloth,store_path="./",id=mesh_id)
    for j in range(50):
        pyflex.step()
        pyflex.render()


    flat_info:task_info=env.get_cur_info()
    

    

    for j in range(50):
        pyflex.step()
        pyflex.render()



    #---------------------calculate query point---------------------
    flat_pos=flat_info.cur_info.vertices
    left_shoulder_world=flat_pos[flat_info.clothes.left_shoulder]
    right_shoulder_world=flat_pos[flat_info.clothes.right_shoulder]
    left_sleeve_world=flat_pos[flat_info.clothes.top_left]
    right_sleeve_world=flat_pos[flat_info.clothes.top_right]
    left_bottom_world=flat_pos[flat_info.clothes.bottom_left]
    right_bottom_world=flat_pos[flat_info.clothes.bottom_right]
    left_sleeve_middle=(left_sleeve_world+left_shoulder_world)/2
    right_sleeve_middle=(right_sleeve_world+right_shoulder_world)/2

    camera_intrinsics=flat_info.config.get_camera_matrix()[0]
    camera_extrinsics=flat_info.config.get_camera_matrix()[1]
    left_shoulder_pixel=world_to_pixel_valid(left_shoulder_world,flat_info.cur_info.depth,camera_intrinsics,camera_extrinsics).astype(np.int32)
    right_shoulder_pixel=world_to_pixel_valid(right_shoulder_world,flat_info.cur_info.depth,camera_intrinsics,camera_extrinsics).astype(np.int32)
    left_sleeve_pixel=world_to_pixel_valid(left_sleeve_world,flat_info.cur_info.depth,camera_intrinsics,camera_extrinsics).astype(np.int32)
    right_sleeve_pixel=world_to_pixel_valid(right_sleeve_world,flat_info.cur_info.depth,camera_intrinsics,camera_extrinsics).astype(np.int32)
    left_bottom_pixel=world_to_pixel_valid(left_bottom_world,flat_info.cur_info.depth,camera_intrinsics,camera_extrinsics).astype(np.int32)
    right_bottom_pixel=world_to_pixel_valid(right_bottom_world,flat_info.cur_info.depth,camera_intrinsics,camera_extrinsics).astype(np.int32)
    left_sleeve_middle_pixel=world_to_pixel_valid(left_sleeve_middle,flat_info.cur_info.depth,camera_intrinsics,camera_extrinsics).astype(np.int32)
    right_sleeve_middle_pixel=world_to_pixel_valid(right_sleeve_middle,flat_info.cur_info.depth,camera_intrinsics,camera_extrinsics).astype(np.int32)

    left_shoulder_pcd=pixel_to_pcd(left_shoulder_pixel,flat_info.cur_info.depth,camera_intrinsics)
    right_shoulder_pcd=pixel_to_pcd(right_shoulder_pixel,flat_info.cur_info.depth,camera_intrinsics)
    left_sleeve_pcd=pixel_to_pcd(left_sleeve_pixel,flat_info.cur_info.depth,camera_intrinsics)
    right_sleeve_pcd=pixel_to_pcd(right_sleeve_pixel,flat_info.cur_info.depth,camera_intrinsics)
    left_bottom_pcd=pixel_to_pcd(left_bottom_pixel,flat_info.cur_info.depth,camera_intrinsics)
    right_bottom_pcd=pixel_to_pcd(right_bottom_pixel,flat_info.cur_info.depth,camera_intrinsics)
    left_sleeve_middle_pcd=pixel_to_pcd(left_sleeve_middle_pixel,flat_info.cur_info.depth,camera_intrinsics)
    right_sleeve_middle_pcd=pixel_to_pcd(right_sleeve_middle_pixel,flat_info.cur_info.depth,camera_intrinsics)
    left_shoulder_pcd=left_shoulder_pcd.reshape(1,3)
    right_shoulder_pcd=right_shoulder_pcd.reshape(1,3)
    left_sleeve_pcd=left_sleeve_pcd.reshape(1,3)
    right_sleeve_pcd=right_sleeve_pcd.reshape(1,3)
    left_bottom_pcd=left_bottom_pcd.reshape(1,3)
    right_bottom_pcd=right_bottom_pcd.reshape(1,3)
    left_sleeve_middle_pcd=left_sleeve_middle_pcd.reshape(1,3)
    right_sleeve_middle_pcd=right_sleeve_middle_pcd.reshape(1,3)

    flat_pc_points=np.array(flat_info.cur_info.points)
    flat_pc_colors=np.array(flat_info.cur_info.colors)
    flat_pc=np.concatenate([flat_pc_points,flat_pc_colors],axis=1)
    flat_pc_index=np.arange(len(flat_pc))
    flat_pc=flat_pc[np.random.choice(flat_pc_index,10000,replace=True)]

    left_shoulder_id=np.argmin(np.linalg.norm(flat_pc[:,:3]-left_shoulder_pcd,axis=1))
    right_shoulder_id=np.argmin(np.linalg.norm(flat_pc[:,:3]-right_shoulder_pcd,axis=1))
    left_sleeve_id=np.argmin(np.linalg.norm(flat_pc[:,:3]-left_sleeve_pcd,axis=1))
    right_sleeve_id=np.argmin(np.linalg.norm(flat_pc[:,:3]-right_sleeve_pcd,axis=1))
    left_bottom_id=np.argmin(np.linalg.norm(flat_pc[:,:3]-left_bottom_pcd,axis=1))
    right_bottom_id=np.argmin(np.linalg.norm(flat_pc[:,:3]-right_bottom_pcd,axis=1))
    left_sleeve_middle_id=np.argmin(np.linalg.norm(flat_pc[:,:3]-left_sleeve_middle_pcd,axis=1))
    right_sleeve_middle_id=np.argmin(np.linalg.norm(flat_pc[:,:3]-right_sleeve_middle_pcd,axis=1))


    flat_pc_ready=torch.from_numpy(flat_pc).float()
    flat_pc_ready[:,2]=6-2*flat_pc_ready[:,2]
    flat_pc_ready=flat_pc_ready.unsqueeze(0)

    fling_info=Fling_Demo(mesh_id=mesh_id,pc=flat_pc_ready,point={"left_shoulder":left_shoulder_id,"right_shoulder":right_shoulder_id,"left_sleeve":left_sleeve_id,"right_sleeve":right_sleeve_id,"left_bottom":left_bottom_id,"right_bottom":right_bottom_id,"left_sleeve_middle":left_sleeve_middle_id,"right_sleeve_middle":right_sleeve_middle_id},pc_ori=flat_pc)
    
    path=os.path.join(store_path,str(mesh_id)+".pkl")
    with open(path,"wb") as f:
        pickle.dump(fling_info,f)
    print("---------------load flat cloth done----------------")


    


    