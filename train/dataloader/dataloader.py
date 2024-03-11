''' 随机选点的dataloader'''
from copy import deepcopy
import random
import socket
import _pickle as pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import open3d as o3d
import functools
import sys

import tqdm
import h5py
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(),"train"))
from base.config import Config


class Dataset(Dataset):
    def __init__(self,deform_path:str,object_path:str,config:Config,flag:str,size=None):
        self.config=config
        self.deform_path=deform_path
        self.object_path=object_path
        self.oriinfo=self.process_dir(self.deform_path)
        self.valid_list=self.get_valid_list(self.oriinfo,config.data_config.deform_level)
        self.mesh_list=list(self.valid_list.keys())
        self.kp_data=self.process_cross_obj(self.object_path)
        if flag=="train":
            self.mesh_list=self.mesh_list[:int(len(self.mesh_list)*config.data_config.factor)]
        else:
            self.mesh_list=self.mesh_list[int(len(self.mesh_list)*config.data_config.factor):]
        self.mesh_num=len(self.mesh_list)
        print("mesh_num:",self.mesh_num)
        self.obj_info_pair=[]  #cross object pair
        for i in range(self.mesh_num):
            for j in range(i+1,self.mesh_num):
                for m in self.valid_list[self.mesh_list[i]]:
                    for n in self.valid_list[self.mesh_list[j]]:
                        self.obj_info_pair.append((m,n))
        self.deform_info_pair=[]
        for i in range(self.mesh_num):#cross_deform_pair
            for j in range(len(self.valid_list[self.mesh_list[i]])):
                for k in range(j+1,len(self.valid_list[self.mesh_list[i]])):
                    self.deform_info_pair.append((self.valid_list[self.mesh_list[i]][j],self.valid_list[self.mesh_list[i]][k]))
                    self.deform_info_pair.append((self.valid_list[self.mesh_list[i]][k],self.valid_list[self.mesh_list[i]][j]))
        self.obj_info_pair_len=len(self.obj_info_pair)
        self.deform_info_pair_len=len(self.deform_info_pair)
        self.deform_index=np.random.randint(0,self.deform_info_pair_len,self.obj_info_pair_len)
        print("dataset complete")

    def process_dir(self,path):
        data:list[dict[str,dict[int,list[(str,None)]]]]=[]

        mesh_list=sorted(list(os.listdir(path)))
        arglist=[(mesh,os.path.join(path,mesh)) for mesh in mesh_list]
        for arg in tqdm.tqdm(arglist):
            data.append(self.process_mesh(arg))
        return data






    def process_cross_obj(self,path):
        data:dict[str,dict[str,np.ndarray]]={}
        for mesh in tqdm.tqdm(os.listdir(path)):
            for file in os.listdir(os.path.join(path,mesh)):
                if file.endswith('50.npz'):
                    info=np.load(os.path.join(path,mesh,file))
                    keypoints=info['keypoints']
                    kp_id=info['keypoint_id']
                    pointcloud=info['pointcloud']
                    data[mesh]={"keypoints":keypoints,"kp_id":kp_id,"pointcloud":pointcloud}
        return data
    
    
    
    def process_mesh(self,arg):
        mesh=arg[0]
        mesh_dir=arg[1]
        data={}
        for iter in os.listdir(mesh_dir):
            iter_dir=os.path.join(mesh_dir,iter)
            for file in os.listdir(iter_dir):
                if file.endswith('.h5'):
                    file_path=os.path.join(iter_dir,file)
                    info=None
                    manipute_times=int(file.split('.')[0])
                    data.setdefault(int(manipute_times),[]).append((file_path,None))
        return {mesh:data}

    def get_valid_list(self,deform_data,deform_level):
        valid_info:dict[str,list]={}
        for mesh_data in deform_data:
            mesh_id=list(mesh_data.keys())[0]
            mesh_form=mesh_data[mesh_id]
            if len(mesh_form)<=0:
                continue
            valid_info[mesh_id]=[(mesh_form[0][0][0],mesh_form[0][0][0],mesh_id)]
            for level in range(1,deform_level):
                if level not in mesh_form.keys():
                    continue
                mesh_level=mesh_form[level]
                for shape in mesh_level:
                    valid_info[mesh_id].append((mesh_form[0][0][0],shape[0],mesh_id))
        return valid_info
        


    def __len__(self):
        return 2*self.obj_info_pair_len
    
    @staticmethod
    def pixel_to_world(pixel_coordinates, depth, camera_intrinsics, camera_extrinsics):
        # 将像素坐标点转换为相机坐标系
        camera_coordinates = np.dot(np.linalg.inv(camera_intrinsics), np.append(pixel_coordinates, 1.0))
        camera_coordinates *= depth

        # 将相机坐标系中的点转换为世界坐标系
        world_point = np.dot(np.linalg.inv(camera_extrinsics), np.append(camera_coordinates, 1.0))
        world_point[2]=-world_point[2]
        return world_point[:3]
    @staticmethod
    def world_to_pixel(world_point, camera_intrinsics, camera_extrinsics):
        # 将世界坐标点转换为相机坐标系
        #u 是宽
        world_point[2]=-world_point[2]
        camera_point = np.dot(camera_extrinsics, np.append(world_point, 1.0))
        # 将相机坐标点转换为像素坐标系
        pixel_coordinates = np.dot(camera_intrinsics, camera_point[:3])
        pixel_coordinates /= pixel_coordinates[2]
        # regulate the pixel_coordinates to the range of image
        pixel_coordinates[0]=np.clip(pixel_coordinates[0],0,camera_intrinsics[0,2]*2-1)
        pixel_coordinates[1]=np.clip(pixel_coordinates[1],0,camera_intrinsics[1,2]*2-1)
        return pixel_coordinates[:2]
    @staticmethod
    def batch_pcd_to_pixel(pcd,camera_intrisics):
        fx=camera_intrisics[0,0]
        fy=camera_intrisics[1,1]
        cx=camera_intrisics[0,2]
        cy=camera_intrisics[1,2]
        x=pcd[:,0]
        y=pcd[:,1]
        z=pcd[:,2]
        u=(fx*x/z)+cx
        v=(fy*y/z)+cy
        return np.array([u,v]).astype(np.int32)
    @staticmethod
    def batch_pixel_to_world(pixels,pc,camera_intrisics,camera_extrisics):
        camera_coordinates=np.dot(np.linalg.inv(camera_intrisics),np.concatenate((pixels,np.ones((pixels.shape[0],1))),axis=1).T)
        camera_coordinates*=pc[:,2]
        world_point=np.dot(np.linalg.inv(camera_extrisics),np.concatenate((camera_coordinates,np.ones((1,camera_coordinates.shape[1]))),axis=0))
        world_point=world_point[:3].T
        world_point[:,2]=-world_point[:,2]
        return world_point
    
    @staticmethod
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

    def get_near_vertices(self,flat_info,id):
        return np.argsort(np.linalg.norm(flat_info['vertices']-flat_info['vertices'][id],axis=1))[:100]

    def pixel_to_pcd(self,pixel_coordinates,depth,camera_intrinsics):
        z=depth[pixel_coordinates[1],pixel_coordinates[0]]
        x=(pixel_coordinates[0]-camera_intrinsics[0,2])*z/camera_intrinsics[0,0]
        y=(pixel_coordinates[1]-camera_intrinsics[1,2])*z/camera_intrinsics[1,1]
        return np.array([x,y,z])
    @staticmethod
    def batch_world_to_pixel(world_points, camera_intrinsics, camera_extrinsics):
        # 将世界坐标点转换为相机坐标系
        world_points[:, 2] = -world_points[:, 2]
        
        # 扩展为齐次坐标并进行相机坐标转换
        camera_points = np.matmul(camera_extrinsics, np.hstack((world_points, np.ones((world_points.shape[0], 1)))).T)
        
        # 将相机坐标点转换为像素坐标系
        pixel_coordinates = np.dot(camera_intrinsics, camera_points[:3,:]).T
        pixel_coordinates /= pixel_coordinates[:, 2].reshape(-1, 1)
        # regulate the pixel_coordinates to the range of image
        pixel_coordinates[:,0]=np.clip(pixel_coordinates[:,0],0,camera_intrinsics[0,2]*2-1)
        pixel_coordinates[:,1]=np.clip(pixel_coordinates[:,1],0,camera_intrinsics[1,2]*2-1)
        return pixel_coordinates[:, :2].astype(np.int32)
    @staticmethod
    def batch_pixel_to_pcd(pixel_coordinates, depth, camera_intrinsics):
            # 获取深度图像中对应像素坐标点的深度值
            z = depth[pixel_coordinates[:, 1], pixel_coordinates[:, 0]]
            
            index=np.arange(len(pixel_coordinates))
            # 计算每个像素坐标点对应的X和Y坐标值
            x = (pixel_coordinates[:, 0].reshape(-1,1) - camera_intrinsics[0, 2]) * z / camera_intrinsics[0, 0]
            y = (pixel_coordinates[:, 1].reshape(-1,1) - camera_intrinsics[1, 2]) * z / camera_intrinsics[1, 1]
            
            # 创建点云坐标（PCD）的numpy数组
            pcd = np.column_stack((x, y, z))
            
            return pcd
    
    @staticmethod
    def process_deform_file(file_path):
        with h5py.File(file_path,'r')as f:
            points = np.array(f['points'][:])
            colors = np.array(f['colors'][:])
            vertices = np.array(f['vertices'][:])
            visible_veritices = np.array(f['visible_vertices'][:])
            visible_indices = np.array(f['visible_indices'][:])
            intrisics = np.array(f['camera_intrisics'][:])
            extrisics = np.array(f['camera_extrisics'][:])
            rgb=np.array(f['rgb'][:])
            depth=np.array(f['depth'][:])
        info={}
        info['points']=points
        info['colors']=colors
        info['vertices']=vertices
        info['visible_veritices']=visible_veritices
        info['visible_indices']=visible_indices
        info['intrisics']=intrisics
        info['extrisics']=extrisics
        info['rgb']=rgb
        info['depth']=depth
        return info
        
    
    def get_cross_shape(self,index):
        re_info1,re_info2=self.obj_info_pair[index//2]
        flat1=self.process_deform_file(re_info1[0])
        deform1=self.process_deform_file(re_info1[1])
        kp1=self.kp_data[re_info1[2]]
        flat2=self.process_deform_file(re_info2[0])
        deform2=self.process_deform_file(re_info2[1])
        kp2=self.kp_data[re_info2[2]]
        pc1=deform1['points']
        pc2=deform2['points']
        colors1=deform1['colors']
        colors2=deform2['colors']
        if pc1.shape[0]<=0:
            pc1=flat1['points']
            colors1=flat1['colors']
            deform1=flat1
        if pc2.shape[0]<=0:
            pc2=flat2['points']
            colors2=flat2['colors']
            deform2=flat2
        #downsample pc1 and pc2 to 10000 points
        index=np.random.choice(pc1.shape[0],5000,replace=True)
        pc1=pc1[index]
        colors1=colors1[index]
        index=np.random.choice(pc2.shape[0],5000,replace=True)
        pc2=pc2[index]
        colors2=colors2[index]
        kp_id1=kp1["kp_id"]
        kp_id2=kp2["kp_id"]

        shape_correspondence=self.get_obj_correspondence(flat1,flat2,deform1,deform2,kp_id1,kp_id2,pc1,pc2)


        pc1=torch.cat((torch.tensor(pc1),torch.tensor(colors1)),dim=1).float()
        pc2=torch.cat((torch.tensor(pc2),torch.tensor(colors2)),dim=1).float()
        pc1[:,2]=6-2*pc1[:,2]
        pc2[:,2]=6-2*pc2[:,2]

        return pc1,pc2,shape_correspondence
    
    def get_cross_deform(self,index):
        re_info1,re_info2=self.deform_info_pair[self.deform_index[index//2]]
        flat=self.process_deform_file(re_info1[0])
        deform1=self.process_deform_file(re_info1[1])
        deform2=self.process_deform_file(re_info2[1])
        pc1=deform1['points']
        pc2=deform2['points']
        colors1=deform1['colors']
        colors2=deform2['colors']
        if pc1.shape[0]<=0:
            pc1=flat['points']
            colors1=flat['colors']
            deform1=flat
        if pc2.shape[0]<=0:
            pc2=flat['points']
            colors2=flat['colors']
            deform2=flat
        #downsample pc1 and pc2 to 10000 points
        index=np.random.choice(pc1.shape[0],5000,replace=True)
        pc1=pc1[index]
        colors1=colors1[index]
        index=np.random.choice(pc2.shape[0],5000,replace=True)
        pc2=pc2[index]
        colors2=colors2[index]

        correspondence=self.get_deform_correspondence(pc1,pc2,flat,deform1,deform2)






        pc1=torch.cat((torch.tensor(pc1),torch.tensor(colors1)),dim=1).float()
        pc2=torch.cat((torch.tensor(pc2),torch.tensor(colors2)),dim=1).float()
        pc1[:,2]=6-2*pc1[:,2]
        pc2[:,2]=6-2*pc2[:,2]
        return pc1,pc2,correspondence
    
    
    
    def get_deform_correspondence(self,pc1,pc2,flat,deform1,deform2):
        deform1_visible=deform1['visible_indices']
        deform2_visible=deform2['visible_indices']
        flat_vertices=flat['vertices']

        correspondences=[]
        deform1_ready=np.random.choice(deform1_visible,200,replace=False)
        for visibleid1 in deform1_ready:
            visibleid1group=self.get_near_vertices(flat,visibleid1)
            for id in visibleid1group:
                if id in deform2_visible:
                    correspondences.append(id)
                    break
        # for visibleid1 in deform1_ready:
        #     if visibleid1 in deform2_visible:
        #         correspondences.append(visibleid1)
        correspondences=np.random.choice(np.array(correspondences),20,replace=True)

        deform1_world=deform1['vertices'][correspondences]
        deform2_world=deform2['vertices'][correspondences]
        # print(deform1['intrisics'])
        # print(deform1['extrisics'])
        deform1_pixel=self.batch_world_to_pixel(deform1_world,deform1['intrisics'],deform1['extrisics'])
        deform2_pixel=self.batch_world_to_pixel(deform2_world,deform2['intrisics'],deform2['extrisics'])
        deform1_pcd=self.batch_pixel_to_pcd(deform1_pixel,deform1['depth'],deform1['intrisics'])
        deform2_pcd=self.batch_pixel_to_pcd(deform2_pixel,deform2['depth'],deform2['intrisics'])
        deform1_pcdid=np.argmin(np.linalg.norm(pc1[np.newaxis,:,:3]-deform1_pcd[:,np.newaxis,:],axis=2),axis=1)
        deform2_pcdid=np.argmin(np.linalg.norm(pc2[np.newaxis,:,:3]-deform2_pcd[:,np.newaxis,:],axis=2),axis=1)
        return torch.cat((torch.tensor(deform1_pcdid).reshape(self.config.train_config.correspondence_num,1),torch.tensor(deform2_pcdid).reshape(self.config.train_config.correspondence_num,1)),dim=1)
    
    #get cross_shape trainning_data
    def get_obj_correspondence(self,flat1,flat2,deform1,deform2,kp_id1,kp_id2,pc1,pc2):
        correspondences=[]
        for i in range(len(kp_id1)):
            keypoint1=None
            keypoint2=None
            keypoint_group1=self.get_near_vertices(flat1,kp_id1[i])
            keypoint_group2=self.get_near_vertices(flat2,kp_id2[i])
            for i in range(len(keypoint_group1)):
                if keypoint_group1[i] in deform1["visible_indices"]:
                    keypoint1=keypoint_group1[i]
                    break
            for i in range(len(keypoint_group2)):
                if keypoint_group2[i] in deform2["visible_indices"]:
                    keypoint2=keypoint_group2[i]
                    break
            if keypoint1==None or keypoint2==None:
                continue
            else:
                correspondences.append((keypoint1,keypoint2))

        correspondences=np.array(correspondences)
        correspondences=correspondences[np.random.choice(len(correspondences),self.config.train_config.correspondence_num,replace=True)]
        kp_world_1=deform1["vertices"][correspondences[:,0]]
        kp_world_2=deform2["vertices"][correspondences[:,1]]
        kp_pixel_1=self.batch_world_to_pixel(kp_world_1,deform1["intrisics"],deform1["extrisics"])
        kp_pixel_2=self.batch_world_to_pixel(kp_world_2,deform2["intrisics"],deform2["extrisics"])
        kp_pcd_1=self.batch_pixel_to_pcd(kp_pixel_1,deform1["depth"],deform1["intrisics"])
        kp_pcd_2=self.batch_pixel_to_pcd(kp_pixel_2,deform2["depth"],deform2["intrisics"])
        kp_pcdid_1=np.argmin(np.linalg.norm(pc1[np.newaxis,:,:3]-kp_pcd_1[:,np.newaxis,:],axis=2),axis=1)
        kp_pcdid_2=np.argmin(np.linalg.norm(pc2[np.newaxis,:,:3]-kp_pcd_2[:,np.newaxis,:],axis=2),axis=1)
        return torch.cat((torch.tensor(kp_pcdid_1).reshape(self.config.train_config.correspondence_num,1),torch.tensor(kp_pcdid_2).reshape(self.config.train_config.correspondence_num,1)),dim=1)
    
    

    def __getitem__(self,index):
        if index%2==0:
            return self.get_cross_shape(index)
        else:
            return self.get_cross_deform(index)


        
