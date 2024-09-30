''' 随机选点且返回距离值的'''
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


class disDataset(Dataset):
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
    
    
    def process_dir(self,path):
        data:list[dict[str,dict[int,list[(str,None)]]]]=[]

        mesh_list=sorted(list(os.listdir(path)))
        arglist=[(mesh,os.path.join(path,mesh)) for mesh in mesh_list]
        for arg in tqdm.tqdm(arglist):
            data.append(self.process_mesh(arg))
        return data

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
        
        

    def get_negative(self,correspondence,flat1,flat2,deform1,deform2,pc1,pc2):
        negative_num=self.config.train_config.num_negative
        corr_num=correspondence.shape[0]
        #negative_id corr_num*negative_num//2
        negative_id_1=np.random.choice(deform1['visible_indices'],(corr_num,negative_num//2),replace=True)
        negative_id_2=np.random.choice(deform2['visible_indices'],(corr_num,negative_num//2),replace=True)
        negative_id_1=negative_id_1.reshape(-1)
        negative_id_2=negative_id_2.reshape(-1)
        #negative_world corr_num*negative_num//2*3
        negative_world_1=deform1['vertices'][negative_id_1]
        negative_world_2=deform2['vertices'][negative_id_2]
        negative_pixel_1=self.batch_world_to_pixel(negative_world_1,deform1['intrisics'],deform1['extrisics'])
        negative_pixel_2=self.batch_world_to_pixel(negative_world_2,deform2['intrisics'],deform2['extrisics'])
        #negative_pcd corr_num*negative_num//2*3
        negative_pcd_1=self.batch_pixel_to_pcd(negative_pixel_1,deform1['depth'],deform1['intrisics'])
        negative_pcd_2=self.batch_pixel_to_pcd(negative_pixel_2,deform2['depth'],deform2['intrisics'])
        negative_pcdid_1=np.argmin(np.linalg.norm(pc1[np.newaxis,:,:3]-negative_pcd_1[:,np.newaxis,:],axis=2),axis=1)
        negative_pcdid_2=np.argmin(np.linalg.norm(pc2[np.newaxis,:,:3]-negative_pcd_2[:,np.newaxis,:],axis=2),axis=1)
        negative_pcdid_1=negative_pcdid_1.reshape(corr_num,negative_num//2)
        negative_pcdid_2=negative_pcdid_2.reshape(corr_num,negative_num//2)
        negative_pcdid=np.concatenate((negative_pcdid_1,negative_pcdid_2),axis=1)
        negative_id_1=negative_id_1.reshape(corr_num,negative_num//2)
        negative_id_2=negative_id_2.reshape(corr_num,negative_num//2)

        negative_dist_1=np.linalg.norm(flat1['vertices'][correspondence[:,0]][:,np.newaxis,:]-flat1['vertices'][negative_id_1],axis=2)
        negative_dist_2=np.linalg.norm(flat2['vertices'][correspondence[:,1]][:,np.newaxis,:]-flat2['vertices'][negative_id_2],axis=2)
        negative_dist=np.concatenate((negative_dist_1,negative_dist_2),axis=1)
        # print("negtive_dist",negative_dist.shape)
        # print("negative_pcdid",negative_pcdid.shape)

        negative=torch.cat((torch.tensor(negative_pcdid).unsqueeze(0),torch.tensor(negative_dist).unsqueeze(0)),dim=0)
        # print("negative",negative.shape)
        return negative



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
        #downsample pc1 and pc2 to 10000 points
        if pc1.shape[0]>10000:
            index=np.random.choice(pc1.shape[0],10000,replace=False)
            pc1=pc1[index]
            colors1=colors1[index]
        if pc2.shape[0]>10000:
            index=np.random.choice(pc2.shape[0],10000,replace=False)
            pc2=pc2[index]
            colors2=colors2[index]
        kp_id1=kp1["kp_id"]
        kp_id2=kp2["kp_id"]

        shape_correspondence,correspondence_id=self.get_obj_correspondence(flat1,flat2,deform1,deform2,kp_id1,kp_id2,pc1,pc2)


        pc1=torch.cat((torch.tensor(pc1),torch.tensor(colors1)),dim=1).float()
        pc2=torch.cat((torch.tensor(pc2),torch.tensor(colors2)),dim=1).float()
        pc1[:,2]=6-2*pc1[:,2]
        pc2[:,2]=6-2*pc2[:,2]

        return pc1,pc2,shape_correspondence,self.get_negative(correspondence_id,flat1,flat2,deform1,deform2,pc1,pc2)
    
    def get_cross_deform(self,index):
        re_info1,re_info2=self.deform_info_pair[self.deform_index[index//2]]
        flat=self.process_deform_file(re_info1[0])
        deform1=self.process_deform_file(re_info1[1])
        deform2=self.process_deform_file(re_info2[1])
        pc1=deform1['points']
        pc2=deform2['points']
        colors1=deform1['colors']
        colors2=deform2['colors']
        #downsample pc1 and pc2 to 10000 points
        if pc1.shape[0]>10000:
            index=np.random.choice(pc1.shape[0],10000,replace=False)
            pc1=pc1[index]
            colors1=colors1[index]
        if pc2.shape[0]>10000:
            index=np.random.choice(pc2.shape[0],10000,replace=False)
            pc2=pc2[index]
            colors2=colors2[index]

        correspondence,correspondence_id=self.get_deform_correspondence(pc1,pc2,flat,deform1,deform2)

        pc1=torch.cat((torch.tensor(pc1),torch.tensor(colors1)),dim=1).float()
        pc2=torch.cat((torch.tensor(pc2),torch.tensor(colors2)),dim=1).float()
        pc1[:,2]=6-2*pc1[:,2]
        pc2[:,2]=6-2*pc2[:,2]
        return pc1,pc2,correspondence,self.get_negative(correspondence_id,flat,flat,deform1,deform2,pc1,pc2)
    
    
    
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
        correspondences=np.random.choice(np.array(correspondences),20,replace=True)

        deform1_world=deform1['vertices'][correspondences]
        deform2_world=deform2['vertices'][correspondences]
        deform1_pixel=self.batch_world_to_pixel(deform1_world,deform1['intrisics'],deform1['extrisics'])
        deform2_pixel=self.batch_world_to_pixel(deform2_world,deform2['intrisics'],deform2['extrisics'])
        deform1_pcd=self.batch_pixel_to_pcd(deform1_pixel,deform1['depth'],deform1['intrisics'])
        deform2_pcd=self.batch_pixel_to_pcd(deform2_pixel,deform2['depth'],deform2['intrisics'])
        deform1_pcdid=np.argmin(np.linalg.norm(pc1[np.newaxis,:,:3]-deform1_pcd[:,np.newaxis,:],axis=2),axis=1)
        deform2_pcdid=np.argmin(np.linalg.norm(pc2[np.newaxis,:,:3]-deform2_pcd[:,np.newaxis,:],axis=2),axis=1)
        correspondences_id=np.concatenate((correspondences.reshape(-1,1),correspondences.reshape(-1,1)),axis=1)
        return torch.cat((torch.tensor(deform1_pcdid).reshape(self.config.train_config.correspondence_num,1),torch.tensor(deform2_pcdid).reshape(self.config.train_config.correspondence_num,1)),dim=1),torch.tensor(correspondences_id)
    
    #get cross_shape trainning_data
    def get_obj_correspondence(self,flat1,flat2,deform1,deform2,kp_id1,kp_id2,pc1,pc2):
        correspondences_id=[]
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
                correspondences_id.append((keypoint1,keypoint2))

        correspondences_id=np.array(correspondences_id)
        correspondences_id=correspondences_id[np.random.choice(len(correspondences_id),self.config.train_config.correspondence_num,replace=True)]
        kp_world_1=deform1["vertices"][correspondences_id[:,0]]
        kp_world_2=deform2["vertices"][correspondences_id[:,1]]
        kp_pixel_1=self.batch_world_to_pixel(kp_world_1,deform1["intrisics"],deform1["extrisics"])
        kp_pixel_2=self.batch_world_to_pixel(kp_world_2,deform2["intrisics"],deform2["extrisics"])
        kp_pcd_1=self.batch_pixel_to_pcd(kp_pixel_1,deform1["depth"],deform1["intrisics"])
        kp_pcd_2=self.batch_pixel_to_pcd(kp_pixel_2,deform2["depth"],deform2["intrisics"])
        kp_pcdid_1=np.argmin(np.linalg.norm(pc1[np.newaxis,:,:3]-kp_pcd_1[:,np.newaxis,:],axis=2),axis=1)
        kp_pcdid_2=np.argmin(np.linalg.norm(pc2[np.newaxis,:,:3]-kp_pcd_2[:,np.newaxis,:],axis=2),axis=1)
        return torch.cat((torch.tensor(kp_pcdid_1).reshape(self.config.train_config.correspondence_num,1),torch.tensor(kp_pcdid_2).reshape(self.config.train_config.correspondence_num,1)),dim=1),torch.tensor(correspondences_id)
    
    

    def __getitem__(self,index):
        if index%2==0:
            return self.get_cross_shape(index)
        else:
            return self.get_cross_deform(index)


        

if __name__=="__main__":
    config=Config()
    dataset=disDataset(deform_path="/home/luhr/correspondence/softgym_cloth/cloth3d_train_data",object_path="/home/luhr/correspondence/softgym_cloth/garmentgym/tops",config=config,flag="train")
    for i in range(100):
        pc1_t,pc2_t,corr_t,negative=dataset[i]
        pc1_t[:,0]-=1
        pc2_t[:,0]+=1
        pc1=o3d.geometry.PointCloud()
        pc2=o3d.geometry.PointCloud()
        pc1.points=o3d.utility.Vector3dVector(pc1_t[:,:3])
        pc2.points=o3d.utility.Vector3dVector(pc2_t[:,:3])
        pc1.colors=o3d.utility.Vector3dVector(pc1_t[:,3:6])
        pc2.colors=o3d.utility.Vector3dVector(pc2_t[:,3:6])
        corr_t=corr_t.numpy()
        corr_t=corr_t.tolist()
        corr_visual=o3d.geometry.LineSet().create_from_point_cloud_correspondences(pc1,pc2,corr_t)
        # 随机生成线条的颜色
        # 随机生成线条的颜色
        colors = np.random.rand(len(corr_t), 3)

        # 设置线条颜色
        corr_visual.colors = o3d.utility.Vector3dVector(colors)
        print("show correspondence")
        o3d.visualization.draw_geometries([pc1,pc2,corr_visual])
        # print("show smooth1")
        # smooth1_visual=o3d.geometry.LineSet().create_from_point_cloud_correspondences(pc1,pc1,smooth_info1[:,:2].numpy().astype(np.int32).tolist())
        # o3d.visualization.draw_geometries([pc1,pc1,smooth1_visual])
        # print("show smooth2")
        # smooth2_visual=o3d.geometry.LineSet().create_from_point_cloud_correspondences(pc2,pc2,smooth_info2[:,:2].numpy().astype(np.int32).tolist())
        # o3d.visualization.draw_geometries([pc2,pc2,smooth2_visual])

        # home=corr_t[0][0]
        # positive=corr_t[1]
        # negative=neg_t[0]
        # negshow_corr=[(int(home),int(nega)) for nega in negative.tolist()]
        # neg_visual=o3d.geometry.LineSet().create_from_point_cloud_correspondences([pc1,pc2,negshow_corr[:30]])
        # o3d.visualization.draw_geometries([pc1,pc2,negshow_corr[:30]])

        
        
      