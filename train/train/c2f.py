'''coarse to fine single can visualize'''
import argparse
from copy import deepcopy
import random
import socket
import cv2
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
import tqdm
import torch.nn.functional as F
from train.model.basic_pn import basic_model
from info_nce import InfoNCE
import tqdm
import matplotlib.pyplot as plt
def standardize_bbox(pcl:torch.tensor):
        # pcl=pcl
        mins = np.amin(pcl, axis=0)
        maxs = np.amax(pcl, axis=0)
        center = (mins + maxs) / 2.
        scale = np.amax(maxs-mins)
        result = ((pcl - center)/scale).astype(np.float32)  # [-0.5, 0.5]
        return result

def colormap(pointcloud):
        base_point = np.copy(pointcloud[0])
        distance = np.zeros((pointcloud.shape[0],1))
        point1 = np.copy(pointcloud[0])
        point2 = np.copy(pointcloud[0])
        for i in range(pointcloud.shape[0]):#最左下的点
            if pointcloud[i][0]+pointcloud[i][1]<base_point[0]+base_point[1]:
                base_point=pointcloud[i]
        for i in range(pointcloud.shape[0]):#最左上的点(255,0,255)
            if pointcloud[i][0]-pointcloud[i][1]<point1[0]-point1[1]:
                point1 = pointcloud[i]
        for i in range(pointcloud.shape[0]):#最右上的点(170,0,255)
            if pointcloud[i][0]+pointcloud[i][1]>point2[0]+point2[1]:
                point2 = pointcloud[i]
        
        base_point[0]-=0.02
        for i in range(pointcloud.shape[0]):
            distance[i] = np.linalg.norm(pointcloud[i] - base_point)
        max_value = np.max(distance)
        min_value = np.min(distance)
        cmap = plt.cm.get_cmap('jet_r')
        colors = cmap((-distance+max_value)/(max_value-min_value))
        colors = np.reshape(colors,(-1,4))
        color_map = np.zeros((pointcloud.shape[0], 3))
        i=0
        for color in colors:
            color_map[i] = color[:3]
            i=i+1
        color_map2 = np.zeros_like(color_map)
        for i in range(pointcloud.shape[0]):
            distance1 = np.linalg.norm(point1-pointcloud[i])
            distance2 = np.linalg.norm(point2-pointcloud[i])
            dis = np.abs(point1[1]-pointcloud[i][1])
            if dis < 0.4:
                color_map2[i] = np.array([75.0/255.0,0.0,130.0/255.0])*distance2/(distance1+distance2) + np.array([1.0,20.0/255.0,147.0/255.0])*distance1/(distance1+distance2)
        for i in range(pointcloud.shape[0]):
            distance1 = np.linalg.norm(point1-pointcloud[i])
            distance2 = np.linalg.norm(point2-pointcloud[i])
            distance3 = np.linalg.norm(point1-point2)
            dis = np.abs(point1[1]-pointcloud[i][1])
            if dis<0.4:
                color_map[i] = color_map[i]*(dis)/(0.4) + (color_map2[i])*(0.4-dis)/(0.4)
            
        return color_map

class visualize:
    def __init__(self,pc1,pc2,correspondence):
        self.pc1=pc1.numpy()
        self.pc2=pc2.numpy()
        self.correspondence=correspondence.cpu().numpy().astype(np.int32)
        self.pcd1=o3d.geometry.PointCloud()
        self.pcd1_position=self.pc1[:,:3]
        self.pcd1_position=standardize_bbox(self.pcd1_position)
        self.pcd1_color=colormap(self.pcd1_position)
        self.pcd1_position[:,0]-=0.6
        self.pcd1.points=o3d.utility.Vector3dVector(self.pcd1_position)
        self.pcd1.colors=o3d.utility.Vector3dVector(self.pcd1_color)
        self.pcd2=o3d.geometry.PointCloud()
        self.pcd2_position=self.pc2[:,:3]
        self.pcd2_position=standardize_bbox(self.pcd2_position)
        self.pcd2_position[:,0]+=0.6
        self.pcd2.points=o3d.utility.Vector3dVector(self.pcd2_position)

        self.pcd2_color=self.pcd1_color[self.correspondence]
        self.pcd2.colors=o3d.utility.Vector3dVector(self.pcd2_color)

        self.query_id=None
    
    def show_gt(self):
        self.pcd2_color=self.pcd1_color[self.correspondence]
        self.pcd2.colors=o3d.utility.Vector3dVector(self.pcd2_color)
        o3d.visualization.draw_geometries([self.pcd1,self.pcd2])
    
    def show_match(self,query_id):
        query_id=query_id.cpu().numpy()
        query_id=query_id.astype(np.int32)
        self.query_id=query_id
        self.pcd2_color=self.pcd1_color[query_id]
        self.pcd2_color=self.pcd2_color.reshape(-1,3)
        self.pcd2.colors=o3d.utility.Vector3dVector(self.pcd2_color)
        o3d.visualization.draw_geometries([self.pcd1,self.pcd2])
    
    def show_mistake(self,mask):
        mask=mask.cpu().numpy()
        self.pcd2_color[mask]=np.array([0,0,0])
        self.pcd2.colors=o3d.utility.Vector3dVector(self.pcd2_color)
        o3d.visualization.draw_geometries([self.pcd1,self.pcd2])


class C2fDataset(Dataset):
    def __init__(self,deform_path:str,device:str,store_path:str,model:basic_model):
        self.deform_path=deform_path
        self.oriinfo=self.process_dir(self.deform_path)
        self.valid_list=self.get_flat_deform_pair(self.oriinfo)
        self.dataset_len=len(self.valid_list)
        self.device=device
        self.store_path=store_path
        self.model=model
        print("mesh_num:",self.dataset_len)
        self.criterion=InfoNCE(negative_mode='paired',temperature=0.1)
        self.optimizer=torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=1e-5)


        
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

    def get_flat_deform_pair(self,deform_data):
        flat_deform_pair=[]
        for mesh_data in deform_data:
            mesh_id=list(mesh_data.keys())[0]
            mesh_form=mesh_data[mesh_id]
            if len(mesh_form)<=0:
                continue
            for level in range(1,21):
                if level not in mesh_form.keys():
                    continue
                mesh_level=mesh_form[level]
                for shape in mesh_level:
                   flat_deform_pair.append((mesh_form[0][0][0],shape[0],mesh_id))
        return flat_deform_pair
        


    def __len__(self):
        return len(self.valid_list)
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
        return torch.cat((u.reshape((-1,1)),v.reshape((-1,1))),dim=1)
    @staticmethod
    def batch_pixel_to_world(pixels,pc,camera_intrisics,camera_extrisics):
        camera_intrisics=torch.inverse(torch.tensor(camera_intrisics)).to(device).double()
        camera_extrisics=torch.inverse(torch.tensor(camera_extrisics)).to(device).double()
        pixels=pixels.double()
        camera_coordinates=torch.matmul(camera_intrisics,torch.cat((pixels,torch.ones((pixels.shape[0],1)).to(device)),dim=1).T)
        camera_coordinates*=pc[:,2]
        world_point=torch.matmul(camera_extrisics,torch.cat((camera_coordinates,torch.ones((1,camera_coordinates.shape[1])).to(device)),dim=0))
        world_point=world_point[:3].T
        world_point[:,2]=-world_point[:,2]
        return world_point
    

    @staticmethod
    def batch_world_to_pixel(world_points, camera_intrinsics, camera_extrinsics):
        # 将世界坐标点转换为相机坐标系
        world_points[:, 2] = -world_points[:, 2]
        camera_intrinsics=torch.tensor(camera_intrinsics).to(device).double()
        camera_extrinsics=torch.tensor(camera_extrinsics).to(device).double()
        
        # 扩展为齐次坐标并进行相机坐标转换
        camera_points = torch.matmul(camera_extrinsics, torch.cat((world_points, torch.ones((world_points.shape[0], 1)).to(device)),dim=1).T)
        
        # 将相机坐标点转换为像素坐标系
        pixel_coordinates = torch.matmul(camera_intrinsics, camera_points[:3,:]).T
        pixel_coordinates /= pixel_coordinates[:, 2].reshape(-1, 1)
        # regulate the pixel_coordinates to the range of image
        pixel_coordinates[:,0]=torch.clip(pixel_coordinates[:,0],0,camera_intrinsics[0,2]*2-1)
        pixel_coordinates[:,1]=torch.clip(pixel_coordinates[:,1],0,camera_intrinsics[1,2]*2-1)
        return pixel_coordinates[:, :2].long()
    @staticmethod
    def batch_pixel_to_pcd(pixel_coordinates, depth, camera_intrinsics):
            # 获取深度图像中对应像素坐标点的深度值
            depth=torch.tensor(depth).to(device).double()
            camera_intrinsics=torch.tensor(camera_intrinsics).to(device).double()
            z = depth[pixel_coordinates[:, 1], pixel_coordinates[:, 0]]
            
            # 计算每个像素坐标点对应的X和Y坐标值
            x = (pixel_coordinates[:, 0].reshape(-1,1) - camera_intrinsics[0, 2]) * z / camera_intrinsics[0, 0]
            y = (pixel_coordinates[:, 1].reshape(-1,1) - camera_intrinsics[1, 2]) * z / camera_intrinsics[1, 1]
            
            # 创建点云坐标（PCD）的numpy数组
            pcd=torch.cat((x,y,z),dim=1)
            
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
            faces=np.array(f['faces'][:])
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
        info['faces']=faces
        return info
        
    
    def get_cross_deform(self,index):
        re_info=self.valid_list[index]
        flat=self.process_deform_file(re_info[0])
        deform=self.process_deform_file(re_info[1])
        pc1=flat['points']
        pc2=deform['points']
        colors1=flat['colors']
        colors2=deform['colors']
        deform_info=deepcopy(deform)
        #downsample pc1 and pc2 to 10000 points
        if pc1.shape[0]>10000:
            index=np.random.choice(pc1.shape[0],10000,replace=False)
            pc1=pc1[index]
            colors1=colors1[index]
        if pc2.shape[0]>10000:
            index=np.random.choice(pc2.shape[0],10000,replace=False)
            pc2=pc2[index]
            colors2=colors2[index]
        
        pc1=torch.tensor(pc1).to(device)
        pc2=torch.tensor(pc2).to(device)

        correspondence=self.get_gt_correspondence(pc1,pc2,flat,deform)





        pc1=pc1.cpu()
        pc2=pc2.cpu()
        pc1=torch.cat((pc1,torch.tensor(colors1)),dim=1).double()
        pc2=torch.cat((pc2,torch.tensor(colors2)),dim=1).double()
        pc1[:,2]=6-2*pc1[:,2]
        pc2[:,2]=6-2*pc2[:,2]
        return pc1,pc2,correspondence
    
    
    
    def get_gt_correspondence(self,pc1,pc2,flat,deform):
        with torch.no_grad():
            pc2_pixel=self.batch_pcd_to_pixel(pc2,flat['intrisics'])
            pc2_pixel=pc2_pixel.long()
            pc2_pixel[:,0]=torch.clip(pc2_pixel[:,0],0,flat['intrisics'][0,2]*2-1)
            pc2_pixel[:,1]=torch.clip(pc2_pixel[:,1],0,flat['intrisics'][1,2]*2-1)
            pc2_world=self.batch_pixel_to_world(pc2_pixel,pc2,flat['intrisics'],flat['extrisics'])

            deform_world_sim=deform['vertices']
            deform_world_sim=torch.tensor(deform['vertices']).to(device).double()
            pc2_world[:,1]=torch.max(deform_world_sim[:,1])+0.1
            
            pc2_world_flat=pc2_world[:,(0,2)]
            deform_world_sim_flat=deform_world_sim[:,(0,2)]
            

            pc2_world_id_flat=torch.zeros((pc2.shape[0],5)).long().to(device)
            for i in range(pc2.shape[0]):
                pc2_world_id_flat[i]=torch.argsort(torch.norm(deform_world_sim_flat-pc2_world_flat[i],dim=1))[:5]
            
            deform_world_sim_flat=deform_world_sim[pc2_world_id_flat.reshape((-1,))]
            deform_world_sim_flat=deform_world_sim_flat.reshape((pc2_world_id_flat.shape[0],pc2_world_id_flat.shape[1],3))

    


            min_index=torch.argmin(torch.abs(deform_world_sim_flat[:,:,1]-pc2_world[:,1].unsqueeze(1)),dim=1)
            pc2_world_id=pc2_world_id_flat.gather(1,min_index.reshape((-1,1))).reshape(-1)

            flat_world_sim=flat['vertices']
            flat_world_sim=torch.tensor(flat_world_sim).to(device).double()
            pc1_world_sim=flat_world_sim[pc2_world_id]
            pc1_pixel_sim=self.batch_world_to_pixel(pc1_world_sim,flat['intrisics'],flat['extrisics'])
            pc1_pixel_sim=pc1_pixel_sim.long()
            pc1_pixel_sim[:,0]=torch.clip(pc1_pixel_sim[:,0],0,flat['intrisics'][0,2]*2-1)
            pc1_pixel_sim[:,1]=torch.clip(pc1_pixel_sim[:,1],0,flat['intrisics'][1,2]*2-1)
            pc1_pcd=self.batch_pixel_to_pcd(pc1_pixel_sim,flat['depth'],flat['intrisics'])
            correspondence_id=torch.zeros((pc1.shape[0],)).long().to(device) 
            
            for i in range(pc1.shape[0]):
                correspondence_id[i]=torch.argmin(torch.norm(pc1-pc1_pcd[i],dim=1))
            return correspondence_id
        


    def process(self,index):
        flat,deform,gt_correspondence=self.get_cross_deform(index)
        vis=visualize(flat,deform,gt_correspondence)
        # vis.show_gt()
        gt_correspondence=gt_correspondence.long()
        for j in range(5):
            flat_feature=self.model(flat.unsqueeze(0).float().to(self.device))[0]
            flat_feature=F.normalize(flat_feature,dim=-1)
            deform_feature=self.model(deform.unsqueeze(0).float().to(self.device))[0]
            deform_feature=F.normalize(deform_feature,dim=-1)
            
            deform_query_result=torch.zeros((flat.shape[0])).to(self.device)
            for j in range(flat.shape[0]):
                deform_query_result[j]=torch.argmax(torch.sum(deform_feature[j]*flat_feature,dim=1))
            
            # vis.show_match(deform_query_result)
            deform_query_result=deform_query_result.long()
            gt_corr_pos=flat[gt_correspondence,:3]
            query_corr_pos=flat[deform_query_result,:3]
            distance=torch.sum((gt_corr_pos-query_corr_pos)**2,dim=1)
            mistake=distance>0.05
            mistake_index=torch.nonzero(mistake).reshape((-1,))
            if len(mistake_index)<=100:
                break
            # vis.show_mistake(mistake)
            deform_query_result=torch.zeros((mistake_index.shape[0]),150).to(self.device)
            for j in range(mistake_index.shape[0]):
                deform_query_result[j]=torch.argsort(torch.sum(deform_feature[mistake_index[j]]*flat_feature,dim=1),descending=True)[:150]
            
            deform_query_result=deform_query_result.long()



            randindex=torch.randperm(flat.shape[0])[:200]

            if mistake_index.shape[0]>500:
                mistake_index=mistake_index[torch.randperm(mistake_index.shape[0])[:500]]

            
            query=deform_feature[mistake_index]
            query=torch.cat((query,deform_feature[randindex]),dim=0)
            positive=flat_feature[gt_correspondence[mistake_index]]
            positive=torch.cat((positive,flat_feature[gt_correspondence[randindex]]),dim=0)
            negative=flat_feature[deform_query_result.reshape((-1,))].reshape((mistake_index.shape[0],150,-1))
            rand_negative_index=torch.randint(0,flat.shape[0],(200,150)).to(self.device)
            negative=torch.cat((negative,flat_feature[rand_negative_index.reshape(-1)].reshape(200,150,-1)),dim=0)
            self.optimizer.zero_grad()
            loss=self.criterion(query,positive,negative)
            loss.backward()
            if loss.item()<4:
                break
            self.optimizer.step()
 

        
        
        
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--deform_path', type=str, default='/home/isaac/correspondence/tshirt_move')
    parser.add_argument('--model_path',type=str,default='/home/isaac/correspondence/UniGarmentManip/checkpoint/tops.pth')
    parser.add_argument('--device',type=str,default='cuda:0')
    parser.add_argument("--log_path",type=str,default="./c2f_log")
    args = parser.parse_args()
    deform_path=args.deform_path
    device=args.device
    model_path=args.model_path
    log_path=args.log_path

    model_store_path=os.makedirs(log_path,exist_ok=True)


    model=basic_model(512)
    model.load_state_dict(torch.load(model_path,map_location=device)['model_state_dict'])
    model=model.to(device)
    

    dataset=C2fDataset(deform_path,device,log_path,model)
    with tqdm.tqdm(total=len(dataset)) as pbar:
        for i in range(len(dataset)):
            dataset.process(i)

            if i % 100 == 0:
                torch.save({"model_state_dict":dataset.model.state_dict(),"optimizer_state_dict":dataset.optimizer.state_dict(),"optimizer":dataset.optimizer},os.path.join(log_path,"model_checkpoint_"+str(i)+".pth"))
            print("process {}th mesh".format(i))
            pbar.update(1)


    

    