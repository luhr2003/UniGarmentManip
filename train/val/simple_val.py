import os
import sys

import numpy as np
curpath=os.getcwd()
sys.path.append(curpath)
sys.path.append(os.path.join(curpath,'train'))

from base.config import Config
import argparse
from base.utils import *
from tensorboardX import SummaryWriter
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
from model.basic_pn import basic_model
import random
from info_nce import InfoNCE
import open3d as o3d

def val_cal_loss(feature1,feature2,correspondence,config:Config):
    # feature1 batchsize*num_points*feature_dim
    # feature2 batchsize*num_points*feature_dim
    # correspondence batchsize*num_correspondence*2
    batchsize=feature1.shape[0]
    num_correspondence=correspondence.shape[1]
    feature_dim=feature1.shape[2]

    batch_index=torch.arange(batchsize).to(config.train_config.device)
    #query batchsize*num_correspondence*feature_dim
    query=torch.stack([feature1[batch_index,correspondence[batch_index,i,0]] for i in range(num_correspondence)],dim=1)

    #positive batchsize*num_correspondence*feature_dim
    positive=torch.stack([feature2[batch_index,correspondence[batch_index,i,1]] for i in range(num_correspondence)],dim=1)

    #negative index batchsize*num_correspondence*negative_num
    negative_index=torch.randint(0,feature2.shape[1],(batchsize,num_correspondence,config.train_config.num_negative)).to(config.train_config.device)

    #negative batchsize*num_correspondence*negative_num*feature_dim
    negative_index=torch.randint(0,num_points,(batchsize,num_correspondence,config.train_config.num_negative))
    negative=torch.zeros(batchsize,num_correspondence,config.train_config.num_negative,feature_dim).to(config.train_config.device)
    for i in range(batchsize):
        for j in range(num_correspondence):
            for k in range(config.train_config.num_negative):
                negative[i,j,k]=feature2[i,negative_index[i,j,k]]

    criterion=InfoNCE(negative_mode="paired",temperature=config.train_config.temperature)
    query=query.reshape(batchsize*num_correspondence,feature_dim)
    positive=positive.reshape(batchsize*num_correspondence,feature_dim)
    negative=negative.reshape(batchsize*num_correspondence,config.train_config.num_negative,feature_dim)
    loss=criterion(query,positive,negative)
    return loss


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


def cal_inference_pair(feature1,feature2,correspondence,config):
    # feature1 batchsize*num_points*feature_dim
    # feature2 batchsize*num_points*feature_dim
    # correspondence batchsize*num_correspondence*2
    batchsize=feature1.shape[0]
    num_correspondence=correspondence.shape[1]
    feature_dim=feature1.shape[2]

    batch_index=torch.arange(batchsize).to(config.train_config.device)

    #query batchsize*num_correspondence*feature_dim
    # query=torch.stack([feature1[batch_index,correspondence[batch_index,i,0]] for i in range(num_correspondence)],dim=1)
    query=feature1.gather(1,correspondence[:,:,0].unsqueeze(-1).expand(-1,-1,feature_dim))
    query=F.normalize(query,dim=-1)
    feature2=F.normalize(feature2,dim=-1)
    #inferece batchsize*num_correspondence
    inference=torch.zeros(batchsize,num_correspondence).to(config.train_config.device)
    # for i in range(batchsize):
    #     for j in range(num_correspondence):
    #         inference[i,j]=torch.argmax(torch.sum(query[i,j]*feature2[i],dim=1,keepdim=True))
    # inference = torch.argmax(torch.sum(query[:,:,None,:] * feature2[:,None,:,:],dim=3),dim=2)# b x n x n x d
    for i in range(batchsize):
        inference[i]=torch.argmax(torch.sum(query[i,:,None,:]*feature2[i,None,:,:],dim=2),dim=1)
    return inference

def cal_distance_accuracy(pc1,pc2,inference,correspondence,config):
    #pc1 batchsize*num_points*3
    #pc2 batchsize*num_points*3

    batchsize=pc1.shape[0]
    num_points=pc1.shape[1]
    num_correspondence=correspondence.shape[1]

    batch_index=torch.arange(batchsize).to(config.train_config.device)
    #pc1_pos
    # pc1_pos=torch.stack([pc1[batch_index,correspondence[batch_index,i,0]] for i in range(num_correspondence)],dim=1)
    pc1_pos=pc1.gather(1,correspondence[:,:,0].unsqueeze(-1).expand(-1,-1,3))

    #gt_pos batchsize*num_correspondence*3
    # gt_pos=torch.stack([pc2[batch_index,correspondence[batch_index,i,1],:3] for i in range(num_correspondence)],dim=1)
    gt_pos=pc2.gather(1,correspondence[:,:,1].unsqueeze(-1).expand(-1,-1,3))

    #inference_pos batchsize*num_correspondence*3
    # print(inference)
    # inference_pos=torch.stack([pc2[batch_index,inference[batch_index,i],:3] for i in range(num_correspondence)],dim=1)
    inference_pos=pc2.gather(1,inference.unsqueeze(-1).expand(-1,-1,3))
    
    #cal distance
    # distance=torch.zeros(batchsize,num_correspondence).to(config.train_config.device)
    # for i in range(batchsize):
    #     for j in range(num_correspondence):
    #         # print(inference_pos[i,j,:3])
    #         # print(gt_pos[i,j,:3])
    #distance batchsize*num_correspondence
    distance=torch.norm(inference_pos.reshape(-1,3)-gt_pos.reshape(-1,3),dim=1).reshape(batchsize,num_correspondence).to(config.train_config.device)
    
    #cal accuracy
    correct=distance<config.train_config.distance_threshold
    #accuracy batchsize
    accuracy=torch.sum(correct,dim=1)/num_correspondence

    return distance.mean().mean(),accuracy.mean()

def visualize(pc1,pc2,inference,correspondence):
    #pc1 batchsize*num_points*3
    #pc2 batchsize*num_points*3
    #inference batchsize*num_correspondence
    #correspondence batchsize*num_correspondence*2
    batchsize=pc1.shape[0]
    num_correspondence=correspondence.shape[1]
    for i in range(batchsize):
        pcd1=o3d.geometry.PointCloud()
        points1=pc1[i][:,:3].cpu().numpy().reshape(-1,3)
        colors1=pc1[i][:,3:].cpu().numpy().reshape(-1,3)
        points1[:,0]-=0.5
        pcd1.points=o3d.utility.Vector3dVector(points1)
        pcd1.colors=o3d.utility.Vector3dVector(colors1)
        pcd2=o3d.geometry.PointCloud()
        points2=pc2[i][:,:3].cpu().numpy().reshape(-1,3)
        colors2=pc2[i][:,3:].cpu().numpy().reshape(-1,3)
        points2[:,0]+=0.5
        pcd2.points=o3d.utility.Vector3dVector(points2)
        pcd2.colors=o3d.utility.Vector3dVector(colors2)
        gt_correspondence=[]
        for j in range(num_correspondence):
            gt_correspondence.append([correspondence[i,j,0],correspondence[i,j,1]])
        inference_correspondence=[]
        for j in range(num_correspondence):
            inference_correspondence.append([correspondence[i,j,0],inference[i,j]])
        
        gt_corr=o3d.geometry.LineSet().create_from_point_cloud_correspondences(pcd1,pcd2,gt_correspondence)
        gt_corr.colors=o3d.utility.Vector3dVector(np.tile(np.array([0,1,0]),(len(gt_correspondence),1)))
        inference_corr=o3d.geometry.LineSet().create_from_point_cloud_correspondences(pcd1,pcd2,inference_correspondence)
        inference_corr.colors=o3d.utility.Vector3dVector(np.tile(np.array([1,0,0]),(len(inference_correspondence),1)))
        o3d.visualization.draw_geometries([pcd1,pcd2,gt_corr,inference_corr])




            










if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--model_path',type=str,default="/home/luhr/correspondence/softgym_cloth/checkpoint/new_naive_train/all_new.pth")
    parser.add_argument('--data_path',type=str,default="/home/luhr/correspondence/softgym_cloth/cloth3d_train_data")

    args=parser.parse_args()
    model_path=args.model_path
    data_path=args.data_path

    config=Config()
    data=process_dir(data_path)
    dataset=fileDataset(deform_path=data_path,object_path="/home/luhr/correspondence/softgym_cloth/garmentgym/cloth3d/train",config=config)
    dataloader=Data.DataLoader(dataset,batch_size=config.train_config.batch_size,shuffle=True,num_workers=4)

    model=basic_model(config.train_config.feature_dim)
    model.load_state_dict(torch.load(model_path,map_location=config.train_config.device)['model_state_dict'])
    model.to(config.train_config.device)

    model.eval()
    total_loss=0    
    for i,(pc1,pc2,correspondence) in enumerate(dataloader):
        batchsize=pc1.shape[0]
        num_points=pc1.shape[1]
        num_correspondence=correspondence.shape[1]

        pc1=pc1.to(config.train_config.device)
        pc2=pc2.to(config.train_config.device)
        correspondence=correspondence.to(config.train_config.device)

        with torch.no_grad():
            feature1=model(pc1)
            feature2=model(pc2)
            loss=val_cal_loss(feature1,feature2,correspondence,config)
            inference=cal_inference_pair(feature1,feature2,correspondence,config)
            inference=inference.long()
            visualize(pc1,pc2,inference,correspondence)
            print(cal_distance_accuracy(pc1,pc2,inference,correspondence,config))