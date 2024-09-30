
import os
import open3d as o3d

import torch
import merger.merger_net as merger_net
import json
import tqdm
import numpy as np
import argparse

arg_parser = argparse.ArgumentParser(description="Predictor for Skeleton Merger on KeypointNet dataset. Outputs a npz file with two arrays: kpcd - (N, k, 3) xyz coordinates of keypoints detected; nfact - (N, 2) normalization factor, or max and min coordinate values in a point cloud.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
arg_parser.add_argument('-i', '--obj-path', type=str, default='./UniGarmentManip/garmentgym/tops',
                        help='Point cloud file folder path from KeypointNet dataset.')
arg_parser.add_argument('-m', '--checkpoint-path', '--model-path', type=str, default='./UniGarmentManip/skeleton/tops.pth',
                        help='Model checkpoint file path to load.')
arg_parser.add_argument('-d', '--device', type=str, default='cuda',
                        help='Pytorch device for predicting.')
arg_parser.add_argument('-k', '--n-keypoint', type=int, default=50,
                        help='Requested number of keypoints to detect.')
arg_parser.add_argument('-b', '--batch', type=int, default=32,
                        help='Batch size.')
arg_parser.add_argument('--max-points', type=int, default=10000,
                        help='Indicates maximum points in each input point cloud.')
ns = arg_parser.parse_args()


def prepare_data(path):
    data=[]
    paths=[]
    ori_data=[]
    for root,dirs,files in os.walk(path):
        for file in files:
            if file.endswith('.pcd'):
                file_path=os.path.join(root,file)
                vertices=o3d.io.read_point_cloud(os.path.join(root,file))
                vertices=np.asarray(vertices.points)
                points=np.array(vertices)
                ori_data.append(points)
                points=points[np.random.choice(len(points),ns.max_points,replace=True)]
                data.append(points)
                paths.append(file_path)
    return ori_data,data,paths

def find_nearest_point(points,kp):
    kp=np.array(kp)
    points=np.array(points)
    dist=np.linalg.norm(kp[np.newaxis,:,:]-points[:,np.newaxis,:],axis=2)
    return np.argmin(dist,axis=0)

def find_nearest_point_group(points,kp):
    kp=np.array(kp)
    points=np.array(points)
    dist=np.linalg.norm(kp[:,np.newaxis,:]-points[np.newaxis,:,:],axis=2)
    return np.argsort(dist,axis=1)[:,:20]

def visualize_pointcloud(pc,kp):
    kp=kp.reshape(-1)
    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(pc)
    colors= np.zeros_like(pc)
    colors[kp]=np.array([1,0,0])
    pcd.colors=o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

def get_id_visualize(pc,kp):
    for i in range(len(kp)):
        visualize_pointcloud(pc,kp[i])
        print(i)


if __name__=='__main__':

    net = merger_net.Net(ns.max_points, ns.n_keypoint).to(ns.device)
    net.load_state_dict(torch.load(ns.checkpoint_path, map_location=torch.device(ns.device))['model_state_dict'])
    net.eval()

    ori_data,kpn_ds,paths=prepare_data(ns.obj_path)

    out_kpcd=[]
    for i in tqdm.tqdm(range(0, len(kpn_ds), ns.batch), unit_scale=ns.batch):
        Q = []
        for j in range(ns.batch):
            if i + j >= len(kpn_ds):
                continue
            pc = kpn_ds[i + j]
            Q.append(pc)
        if len(Q) == 1:
            Q.append(Q[-1])
        with torch.no_grad():
            recon, key_points, kpa, emb, null_activation = net(torch.Tensor(np.array(Q)).to(ns.device))
        for kp in key_points:
            out_kpcd.append(kp)
    for i in range(len(out_kpcd)):
        out_kpcd[i] = out_kpcd[i].cpu().numpy()
    for i in range(len(out_kpcd)):
        kp_id=find_nearest_point(ori_data[i],out_kpcd[i])
        print("save keypoints to "+paths[i].replace('.obj','keypoints.npz'+str(ns.n_keypoint)))
        np.savez(paths[i].replace('.obj','keypoints.npz'+str(ns.n_keypoint)),keypoints=out_kpcd[i],keypoint_id=kp_id,pointcloud=ori_data[i])
