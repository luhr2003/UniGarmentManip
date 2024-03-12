from copy import deepcopy
import json
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
from garmentgym.garmentgym.env.bimanual_hang import BimanualHangEnv
import open3d as o3d
import torch.nn.functional as F
import pyflex
from typing import List
from garmentgym.garmentgym.base.config import Task_result




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




if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str,default="simple")
    parser.add_argument('--demonstration',type=str,default='/home/isaac/correspondence/UniGarmentManip/demonstration/hang/double/00044')
    parser.add_argument('--current_cloth',type=str,default='/home/isaac/correspondence/UniGarmentManip/garmentgym/cloth3d/train')
    parser.add_argument('--model_path',type=str,default='/home/isaac/correspondence/UniGarmentManip/checkpoint/tops.pth')
    parser.add_argument('--mesh_id',type=str,default='01500')
    parser.add_argument('--log_file', type=str,default="double_hang_from_flat.pkl")
    parser.add_argument('--store_dir',type=str,default="hang_test")
    parser.add_argument("--device",type=str,default="cuda:0")

    args=parser.parse_args()
    demonstration=args.demonstration
    current_cloth=args.current_cloth
    model_path=args.model_path
    mesh_id=args.mesh_id
    store_dir=args.store_dir
    log_file=args.log_file
    device=args.device

    print("---------------load model----------------")
    # load model
    model=basic_model(512).to(device)
    model.load_state_dict(torch.load(model_path,map_location=device)["model_state_dict"])
    print('load model from {}'.format(model_path))
    model.eval()

    print("---------------load demonstration sequence----------------")
    # load demonstration sequence
    info_sequence=list()
    for i in sorted(os.listdir(demonstration)):
        if i.endswith('.pkl'):
            with open(os.path.join(demonstration,i),'rb') as f:
                print("load {}".format(i))
                info_sequence.append(pickle.load(f))


    print("---------------load flat cloth----------------")
    # load flat cloth
    env=BimanualHangEnv(current_cloth,store_path=store_dir,id=mesh_id)
    for j in range(50):
        pyflex.step()
        pyflex.render()

    print("---------------start deform----------------")
    # env.move_sleeve()
    # env.move_bottom()
    # env.move_left_right()
    # env.move_middle()


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
        env.update_camera(1)
        env.execute_action([action_function,action_world])
        


        hang_result=env.check_hang()
        print("hang result",hang_result)
        #-------------save result--------------
        log_file_path=os.path.join(store_dir,log_file)
        with open(log_file_path,"rb") as f:
            task_result:Task_result=pickle.load(f)
            task_result.current_num+=1
            if hang_result:
                task_result.success_num+=1
            else:
                task_result.fail_num+=1
            task_result.success_rate=task_result.success_num/task_result.current_num
            task_result.result_dict[mesh_id]=hang_result
        with open(log_file_path,"wb") as f:
            pickle.dump(task_result,f)
        print(task_result)
        # for j in range(100):
        #     pyflex.step()
        #     pyflex.render()
        
        # is_hang=True
        
        # cur_pos=pyflex.get_positions().reshape(-1,4)[:,:3]
        # cloth_pos=cur_pos[:env.clothes.mesh.num_particles]
        
        #ground_height_threshold = 0.05  # 假设地面高度为0，可以根据实际情况调整
        
        
        
        
        
        
        # 调用检测衣物挂在衣架的函数
        #print(check_cloth_hanging(env.cloth_points, env.cloth_depth, env.camera_intrinsics, env.camera_extrinsics, env.ground_plane))

        

        # # -------------visualize-------------
        # # visualize
        # if i == 0:
        #     pcd1=o3d.geometry.PointCloud()
        #     points1=demo_pc_ready[0][:,:3].cpu().numpy().reshape(-1,3)
        #     colors1=demo_pc_ready[0][:,3:].cpu().numpy().reshape(-1,3)
        #     points1[:,0]-=0.5
        #     pcd1.points=o3d.utility.Vector3dVector(points1)
        #     pcd1.colors=o3d.utility.Vector3dVector(colors1)
        #     pcd2=o3d.geometry.PointCloud()
        #     points2=cur_pc_ready[0][:,:3].cpu().numpy().reshape(-1,3)
        #     colors2=cur_pc_ready[0][:,3:].cpu().numpy().reshape(-1,3)
        #     points2[:,0]+=0.5
        #     pcd2.points=o3d.utility.Vector3dVector(points2)
        #     pcd2.colors=o3d.utility.Vector3dVector(colors2)
        #     inference_correspondence=[]
        #     for i in range(len(action_id)):
        #         inference_correspondence.append([action_id[i],cur_pcd[i]])
            
        #     inference_corr=o3d.geometry.LineSet().create_from_point_cloud_correspondences(pcd1,pcd2,inference_correspondence)
        #     inference_corr.colors=o3d.utility.Vector3dVector(np.tile(np.array([0,0,1]),(len(inference_correspondence),1)))
        #     o3d.visualization.draw_geometries([pcd1,pcd2,inference_corr])




    

