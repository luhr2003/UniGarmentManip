import argparse
import pathlib
import sys
import os

import cv2
curpath=os.getcwd()
sys.path.append(curpath)
curpathlib=curpath.split('/')
curpath='/'.join(curpathlib[:-3])
sys.path.append(curpath)
from pathlib import Path
import open3d as o3d
import numpy as np
import os
from PIL import Image
from garmentgym.base.config import Config



def rgbd2pcd(rgb_image_path, depth_image_path,pcd_path):
    # 读取RGB和深度图像
    rgb_image = cv2.imread(rgb_image_path)
    depth_image = cv2.imread(depth_image_path, -1)

    # 创建Open3D中的PointCloud对象
    point_cloud = o3d.geometry.PointCloud()

    depth = depth_image.flatten()
    height, width = depth_image.shape
# 找到非零深度值的索引
    nonzero_indices = np.nonzero(depth)[0]

    # 计算对应的像素坐标
    u, v = np.meshgrid(range(width), range(height))
    u = u.flatten()
    v = v.flatten()

    # 获取相机内参
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(Config().camera_config.cam_size[0], Config().camera_config.cam_size[1], Config().get_camera_matrix()[0])
    fx = camera_intrinsics.intrinsic_matrix[0, 0]
    fy = camera_intrinsics.intrinsic_matrix[1, 1]
    cx = camera_intrinsics.intrinsic_matrix[0, 2]
    cy = camera_intrinsics.intrinsic_matrix[1, 2]

    # 计算三维坐标
    z = depth[nonzero_indices]
    x = (u[nonzero_indices] - cx) * z / fx
    y = (v[nonzero_indices] - cy) * z / fy

    points = np.column_stack((x, y, z))

    # 获取颜色值
    colors = rgb_image.reshape(-1, 3) / 255.0
    colors = colors[nonzero_indices]

    # 设置点云的坐标和颜色
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    print(np.asarray(point_cloud.points))
    print(min(np.asarray(point_cloud.points)[:,2]))
    o3d.io.write_point_cloud(pcd_path, point_cloud)

def show_pcd(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    o3d.visualization.draw_geometries([pcd])

def preprocess(rgb_path, depth_path,processed_rgb_path,processed_depth_path):
    depth=Image.open(depth_path)
    depth=np.array(depth)
    depth[depth>5]=0
    new_depth=Image.fromarray(depth)
    print(processed_depth_path)
    new_depth.save(processed_depth_path)
    rgb=Image.open(rgb_path)
    rgb=np.array(rgb)
    new_rgb=Image.fromarray(rgb)
    new_rgb.save(processed_rgb_path)


if __name__ == '__main__':
    # parser=argparse.ArgumentParser()
    # parser.add_argument("--path",type=str,default="/home/luhr/correspondence/softgym++/data")
    # args=parser.parse_args()
    # p=Path(args.path) 
    # filelist=sorted(list(p.rglob("*.jpg")))
    # for file in filelist:
    #     if "processed" in str(file):
    #         continue
    #     rgb_path=str(file)
    #     depth_path=str(file).replace("jpg","png").replace("rgb","depth")
    #     processed_rgb_path=rgb_path.replace(".jpg","_processed.jpg")
    #     processed_depth_path=depth_path.replace(".png","_processed.png")
    #     pcd_path=str(file).replace("jpg","pcd").replace("rgb","")
    #     preprocess(rgb_path,depth_path,processed_rgb_path,processed_depth_path)
    #     rgbd2pcd(processed_rgb_path, processed_depth_path, pcd_path)
    #     show_pcd(pcd_path)
    #     break
    rgb_path="/home/luhr/correspondence/softgym_cloth/data/fold_show_7.png"
    depth_path="/home/luhr/correspondence/softgym_cloth/data/fold_depth_show_7.png"
    processed_rgb_path="/home/luhr/correspondence/softgym_cloth/data/fold_show_processed.png"
    processed_depth_path="/home/luhr/correspondence/softgym_cloth/data/fold_depth_show_processed.png"
    pcd_path="/home/luhr/correspondence/softgym_cloth/data/fold_show.pcd"

    preprocess(rgb_path,depth_path,processed_rgb_path,processed_depth_path)
    rgbd2pcd(rgb_path, processed_depth_path, pcd_path)
    show_pcd(pcd_path)