import os
import pickle
import sys

import numpy as np
curpath=os.getcwd()
sys.path.append(curpath)
curpathlib=curpath.split('/')
curpath='/'.join(curpathlib[:-3])
import cv2
from garmentgym.base.record import cross_Deform_info,Action
import open3d as o3d

def load(path):
    with open(path,"rb") as f:
        info=pickle.load(f)
    return info

def show(info:cross_Deform_info):
    gt_pcd=o3d.geometry.PointCloud()
    gt_pcd.points=o3d.utility.Vector3dVector(info.cur_info.vertices)
    o3d.visualization.draw_geometries([gt_pcd])
    pcd=o3d.geometry.PointCloud()
    info.cur_info.points=np.array(info.cur_info.points)
    print(info.cur_info.points)
    info.cur_info.points[:,2]=6-info.cur_info.points[:,2]*2
    pcd.points=o3d.utility.Vector3dVector(info.cur_info.points)
    pcd.colors=o3d.utility.Vector3dVector(info.cur_info.colors)
    o3d.visualization.draw_geometries([pcd])
    partial_pcd=o3d.geometry.PointCloud()
    partial_pcd.points=o3d.utility.Vector3dVector(info.cur_info.partial_pcd_points)
    o3d.visualization.draw_geometries([partial_pcd])
    pcd_form=o3d.geometry.PointCloud()
    points_form=[]
    for indices in info.cur_info.corr_idx:
        points_form.append(info.cur_info.vertices[indices])
    pcd_form.points=o3d.utility.Vector3dVector(points_form)
    o3d.visualization.draw_geometries([pcd_form])
    # cv2.imshow("rgb",info.cur_info.rgb)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
if __name__=="__main__":
    info:cross_Deform_info=load("./test2.pkl")
    print(info.config)
    print(info.action.action_world)
    show(info)
    