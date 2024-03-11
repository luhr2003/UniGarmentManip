import sys
import os
from typing import Any

import trimesh
from garmentgym.base.clothes import Clothes
curpath=os.getcwd()
sys.path.append(curpath)
import pyflex
from garmentgym.base.config import Config
import numpy as np
import open3d as o3d
import mesh_raycast
import cv2


class Basic_info:
    def __init__(self):
        self.mesh_path=None
        self.mesh_id=-1
        self.manipulate_time=-1
        self.prefix=None

    def __setitem__(self,key,value):
        setattr(self,key,value)
    def __getitem__(self,key):
        return getattr(self,key)

class Action:
    def __init__(self,action:list):
        self.action_world=action
    def append(self,action_world):
        self.action_world.append(action_world)

class cur_Info:
    def __init__(self,config:Config):
        self.position=None
        self.vertices=None
        self.edges=None
        self.faces=None
        self.rgb=None
        self.depth=None
        self.mesh=None
        self.pcd=None
        self.points=None
        self.colors=None
        self.corr_idx=None
        self.velocities=None
        self.world_points=None
        self.nonzero_indices=None
        self.visible_indices=[]
        self.config=config
        self.visible_vertices=None
        self.partial_pcd_points=None

    
    def rgbd2pcd(self,rgb,depth):
        # 找到非零深度值的索引
        height,width = depth.shape[:2]
        depth=depth.flatten()
        nonzero_indices = np.nonzero(depth)[0]
        # 计算对应的像素坐标
        u, v = np.meshgrid(range(width), range(height))
        u = u.flatten()
        v = v.flatten()

        # 获取相机内参
        intrinsic_matrix=Config().get_camera_matrix()[0]
        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(Config().camera_config.cam_size[0], Config().camera_config.cam_size[1],fx=intrinsic_matrix[0,0],fy=intrinsic_matrix[1,1],cx=intrinsic_matrix[0,2],cy=intrinsic_matrix[1,2])
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
        colors = rgb.reshape(-1, 3) / 255.0
        colors = colors[nonzero_indices]


        point_cloud=o3d.geometry.PointCloud()
        point_cloud.points=o3d.utility.Vector3dVector(points)
        point_cloud.colors=o3d.utility.Vector3dVector(colors)
        return points,colors
    
    @staticmethod
    def is_vertex_visible(vertex, viewer_position, mesh):
        # 创建从viewer_position到vertex的射线
        triangles=mesh.vertices[mesh.faces]
        triangles=np.array(triangles,dtype="f4")
        vertex=np.array(vertex)
        viewer_position=np.array(viewer_position)
        ray_origin = viewer_position
        ray_direction = vertex - viewer_position
        # ray_direction =ray_direction.reshape(-1,3)

        # 检测射线与mesh是否相交
        result = mesh_raycast.raycast(ray_origin, ray_direction, mesh=triangles)
        if len(result)==0:
            return True
        first_result = min(result, key=lambda x: x['distance'])
        # 如果相交点存在且距离小于等于射线长度，则vertex被遮挡
        if first_result is not None and first_result['distance'] <= np.linalg.norm(ray_direction):
            return False
        
        return True


    def record(self):
        self.position=pyflex.get_positions().reshape(-1,4)
        self.vertices=self.position[:,:3]
        self.edges=pyflex.get_edges()
        self.faces=pyflex.get_faces().reshape(-1,3)
        self.velocities=pyflex.get_velocities().reshape(-1,4)
        self.rgb,self.depth=pyflex.render_cloth()
        self.rgb=np.flip(self.rgb.reshape([self.config.camera_config.cam_size[0],self.config.camera_config.cam_size[1],4]),0)[:,:,:3].astype(np.uint8)
        self.depth=np.flip(self.depth.reshape([self.config.camera_config.cam_size[0],self.config.camera_config.cam_size[1],1]),0).astype(np.float32)
        self.depth[self.depth>3]=0
        self.mesh=trimesh.Trimesh(self.vertices,self.faces)
        self.points,self.colors=self.rgbd2pcd(self.rgb,self.depth)
        self.nonzero_indices=np.nonzero(self.depth.flatten())[0]
        print("calculating ray cast")
        for i in range(self.config.cloth_config.num_particles):
            if self.is_vertex_visible(self.vertices[i],self.config.camera_config.cam_position,self.mesh):
                self.visible_indices.append(i)
        self.partial_pcd_points=self.vertices[self.visible_indices]
        self.visible_vertices=self.vertices[self.visible_indices]
        self.world_points_map=self.pixel_to_world(self.depth,self.config.get_camera_matrix()[0],self.config.get_camera_matrix()[1])
        self.world_points=self.world_points_map.reshape(-1,3)
        self.world_points=self.world_points[self.nonzero_indices].reshape(-1,3)
        self.corr_idx=self.get_corr_idx(self.visible_vertices,self.world_points.reshape(-1,3))

    @staticmethod
    def get_corr_idx(vertices, world_points):
        world_points[:,1]=0.1
        corr_idx = np.argmin(np.linalg.norm(vertices[np.newaxis,:,:] - world_points[:,np.newaxis,:], axis=2), axis=1)
        return corr_idx
    @staticmethod
    def pixel_to_world(depth_map, camera_intrinsics, camera_extrinsics):
        # 获取深度图的宽度和高度
        depth_map=depth_map.squeeze()
        height, width = depth_map.shape

        # 创建网格坐标
        x, y = np.meshgrid(np.arange(width), np.arange(height))

        # 将像素坐标转换为相机坐标系
        camera_coordinates = np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))
        camera_coordinates = np.dot(np.linalg.inv(camera_intrinsics), camera_coordinates)

        # 根据深度值缩放相机坐标系中的点，并筛选非零深度值对应的坐标
        valid_indices = depth_map.flatten() != 0
        camera_coordinates[:,valid_indices] *= depth_map.flatten()[valid_indices]
        camera_coordinates=camera_coordinates[:,valid_indices]
        # 将相机坐标系中的点转换为世界坐标系
        world_coordinates = np.dot(np.linalg.inv(camera_extrinsics), np.vstack((camera_coordinates, np.ones_like(x.flatten()[valid_indices]))))
        world_coordinates[2] = -world_coordinates[2]

        # 构建世界坐标并重新调整为深度图的形状
        world_coords = np.zeros((height*width, 3))
        world_coords[valid_indices] = world_coordinates[:3].T
        world_coords = world_coords.reshape((height, width, 3))

        return world_coords
        
        



class cross_Deform_info():
    def __init__(self,prefix:str="cross_deform",suffix:str=""):
        self.basic_info:Basic_info=Basic_info()
        self.config:Config=None
        self.clothes:Clothes=None
        self.action:Action=None
        self.prefix=prefix
        self.suffix=suffix
        self.cur_info:cur_Info=None
        
    def __getitem__(self,key):
        return getattr(self,key)
    def __setitem__(self,key,value):
        setattr(self,key,value)
    def add(self,**kwargs):
        for key in kwargs:
            self[key]=kwargs[key]
    def update(self,action):
        self.cur_info=cur_Info(self.config)
        self.cur_info.record()
        self.action=Action(action)
        self.basic_info.manipulate_time=len(self.action.action_world)
        self.clothes.update_info()
    def init(self):
        self.basic_info.prefix=self.prefix
        self.basic_info.manipulate_time=0
        self.basic_info.mesh_path=self.clothes.path
        self.basic_info.mesh_id=self.clothes.id
        
        
        
        
class task_cur_Info:
    def __init__(self,config:Config):
        self.position=None
        self.vertices=None
        self.edges=None
        self.faces=None
        self.rgb=None
        self.depth=None
        self.mesh=None
        self.pcd=None
        self.points=None
        self.colors=None
        self.velocities=None
        self.config=config

    def rgbd2pcd(self,rgb,depth):
        # 找到非零深度值的索引
        height,width = depth.shape[:2]
        depth=depth.flatten()
        nonzero_indices = np.nonzero(depth)[0]
        # 计算对应的像素坐标
        u, v = np.meshgrid(range(width), range(height))
        u = u.flatten()
        v = v.flatten()

        # 获取相机内参
        intrinsic_matrix=Config().get_camera_matrix()[0]
        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(Config().camera_config.cam_size[0], Config().camera_config.cam_size[1],fx=intrinsic_matrix[0,0],fy=intrinsic_matrix[1,1],cx=intrinsic_matrix[0,2],cy=intrinsic_matrix[1,2])
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
        colors = rgb.reshape(-1, 3) / 255.0
        colors = colors[nonzero_indices]


        point_cloud=o3d.geometry.PointCloud()
        point_cloud.points=o3d.utility.Vector3dVector(points)
        point_cloud.colors=o3d.utility.Vector3dVector(colors)
        return points,colors





    def record(self):
        self.position=pyflex.get_positions().reshape(-1,4)
        self.vertices=self.position[:,:3]
        self.edges=pyflex.get_edges()
        self.faces=pyflex.get_faces().reshape(-1,3)
        self.velocities=pyflex.get_velocities().reshape(-1,4)
        self.rgb,self.depth=pyflex.render_cloth()
        self.rgb=np.flip(self.rgb.reshape([self.config.camera_config.cam_size[0],self.config.camera_config.cam_size[1],4]),0)[:,:,:3].astype(np.uint8)
        self.depth=np.flip(self.depth.reshape([self.config.camera_config.cam_size[0],self.config.camera_config.cam_size[1],1]),0).astype(np.float32)
        self.depth[self.depth>3]=0
        self.mesh=trimesh.Trimesh(self.vertices,self.faces)
        self.points,self.colors=self.rgbd2pcd(self.rgb,self.depth)

    
    
    
    
class task_info():
    def __init__(self):
        self.basic_info:Basic_info=Basic_info()
        self.config:Config=None
        self.clothes:Clothes=None
        self.action=[]
        self.rgb=None
        self.depth=None
        
        
    def __getitem__(self,key):
        return getattr(self,key)
    def __setitem__(self,key,value):
        setattr(self,key,value)
    def add(self,**kwargs):
        for key in kwargs:
            self[key]=kwargs[key]
    def update(self,action):
        self.cur_info=task_cur_Info(self.config)
        self.cur_info.record()
        self.action=action
        self.basic_info.manipulate_time=len(self.action)
        self.clothes.update_info()
        print("update",self.cur_info.depth.max())
    
    def init(self):
        self.basic_info.manipulate_time=0
        self.basic_info.mesh_path=self.clothes.path
        self.basic_info.mesh_id=self.clothes.id
        
 

