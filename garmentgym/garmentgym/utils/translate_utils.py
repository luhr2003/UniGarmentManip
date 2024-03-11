import sys
import os
curpath=os.getcwd()
sys.path.append(curpath)
curpathlib=curpath.split('/')
curpath='/'.join(curpathlib[:-3])
sys.path.append(curpath)
import pyflex
from garmentgym.base.config import Config



import numpy as np

def world_to_pixel(world_point, camera_intrinsics, camera_extrinsics):
    # 将世界坐标点转换为相机坐标系
    #u 是宽
    world_point[2]=-world_point[2]
    camera_point = np.dot(camera_extrinsics, np.append(world_point, 1.0))
    # 将相机坐标点转换为像素坐标系
    pixel_coordinates = np.dot(camera_intrinsics, camera_point[:3])
    pixel_coordinates /= pixel_coordinates[2]
    return pixel_coordinates[:2]

def world_to_pixel_hard(world_point,camera_size):
    return np.array([world_point[0]/0.85*camera_size/2+camera_size/2,world_point[2]/0.85*camera_size/2+camera_size/2])

def pixel_to_world_hard(pixel_point,camera_size):
    return np.array([(pixel_point[0]-camera_size/2)*0.85/camera_size*2,0,(pixel_point[1]-camera_size/2)*0.85/camera_size*2])

# # 输入相机内参和外参
# camera_intrinsics,camera_extrinsics = Config().get_camera_matrix()

# # 输入世界坐标点
# world_point = np.array([0.82842712,0,0.82842712])

# # 调用函数将世界坐标点转换为像素坐标系
# pixel_coordinates = world_to_pixel(world_point, camera_intrinsics, camera_extrinsics)


def pixel_to_world(pixel_coordinates, depth, camera_intrinsics, camera_extrinsics):
    # 将像素坐标点转换为相机坐标系
    camera_coordinates = np.dot(np.linalg.inv(camera_intrinsics), np.append(pixel_coordinates, 1.0))
    camera_coordinates *= depth

    # 将相机坐标系中的点转换为世界坐标系
    world_point = np.dot(np.linalg.inv(camera_extrinsics), np.append(camera_coordinates, 1.0))
    world_point[2]=-world_point[2]
    return world_point[:3]


# 调用函数将像素坐标转换为世界坐标系
pixel_coordinates = np.array([0,0])
depth = 2
camera_intrinsics,camera_extrinsics = Config().get_camera_matrix()
world_point = pixel_to_world(pixel_coordinates, depth, camera_intrinsics, camera_extrinsics)
print(world_point)


def depth_map_to_world(depth_map, camera_intrinsics, camera_extrinsics):
    # 获取深度图的宽度和高度
    height, width = depth_map.shape

    # 创建网格坐标
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # 将像素坐标转换为相机坐标系
    camera_coordinates = np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))
    camera_coordinates = np.dot(np.linalg.inv(camera_intrinsics), camera_coordinates)

    # 根据深度值缩放相机坐标系中的点，并筛选非零深度值对应的坐标
    valid_indices = depth_map.flatten() != 0
    camera_coordinates *= depth_map.flatten()[valid_indices]

    # 将相机坐标系中的点转换为世界坐标系
    world_coordinates = np.dot(np.linalg.inv(camera_extrinsics), np.vstack((camera_coordinates, np.ones_like(x.flatten()))))
    print(world_coordinates.shape)
    world_coordinates[2] = -world_coordinates[2]

    # 构建世界坐标并重新调整为深度图的形状
    world_coords = np.zeros((height*width, 3))
    world_coords[valid_indices] = world_coordinates[:3].T
    world_coords = world_coords.reshape((height, width, 3))

    return world_coords


camera_intrinsics,camera_extrinsics = Config().get_camera_matrix()
depth_map=np.ones((720,720))*2
# 调用函数将深度图转换为世界坐标系
# world_coords = depth_map_to_world(depth_map, camera_intrinsics, camera_extrinsics)
# print(world_coords)



