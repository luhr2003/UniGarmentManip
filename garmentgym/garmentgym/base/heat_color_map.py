import numpy as np
import mitsuba as mi
import os
import matplotlib.pyplot as plt

def generate_random_number():
    while True:
        random_number = np.random.normal()
        if -20 <= random_number <= 20:
            return random_number

def four_colormap(feature, pts,
            select_point1,
            select_point2,
            select_point3,
            select_point4):
    color_map=np.zeros((pts.shape[0],3),dtype=np.float32)
    limit1 = np.random.uniform(0.13,0.26)
    limit2 = np.random.uniform(0.13,0.26)
    limit3 = np.random.uniform(0.13,0.26)
    limit4 = np.random.uniform(0.13,0.26)
    for i in range(feature.shape[0]):
        distance1 = np.clip(np.linalg.norm(pts[i] - select_point1)+generate_random_number()/180,0,1)
        distance2 = np.clip(np.linalg.norm(pts[i] - select_point2)+generate_random_number()/180,0,1)
        distance3 = np.clip(np.linalg.norm(pts[i] - select_point3)+generate_random_number()/130,0,1)
        distance4 = np.clip(np.linalg.norm(pts[i] - select_point4)+generate_random_number()/130,0,1)
        a=np.random.uniform(0.06, limit1)
        b=np.random.uniform(0.06, limit2)
        c=np.random.uniform(0.06, limit3)
        d=np.random.uniform(0.06, limit4)
        if distance1< a:
            color_map[i]=np.array([1.0,0.0,0.0])*(a-distance1)/a + np.array([0.58,0.8,1.0])*distance1/a
        elif distance2< b:
            color_map[i]=np.array([255.0/255.0,80.0/255.0,5.0/255.0])*(b-distance2)/b + np.array([0.58,0.8,1.0])*distance2/b
        elif distance3< c:
            color_map[i]=np.array([0.0,0.0,1.0])*(c-distance3)/c + np.array([0.58,0.8,1.0])*distance3/c
        elif distance4< d:
            color_map[i]=np.array([187.0/255.0,0.0,1.0])*(d-distance4)/d + np.array([0.58,0.8,1.0])*distance4/d
        else:
            color_map[i]=np.array([0.58,0.8,1.0]).astype(np.float32)
    return color_map


def one_colormap(feature, pts,
                select_point1):
        color_map=np.zeros((pts.shape[0],3),dtype=np.float32)
        limit1 = np.random.uniform(0.13,0.26)
        for i in range(feature.shape[0]):
            distance1 = np.clip(np.linalg.norm(pts[i] - select_point1)+generate_random_number()/180,0,1)
            a=np.random.uniform(0.06, limit1)
            if distance1< a:
                color_map[i]=np.array([1.0,0.0,0.0])*(a-distance1)/a + np.array([0.58,0.8,1.0])*distance1/a
            else:
                color_map[i]=np.array([0.58,0.8,1.0]).astype(np.float32)
        return color_map

def two_grasp_colormap(feature, pts,
                select_point1,
                select_point2):
        color_map=np.zeros((pts.shape[0],3),dtype=np.float32)
        limit1 = np.random.uniform(0.13,0.26)
        limit2 = np.random.uniform(0.13,0.26)
        for i in range(feature.shape[0]):
            distance1 = np.clip(np.linalg.norm(pts[i] - select_point1)+generate_random_number()/180,0,1)
            distance2 = np.clip(np.linalg.norm(pts[i] - select_point2)+generate_random_number()/180,0,1)
            a=np.random.uniform(0.06, limit1)
            b=np.random.uniform(0.06, limit2)
            if distance1< a:
                color_map[i]=np.array([1.0,0.0,0.0])*(a-distance1)/a + np.array([0.58,0.8,1.0])*distance1/a
            elif distance2< b:
                color_map[i]=np.array([255.0/255.0,80.0/255.0,5.0/255.0])*(b-distance2)/b + np.array([0.58,0.8,1.0])*distance2/b
            else:
                color_map[i]=np.array([0.58,0.8,1.0]).astype(np.float32)
        return color_map

def grasp_place_colormap(feature, pts,
                select_point1,
                select_point3):
        color_map=np.zeros((pts.shape[0],3),dtype=np.float32)
        limit1 = np.random.uniform(0.13,0.26)
        limit3 = np.random.uniform(0.13,0.26)
        for i in range(feature.shape[0]):
            distance1 = np.clip(np.linalg.norm(pts[i] - select_point1)+generate_random_number()/180,0,1)
            distance3 = np.clip(np.linalg.norm(pts[i] - select_point3)+generate_random_number()/130,0,1)
            a=np.random.uniform(0.06, limit1)
            c=np.random.uniform(0.06, limit3)
            if distance1< a:
                color_map[i]=np.array([1.0,0.0,0.0])*(a-distance1)/a + np.array([0.58,0.8,1.0])*distance1/a
            elif distance3< c:
                color_map[i]=np.array([0.0,0.0,1.0])*(c-distance3)/c + np.array([0.58,0.8,1.0])*distance3/c
            else:
                color_map[i]=np.array([0.58,0.8,1.0]).astype(np.float32)
        return color_map
