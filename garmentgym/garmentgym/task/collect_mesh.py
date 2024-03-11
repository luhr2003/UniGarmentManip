import argparse
import pickle
import random
import sys
import time
import os

import cv2
import numpy as np
import tqdm


import trimesh
import mesh_raycast

curpath=os.getcwd()
sys.path.append(curpath)
sys.path.append(curpath+"/garmentgym")
from garmentgym.utils.init_env import init_env
import pyflex
from garmentgym.base.clothes_env import ClothesEnv
from garmentgym.base.clothes import Clothes
from copy import deepcopy
from garmentgym.clothes_hyper import hyper
from garmentgym.base.config import *
from garmentgym.utils.exceptions import MoveJointsException
from garmentgym.utils.flex_utils import center_object, wait_until_stable
from garmentgym.utils.translate_utils import world_to_pixel
from multiprocessing import Pool,Process
import open3d as o3d
from garmentgym.base.record import cur_Info

task_config = {"task_config": {
    'observation_mode': 'cam_rgb',
    'action_mode': 'pickerpickplace',
    'num_picker': 1,
    'render': True,
    'headless': False,
    'horizon': 100,
    'action_repeat': 8,
    'render_mode': 'cloth',
}}

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print("warning: dir %s already exists" % path)


class Basic_action(ClothesEnv):
    def __init__(self,mesh_category_path:str,gui=True,store_path="./",prefix="",id=-1):
        self.config=Config(task_config)
        self.id=id
        self.clothes=Clothes(name="cloth"+str(id),config=self.config,mesh_category_path=mesh_category_path,id=id)
        super().__init__(mesh_category_path=mesh_category_path,config=self.config,clothes=self.clothes)
        print(self.config)
        self.empty_scene(self.config)
        print("establish scene complete")
        self.store_path=store_path
        self.gui=gui
        self.gui=self.config.basic_config.gui
        center_object()
        self.action_tool.reset([0,0.2,0])
        self.prefix=prefix
        print("init complete")
        pyflex.step()
        if gui:
            pyflex.render()
            
    
        
        self.info=Basic_action(self.prefix)
        self.action=[]
        self.info.add(config=self.config,clothes=self.clothes,action=Action(self.action))
        self.info.init()
        
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
        self.visible_vertices=None
        self.partial_pcd_points=None

        self.grasp_states=[True,True]

    
    def record_mesh(self):
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
        self.mesh.export('/home/yiyan/correspondence/softgym_cloth/garmentgym/garmentgym/task/mesh.obj')
        print("11111111222222222222")
        
    
    
    def random_point_lift_up_drop(self):
        # Choose random pick point on cloth
        pickpoint = random.randint(0, self.clothes.mesh.num_particles - 1)
        curr_pos = pyflex.get_positions()
        original_inv_mass = curr_pos[pickpoint * 4 + 3]
        # Set the mass of the pickup point to infinity so that
        # it generates enough force to the rest of the cloth
        curr_pos[pickpoint * 4 + 3] = 0
        pyflex.set_positions(curr_pos)
        # Choose random height to fix random pick point on cloth to
        pickpoint_pos = curr_pos[pickpoint * 4: pickpoint * 4 + 3].copy()
        height = np.random.random(1) * 1.0 + 0.5
        print(pickpoint_pos)

        # Move cloth up slowly...
        init_height = pickpoint_pos[1]
        speed = 0.005
        for j in range(int(1/speed)):
            curr_pos = pyflex.get_positions()
            curr_vel = pyflex.get_velocities()
            pickpoint_pos[1] = (height-init_height)*(j*speed) + init_height
            curr_pos[pickpoint * 4: pickpoint * 4 + 3] = pickpoint_pos
            curr_pos[pickpoint * 4 + 3] = 0
            curr_vel[pickpoint * 3: pickpoint * 3 + 3] = [0, 0, 0]
            pyflex.set_positions(curr_pos)
            pyflex.set_velocities(curr_vel)
            pyflex.step()
            curr_pos_after=pyflex.get_positions()
            print("curr_pos_set",pickpoint_pos)
            print("curr_pos_after",curr_pos_after[pickpoint * 4: pickpoint * 4 + 3])
            if self.gui:
                pyflex.render()

        wait_until_stable(gui=self.gui)

        # Reset to previous cloth parameters and drop cloth
        curr_pos = pyflex.get_positions()
        curr_pos[pickpoint * 4 + 3] = original_inv_mass
        pyflex.set_positions(curr_pos)
        for j in range(300):
            pyflex.step()
            if self.gui:
                pyflex.render()

    def pick_and_place_primitive(
        self, p1, p2, lift_height=0.1):
        # prepare primitive params
        pick_pos, place_pos = p1.copy(), p2.copy()
        pick_pos[1] += 0.03
        place_pos[1] += 0.03 + 0.05

        prepick_pos = pick_pos.copy()
        prepick_pos[1] = lift_height
        preplace_pos = place_pos.copy()
        preplace_pos[1] = lift_height
        print("prepick_pos", prepick_pos)
        print("preplace_pos", preplace_pos)
        print("pick_pos", pick_pos)
        print("place_pos", place_pos)

        # execute action
        self.set_grasp(False)
        self.movep([prepick_pos], speed=8e-1)
        self.movep([pick_pos], speed=6e-1)
        self.set_grasp(True)
        self.movep([prepick_pos], speed=1e-1)
        self.movep([preplace_pos], speed=2e-2)
        self.movep([place_pos], speed=2e-2)
        self.set_grasp(False)
        self.movep([preplace_pos], speed=8e-2)
        self.hide_end_effectors()
    
    def movep(self, pos, speed=None, limit=1000, min_steps=None, eps=1e-4):
        if speed is None:
            speed = 0.08
        target_pos = np.array(pos)
        for step in range(limit):
            curr_pos = self.action_tool._get_pos()[0]
            deltas = [(targ - curr)
                      for targ, curr in zip(target_pos, curr_pos)]
            dists = [np.linalg.norm(delta) for delta in deltas]
            if all([dist < eps for dist in dists]) and\
                    (min_steps is None or step > min_steps):
                return
            action = []
            for targ, curr, delta, dist, gs in zip(
                    target_pos, curr_pos, deltas, dists, self.grasp_states):
                if dist < speed:
                    action.extend([*targ, float(gs)])
                else:
                    delta = delta/dist
                    action.extend([*(curr+delta*speed), float(gs)])
            action = np.array(action)
            self.action_tool.step(action)


        raise MoveJointsException
    
    def reset_end_effectors(self):
        self.movep([[0.5, 0.5, -0.5]], speed=8e-2)

    def hide_end_effectors(self):
        self.movep([[0.5, 0.5, -1]], speed=5e-2)


    def set_grasp(self, grasp):
        if type(grasp) == bool:
            self.grasp_states = [grasp] * len(self.grasp_states)
        elif len(grasp) == len(self.grasp_states):
            self.grasp_states = grasp
        else:
            raise Exception()
    

    def stretch_cloth_regular(self,
                      grasp_dist: float,
                      fling_height: float = 0.7,
                      max_grasp_dist: float = 0.7,
                      increment_step=0.02):
        # keep stretching until cloth is tight
        # i.e.: the midpoint of the grasped region
        # stops moving
        left, right = self.action_tool._get_pos()[0]
        print("left", left)
        print("right", right)
        left[1] = fling_height
        right[1] = fling_height
        midpoint = (left + right)/2
        direction = left - right
        direction = direction/np.linalg.norm(direction)
        self.movep([left, right], speed=8e-4, min_steps=20)
        stable_steps = 0
        cloth_midpoint = 1e2
        print("lift end")
        while True:
            positions = pyflex.get_positions().reshape((-1, 4))[:, :3]
            # get midpoints
            high_positions = positions[positions[:, 1] > fling_height-0.1, ...]
            # if (high_positions[:, 0] < 0).all() or \
            #         (high_positions[:, 0] > 0).all():
            #     # single grasp
            #     return grasp_dist
            positions = [p for p in positions]
            positions.sort(
                key=lambda pos: np.linalg.norm(pos[[0, 2]]-midpoint[[0, 2]]))
            new_cloth_midpoint = positions[0]
            stable = np.linalg.norm(
                new_cloth_midpoint - cloth_midpoint) < 3e-2
            if stable:
                stable_steps += 1
            else:
                stable_steps = 0
            stretched = stable_steps > 2
            if stretched:
                return grasp_dist
            cloth_midpoint = new_cloth_midpoint
            grasp_dist += increment_step
            left = midpoint + direction*grasp_dist/2
            right = midpoint - direction*grasp_dist/2
            self.movep([left, right], speed=5e-4)
            if grasp_dist > max_grasp_dist:
                return max_grasp_dist
            
    def lift_cloth(self,
                   grasp_dist: float,
                   fling_height: float = 0.7,
                   increment_step: float = 0.03,
                   max_height=0.7,
                   height_offset : float = 0.1):
        while True:
            positions = pyflex.get_positions().reshape((-1, 4))[:, :3]
            heights = positions[:, 1][:self.clothes.mesh.num_particles]

            if heights.min() > height_offset + 0.05:
                fling_height -= increment_step
            elif heights.min() < height_offset - 0.05:
                fling_height += increment_step 

            self.movep([[grasp_dist/2, fling_height, -0.3],
                        [-grasp_dist/2, fling_height, -0.3]], speed=1e-3)
            

            return fling_height
    def step_fn(gui=True):
        pyflex.step()
        if gui:
            pyflex.render()
    def show_key_points(self):
        curr_pos=pyflex.get_positions().reshape(-1,4)
        point_pos=curr_pos[self.clothes.top_right][:3].copy()
        next_pos=curr_pos[self.clothes.top_right][:3].copy()
        next_pos[0]-=0.5
        # self.action_tool.shape_move(next_pos)
        self.pick_and_place_primitive(point_pos,next_pos)
        for j in range(100):
            pyflex.step()
            pyflex.render()
    def show_points(self):
        curr_pos=pyflex.get_positions().reshape(-1,4)
        for index in self.clothes.keypoint:
            shape_pos=curr_pos[index][:3].copy()
            shape_pos[1]+=0.2
            self.action_tool.shape_move(shape_pos)
            time.sleep(1)
    def show_position(self,position):
        self.action_tool.shape_move(position)
    def test_camera(self):
        indix=self.clothes.top_right
        curr_pos=pyflex.get_positions().reshape(-1,4)
        point_pos=curr_pos[indix][:3].copy()
        rgb_camera_test,depth_camera_test=pyflex.render()
        rgb_camera_test=np.flip(rgb_camera_test.reshape([self.config.camera_config.cam_size[0],self.config.camera_config.cam_size[1],4]),0)[:,:,:3].astype(np.uint8)
        circle_point=world_to_pixel(point_pos,self.config.get_camera_matrix()[0],self.config.get_camera_matrix()[1])
        print("circle_point",circle_point)
        cv2.circle(rgb_camera_test,(int(round(circle_point[0])),int(round(circle_point[1]))),5,(0,0,255),-1)
        cv2.imwrite("rgb_camera_test_circle.png",rgb_camera_test)

    def test_depth(self):
        indix=self.clothes.top_right
        curr_pos=pyflex.get_positions().reshape(-1,4)
    def move_key_points_inside(self):
        curr_pos=pyflex.get_positions().reshape(-1,4)
        index=random.choice(self.clothes.keypoint)
        point_pos=curr_pos[index][:3].copy()
        next_pos=curr_pos[index][:3].copy()
        while np.linalg.norm(next_pos-point_pos)<0.3:
            next_pos=curr_pos[random.choice(range(0,self.clothes.mesh.num_particles))][:3].copy()
        self.pick_and_place_primitive(point_pos,next_pos)
        for j in range(50):
            pyflex.step()
            pyflex.render()


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

def pixel_to_world_hard(pixel_point,camera_size):
    return np.array([(pixel_point[0]-camera_size/2)*0.85/camera_size*2,0,(pixel_point[1]-camera_size/2)*0.85/camera_size*2])
                            
if __name__ == "__main__":
    print("hello")
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--mesh_path', type=str,default="/media/luhr/BAE7BC3408D3F9D7/study/correspondence/data/test_t1/Dress")
    parser.add_argument('--store_path', type=str,default="./debug")
    parser.add_argument('--prefix', type=str,default="")
    parser.add_argument('--iter',type=int,default=0)
    parser.add_argument('--mesh_id',type=str,default=-1)
    args=parser.parse_args()

    mesh_path=args.mesh_path
    store_path=args.store_path
    prefix=args.prefix
    mesh_id=args.mesh_id
    iter=args.iter
    make_dir(store_path)
    env=Basic_action(mesh_path,prefix=prefix+"iterate"+str(iter),store_path=store_path,id=mesh_id)
    
    
    # env=Basic_action("garmentgym/cloth3d/train")
    # env=Basic_action("/home/yiyan/correspondence/softgym_cloth/garmentgym/dress")
    env.hide_end_effectors()
    for j in range(50):
        pyflex.step()
        pyflex.render()
    #env.move_key_points_inside()
    
    # /media/luhr/BAE7BC3408D3F9D7/study/correspondence/data/test_t1/Dress
    # "/media/luhr/BAE7BC3408D3F9D7/study/correspondence/data/test_t1/Jumpsuit"
    # /media/luhr/BAE7BC3408D3F9D7/study/correspondence/data/test_t1/Trousers
    for j in range(50):
        pyflex.step()
        pyflex.render()
    
    # env.stretch_cloth_regular(0.5)
    # env.show_position(pixel_to_world_hard((0,0),720))
    env.record_mesh()
  
    


