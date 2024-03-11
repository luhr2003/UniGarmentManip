import argparse
import pickle
import random
import sys
import time
import os

import cv2
import tqdm

curpath=os.getcwd()
sys.path.append(curpath)

import open3d as o3d
from garmentgym.garmentgym.utils.init_env import init_env
import pyflex
from garmentgym.garmentgym.base.clothes_env import ClothesEnv
from garmentgym.garmentgym.base.clothes import Clothes
from copy import deepcopy
from garmentgym.garmentgym.clothes_hyper import hyper
from garmentgym.garmentgym.base.config import *
from garmentgym.garmentgym.utils.exceptions import MoveJointsException
from garmentgym.garmentgym.utils.flex_utils import center_object, wait_until_stable
from multiprocessing import Pool,Process
from garmentgym.garmentgym.utils.translate_utils import pixel_to_world, pixel_to_world_hard, world_to_pixel, world_to_pixel_hard
from garmentgym.garmentgym.utils.basic_utils import make_dir
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

from garmentgym.garmentgym.base.record import task_info



class FoldEnv(ClothesEnv):
    def __init__(self,mesh_category_path:str,gui=True,store_path="./",id=None):
        self.config=Config(task_config)
        self.id=id
        self.clothes=Clothes(name="cloth"+str(id),config=self.config,mesh_category_path=mesh_category_path,id=id)
        super().__init__(mesh_category_path=mesh_category_path,config=self.config,clothes=self.clothes)
        self.store_path=store_path
        self.empty_scene(self.config)
        self.gui=gui
        self.gui=self.config.basic_config.gui
        center_object()
        self.action_tool.reset([0,0.1,0])
        pyflex.step()
        if gui:
            pyflex.render()
        
        self.info=task_info()
        self.action=[]
        self.info.add(config=self.config,clothes=self.clothes)
        self.info.init()
        
        self.grasp_states=[True,True]

        self.num_particles = self.clothes.mesh.num_particles    #我瞎改的
        self.particle_radius=0.00625
        
        
        
        
        
        
        
    def record_info(self):
        self.info.update(self.action)
        make_dir(os.path.join(self.store_path,str(self.id)))
        self.curr_store_path=os.path.join(self.store_path,str(self.id),str(len(self.action))+".pkl")
        with open(self.curr_store_path,"wb") as f:
            pickle.dump(self.info,f)
    
    def get_cur_info(self):
        self.info.update(self.action)
        return self.info

        
    
    def pick_and_place_primitive(
        self, p1, p2, lift_height=0.15):
        # prepare primitive params
        pick_pos, place_pos = p1.copy(), p2.copy()
        pick_pos[1]=0.03
        place_pos[1]=0.05

        prepick_pos = pick_pos.copy()
        prepick_pos[1] = lift_height
        preplace_pos = place_pos.copy()
        preplace_pos[1] = lift_height

        # execute action
        self.set_grasp(False)
        self.movep([prepick_pos], speed=8e-2)
        self.movep([pick_pos], speed=6e-2)
        self.set_grasp(True)
        self.movep([prepick_pos], speed=1e-2)
        self.movep([preplace_pos], speed=2e-2)
        self.movep([place_pos], speed=1e-2)
        self.set_grasp(False)
        self.movep([preplace_pos], speed=10e-2)
        self.hide_end_effectors()
        
    def top_pick_and_place_primitive(
        self, p1, p2, lift_height=0.15):
        # prepare primitive params
        pick_pos, place_pos = p1.copy(), p2.copy()
        pick_pos[1] += 0.06
        place_pos[1] += 0.03 + 0.05

        prepick_pos = pick_pos.copy()
        prepick_pos[1] = lift_height
        preplace_pos = place_pos.copy()
        preplace_pos[1] = lift_height

        # execute action
        self.set_grasp(False)
        self.movep([prepick_pos], speed=8e-2)
        self.movep([pick_pos], speed=6e-2)
        self.set_grasp(True)
        self.movep([prepick_pos], speed=1e-2)
        self.movep([preplace_pos], speed=2e-2)
        self.movep([place_pos], speed=2e-2)
        self.set_grasp(False)
        self.movep([preplace_pos], speed=8e-2)
        self.hide_end_effectors()
    
    
    def pick_and_change_route(
        self, p1, p2,p3 ,lift_height=0.15):
        # prepare primitive params
        pick_pos, mid_pos,place_pos = p1.copy(), p2.copy(),p3.copy()
        pick_pos[1] -= 0.04
        place_pos[1] += 0.03 + 0.05
        mid_pos[1] += 0.03 + 0.05

        prepick_pos = pick_pos.copy()
        prepick_pos[1] = lift_height
        premid_pos = mid_pos.copy()
        premid_pos[1] = lift_height
        preplace_pos = place_pos.copy()
        preplace_pos[1] = lift_height

        # execute action
        self.set_grasp(False)
        self.movep([prepick_pos], speed=8e-2)
        self.movep([pick_pos], speed=6e-2)
        self.set_grasp(True)
        self.movep([prepick_pos], speed=1e-2)
        self.movep([premid_pos], speed=2e-2)
        self.movep([mid_pos], speed=2e-2)
        
        self.movep([premid_pos], speed=1e-2)
    
        self.movep([preplace_pos], speed=1e-2)
        self.movep([place_pos], speed=2e-2)
        self.set_grasp(False)
        self.movep([preplace_pos], speed=8e-2)
        self.hide_end_effectors()
    

    def pick_change_nodown(
        self, p1, p2,p3 ,lift_height=0.15):
        # prepare primitive params
        pick_pos, mid_pos,place_pos = p1.copy(), p2.copy(),p3.copy()
        pick_pos[1] -= 0.04
        place_pos[1] += 0.03 + 0.05
        mid_pos[1] += 0.03 + 0.05

        prepick_pos = pick_pos.copy()
        prepick_pos[1] = lift_height
        premid_pos = mid_pos.copy()
        premid_pos[1] = lift_height
        preplace_pos = place_pos.copy()
        preplace_pos[1] = lift_height

        # execute action
        self.set_grasp(False)
        self.movep([prepick_pos], speed=8e-2)
        self.movep([pick_pos], speed=6e-2)
        self.set_grasp(True)
        self.movep([prepick_pos], speed=1e-2)
        self.movep([premid_pos], speed=2e-2)

        self.movep([preplace_pos], speed=1e-2)
        self.movep([place_pos], speed=2e-2)
        self.set_grasp(False)
        self.movep([preplace_pos], speed=8e-2)
        self.hide_end_effectors()

    def get_current_covered_area(self,cloth_particle_num, cloth_particle_radius: float = 0.00625):
        """
        Calculate the covered area by taking max x,y cood and min x,y 
        coord, create a discritized grid between the points
        :param pos: Current positions of the particle states
        """
        pos = pyflex.get_positions()
        pos = np.reshape(pos, [-1, 4])[:cloth_particle_num]
        min_x = np.min(pos[:, 0])
        min_y = np.min(pos[:, 2])
        max_x = np.max(pos[:, 0])
        max_y = np.max(pos[:, 2])
        init = np.array([min_x, min_y])
        span = np.array([max_x - min_x, max_y - min_y]) / 100.
        pos2d = pos[:, [0, 2]]

        offset = pos2d - init
        slotted_x_low = np.maximum(np.round((offset[:, 0] - cloth_particle_radius) / span[0]).astype(int), 0)
        slotted_x_high = np.minimum(np.round((offset[:, 0] + cloth_particle_radius) / span[0]).astype(int), 100)
        slotted_y_low = np.maximum(np.round((offset[:, 1] - cloth_particle_radius) / span[1]).astype(int), 0)
        slotted_y_high = np.minimum(np.round((offset[:, 1] + cloth_particle_radius) / span[1]).astype(int), 100)
        # Method 1
        grid = np.zeros(10000)  # Discretization
        listx = self.vectorized_range1(slotted_x_low, slotted_x_high)
        listy = self.vectorized_range1(slotted_y_low, slotted_y_high)
        listxx, listyy = self.vectorized_meshgrid1(listx, listy)
        idx = listxx * 100 + listyy
        idx = np.clip(idx.flatten(), 0, 9999)
        grid[idx] = 1
        return np.sum(grid) * span[0] * span[1]

            
    def vectorized_range1(self,start, end):
        """  Return an array of NxD, iterating from the start to the end"""
        N = int(np.max(end - start)) + 1
        idxes = np.floor(np.arange(N) * (end - start)
                        [:, None] / N + start[:, None]).astype('int')
        return idxes

    def vectorized_meshgrid1(self,vec_x, vec_y):
        """vec_x in NxK, vec_y in NxD. Return xx in Nx(KxD) and yy in Nx(DxK)"""
        N, K, D = vec_x.shape[0], vec_x.shape[1], vec_y.shape[1]
        vec_x = np.tile(vec_x[:, None, :], [1, D, 1]).reshape(N, -1)
        vec_y = np.tile(vec_y[:, :, None], [1, 1, K]).reshape(N, -1)
        return vec_x, vec_y

    
    
    
    
    def hide_end_effectors(self):
        self.movep([[0.5, 0.5, -1]], speed=5e-2)
        
    def two_hide_end_effectors(self):
        self.two_movep([[0.5, 0.5, -1],[0.5,0.5,-1]], speed=5e-2)

    def set_grasp(self, grasp):
        if type(grasp) == bool:
            self.grasp_states = [grasp] * len(self.grasp_states)
        elif len(grasp) == len(self.grasp_states):
            self.grasp_states = grasp
        else:
            raise Exception()
             
    
    def step_fn(gui=True):
        pyflex.step()
        if gui:
            pyflex.render()
    def show_position(self):
        self.action_tool.shape_move(np.array([0.9,0,0.9]))
    
    def movep(self, pos, speed=None, limit=1000, min_steps=None, eps=1e-4):
        if speed is None:
            speed = 2
        target_pos=pos
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
            if self.gui:
                pyflex.render()
            action=np.array(action)
            self.action_tool.step(action)
        raise MoveJointsException

    def move_sleeve(self):
        left_id=self.clothes.top_left
        right_id=self.clothes.top_right
        cur_pos=np.array(pyflex.get_positions()).reshape(-1,4)[:,:3]
        cur_left_pos=cur_pos[left_id]
        cur_right_pos=cur_pos[right_id]
        next_left_pos=deepcopy(cur_left_pos)
        next_left_pos[0]+=random.uniform(-0.2,0.2)
        next_left_pos[2]+=random.uniform(0.2,0.2)
        self.pick_and_place_primitive(cur_left_pos,next_left_pos)
        cur_right_pos=deepcopy(cur_right_pos)
        next_right_pos=deepcopy(cur_right_pos)
        next_right_pos[0]+=random.uniform(-0.2,0.2)
        next_right_pos[2]+=random.uniform(-0.2,0.2)
        self.pick_and_place_primitive(cur_right_pos,next_right_pos)
    def move_bottom(self):
        left_id=self.clothes.bottom_left
        right_id=self.clothes.bottom_right
        cur_pos=np.array(pyflex.get_positions()).reshape(-1,4)[:,:3]
        cur_left_pos=cur_pos[left_id]
        cur_right_pos=cur_pos[right_id]
        next_left_pos=deepcopy(cur_left_pos)
        next_left_pos[0]+=random.uniform(-0.3,0.3)
        next_left_pos[2]+=random.uniform(-0.3,0.3)
        self.pick_and_place_primitive(cur_left_pos,next_left_pos)
        cur_right_pos=deepcopy(cur_right_pos)
        next_right_pos=deepcopy(cur_right_pos)
        next_right_pos[0]+=random.uniform(-0.3,0.3)
        next_right_pos[2]+=random.uniform(-0.3,0.3)
        self.pick_and_place_primitive(cur_right_pos,next_right_pos)
    

    
    def move_middle(self):
        middle_id=self.clothes.middle_point
        cur_pos=np.array(pyflex.get_positions()).reshape(-1,4)[:,:3]
        cur_middle_pos=cur_pos[middle_id]
        next_middle_pos=deepcopy(cur_middle_pos)
        next_middle_pos[0]+=random.uniform(-0.5,0.5)
        next_middle_pos[2]+=random.uniform(-0.5,0.5)
        self.pick_and_place_primitive(cur_middle_pos,next_middle_pos)
    
    def move_left_right(self):
        left_id=self.clothes.left_point
        right_id=self.clothes.right_point
        cur_pos=np.array(pyflex.get_positions()).reshape(-1,4)[:,:3]
        cur_left_pos=cur_pos[left_id]
        cur_right_pos=cur_pos[right_id]
        next_left_pos=deepcopy(cur_left_pos)
        next_left_pos[0]+=random.uniform(-0.5,0.5)
        next_left_pos[2]+=random.uniform(-0.5,0.5)
        self.pick_and_place_primitive(cur_left_pos,next_left_pos)
        cur_right_pos=deepcopy(cur_right_pos)
        next_right_pos=deepcopy(cur_right_pos)
        next_right_pos[0]+=random.uniform(-0.5,0.5)
        next_right_pos[2]+=random.uniform(-0.5,0.5)
        self.pick_and_place_primitive(cur_right_pos,next_right_pos)
    def move_top_bottom(self):
        top_id=self.clothes.top_point
        bottom_id=self.clothes.bottom_point
        cur_pos=np.array(pyflex.get_positions()).reshape(-1,4)[:,:3]
        cur_top_pos=cur_pos[top_id]
        cur_bottom_pos=cur_pos[bottom_id]
        next_top_pos=deepcopy(cur_top_pos)
        next_top_pos[0]+=random.uniform(-0.5,0.5)
        next_top_pos[2]+=random.uniform(-0.5,0.5)
        self.pick_and_place_primitive(cur_top_pos,next_top_pos)
        cur_bottom_pos=deepcopy(cur_bottom_pos)
        next_bottom_pos=deepcopy(cur_bottom_pos)
        next_bottom_pos[0]+=random.uniform(-0.5,0.5)
        next_bottom_pos[2]+=random.uniform(-0.5,0.5)
        self.pick_and_place_primitive(cur_bottom_pos,next_bottom_pos)


        

    
    def execute_action(self,action):
        function=action[0]
        arg=action[1]
        if function=="pick_and_place_primitive":
            self.pick_and_place_primitive(arg[0],arg[1])
        elif function=="top_pick_and_place_primitive":
            self.top_pick_and_place_primitive(arg[0],arg[1])
        elif function=="pick_and_change_route":
            self.pick_and_change_route(arg[0],arg[1],arg[2])
        elif function=="pick_change_nodown":
            self.pick_change_nodown(arg[0],arg[1],arg[2])

    def wait_until_stable(self,max_steps=300,
                      tolerance=1e-2,
                      gui=False,
                      step_sim_fn=lambda: pyflex.step()):
        for _ in range(max_steps):
            particle_velocity = pyflex.get_velocities()
            if np.abs(particle_velocity).max() < tolerance:
                return True
            step_sim_fn()
            if gui:
                pyflex.render()
        return False
    
    
    def compute_coverage(self):
        return self.get_current_covered_area(self.num_particles, self.particle_radius)
    
    def check_success(self,type:str):
        initial_area=self.clothes.init_coverage
        init_mask=self.clothes.init_cloth_mask
        if type=="funnel":

            rate_boundary=0.7
            shoulder_boundary=0.3
            sleeve_boundary=0.35
            rate_boundary_upper=0.25
            


            self.wait_until_stable()
            
            cur_pos=pyflex.get_positions().reshape(-1,4)[:,:3]
            cloth_pos=cur_pos[:self.clothes.mesh.num_particles]
            cloth_pos=np.array(cloth_pos)
            
            final_area=self.compute_coverage()
            print("final_area=",final_area)
            
            rate=final_area/initial_area
            print("rate=",rate)

            
            bottom_left=cloth_pos[self.clothes.bottom_left][:3].copy()
            bottom_right=cloth_pos[self.clothes.bottom_right][:3].copy()
            top_left=cloth_pos[self.clothes.top_left][:3].copy()
            top_right=cloth_pos[self.clothes.top_right][:3].copy()
            right_shoulder=cloth_pos[self.clothes.right_shoulder][:3].copy()
            left_shoulder=cloth_pos[self.clothes.left_shoulder][:3].copy()
            
            left_sleeve_distance=np.linalg.norm(top_left-bottom_left)
            right_sleeve_distance=np.linalg.norm(top_right-bottom_right)
            left_shoulder_distance=np.linalg.norm(bottom_left-left_shoulder)
            right_shoulder_distance=np.linalg.norm(bottom_right-right_shoulder)
            print("left_sleeve_distance=",left_sleeve_distance)
            print("right_sleeve_distance=",right_sleeve_distance)
            print("left_shoulder_distance=",left_shoulder_distance)
            print("right_shoulder_distance=",right_shoulder_distance)

            if rate>rate_boundary_upper and rate<rate_boundary \
            and left_shoulder_distance<shoulder_boundary and right_shoulder_distance<shoulder_boundary \
            and left_sleeve_distance<sleeve_boundary and right_sleeve_distance<sleeve_boundary:
                return True
            else:
                return False
        
            
        elif type=="simple":

            rate_boundary=0.5
            shoulder_boundary=0.35
            sleeve_boundary=0.3
            rate_boundary_upper=0.25
            

            self.wait_until_stable()
            
            cur_pos=pyflex.get_positions().reshape(-1,4)[:,:3]
            cloth_pos=cur_pos[:self.clothes.mesh.num_particles]
            cloth_pos=np.array(cloth_pos)
            
            final_area=self.compute_coverage()
            print("final_area=",final_area)
            
            rate=final_area/initial_area
            print("rate=",rate)

            
            bottom_left=cloth_pos[self.clothes.bottom_left][:3].copy()
            bottom_right=cloth_pos[self.clothes.bottom_right][:3].copy()
            top_left=cloth_pos[self.clothes.top_left][:3].copy()
            top_right=cloth_pos[self.clothes.top_right][:3].copy()
            right_shoulder=cloth_pos[self.clothes.right_shoulder][:3].copy()
            left_shoulder=cloth_pos[self.clothes.left_shoulder][:3].copy()
            
            left_sleeve_distance=np.linalg.norm(top_left-right_shoulder)
            right_sleeve_distance=np.linalg.norm(top_right-left_shoulder)
            left_shoulder_distance=np.linalg.norm(bottom_left-left_shoulder)
            right_shoulder_distance=np.linalg.norm(bottom_right-right_shoulder)
            print("left_sleeve_distance=",left_sleeve_distance)
            print("right_sleeve_distance=",right_sleeve_distance)
            print("left_shoulder_distance=",left_shoulder_distance)
            print("right_shoulder_distance=",right_shoulder_distance)
            
            #sleeve_boundary=np.linalg.norm(top_left-top_right)

            if rate>rate_boundary_upper and rate<rate_boundary \
            and left_shoulder_distance<shoulder_boundary and right_shoulder_distance<shoulder_boundary \
            and left_sleeve_distance<=sleeve_boundary and right_sleeve_distance<=sleeve_boundary:
                return True
            else:
                return False
        
        
        elif type=="left_right":

            rate_boundary=0.7
            shoulder_boundary=0.3
            bottom_boundary=0.35
            sleeve_boundary=0.45
            rate_boundary_upper=0.25
            

            self.wait_until_stable()
            
            cur_pos=pyflex.get_positions().reshape(-1,4)[:,:3]
            cloth_pos=cur_pos[:self.clothes.mesh.num_particles]
            cloth_pos=np.array(cloth_pos)
            
            final_area=self.compute_coverage()
            print("final_area=",final_area)
            
            rate=final_area/initial_area
            print("rate=",rate)

            
            bottom_left=cloth_pos[self.clothes.bottom_left][:3].copy()
            bottom_right=cloth_pos[self.clothes.bottom_right][:3].copy()
            top_left=cloth_pos[self.clothes.top_left][:3].copy()
            top_right=cloth_pos[self.clothes.top_right][:3].copy()
            right_shoulder=cloth_pos[self.clothes.right_shoulder][:3].copy()
            left_shoulder=cloth_pos[self.clothes.left_shoulder][:3].copy()
            
            left_sleeve_distance=np.linalg.norm(top_left-bottom_left)
            right_sleeve_distance=np.linalg.norm(top_right-bottom_right)
            bottom_distance=np.linalg.norm(bottom_left-bottom_right)
            shoulder_distance=np.linalg.norm(left_shoulder-right_shoulder)
            print("left_sleeve_distance=",left_sleeve_distance)
            print("right_sleeve_distance=",right_sleeve_distance)
            print("bottom_distance=",bottom_distance)
            # print("shoulder_distance=",shoulder_distance)
            
            sleeve_boundary=np.linalg.norm(left_shoulder-bottom_left)
            print("sleeve_boundary",sleeve_boundary)

            if rate>rate_boundary_upper and rate<rate_boundary \
            and shoulder_distance<shoulder_boundary and bottom_distance<bottom_boundary \
            and left_sleeve_distance<sleeve_boundary and right_sleeve_distance<sleeve_boundary:
                return True
            else:
                return False
            
        
        elif type=="jinteng":

            rate_boundary=0.5
            shoulder_boundary=0.35
            sleeve_boundary=0.5
            rate_boundary_upper=0.25
            


            self.wait_until_stable()
            
            cur_pos=pyflex.get_positions().reshape(-1,4)[:,:3]
            cloth_pos=cur_pos[:self.clothes.mesh.num_particles]
            cloth_pos=np.array(cloth_pos)
            
            final_area=self.compute_coverage()
            print("final_area=",final_area)
            
            rate=final_area/initial_area
            print("rate=",rate)

            
            bottom_left=cloth_pos[self.clothes.bottom_left][:3].copy()
            bottom_right=cloth_pos[self.clothes.bottom_right][:3].copy()
            top_left=cloth_pos[self.clothes.top_left][:3].copy()
            top_right=cloth_pos[self.clothes.top_right][:3].copy()
            right_shoulder=cloth_pos[self.clothes.right_shoulder][:3].copy()
            left_shoulder=cloth_pos[self.clothes.left_shoulder][:3].copy()
            
            left_sleeve_distance=np.linalg.norm(top_left-bottom_left)
            right_sleeve_distance=np.linalg.norm(top_right-bottom_right)
            left_shoulder_distance=np.linalg.norm(bottom_left-left_shoulder)
            right_shoulder_distance=np.linalg.norm(bottom_right-right_shoulder)
            print("left_sleeve_distance=",left_sleeve_distance)
            print("right_sleeve_distance=",right_sleeve_distance)
            print("left_shoulder_distance=",left_shoulder_distance)
            print("right_shoulder_distance=",right_shoulder_distance)

            if rate>rate_boundary_upper and rate<rate_boundary \
            and left_shoulder_distance<shoulder_boundary and right_shoulder_distance<shoulder_boundary \
            and left_sleeve_distance<sleeve_boundary and right_sleeve_distance<sleeve_boundary:
                return True
            else:
                return False
        
        elif type=="simple":

            rate_boundary=0.5
            shoulder_boundary=0.3
            sleeve_boundary=0.5
            rate_boundary_upper=0.25
            

            self.wait_until_stable()
            
            cur_pos=pyflex.get_positions().reshape(-1,4)[:,:3]
            cloth_pos=cur_pos[:self.clothes.mesh.num_particles]
            cloth_pos=np.array(cloth_pos)
            
            final_area=self.compute_coverage()
            print("final_area=",final_area)
            
            rate=final_area/initial_area
            print("rate=",rate)

            
            bottom_left=cloth_pos[self.clothes.bottom_left][:3].copy()
            bottom_right=cloth_pos[self.clothes.bottom_right][:3].copy()
            top_left=cloth_pos[self.clothes.top_left][:3].copy()
            top_right=cloth_pos[self.clothes.top_right][:3].copy()
            right_shoulder=cloth_pos[self.clothes.right_shoulder][:3].copy()
            left_shoulder=cloth_pos[self.clothes.left_shoulder][:3].copy()
            
            left_sleeve_distance=np.linalg.norm(top_left-right_shoulder)
            right_sleeve_distance=np.linalg.norm(top_right-left_shoulder)
            left_shoulder_distance=np.linalg.norm(bottom_left-left_shoulder)
            right_shoulder_distance=np.linalg.norm(bottom_right-right_shoulder)
            print("left_sleeve_distance=",left_sleeve_distance)
            print("right_sleeve_distance=",right_sleeve_distance)
            print("left_shoulder_distance=",left_shoulder_distance)
            print("right_shoulder_distance=",right_shoulder_distance)
            
            #sleeve_boundary=np.linalg.norm(top_left-top_right)

            if rate>rate_boundary_upper and rate<rate_boundary \
            and left_shoulder_distance<shoulder_boundary and right_shoulder_distance<shoulder_boundary \
            and left_sleeve_distance<=sleeve_boundary and right_sleeve_distance<=sleeve_boundary:
                return True
            else:
                return False
        
        
        elif type=="leftright":

            rate_boundary=0.7
            shoulder_boundary=0.15
            bottom_boundary=0.15
            #sleeve_boundary=0.2
            rate_boundary_upper=0.35
            

            self.wait_until_stable()
            
            cur_pos=pyflex.get_positions().reshape(-1,4)[:,:3]
            cloth_pos=cur_pos[:self.clothes.mesh.num_particles]
            cloth_pos=np.array(cloth_pos)
            
            final_area=self.compute_coverage()
            print("final_area=",final_area)
            
            rate=final_area/initial_area
            print("rate=",rate)

            
            bottom_left=cloth_pos[self.clothes.bottom_left][:3].copy()
            bottom_right=cloth_pos[self.clothes.bottom_right][:3].copy()
            top_left=cloth_pos[self.clothes.top_left][:3].copy()
            top_right=cloth_pos[self.clothes.top_right][:3].copy()
            right_shoulder=cloth_pos[self.clothes.right_shoulder][:3].copy()
            left_shoulder=cloth_pos[self.clothes.left_shoulder][:3].copy()
            
            left_sleeve_distance=np.linalg.norm(top_left-bottom_left)
            right_sleeve_distance=np.linalg.norm(top_right-bottom_right)
            bottom_distance=np.linalg.norm(bottom_left-bottom_right)
            shoulder_distance=np.linalg.norm(left_shoulder-right_shoulder)
            print("left_sleeve_distance=",left_sleeve_distance)
            print("right_sleeve_distance=",right_sleeve_distance)
            print("bottom_distance=",bottom_distance)
            print("shoulder_distance=",shoulder_distance)
            
            sleeve_boundary=np.linalg.norm(left_shoulder-bottom_left)

            if rate>rate_boundary_upper and rate<rate_boundary \
            and shoulder_distance<shoulder_boundary and bottom_distance<bottom_boundary \
            and left_sleeve_distance<sleeve_boundary and right_sleeve_distance<sleeve_boundary:
                return True
            else:
                return False
            
        
        elif type=="jinteng":

            rate_boundary=0.5
            shoulder_boundary=0.3
            sleeve_boundary=0.35
            rate_boundary_upper=0.25
            


            self.wait_until_stable()
            
            cur_pos=pyflex.get_positions().reshape(-1,4)[:,:3]
            cloth_pos=cur_pos[:self.clothes.mesh.num_particles]
            cloth_pos=np.array(cloth_pos)
            
            final_area=self.compute_coverage()
            print("final_area=",final_area)
            
            rate=final_area/initial_area
            print("rate=",rate)

            
            bottom_left=cloth_pos[self.clothes.bottom_left][:3].copy()
            bottom_right=cloth_pos[self.clothes.bottom_right][:3].copy()
            top_left=cloth_pos[self.clothes.top_left][:3].copy()
            top_right=cloth_pos[self.clothes.top_right][:3].copy()
            right_shoulder=cloth_pos[self.clothes.right_shoulder][:3].copy()
            left_shoulder=cloth_pos[self.clothes.left_shoulder][:3].copy()
            
            left_sleeve_distance=np.linalg.norm(top_left-bottom_left)
            right_sleeve_distance=np.linalg.norm(top_right-bottom_right)
            left_shoulder_distance=np.linalg.norm(bottom_left-left_shoulder)
            right_shoulder_distance=np.linalg.norm(bottom_right-right_shoulder)
            print("left_sleeve_distance=",left_sleeve_distance)
            print("right_sleeve_distance=",right_sleeve_distance)
            print("left_shoulder_distance=",left_shoulder_distance)
            print("right_shoulder_distance=",right_shoulder_distance)

            if rate>rate_boundary_upper and rate<rate_boundary \
            and left_shoulder_distance<shoulder_boundary and right_shoulder_distance<shoulder_boundary \
            and left_sleeve_distance<sleeve_boundary and right_sleeve_distance<sleeve_boundary:
                return True
            else:
                return False
    
        
        elif type=='trousers_fold':
            rate_boundary=0.5
            top_boundary=0.6
            bottom_boundary=0.3
            updown_boundary=0.6
            rate_boundary_upper=0.25
            


            self.wait_until_stable()
            
            cur_pos=pyflex.get_positions().reshape(-1,4)[:,:3]
            cloth_pos=cur_pos[:self.clothes.mesh.num_particles]
            cloth_pos=np.array(cloth_pos)
            
            final_area=self.compute_coverage()
            print("final_area=",final_area)
            
            rate=final_area/initial_area
            print("rate=",rate)

            
            bottom_left=cloth_pos[self.clothes.bottom_left][:3].copy()
            bottom_right=cloth_pos[self.clothes.bottom_right][:3].copy()
            top_left=cloth_pos[self.clothes.top_left][:3].copy()
            top_right=cloth_pos[self.clothes.top_right][:3].copy()
            
            
            undown_distance1=np.linalg.norm(top_left-bottom_left)
            updown_distance2=np.linalg.norm(top_right-bottom_right)
            bottom_distance=np.linalg.norm(bottom_left-bottom_right)
            top_distance=np.linalg.norm(top_right-top_left)
            print("undown_distance1=",undown_distance1)
            print("updown_distance2=",updown_distance2)
            print("bottom_distance=",bottom_distance)
            print("top_distance=",top_distance)

            if rate>rate_boundary_upper and rate<rate_boundary \
            and updown_distance2<updown_boundary and undown_distance1<updown_boundary \
            and bottom_distance<bottom_boundary and top_distance<top_boundary:
                return True
            else:
                return False
            
        elif type=='dress_fold':
            rate_boundary=0.5
            top_boundary=0.6
            bottom_boundary=0.3
            updown_boundary=0.6
            rate_boundary_upper=0.25
            


            self.wait_until_stable()
            
            cur_pos=pyflex.get_positions().reshape(-1,4)[:,:3]
            cloth_pos=cur_pos[:self.clothes.mesh.num_particles]
            cloth_pos=np.array(cloth_pos)
            
            final_area=self.compute_coverage()
            print("final_area=",final_area)
            
            rate=final_area/initial_area
            print("rate=",rate)

            
            bottom_left=cloth_pos[self.clothes.bottom_left][:3].copy()
            bottom_right=cloth_pos[self.clothes.bottom_right][:3].copy()
            top_left=cloth_pos[self.clothes.top_left][:3].copy()
            top_right=cloth_pos[self.clothes.top_right][:3].copy()
            
            
            undown_distance1=np.linalg.norm(top_left-bottom_left)
            updown_distance2=np.linalg.norm(top_right-bottom_right)
            bottom_distance=np.linalg.norm(bottom_left-bottom_right)
            top_distance=np.linalg.norm(top_right-top_left)
            print("undown_distance1=",undown_distance1)
            print("updown_distance2=",updown_distance2)
            print("bottom_distance=",bottom_distance)
            print("top_distance=",top_distance)

            if rate>rate_boundary_upper and rate<rate_boundary \
            and updown_distance2<updown_boundary and undown_distance1<updown_boundary \
            and bottom_distance<bottom_boundary and top_distance<top_boundary:
                return True
            else:
                return False

    
    
    
    