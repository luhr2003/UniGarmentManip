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
from garmentgym.garmentgym.base.clothes_hang import ClothesHangEnv
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
    'headless': True,
    'horizon': 100,
    'action_repeat': 8,
    'render_mode': 'cloth',
}}

from garmentgym.garmentgym.base.record import task_info



class HangEnv(ClothesHangEnv):
    def __init__(self,mesh_category_path:str,gui=True,store_path="./",id=None):
        self.config=Config(task_config)
        self.id=id
        print("singel test mesh id:{}".format(id))
        self.clothes=Clothes(name="cloth"+str(id),config=self.config,mesh_category_path=mesh_category_path,id=id)
        super().__init__(mesh_category_path=mesh_category_path,config=self.config,clothes=self.clothes)
        self.store_path=store_path
        self.empty_scene(self.config)
        self.add_cloth(self.config)
        
        self.cur_pos=pyflex.get_positions().reshape(-1,4)[:,:3]
        self.cloth_pos=self.cur_pos[:self.clothes.mesh.num_particles]
        
        
        self.action_tool.reset([0,0.5,0])
        self.add_hang()# modify this to fucntion to add more hanging
        pyflex.step()
        pyflex.render()
        
        self.info=task_info()
        self.action=[]
        self.info.add(config=self.config,clothes=self.clothes)
        self.info.init()
        
        self.grasp_states=[True,True]
        print("init_complete")
        
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
        self, p1, p2, lift_height=0.2):
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
    def pick_and_hold(
        self, p1, p2, lift_height=0.2):
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
        self.set_grasp(True)
        
    def top_pick_and_place_primitive(
        self, p1, p2, lift_height=0.3):
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
    

    
    def pick(self,p1,p2):
        '''a demo funciton to show clothes can hang up sucessfully'''
        #self.set_grasp(False)
        self.movep([p1], speed=2e-2)
        self.set_grasp(True)
        env.update_camera(0)
        
        
        self.movep([p2], speed=2e-2)
        self.set_grasp(False)
        self.hide_end_effectors()
    
        
    def middle_step(self,p1):
        self.set_grasp(True)
        self.update_camera(0)
        self.movep([p1],speed=2e-2)
        
        
    
    def start_step(self,p1):
        self.set_grasp(False)
        self.movep([p1],speed=2e-2)
        self.set_grasp(True)
        p1_high=p1.copy()
        p1_high[1]+=0.2
        self.movep([p1_high],speed=2e-2)
        
    def final_step(self,p1):
        self.set_grasp(True)
        self.update_camera(1)
        self.movep([p1],speed=2e-2)
        self.set_grasp(False)
        self.hide_end_effectors()

    def hang_trajectory(self,p1):
        # self.set_grasp(False)
        # self.movep([p1],speed=2e-2)
        # self.set_grasp(True)
        p1_high=p1.copy()
        p1_high[1]+=0.2
        # self.movep([p1_high],speed=2e-2)
        self.pick_and_hold(p1,p1_high)
        self.set_grasp(True)
        self.update_camera(1)
        # p1=[0.52,1.5,-0.4]
        p1=[0.49,2.5,-0.4]
        #p1=[0.65,1.5,-0.5]
        self.movep([p1],speed=2e-2)
        self.set_grasp(True)
        self.update_camera(1)
        # p1=[0.67,1.1,-0.5]
        p1=[0.75,1.8,-0.63]
        #p1=[0.7,1.2,-0.64]
        self.movep([p1],speed=2e-2)
        self.set_grasp(False)
        self.hide_end_effectors()
        

    
    
    def hide_end_effectors(self):
        self.movep([[0.5, 0.5, -1]], speed=5e-2)
        

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
            pyflex.render()
            action=np.array(action)
            self.action_tool.step(action)
        raise MoveJointsException

    def move_sleeve(self):
        print("move_sleeve")
        left_id=self.clothes.top_left
        right_id=self.clothes.top_right
        cur_pos=np.array(pyflex.get_positions()).reshape(-1,4)[:,:3]
        cur_left_pos=cur_pos[left_id]
        cur_right_pos=cur_pos[right_id]
        next_left_pos=deepcopy(cur_left_pos)
        next_left_pos[0]+=random.uniform(-0.2,0.2)
        next_left_pos[2]+=random.uniform(-0.2,0.2)
        self.pick_and_place_primitive(cur_left_pos,next_left_pos)
        cur_right_pos=deepcopy(cur_right_pos)
        next_right_pos=deepcopy(cur_right_pos)
        next_right_pos[0]+=random.uniform(-0.2,0.2)
        next_right_pos[2]+=random.uniform(-0.2,0.2)
        self.pick_and_place_primitive(cur_right_pos,next_right_pos)
    def move_bottom(self):
        print("move_bottom")
        left_id=self.clothes.bottom_left
        right_id=self.clothes.bottom_right
        cur_pos=np.array(pyflex.get_positions()).reshape(-1,4)[:,:3]
        cur_left_pos=cur_pos[left_id]
        cur_right_pos=cur_pos[right_id]
        next_left_pos=deepcopy(cur_left_pos)
        next_left_pos[0]+=random.uniform(-0.5,0.2)
        next_left_pos[2]+=random.uniform(-0.2,0.5)
        self.pick_and_place_primitive(cur_left_pos,next_left_pos)
        cur_right_pos=deepcopy(cur_right_pos)
        next_right_pos=deepcopy(cur_right_pos)
        next_right_pos[0]+=random.uniform(-0.2,0.5)
        next_right_pos[2]+=random.uniform(-0.2,0.5)
        self.pick_and_place_primitive(cur_right_pos,next_right_pos)
    
    def move_shoulders(self):
        print("move_shoulders")
        right_shoulder_id=self.clothes.right_shoulder
        left_shoulder_id=self.clothes.left_shoulder
        cur_pos=np.array(pyflex.get_positions()).reshape(-1,4)[:,:3]
        cur_right_shoulder_pos=cur_pos[right_shoulder_id]
        next_right_shoulder_pos=deepcopy(cur_right_shoulder_pos)
        next_right_shoulder_pos[0]+=random.uniform(0,1)
        next_right_shoulder_pos[2]+=random.uniform(-0.5,0.5)
        self.pick_and_place_primitive(cur_right_shoulder_pos,next_right_shoulder_pos)
        cur_pos=np.array(pyflex.get_positions()).reshape(-1,4)[:,:3]
        cur_left_shoulder_pos=cur_pos[left_shoulder_id]
        next_left_shoulder_pos=deepcopy(cur_left_shoulder_pos)
        next_left_shoulder_pos[0]+=random.uniform(-1,0)
        next_left_shoulder_pos[2]+=random.uniform(-0.5,0.5)
        self.pick_and_place_primitive(cur_left_shoulder_pos,next_left_shoulder_pos)


    
    def move_middle(self):
        print("move_middle")
        middle_id=self.clothes.middle_point
        cur_pos=np.array(pyflex.get_positions()).reshape(-1,4)[:,:3]
        cur_middle_pos=cur_pos[middle_id]
        next_middle_pos=deepcopy(cur_middle_pos)
        next_middle_pos[0]+=random.uniform(-0.5,0.5)
        next_middle_pos[2]+=random.uniform(-0.5,0.5)
        self.pick_and_place_primitive(cur_middle_pos,next_middle_pos)
    
    def move_left_right(self):
        print("move_left_right")
        left_id=self.clothes.left_point
        right_id=self.clothes.right_point
        cur_pos=np.array(pyflex.get_positions()).reshape(-1,4)[:,:3]
        cur_left_pos=cur_pos[left_id]
        next_left_pos=deepcopy(cur_left_pos)
        next_left_pos[0]+=random.uniform(-0.5,1)
        next_left_pos[2]+=random.uniform(-0.7,0.7)
        self.pick_and_place_primitive(cur_left_pos,next_left_pos)
        cur_pos=np.array(pyflex.get_positions()).reshape(-1,4)[:,:3]
        cur_right_pos=cur_pos[right_id]
        next_right_pos=deepcopy(cur_right_pos)
        next_right_pos[0]+=random.uniform(-1,0.5)
        next_right_pos[2]+=random.uniform(-0.7,0.7)
        self.pick_and_place_primitive(cur_right_pos,next_right_pos)
    def move_top_bottom(self):
        print("move_top_bottom")
        top_id=self.clothes.top_point
        bottom_id=self.clothes.bottom_point
        cur_pos=np.array(pyflex.get_positions()).reshape(-1,4)[:,:3]
        cur_top_pos=cur_pos[top_id]
        next_top_pos=deepcopy(cur_top_pos)
        next_top_pos[0]+=random.uniform(-0.5,0.5)
        next_top_pos[2]+=random.uniform(-1,0)
        self.pick_and_place_primitive(cur_top_pos,next_top_pos)
        cur_pos=np.array(pyflex.get_positions()).reshape(-1,4)[:,:3]
        cur_bottom_pos=cur_pos[bottom_id]
        next_bottom_pos=deepcopy(cur_bottom_pos)
        next_bottom_pos[0]+=random.uniform(-0.5,0.5)
        next_bottom_pos[2]+=random.uniform(-0.5,0.5)
        self.pick_and_place_primitive(cur_bottom_pos,next_bottom_pos)


    def updown(self):
        print("updown")
        left_shoulder_id=self.clothes.left_shoulder
        cur_pos=np.array(pyflex.get_positions()).reshape(-1,4)[:,:3]
        left_pos=cur_pos[left_shoulder_id]
        left_pos[0]+=0.25
        left_pos[2]+=0.15
        next_left_pos=deepcopy(left_pos)
        next_left_pos[1]+=random.uniform(1,1.5)
        next_left_pos[2]+=random.uniform(-0.5,-0)
        self.pick_and_place_primitive(left_pos,next_left_pos,0.8)
        

    
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
        elif function=="start_step":
            self.start_step(arg[0])
        elif function=="middle_step":
            self.middle_step(arg[0])
        elif function=="final_step":
            self.final_step(arg[0])
        elif function=="hang_trajectory":
            self.hang_trajectory(arg[0])
    
    def wait_until_stable(self,max_steps=30,
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
    
    def check_hang(self,height=0.0052,distance=0.5):
        self.wait_until_stable()
        cur_pos=pyflex.get_positions().reshape(-1,4)[:,:3]
        cloth_pos=cur_pos[:self.clothes.mesh.num_particles]
        cloth_pos=np.array(cloth_pos)
        min_height=np.min(cloth_pos[:,1])
        if min_height<=height:
            return False
        else:
            if distance != None:
                top_pos=cur_pos[self.clothes.top_point]
                end_pos=np.array([0.8,1.7,-0.8])
                print(np.linalg.norm(top_pos-end_pos))
                if np.linalg.norm(top_pos-end_pos)>distance:
                    return False
                else:
                    return True
            else:
                return True
            

if __name__=="__main__":
    #change mesh_category path to your own path
    #change id to demo shirt id
    env=HangEnv(mesh_category_path="/home/yiyan/correspondence/softgym_cloth/garmentgym/cloth3d/train",gui=True,store_path="./",id="00044")
    env.update_camera(1)
    for j in range(100):
        pyflex.step()
        pyflex.render()
        
    
    
    flat_pos=pyflex.get_positions().reshape(-1,4)
    
    middle_shoulder =flat_pos[env.clothes.right_shoulder][:3]
    middle_shoulder[0]=(flat_pos[env.clothes.left_shoulder][0]+middle_shoulder[0])/2
    middle_shoulder[2]+=0.15
       
    env.update_camera(1)
    # env.start_step(middle_shoulder)
    # env.update_camera(0)
    # env.middle_step([0.65,1.5,-0.5])
    # env.final_step([0.7,1.2,-0.64])
   # env.pick(middle_shoulder,[0.8,1.5,-0.8])
    env.hang_trajectory(middle_shoulder)
        
    
    
        
    

    
    
    
    