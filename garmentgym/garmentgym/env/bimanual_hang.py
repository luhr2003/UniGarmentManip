import argparse
import pickle
import random
import sys
import time
import os

import cv2
import torch
import tqdm

curpath=os.getcwd()
sys.path.append(curpath)
sys.path.append(curpath+"/garmentgym")   

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
    'num_picker': 2,
    'render': True,
    'headless': False,
    'horizon': 100,
    'action_repeat': 8,
    'render_mode': 'cloth',
}}

from garmentgym.garmentgym.base.record import task_info



class BimanualHangEnv(ClothesHangEnv):
    def __init__(self,mesh_category_path:str,gui=True,store_path="./",id=-1):
        self.config=Config(task_config)
        print("double load_cloth",id)
        self.id=id
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
        
    def record_info(self,id):
        if self.store_path is None:
            return
        self.info.update(self.action)
        make_dir(os.path.join(self.store_path,"task_info"))
        self.curr_store_path=os.path.join(self.store_path,"task_info",str(id)+".pkl")
        with open(self.curr_store_path,"wb") as f:
            pickle.dump(self.info,f)
    def throw_down(self):
        self.two_pick_and_place_primitive([0,0,0],[0,2,0],[0.5,0.5,-1],[0.5,0.5,-1],lift_height=1.2)
    
    

    def record_heatmap(self,id,demo_pc,deform_pc,grasp_id):
        make_dir(os.path.join(self.store_path,"heatmap"))
        self.curr_store_path=os.path.join(self.store_path,"heatmap",str(id)+".pt")
        demo_pc=demo_pc.cpu()
        deform_pc=deform_pc.cpu()
        grasp_id=torch.tensor(grasp_id)
        torch.save({"demo_pc":demo_pc,"deform_pc":deform_pc,"grasp_id":grasp_id},self.curr_store_path)
        
    
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
        self.set_grasp(False)
        self.movep([p1], speed=2e-2)
        self.set_grasp(True)
        self.movep([p2], speed=2e-2)
        self.set_grasp(False)
        self.hide_end_effectors()
    

    def two_pick_and_down(self, p1_s,p1_m ,p1_e, p2_s,p2_m,p2_e,lift_height=0.5,down_height=0.03):
    # prepare primitive params
        pick_pos1, mid_pos1,place_pos1 = p1_s.copy(),p1_m.copy(), p1_e.copy()
        pick_pos2, mid_pos2,place_pos2 = p2_s.copy(),p2_m.copy(), p2_e.copy()

        pick_pos1[1] += down_height
        mid_pos1[1]+=down_height+0.04
        place_pos1[1] += 0.03 + 0.05
        pick_pos2[1] += 0.03
        mid_pos2[1]+=0.03
        place_pos2[1] += 0.03 + 0.05

        prepick_pos1 = pick_pos1.copy()
        prepick_pos1[1] = lift_height
        premid_pos1 = mid_pos1.copy()
        premid_pos1[1] = lift_height
        preplace_pos1 = place_pos1.copy()
        preplace_pos1[1] = lift_height
        prepick_pos2 = pick_pos2.copy()
        prepick_pos2[1] = lift_height
        premid_pos2 = mid_pos2.copy()
        premid_pos2[1]=lift_height
        preplace_pos2 = place_pos2.copy()
        preplace_pos2[1] = lift_height

        # execute action
        self.set_grasp([False, False])
        self.two_movep([prepick_pos1, prepick_pos2], speed=1e-2)  # 修改此处
        self.two_movep([pick_pos1, pick_pos2], speed=1e-2)  # 修改此处
        self.set_grasp([True, True])
        self.two_movep([prepick_pos1, prepick_pos2], speed=1e-2)  # 修改此处
        self.two_movep([premid_pos1,premid_pos2], speed=1e-2)  # 修改此处
        self.two_movep([mid_pos1,mid_pos2], speed=1e-2)  # 修改此处
        self.two_movep([premid_pos1,premid_pos2], speed=1e-2)  # 修改此处
        
        self.two_movep([preplace_pos1, preplace_pos2], speed=1e-2)  # 修改此处
        self.two_movep([place_pos1, place_pos2], speed=1e-2)  # 修改此处
        self.set_grasp([False, False])
        self.two_movep([preplace_pos1, preplace_pos2], speed=1e-2)  # 修改此处
        self.two_hide_end_effectors()
    
    
    
    def two_pick_change_nodown(
        self, p1, p2,p3 ,p4,p5,p6,lift_height=0.5):
        # prepare primitive params
        pick_pos1, mid_pos1,place_pos1 = p1.copy(), p2.copy(),p3.copy()
        pick_pos2, mid_pos2,place_pos2 = p4.copy(), p5.copy(),p6.copy()
        pick_pos1[1] -= 0.04
        place_pos1[1] += 0.03 + 0.05
        mid_pos1[1] += 0.03 + 0.05
        pick_pos2[1] -= 0.04
        place_pos2[1] += 0.03 + 0.05
        mid_pos2[1] += 0.03 + 0.05

        prepick_pos1 = pick_pos1.copy()
        prepick_pos1[1] = lift_height
        premid_pos1 = mid_pos1.copy()
        premid_pos1[1] = lift_height
        preplace_pos1 = place_pos1.copy()
        preplace_pos1[1] = lift_height
        
        prepick_pos2 = pick_pos2.copy()
        prepick_pos2[1] = lift_height
        premid_pos2 = mid_pos2.copy()
        premid_pos2[1] = lift_height
        preplace_pos2 = place_pos2.copy()
        preplace_pos2[1] = lift_height

        # execute action
        self.set_grasp(False)
        self.two_movep([prepick_pos1,prepick_pos2], speed=8e-2)
        self.two_movep([pick_pos1,pick_pos2], speed=6e-2)
        self.set_grasp(True)
        self.two_movep([prepick_pos1,prepick_pos2], speed=1e-2)
        self.two_movep([premid_pos1,premid_pos2], speed=2e-2)

        self.two_movep([preplace_pos1,preplace_pos2], speed=1e-2)
        self.two_movep([place_pos1,place_pos2], speed=2e-2)
        self.set_grasp(False)
        self.two_movep([preplace_pos1,preplace_pos2], speed=8e-2)
        self.two_hide_end_effectors()

    def two_pick_and_place_primitive(self, p1_s, p1_e, p2_s,p2_e,lift_height=0.4,down_height=0.03):
    # prepare primitive params
        pick_pos1, place_pos1 = p1_s.copy(), p1_e.copy()
        pick_pos2, place_pos2 = p2_s.copy(), p2_e.copy()

        pick_pos1[1] += down_height
        place_pos1[1] += 0.03 + 0.05
        pick_pos2[1] += down_height
        place_pos2[1] += 0.03 + 0.05

        prepick_pos1 = pick_pos1.copy()
        prepick_pos1[1] = lift_height
        preplace_pos1 = place_pos1.copy()
        preplace_pos1[1] = lift_height
        prepick_pos2 = pick_pos2.copy()
        prepick_pos2[1] = lift_height
        preplace_pos2 = place_pos2.copy()
        preplace_pos2[1] = lift_height

        # execute action
        self.set_grasp([False, False])
        self.two_movep([prepick_pos1, prepick_pos2], speed=8e-2)  # 修改此处
        self.two_movep([pick_pos1, pick_pos2], speed=8e-2)  # 修改此处
        self.set_grasp([True, True])
        self.two_movep([prepick_pos1, prepick_pos2], speed=1e-2)  # 修改此处
        self.two_movep([preplace_pos1, preplace_pos2], speed=1e-2)  # 修改此处
        self.two_movep([place_pos1, place_pos2], speed=1e-2)  # 修改此处
        self.set_grasp([False, False])
        self.two_hide_end_effectors()
        # self.two_movep([preplace_pos1, preplace_pos2], speed=1e-2)  # 修改此处
        # self.two_hide_end_effectors()
    
    def two_pick_and_place_hold(self, p1_s, p1_e, p2_s,p2_e,lift_height=0.4,down_height=0.03):
    # prepare primitive params
        pick_pos1, place_pos1 = p1_s.copy(), p1_e.copy()
        pick_pos2, place_pos2 = p2_s.copy(), p2_e.copy()

        pick_pos1[1] += down_height
        place_pos1[1] += 0.03 + 0.05
        pick_pos2[1] += down_height
        place_pos2[1] += 0.03 + 0.05

        prepick_pos1 = pick_pos1.copy()
        prepick_pos1[1] = lift_height
        preplace_pos1 = place_pos1.copy()
        preplace_pos1[1] = lift_height
        prepick_pos2 = pick_pos2.copy()
        prepick_pos2[1] = lift_height
        preplace_pos2 = place_pos2.copy()
        preplace_pos2[1] = lift_height

        # execute action
        self.set_grasp([False, False])
        self.step_sim_fn()
        self.two_movep([prepick_pos1, prepick_pos2], speed=8e-2)  # 修改此处
        self.step_sim_fn()
        self.two_movep([pick_pos1, pick_pos2], speed=8e-2)  # 修改此处
        self.set_grasp([True, True])
        self.two_movep([prepick_pos1, prepick_pos2], speed=1e-2)  # 修改此处
        self.step_sim_fn()
        self.two_movep([preplace_pos1, preplace_pos2], speed=1e-2)  # 修改此处
        self.step_sim_fn()
        self.two_movep([place_pos1, place_pos2], speed=1e-2)  # 修改此处
        self.step_sim_fn()
        # self.set_grasp([False, False])
        # self.two_hide_end_effectors()
        # self.two_movep([preplace_pos1, preplace_pos2], speed=1e-2)  # 修改此处
        # self.two_hide_end_effectors()
        


    def two_final(self, p1_e, p2_e,lift_height=0.5,down_height=0.03):
    # prepare primitive params
        place_pos1 = p1_e.copy()
        place_pos2 = p2_e.copy()
        place_pos1[1] += 0.03 + 0.05
        place_pos2[1] += 0.03 + 0.05

        # execute action
        self.set_grasp([True, True])
        self.step_sim_fn()
        self.two_movep([place_pos1, place_pos2], speed=2e-2)  # 修改此处
        self.step_sim_fn()
        self.set_grasp([False, False])
        self.two_hide_end_effectors()
        self.step_sim_fn()

    def two_hang_trajectory(self,p1s,p2s):
        self.step_sim_fn()
        p1e=[0.3,2.5,-0.41]
        p2e=[0.54,2.5,-0.33]
        #p1e=[0.37,1.55,-0.51]
        #p2e=[0.62,1.55,-0.42]
        self.set_grasp([True,True])
        self.two_pick_and_place_hold(p1s,p1e,p2s,p2e)
        self.step_sim_fn()
        p1f=[0.54,1.8,-0.58]
        p2f=[0.69,1.8,-0.59]
        # p1f=[0.5,1.2,-0.64]
        # p2f=[0.75,1.2,-0.55]
        self.two_final(p1f,p2f)
        self.step_sim_fn()
    # def two_hang_trajectory(self,p1s,p2s):
    #     self.step_sim_fn()
    #     p1e=[0.3,2.5,-0.41]
    #     p2e=[0.54,2.5,-0.33]
    #     #p1e=[0.37,1.55,-0.51]
    #     #p2e=[0.62,1.55,-0.42]
    #     self.set_grasp([True,True])
    #     self.two_pick_and_place_hold(p1s,p1e,p2s,p2e)
    #     self.step_sim_fn()
    #     p1f=[0.64,1.8,-0.78]
    #     p2f=[0.89,1.8,-0.69]
    #     # p1f=[0.5,1.2,-0.64]
    #     # p2f=[0.75,1.2,-0.55]
    #     self.two_final(p1f,p2f)
    #     self.step_sim_fn()

    
    
    def two_movep(self, pos, speed=None, limit=1000, min_steps=None, eps=1e-4):
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
            
            for targ, curr in zip(target_pos[:-1], curr_pos):  # 去掉目标位置的最后一个元素
                delta = targ - curr  # 计算位置差值
                dist = np.linalg.norm(delta)
            
            
            
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
    
    
    def hide_end_effectors(self):
        self.movep([[0.5, 0.5, -1]], speed=5e-2)
        
    def two_hide_end_effectors(self):
        self.two_movep([[0.5, 2.0, -1],[0.5,2.0,-1]], speed=5e-2)
        

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
        print("move sleeve")
        left_id=self.clothes.top_left
        right_id=self.clothes.top_right
        cur_pos=np.array(pyflex.get_positions()).reshape(-1,4)[:,:3]
        cur_left_pos=cur_pos[left_id]
        cur_right_pos=cur_pos[right_id]
        next_left_pos=deepcopy(cur_left_pos)
        next_left_pos[0]+=random.uniform(-0.2,0.5)
        next_left_pos[2]+=random.uniform(-0.4,0.4)
        # self.pick_and_place_primitive(cur_left_pos,next_left_pos)
        cur_right_pos=deepcopy(cur_right_pos)
        next_right_pos=deepcopy(cur_right_pos)
        next_right_pos[0]+=random.uniform(-0.5,0.2)
        next_right_pos[2]+=random.uniform(-0.4,0.4)
        # self.pick_and_place_primitive(cur_right_pos,next_right_pos)
        self.two_pick_and_place_primitive(cur_left_pos,next_left_pos,cur_right_pos,next_right_pos)
    def move_bottom(self):
        print("move bottom")
        left_id=self.clothes.bottom_left
        right_id=self.clothes.bottom_right
        cur_pos=np.array(pyflex.get_positions()).reshape(-1,4)[:,:3]
        cur_left_pos=cur_pos[left_id]
        cur_right_pos=cur_pos[right_id]
        next_left_pos=deepcopy(cur_left_pos)
        next_left_pos[0]+=random.uniform(-0.5,0.5)
        next_left_pos[2]+=random.uniform(-0.5,0.5)
        # self.pick_and_place_primitive(cur_left_pos,next_left_pos)
        cur_right_pos=deepcopy(cur_right_pos)
        next_right_pos=deepcopy(cur_right_pos)
        next_right_pos[0]+=random.uniform(-0.5,0.5)
        next_right_pos[2]+=random.uniform(-0.5,0.5)
        # self.pick_and_place_primitive(cur_right_pos,next_right_pos)
        self.two_pick_and_place_primitive(cur_left_pos,next_left_pos,cur_right_pos,next_right_pos)
    

    
    def move_middle(self):
        print("move middle")
        middle_id=self.clothes.middle_point
        cur_pos=np.array(pyflex.get_positions()).reshape(-1,4)[:,:3]
        cur_middle_pos=cur_pos[middle_id]
        next_middle_pos=deepcopy(cur_middle_pos)
        next_middle_pos[0]+=random.uniform(-0.5,0.5)
        next_middle_pos[2]+=random.uniform(-0.5,0.5)
        self.pick_and_place_primitive(cur_middle_pos,next_middle_pos)
    
    def move_left_right(self):
        print("move left right")
        left_id=self.clothes.left_point
        right_id=self.clothes.right_point
        cur_pos=np.array(pyflex.get_positions()).reshape(-1,4)[:,:3]
        cur_left_pos=cur_pos[left_id]
        cur_right_pos=cur_pos[right_id]
        next_left_pos=deepcopy(cur_left_pos)
        next_left_pos[0]+=random.uniform(-0.5,1)
        next_left_pos[2]+=random.uniform(-0.7,0.7)
        # self.pick_and_place_primitive(cur_left_pos,next_left_pos)
        cur_right_pos=deepcopy(cur_right_pos)
        next_right_pos=deepcopy(cur_right_pos)
        next_right_pos[0]+=random.uniform(-1,0.5)
        next_right_pos[2]+=random.uniform(-0.7,0.7)
        # self.pick_and_place_primitive(cur_right_pos,next_right_pos)
        self.two_pick_and_place_primitive(cur_left_pos,next_left_pos,cur_right_pos,next_right_pos)
    def move_top_bottom(self):
        print("move top bottom")
        top_id=self.clothes.top_point
        bottom_id=self.clothes.bottom_point
        cur_pos=np.array(pyflex.get_positions()).reshape(-1,4)[:,:3]
        cur_top_pos=cur_pos[top_id]
        cur_bottom_pos=cur_pos[bottom_id]
        next_top_pos=deepcopy(cur_top_pos)
        next_top_pos[0]+=random.uniform(-0.5,0.5)
        next_top_pos[2]+=random.uniform(-0.5,0.5)
        # self.pick_and_place_primitive(cur_top_pos,next_top_pos)
        cur_bottom_pos=deepcopy(cur_bottom_pos)
        next_bottom_pos=deepcopy(cur_bottom_pos)
        next_bottom_pos[0]+=random.uniform(-0.5,0.5)
        next_bottom_pos[2]+=random.uniform(-0.5,0.5)
        # self.pick_and_place_primitive(cur_bottom_pos,next_bottom_pos)
        self.two_pick_and_place_primitive(cur_top_pos,next_top_pos,cur_bottom_pos,next_bottom_pos)

    def shoudle_random(self):
        print("random")
        cur_pos=np.array(pyflex.get_positions())[:,:3]
        left_shoulder_id=self.clothes.left_shoulder
        right_shoulder_id=self.clothes.right_shoulder
        left_pos=cur_pos[left_shoulder_id]
        right_pos=cur_pos[right_shoulder_id]
        next_left_pos=deepcopy(left_pos)
        next_right_pos=deepcopy(right_pos)
        next_left_pos[1]+=random.uniform(0,1)
        next_right_pos[1]+=random.uniform(0,1)
        self.two_pick_and_place_primitive(left_pos,next_left_pos,right_pos,next_right_pos)
        for j in range(50):
            pyflex.step()
            pyflex.render()
        

    def updown(self):
        print("updown")
        left_bottom_id=self.clothes.bottom_left
        right_bottom_id=self.clothes.bottom_right
        cur_pos=np.array(pyflex.get_positions()).reshape(-1,4)[:,:3]
        left_pos=cur_pos[left_bottom_id]
        right_pos=cur_pos[right_bottom_id]
        next_left_pos=deepcopy(left_pos)
        next_right_pos=deepcopy(right_pos)
        next_left_pos[1]+=0.5
        next_right_pos[1]+=0.5
        next_left_pos[2]-=random.uniform(0,0.4)
        next_right_pos[2]-=random.uniform(0,0.4)
        self.two_pick_and_place_primitive(left_pos,next_left_pos,right_pos,next_right_pos,0.8)
        self.set_grasp([False,False])
        for j in range(50):
            pyflex.step()
            pyflex.render()
        
        

    
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
        elif function=="two_hang_trajectory":
            self.two_hang_trajectory(arg[0],arg[1])
        else:
            "print_error"
    
    def wait_until_stable(self,max_steps=30,
                      tolerance=1e-2,
                      gui=True,
                      step_sim_fn=lambda: pyflex.step()):
        for _ in range(max_steps):
            particle_velocity = pyflex.get_velocities()
            if np.abs(particle_velocity).max() < tolerance:
                return True
            step_sim_fn()
            if gui:
                pyflex.render()
        return False
    
    def check_hang(self,height=0.0052,distance=0.75):
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
    env=BimanualHangEnv(mesh_category_path="/home/yiyan/correspondence/softgym_cloth/garmentgym/cloth3d/val",gui=True,store_path="./",id="03321")
    env.update_camera(1)
    for j in range(100):
        pyflex.step()
        pyflex.render()
        
        
    flat_pos=pyflex.get_positions().reshape(-1,4)[:,:3]
    env.update_camera(0)
    left_collor=flat_pos[env.clothes.left_shoulder].copy()
    left_collor[0]+=0.06
    left_collor[2]+=0.08
    
    right_collor=flat_pos[env.clothes.right_shoulder].copy()
    right_collor[0]-=0.06
    right_collor[2]+=0.08
    
    # env.two_pick_and_place_primitive(left_collor,[0.37,1.55,-0.51],right_collor,[0.62,1.55,-0.42])
    # env.two_final([0.5,1.2,-0.64],[0.75,1.2,-0.55])
    #env.two_pick_change_nodown(flat_pos[env.clothes.left_shoulder],[0.4,1.8,-0.54],[0.45,1.2,-0.59],flat_pos[env.clothes.right_shoulder],[0.65,1.8,-0.45],[0.7,1.2,-0.5])
    #env.pick([0,0,0],[0.8,1.5,-0.8])
    env.two_hang_trajectory(left_collor,right_collor)
    
    
        
    

    
    
    
    