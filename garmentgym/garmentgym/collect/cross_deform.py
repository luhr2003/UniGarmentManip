
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
from garmentgym.utils.init_env import init_env
import pyflex
from garmentgym.garmentgym.base.clothes_env import ClothesEnv
from garmentgym.garmentgym.base.clothes import Clothes
from copy import deepcopy
from garmentgym.clothes_hyper import hyper
from garmentgym.garmentgym.base.config import *
from garmentgym.utils.exceptions import MoveJointsException
from garmentgym.utils.flex_utils import center_object, wait_until_stable
from multiprocessing import Pool,Process
from garmentgym.utils.translate_utils import pixel_to_world, pixel_to_world_hard, world_to_pixel, world_to_pixel_hard
from garmentgym.utils.basic_utils import make_dir
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

from garmentgym.garmentgym.base.record import cross_Deform_info,Action



class cross_deform(ClothesEnv):
    def __init__(self,mesh_category_path:str,gui=True,store_path="./",prefix="",id=-1):
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
        self.prefix=prefix
        pyflex.step()
        if gui:
            pyflex.render()
        
        self.info=cross_Deform_info(self.prefix)
        self.action=[]
        self.info.add(config=self.config,clothes=self.clothes,action=Action(self.action))
        self.info.init()
        

        self.grasp_states=[True,True]
    def record_info(self):
        self.info.update(self.action)
        make_dir(os.path.join(self.store_path,str(self.id),self.prefix))
        self.curr_store_path=os.path.join(self.store_path,str(self.id),self.prefix,str(len(self.action))+".pkl")
        with open(self.curr_store_path,"wb") as f:
            pickle.dump(self.info,f)

        

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
    def show_position(self):
        self.action_tool.shape_move(np.array([0.9,0,0.9]))
    
    def movep(self, pos, speed=None, limit=1000, min_steps=None, eps=1e-4):
        if speed is None:
            speed = 0.5
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
            if self.gui:
                pyflex.render()
            self.action_tool.step(action)

        raise MoveJointsException
    def move_key_points(self,var):
        curr_pos=pyflex.get_positions().reshape(-1,4)
        index=random.choice(self.clothes.keypoint)
        point_pos=curr_pos[index][:3].copy()
        next_pos=curr_pos[index][:3].copy()
        while np.linalg.norm(next_pos-point_pos)<0.3:
            next_pos[0]+=random.uniform(-var,var)
            next_pos[2]+=random.uniform(-var,var)

        self.pick_and_place_primitive(point_pos,next_pos)
        for j in range(50):
            pyflex.step()
            pyflex.render()
        self.action.append([index,point_pos,next_pos])
        self.record_info()
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
        self.action.append([index,point_pos,next_pos])
        self.record_info()

    def start(self,len):
        self.record_info()
        for j in range(len):
            self.move_key_points_inside()


                            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--mesh_path', type=str,default="/home/luhr/correspondence/softgym_cloth/tops")
    parser.add_argument('--store_path', type=str,default="./cloth3d_train_data")
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
    env=cross_deform(mesh_path,prefix=prefix+"iterate"+str(iter),store_path=store_path,id=mesh_id)
    for j in range(50):
        pyflex.step()
        pyflex.render()
    env.start(10)
    for j in range(50):
        pyflex.step()
        pyflex.render()
    


