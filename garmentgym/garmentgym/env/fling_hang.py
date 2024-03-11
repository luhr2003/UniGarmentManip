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
sys.path.append(curpath+"/garmentgym")   

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
from garmentgym.garmentgym.utils.flex_utils import center_object, wait_until_stable
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


def get_image(width=480, height=480):

    rgb, depth, _ = pyflex.render(uv=False)
    # Need to reverse the height dimension
    rgb = np.flip(rgb.reshape([480, 480, 4]), 0)[:, :, :3].astype(np.uint8)
    depth = np.flip(depth.reshape([480, 480]), 0)
    if (width != rgb.shape[0] or height != rgb.shape[1]) and \
            (width is not None and height is not None):
        rgb = cv2.resize(rgb, (480, 480))
        depth = cv2.resize(depth, (480, 480))
    return rgb, depth




class FlingHangEnv(ClothesEnv):
    def __init__(self,mesh_category_path:str,config:Config,gui=True,store_path="./",id=-1):
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
        
        self.add_hang()# modify this to fucntion to add more hanging
        pyflex.step()
        pyflex.render()
        
        self.info=task_info()
        self.action=[]
        self.info.add(config=self.config,clothes=self.clothes)
        self.info.init()
        self.grasp_height=0.05
        self.grasp_states=[True,True]
        
        # visualizations
        self.dump_all_visualizations = False
        self.dump_visualizations = False
        
        
        self.record_task_config=False
        self.env_end_effector_positions = []
        self.env_mesh_vertices = []
        self.gui_step=0
        
        self.num_particles = 100000    #我瞎改的
        self.fling_speed=9e-2
        self.adaptive_fling_momentum=-1
        self.particle_radius=0.00625
        
        self.up_camera=config["camera_config"]()
        self.vertice_camera=deepcopy(config.camera_config)
        self.vertice_camera.cam_position=[0, 3.5, 2.5]
        self.vertice_camera.cam_angle=[0,-np.pi/5,0]

        self.record_info_id=0
        
    def add_hang(self):
        '''
        modify this function to change the position of the hang
        center is the position of the hang
        param is the parameter of the hang which orginized as length*height*width
        you can modify the quat to change the orientation of the hang
        '''
        # 添加立柱
        center_vertic=[0.8,0,-0.8]
        quat = self.quatFromAxisAngle([0, 0,-1.],0.)
        # 长*高*宽
        param_vertic=np.array([0.05,1.9,0.05])
        pyflex.add_box(param_vertic,center_vertic,quat)

        # 添加横杆
        quat = self.quatFromAxisAngle([0, 1, -1.], np.pi/3)
        center_horiz=[0.8,1.7,-0.8]
        param_horiz=np.array([0.6,0.02,0.02])
        pyflex.add_box(param_horiz,center_horiz,quat)

        hang_state=pyflex.get_shape_states()
        # print(np.array(hang_state).reshape(-1,14))
        pyflex.set_shape_states(hang_state)
    
        for j in range(10):
            pyflex.step()
            pyflex.render()  
    
    @staticmethod
    def quatFromAxisAngle(axis, angle):
        '''
        given a rotation axis and angle, return a quatirian that represents such roatation.
        '''
        axis /= np.linalg.norm(axis)

        half = angle * 0.5
        w = np.cos(half)

        sin_theta_over_two = np.sin(half)
        axis *= sin_theta_over_two

        quat = np.array([axis[0], axis[1], axis[2], w])

        return quat

        
        
        
    def record_info(self,id):
        if self.store_path is None:
            return
        self.info.update(self.action)
        make_dir(os.path.join(self.store_path,"task_info"))
        self.curr_store_path=os.path.join(self.store_path,"task_info",str(id)+".pkl")
        with open(self.curr_store_path,"wb") as f:
            pickle.dump(self.info,f)
    
    def updown(self):
        left_shoulder_id=self.clothes.top_left
        right_shoulder_id=self.clothes.top_right
        cur_pos=np.array(pyflex.get_positions()).reshape(-1,4)[:,:3]
        left_pos=cur_pos[left_shoulder_id]
        right_pos=cur_pos[right_shoulder_id]
        next_left_pos=deepcopy(left_pos)
        next_right_pos=deepcopy(right_pos)
        next_left_pos[1]+=1
        next_right_pos[1]+=1
        #next_left_pos[2]+=random.uniform(0.5,1)
        #next_right_pos[2]+=random.uniform(0.5,1)
        for j in range(50):
            self.step_sim_fn()
        self.two_pick_and_place_primitive(left_pos,next_left_pos,right_pos,next_right_pos,0.8)
    
    
    def get_cur_info(self):
        center_object()
        self.update_camera(0)
        for j in range(10):
            pyflex.step()
            pyflex.render()
        self.info.update(self.action)
        return self.info
    
    
    
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
        self.set_colors([False,False])
        self.two_movep([[0.5, 0.5, -1],[0.5,0.5,-1]], speed=5e-2)
        
        
    def two_pick_and_place_primitive(self, p1_s, p1_e, p2_s,p2_e,lift_height=0.3,down_height=0.03):
    # prepare primitive params
        pick_pos1, place_pos1 = p1_s.copy(), p1_e.copy()
        pick_pos2, place_pos2 = p2_s.copy(), p2_e.copy()

        pick_pos1[1] += down_height
        place_pos1[1] += 0.2
        pick_pos2[1] += down_height
        place_pos2[1] += 0.2

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
        self.two_movep([prepick_pos1, prepick_pos2], speed=0.8)  # 修改此处
        self.two_movep([pick_pos1, pick_pos2], speed=0.8)  # 修改此处
        self.set_grasp([True, True])
        self.two_movep([prepick_pos1, prepick_pos2], speed=2e-2)  # 修改此处
        self.two_movep([preplace_pos1, preplace_pos2], speed=2e-2)  # 修改此处
        self.two_movep([place_pos1, place_pos2], speed=2e-2)  # 修改此处
        self.set_grasp([False, False])
        self.two_movep([preplace_pos1, preplace_pos2], speed=0.8)  # 修改此处
        self.two_hide_end_effectors()
        
    
    def reset_end_effectors(self):
        self.fling_movep([[0.5, 0.5, -0.5], [-0.5, 0.5, -0.5]], speed=8e-2)
    
    
    def fling_movep(self, pos, speed=None, limit=1000, min_steps=None, eps=1e-4):
        if speed is None:
            if self.dump_visualizations:
                speed = self.default_speed
            else:
                speed = 0.1
        target_pos = np.array(pos)
        
        # 检查维度，确保 target_pos 和 curr_pos 具有相同的维度
        if target_pos.shape != (len(target_pos), 3):
            target_pos = target_pos[:, :3]
        
        
        for step in range(limit):
            curr_pos = self.action_tool._get_pos()[0]
            
            # 检查维度，确保 target_pos 和 curr_pos 具有相同的维度
            if curr_pos.shape != (len(curr_pos), 3):
                curr_pos = curr_pos[:, :3]
            
            
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

            if step % 4 == 0 and self.dump_visualizations:
                if 'top' not in self.env_video_frames:
                    self.env_video_frames['top'] = []

                self.env_video_frames['top'].append(
                    np.squeeze(np.array(get_image()[0])))
                

        raise MoveJointsException
    
    
    def set_grasp(self, grasp):
        if type(grasp) == bool:
            self.grasp_states = [grasp] * len(self.grasp_states)
        elif len(grasp) == len(self.grasp_states):
            self.grasp_states = grasp
        else:
            raise Exception()
    
    
    
    
    def fling_primitive(self, dist, fling_height, fling_speed, cloth_height):
    
        x = cloth_height/2

        x_release = x * 0.9 * self.adaptive_fling_momentum
        x_drag = x * self.adaptive_fling_momentum
      
        # lower
        self.fling_movep([[dist/2, self.grasp_height*2+0.1, x_release-0.5],
                    [-dist/2, self.grasp_height*2+0.1, x_release-0.5]], speed=0.04)
        for j in range(20):
            pyflex.step()
            pyflex.render()
        
        self.fling_movep([[dist/2, self.grasp_height, x_drag-0.6],
                    [-dist/2, self.grasp_height, x_drag-0.6]], speed=0.05)
        # release
        self.set_grasp(False)
        print("release")
        if self.dump_visualizations:
            self.fling_movep(
                [[dist/2, self.grasp_height*2, -x_drag],
                 [-dist/2, self.grasp_height*2, -x_drag]], min_steps=10)
        self.reset_end_effectors()

    
    def pick_and_fling_primitive_new(
            self, p2, p1):

        left_grasp_pos, right_grasp_pos = p1, p2

        left_grasp_pos[1] += self.grasp_height 
        right_grasp_pos[1] += self.grasp_height

        # grasp distance
        dist = np.linalg.norm(
            np.array(left_grasp_pos) - np.array(right_grasp_pos))
        
        APPROACH_HEIGHT = 0.6
        pre_left_grasp_pos = (left_grasp_pos[0], APPROACH_HEIGHT, left_grasp_pos[2])
        pre_right_grasp_pos = (right_grasp_pos[0], APPROACH_HEIGHT, right_grasp_pos[2])
        
        self.grasp_states=[False,False]
        #approach from the top (to prevent collisions)
        self.fling_movep([pre_left_grasp_pos, pre_right_grasp_pos], speed=0.6)
        self.fling_movep([left_grasp_pos, right_grasp_pos], speed=0.5)

        # only grasp points on cloth
        self.grasp_states = [True, True]

        PRE_FLING_HEIGHT = 1.5
        #lift up cloth
        self.fling_movep([[left_grasp_pos[0], PRE_FLING_HEIGHT, left_grasp_pos[2]],\
             [right_grasp_pos[0], PRE_FLING_HEIGHT, right_grasp_pos[2]]], speed=0.05)
        print("fling step1")

        for j in range(50):
            pyflex.step()
            pyflex.render()

        self.fling_movep([[left_grasp_pos[0]+0.5, PRE_FLING_HEIGHT, left_grasp_pos[2]],\
                [right_grasp_pos[0]-0.5, PRE_FLING_HEIGHT, right_grasp_pos[2]]], speed=0.02)

        for j in range(50):
            pyflex.step()
            pyflex.render()

        # lift to prefling
        self.fling_movep([[left_grasp_pos[0]+0.5, PRE_FLING_HEIGHT, left_grasp_pos[2]+0.8],\
             [right_grasp_pos[0]-0.5, PRE_FLING_HEIGHT, right_grasp_pos[2]+0.8]], speed=0.08)
        print("fling step2")
        
        for j in range(100):
            pyflex.step()
            pyflex.render()
      
        # wait_until_stable(20, tolerance=0.005)

        positions = pyflex.get_positions().reshape((-1, 4))[:, :3]
        heights = positions[:self.num_particles][:, 1]
        cloth_height = heights.max() - heights.min()

        self.fling_primitive(
            dist=dist,
            fling_height=PRE_FLING_HEIGHT-0.4,
            fling_speed=self.fling_speed,
            cloth_height=cloth_height,
            )
        
        for j in range(50):
            pyflex.step()
            pyflex.render()
        center_object()

    
    def pick_and_fling_primitive(
            self, p2, p1):

        left_grasp_pos, right_grasp_pos = p1, p2

        left_grasp_pos[1] += self.grasp_height 
        right_grasp_pos[1] += self.grasp_height

        # grasp distance
        dist = np.linalg.norm(
            np.array(left_grasp_pos) - np.array(right_grasp_pos))
        
        APPROACH_HEIGHT = 0.6
        pre_left_grasp_pos = (left_grasp_pos[0], APPROACH_HEIGHT, left_grasp_pos[2])
        pre_right_grasp_pos = (right_grasp_pos[0], APPROACH_HEIGHT, right_grasp_pos[2])
        
        self.grasp_states=[False,False]
        #approach from the top (to prevent collisions)
        self.fling_movep([pre_left_grasp_pos, pre_right_grasp_pos], speed=0.6)
        self.fling_movep([left_grasp_pos, right_grasp_pos], speed=0.5)

        # only grasp points on cloth
        self.grasp_states = [True, True]
        # if self.dump_visualizations:
        #     self.movep([left_grasp_pos, right_grasp_pos], min_steps=10)

        PRE_FLING_HEIGHT = 1
        temp=max(right_grasp_pos[2],left_grasp_pos[2])
        #lift up cloth
        self.fling_movep([[left_grasp_pos[0], PRE_FLING_HEIGHT, temp+0.5],\
             [right_grasp_pos[0], PRE_FLING_HEIGHT, temp+0.5]], speed=0.06)
        print("fling step1")
        # lift to prefling
        self.fling_movep([[left_grasp_pos[0], PRE_FLING_HEIGHT+0.4, left_grasp_pos[2]+0.2],\
             [right_grasp_pos[0], PRE_FLING_HEIGHT+0.4, right_grasp_pos[2]+0.2]], speed=0.06)
        print("fling step2")
        
        for j in range(100):
            pyflex.step()
            pyflex.render()
      
        # wait_until_stable(20, tolerance=0.005)

        positions = pyflex.get_positions().reshape((-1, 4))[:, :3]
        heights = positions[:self.num_particles][:, 1]
        cloth_height = heights.max() - heights.min()

        self.fling_primitive(
            dist=dist,
            fling_height=PRE_FLING_HEIGHT-0.4,
            fling_speed=self.fling_speed,
            cloth_height=cloth_height,
            )
        
    def pick_and_fling_primitive_bottom(
            self, p2, p1):

        left_grasp_pos, right_grasp_pos = p1, p2

        left_grasp_pos[1] += self.grasp_height 
        right_grasp_pos[1] += self.grasp_height

        # grasp distance
        dist = np.linalg.norm(
            np.array(left_grasp_pos) - np.array(right_grasp_pos))
        
        APPROACH_HEIGHT = 0.6
        pre_left_grasp_pos = (left_grasp_pos[0], APPROACH_HEIGHT, left_grasp_pos[2])
        pre_right_grasp_pos = (right_grasp_pos[0], APPROACH_HEIGHT, right_grasp_pos[2])
        
        self.grasp_states=[False,False]
        #approach from the top (to prevent collisions)
        self.fling_movep([pre_left_grasp_pos, pre_right_grasp_pos], speed=0.6)
        self.fling_movep([left_grasp_pos, right_grasp_pos], speed=0.5)

        # only grasp points on cloth
        self.grasp_states = [True, True]

        PRE_FLING_HEIGHT = 2
        #lift up cloth
        self.fling_movep([[left_grasp_pos[0], PRE_FLING_HEIGHT, left_grasp_pos[2]],\
             [right_grasp_pos[0], PRE_FLING_HEIGHT, right_grasp_pos[2]]], speed=0.05)
        print("fling step1")

        for j in range(50):
            pyflex.step()
            pyflex.render()

        self.fling_movep([[left_grasp_pos[0]+0.5, PRE_FLING_HEIGHT, left_grasp_pos[2]],\
                [right_grasp_pos[0]-0.5, PRE_FLING_HEIGHT, right_grasp_pos[2]]], speed=0.02)

        for j in range(50):
            pyflex.step()
            pyflex.render()

        # lift to prefling
        self.fling_movep([[left_grasp_pos[0]+0.5, PRE_FLING_HEIGHT, left_grasp_pos[2]-0.8],\
             [right_grasp_pos[0]-0.5, PRE_FLING_HEIGHT, right_grasp_pos[2]-0.8]], speed=0.08)
        print("fling step2")
        
        for j in range(100):
            pyflex.step()
            pyflex.render()
      
        # wait_until_stable(20, tolerance=0.005)

        self.fling_movep([[left_grasp_pos[0]+0.2, PRE_FLING_HEIGHT/2, left_grasp_pos[2]],\
                [right_grasp_pos[0]-0.2, PRE_FLING_HEIGHT/2, right_grasp_pos[2]]], speed=0.08)
        for j in range(50):
            pyflex.step()
            pyflex.render()
        self.fling_movep([[left_grasp_pos[0], 0.05, left_grasp_pos[2]+0.8],\
                [right_grasp_pos[0],0.05, right_grasp_pos[2]+0.8]], speed=0.04)
        
        for j in range(50):
            pyflex.step()
            pyflex.render()
        
        self.set_grasp(False)
        print("release")
        self.reset_end_effectors()
        
        center_object()
        
        
            
    
    def move_sleeve(self,var):
        left_id=self.clothes.top_left
        right_id=self.clothes.top_right
        cur_pos=np.array(pyflex.get_positions()).reshape(-1,4)[:,:3]
        cur_left_pos=cur_pos[left_id]
        cur_right_pos=cur_pos[right_id]
        next_left_pos=deepcopy(cur_left_pos)
        next_left_pos[0]+=random.uniform(-var,var)
        next_left_pos[2]+=random.uniform(-var,var)
        # self.pick_and_place_primitive(cur_left_pos,next_left_pos)
        cur_right_pos=deepcopy(cur_right_pos)
        next_right_pos=deepcopy(cur_right_pos)
        next_right_pos[0]+=random.uniform(-var,var)
        next_right_pos[2]+=random.uniform(-var,var)
        # self.pick_and_place_primitive(cur_right_pos,next_right_pos)
        self.two_pick_and_place_primitive(cur_left_pos,next_left_pos,cur_right_pos,next_right_pos)
    def move_bottom(self,var):
        left_id=self.clothes.bottom_left
        right_id=self.clothes.bottom_right
        cur_pos=np.array(pyflex.get_positions()).reshape(-1,4)[:,:3]
        cur_left_pos=cur_pos[left_id]
        cur_right_pos=cur_pos[right_id]
        next_left_pos=deepcopy(cur_left_pos)
        next_left_pos[0]+=random.uniform(-var,var)
        next_left_pos[2]+=random.uniform(-var,var)
        # self.pick_and_place_primitive(cur_left_pos,next_left_pos)
        cur_right_pos=deepcopy(cur_right_pos)
        next_right_pos=deepcopy(cur_right_pos)
        next_right_pos[0]+=random.uniform(-var,var)
        next_right_pos[2]+=random.uniform(-var,var)
        # self.pick_and_place_primitive(cur_right_pos,next_right_pos)
        self.two_pick_and_place_primitive(cur_left_pos,next_left_pos,cur_right_pos,next_right_pos)
    

    
    def move_middle(self,var):
        middle_id=self.clothes.middle_point
        cur_pos=np.array(pyflex.get_positions()).reshape(-1,4)[:,:3]
        cur_middle_pos=cur_pos[middle_id]
        next_middle_pos=deepcopy(cur_middle_pos)
        next_middle_pos[0]+=random.uniform(-var,var)
        next_middle_pos[2]+=random.uniform(-var,var)
        self.pick_and_place_primitive(cur_middle_pos,next_middle_pos)
    
    def move_left_right(self,var):
        left_id=self.clothes.left_point
        right_id=self.clothes.right_point
        cur_pos=np.array(pyflex.get_positions()).reshape(-1,4)[:,:3]
        cur_left_pos=cur_pos[left_id]
        cur_right_pos=cur_pos[right_id]
        next_left_pos=deepcopy(cur_left_pos)
        next_left_pos[0]+=random.uniform(-var,var)
        next_left_pos[2]+=random.uniform(-var,var)
        # self.pick_and_place_primitive(cur_left_pos,next_left_pos)
        cur_right_pos=deepcopy(cur_right_pos)
        next_right_pos=deepcopy(cur_right_pos)
        next_right_pos[0]+=random.uniform(-var,var)
        next_right_pos[2]+=random.uniform(-var,var)
        # self.pick_and_place_primitive(cur_right_pos,next_right_pos)
        self.two_pick_and_place_primitive(cur_left_pos,next_left_pos,cur_right_pos,next_right_pos)
    def move_top_bottom(self,var):
        top_id=self.clothes.top_point
        bottom_id=self.clothes.bottom_point
        cur_pos=np.array(pyflex.get_positions()).reshape(-1,4)[:,:3]
        cur_top_pos=cur_pos[top_id]
        cur_bottom_pos=cur_pos[bottom_id]
        next_top_pos=deepcopy(cur_top_pos)
        next_top_pos[0]+=random.uniform(-var,var)
        next_top_pos[2]+=random.uniform(-var,var)
        # self.pick_and_place_primitive(cur_top_pos,next_top_pos)
        cur_bottom_pos=deepcopy(cur_bottom_pos)
        next_bottom_pos=deepcopy(cur_bottom_pos)
        next_bottom_pos[0]+=random.uniform(-var,var)
        next_bottom_pos[2]+=random.uniform(-var,var)
        # self.pick_and_place_primitive(cur_bottom_pos,next_bottom_pos)
        self.two_pick_and_place_primitive(cur_top_pos,next_top_pos,cur_bottom_pos,next_bottom_pos)


    

    def execute_action(self,action):
        function=action[0]
        args=action[1]
        if function=="two_pick_and_place_primitive":
            self.two_pick_and_place_primitive(*args)
        elif function=="pick_and_fling_primitive":
            self.pick_and_fling_primitive(*args)
        elif function=="two_hang_trajectory":
            self.two_hang_trajectory(*args)
            
            
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

    def update_camera(self,id):
        if id ==0:
            pyflex.set_camera(self.up_camera)
            for j in range(5):
                pyflex.step()
                pyflex.render()
        else:
            pyflex.set_camera(self.vertice_camera())
            for j in range(5):
                pyflex.step()
                pyflex.render()
    
    def two_hang_trajectory(self,p1s,p2s):
        p1e=[0.3,2.6,-0.41]
        p2e=[0.54,2.6,-0.33]
        #p1e=[0.37,1.55,-0.51]
        #p2e=[0.62,1.55,-0.42]
        self.set_grasp([True,True])
        self.two_pick_and_place_hold(p1s,p1e,p2s,p2e)
        p1f=[0.64,1.8,-0.78]
        p2f=[0.89,1.8,-0.69]
        # p1f=[0.5,1.2,-0.64]
        # p2f=[0.75,1.2,-0.55]
        self.two_final(p1f,p2f)

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
        self.two_movep([prepick_pos1, prepick_pos2], speed=8e-1)  # 修改此处
        self.two_movep([pick_pos1, pick_pos2], speed=8e-1)  # 修改此处
        self.set_grasp([True, True])
        self.two_movep([prepick_pos1, prepick_pos2], speed=3e-2)  # 修改此处
        self.two_movep([preplace_pos1, preplace_pos2], speed=3e-2)  # 修改此处
        self.two_movep([place_pos1, place_pos2], speed=3e-2)  # 修改此处
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
        self.two_movep([place_pos1, place_pos2], speed=4e-2)  # 修改此处
        self.set_grasp([False, False])
        self.two_hide_end_effectors()
              
    def compute_coverage(self):
        return self.get_current_covered_area(self.num_particles, self.particle_radius)

    def check_hang(self,height=0.0052,distance=0.52):
        wait_until_stable(self)
        cur_pos=pyflex.get_positions().reshape(-1,4)[:,:3]
        cloth_pos=cur_pos[:self.clothes.mesh.num_particles]
        cloth_pos=np.array(cloth_pos)
        min_height=np.min(cloth_pos[:,1])
        if min_height<=height:
            return False
        else:
            if distance != None:
                top_pos=cur_pos[self.clothes.top_point]
                end_pos=np.array([0.8,1.8,-0.8])
                print(np.linalg.norm(top_pos-end_pos))
                if np.linalg.norm(top_pos-end_pos)>distance:
                    return False
                else:
                    return True
            else:
                return True
    
    def throw_down(self):
        self.two_pick_and_place_primitive([0,0,0],[0,2,0],[0.5,0.5,-1],[0.5,0.5,-1])


if __name__=="__main__":
    #change mesh_category path to your own path
    #change id to demo shirt id
    config=Config()
    env=FlingHangEnv(mesh_category_path="/home/yiyan/correspondence/softgym_cloth/garmentgym/cloth3d/train",gui=True,store_path="./",id="00044",config=config)

    for j in range(100):
        pyflex.step()
        pyflex.render()
        
    print("---------------start deform----------------")
    env.move_sleeve()
    env.move_bottom()
    
    flat_pos=pyflex.get_positions().reshape(-1,4)[:,:3]
    
    initial_area = env.compute_coverage()
    print(initial_area)
    
    env.update_camera(2)
    
    
    fling_points=[]
    fling_points.append([flat_pos[env.clothes.left_shoulder],flat_pos[env.clothes.right_shoulder]])
    fling_points.append([flat_pos[env.clothes.bottom_left],flat_pos[env.clothes.bottom_right]])
    fling_points.append([flat_pos[env.clothes.top_left],flat_pos[env.clothes.top_right]])
    
    env.pick_and_fling_primitive(fling_points[0][0],fling_points[0][1])
    
    final_area =env.compute_coverage()
    print(final_area)
    
    print(final_area/initial_area)
    
    #env.pick_and_fling_primitive(fling_points[0][0],fling_points[0][1])
    #env.pick_and_fling_primitive(fling_points[1][0],fling_points[1][1])
    #env.pick_and_fling_primitive(fling_points[2][0],fling_points[2][1])
    
    #-----------hang--------------
    flat_pos=pyflex.get_positions().reshape(-1,4)[:,:3]
    env.update_camera(2)
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
    