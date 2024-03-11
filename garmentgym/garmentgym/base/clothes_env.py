import colorsys
import os
import cv2
import numpy as np
import sys

import torch

from garmentgym.base.clothes import Clothes
from garmentgym.utils.flex_utils import center_object,set_random_cloth_color, set_state
import pyflex
from garmentgym.env.flex_env import FlexEnv
from garmentgym.utils.tool_utils import PickerPickPlace, Pickerpoint
from copy import deepcopy
from gym.spaces import Box
from garmentgym.clothes_hyper import hyper
from garmentgym.base.config import *
from garmentgym.garmentgym.base.heatmap_render import get_four_points_heatmap,get_grasp_place_heatmap,get_one_point_heatmap,get_two_grasp_heatmap,get_six_points_heatmap

class ClothesEnv(FlexEnv):
    def __init__(self,mesh_category_path:str,config:Config,clothes:Clothes=None,store_path:str=None):
        self.config=config
        self.render_mode = config.task_config.render_mode
        self.action_mode = config.task_config.action_mode
        self.cloth_particle_radius = config.task_config.particle_radius
        if clothes is None:
            self.clothes=Clothes(name="cloth_random",mesh_category_path=mesh_category_path,config=config)
        self.config.update(self.clothes.get_cloth_config())
        super().__init__(config=config)

        self.action_tool = PickerPickPlace(num_picker=config.task_config.num_picker, particle_radius=config.task_config.particle_radius, env=self, picker_threshold=config.task_config.picker_threshold,
                                               picker_low=(-5, 0., -5), picker_high=(5, 5, 5),picker_radius=config.task_config.picker_radius,picker_size=config.task_config.picker_size)
        
        self.store_path=store_path
        self.image_buffer=[]
    

    def step_sim_fn(self):
        pyflex.step()
        rgb,depth=pyflex.render()
        rgb=np.flip(rgb.reshape([self.config.camera_config.cam_size[0],self.config.camera_config.cam_size[1],4]),0)[:,:,:3].astype(np.uint8)
        cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR,rgb) 
        self.image_buffer.append(rgb)
    
    def export_image(self):
        trajectory_store_path=os.path.join(self.store_path,"trajectory")
        print("export image to %s"%trajectory_store_path   )
        os.makedirs(trajectory_store_path,exist_ok=True)
        video_path=os.path.join(trajectory_store_path,"video.mp4")
        video=cv2.VideoWriter(video_path,cv2.VideoWriter_fourcc(*'mp4v'),30,(self.config.camera_config.cam_size[0],self.config.camera_config.cam_size[1]))
        for i in range(len(self.image_buffer)):
            rgb=self.image_buffer[i]
            cv2.imwrite(os.path.join(trajectory_store_path,"image_%d.png"%i),rgb)
            video.write(rgb)
        video.release()

    def get_heatmap(self,id,pc,select_points):
        if type(pc)==torch.Tensor:
            pc=pc.cpu().numpy()
            pc=pc[0]
        pc=pc[:,:3]
        heatmap_path=os.path.join(self.store_path,"heatmap")
        if len(select_points)==1:
            get_one_point_heatmap(pc,pc[select_points[0]],np.zeros(512),save_path=heatmap_path,name=str(id))
        if len(select_points)==2:
            get_two_grasp_heatmap(pc,pc[select_points[0]],np.zeros(512),pc[select_points[1]],np.zeros(512),save_path=heatmap_path,name=str(id))
        if len(select_points)==4:
            get_four_points_heatmap(pc,pc[select_points[0]],np.zeros(512),pc[select_points[1]],np.zeros(512),pc[select_points[2]],np.zeros(512),pc[select_points[3]],np.zeros(512),save_path=heatmap_path,name=str(id))
        if len(select_points)==6:
            get_six_points_heatmap(pc,pc[select_points[0]],np.zeros(512),pc[select_points[1]],np.zeros(512),pc[select_points[2]],np.zeros(512),pc[select_points[3]],np.zeros(512),pc[select_points[4]],np.zeros(512),pc[select_points[5]],np.zeros(512),save_path=heatmap_path,name=str(id))


    def set_clothes(self,config,state=None,render_mode='cloth',step_sim_fn=lambda: pyflex.step()):
        if render_mode == 'particle':
            render_mode = 1
        elif render_mode == 'cloth':
            render_mode = 2
        elif render_mode == 'both':
            render_mode = 3
        camera_params = config['camera_config']
        print(camera_params)
        env_idx = hyper['softgym_clothes_env']
        scene_params = np.array([
            *config["cloth_config"]['cloth_pos'],
            *config["cloth_config"]['cloth_size'],
            *config["cloth_config"]['cloth_stiff'],
            render_mode,
            *camera_params['cam_position'][:],
            *camera_params['cam_angle'][:],
            camera_params['cam_size'][0],
            camera_params['cam_size'][1],
            config["cloth_config"]['cloth_mass'],
            config["cloth_config"]['flip_mesh']])
        print(scene_params)
        pyflex.set_clothes(
            scene_idx=env_idx,
            scene_params=scene_params,
            vertices=config["cloth_config"]['mesh_verts'],
            stretch_edges=config["cloth_config"]['mesh_stretch_edges'],
            bend_edges=config["cloth_config"]['mesh_bend_edges'],
            shear_edges=config["cloth_config"]['mesh_shear_edges'],
            faces=config["cloth_config"]['mesh_faces'],
            thread_idx=0)
        step_sim_fn()
        set_random_cloth_color()
        if state is not None:
            set_state(state)
        self.clothes.flatten_cloth()
        for j in range(300):
            self.step_sim_fn()
        return deepcopy(config)
    

    def empty_scene(self,config,state=None,render_mode='cloth',step_sim_fn=lambda: pyflex.step(),):

        if render_mode == 'particle':
            render_mode = 1
        elif render_mode == 'cloth':
            render_mode = 2
        elif render_mode == 'both':
            render_mode = 3
        
        pyflex.empty(6, config["scene_config"]())
        pyflex.add_cloth_mesh(
            position=config["cloth_config"]['cloth_pos'], 
            verts=config["cloth_config"]['mesh_verts'], 
            faces=config["cloth_config"]['mesh_faces'], 
            stretch_edges=config["cloth_config"]['mesh_stretch_edges'], 
            bend_edges=config["cloth_config"]['mesh_bend_edges'], 
            shear_edges=config["cloth_config"]['mesh_shear_edges'], 
            stiffness=config["cloth_config"]['cloth_stiff'], 
            mass=config["cloth_config"]['cloth_mass'])
        self.clothes.init_info()
        self.clothes.update_info()

        random_state = np.random.RandomState(np.abs(int(np.sum(config["cloth_config"]['mesh_verts']))))
        hsv_color = [
            random_state.uniform(0.0, 1.0),
            random_state.uniform(0.0, 1.0),
            random_state.uniform(0.6, 0.9)
        ]

        rgb_color = colorsys.hsv_to_rgb(*hsv_color)
        pyflex.change_cloth_color(rgb_color)
        pyflex.set_camera(config["camera_config"]())
        step_sim_fn()

        if state is not None:
            pyflex.set_positions(state['particle_pos'])
            pyflex.set_velocities(state['particle_vel'])
            pyflex.set_shape_states(state['shape_pos'])
            pyflex.set_phases(state['phase'])


        for j in range(50):
            self.step_sim_fn()
        
        return deepcopy(config)

    def get_default_config(self):
        return self.config 