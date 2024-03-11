import colorsys
import os
import cv2
import numpy as np
import sys
sys.path.append(os.getcwd()+"/garmentgym")

from garmentgym.base.clothes import Clothes
from garmentgym.utils.flex_utils import center_object,set_random_cloth_color, set_state
import pyflex
from garmentgym.env.flex_env import FlexEnv
from garmentgym.utils.tool_utils import PickerPickPlace, Pickerpoint
from copy import deepcopy

from garmentgym.clothes_hyper import hyper
from garmentgym.base.config import *
from garmentgym.garmentgym.base.heatmap_render import get_four_points_heatmap,get_grasp_place_heatmap,get_one_point_heatmap,get_two_grasp_heatmap


class ClothesHangEnv(FlexEnv):
    def __init__(self,mesh_category_path:str,config:Config,clothes:Clothes=None):
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
        # self.action_tool = Pickerpoint(num_picker=config.task_config.num_picker, particle_radius=config.task_config.particle_radius, env=self, picker_threshold=config.task_config.picker_threshold,
        #                                        picker_low=(-5, 0., -5), picker_high=(5, 5, 5),picker_radius=config.task_config.picker_radius,picker_size=config.task_config.picker_size)
        self.up_camera=config["camera_config"]()
        self.vertice_camera=deepcopy(config.camera_config)
        self.vertice_camera.cam_position=[0, 2.7,  3]    #second is height;third is y

        self.vertice_camera.cam_angle=[0,-np.pi/7,0]     #5
        self.image_buffer=[]

    def get_heatmap(self,id,pc,select_points):
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




    def step_sim_fn(self):
        pyflex.step()
        rgb,depth=pyflex.render()
        rgb=np.flip(rgb.reshape([self.config.camera_config.cam_size[0],self.config.camera_config.cam_size[1],4]),0)[:,:,:3].astype(np.uint8)
        cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR,rgb)      
        self.image_buffer.append(rgb)
    
    def export_image(self):
        trajectory_store_path=os.path.join(self.store_path,"trajectory")
        os.makedirs(trajectory_store_path,exist_ok=True)
        video_path=os.path.join(trajectory_store_path,"video.mp4")
        video=cv2.VideoWriter(video_path,cv2.VideoWriter_fourcc(*'mp4v'),30,(self.config.camera_config.cam_size[0],self.config.camera_config.cam_size[1]))
        for i in range(len(self.image_buffer)):
            rgb=self.image_buffer[i]
            cv2.imwrite(os.path.join(trajectory_store_path,"image_%d.png"%i),rgb)
            video.write(rgb)
        video.release()

    def update_camera(self,id):
        if id ==0:
            pyflex.set_camera(self.up_camera)
            for j in range(5):
                self.step_sim_fn()
        else:
            pyflex.set_camera(self.vertice_camera())
            for j in range(5):
                self.step_sim_fn()
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


    def empty_scene(self,config,state=None,render_mode='cloth',step_sim_fn=lambda: pyflex.step(),):
        pyflex.empty(6, config["scene_config"]())
        
        pyflex.set_camera(config["camera_config"]())
        step_sim_fn()

        if state is not None:
            pyflex.set_positions(state['particle_pos'])
            pyflex.set_velocities(state['particle_vel'])
            pyflex.set_shape_states(state['shape_pos'])
            pyflex.set_phases(state['phase'])


        for j in range(10):
            self.step_sim_fn()
        
        return deepcopy(config)
    
    def add_cloth(self,config):
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
        center_object()

    def get_default_config(self):
        return self.config 

    def add_hang(self):
        '''
        modify this function to change the position of the hang
        center is the position of the hang
        param is the parameter of the hang which orginized as length*height*width
        you can modify the quat to change the orientation of the hang
        '''
        # 添加立柱
        center_vertic=[0.8,0,-0.8]
        quat = self.quatFromAxisAngle([0, 0, -1.], 0.)
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
            self.step_sim_fn()


if __name__=="__main__":
    config=Config()
    env=ClothesHangEnv(mesh_category_path="/home/yiyan/correspondence/softgym_cloth/garmentgym/cloth3d/train",config=config)
    env.empty_scene(config)
    env.add_cloth(config)
    for j in range(100):
        pyflex.step()
        pyflex.render()
    env.update_camera(1)
    for j in range(50):
        pyflex.step()
        pyflex.render()
    env.add_hang()
    for j in range(100):
        pyflex.step()
        pyflex.render()