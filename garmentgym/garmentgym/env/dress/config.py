from typing import Any
import numpy as np
from scipy.spatial.transform import Rotation


class Config:
    def __init__(self,kwargs:dict=None) -> None:
        self.basic_config=basic_Config()
        self.task_config=task_Config()
        self.camera_config=camera_Config()
        self.cloth_config=cloth_Config()
        self.scene_config=scene_Config()

        if kwargs is not None:
            self.update(kwargs)

    def __getitem__(self,key):
        return getattr(self,key)
    
    def update(self,kwargs):
        for key in kwargs:
            self[key].update(kwargs[key])
    def __str__(self):
        return "basic_config"+str(self.basic_config)+"\n"+"scene_config"+str(self.scene_config)+'\n'+"task_config"+str(self.task_config)+'\n'+"camera_config"+str(self.camera_config)+'\n'+"cloth_config"+str(self.cloth_config)
    def __call__(self):
        return self.basic_config(),self.task_config(),self.camera_config(),self.cloth_config(),self.scene_config()    
    
    def get_camera_matrix(self):
        focal_length = self.camera_config.cam_size[0] / 2 / np.tan(self.camera_config.cam_fov / 2)
        cam_intrinsics = np.array([[focal_length, 0, float(self.camera_config.cam_size[1])/2],
                                [0, focal_length, float(self.camera_config.cam_size[0])/2],
                                [0, 0, 1]])
        cam_extrinsics = np.eye(4)
        rotation_matrix = Rotation.from_euler('xyz', [self.camera_config.cam_angle[1], np.pi - self.camera_config.cam_angle[0], np.pi], degrees=False).as_matrix()
        cam_extrinsics[:3, :3] = rotation_matrix
        cam_extrinsics[0, 3] = self.camera_config.cam_position[0]
        cam_extrinsics[1, 3] = self.camera_config.cam_position[2]
        cam_extrinsics[2, 3] = self.camera_config.cam_position[1]
        

        return cam_intrinsics, cam_extrinsics


class task_Config:
    def __init__(self):
        self.observation_mode='cam_rgb'
        self.action_mode='pickerpickplace'
        self.num_picker=2
        self.render=True
        self.headless=False
        self.horizon=100
        self.action_repeat=8
        self.render_mode='cloth'
        self.picker_radius=0.1
        self.picker_threshold=0.005
        self.particle_radius=0.00625
        self.picker_size=0.05
        self.speed=0.01
    def update(self,kwargs):
        for key in kwargs:
            setattr(self,key,kwargs[key])
    def __getitem__(self,key):
        return getattr(self,key)
    def __str__(self):
        return str(self.__dict__)
    def reset(self):
        self.__init__()
    def __call__(self):
        return self.__dict__

class camera_Config:
    def __init__(self):
        self.name='default_camera'
        self.render_type=["cloth"]
        self.cam_position=[0, 2.5, 0.0]
        self.cam_angle=[0,-np.pi/2,0]
        self.cam_size=[720,720]
        # self.cam_fov=39.5978/180*np.pi
        self.cam_fov=np.pi/4
    def update(self,kwargs):
        for key in kwargs:
            setattr(self,key,kwargs[key])
    def __getitem__(self,key):
        return getattr(self,key)
    def __str__(self):
        return str(self.__dict__)
    def reset(self):
        self.__init__()
    def __call__(self):
        val=self.__dict__
        if 'name' in val:
            val.pop('name')
        return val
    
class cloth_Config:
    def __init__(self):
        self.scale=1.2
        self.cloth_pos=[0.0, 0, 0.0]
        self.cloth_size_scale=1.2
        self.cloth_size=[-1,-1]
        self.cloth_stiff=(1.5, 0.04, 0.04)
        self.mesh_verts=None
        self.mesh_faces=None
        self.mesh_nocs_verts=None
        self.mesh_shear_edges=None
        self.mesh_bend_edges=None
        self.mesh_stretch_edges=None
        self.cloth_mass=5
        self.flip_mesh=0
        self.num_particles=-1
    def update(self,kwargs):
        for key in kwargs:
            setattr(self,key,kwargs[key])
    def __getitem__(self,key):
        return getattr(self,key)
    def __str__(self):
        return str(self.__dict__)
    def reset(self):
        self.__init__()
    def __call__(self):
        return self.__dict__
    
class scene_Config:
    def __init__(self):
        self.scene_id=0
        self.radius=0.01
        self.buoyancy=0
        self.numExtraParticles=50000
        self.collisionDistance=0.0006
        self.msaaSamples=0
    def update(self,kwargs):
        for key in kwargs:
            setattr(self,key,kwargs[key])
    def __getitem__(self,key):
        return getattr(self,key)
    def __str__(self):
        return str(self.__dict__)
    def reset(self):
        self.__init__()
    def __call__(self):
        return self.__dict__
    
class basic_Config:
    def __init__(self):
        self.grasp_height_single=0.3
        self.grasp_height_double=0.05
        self.gui=True
    def update(self,kwargs):
        for key in kwargs:
            setattr(self,key,kwargs[key])
    def __getitem__(self,key):
        return getattr(self,key)
    def __str__(self):
        return str(self.__dict__)
    def reset(self):
        self.__init__()
    def __call__(self):
        return self.__dict__