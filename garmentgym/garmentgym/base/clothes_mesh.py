import trimesh
from garmentgym.utils.clothes_utils import load_cloth_mesh,load_cloth_urs
import numpy as np
class clothes_mesh:
    def __init__(self,path:str,name:str,need_urs:bool) -> None:
        self.name=name
        self.path=path
        self.need_urs=need_urs
        self.urs=None
        self.vertices=np.array([])
        self.faces=np.array([])
        self.stretch_edges=np.array([])
        self.bend_edges=np.array([])
        self.shear_edges=np.array([])
        self.cloth_pos=np.array([])
        self.cloth_size_scale=np.array([])
        self.cloth_mass=np.array([])
        self.cloth_stiff=np.array([])
        self.cloth_trimesh=None
        self.num_particles=-1
        self.flattened_area=-1
        self.load_mesh()
        self.cloth_height=-1
        self.cloth_width=-1
        assert self.vertices.shape[0]>0,'vertices is empty,load_mesh_failed'
    def load_mesh(self):
        if not self.need_urs:
            self.vertices,self.faces,self.stretch_edges,self.bend_edges,self.shear_edges=load_cloth_mesh(self.path)
        else:
            self.vertices,self.faces,self.stretch_edges,self.bend_edges,self.shear_edges,self.urs=load_cloth_urs(self.path)
    def set_config(self,cloth_pos:np.ndarray,cloth_size_scale:np.ndarray,cloth_mass:np.ndarray,cloth_stiff:np.ndarray):
        self.cloth_pos=cloth_pos
        self.cloth_size_scale=cloth_size_scale
        self.cloth_mass=cloth_mass
        self.cloth_stiff=cloth_stiff
        self.vertices=self.vertices*self.cloth_size_scale
        self.num_particles=self.vertices.shape[0]
        self.cloth_trimesh=trimesh.Trimesh(self.vertices,self.faces)
    
    def get_config_dict(self):
        if not self.need_urs:
            return {'cloth_config':{
                'cloth_pos':self.cloth_pos,
                'cloth_size_scale':self.cloth_size_scale,
                'cloth_mass':self.cloth_mass,
                'cloth_stiff':self.cloth_stiff,
                'cloth_size':[-1,-1],
                'mesh_verts':self.vertices.reshape(-1),
                'mesh_faces':self.faces.reshape(-1),
                'mesh_stretch_edges':self.stretch_edges.reshape(-1),
                'mesh_bend_edges':self.bend_edges.reshape(-1),
                'mesh_shear_edges':self.shear_edges.reshape(-1),
                'flip_mesh':0,
                'num_particles':self.num_particles,
            }}
        else:
            return {'cloth_config':{
                'cloth_pos':self.cloth_pos,
                'cloth_size_scale':self.cloth_size_scale,
                'cloth_mass':self.cloth_mass,
                'cloth_stiff':self.cloth_stiff,
                'mesh_verts':self.vertices.reshape(-1),
                'mesh_faces':self.faces.reshape(-1),
                'mesh_stretch_edges':self.stretch_edges.reshape(-1),
                'mesh_bend_edges':self.bend_edges.reshape(-1),
                'mesh_shear_edges':self.shear_edges.reshape(-1),
                'mesh_nocs_verts':self.urs.reshape(-1),
            }}
        