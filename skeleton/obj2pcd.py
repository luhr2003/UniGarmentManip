import open3d as o3d
import os
import numpy as np
import argparse

def load_cloth_mesh(path):
    vertices, faces = [], []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        # 3D vertex
        if line.startswith('v '):
            vertices.append([float(n)
                             for n in line.replace('v ', '').split(' ')])
    
    return np.array(vertices)


def get_rotation_matrix(rotationVector, angle):
    angle = float(angle)
    axis = rotationVector/np.sqrt(np.dot(rotationVector , rotationVector))
    a = np.cos(angle/2)
    b,c,d = -axis*np.sin(angle/2.)
    return np.array( [ [a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                       [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                       [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c] ])




if __name__=="__main__":
    parser= argparse.ArgumentParser()
    parser.add_argument("--data_path",type=str, default="")

    args=parser.parse_args()
    data_path=args.data_path

    for root,dirs,files in os.walk(data_path):
        for file in files:
            if file.endswith(".obj"):
                print(os.path.join(root,file))
                vertices=load_cloth_mesh(os.path.join(root,file))
                pcd=o3d.geometry.PointCloud()
                pcd.points=o3d.utility.Vector3dVector(vertices)
                print("write to ",os.path.join(root,file.replace(".obj",".pcd")))
                # o3d.visualization.draw_geometries([pcd])
                o3d.io.write_point_cloud(os.path.join(root,file.replace(".obj",".pcd")),pcd)
