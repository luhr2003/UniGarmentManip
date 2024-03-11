'''this file is used to convert the pickle file to h5 file for only useful data'''
import pickle
import sys
import os
from multiprocessing import Pool
import h5py
import argparse

from tqdm import tqdm
curpath=os.getcwd()
sys.path.append(curpath)
sys.path.append(curpath+'/garmentgym')

def pickle2h5(file_path):
    try:
        with open(file_path, 'rb') as f:
            info: cross_Deform_info = pickle.load(f)
    except:
        return
    h5_path = file_path.replace('.pkl', 'h5')
    with h5py.File(h5_path, 'w') as f:
        points = info.cur_info.points
        colors = info.cur_info.colors
        vertices = info.cur_info.vertices
        visible_indices = info.cur_info.visible_indices
        visible_vertices = info.cur_info.visible_vertices
        camera_intrisics = info.config.get_camera_matrix()[0]
        camera_extrisics = info.config.get_camera_matrix()[1]
        rgb=info.cur_info.rgb
        depth=info.cur_info.depth
        faces=info.cur_info.faces
        points_set = f.create_dataset('points', data=points)
        colors_set = f.create_dataset('colors', data=colors)
        vertices_set = f.create_dataset('vertices', data=vertices)
        visible_veritices_set = f.create_dataset('visible_vertices', data=visible_vertices)
        visible_indices_set = f.create_dataset('visible_indices', data=visible_indices)
        intrisics_set = f.create_dataset('camera_intrisics', data=camera_intrisics)
        extrisics_set = f.create_dataset('camera_extrisics', data=camera_extrisics)
        rgb_set=f.create_dataset('rgb',data=rgb)
        depth_set=f.create_dataset('depth',data=depth)
        faces_set=f.create_dataset('faces',data=faces)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/home/luhr/correspondence/softgym_cloth/cloth3d_train_data")

    data_path = parser.parse_args().data_path

    # 获取所有.pkl文件的路径
    file_paths = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.pkl'):
                print(file)
                file_paths.append(os.path.join(root, file))

    # 创建进程池，指定最大进程数为4
    pool = Pool(processes=20)
    

    # 使用进程池并行执行任务
    with tqdm(total=len(file_paths)) as pbar:
        for _ in pool.imap_unordered(pickle2h5, file_paths):
            pbar.update(1)

    # 关闭进程池
    pool.close()
    pool.join()