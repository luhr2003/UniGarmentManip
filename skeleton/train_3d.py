import os
import random
import argparse
import contextlib

import numpy as np
import torch
import torch.optim as optim
from merger.data_flower import all_h5
from merger.merger_net import Net
from merger.composed_chamfer import composed_sqrt_chamfer
import open3d as o3d
from tensorboardX import SummaryWriter


arg_parser = argparse.ArgumentParser(description="Training Skeleton Merger. Valid .h5 files must contain a 'data' array of shape (N, n, 3) and a 'label' array of shape (N, 1).", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
arg_parser.add_argument('-t','--data_dir',type=str,default='')
arg_parser.add_argument('-m', '--checkpoint-path', '--model-path', type=str, default='jumpsuit.pt',
                        help='Model checkpoint file path for saving.')
arg_parser.add_argument('-k', '--n-keypoint', type=int, default=50,
                        help='Requested number of keypoints to detect.')
arg_parser.add_argument('-d', '--device', type=str, default='cuda',
                        help='Pytorch device for training.')
arg_parser.add_argument('-b', '--batch', type=int, default=32,
                        help='Batch size.')
arg_parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='Number of epochs to train.')
arg_parser.add_argument('--max-points', type=int, default=5000,
                        help='Indicates maximum points in each input point cloud.')
arg_parser.add_argument('--log_dir',type=str,default="log")
arg_parser.add_argument('--prefix',type=str,default="")


def L2(embed):
    return 0.01 * (torch.sum(embed ** 2))


def feed(net, optimizer, x_set, train, shuffle, batch, epoch,logs):
    running_loss = 0.0
    running_lrc = 0.0
    running_ldiv = 0.0
    net.train(train)
    if shuffle:
        x_set = list(x_set)
        random.shuffle(x_set)
    with contextlib.suppress() if train else torch.no_grad():
        for i in range(len(x_set) // batch):
            idx = slice(i * batch, (i + 1) * batch)
            refp = next(net.parameters())
            batch_x = torch.tensor(x_set[idx], device=refp.device)
            if train:
                optimizer.zero_grad()
            RPCD, KPCD, KPA, LF, MA = net(batch_x)
            blrc = composed_sqrt_chamfer(batch_x, RPCD, MA)
            bldiv = L2(LF)
            loss = blrc + bldiv
            if train:
                loss.backward()
                optimizer.step()
    
            # print statistics
            running_lrc += blrc.item()
            running_ldiv += bldiv.item()
            running_loss += loss.item()
            print('[%s%d, %4d] loss: %.4f Lrc: %.4f Ldiv: %.4f' %
                  ('VT'[train], epoch, i, running_loss / (i + 1), running_lrc / (i + 1), running_ldiv / (i + 1)))
            with open(logs,'a') as f:
                f.write('[%s%d, %4d] loss: %.4f Lrc: %.4f Ldiv: %.4f\n' %
                  ('VT'[train], epoch, i, running_loss / (i + 1), running_lrc / (i + 1), running_ldiv / (i + 1)))

    return running_loss / (i + 1), running_lrc / (i + 1), running_ldiv / (i + 1)


def process_data(dataset):
    data=[]
    for root, dirs, files in os.walk(dataset):
        for file in files:
            if file.endswith(".pcd"):
                pcd=o3d.io.read_point_cloud(os.path.join(root,file))
                points=np.asarray(pcd.points)
                # down sample to 10000 points
                points=points[np.random.choice(points.shape[0],5000,replace=True)]
                data.append(np.expand_dims(points,0).astype(np.float32))

    return np.concatenate(data,axis=0)

def make_dir(path):
    os.makedirs(path,exist_ok=True)




if __name__ == '__main__':
    ns = arg_parser.parse_args()
    dataset=ns.data_dir
    batch = ns.batch
    x = process_data(dataset)
    prefix=ns.prefix
    print(x.shape)
    net = Net(ns.max_points, ns.n_keypoint).to(ns.device)
    optimizer = optim.Adadelta(net.parameters(), eps=1e-2)

    logdir=ns.log_dir
    make_dir(logdir)
    logdir=os.path.join(logdir,prefix)
    make_dir(logdir)
    log_dir_path=os.path.join(logdir,"log")
    make_dir(log_dir_path)
    log_path=os.path.join(log_dir_path,"train.txt")
    tensorboard_dir=os.path.join(logdir,"tensorboard")
    make_dir(tensorboard_dir)
    checkpoint_dir=os.path.join(logdir,"checkpoints")
    make_dir(checkpoint_dir)

    writer=SummaryWriter(logdir=tensorboard_dir)

    for epoch in range(ns.epochs):
        feed(net, optimizer, x, True, True, batch, epoch, log_path)
        current_checkpoint=os.path.join(checkpoint_dir,str(epoch)+".pth")
        torch.save({
            'epoch': epoch,
            'net':net,
            'model_state_dict':net.state_dict(),
        }, current_checkpoint)


