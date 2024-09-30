
import os
import sys
curpath=os.getcwd()
sys.path.append(curpath)
sys.path.append(os.path.join(curpath,'code'))
from train.dataloader.dataloader import Dataset
from base.config import Config
import argparse
from base.utils import *
from tensorboardX import SummaryWriter
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
from model.basic_pn import basic_model
import random
from info_nce import InfoNCE
from val.simple_val import *
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def train(config:Config,writer:SummaryWriter,checkpoint_dir:str,log_train_path:str,log_val_path:str):
    # produce dataset train val dataloader
    train_dataset=Dataset(deform_path=deform_path,object_path=object_path,config=config,flag="train")
    val_dataset=Dataset(deform_path=deform_path,object_path=object_path,config=config,flag="val")
    train_dataloader=DataLoaderX(train_dataset,batch_size=config.train_config.batch_size,shuffle=True,num_workers=20)
    val_dataloader=DataLoaderX(val_dataset,batch_size=config.train_config.batch_size,shuffle=True,num_workers=20)
    model=basic_model(config.train_config.feature_dim)
    if resume:
        assert resume_path is not None,"resume path is None"
        print("resume from {}".format(resume_path))
        model.load_state_dict(torch.load(resume_path,map_location=config.train_config.device)['model_state_dict'])
        optimizer=torch.load(resume_path,map_location=config.train_config.device)['optimizer']
    else:
        # produce optimizer
        optimizer=torch.optim.Adam(model.parameters(),lr=config.train_config.lr,weight_decay=config.train_config.weight_decay)

    # produce model
    model.to(config.train_config.device)


    # produce loss function
    criterion=InfoNCE(negative_mode='paired',temperature=config.train_config.temperature)


    # train
    train_step=0
    val_step=0 
    for epoch in range(config.train_config.epoch):
        model.train()
        train_length=len(train_dataloader)
        total_loss=0
        for i,(pc1,pc2,correspondence) in enumerate(train_dataloader):
            # pc1 batchsize*num_points*6
            # pc2 batchsize*num_points*6
            # correspondence batchsize*num_correspondence*2
            batchsize=pc1.shape[0]
            num_points=pc1.shape[1]
            num_correspondence=correspondence.shape[1]



            pc1=pc1.to(config.train_config.device)
            pc2=pc2.to(config.train_config.device)
            correspondence=correspondence.to(config.train_config.device)


            #pc1_output batchsize*num_points*feature_dim
            #pc2_output batchsize*num_points*feature_dim
            pc1_output=model(pc1)
            pc2_output=model(pc2)
            feature_dim=pc1_output.shape[2]



            batch_index=torch.arange(0,pc1.shape[0])

            # query batchsize*num_correspondence*feature_dim
            # query=torch.stack([pc1_output[batch_index,correspondence[:,i,0]]for i in range(correspondence.shape[1])],dim=1)
            # print("query shape",query.shape)
            query=pc1_output.gather(1,correspondence[:,:,0].unsqueeze(2).expand(-1,-1,feature_dim))


            # positive batchsize*num_correspondence*feature_dim
            # positive=torch.stack([pc2_output[batch_index,correspondence[:,i,1]]for i in range(correspondence.shape[1])],dim=1)
            # print("positive shape",positive.shape)
            positive=pc2_output.gather(1,correspondence[:,:,1].unsqueeze(2).expand(-1,-1,feature_dim))



            # negative batchsize*num_correspondence*num_negative*feature_dim
            # negative here is random select not in correspondence
            # negative_index batchsize*num_correspondence*num_negative
            negative_index=torch.randint(0,num_points,(batchsize,num_correspondence,config.train_config.num_negative)).to(config.train_config.device)
            negative_index=negative_index.reshape(batchsize,num_correspondence*config.train_config.num_negative)
            negative=pc2_output.gather(1,negative_index.unsqueeze(2).expand(-1,-1,feature_dim))
            negative=negative.reshape(batchsize,num_correspondence,config.train_config.num_negative,feature_dim)
            # print("negative shape",negative.shape)



            num_negative=negative.shape[2]
            query=query.reshape(batchsize*num_correspondence,feature_dim)
            positive=positive.reshape(batchsize*num_correspondence,feature_dim)
            negative=negative.reshape(batchsize*num_correspondence,num_negative,feature_dim)


            loss=criterion(query,positive,negative)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss+=loss.item()

            inference = cal_inference_pair(pc1_output,pc2_output,correspondence,config).long()
            distance,accuracy=cal_distance_accuracy(pc1,pc2,inference,correspondence,config)
            distance=distance.item()
            accuracy=accuracy.item()




            print("train: epoch: {}, batch: {},loss: {}, average_loss: {}, distance: {}, accuracy: {}".format(epoch,i,loss.item(),total_loss/(i+1),distance,accuracy))
            

            with open(log_train_path,'a') as f:
                f.write("train: epoch: {}, batch: {},loss: {}, average_loss: {}, distance: {}, accuracy: {}\n".format(epoch,i,loss.item(),total_loss/(i+1),distance,accuracy))
            writer.add_scalar('train_loss',loss.item(),train_step)
            writer.add_scalar('train_average_loss',total_loss/(i+1),train_step)
            writer.add_scalar('train_distance',distance,train_step)
            writer.add_scalar('train_accuracy',accuracy,train_step)
            train_step+=1
            
            if i > config.train_config.batch_num:
                break
            if i % config.record_config.record_interval == 0:
                torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'optimizer':optimizer,"model":model},os.path.join(checkpoint_dir,'checkpoint_{}_{}.pth'.format(epoch,i)))


        model.eval()
        with torch.no_grad():
            val_total_loss=0
            for i,(pc1,pc2,correspondence) in enumerate(val_dataloader):
                # pc1 batchsize*num_points*6
                # pc2 batchsize*num_points*6
                # correspondence batchsize*num_correspondence*2
                batchsize=pc1.shape[0]
                num_points=pc1.shape[1]
                num_correspondence=correspondence.shape[1]



                pc1=pc1.to(config.train_config.device)
                pc2=pc2.to(config.train_config.device)
                correspondence=correspondence.to(config.train_config.device)




                #pc1_output batchsize*num_points*feature_dim
                #pc2_output batchsize*num_points*feature_dim
                pc1_output=model(pc1)
                pc2_output=model(pc2)
                feature_dim=pc1_output.shape[2]


                batch_index=torch.arange(0,pc1.shape[0])

                # query batchsize*num_correspondence*feature_dim
                query=pc1_output.gather(1,correspondence[:,:,0].unsqueeze(2).expand(-1,-1,feature_dim))


                # positive batchsize*num_correspondence*feature_dim
                positive=pc2_output.gather(1,correspondence[:,:,1].unsqueeze(2).expand(-1,-1,feature_dim))



                # negative batchsize*num_correspondence*num_negative*feature_dim
                # negative here is random select not in correspondence
                # negative_index batchsize*num_correspondence*num_negative
                negative_index=torch.randint(0,num_points,(batchsize,num_correspondence,config.train_config.num_negative)).to(config.train_config.device)
                negative_index=negative_index.reshape(batchsize,num_correspondence*config.train_config.num_negative)
                negative=pc2_output.gather(1,negative_index.unsqueeze(2).expand(-1,-1,feature_dim))
                negative=negative.reshape(batchsize,num_correspondence,config.train_config.num_negative,feature_dim)



                
                num_negative=negative.shape[2]
                query=query.reshape(batchsize*num_correspondence,feature_dim)
                positive=positive.reshape(batchsize*num_correspondence,feature_dim)
                negative=negative.reshape(batchsize*num_correspondence,num_negative,feature_dim)
                loss=criterion(query,positive,negative)
                val_total_loss+=loss.item()

                inference = cal_inference_pair(pc1_output,pc2_output,correspondence,config).long()
                distance,accuracy=cal_distance_accuracy(pc1,pc2,inference,correspondence,config)
                distance=distance.item()
                accuracy=accuracy.item()



                print("val: epoch:{},batch:{},loss:{},average_loss:{},distance:{},accuracy:{}".format(epoch,i,loss.item(),val_total_loss/(i+1),distance,accuracy))
                with open(log_val_path,'a') as f:
                    f.write("val: epoch:{},batch:{},loss:{},average_loss:{},distance:{},accuracy:{}\n".format(epoch,i,loss.item(),val_total_loss/(i+1),distance,accuracy))
                writer.add_scalar('val_loss',loss.item(),val_step)
                writer.add_scalar('val_average_loss',val_total_loss/(i+1),val_step)
                writer.add_scalar('val_distance',distance,val_step)
                writer.add_scalar('val_accuracy',accuracy,val_step)
                if i > config.train_config.batch_num//100:
                    break
            torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'optimizer':optimizer,"model":model},os.path.join(checkpoint_dir,'checkpoint_{}.pth'.format(epoch)))



    

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--logs_dir',type=str,default="./logs/aver_nce/")
    parser.add_argument('--prefix',type=str,default="aver_nce")
    parser.add_argument('--resume',type=bool,default=False)
    parser.add_argument('--resume_path',type=str,default="")
    parser.add_argument("--device",type=str,default="cuda:0")
    parser.add_argument("--deform_path",type=str,default="/home/isaac/correspondence/tshirt_move")
    parser.add_argument("--object_path",type=str,default="./UniGarmentManip/garmentgym/tops")

    args=parser.parse_args()
    logs_dir=args.logs_dir
    prefix=args.prefix
    resume=args.resume
    resume_path=args.resume_path
    device=args.device
    deform_path=args.deform_path
    object_path=args.object_path

    force_mkdir(logs_dir)
    curdirs=os.path.join(logs_dir,prefix)
    force_mkdir(curdirs)
    tensorboard_dir=os.path.join(curdirs,'tensorboard')
    force_mkdir(tensorboard_dir)
    checkpoint_dir=os.path.join(curdirs,'checkpoint')
    force_mkdir(checkpoint_dir)
    log_dir=os.path.join(curdirs,'log')
    force_mkdir(log_dir)
    log_train_path=os.path.join(log_dir,'train.txt')
    log_val_path=os.path.join(log_dir,'val.txt')


    config=Config()
    config.train_config.device=device
    torch.cuda.set_device(config.train_config.device)

    writer=SummaryWriter(tensorboard_dir)
    train(config,writer,checkpoint_dir,log_train_path,log_val_path)
    
