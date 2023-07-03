import argparse, time, os, pickle
import numpy as np
import math
import random
import subprocess
import os, psutil

import dgl
import torch
import torch.optim as optim
import pdb

from models_final import SHARC, main_model, Segmentation
from dataset_final import SHARCDataset_withmfcc, SHARCDataset
from utils_train import *
import time
import sys
###########
# ArgParser
def arguments():
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--train_files', type=str)
    parser.add_argument('--levels', type=str, default='10')
    parser.add_argument('--faiss_gpu', action='store_true')
    parser.add_argument('--model_filename', type=str, default='lander.pth')

    # KNN
    parser.add_argument('--knn_k', type=str, default='10')
    parser.add_argument('--num_workers', type=int, default=0)

    # Model
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--num_conv', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--gat', action='store_true')
    parser.add_argument('--gat_k', type=int, default=1)
    parser.add_argument('--balance', action='store_true')
    parser.add_argument('--use_cluster_feat', action='store_true')
    parser.add_argument('--use_focal_loss', action='store_true')

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--ngpu', type=str, default=0)

    #Xvector training 
    parser.add_argument('--featspath', type=str, default=None)
    parser.add_argument('--reco2utt_list', type=str, default=None)
    parser.add_argument('--segments_list', type=str, default=None)
    parser.add_argument('--dataset_str', type=str, default=None)
    parser.add_argument('--xvec_model_weight_path',type=str,default=None)
    parser.add_argument('--xvec_dim',type=int,default=512)
    parser.add_argument('--file_pairs', type=str, default=None)
    parser.add_argument('--xvecpath', type=str, default=None)
    parser.add_argument('--labelspath', type=str, default=None)
    parser.add_argument('--model_savepath',type=str,default='e2e_lander.pth')
    parser.add_argument('--filegroupcount',type=int,default=1)
    parser.add_argument('--feats_norm',type=int,default=0,help='Normalize features before GNN')
    parser.add_argument('--resume',type=int,default=0, help='resume the training')
    parser.add_argument('--mymode', type=str, default='train',help='options are train/val/val_orgxvec')
    parser.add_argument('--fulltrain',type=int,default=0, help=' full end to end training')
    parser.add_argument('--pldamodel',type=str,default=None)
    parser.add_argument('--xvecbeta',type=float,default=0.5)
    parser.add_argument('--modified',type=int,default=0)
    parser.add_argument('--withcheckpoint',type=int,default=0, help='model is a dictionary with model and optimizerm or not')
    parser.add_argument('--samelr',type=int,default=0, help='same lr for xvec and SHARC')
    args = parser.parse_args()
    # print(args)
    return args
args = arguments()
def cuda_gpu_available():
    try:
        gpu_available_info = subprocess.Popen('/state/partition1/softwares/Kaldi_Sept_2020/kaldi/src/nnet3bin/cuda-gpu-available', stderr=subprocess.PIPE)
        gpu_available_info = gpu_available_info.stderr.read().decode('utf-8')
        Node = gpu_available_info[86:97]
        gpu = int(gpu_available_info[gpu_available_info.index('active GPU is') + 15])
    except:
        return False
    print("Successfully selected the free GPU {} in {}.".format(gpu, Node))
    return gpu



###########################
# Environment Configuration

if int(args.ngpu) >= 0 :
    # args.ngpu = cuda_gpu_available()
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.ngpu)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    print('GPU selected: ',args.ngpu)
    # torch.cuda.set_device(args.ngpu)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    device = 'cpu'
print(device)

#dataloader for batch-iterating over a set of nodes, generating the list
#  of message flow graphs (MFGs) as computation dependency of the said minibatch
def set_train_sampler_loader(g, k):
    fanouts = [k-1 for i in range(args.num_conv + 1)]
    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
    # fix the number of edges
    train_dataloader = dgl.dataloading.NodeDataLoader(
        g, torch.arange(g.number_of_nodes()), sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers
    )
    return train_dataloader

#train_loaders = []
#for gidx, g in enumerate(gs):
#    train_dataloader = set_train_sampler_loader(gs[gidx], ks[gidx])
#    train_loaders.append(train_dataloader)

##################

##################
# Data Preparation
#with open(args.data_path, 'rb') as f:
#    features, labels = pickle.load(f)

#features = np.load("data/dihard/non_overlap_feats.npy")
#labels = np.load("data/dihard/non_overlap_labels.npy")
#features = features[:-25859,:]
#labels = labels[:-25859]

mode = args.mode.split(',')
if 'ami'in args.dataset_str:
    uniqidpath="data/ami_train/labels_train/uniq_spk_ids"
else:
    uniqidpath="librivox_uniq_spk_ids"
    pair_list = open(args.file_pairs).readlines()
    fid2fname = {}
    for line in pair_list:
        fid, fname = line.split()
        fid2fname[fid] = fname

with open(uniqidpath,"r") as f:
    spk_ids = f.readlines()

spk_dct = {}
for i,spk_id in enumerate(spk_ids):
    spk_dct[spk_id[:-1]] = i

# xvec_files=f'{args.train_files}_xvecs'
# lab_files=f'{args.train_files}_labels'
# with open(xvec_files,"r") as f:
#     xvec_files = f.readlines()
# with open(lab_files,"r") as f:
#     lab_files = f.readlines()


k_list = [int(k) for k in args.knn_k.split(',')]
lvl_list = [int(l) for l in args.levels.split(',')]

featsdict = {}
with open(args.featspath) as fpath:
    for line in fpath: 
        key, value = line.split(" ",1)
        featsdict[key] = value.rsplit()[0]

# segmentsfile = f'{args.segments_list}'
# frame_step = 0.01
# speech_segments = Segmentation.read_segments_file(
#                 segmentsfile, step=frame_step)

reco2utt_list = open(args.reco2utt_list).readlines()
reco2utt_dict = {}
for line in reco2utt_list:
    rec, utt = line.split(" ",1)
    reco2utt_dict[rec] = utt

def train(model,opt, schedulerdict=None):
    filepath = 'lists/{}/train.list'.format(args.dataset_str)
    train_list = np.genfromtxt(filepath,dtype=float).astype(int)
    filelist = np.array(list(reco2utt_dict.keys()))
    train_filelist = filelist[train_list]
    filegroup_count = args.filegroupcount
    ###############
    # Training Loop
    for epoch in range(args.epochs):
        if args.resume>epoch:
            continue
        gs = []
        nbrs = []
        ks = []
        counter = 0
        labels_group = []

        # for xf, lf in zip(xvec_files[-10:],lab_files[-10:]):
        loss_den_val_total = []
        loss_conn_val_total = []
        loss_val_total = []
        num_graphs = 0
        batch = 0
        for recid in train_filelist:
            counter += 1
            feats_list = []
            labels = []
            # feats_fname = xf[:-1]
            # labels_fname = lf[:-1]
            feats_fname = f'{args.xvecpath}/{recid}.npy'
            labels_fname = f'{args.labelspath}/labels_{fid2fname[recid]}'
            # print(feats_fname,labels_fname)
            features = np.load(feats_fname) 
            # recid = feats_fname.split('/')[-1].split('.')[0]
            print(recid)
            
            reco2utt = reco2utt_dict[recid]
            # segmentation = speech_segments[feats_fname]
            mfcc_feats,idx_xvec = load_mfcc_feats_nosilence(args,recid,featsdict,reco2utt,featsbatch=1)
            clean_ind = []
            with open(labels_fname,"r") as f:
                label_file = f.readlines()
            for i,line in enumerate(label_file):
                label = line.split()
                if len(label) == 2:
                    clean_ind.append(i)
                    feats_list.append(features[i])
                    labels.append(spk_dct[label[1]])
            #pdb.set_trace()
            clean_ind = np.array(clean_ind)
            mfcc_feats = mfcc_feats[clean_ind]
            print('per file mfcc: ',mfcc_feats.shape)
            if counter % filegroup_count == 1 or filegroup_count == 1:
                features_group = np.vstack(feats_list)    
                mfcc_features_group = mfcc_feats.copy()
            else:
                features_group = np.vstack((features_group,feats_list))
                mfcc_features_group = np.vstack((mfcc_features_group,mfcc_feats))
                
            labels_group.extend(labels)
            
            if counter % filegroup_count == 0:
                
                mfcc_features = torch.FloatTensor(mfcc_features_group).to(device)
                org_xvecs = torch.FloatTensor(features_group).to(device)
                print('mfcc: ',mfcc_features.shape)
                xvecs = model.extract_xvecs(mfcc_features,org_xvecs)
                labels_group = np.array(labels_group)
                print("Create graphs for recording %s"%(recid))
            #dataset preparation includes creating ground truth graph levels for different level list with different knn
                for k, l in zip(k_list, lvl_list):
                    dataset = SHARCDataset_withmfcc(mfcc_xvec_features=xvecs, features=features_group, 
                                                    labels=labels_group, mode=mode, k=k,
                                                    levels=l,device=device,feats_norm=args.feats_norm)
                    
                    gs += [g for g in dataset.gs]
                    ks += [k for g in dataset.gs]
                    nbrs += [nbr for nbr in dataset.nbrs]
                labels_group = []
                num_graphs = len(gs)
                
            if (epoch ==0 and counter == filegroup_count) or args.resume > 0:
                num_batches = math.ceil((num_graphs*len(train_filelist))/(filegroup_count*args.batch_size))
                print('num_batches: ',num_batches)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(opt,
                                                            T_max=args.epochs * num_batches,
                                                            eta_min=1e-5)
                if args.resume>0:
                    scheduler.load_state_dict(schedulerdict)
            if num_graphs >= args.batch_size:
                # bp()
                print("Num graphs = %d"%(len(gs)))
                print('Graphs Prepared.',flush=True)
                graph_batch = list(range(num_graphs))
                random.shuffle(graph_batch)
                opt.zero_grad()
                for graph_id in graph_batch:
                    # get the feature for the input_nodes
                    g = gs[graph_id].to(device)
                    
                    output_bipartite = model(g)
                    loss, loss_den_val, loss_conn_val = model.compute_loss(output_bipartite)
                    loss_den_val_total.append(loss_den_val)
                    loss_conn_val_total.append(loss_conn_val)
                    loss_val_total.append(loss.item())
                    loss.backward(retain_graph=True)
                    del g
                    torch.cuda.empty_cache()
                    if (batch + 1) % 10 == 0:
                        print('epoch: %d, batch: %d / %d, loss: %.6f, loss_den: %.6f, loss_conn: %.6f'%
                            (epoch, batch, num_batches, loss.item(), loss_den_val, loss_conn_val))

                opt.step()
                scheduler.step()
                gs = []
                nbrs = []
                ks = []
                num_graphs = 0
                batch = batch + 1
                torch.cuda.empty_cache()
                
                
        print('############################################################################################')
        print('epoch: %d loss: %.6f loss_den: %.6f loss_conn: %.6f'%
        (epoch, np.array(loss_val_total).mean(),
        np.array(loss_den_val_total).mean(), np.array(loss_conn_val_total).mean()),flush=True)
        print('############################################################################################')
        if (epoch+1) % 5 == 0:
            checkpoints ={}
            checkpoints['model'] = model.state_dict()
            checkpoints['optimizer'] = opt.state_dict()
            checkpoints['scheduler'] = scheduler.state_dict()
            torch.save(model.state_dict(), f'{args.model_savepath}/model_final.pth')
            torch.save(checkpoints,f'{args.model_savepath}/model_{epoch+1}_snapshot.pth')

    checkpoints ={}
    checkpoints['model'] = model.state_dict()
    checkpoints['optimizer'] = opt.state_dict()
    checkpoints['scheduler'] = scheduler.state_dict()
    torch.save(checkpoints,f'{args.model_savepath}/model_final.pth')
    # torch.save(model.state_dict(), args.model_savepath)


def train_ami(model,opt, schedulerdict=None):
    '''
    one graph is created from 500 frames of one file and then no. of graphs is batch size =2,
    then go to next 500 frames of the same and repeat 
    '''
    filepath = 'lists/{}/shuffled_list.txt'.format(args.dataset_str)
    train_list = np.genfromtxt(filepath,dtype=float).astype(int)
    filelist = np.array(list(reco2utt_dict.keys()))
    train_filelist = filelist[train_list]
    filegroup_count = args.filegroupcount
    num_graphs_tentative = 2
    
    num_batches = math.ceil((num_graphs_tentative*len(train_filelist))/(filegroup_count*args.batch_size))
    print('num_batches: ',num_batches)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt,
                                                    T_max=args.epochs * num_batches,
                                                    eta_min=1e-5)
    if args.resume>0:
        scheduler.load_state_dict(schedulerdict)
    ###############
    # Training Loop
    for epoch in range(args.epochs):
        if args.resume>epoch:
            continue
        gs = []
        nbrs = []
        ks = []
        counter = 0
        labels_group = []

        # for xf, lf in zip(xvec_files[-10:],lab_files[-10:]):
        loss_den_val_total = []
        loss_conn_val_total = []
        loss_val_total = []
        num_graphs = 0
        batch = 0
        for recid in train_filelist:
            counter += 1
            feats_list = []
            labels = []
            # feats_fname = xf[:-1]
            # labels_fname = lf[:-1]
            feats_fname = f'{args.xvecpath}/{recid}.npy'
            labels_fname = f'{args.labelspath}/labels_{recid}'
            # print(feats_fname,labels_fname)
            features = np.load(feats_fname) 
            # recid = feats_fname.split('/')[-1].split('.')[0]
            print(recid)
            
            reco2utt = reco2utt_dict[recid]
            # segmentation = speech_segments[feats_fname]
            mfcc_feats,idx_xvec = load_mfcc_feats_nosilence(args,recid,featsdict,reco2utt,featsbatch=1)
            clean_ind = []
            with open(labels_fname,"r") as f:
                label_file = f.readlines()
            for i,line in enumerate(label_file):
                label = line.split()
                if len(label) == 2:
                    clean_ind.append(i)
                    feats_list.append(features[i])
                    labels.append(spk_dct[label[1]])
            #pdb.set_trace()
            clean_ind = np.array(clean_ind)
            mfcc_feats = mfcc_feats[clean_ind]
            print('per file mfcc: ',mfcc_feats.shape)
            if counter % filegroup_count == 1 or filegroup_count == 1:
                features_group = np.vstack(feats_list)    
                mfcc_features_group = mfcc_feats.copy()
            else:
                features_group = np.vstack((features_group,feats_list))
                mfcc_features_group = np.vstack((mfcc_features_group,mfcc_feats))
                
            labels_group.extend(labels)
            
            if counter % filegroup_count == 0:
                # bp()
                # mfcc_features = torch.FloatTensor(mfcc_features_group).to(device)
                # org_xvecs = torch.FloatTensor(features_group).to(device)
                print('mfcc: ',mfcc_features_group.shape)
                maxsize = 500
                # if mfcc_features.shape[0]<=maxsize:
                #     xvecs = model.extract_xvecs(mfcc_features,org_xvecs)
                # else:
                
                count = 0
                while count < mfcc_features_group.shape[0]:
                    mfcc_features = torch.FloatTensor(mfcc_features_group[count:count+maxsize]).to(device)
                    org_xvecs = torch.FloatTensor(features_group[count:count+maxsize]).to(device)
                    xvecs= model.extract_xvecs(mfcc_features,org_xvecs)
                    

                    labels_group_sub = np.array(labels_group)[count:count+maxsize]
                    features_group_sub = features_group[count:count+maxsize]
                    print("Create graphs for recording %s"%(recid))
                    #dataset preparation includes creating ground truth graph levels for different level list with different knn
                    for k, l in zip(k_list, lvl_list):
                        dataset = SHARCDataset_withmfcc(mfcc_xvec_features=xvecs, features=features_group_sub, 
                                                        labels=labels_group_sub, mode=mode, k=k,
                                                        levels=l,device=device,feats_norm=args.feats_norm,
                                                        pldamodel=args.pldamodel)
                        
                        gs += [g for g in dataset.gs]
                        ks += [k for g in dataset.gs]
                        nbrs += [nbr for nbr in dataset.nbrs]
                
                    num_graphs = len(gs)
                    print('num_graphs: ',num_graphs)
                    # if (epoch ==0 and counter == filegroup_count and count ==0) or args.resume > 0:
                    #     num_batches = math.ceil((num_graphs*len(train_filelist))/(filegroup_count*args.batch_size))
                    #     print('num_batches: ',num_batches)
                    #     scheduler = optim.lr_scheduler.CosineAnnealingLR(opt,
                    #                                                 T_max=args.epochs * num_batches,
                    # #                                                 eta_min=1e-5)
                    #     if args.resume>0:
                    #         scheduler.load_state_dict(schedulerdict)
                    # bp()
                    if num_graphs >= args.batch_size:
                        # bp()
                        print("Num graphs = %d"%(len(gs)))
                        print('Graphs Prepared.',flush=True)
                        graph_batch = list(range(num_graphs))
                        # random.shuffle(graph_batch)
                        opt.zero_grad()
                        for graph_id in graph_batch:
                            # get the feature for the input_nodes
                            g = gs[graph_id].to(device)
                            
                            output_bipartite = model(g)
                            loss, loss_den_val, loss_conn_val = model.compute_loss(output_bipartite)
                            loss_den_val_total.append(loss_den_val)
                            loss_conn_val_total.append(loss_conn_val)
                            loss_val_total.append(loss.item())
                            time.sleep(1)
                            loss.backward(retain_graph=True)
                            
                            del g
                            if (batch + 1) % 1 == 0:
                                print('epoch: %d, batch: %d / %d, loss: %.6f, loss_den: %.6f, loss_conn: %.6f'%
                                    (epoch, batch, num_batches, loss.item(), loss_den_val, loss_conn_val))

                        opt.step()
                        scheduler.step()
                        gs = []
                        nbrs = []
                        ks = []
                        num_graphs = 0
                        batch = batch + 1
                        del loss
                        del xvecs
                        del dataset
                        torch.cuda.empty_cache()
                    count = count +maxsize
                    # bp()
                labels_group = []
                
        print('############################################################################################')
        print('Final_epoch: %d loss: %.6f loss_den: %.6f loss_conn: %.6f'%
        (epoch, np.array(loss_val_total).mean(),
        np.array(loss_den_val_total).mean(), np.array(loss_conn_val_total).mean()),flush=True)
        print('############################################################################################')
        if (epoch+1) % 2 == 0:
            checkpoints ={}
            checkpoints['model'] = model.state_dict()
            checkpoints['optimizer'] = opt.state_dict()
            checkpoints['scheduler'] = scheduler.state_dict()
            torch.save(model.state_dict(), f'{args.model_savepath}/model_final.pth')
            torch.save(checkpoints,f'{args.model_savepath}/model_{epoch+1}_snapshot.pth')

    checkpoints ={}
    checkpoints['model'] = model.state_dict()
    checkpoints['optimizer'] = opt.state_dict()
    checkpoints['scheduler'] = scheduler.state_dict()
    torch.save(checkpoints,f'{args.model_savepath}/model_final.pth')
    # torch.save(model.state_dict(), args.model_savepath)

def train_ami_modified3(model,opt, schedulerdict=None,milestone=5):
    '''
    one graph is created from 500 frames of one file and then no. of graphs is batch size =2,
    then go to next 500 frames of the same and repeat, make it faster
    '''
    # bp()
    filepath = 'lists/{}/shuffled_list.txt'.format(args.dataset_str)
    train_list = np.genfromtxt(filepath,dtype=float).astype(int)
    filelist = np.array(list(reco2utt_dict.keys()))
    train_filelist = filelist[train_list]
    filegroup_count = args.filegroupcount
    num_graphs_tentative = 2
    
    num_batches = 526
    # num_batches = math.ceil((num_graphs_tentative*len(train_filelist))/(filegroup_count*args.batch_size))
    print('num_batches: ',num_batches)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt,
                                                    T_max=args.epochs * num_batches,
                                                    eta_min=1e-5)
    if args.resume>0:
        scheduler.load_state_dict(schedulerdict)

    labels_dict = {}
    clean_ind_dict = {}
    ###############
    # Training Loop
    for epoch in range(args.epochs):
        if args.resume>epoch:
            continue
        gs = []
        nbrs = []
        ks = []
        counter = 0
        labels_group = []

       
        loss_den_val_total = []
        loss_conn_val_total = []
        loss_val_total = []
        num_graphs = 0
        batch = 0
        for recid in train_filelist:
            counter += 1
            feats_list = []
            labels = []
            
            feats_fname = f'{args.xvecpath}/{recid}.npy'
            labels_fname = f'{args.labelspath}/labels_{recid}'
            # print(feats_fname,labels_fname)
            features = np.load(feats_fname) 
            print(recid)
            
            reco2utt = reco2utt_dict[recid]
            count = 0
            maxsize = 500
            # segmentation = speech_segments[feats_fname]
            while count + maxsize < features.shape[0]:
                if count == 0 and not bool(clean_ind_dict):
                    labels = []
                    labels_fname = f'{args.labelspath}/labels_{recid}'
                    clean_ind = []
                    with open(labels_fname,"r") as f:
                        label_file = f.readlines()
                    for i,line in enumerate(label_file):
                        label = line.split()
                        if len(label) == 2:
                            clean_ind.append(i)
                            labels.append(spk_dct[label[1]])
                  
                    clean_ind = np.array(clean_ind)
                    clean_ind_dict[recid] = clean_ind
                    labels_dict[recid] = labels
                else:
                    clean_ind = clean_ind_dict[recid]
                    labels = labels_dict[recid]

                clean_ind = np.array(clean_ind)
                
                # xvectors
                
                feats_list = features[clean_ind]
                feats_list = feats_list[count:count+maxsize]
                if feats_list.shape[0]< maxsize:
                    # counter = counter - 1
                    continue
                mfcc_feats = load_mfcc_feats_nosilence_sub(args,recid,featsdict,reco2utt,start=count,end=count+maxsize,featsbatch=1,clean_ind=clean_ind)
                    
                print('per file mfcc: ',mfcc_feats.shape)
                if counter % filegroup_count == 1 or filegroup_count == 1:
                    features_group = feats_list.copy()   
                    mfcc_features_group = mfcc_feats.copy()
                else:
                    features_group = np.vstack((features_group,feats_list))
                    mfcc_features_group = np.vstack((mfcc_features_group,mfcc_feats))
                    
                labels_group.extend(labels[count:count+maxsize])
            
                if counter % filegroup_count == 0:

                    print('mfcc: ',mfcc_features_group.shape)
                    
                    mfcc_features = torch.FloatTensor(mfcc_features_group).to(device)
                    org_xvecs = torch.FloatTensor(features_group).to(device)
                    xvecs= model.extract_xvecs(mfcc_features,org_xvecs)
                    
                    labels_group_sub = np.array(labels_group)
                    features_group_sub = features_group
                    print("Create graphs for recording %s"%(recid))
                    #dataset preparation includes creating ground truth graph levels for different level list with different knn
                    for k, l in zip(k_list, lvl_list):
                        dataset = SHARCDataset_withmfcc(mfcc_xvec_features=xvecs, features=features_group_sub, 
                                                        labels=labels_group_sub, mode=mode, k=k,
                                                        levels=l,device=device,feats_norm=args.feats_norm,
                                                        pldamodel=args.pldamodel)
                        
                        gs += [g for g in dataset.gs]
                        ks += [k for g in dataset.gs]
                        nbrs += [nbr for nbr in dataset.nbrs]
                
                    num_graphs = len(gs)
                    print('num_graphs: ',num_graphs)
                    # if (epoch ==0 and counter == filegroup_count and count ==0) or args.resume > 0:
                    #     num_batches = math.ceil((num_graphs*len(train_filelist))/(filegroup_count*args.batch_size))
                    #     print('num_batches: ',num_batches)
                    #     scheduler = optim.lr_scheduler.CosineAnnealingLR(opt,
                    #                                                 T_max=args.epochs * num_batches,
                    # #                                                 eta_min=1e-5)

                    if num_graphs >= args.batch_size:

                        print("Num graphs = %d"%(len(gs)))
                        print('Graphs Prepared.',flush=True)
                        graph_batch = list(range(num_graphs))
                        # random.shuffle(graph_batch)
                        opt.zero_grad()
                        for graph_id in graph_batch:
                            # get the feature for the input_nodes
                            g = gs[graph_id].to(device)
                            
                            output_bipartite = model(g)
                            loss, loss_den_val, loss_conn_val = model.compute_loss(output_bipartite)
                            loss_den_val_total.append(loss_den_val)
                            loss_conn_val_total.append(loss_conn_val)
                            loss_val_total.append(loss.item())
                            time.sleep(1)
                            loss.backward(retain_graph=True)
                            
                            del g
                            if (batch + 1) % 1 == 0:
                                print('epoch: %d, batch: %d / %d, loss: %.6f, loss_den: %.6f, loss_conn: %.6f'%
                                    (epoch, batch, num_batches, loss.item(), loss_den_val, loss_conn_val))

                        opt.step()
                        scheduler.step()
                        gs = []
                        nbrs = []
                        ks = []
                        num_graphs = 0
                        batch = batch + 1
                        del loss
                       
                        torch.cuda.empty_cache()
                    count = count +maxsize
                    # bp()
                labels_group = []
                
        print('############################################################################################')
        print('Final_epoch: %d loss: %.6f loss_den: %.6f loss_conn: %.6f'%
        (epoch, np.array(loss_val_total).mean(),
        np.array(loss_den_val_total).mean(), np.array(loss_conn_val_total).mean()),flush=True)
        print('############################################################################################')
        if (epoch+1) % milestone == 0:
            checkpoints ={}
            checkpoints['model'] = model.state_dict()
            checkpoints['optimizer'] = opt.state_dict()
            checkpoints['scheduler'] = scheduler.state_dict()
            torch.save(model.state_dict(), f'{args.model_savepath}/model_final.pth')
            torch.save(checkpoints,f'{args.model_savepath}/model_{epoch+1}_snapshot.pth')

    checkpoints ={}
    checkpoints['model'] = model.state_dict()
    checkpoints['optimizer'] = opt.state_dict()
    checkpoints['scheduler'] = scheduler.state_dict()
    torch.save(checkpoints,f'{args.model_savepath}/model_final.pth')
   
def val(model,reco2utt_dict):
    epoch = args.epochs
    filepath = 'lists/{}/val.list'.format(args.dataset_str)
    val_list = np.genfromtxt(filepath,dtype=float).astype(int)
    filelist = np.array(list(reco2utt_dict.keys()))
    val_filelist = filelist[val_list]
    val_filelist = val_filelist[:20]
    filegroup_count = 1
    checkpoints = torch.load(args.model_savepath, map_location=device)
    # model.load_state_dict(checkpoints['model'])
    gs = []
    nbrs = []
    ks = []
    counter = 0
    labels_group = []
    
    
    model.eval()
    for recid in val_filelist:
        counter += 1
        feats_list = []
        labels = []
        
        feats_fname = f'{args.xvecpath}/{recid}.npy'
        labels_fname = f'{args.labelspath}/labels_{fid2fname[recid]}'
        # print(feats_fname,labels_fname)
        features = np.load(feats_fname) 
        
        print(recid,features.shape)
        
        reco2utt = reco2utt_dict[recid]
        # segmentation = speech_segments[feats_fname]
        mfcc_feats,idx_xvec = load_mfcc_feats_nosilence(args,recid,featsdict,reco2utt,featsbatch=1)
        clean_ind = []
        with open(labels_fname,"r") as f:
            label_file = f.readlines()
        for i,line in enumerate(label_file):
            label = line.split()
            if len(label) == 2:
                clean_ind.append(i)
                feats_list.append(features[i])
                labels.append(spk_dct[label[1]])
        #pdb.set_trace()
        clean_ind = np.array(clean_ind)
        mfcc_feats = mfcc_feats[clean_ind]
        
        if counter % filegroup_count == 1 or filegroup_count == 1:
            features_group = np.vstack(feats_list)    
            mfcc_features_group = mfcc_feats.copy()
        else:
            features_group = np.vstack((features_group,feats_list))
            mfcc_features_group = np.vstack((mfcc_features_group,mfcc_feats))
            
        labels_group.extend(labels)
        
        if counter % filegroup_count == 0:
            
            mfcc_features = torch.FloatTensor(mfcc_features_group).to(device)
            org_xvecs = torch.FloatTensor(features_group).to(device)
            print('mfcc: ',mfcc_features.shape)
            xvecs = model.extract_xvecs(mfcc_features,org_xvecs)
            labels_group = np.array(labels_group)
            print("Create graphs for recording %s"%(recid))
        #dataset preparation includes creating ground truth graph levels for different level list with different knn
            for k, l in zip(k_list, lvl_list):
                dataset = SHARCDataset_withmfcc(mfcc_xvec_features=xvecs, features=features_group, 
                                                labels=labels_group, mode=mode, k=k,
                                                levels=l,device=device)
                
                gs += [g for g in dataset.gs]
                ks += [k for g in dataset.gs]
                nbrs += [nbr for nbr in dataset.nbrs]
            labels_group = []
    print("Num graphs = %d"%(len(gs)))
    print('Graphs Prepared.',flush=True)
    num_graphs = len(gs)
    loss_den_val_total = []
    loss_conn_val_total = []
    loss_val_total = []
    graph_batch = list(range(num_graphs))
    for graph_id in graph_batch:
        # get the feature for the input_nodes
        g = gs[graph_id].to(device)
        
        output_bipartite = model(g)
        loss, loss_den_val, loss_conn_val = model.compute_loss(output_bipartite)
        loss_den_val_total.append(loss_den_val)
        loss_conn_val_total.append(loss_conn_val)
        loss_val_total.append(loss.item())
        del g
        torch.cuda.empty_cache()
    print('############################################################################################')
    print('epoch: %d val_loss: %.6f val_loss_den: %.6f val_loss_conn: %.6f'%
    (epoch, np.array(loss_val_total).mean(),
    np.array(loss_den_val_total).mean(), np.array(loss_conn_val_total).mean()),flush=True)
    print('############################################################################################')

def val_orgxvec(model,reco2utt_dict):
    
    epoch = args.epochs
    filepath = 'lists/{}/val.list'.format(args.dataset_str)
    val_list = np.genfromtxt(filepath,dtype=float).astype(int)
    filelist = np.array(list(reco2utt_dict.keys()))
    val_filelist = filelist[val_list]
    filegroup_count = 1
    checkpoints = torch.load(args.model_savepath, map_location=device)
    # model.load_state_dict(checkpoints['model'])
    gs = []
    nbrs = []
    ks = []
    counter = 0
    labels_group = []
    
    # for xf, lf in zip(xvec_files[-10:],lab_files[-10:]):
    model.eval()
    for recid in val_filelist[:10]:
        counter += 1
        feats_list = []
        labels = []
        # feats_fname = xf[:-1]
        # labels_fname = lf[:-1]
        feats_fname = f'{args.xvecpath}/{recid}.npy'
        labels_fname = f'{args.labelspath}/labels_{fid2fname[recid]}'
        # print(feats_fname,labels_fname)
        features = np.load(feats_fname) 
        # recid = feats_fname.split('/')[-1].split('.')[0]
        print(recid,features.shape)
        
        reco2utt = reco2utt_dict[recid]
        # segmentation = speech_segments[feats_fname]
        
        clean_ind = []
        with open(labels_fname,"r") as f:
            label_file = f.readlines()
        for i,line in enumerate(label_file):
            label = line.split()
            if len(label) == 2:
                clean_ind.append(i)
                feats_list.append(features[i])
                labels.append(spk_dct[label[1]])
        #pdb.set_trace()
        clean_ind = np.array(clean_ind)

        
        if counter % filegroup_count == 1 or filegroup_count == 1:
            features_group = np.vstack(feats_list)    
        else:
            features_group = np.vstack((features_group,feats_list))
        labels_group.extend(labels)
        
        if counter % filegroup_count == 0:
            org_xvecs = torch.FloatTensor(features_group).to(device)
            labels_group = np.array(labels_group)
            print("Create graphs for recording %s"%(recid))
        #dataset preparation includes creating ground truth graph levels for different level list with different knn
            for k, l in zip(k_list, lvl_list):
                dataset = LanderDataset(features=features_group, 
                                        labels=labels_group, mode=mode, k=k,
                                        levels=l)
                # dataset = SHARCDataset_withmfcc(mfcc_xvec_features=org_xvecs, features=features_group, 
                #                                 labels=labels_group, mode=mode, k=k,
                #                                 levels=l,device=device)
                
                gs += [g for g in dataset.gs]
                ks += [k for g in dataset.gs]
                nbrs += [nbr for nbr in dataset.nbrs]
            labels_group = []
    print("Num graphs = %d"%(len(gs)))
    print('Graphs Prepared.',flush=True)
    num_graphs = len(gs)
    loss_den_val_total = []
    loss_conn_val_total = []
    loss_val_total = []
    graph_batch = list(range(num_graphs))
    for graph_id in graph_batch:
        # get the feature for the input_nodes
        g = gs[graph_id].to(device)
        
        output_bipartite = model(g)
        loss, loss_den_val, loss_conn_val = model.compute_loss(output_bipartite)
        loss_den_val_total.append(loss_den_val)
        loss_conn_val_total.append(loss_conn_val)
        loss_val_total.append(loss.item())
        del g
        torch.cuda.empty_cache()
    print('############################################################################################')
    print('epoch: %d val_loss: %.6f val_loss_den: %.6f val_loss_conn: %.6f'%
    (epoch, np.array(loss_val_total).mean(),
    np.array(loss_den_val_total).mean(), np.array(loss_conn_val_total).mean()),flush=True)
    print('############################################################################################')


def main():
    print('main')

##########################################################################################
# Model Definition
feature_dim = args.xvec_dim
torch.cuda.manual_seed(940105)

model = main_model(xvecmodelpath=args.xvec_model_weight_path,
               model_filename = args.model_filename,
               feature_dim=feature_dim, nhid=args.hidden,
               num_conv=args.num_conv, dropout=args.dropout,
               use_GAT=args.gat, K=args.gat_k,
               balance=args.balance,
               use_cluster_feat=args.use_cluster_feat,
               use_focal_loss=args.use_focal_loss,
               fulltrain=args.fulltrain,
               xvecbeta=args.xvecbeta,withcheckpoint=args.withcheckpoint)
if args.resume>0:
    checkpoints = torch.load(f'{args.model_savepath}_{args.resume+1}_snapshot.pth', map_location=device)
    model.load_state_dict(checkpoints['model'])
model = model.to(device)


model.train1()

#################
# Hyperparameters
milestone = 5 # saving model at every milestone
if not args.samelr:
    opt = optim.SGD([
                        {'params': model.mylander.parameters()},
                        {'params' : model.e2e_xvecs.parameters(),'lr':args.lr/1000}
                    ], lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)

else:
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                weight_decay=args.weight_decay)
    milestone = 2
if args.resume>0:
    opt.load_state_dict(checkpoints['optimizer'])
    
# keep num_batch_per_loader the same for every sub_dataloader
#num_batch_per_loader = len(train_loaders[0])
#train_loaders = [iter(train_loader) for train_loader in train_loaders]
#num_loaders = len(train_loaders)

process = psutil.Process(os.getpid())
print(process.memory_info().rss)
#pdb.set_trace()
if args.mymode=='train':
    print('Start Training.',flush=True)
    if args.dataset_str == 'ami_sdm_train' or args.dataset_str == 'ami_sdm_train_mdm':
        if args.modified == 3:
            train_ami_modified3(model,opt,milestone=milestone)
        else:
            train_ami(model,opt)
    else:
        if args.resume>0:
            train(model,opt,checkpoints['scheduler'])
        else:
            train(model,opt)
        
elif args.mymode=='val':
    val(model,reco2utt_dict)
elif args.mymode=='val_orgxvec':
    model = LANDER(feature_dim=feature_dim, nhid=args.hidden,
               num_conv=args.num_conv, dropout=args.dropout,
               use_GAT=args.gat, K=args.gat_k,
               balance=args.balance,
               use_cluster_feat=args.use_cluster_feat,
               use_focal_loss=args.use_focal_loss)
    if args.model_filename is not None:
            model.load_state_dict(torch.load(args.model_filename, map_location=device))
    model = model.to(device)
    val_orgxvec(model,reco2utt_dict)

