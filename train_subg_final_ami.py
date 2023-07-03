import argparse, time, os, pickle
from calendar import c
import numpy as np
import math
import random
import subprocess
import os, psutil

import dgl
import torch
import torch.optim as optim
from pdb import set_trace as bp
from models_final import SHARC
from dataset_final import SHARCDataset

import sys
from services.kaldi_io import read_vec_flt
from torch.optim.swa_utils import AveragedModel, SWALR

from torch.utils.tensorboard import SummaryWriter
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

    #paths
    parser.add_argument('--filegroupcount',type=int,default=10)
    parser.add_argument('--xvec_dim',type=int,default=512)
    parser.add_argument('--file_pairs', type=str, default=None)
    parser.add_argument('--xvecpath', type=str, default=None)
    parser.add_argument('--labelspath', type=str, default=None)
    parser.add_argument('--reco2utt_list', type=str, default=None)
    parser.add_argument('--dataset_str', type=str, default=None)
    parser.add_argument('--feats_norm',type=int,default=0,help='Normalize features before GNN')
    parser.add_argument('--model_filename_init', type=str, default=None,help='Initialize the SHARC model')
    parser.add_argument('--model_savepath', type=str, default=None)
    parser.add_argument('--ngpu',type=int)
    parser.add_argument('--xvecdim',type=int,default=512)
    parser.add_argument('--pldamodel',type=str,default=None)
    parser.add_argument('--loss_weight',type=float,default=0.5)
    parser.add_argument('--fixfilecount', action='store_true')
    parser.add_argument('--labelspath_test', type=str, default=None)
    parser.add_argument('--reco2utt_list_test', type=str, default=None)
    parser.add_argument('--xvecpath_test', type=str, default=None)
    parser.add_argument('--knn_k_val', type=str, default='60')
    parser.add_argument('--overlap', action='store_true')
    parser.add_argument('--isswa',type=str,default=None,help='stochastic weighted averaging')
    parser.add_argument('--ldatransform',type=str,default=None)

    # different lander architecture
    parser.add_argument('--lander_type',type=str,default=None)
    parser.add_argument('--temp_param', type=str, default='5,0.95') # temporal continuity parameters, neb, beta1
    
    parser.add_argument('--outpath',type=str,default=None)
    args = parser.parse_args()  
    return args
args = arguments()
# print(args)

def cuda_gpu_available():
    try:
        gpu_available_info = subprocess.Popen('/state/partition1/softwares/Kaldi_Jan_2023/kaldi/src/nnet3bin/cuda-gpu-available', stderr=subprocess.PIPE)
        gpu_available_info = gpu_available_info.stderr.read().decode('utf-8')
        Node = gpu_available_info[86:97]
        gpu = int(gpu_available_info[gpu_available_info.index('active GPU is') + 15])
    except:
        return False
    print("Successfully selected the free GPU {} in {}.".format(gpu, Node))
    return gpu

#######################
writer = SummaryWriter(f'{args.outpath}/')
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


mode = args.mode.split(',')

k_list = [int(k) for k in args.knn_k.split(',')]
lvl_list = [int(l) for l in args.levels.split(',')]
    
def prepare_dataset_test(set='test'):
    k_list_val = [int(k) for k in args.knn_k_val.split(',')]
    
    # prefix = '/data1/prachis/SRE_19/Self_supervised_clustering/tools_diar'
    prefix = ''
    with open("uniq_spkr_list","r") as f:
        spk_ids = f.readlines()
    spk_dct = {}
    for i,spk_id in enumerate(spk_ids):
        spk_dct[spk_id[:-1]] = i
        
    reco2utt_list_test = open(args.reco2utt_list_test).readlines()
    reco2utt_dict_test = {}
    for line in reco2utt_list_test:
        rec, utt = line.split(" ",1)
        reco2utt_dict_test[rec] = utt
    
    
    featsdict = {}
    with open(args.xvecpath_test) as fpath:
        for line in fpath: 
            key, value = line.split(" ",1)
            featsdict[key] = value.rsplit()[0]
    # bp()
    train_filelist = np.array(list(reco2utt_dict_test.keys()))
    # train_filelist = train_filelist[:1]
    filegroup_count = 1
    gs = []
    nbrs = []
    ks = []
    counter = 0
    labels_group = []
   
    for recid in train_filelist:
        counter += 1
        feats_list = []
        labels = []
        labels_fname = f'{args.labelspath_test}/labels_{recid}'
        # labels = [int(labels[i].split()[-1]) for i in range(len(labels))]
        
        with open(labels_fname,"r") as f:
            label_file = f.readlines()
        for i,line in enumerate(label_file):
            label = line.split()
            if not args.overlap:
                if len(label) == 2:
                    mykey = label[0] #'_'.join(label)
                    features = read_vec_flt(f'{prefix}/{featsdict[mykey]}')
                    feats_list.append(features)
                    labels.append(int(label[1]))
            else:
                if len(label) <= 3 :
                    mykey = label[0] #'_'.join(label)
                    features = read_vec_flt(f'{prefix}/{featsdict[mykey]}')
                    feats_list.append(features)
                    labels.append(int(label[1]))
        #pdb.set_trace()
        if counter % filegroup_count == 1 or filegroup_count == 1:
            features_group = np.vstack(feats_list)     
        else:
            features_group = np.vstack((features_group,feats_list))
        labels_group.extend(labels)
        
        if counter % filegroup_count == 0:
            labels_group = np.array(labels_group)
            print("Create graphs for recording %s"%(recid))
         
        #dataset preparation includes creating ground truth graph levels for different level list with different knn
            for k, l in zip(k_list_val, lvl_list):
                dataset = SHARCDataset(features=features_group, labels=labels_group, mode=mode, k=k,
                                            levels=l,pldamodel=args.pldamodel,feats_norm=args.feats_norm,ldatransform=args.ldatransform,
                                            temp_param=args.temp_param)
                gs += [g for g in dataset.gs]
                ks += [k for g in dataset.gs]
                nbrs += [nbr for nbr in dataset.nbrs]
            labels_group = []
           
            counter = 0

    print("Num graphs = %d"%(len(gs)))
    print('Dataset Prepared.',flush=True)
    return gs



def prepare_dataset(set='train',prefix='/data1/prachis/Dihard_2020/gae-pytorch/gae/tools_diar/',spk_ids_list="data/ami_train/labels_train/uniq_spk_ids"):
    
    with open(spk_ids_list,"r") as f:
        spk_ids = f.readlines()

    spk_dct = {}
    for i,spk_id in enumerate(spk_ids):
        spk_dct[spk_id[:-1]] = i


    featsdict = {}
    with open(args.xvecpath) as fpath:
        for line in fpath: 
            key, value = line.split(" ",1)
            featsdict[key] = value.rsplit()[0]

    reco2utt_list = open(args.reco2utt_list).readlines()
    reco2utt_dict = {}
    for line in reco2utt_list:
        rec, utt = line.split(" ",1)
        reco2utt_dict[rec] = utt
    
    if set=='train':
        filepath = 'lists/{}/shuffled_list.txt'.format(args.dataset_str)
    elif set=='val':
        filepath = 'lists/{}/{}.list'.format(args.dataset_str,set)
    train_list = np.genfromtxt(filepath,dtype=float).astype(int)
    
    # filelist = np.array(list(reco2utt_dict.keys()))
    filelist = np.genfromtxt(f'lists/{args.dataset_str}/{args.dataset_str}.list',dtype=str)
    train_filelist = filelist[train_list]
    # train_filelist = train_filelist[45:] # for checking
    
    filegroup_count = args.filegroupcount
    gs = []
    nbrs = []
    ks = []
    counter = 0
    labels_group = []

    for recid in train_filelist:
        counter += 1
        feats_list = []
        labels = []
        labels_fname = f'{args.labelspath}/labels_{recid}'
        with open(labels_fname,"r") as f:
            label_file = f.readlines()
        for i,line in enumerate(label_file):
            label = line.split()
            if not args.overlap:
                if len(label) == 2 :
                    mykey = label[0] 
                    features = read_vec_flt(f'{prefix}/{featsdict[mykey]}') # check which features are empty
                    if len(features) ==0:
                        bp()
                    feats_list.append(features)
                    labels.append(spk_dct[label[1]])
            else:
                if len(label) <= 3 :
                    mykey = label[0] 
                    features = read_vec_flt(f'{args.xvecpath.split("exp/")[0]}/{featsdict[mykey]}')
                    feats_list.append(features)
                    labels.append(spk_dct[label[1]])
  
        print("Create graphs for recording %s"%(recid))
        if counter % filegroup_count == 1 or filegroup_count == 1:
            features_group = np.vstack(feats_list) 
        else:
            features_group = np.vstack((features_group,feats_list))
        labels_group.extend(labels)
        
        if counter % filegroup_count == 0:
            labels_group = np.array(labels_group)
            print("Create graphs for recording %s"%(recid))
         
        #dataset preparation includes creating ground truth graph levels for different level list with different knn
            for k, l in zip(k_list, lvl_list):
                dataset = SHARCDataset(features=features_group, labels=labels_group, mode=mode, k=k,
                                            levels=l,pldamodel=args.pldamodel,feats_norm=args.feats_norm,ldatransform=args.ldatransform,
                                            temp_param=args.temp_param
                                            )
                
                gs += [g for g in dataset.gs]
                ks += [k for g in dataset.gs]
                nbrs += [nbr for nbr in dataset.nbrs]
            labels_group = []
            if not args.fixfilecount:
                filegroup_count = np.random.randint(1,args.filegroupcount)
            counter = 0

    print("Num graphs = %d"%(len(gs)))
    print('Dataset Prepared.',flush=True)
    return gs



def val(epoch,model):
    model.eval()
    loss_den_val_total = []
    loss_conn_val_total = []
    loss_val_total = []
    # random.shuffle(graph_ids_val)
    for batch in range(num_batches_val):
        opt.zero_grad()
        try:
            graph_batch = graph_ids_val[batch*args.batch_size:(batch+1)*args.batch_size]
        except:
            graph_batch = graph_ids_val[batch*args.batch_size:]

        for graph_id in graph_batch:
            
            # get the feature for the input_nodes
            g = gs_val[graph_id].to(device)
            output_bipartite = model(g)
            loss, loss_den_val, loss_conn_val = model.compute_loss(output_bipartite)
            loss_den_val_total.append(loss_den_val)
            loss_conn_val_total.append(loss_conn_val)
            loss_val_total.append(loss.item())
    print('Final_epoch: %d val_loss: %.6f val_loss_den: %.6f val_loss_conn: %.6f'%
            (epoch, np.array(loss_val_total).mean(),
            np.array(loss_den_val_total).mean(), np.array(loss_conn_val_total).mean()),flush=True)

    writer.add_scalar('Val loss',
                            np.array(loss_val_total).mean(),
                            epoch)
    writer.add_scalar('Val loss_den',
                            np.array(loss_den_val_total).mean(),epoch)

    writer.add_scalar('Val loss_conn',
                            np.array(loss_conn_val_total).mean(),
                            epoch)
    # sys.stdout.flush()
    
def train(graph_ids):
    ###############
    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        loss_den_val_total = []
        loss_conn_val_total = []
        loss_val_total = []
        random.shuffle(graph_ids)
        for batch in range(num_batches):
            opt.zero_grad()
            try:
                graph_batch = graph_ids[batch*args.batch_size:(batch+1)*args.batch_size]
            except:
                graph_batch = graph_ids[batch*args.batch_size:]

            for graph_id in graph_batch:
                
                # get the feature for the input_nodes
                g = gs[graph_id].to(device)
                output_bipartite = model(g)
                loss, loss_den_val, loss_conn_val = model.compute_loss(output_bipartite)
                loss_den_val_total.append(loss_den_val)
                loss_conn_val_total.append(loss_conn_val)
                loss_val_total.append(loss.item())
                loss.backward()
                del g
                torch.cuda.empty_cache()
                if (batch + 1) % 10 == 0:
                    print('epoch: %d, batch: %d / %d, loss: %.6f, loss_den: %.6f, loss_conn: %.6f'%
                        (epoch, batch, num_batches, loss.item(), loss_den_val, loss_conn_val))
                    # sys.stdout.flush()
            
            opt.step()

            torch.cuda.empty_cache()
            scheduler.step()
        
        print('Final_epoch: %d loss: %.6f loss_den: %.6f loss_conn: %.6f'%
            (epoch, np.array(loss_val_total).mean(),
            np.array(loss_den_val_total).mean(), np.array(loss_conn_val_total).mean()),flush=True)
        # sys.stdout.flush()
        writer.add_scalar('training loss',
                            np.array(loss_val_total).mean(),
                            epoch)
        writer.add_scalar('training loss_den',
                            np.array(loss_den_val_total).mean(),
                            epoch)
        writer.add_scalar('training loss_conn',
                            np.array(loss_conn_val_total).mean(),
                            epoch)
        val(epoch,model)
        checkpoints ={}
        checkpoints['model'] = model.state_dict()
        checkpoints['optimizer'] = opt.state_dict()
        checkpoints['scheduler'] = scheduler.state_dict()
        torch.save(checkpoints, f'{args.model_savepath}/model_final.pth')
        if (epoch+1) % 20 == 0:
            
            torch.save(checkpoints,f'{args.model_savepath}/model_{epoch+1}_snapshot.pth')
        
    checkpoints ={}
    checkpoints['model'] = model.state_dict()
    checkpoints['optimizer'] = opt.state_dict()
    checkpoints['scheduler'] = scheduler.state_dict()
    torch.save(checkpoints, f'{args.model_savepath}/model_final.pth')


def train_swa1(graph_ids):
    ###############
    # Training Loop
    swa_start = 50
    swa_scheduler = SWALR(opt, swa_lr=args.lr/10)
    swa_model = AveragedModel(model)

    for epoch in range(args.epochs):
        model.train()
        loss_den_val_total = []
        loss_conn_val_total = []
        loss_val_total = []
        random.shuffle(graph_ids)
        for batch in range(num_batches):
            opt.zero_grad()
            try:
                graph_batch = graph_ids[batch*args.batch_size:(batch+1)*args.batch_size]
            except:
                graph_batch = graph_ids[batch*args.batch_size:]

            for graph_id in graph_batch:
                
                # get the feature for the input_nodes
                g = gs[graph_id].to(device)
                output_bipartite = model(g)
                loss, loss_den_val, loss_conn_val = model.compute_loss(output_bipartite)
                loss_den_val_total.append(loss_den_val)
                loss_conn_val_total.append(loss_conn_val)
                loss_val_total.append(loss.item())
                loss.backward()
                del g
                torch.cuda.empty_cache()
                if (batch + 1) % 10 == 0:
                    print('epoch: %d, batch: %d / %d, loss: %.6f, loss_den: %.6f, loss_conn: %.6f'%
                        (epoch, batch, num_batches, loss.item(), loss_den_val, loss_conn_val))
                    # sys.stdout.flush()
            opt.step()
            if epoch > swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                scheduler.step()
                torch.cuda.empty_cache()
       
        print('Final_epoch: %d loss: %.6f loss_den: %.6f loss_conn: %.6f'%
            (epoch, np.array(loss_val_total).mean(),
            np.array(loss_den_val_total).mean(), np.array(loss_conn_val_total).mean()),flush=True)
        # sys.stdout.flush()
        val(epoch,model)
        checkpoints ={}
        checkpoints['model'] = model.state_dict()
        checkpoints['optimizer'] = opt.state_dict()
        checkpoints['scheduler'] = scheduler.state_dict()
        torch.save(checkpoints, f'{args.model_savepath}/model_final.pth')
        if (epoch+1) % 10 == 0:
            checkpoints['swa_model'] = swa_model.state_dict()
            torch.save(checkpoints,f'{args.model_savepath}/model_{epoch+1}_snapshot.pth')
        
    checkpoints ={}
    checkpoints['model'] = model.state_dict()
    checkpoints['swa_model'] = swa_model.state_dict()
    checkpoints['optimizer'] = opt.state_dict()
    checkpoints['scheduler'] = scheduler.state_dict()
    torch.save(checkpoints, f'{args.model_savepath}/model_final.pth')


def train_swa2(graph_ids):
    ###############
    # Training Loop
    # bp()
    swa_start = 50
    import torchcontrib
    swa_opt = torchcontrib.optim.SWA(opt, swa_start=swa_start, swa_freq=5, swa_lr=args.lr/10)
    
    for epoch in range(args.epochs):
        model.train()
        loss_den_val_total = []
        loss_conn_val_total = []
        loss_val_total = []
        random.shuffle(graph_ids)
        for batch in range(num_batches):
            swa_opt.zero_grad()
            try:
                graph_batch = graph_ids[batch*args.batch_size:(batch+1)*args.batch_size]
            except:
                graph_batch = graph_ids[batch*args.batch_size:]

            for graph_id in graph_batch:
                
                # get the feature for the input_nodes
                g = gs[graph_id].to(device)
                output_bipartite = model(g)
                loss, loss_den_val, loss_conn_val = model.compute_loss(output_bipartite)
                loss_den_val_total.append(loss_den_val)
                loss_conn_val_total.append(loss_conn_val)
                loss_val_total.append(loss.item())
                loss.backward()
                del g
                torch.cuda.empty_cache()
                if (batch + 1) % 10 == 0:
                    print('epoch: %d, batch: %d / %d, loss: %.6f, loss_den: %.6f, loss_conn: %.6f'%
                        (epoch, batch, num_batches, loss.item(), loss_den_val, loss_conn_val))
                    # sys.stdout.flush()
            swa_opt.step()
            torch.cuda.empty_cache()
            scheduler.step()
        if epoch >= swa_start:
            swa_opt.swap_swa_sgd()
        print('Final_epoch: %d loss: %.6f loss_den: %.6f loss_conn: %.6f'%
            (epoch, np.array(loss_val_total).mean(),
            np.array(loss_den_val_total).mean(), np.array(loss_conn_val_total).mean()),flush=True)
        # sys.stdout.flush()
        val(epoch,model)
        checkpoints ={}
        checkpoints['model'] = model.state_dict()
        checkpoints['optimizer'] = swa_opt.state_dict()
        checkpoints['scheduler'] = scheduler.state_dict()
        torch.save(checkpoints, f'{args.model_savepath}/model_final.pth')
        if (epoch+1) % 10 == 0:
            
            torch.save(checkpoints,f'{args.model_savepath}/model_{epoch+1}_snapshot.pth')
        
    checkpoints ={}
    checkpoints['model'] = model.state_dict()
    checkpoints['optimizer'] = swa_opt.state_dict()
    checkpoints['scheduler'] = scheduler.state_dict()
    torch.save(checkpoints, f'{args.model_savepath}/model_final.pth')


def main():
    print('main')
##################
# Model Definition
feature_dim = args.xvecdim
torch.cuda.manual_seed(940105)

model = SHARC(feature_dim=feature_dim, nhid=args.hidden,
            num_conv=args.num_conv, dropout=args.dropout,
            use_GAT=args.gat, K=args.gat_k,
            balance=args.balance,
            use_cluster_feat=args.use_cluster_feat,
            use_focal_loss=args.use_focal_loss,beta=args.loss_weight,
            )

model = model.to(device)
#################
# Hyperparameters
opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                weight_decay=args.weight_decay)

# keep num_batch_per_loader the same for every sub_dataloader
#num_batch_per_loader = len(train_loaders[0])
#train_loaders = [iter(train_loader) for train_loader in train_loaders]
#num_loaders = len(train_loaders)
if 'displace' in args.dataset_str:
    spk_ids_list =  f'lists/{args.dataset_str}/uniq_spk_ids'
    gs = prepare_dataset(set='train',prefix='',spk_ids_list=spk_ids_list)
else:
    gs = prepare_dataset(set='train')
num_graphs = len(gs)
num_batches = math.ceil(num_graphs/args.batch_size)
graph_ids = list(range(num_graphs))
scheduler = optim.lr_scheduler.CosineAnnealingLR(opt,
                                                 T_max=args.epochs * num_batches,
                                                 eta_min=1e-5)

if args.model_filename_init is not None:
    checkpoints = torch.load(args.model_filename_init, map_location=device)
    model.load_state_dict(checkpoints['model'])
    opt.load_state_dict(checkpoints['optimizer'])
    scheduler.load_state_dict(checkpoints['scheduler'])


model.train()

if args.xvecpath_test is not None:
     gs_val = prepare_dataset_test(set='test')
else:
    if 'displace' in args.dataset_str:
        gs_val = prepare_dataset(set='val',prefix='',spk_ids_list=spk_ids_list)
    else:
        gs_val = prepare_dataset(set='val')
num_graphs_val = len(gs_val)
num_batches_val = math.ceil(num_graphs_val/args.batch_size)
graph_ids_val = list(range(num_graphs_val))

print('Start Training.',flush=True)
process = psutil.Process(os.getpid())
print(process.memory_info().rss)

if args.isswa is None or args.isswa=='None': # default
    train(graph_ids)
elif args.isswa == '_swa1': # weighted averaging for regularization
    train_swa1(graph_ids)
elif args.isswa == '_swa2':
    train_swa2(graph_ids)
    
#pdb.set_trace()
