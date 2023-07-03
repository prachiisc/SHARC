import argparse, time, os, pickle
import numpy as np

import dgl
import torch
import torch.optim as optim
import pdb
from matplotlib import pyplot as plt

from models_final import SHARC, main_model
from dataset_final import SHARCDataset

# from utils_final import evaluation, decode, build_next_level, stop_iterating, get_cosine_mat, get_PLDA_mat, l2norm  
from utils_final import *
from utils_train import *
from torch.optim.swa_utils import AveragedModel, SWALR

from collections import defaultdict
import subprocess

import sys
import warnings
warnings.filterwarnings("ignore")
###########
# ArgParser
def arguments():
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--model_filename', type=str, default=None)
    parser.add_argument('--faiss_gpu', action='store_true')
    parser.add_argument('--PLDA', action='store_true')
    parser.add_argument('--num_workers', type=int, default=0)

    # HyperParam
    parser.add_argument('--knn_k', type=int, default=10)
    parser.add_argument('--levels', type=int, default=1)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--threshold', type=str, default='prob')
    parser.add_argument('--metrics', type=str, default='pairwise,bcubed,nmi')
    parser.add_argument('--early_stop', action='store_true')

    # Model
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--num_conv', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--gat', action='store_true')
    parser.add_argument('--gat_k', type=int, default=1)
    parser.add_argument('--balance', action='store_true')
    parser.add_argument('--use_cluster_feat', action='store_true')
    parser.add_argument('--use_focal_loss', action='store_true')
    parser.add_argument('--use_gt', action='store_true')

    #Xvector 
    parser.add_argument('--reco2utt_list', type=str, default=None)
    parser.add_argument('--segments_list', type=str, default=None)
    parser.add_argument('--dataset_str', type=str, default=None)
    parser.add_argument('--xvec_dim',type=int,default=512)
    parser.add_argument('--file_pairs', type=str, default=None)
    parser.add_argument('--xvecpath', type=str, default=None)
    parser.add_argument('--labelspath', type=str, default=None)
    parser.add_argument('--feats_file',type=str,default=None)
    parser.add_argument('--feats_norm',type=int,default=0,help='Normalize features before GNN')
    parser.add_argument('--pldamodel', type=str, default=None)
    parser.add_argument('--withcheckpoint', action='store_true')
    # for parallel processing
    parser.add_argument('--splitlist',type=str,default=None)
    # Subgraph
    parser.add_argument('--batch_size', type=int, default=4096)
    
    
    # generate rttms 
    parser.add_argument('--segments',type=str,default=None)
    parser.add_argument('--rttm_ground_path',type=str,default=None)
    parser.add_argument('--which_python',type=str,default='python')
    parser.add_argument('--ldatransform',type=str,default=None)
    parser.add_argument('--temp_param', type=str, default='5,0.95') # temporal continuity parameters, neb, beta1
    parser.add_argument('--cluster_features', action='store_true')
    args = parser.parse_args()
    return args

args = arguments()
# print(args,flush=True)

###########################
# Environment Configuration
#if torch.cuda.is_available():
#    device = torch.device('cuda')
#else:
device = torch.device('cpu')

def write_results_dict(fname, output_file, results_dict, reco2utt):
        """Writes the results in label file"""
        f = fname
        output_label = open(output_file+'/'+f+'.labels','w')
        
        hypothesis = results_dict[f]
        meeting_name = f
        reco = fname
        utts = reco2utt.rstrip().split()
        if reco == meeting_name:
            for j,utt in enumerate(utts):
                towrite = utt +' '+str(hypothesis[j])+'\n'
                output_label.writelines(towrite)
        output_label.close()

        rttm_channel=1
        segmentsfile = args.segments+'/'+f+'.segments'
        python = args.which_python
        # python = '/home/prachis/miniconda3/envs/mytorch/bin/python'
        kaldi_recipe_path="./"
        cmd = '{} {}/diarization/make_rttm.py --rttm-channel  {} {} {}/{}.labels {}/{}.rttm' .format(python,kaldi_recipe_path,rttm_channel, segmentsfile,output_file,f,output_file,f)        
        os.system(cmd)

def compute_score(rttm_gndfile,rttm_newfile,outpath,overlap):
      fold_local='services/'
      scorecode='score.py -r '

      if not overlap:
          cmd = '{} {}/dscore-master/{}{} --ignore_overlaps --collar 0.25 -s {} > {}.txt 2> err.txt'.format(args.which_python,fold_local,scorecode,rttm_gndfile,rttm_newfile,outpath)
          # cmd=args.which_python +' '+ fold_local + 'dscore-master/' + scorecode + rttm_gndfile + ' --ignore_overlaps --collar 0.25 -s ' + rttm_newfile + ' > ' + outpath + '.txt'
          os.system(cmd)
          bashCommand="cat {}.txt | grep OVERALL |awk '{{print $4}}'".format(outpath)
      else:
          cmd = '{} {}/dscore-master/{}{} -s {} > {}_overlap.txt 2> err.txt'.format(args.which_python,fold_local,scorecode,rttm_gndfile,rttm_newfile,outpath)
          # cmd=args.which_python + ' '+ fold_local + 'dscore-master/' + scorecode + rttm_gndfile + ' -s ' + rttm_newfile + ' > ' + outpath + '.txt'
          os.system(cmd)
          bashCommand="cat {}_overlap.txt | grep OVERALL |awk '{{print $4}}'".format(outpath)
      output=subprocess.check_output(bashCommand,shell=True)
      return float(output.decode('utf-8').rstrip())

def get_features(feats_fname,reco2utt):

    if 'vox' in args.dataset_str :
        prefix = '/data1/prachis/Dihard_2020/gae-pytorch/gae/tools_diar/'
    else:
        prefix = ''
    
    featsdict = {}
    with open(feats_fname) as fpath:
        for line in fpath: 
            key, value = line.split(" ",1)
          
            featsdict[key] = value.rsplit()[0]
    
    utts = reco2utt.rstrip().split()
    feats_list = []

    for j,utt in enumerate(utts):
        
        features = read_vec_flt(f'{prefix}{featsdict[utt]}')
        feats_list.append(features)
    
    features = np.array(feats_list)

    return features

def get_labels(labels_fname,Nfeats):
    
    try:
        fptr = open(labels_fname,'r')
        labels = fptr.readlines()
        labels_all = labels.copy()
        fptr.close()
        labels_list = []
        spks_count = []
        # labels_all_list = []
        if 'ami' in args.dataset_str and 'train' not in args.dataset_str :
            for i in range(len(labels)):
                labels_list.append(int(labels[i].split()[-1]))
                spks_count.append(len(labels_all[i].split()[1:]))
                # labels_all_list.append(np.array(labels_all[i].split()[1:]).astype(int))
        else:
            for i in range(len(labels)):
                labels_list.append(spk_dct[labels[i].split()[-1]])
                spks_count.append(len(labels_all[i].split()[1:]))
                # labels_all_list.append(np.array(labels_all[i].split()[1:]) for i in range(len(labels)))
        
        labels = labels_list
        # labels_all = labels_all_list
        # spks_count = [len(spk) for spk in labels_all]
        spks_count = np.array(spks_count)
        overlap_ind = np.where(spks_count>1)[0]
        clean_ind = np.where(spks_count==1)[0]
        labels = np.array(labels)
    except:
        
        labels = np.zeros((Nfeats,),dtype=int) #dummy
        return labels,[],[]
    
    return labels, overlap_ind, clean_ind

def add_temporal_weight(N,edges):
    col,row=edges
    neb,beta1 = args.temp_param.split(',')
    neb=int(neb)
    beta1=float(beta1)
    toep = np.abs(np.arange(N).reshape(N,1)-np.arange(N).reshape(1,N))
    toep[toep>neb] = neb
    weighting = beta1**(toep)
    return weighting[row,col]

def test_withplda(recid,model):
    original_tau = args.tau
    tau = args.tau
    reco2utt = reco2utt_dict[recid]
   

    feats_fname = f'{args.xvecpath}/{recid}.npy'
    labels_fname = f'{args.labelspath}/labels_{recid}'
    try:
        features = np.load(feats_fname)   
    except:
        feats_fname = f'{args.xvecpath}/xvector.scp'
        features = get_features(feats_fname,reco2utt)
    
    Nfeats = len(features)
    labels,overlap_ind,clean_ind= get_labels(labels_fname,Nfeats)

    print(features.shape,labels.shape,flush=True)
   
    # number of edges added is the indicator for early stopping
    num_edges_add_last_level = np.Inf
    ##################################

    affinity_mat = None
    if "rec_aff" not in mode: 
        if "PLDA" in mode:
            if "proc_feats" in mode:
                affinity_mat, features = get_PLDA_mat(features,args.pldamodel)
            else:
                affinity_mat, _ = get_PLDA_mat(features,args.pldamodel,temp_param=args.temp_param)
            # features = l2norm(features.astype('float32'))
        elif "cosine" in mode:
            if "proc_feats" in mode:
                affinity_mat, features = get_cosine_mat(features)
            else:
                affinity_mat, _ = get_cosine_mat(features)
            # features = l2norm(features.astype('float32'))
        # global_features = features.copy()
    
    if args.feats_norm:
        global_features = l2norm(features.astype('float32'))
    else:
        global_features = features.copy()
    
    dataset = SHARCDataset(features=features, labels=labels, mode=mode, k=args.knn_k,
                            levels=1, affinity_mat=affinity_mat, must_run=True, 
                            feats_norm=args.feats_norm,
                            pldamodel=args.pldamodel,ldatransform=args.ldatransform,temp_param=args.temp_param)
    print("graph level created")
    g = dataset.gs[0]
    g.ndata['pred_den'] = torch.zeros((g.number_of_nodes()))
    g.edata['prob_conn'] = torch.zeros((g.number_of_edges(), 2))
    global_labels = labels.copy()
    ids = np.arange(g.number_of_nodes())
    global_edges = ([], [])
    global_peaks = np.array([], dtype=np.long)
    global_edges_len = len(global_edges[0])
    global_num_nodes = g.number_of_nodes()
    global_num_nodes = g.number_of_nodes()

    print("DataLoader Created")

    # Predict connectivity and density
    for level in range(args.levels):
        if not args.use_gt:
            with torch.no_grad():
                output_bipartite = model(g)
            g.ndata['pred_den'] = output_bipartite.dstdata['pred_den'].to('cpu')
            g.edata['prob_conn'] = output_bipartite.edata['prob_conn'].to('cpu')
            if 'temporalProb' in mode:
                g.edata['prob_conn'][:,1] = g.edata['prob_conn'][:,1]*add_temporal_weight(g.number_of_nodes(),g.edges())
                g.edata['prob_conn'][:,0] = 1-g.edata['prob_conn'][:,1]
            torch.cuda.empty_cache()

        new_pred_labels, peaks,\
            global_edges, global_pred_labels, global_peaks = decode(g, tau, args.threshold, args.use_gt, mode,
                                                                    ids, global_edges, global_num_nodes,
                                                                    global_peaks)
        print("labels Predicted for level:%d"%(level))
        ids = ids[peaks]
        cluster_counts = np.unique(global_pred_labels,return_counts=True)
        
        print(f'cluster distribution: {cluster_counts}')

        if "rec_aff" not in mode:
            if "PLDA" in mode or "cosine" in mode:
                affinity_mat = affinity_mat[peaks][:,peaks]
        new_global_edges_len = len(global_edges[0])
        num_edges_add_this_level = new_global_edges_len - global_edges_len

        
        if stop_iterating(level, args.levels, args.early_stop, num_edges_add_this_level, num_edges_add_last_level, args.knn_k):
            print("Exiting due to early stop in level %d"%(level))
            break
        global_edges_len = new_global_edges_len
        num_edges_add_last_level = num_edges_add_this_level
        
        # build new dataset
        features, labels, cluster_features = build_next_level(features, labels, peaks,
                                                            global_features, global_pred_labels, global_peaks)
        print("build next level")
        # After the first level, the number of nodes reduce a lot. Using cpu faiss is faster.
        dataset = SHARCDataset(features=features, labels=labels, mode=mode, k=args.knn_k,
                                levels=1, cluster_features = cluster_features, affinity_mat=affinity_mat,
                                feats_norm=args.feats_norm,pldamodel=args.pldamodel)
        
        print("Next graph level created")
        
        if len(dataset.gs) == 0:
            print("Exiting due to empty dataset in level %d"%(level))
            break

        g = dataset.gs[0]
        
        g.ndata['pred_den'] = torch.zeros((g.number_of_nodes()))

        g.edata['prob_conn'] = torch.zeros((g.number_of_edges(), 2))
        
    print("Levels in output graph %d"%(level))
    evaluation(global_pred_labels, global_labels, args.metrics)
    ################################################################################################################
    # Generate RTTMs
    overlap = 1
    pref = f'k{args.knn_k}_tau{original_tau}'
    results_dict = defaultdict(np.array)
    results_dict[recid]=global_pred_labels
      
    out_file=args.out_path+'/'+'final_{}rttms/'.format(pref)

    mkdir_p(out_file)
    outpath=out_file +'/'+recid
    rttm_newfile=out_file+'/'+recid+'.rttm'
    
    rttm_gndfile = args.rttm_ground_path+'/'+recid+'.rttm'
    write_results_dict(recid, out_file, results_dict, reco2utt)

    der = compute_score(rttm_gndfile,rttm_newfile,outpath,0)
    if overlap:
        overlap_der = compute_score(rttm_gndfile,rttm_newfile,outpath,1)
        print("\n%s  overlap DER: %.2f" % (recid, overlap_der))
    print("\n%s DER: %.2f" % (recid, der))



def main():
    print('main')
##################
# Data Preparation
reco2utt_list = open(args.reco2utt_list).readlines()
reco2utt_dict = {}
for line in reco2utt_list:
    rec, utt = line.split(" ",1)
    reco2utt_dict[rec] = utt

if 'ami' in args.dataset_str:
    # with open("data/ami_train/uniq_ids_subtrain","r") as f:
    # with open("data/ami_train/labels_train/uniq_spk_ids","r") as f:
    if 'train' in args.dataset_str:
        uniq_spk_ids = f"lists/{args.dataset_str}/spkall.list"
    else:
        uniq_spk_ids = "uniq_spkr_list_ami"
 
    with open(uniq_spk_ids,"r") as f:
        spk_ids = f.readlines()
elif 'vox_diar' in args.dataset_str:
    with open("uniq_spkr_list_vox","r") as f:
        spk_ids = f.readlines()
else:
    spklist = f'lists/{args.dataset_str}/spkall.list'
    with open(spklist,"r") as f:
        spk_ids = f.readlines()

#print(spk_ids)
spk_dct = {}
for i,spk_id in enumerate(spk_ids):
    spk_dct[spk_id[:-1]] = i
    
########################################################################################
mode = args.mode.split(",")

##################
# Model Definition
if not args.use_gt:
    feature_dim = args.xvec_dim
    model = SHARC(feature_dim=feature_dim, nhid=args.hidden,
                   num_conv=args.num_conv, dropout=args.dropout,
                   use_GAT=args.gat, K=args.gat_k,
                   balance=args.balance,
                   use_cluster_feat=args.use_cluster_feat,
                   use_focal_loss=args.use_focal_loss)
    
    if not args.withcheckpoint:
        model.load_state_dict(torch.load(args.model_filename, map_location=device))
    else:
        checkpoints = torch.load(args.model_filename, map_location=device)
        if 'swa_model' in checkpoints:
            swa_model = AveragedModel(model)
            swa_model.load_state_dict(checkpoints['swa_model'])
            model = swa_model
        else:
            model.load_state_dict(checkpoints['model'])
    
    model = model.to(device)
    model.eval()
# bp()
print("Model Loaded")

########################################################################################################

if args.splitlist is not None:
    filelist = np.genfromtxt(args.splitlist,dtype=float).astype(int).reshape(-1,)
    recsublist = np.genfromtxt(args.feats_file,dtype=str)[filelist]
    
    for recid in recsublist:
        if args.pldamodel is None:
            print("Please provide PLDA model in pkl format")
        else:
            test_withplda(recid,model)                
else:
    recid = args.feats_file
    test_withplda(recid,model)

