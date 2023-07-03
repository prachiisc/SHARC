import argparse, time, os, pickle
import numpy as np
# from dataset_final.dataset import LanderDataset_withmfcc

import dgl
import torch
import torch.optim as optim
import pdb
from matplotlib import pyplot as plt

from models_final import main_model
from dataset_final import SHARCDataset_withmfcc

from utils_final import *
from utils_train import *

from collections import defaultdict
import subprocess

import sys
import warnings
from services.kaldi_io import *

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
    parser.add_argument('--feats_file',type=str,default=None)
    parser.add_argument('--feats_norm',type=int,default=0,help='Normalize features before GNN')
    parser.add_argument('--mymode',type=str,default='test')
    parser.add_argument('--ark_file',type=str,default=None)
    parser.add_argument('--fulltrain',type=int,default=0)
    parser.add_argument('--pldamodel', type=str, default=None)
    # for parallel processing
    parser.add_argument('--splitlist',type=str,default=None)
    # Subgraph
    parser.add_argument('--batch_size', type=int, default=4096)
    # output xvectors 
    parser.add_argument('--xvecpath_out', type=str, default=None,help='store model xvectors')
    parser.add_argument('--xvecbeta',type=float,default=0.5)
    # generate rttms 
    parser.add_argument('--segments',type=str,default=None)
    parser.add_argument('--rttm_ground_path',type=str,default=None)
    parser.add_argument('--which_python',type=str,default='python')
    parser.add_argument('--model_filename_e2e',type=str,default=None,help='save the new lander model')

    # temporal continuity
    parser.add_argument('--temp_param', type=str, default='5,0.95') # temporal continuity parameters, neb, beta1
    
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

def extract_xvectors(recid,model):
    reco2utt = reco2utt_dict[recid]
    
    mfcc_feats,_ = load_mfcc_feats_nosilence(args,recid,featsdict,reco2utt,featsbatch=1)
    feats_fname = f'{args.xvecpath}/{recid}.npy'
    features = np.load(feats_fname)
    mfcc_features = torch.FloatTensor(mfcc_feats).to(device)
    org_xvecs = torch.FloatTensor(features).to(device)
    print('mfcc: ',mfcc_features.shape)
    print(features.shape,flush=True)
    mfcc_xvec_features = model.extract_xvecs(mfcc_features,org_xvecs)
    xvec_features = mfcc_xvec_features.cpu().detach().numpy()
    np.save(f'{args.xvecpath_out}/{recid}.npy',xvec_features)
    print(f'{args.xvecpath_out}/{recid}.npy')
    
def extract_xvectors_pldafeats(recid,model,fid2fname,arkf):
    print(recid)
    reco2utt = reco2utt_dict[recid]
    # bp()
    labels_fname = f'{args.labelspath}/labels_{fid2fname[recid]}'
    mfcc_feats,_ = load_mfcc_feats_nosilence(args,recid,featsdict,reco2utt,featsbatch=1)
    feats_fname = f'{args.xvecpath}/{recid}.npy'
    features = np.load(feats_fname)

    with open(labels_fname,"r") as f:
        label_file = f.readlines()
    clean_ind = []  
    for i,line in enumerate(label_file):
        label = line.split()
        if len(label) == 2:
            clean_ind.append(i)
            
    clean_ind = np.array(clean_ind)
    labels = np.array(label_file)[clean_ind]
    mfcc_features = torch.FloatTensor(mfcc_feats[clean_ind]).to(device)
    org_xvecs = torch.FloatTensor(features[clean_ind]).to(device)
    print('mfcc: ',mfcc_features.shape)
    print(features.shape,flush=True)
    mfcc_xvec_features = model.extract_xvecs(mfcc_features,org_xvecs)
    xvec_features = mfcc_xvec_features.cpu().detach().numpy()
    
    # np.save(f'{args.xvecpath_out}/{recid}.npy',xvec_features)
    # print(f'{args.xvecpath_out}/{recid}.npy')

    xvecdict = {} 
    for i,line in enumerate(labels):
        utt,spkid = line.split()
        xvecs = xvec_features[i]
        key = f'{spkid}_{utt}' 
        xvecdict[key] = xvecs 
        write_vec_flt(arkf, xvecs, key=key)


def get_labels(labels_fname,Nfeats):
    try:
        fptr = open(labels_fname,'r')
        labels = fptr.readlines()
        fptr.close()
        if 'ami' in args.dataset_str:
            labels = [int(labels[i].split()[-1]) for i in range(len(labels))]
        else:
            labels = [spk_dct[labels[i].split()[-1]] for i in range(len(labels))]
        labels = np.array(labels)
    except:
        labels = np.zeros((Nfeats,),dtype=int) #dummy

    return labels

def test(recid,model):
    reco2utt = reco2utt_dict[recid]
    pred_fname = f'{args.out_path}/{recid}_pred_labels_k{args.knn_k}_tau{args.tau}.txt'
    print(pred_fname)
    # if os.path.exists(pred_fname):
    #     return
    mfcc_feats,idx_xvec = load_mfcc_feats_nosilence(args,recid,featsdict,reco2utt,featsbatch=1)
    feats_fname = f'{args.xvecpath}/{recid}.npy'
    features = np.load(feats_fname)
    labels_fname = f'{args.labelspath}/labels_{recid}'
    Nfeats = features.shape[0]
    labels = get_labels(labels_fname,Nfeats)
    
    print(features.shape,labels.shape,flush=True)
    # number of edges added is the indicator for early stopping
    num_edges_add_last_level = np.Inf
    ##################################
    mfcc_features = torch.FloatTensor(mfcc_feats).to(device)
    org_xvecs = torch.FloatTensor(features).to(device)
    print('mfcc: ',mfcc_features.shape)

    fullbatchsize,n_frames,D = mfcc_features.shape
    batchsize = min(500,fullbatchsize) 
    batch_count = int(fullbatchsize/batchsize)
    cur_batch = batchsize*batch_count
    remainder = fullbatchsize - cur_batch

    mfcc_features_batch = mfcc_features[:batchsize*batch_count].reshape(batch_count,batchsize,n_frames,D)
    org_xvecs_batch = org_xvecs[:batchsize*batch_count].reshape(batch_count,batchsize,args.xvec_dim)
    mfcc_xvec_features = []
    
    for i in range(batch_count):
        temp_xvecs = model.extract_xvecs(mfcc_features_batch[i],org_xvecs_batch[i]).detach().cpu().numpy()
        mfcc_xvec_features.append(temp_xvecs) #batch,N,D
        del temp_xvecs
        print('count:',i)
   
    if remainder > 0:
        print('count:',i)
        mfcc_features = mfcc_features[cur_batch:]
        org_xvecs = org_xvecs[cur_batch:]
        temp_xvecs = model.extract_xvecs(mfcc_features,org_xvecs).detach().cpu().numpy()
        mfcc_xvec_features.append(temp_xvecs)

    mfcc_xvec_features = np.concatenate(mfcc_xvec_features,axis=0)
    mfcc_xvec_features = torch.from_numpy(mfcc_xvec_features)
    if args.feats_norm:
        global_features = mfcc_xvec_features/torch.norm(mfcc_xvec_features,dim=1).reshape(-1,1)
    else:
        global_features = mfcc_xvec_features
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
          
    dataset = SHARCDataset_withmfcc(mfcc_xvec_features=mfcc_xvec_features, features=features, labels=labels, mode=mode, k=args.knn_k,
                            levels=1, affinity_mat=affinity_mat, must_run=True, 
                            feats_norm=args.feats_norm,
                            pldamodel=args.pldamodel,
                            temp_param=args.temp_param)
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
            torch.cuda.empty_cache()

        new_pred_labels, peaks,\
            global_edges, global_pred_labels, global_peaks = decode(g, args.tau, args.threshold, args.use_gt, mode,
                                                                    ids, global_edges, global_num_nodes,
                                                                    global_peaks)
        print("labels Predicted for level:%d"%(level))
        ids = ids[peaks]
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
        mfcc_xvec_features, features, labels, cluster_features = build_next_level_e2e(mfcc_xvec_features,features, labels, peaks,
                                                            global_features, global_pred_labels, global_peaks)
        print("build next level")
        # After the first level, the number of nodes reduce a lot. Using cpu faiss is faster.
        dataset = SHARCDataset_withmfcc(mfcc_xvec_features=mfcc_xvec_features, features=features, labels=labels, mode=mode, k=args.knn_k,
                                levels=1, cluster_features = cluster_features, affinity_mat=affinity_mat, 
                                feats_norm=args.feats_norm,
                                pldamodel=args.pldamodel)
        print("Next graph level created")
        if len(dataset.gs) == 0:
            print("Exiting due to empty dataset in level %d"%(level))
            break
        g = dataset.gs[0]
        g.ndata['pred_den'] = torch.zeros((g.number_of_nodes()))
        g.edata['prob_conn'] = torch.zeros((g.number_of_edges(), 2))

    print("Levels in output graph %d"%(level))
    np.savetxt(pred_fname,global_pred_labels,fmt="%d",delimiter='\n')
    evaluation(global_pred_labels, global_labels, args.metrics)
    ################################################################################################################
    # Generate RTTMs
    overlap = 1
    pref = f'k{args.knn_k}_tau{args.tau}'
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
featsdict = {}
with open(args.featspath) as fpath:
    for line in fpath: 
        key, value = line.split(" ",1)
        featsdict[key] = value.rsplit()[0]

reco2utt_list = open(args.reco2utt_list).readlines()
reco2utt_dict = {}
for line in reco2utt_list:
    rec, utt = line.split(" ",1)
    reco2utt_dict[rec] = utt

if "vox" in args.dataset_str:
    spklist = "uniq_spkr_list_vox"
elif "ami" in  args.dataset_str:
    spklist = "uniq_spk_ids_ami"
with open(spklist,"r") as f:
    spk_ids = f.readlines()
spk_dct = {}
for i,spk_id in enumerate(spk_ids):
    spk_dct[spk_id[:-1]] = i
    
########################################################################################
mode = args.mode.split(",")

##################
# Model Definition
if not args.use_gt:
    feature_dim = args.xvec_dim
    model = main_model(xvecmodelpath=args.xvec_model_weight_path,
               model_filename = args.model_filename,
               feature_dim=feature_dim, nhid=args.hidden,
               num_conv=args.num_conv, dropout=args.dropout,
               use_GAT=args.gat, K=args.gat_k,
               balance=args.balance,
               use_cluster_feat=args.use_cluster_feat,
               use_focal_loss=args.use_focal_loss,
               device = device,
               fulltrain=args.fulltrain,
               xvecbeta=args.xvecbeta)
    checkpoints = torch.load(args.model_savepath, map_location=device)
    model.load_state_dict(checkpoints['model'])
    
    model = model.to(device)
    model.eval()
# bp()
print("Model Loaded")

########################################################################################################

if args.splitlist is not None:
    filelist = np.genfromtxt(args.splitlist,dtype=float).astype(int).reshape(-1,)
    recsublist = np.genfromtxt(args.feats_file,dtype=str)[filelist]
    if args.mymode=='test':
        for recid in recsublist:
            test(recid,model)
    elif args.mymode=='extract_xvec':
        if not os.path.exists(args.model_filename_e2e):
            torch.save(model.mylander.state_dict(),args.model_filename_e2e)
        for recid in recsublist:
            extract_xvectors(recid,model)
    elif args.mymode == 'extract_pldafeats':
        pair_list = open(args.file_pairs).readlines()
        fid2fname = {}
        for line in pair_list:
            fid, fname = line.split()
            fid2fname[fid] = fname
        with open(args.ark_file,'wb') as arkf:
            for recid in recsublist:
                extract_xvectors_pldafeats(recid,model,fid2fname,arkf)
else:
    recid = args.feats_file
    test(recid,model)
