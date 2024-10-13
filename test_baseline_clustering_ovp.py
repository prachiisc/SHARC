import argparse, time, os, pickle
import numpy as np
# from dataset_final.dataset import LanderDataset_withmfcc

import dgl
import torch
import torch.optim as optim
import pdb
from matplotlib import pyplot as plt
from dataset_final.dataset import LanderDataset_overlap

from models_final import LANDER, main_model
from dataset_final import LanderDataset_withmfcc, LanderDataset, LanderDset
from models_final.lander_e2e import LANDER_overlap,LANDER_overlap_sharc

# from utils_final import evaluation, decode, build_next_level, stop_iterating, get_cosine_mat, get_PLDA_mat, l2norm  
from utils_final import *
from utils_train import *
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.metrics import det_curve

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
    parser.add_argument('--model_filename_init', type=str, default=None) # initialize with model1
    parser.add_argument('--approach', type=str, default=None) # type of approaches sriram, myapproach, myapproach_withprob
    parser.add_argument('--labels_dir', type=str, default=None)
    
    # second pass
    parser.add_argument('--k_2ndpass', type=int, default=None)
    parser.add_argument('--overlap_th', type=float, default=0.5)
    parser.add_argument('--density_gap', type=float, default=0.0,help='gap between overlap node and clean node density')
    parser.add_argument('--modestat', type=int, default=None,help='number of 2nd pass neighbours to consider to compute mode of the labels')

    # use external overlap 
    parser.add_argument('--labelspath_pyannote', type=str, default=None,help='pyannote overlapping regions')
    parser.add_argument('--reco2utt_list_fine', type=str, default=None,help='fine resolution')
    parser.add_argument('--labelspath_fine', type=str, default=None,help='fine resolution')
    parser.add_argument('--xvecpath_fine', type=str, default=None,help='fine resolution')
    parser.add_argument('--fine2coarselabels', type=str, default=None,help='fine2coarse resolution')
    parser.add_argument('--intracluster_per', type=float, default=0.1, help='how much proportion of intra cluster samples to consider.')
    parser.add_argument('--out_path_org',type=str,help='path of first pass results')
    parser.add_argument('--overlap_filename',type=str,help='path of overlap detection results in seconds')
    
    args = parser.parse_args()
    return args

args = arguments()

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
                towrite = utt +'\t'+str(hypothesis[j])+'\n'
                output_label.writelines(towrite)
        output_label.close()

        rttm_channel=1
        segmentsfile = args.segments+'/'+f+'.segments'
        python = args.which_python
        # python = '/home/prachis/miniconda3/envs/mytorch/bin/python'
        kaldi_recipe_path="./"
        cmd = '{} {}/diarization/make_rttm.py --rttm-channel  {} {} {}/{}.labels {}/{}.rttm' .format(python,kaldi_recipe_path,rttm_channel, segmentsfile,output_file,f,output_file,f)        
        os.system(cmd)

def load_rttm(rttmfile,step=100):
    rttm = np.genfromtxt(rttmfile, dtype='str')
    rttm_idx = np.asarray([False,False,False,True,True,False,False,True,False, False])
    rttm = rttm[:,rttm_idx] 
    
    # unique_spks,spks = np.unique(rttm[:,-1],return_index=True)
    spks = rttm[:,-1]
    rttm = rttm[:,:2].astype(float)
    rttm[:,1] = rttm[:,0]+rttm[:,1] # convert duration to end point
    rttm = rttm *step # convert to frames
    rttm = rttm.astype(int)
    n_frames = rttm[-1,1]+1
    ref = np.ones(n_frames,dtype=int)*-1
    ref = ref.astype(str)
    for i, line in enumerate(rttm):
        start = line[0]
        end = line[1]
        ref[start:end+1] = spks[i] 
    return ref


def write_rttm_file(rttm_path_org, rttm_path, labels_org, channel=1, step=0.01, precision=2):
    """Write RTTM file.

    Parameters
    ----------
    rttm_path : Path
        Path to output RTTM file.

    labels : ndarray, (n_frames,)
        Array of predicted speaker labels. See ``get_labels`` for explanation.

    channel : int, optional
        Channel (0-indexed) to output segments for.
        (Default: 0)

    step : float, optional
        Duration in seconds between onsets of frames.
        (Default: 0.01)

    precision : int, optional
        Output ``precision`` digits.
        (Default: 2)
    """
    
    # Determine indices of onsets/offsets of speaker turns.

    unique_spks,labels = np.unique(labels_org,return_inverse=True)
    
    is_cp = np.diff(labels, n=1, prepend=-999, append=-999) != 0
    cp_inds = np.nonzero(is_cp)[0]
    bis = cp_inds[:-1]  # Last changepoint is "fake".
    eis = cp_inds[1:] -1
    
    N = len(bis)
    # rttm_idx = np.asarray([False,False,False,True,True,False,False,True,False, False])
    rttm = np.empty((N,10)).astype(str)
    
    rttm[:,0] =  "SPEAKER"
    rttm[:,1] = recid
    rttm[:,2] = channel
    rttm[:,3] = np.round(bis * step,2).astype(str)
    rttm[:,4] = np.round((eis-bis) * step,2).astype(str)
    rttm[:,5] = "<NA>"
    rttm[:,6] = "<NA>"
    rttm[:,7] = labels_org[bis]
    rttm[:,8] = "<NA>"
    rttm[:,9] = "<NA>"

    
    # remove the silence regions, labels_org=-1
    speech_ind = np.where(labels_org[bis]!='-1')[0]
    rttm = rttm[speech_ind]

    rttm_org = np.genfromtxt(rttm_path_org,dtype=str)
    rttm = np.concatenate((rttm_org,rttm),axis=0)
    np.savetxt(rttm_path,rttm, fmt='%s')

def get_overlap_labels(recid, n_frames,step=100,rttm_org=None,mode=['pyannote']):
    
    if 'pyannote' in mode:
        if args.labelspath_pyannote is None:
            overlap_labels = np.zeros((n_frames,),dtype=int)
            overlap_filename = f'{args.overlap_filename}/{recid}.txt'
            if os.path.isfile(overlap_filename):
                overlap_idx = np.genfromtxt(overlap_filename,dtype=float)
                if len(overlap_idx) == 0:
                    print(recid,": not predicted overlaps")
                    return overlap_labels
                overlap_idx = overlap_idx.reshape(-1,4) # 4 columns
                
                overlap_idx = overlap_idx[:,[1,2]]
                overlap_idx = overlap_idx*step
            overlap_idx= overlap_idx.astype(int)
           
            for start,end in overlap_idx:
                overlap_labels[start:end+1] = 1
        else:
            overlap_filename = f'{args.labelspath_pyannote}/labels_{recid}'
            overlap_labels_speech =  np.genfromtxt(overlap_filename,dtype=float)[:,1]
            overlap_labels_speech = overlap_labels_speech.astype(int)
            ref = load_rttm(f'{rttm_org}')
            overlap_labels = np.zeros_like(ref.astype(int))
            overlap_labels[ref!='-1'] = overlap_labels_speech # paste to speech regions
            # need to add silence frames
            # can use rttm or segments file
    else:
        overlap_labels = load_gnd_overlap(recid)
    
    return overlap_labels

def compute_score(rttm_gndfile,rttm_newfile,outpath,overlap):
    fold_local='services/'
    scorecode='score.py -r '

    if not overlap:
        cmd = '{} {}/dscore-master/{}{} --ignore_overlaps -s {} > {}.txt 2> err.txt'.format(args.which_python,fold_local,scorecode,rttm_gndfile,rttm_newfile,outpath)
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


def generate_rttms_overlaponly(recid,labelfull,reco2utt,overlap_th,k_2ndpass,mode):
    overlap = 1
    pref = f'2ndpassk{k_2ndpass}_density_gap{args.density_gap}_overlaponly'
    results_dict = defaultdict(np.array)
    results_dict_org = defaultdict(np.array)

    out_file=args.out_path+'/'+'final_{}rttms/'.format(pref)
    mkdir_p(out_file)

    pref_org = f'firstpass_'
    out_file_org=args.out_path+'/'+'final_{}rttms/'.format(pref_org)
    mkdir_p(out_file_org)
    rttm_file_org=args.out_path+'/'+'final_{}rttms/{}.rttm'.format(pref_org,recid)

    results_dict[recid]=labelfull[:,1]
    results_dict_org[recid] = labelfull[:,0]
    
    write_results_dict(recid, out_file, results_dict, reco2utt)

    write_results_dict(recid, out_file_org, results_dict_org, reco2utt)
    # read overlaps
    step = 100
    
    ref = load_rttm(f'{out_file}/{recid}.rttm')
    n_frames = len(ref)
    
    overlap_labels = get_overlap_labels(recid, n_frames,step,rttm_file_org,mode=mode)
    n_frames = min(n_frames,len(overlap_labels))
    overlap_labels = overlap_labels[:n_frames]
    ref = ref[:n_frames]
    ref[overlap_labels!=1] = '-1'

    # remaining steps
    # convert this frames into rttm
    # then concatenate with the single speaker rttm
    # check the performance
    # repeat the same process with ground truth labels
    # if it improves then we can start the GNN based training
    pref_overlap = f'2ndpassk{k_2ndpass}_density_gap{args.density_gap}_overlap'
    
    outpath = args.out_path+'/'+'final_{}rttms/'.format(pref_overlap)
    rttm_newfile=args.out_path+'/'+'final_{}rttms/{}.rttm'.format(pref_overlap,recid)
    mkdir_p(outpath)
    
    write_rttm_file(rttm_file_org, rttm_newfile,ref)
    rttm_gndfile = args.rttm_ground_path+'/'+recid+'.rttm'
    outpath=outpath +'/'+recid

    der = compute_score(rttm_gndfile,rttm_newfile,outpath,0)
    if overlap:
        overlap_der = compute_score(rttm_gndfile,rttm_newfile,outpath,1)
        print("\n%s  overlap DER: %.2f" % (recid, overlap_der))
    print("\n%s DER: %.2f" % (recid, der))
    
def get_features(feats_fname,reco2utt):
    # if args.dataset_str == "vox_diar" or args.dataset_str == "lib_vox_cv_all":
    if args.dataset_str == "lib_vox_cv_all":
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


def test_withplda(recid):
    reco2utt = reco2utt_dict[recid]
    feats_fname = f'{args.xvecpath}/{recid}.npy'
    
    try:
        features = np.load(feats_fname)   
    except:
        feats_fname = f'{args.xvecpath}/xvector.scp'
        features = get_features(feats_fname,reco2utt)
    
    if args.labels_dir is not None:
        print('labels provided!')
        global_pred_labels = np.genfromtxt(f'{args.labels_dir}/{recid}.labels',dtype=str)[:,1]
        _,global_pred_labels = np.unique(global_pred_labels,return_inverse=True)
       
    test_withplda_2ndpass(global_pred_labels,reco2utt,features,recid)

def test_withplda_2ndpass(global_pred_labels,reco2utt,features,recid=None):
    # use model 1 output , take model2 embeddings , change affinity matrix based on model1 output
    # then recreate new graph using new knn, then only perform scoring on the new edges.
    # Data Preparation
    
    global_num_nodes = len(global_pred_labels)
    similarity = global_pred_labels.reshape(-1,1) == global_pred_labels.reshape(-1,1).T
    similarity = 1- similarity
   
    k = args.knn_k
    tau = args.overlap_th
    
    # create another graph taking new nearest neighbours
    if "PLDA" in mode:
        if "proc_feats" in mode:
                affinity_mat, features = get_PLDA_mat(features,args.pldamodel,temp_param=args.temp_param)
        else:
            affinity_mat, _ = get_PLDA_mat(features,args.pldamodel,temp_param=args.temp_param)
    

    affinity_mat = affinity_mat*similarity
    # args.knn_k/2, k=10
    cluster_ids = np.unique(global_pred_labels)

    if args.k_2ndpass is not None:
        k = args.k_2ndpass
    else:
        k = args.knn_k
    

    ids = np.arange(global_num_nodes)
    tau = args.overlap_th

    max_overlaps_rows = ids

    prob_conn = affinity_mat
    # based on prob_matrix 

    import statistics
    if args.modestat is None:
        modestat = k   #k_2nd_pass
    else:
        modestat = args.modestat
    # check mode of top modestat scores
    a1 =  np.argsort(prob_conn,axis=1)[:,::-1]
    a2 = global_pred_labels[a1[:,:modestat]]
    org_labels_predicted_clusters = a1[:,0]
    for i,org in enumerate(a2):
        org_labels_predicted_clusters[i] = statistics.mode(org) 


    if 'pyannote' in mode or 'ground' in mode:
        org_labels_predicted_clusters[np.where(prob_conn.sum(1)==0)[0]] = -1 # if the probability is not greater than tau

    labelfull = np.ones((global_num_nodes,2)) * -1
    labelfull[:,0] = global_pred_labels
    
    if len(cluster_ids)>1:
        # plda mat based speakers
        labelfull[max_overlaps_rows,1] = org_labels_predicted_clusters
    
    

    # compute fa and missrate of overlap detection
    pred_ovp_ind = np.where(labelfull[:,1]>-1)[0]
    pred_ovp = np.zeros((global_num_nodes,))
    pred_ovp[pred_ovp_ind] = 1

        
    generate_rttms_overlaponly(recid,labelfull.astype(int),reco2utt,tau,k,mode)

def main():
    print('main')
##################
# Data Preparation
reco2utt_list = open(args.reco2utt_list).readlines()
reco2utt_dict = {}
for line in reco2utt_list:
    rec, utt = line.split(" ",1)
    reco2utt_dict[rec] = utt

########################################################################################
mode = args.mode.split(",")

if args.splitlist is not None:
    filelist = np.genfromtxt(args.splitlist,dtype=float).astype(int).reshape(-1,)
    recsublist = np.genfromtxt(args.feats_file,dtype=str)[filelist]
    for recid in recsublist:
        test_withplda(recid)