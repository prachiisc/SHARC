from __future__ import division
from __future__ import print_function

import argparse
import time
import random
from unicodedata import normalize
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from pdb import set_trace as bp
import os
import sys
from collections import defaultdict
import subprocess
import pickle as pkl
#sys.path.insert(0,'gae/')
# sys.path.insert(0,'services/')

from scipy.special import expit,logit
from utils_cluster import * 
from utils_final import *
import networkx as nx
from scipy import sparse

# load_data, mask_test_edges, preprocess_graph, get_roc_score, plot_results_full
# from utils import load_data_dihard, load_data_dihard_plda, mask_val_edges, load_data_dihard_pldaSpecc
# from utils import sparse_to_tuple, get_roc_score_modified, load_data_dihard_pic, mask_val_edges_simplified, mkdir_p
# import services.pic_dihard as pic
from services.run_spectralclustering import do_spectral_clustering
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, accuracy_score
from scipy.ndimage import gaussian_filter

import services.agglomerative as ahc
import matplotlib as mlt
from matplotlib import pyplot as plt
from tqdm import tqdm

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# from set_gpu import cuda_gpu_available

from validate_clustering import *
# validate_path_integral, validate_path_integral_ami, validate_ahc, validate_spectral_clustering_weighted, validate_clustering_weighted
from sklearn.metrics import f1_score
# from hungarian_algorithm import algorithm
# from services.softkmeans import soft_k_means

from utils_train import * # kaldi_io
mlt.use('Agg')
# #select device
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gcn_ae', help="models used")
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')

    parser.add_argument('--dataset_str', type=str, default='dihard_dev_2020_track1_fbank_jhu', help='type of dataset.')
    parser.add_argument('--which_python', type=str, default='/home/prachis/miniconda3/envs/mytorch/bin/python', help='python path.')
    parser.add_argument('--pldamodel', type=str, default=None, help='Plda model path.')

    parser.add_argument('--segments', type=str, default='/data1/prachis/Dihard_2020/SSC/lists/dihard_dev_2020_track1/segments_xvec/', help='segments path.')
    parser.add_argument('--reco2utt_list', type=str, default='/data1/prachis/Dihard_2020/SSC/lists/dihard_dev_2020_track1/tmp/spk2utt', help='reco2utt path.')
    parser.add_argument('--reco2num_list', type=str, default='/data1/prachis/Dihard_2020/SSC/lists/dihard_dev_2020_track1/tmp/reco2num_spk', help='reco2utt path.')
    parser.add_argument('--rttm_ground_path', type=str, default='/data1/prachis/Dihard_2020/Dihard_2020_track1/data/dihard_dev_2020_track1/filewise_rttms/', help='ground truth rttm path of rttm.')
    parser.add_argument('--xvecpath', type=str,default=None)
    parser.add_argument('--filename', type=str,default=None)
    parser.add_argument('--extract_feats', action='store_true')

    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--outf', type=str, default='exp/results/', help='output path.')
    parser.add_argument('--graphfold', type=str, default='exp/graphs/', help='train val graphs path.')
    parser.add_argument('--clustering', type=str, default='spectral', help='spectral/pic')
    parser.add_argument('--useoverlap', type=int, default=1)
    parser.add_argument('--ngpu', type=str, default='0')
    # parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--splitpath', type=str, default=None)
    parser.add_argument('--savedmodel', type=str, default=None,help='path of supervised trained model')
    parser.add_argument('--K', type=int, default=30)
    parser.add_argument('--z', type=float, default=0.1)
    parser.add_argument('--nb', type=int, default=5)
    parser.add_argument('--beta', type=float, default=0.95)
    parser.add_argument('--pref', type=str, default=None)
    parser.add_argument('--scale', type=int, default=1)
    parser.add_argument('--PLDA', type=str, default='dihard')
    args = parser.parse_args()

    return args

args = arguments()

if int(args.ngpu) >= 0 :
    args.ngpu = cuda_gpu_available()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.ngpu)

    print('GPU selected: ',args.ngpu)
    # torch.cuda.set_device(args.ngpu)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    device = 'cpu'

def get_knn(A,myK=50):
    B = -A.copy()
    K = myK
    N = B.shape[0]
    K = min(K,N-1)
    B[np.diag_indices(N)] = -np.inf
    sortedDist = np.sort(B,axis=1)
    NNIndex = np.argsort(B,axis=1)
    NNIndex = NNIndex[:,:K+1]
    ND = -sortedDist[:, 1:K+1].copy()
    NI = NNIndex[:, 1:K+1].copy()
    XI = np.dot(np.arange(N).reshape(-1,1),np.ones((1,K),dtype=int))
    graphW = coo_matrix((ND.reshape(-1), (XI.reshape(-1),  NI.reshape(-1))), shape=(N, N)).toarray()
    graphW[np.diag_indices(N)]=0
    
    adj = graphW
    adj = sparse.csr_matrix(adj)
    return adj

def get_clusterinit(distance_matrix,plda_dist,n_clusters=None, clusteringinit='spectral'):
        clusteringinit = 'pic'
        n_nodes = distance_matrix.shape[0]
        if clusteringinit == 'spectral':
            # pref='_baseline_threshold{}'.format(args.threshold)
            # pref='_baseline_threshold{}_scaledaffinity'.format(args.threshold)
            # # bp()
            # scoretype =  None
            scoretype = 'laplacian'
            # pref='_baseline_nolaplace'
            # pref='_baseline_scaled10_nolaplace'
            # pref='_baseline_scaled10_'
            
            # _ = validate_spectral_clustering(filename,distance_matrix,reco2utt,1,1,n_clusters,pref=pref,rttm_gndfile=rttm_gndfile,scoretype=scoretype)
            # _ = validate_spectral_clustering_threshold(filename,distance_matrix,reco2utt,1,1,threshold=args.threshold,pref=pref,rttm_gndfile=rttm_gndfile)

            if n_clusters is not None:
                minK = n_clusters
                maxK = n_clusters
                th = 1e-2
            else:
                minK = 1
                maxK = 10
                th = 0.5
            labelfull = do_spectral_clustering(distance_matrix,
                                                    minclusters=minK,
                                                    maxclusters=maxK,
                                                    truek=4,custom_dist='cosine',
                                                    scoretype='laplacian',
                                                    stop_eigenvalue=th)  
            n_clusters = len(np.unique(labelfull))
        else:
            distance_matrix = distance_matrix *(distance_matrix + plda_dist)
            distance_matrix = distance_matrix/distance_matrix.max()
            N = distance_matrix.shape[0]
            # clusteringinit == 'pic':
            # pref='_baseline_'
            flag = 1
            # _ = validate_path_integral(args,filename,distance_matrix,reco2utt,flag,1,n_clusters,pref=pref,rttm_gndfile=rttm_gndfile)
            if flag==1:
                # z = 0.01
                # K = nframe
                neb = 5
                beta1 = 0.95
                toep = np.abs(np.arange(N).reshape(N,1)-np.arange(N).reshape(1,N))
                toep[toep>neb] = neb
                weighting = beta1**(toep)
                distance_matrix = weighting*distance_matrix
            else:
                distance_matrix = distance_matrix
                
            labelfull=np.arange(distance_matrix.shape[0])
            clusterlen=[1]*len(labelfull)    
            N = len(labelfull)
            # plda scores
            z=0.1
            K = 30
            final_k = min(K, N - 1) 
            threshold = None
            mypic =pic.PIC_ami_threshold(n_clusters,clusterlen,labelfull,distance_matrix.copy(),threshold,K=final_k,z=z) 
            nframe = distance_matrix.shape[0]
            if flag == 2:
                if threshold == None:
                    labelfull,_ = mypic.gacCluster_oracle_org()
                else:
                    if n_clusters > 1:
                        labelfull,clusterlen = mypic.gacCluster_org()
                    else:
                        labelfull = np.zeros((nframe,1))
                        clusterlen = [nframe]
            else:
                if threshold == None:
                    labelfull,_= mypic.gacCluster_oracle()
                else:
                    if n_clusters > 1:
                        labelfull,_ = mypic.gacCluster()
                    else:
                        labelfull = np.zeros((nframe,1))
                        clusterlen = [nframe]
           
        
        # flag = 2
        # pref='_baseline_'
        # _ = validate_ahc(args,filename,distance_matrix,reco2utt,flag,1,n_clusters,pref=pref,rttm_gndfile=rttm_gndfile)


        labelfull = labelfull.reshape(-1,1)
        adj_bool = labelfull == labelfull.T
        adj_label = np.ones((n_nodes,n_nodes))*adj_bool
        final_adj = adj_label * (adj_label + plda_dist)
        # final_adj = adj_label
        # final_adj =  np.maximum(adj_label,plda_dist)
        
        # final_adj /=np.max(final_adj)
        adj_label_no_diagonal = final_adj - np.eye((final_adj.shape[0]))
        # adj_label_no_diagonal = get_knn(adj_label_no_diagonal,myK=50)

        return adj_label_no_diagonal
        

def get_clusterinit_both(distance_matrix,plda_dist,n_clusters=None, clusteringinit='spectral',filename=None):
        # clusteringinit = 'pic'
        n_nodes = distance_matrix.shape[0]
        if 1: #clusteringinit == 'spectral':
            # pref='_baseline_threshold{}'.format(args.threshold)
            # pref='_baseline_threshold{}_scaledaffinity'.format(args.threshold)
            # # bp()
            # scoretype =  None
            scoretype = 'laplacian'
            # pref='_baseline_nolaplace'
            # pref='_baseline_scaled10_nolaplace'
            # pref='_baseline_scaled10_'
            
            # _ = validate_spectral_clustering(filename,distance_matrix,reco2utt,1,1,n_clusters,pref=pref,rttm_gndfile=rttm_gndfile,scoretype=scoretype)
            # _ = validate_spectral_clustering_threshold(filename,distance_matrix,reco2utt,1,1,threshold=args.threshold,pref=pref,rttm_gndfile=rttm_gndfile)

            if n_clusters is not None:
                minK = n_clusters
                maxK = n_clusters
                th = 1e-2
            else:
                minK = 1
                maxK = 10
                th = 0.5
            labelfull_spec = do_spectral_clustering(distance_matrix,
                                                    minclusters=minK,
                                                    maxclusters=maxK,
                                                    truek=4,custom_dist='cosine',
                                                    scoretype='laplacian',
                                                    stop_eigenvalue=th)  
            n_clusters = len(np.unique(labelfull_spec))
        if 1:
           
            labelspath = 'exp_march/results_spec_sup_ae_norm_cosinesoftmaxloss_angleproto_clean_0_xvec0.75shift_norm_PLDA_scaled/_avg_accumgradient_use_gnd_adj_withadjplda/ami_dev_fbank_0.75s/results_sup_pic_widePLDA/final_pic_knnpldainit50_sup_affine_K30_z0.1_nb5_beta0.95_Model40rttms/'
            if os.path.isfile(f'{labelspath}/{filename}.labels'):
                labelfull = np.genfromtxt(f'{labelspath}/{filename}.labels',dtype=str)[:,1]
            else:
                distance_matrix = distance_matrix *(distance_matrix + plda_dist)
                distance_matrix = distance_matrix/distance_matrix.max()
                N = distance_matrix.shape[0]
                # clusteringinit == 'pic':
                # pref='_baseline_'
                flag = 1
                # _ = validate_path_integral(args,filename,distance_matrix,reco2utt,flag,1,n_clusters,pref=pref,rttm_gndfile=rttm_gndfile)
                
                if flag==1:
                    # z = 0.01
                    # K = nframe
                    neb = 5
                    beta1 = 0.95
                    toep = np.abs(np.arange(N).reshape(N,1)-np.arange(N).reshape(1,N))
                    toep[toep>neb] = neb
                    weighting = beta1**(toep)
                    distance_matrix = weighting*distance_matrix
                else:
                    distance_matrix = distance_matrix
                
                labelfull=np.arange(distance_matrix.shape[0])
                clusterlen=[1]*len(labelfull)    
                N = len(labelfull)
                # plda scores
                z=0.1
                K = 30
                final_k = min(K, N - 1) 
                threshold = None
                mypic =pic.PIC_ami_threshold(n_clusters,clusterlen,labelfull,distance_matrix.copy(),threshold,K=final_k,z=z) 
                nframe = distance_matrix.shape[0]
                if flag == 2:
                    if threshold == None:
                        labelfull,_ = mypic.gacCluster_oracle_org()
                    else:
                        if n_clusters > 1:
                            labelfull,clusterlen = mypic.gacCluster_org()
                        else:
                            labelfull = np.zeros((nframe,1))
                            clusterlen = [nframe]
                else:
                    if threshold == None:
                        labelfull,_= mypic.gacCluster_oracle()
                    else:
                        if n_clusters > 1:
                            labelfull,_ = mypic.gacCluster()
                        else:
                            labelfull = np.zeros((nframe,1))
                            clusterlen = [nframe]
           
        
        # flag = 2
        # pref='_baseline_'
        # _ = validate_ahc(args,filename,distance_matrix,reco2utt,flag,1,n_clusters,pref=pref,rttm_gndfile=rttm_gndfile)

        for i,lfull in enumerate(labelfull):
            labelfull[i] = lfull.split('[')[1].split(']')[0]
       
        labelfull = labelfull.astype(int)
        labelfull = labelfull.reshape(-1,1)
        
        adj_bool = labelfull == labelfull.T
        adj_label = np.ones((n_nodes,n_nodes))*adj_bool

        labelfull_spec = labelfull_spec.reshape(-1,1)
        adj_bool_spec = labelfull_spec == labelfull_spec.T
        adj_label_spec = np.ones((n_nodes,n_nodes))*adj_bool_spec
        # final_adj = np.maximum(adj_label,adj_label_spec)
        # final_adj = adj_label_spec*(adj_label+adj_label_spec)
        # final_adj = adj_label * (adj_label + plda_dist)
        final_adj = 0.7*adj_label + 0.3*adj_label_spec
        # final_adj = final_adj * (final_adj + plda_dist)
        # final_adj = adj_label
        # final_adj =  np.maximum(adj_label,plda_dist)
        
        final_adj /=np.max(final_adj)
        adj_label_no_diagonal = final_adj - np.eye((final_adj.shape[0]))
        # adj_label_no_diagonal = get_knn(adj_label_no_diagonal,myK=50)

        # return adj_label_no_diagonal
        return labelfull
        

class CustomDataset(Dataset):
    def __init__(self, datalist,set):
        # self.data = torch.FloatTensor(data).to(device)
        # self.labels = torch.tensor(labels).to(device).long().reshape(-1,1)
        # self.lengths = torch.tensor(lengths).to(device).long()
       
        self.datalist = datalist
        self.reco2num = open(args.reco2num_list).readlines()
        self.pair_list = open(args.reco2utt_list).readlines()
        self.set = set
        self.rotate = 0
        self.p = 0.4
        self.ntrain = len(np.unique(self.datalist))
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, nidx):
        
        idx = self.datalist[nidx]
        reco2utt = self.pair_list[idx]
        
        n_clusters = int(self.reco2num[idx].rsplit()[1])
        
        filename = reco2utt.split()[0]

        adj, features,clean_ind,ovpmax2_ind = load_data_dihard(args.dataset_str,filename,set=self.set)
        # features = features/np.linalg.norm(features,axis=1).reshape(-1,1)
        
        # if random.random() < self.p :
        if nidx >= self.ntrain and self.set == 'train':
            self.rotate = 1
        else:
            self.rotate = 0
        if self.rotate:
            rotation_mat = torch.FloatTensor(SO.rvs(features.shape[1]))
            features = torch.matmul(features, rotation_mat)
        scale = 10
        adj_plda,output_new = load_data_dihard_plda(args.dataset_str,filename,device,n_clusters,set=self.set,scale=scale)
        adj_plda_no_diagonal = adj_plda - sp.dia_matrix((adj_plda.diagonal()[np.newaxis, :], [0]), shape=adj_plda.shape)
        
        adj_label = adj + sp.eye(adj.shape[0])
        if self.set == 'train':
            if not args.useoverlap :
                adj_label = adj_label[np.ix_(clean_ind,clean_ind)]
                adj_plda_no_diagonal = adj_plda_no_diagonal.tocsr()[np.ix_(clean_ind,clean_ind)]
                features = features[clean_ind]
            else :
                adj_label = adj_label[np.ix_(ovpmax2_ind,ovpmax2_ind)]
                adj_plda_no_diagonal = adj_plda_no_diagonal.tocsr()[np.ix_(ovpmax2_ind,ovpmax2_ind)]
                features = features[ovpmax2_ind]
            
            
        n_nodes, _ = features.shape
        if self.set == 'train':
            sum_mask = adj_label.sum(1)
            indnonzero = np.where(sum_mask!=1.0)[0]
            
            if len(indnonzero) != n_nodes:
                # bp()
                features = features[indnonzero]
                adj_label = adj_label[np.ix_(indnonzero,indnonzero)]
                adj_plda_no_diagonal = adj_plda_no_diagonal[np.ix_(indnonzero,indnonzero)]
                n_nodes = len(indnonzero)
        adj_norm = preprocess_graph(adj_plda_no_diagonal)
        
        # print(filename)
        
        # plt.figure()
        # plt.imshow(output_new)
        # plt.colorbar()
        # plt.savefig(f'validation_scores/{filename}_pldascaledsigmoid.png')
        
        # plt.figure()
        # plt.imshow(adj_plda_no_diagonal.toarray())
        # plt.colorbar()
        # plt.savefig(f'validation_scores/{filename}_adjplda_nodiag.png')
        
        # plt.figure()
        # plt.imshow(adj_norm.to_dense())
        # plt.colorbar()
        # plt.savefig(f'validation_scores/{filename}_adjnorm.png')
        
        # plt.figure()
        # plt.imshow(adj.toarray())
        # plt.colorbar()
        # plt.savefig(f'validation_scores/{filename}_adjgndlabel.png')
        
        
        # adj_norm = adj_norm.to(device)
        # adj_label = torch.FloatTensor(adj_label.toarray()).to(device)

        return features,adj_label.toarray(),adj_norm.to_dense(),n_clusters,n_nodes,filename


def unique(arr, return_ind=False):
    if return_ind:
        k = 0
        d = dict()
        uniques = np.empty(arr.size, dtype=arr.dtype)
        indexes = np.empty(arr.size, dtype='i')
        for i, a in enumerate(arr):
            if a in d:
                indexes[i] = d[a]
            else:
                indexes[i] = k
                uniques[k] = a
                d[a] = k
                k += 1
        return uniques[:k], indexes
    else:
        _, idx = np.unique(arr, return_index=True)
        return arr[np.sort(idx)]

def write_results_dict(fname, output_file, results_dict, reco2utt):
        """Writes the results in label file"""
        f = fname
        output_label = open(output_file+'/'+f+'.labels','w')

        hypothesis = results_dict[f]
        meeting_name = f
        reco = reco2utt.split()[0]
        utts = reco2utt.rstrip().split()[1:]
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
    #   bp()
      # print('--------------------------------------------------')
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
      
      # print('----------------------------------------------------')
      # subprocess.check_call(cmd,stderr=subprocess.STDOUT)
      # print('scoring ',rttm_gndfile)
      
      output=subprocess.check_output(bashCommand,shell=True)
      # output=subprocess.check_output(bashCommand,stderr=subprocess.DEVNULL)
      return float(output.decode('utf-8').rstrip())



class baseline_clustering:
    def __init__(self,train_dataloader=None,val_dataloader=None,**kwargs):
        self.feat_dim = kwargs['feat_dim']
       
        self.test_freq = kwargs['test_freq']
        self.final =0
        self.forcing_label = 0
        self.results_dict={}
        self.results = defaultdict(list)

        self.trainloader = train_dataloader
        self.valloader = val_dataloader
        if 'rttm_gndval' in kwargs.keys():
            self.rttm_gnd_val = kwargs['rttm_gndval']
        if 'out_file_base' in kwargs.keys():
            self.out_file_base = kwargs['out_file_base']
        if 'val_list' in kwargs.keys():
            self.val_list = kwargs['val_list']
        if 'val_batch' in kwargs.keys():
            self.val_batchsize = kwargs['val_batch']
        pos_weight = torch.FloatTensor([100]).to(device) # give more weightage to positives
        self.bceloss = torch.nn.BCEWithLogitsLoss(reduction='sum',pos_weight=pos_weight)
        
        self.beta = 0.7

    def generate_mapping(self,true_labels,pred_labels):
        # bp()
        true_labels =  true_labels.numpy()
        pred_labels_cpu = pred_labels.detach().numpy()
        # correlation
        # affinity = true_labels.T @ pred_labels_cpu
        # bp()
        # BCE
        logpred_labels = np.log(pred_labels_cpu)
        logpred_labels_inv = np.log(1-pred_labels_cpu)

        affinity = true_labels.T @ logpred_labels + (1-true_labels).T@ logpred_labels_inv
        
        nspks = len(affinity)
        gdict = {}
        for rowid,row in enumerate(affinity):
            tempdict = {}
            for colid,col in enumerate(row):
                tempdict['pred_'+str(colid)] = col
            gdict['true_'+str(rowid)]=tempdict
        # bp()
        mappinglist = algorithm.find_matching(gdict, matching_type = 'max', return_type = 'list')
        transform = np.zeros((nspks,),dtype=int)
        for line in mappinglist:
            pair = line[0]
            transform[int(pair[0].split('_')[-1])] = int(pair[1].split('_')[-1])
        # bp()
        # print(transform)
        pred_labels = pred_labels[:,transform]
        return pred_labels


    def gae_softkmeans_clustering(self,datalist):
        reco2num = open(args.reco2num_list).readlines()
        pair_list = open(args.reco2utt_list).readlines()
        
        for nidx in range(datalist):
            idx = self.datalist[nidx]
            reco2utt = pair_list[idx]
            
            n_clusters = int(reco2num[idx].rsplit()[1])
            
            filename = reco2utt.split()[0]
            # adj, features,clean_ind = load_data_dihard_val_weightadj(args.dataset_str,filename,set='train')
        
            # adj, features,clean_ind = load_data_dihard(args.dataset_str,filename)
            scale = 10
            adj_plda,features,distance_matrix = load_data_dihard_plda_spectral(args.dataset_str,filename,n_clusters,scale=scale)
            features = features/np.linalg.norm(features,axis=1).reshape(-1,1)
            
            adj_plda_no_diagonal = adj_plda - sp.dia_matrix((adj_plda.diagonal()[np.newaxis, :], [0]), shape=adj_plda.shape)
            
            self.model.load_state_dict(torch.load(f'{args.savedmodel}/{filename}.pth'))
            adj_norm = preprocess_graph(adj_plda_no_diagonal)
            hout = self.model(features,adj_norm)
            x = hout.data.cpu().numpy()
            W = soft_k_means(x, K=n_clusters)
            
            
            rttm_basefile = self.out_file_base + '/'+filename+'.rttm'
            outpath_base = self.out_file_base + '/val_der_'+filename
            rttm_gndfile = f'{self.rttm_gnd_val}/{filename}.rttm'
            reco2utt = pair_list[idx]
            print(f'File: {filename}\n',flush=True)
            print('\n---------------Baseline ----------------------------\n')
            overlap = 1                
            base_der = compute_score(rttm_gndfile,rttm_basefile,outpath_base,0)
            if overlap:
                overlap_base_der = compute_score(rttm_gndfile,rttm_basefile,outpath_base,overlap)
                print("\noverlap DER: %.2f" % (overlap_base_der))
            print("\nDER: %.2f" % (base_der))
            # pref = 
            overlap_th = 0.5
            print('overlap_th: ',overlap_th)
            # pref='_selfsup_softmax_weighted_ovpth{}_'.format(overlap_th)
            pref='_selfsup_softmax_ovpth{}_{}epoch_'.format(overlap_th,args.epochs)
            overlap_der = validate_clustering_weighted(args,filename,W,reco2utt,1,1,pref=pref,rttm_gndfile=rttm_gndfile,overlap_th=overlap_th)

        
        
    def gae_spectral_clustering(self,args,datalist,pldamodel=None):
        reco2num = open(args.reco2num_list).readlines()
        pair_list = open(args.reco2utt_list).readlines()
        
        for nidx in range(len(datalist)):
            idx = datalist[nidx]
            reco2utt = pair_list[idx]
            
            n_clusters = int(reco2num[idx].rsplit()[1])
            
            filename = reco2utt.split()[0]
            
            scale = args.scale
            try:
                features = np.load(feats_fname)   
            except:
                feats_fname = f'{args.xvecpath}/xvector.scp'
                features = get_features_filewise(feats_fname,reco2utt.split(" ",1)[1])
            if args.PLDA == 'dihard':
                adj_plda,distance_matrix = load_data_simu_plda(args.dataset_str,filename,n_clusters,scale=scale)
            elif args.PLDA == 'ami':
                _,distance_matrix = load_data_simu_plda_ami(args.dataset_str,filename,n_clusters,scale=scale,xvecpath=args.xvecpath,isadj=None,pldamodel=pldamodel,X=features)
                # distance_matrix, _ = get_PLDA_mat(features,pldamodel,temp_param=args.temp_param)
    
            elif args.PLDA == 'libvox':
                    adj_plda,distance_matrix = load_data_simu_plda_libvox(args.dataset_str,filename,n_clusters,scale=scale,clustering=args.clustering)
            else:
                print('check PLDA argument')
            
            affinity = distance_matrix 
            # affinity = (affinity+ 1.0)/2.0
            # features = features/np.linalg.norm(features,axis=1).reshape(-1,1)
            
            # adj_plda_no_diagonal = adj_plda - sp.dia_matrix((adj_plda.diagonal()[np.newaxis, :], [0]), shape=adj_plda.shape)
                        
            rttm_gndfile = f'{self.rttm_gnd_val}/{filename}.rttm'
           
            print('ground clusters:', n_clusters)
            print(f'File: {filename}\n',flush=True)
            print('\n---------------Baseline ----------------------------\n')
            
            # print('\n---------------after training ----------------------------\n')
            scoretype = 'laplacian'
            if args.threshold is None:
           
                pref=args.pref
                overlap_der = validate_spectral_clustering(filename,affinity,reco2utt,1,1,n_clusters,pref=pref,rttm_gndfile=rttm_gndfile,scoretype=scoretype)
            else:
            #     # pref = f'_threshold{args.threshold}_sigmoid'
                # pref = f'_threshold{args.threshold}_baselinescaled10'
                # pref=f'_baseline_scaled10_AmiPLDA_threshold{args.threshold}_'
                pref=args.pref
                _ = validate_spectral_clustering_threshold(filename,affinity,reco2utt,1,1,threshold=args.threshold,pref=pref,rttm_gndfile=rttm_gndfile)

 
    def gae_pic_clustering_amiplda(self,args,datalist):
        reco2num = open(args.reco2num_list).readlines()
        pair_list = open(args.reco2utt_list).readlines()
        
        for nidx in range(len(datalist)):
            idx = datalist[nidx]
            reco2utt = pair_list[idx]
            
            n_clusters = int(reco2num[idx].rsplit()[1])
            
            filename = reco2utt.split()[0]
           
            scale = 1
            # adj, features,clean_ind,ovpmax2_ind = load_data_simu(args.dataset_str,filename,set='val')
            # overlap_ind = np.arange(features.shape[0])
           
            # overlap_ind[clean_ind] = -1
            # overlap_ind = overlap_ind[overlap_ind>=0]
            # adj_label = adj + sp.eye(adj.shape[0])
            # adj_label[np.ix_(overlap_ind,clean_ind)] = 0.5
            # adj_label[np.ix_(clean_ind,overlap_ind)] = 0.5
            # adj_label[np.ix_(overlap_ind,overlap_ind)] = 0.5
            
            # affinity = adj_label.toarray()
            adj_plda,features,distance_matrix = load_data_simu_plda_ami_feats_knn(args.dataset_str,filename,n_clusters,scale=scale)
            # scale=1
            # _,distance_matrix = load_data_simu_plda(args.dataset_str,filename,n_clusters,scale=scale)
            affinity = distance_matrix
            
            # features = features/np.linalg.norm(features,axis=1).reshape(-1,1)
            
            # adj_plda_no_diagonal = adj_plda - sp.dia_matrix((adj_plda.diagonal()[np.newaxis, :], [0]), shape=adj_plda.shape)
            
            # # # # # # self.model.load_state_dict(torch.load(f'{args.savedmodel}/{filename}.pth'))
            # adj_norm = preprocess_graph(adj_plda_no_diagonal)
            # emb = self.model(features,adj_norm)
            # hidden_emb = emb.data.cpu().numpy()
            # # affinity = np.dot(hidden_emb, hidden_emb.T)
            # affinity = hidden_emb

            # os.system(f'mkdir -p {args.outf}/score_affine')
            # filepath = f'{args.outf}/score_affine/{filename}.npy'
            # np.save(filepath,affinity)
            # bp()

            print(np.min(affinity),np.max(affinity))
            
            # scale=10
            # affinity = expit(affinity)
           
            # affinity[affinity<0] = 0.0
            # affinity =  (affinity+1)/2
            # if np.min(affinity)<0:
            #     affinity = (affinity - np.min(affinity))/(np.max(affinity)-np.min(affinity))
            # affinity = affinity * distance_matrix
            # affinity = distance_matrix + affinity
            # beta = 0.7
            # affinity = affinity * (distance_matrix + affinity)
            # affinity = affinity/np.max(affinity)
            repeat = 0
            if repeat:
                adj_plda_no_diagonal = get_knn(affinity)
                # adj_plda_no_diagonal = adj_plda - sp.dia_matrix((adj_plda.diagonal()[np.newaxis, :], [0]), shape=adj_plda.shape)
                # # # # # self.model.load_state_dict(torch.load(f'{args.savedmodel}/{filename}.pth'))
                adj_norm = preprocess_graph(adj_plda_no_diagonal)
                emb = self.model(features,adj_norm)
                hidden_emb = emb.data.cpu().numpy()
                # affinity = np.dot(hidden_emb, hidden_emb.T)
                affinity = hidden_emb
                affinity = expit(affinity)
                affinity = affinity * (distance_matrix + affinity)
                affinity = affinity/np.max(affinity)
                



            rttm_gndfile = f'{self.rttm_gnd_val}/{filename}.rttm'
            reco2utt = pair_list[idx]
            print('ground clusters:', n_clusters)
            print(f'File: {filename}\n',flush=True)
            print('\n---------------Baseline ----------------------------\n')
            scoretype = 'laplacian'
            overlap = 1                
            pref ='_baseline_'
            # pref = f'_threshold{args.threshold}_baseline_'
            # _ = validate_spectral_clustering_threshold(filename,distance_matrix,reco2utt,1,1,threshold=args.threshold,pref=pref,rttm_gndfile=rttm_gndfile)
            
            rttm_basefile = self.out_file_base + '/'+filename+'.rttm'
            outpath_base = self.out_file_base + '/val_der_'+filename
            try:
                base_der = compute_score(rttm_gndfile,rttm_basefile,outpath_base,0)
                if overlap:
                    overlap_base_der = compute_score(rttm_gndfile,rttm_basefile,outpath_base,overlap)
                    print("\noverlap DER: %.2f" % (overlap_base_der))
                print("\nDER: %.2f" % (base_der))
            except:
                flag=2
                
                overlap_der = validate_path_integral_ami(args,filename,distance_matrix,reco2utt,flag,1,n_clusters,pref=pref,rttm_gndfile=rttm_gndfile)
            # continue
            print('\n---------------after training ----------------------------\n')
            # scoretype = 'laplacian'
            flag=1
            if args.threshold is None:
            #     # pref = f'_spectralthreshold0.4init_sigmoid'
            #     # pref = f'_spectralthreshold0.5init_'
            #     # pref='_baseline_scaled10_'
                # pref='_sup_'
            #     # pref='_sup_scaled10_'
                # pref='_sup_affine_'
                # pref='_sup_affine_sigmoid_'
                # pref=f'_sup_affine_relu_K30_Model{args.epochs}_nb2'
                # pref=f'_sup_affine_relu_K50_Model{args.epochs}'
                K=30
                
                # pref='_baseline_widePLDA_'
                # pref=f'_baseline_AmiPLDA_K{args.K}_z{args.z}_nb{args.nb}_beta{args.beta}'
                # pref=f'_baseline_AmiPLDA_K{args.K}_z{args.z}_nb{args.nb}_beta{args.beta}'
                pref=args.pref
                # pref=f'_sup_affine_gnd_K{K}_z{args.z}_beta{args.beta}_flag{flag}_weighted0.5_overlap_Model{args.epochs}'

            else:
                pref=f'_baseline_AmiPLDA_K{args.K}_z{args.z}_nb{args.nb}_beta{args.beta}_threshold{args.threshold}'
            overlap_der = validate_path_integral_ami(args,filename,affinity,reco2utt,flag,1,n_clusters,pref=pref,rttm_gndfile=rttm_gndfile)
            # overlap_der = validate_path_integral_ami_gnd(args,filename,affinity,reco2utt,flag,1,n_clusters,pref=pref,rttm_gndfile=rttm_gndfile,K=K,clean_ind=clean_ind)
            # overlap_der = validate_gnd(args,filename,affinity,reco2utt,flag,1,n_clusters,pref=pref,rttm_gndfile=rttm_gndfile)

    
    def gae_pic_clustering(self,args,datalist):
        reco2num = open(args.reco2num_list).readlines()
        pair_list = open(args.reco2utt_list).readlines()
        
        for nidx in range(len(datalist)):
            idx = datalist[nidx]
            reco2utt = pair_list[idx]
            
            n_clusters = int(reco2num[idx].rsplit()[1])
            
            filename = reco2utt.split()[0]
           
            scale = 1
            # adj, features,clean_ind,ovpmax2_ind = load_data_simu(args.dataset_str,filename,set='val')
            # overlap_ind = np.arange(features.shape[0])
           
            # overlap_ind[clean_ind] = -1
            # overlap_ind = overlap_ind[overlap_ind>=0]
            # adj_label = adj + sp.eye(adj.shape[0])
            # adj_label[np.ix_(overlap_ind,clean_ind)] = 0.5
            # adj_label[np.ix_(clean_ind,overlap_ind)] = 0.5
            # adj_label[np.ix_(overlap_ind,overlap_ind)] = 0.5
            
            # affinity = adj_label.toarray()
            adj_plda,features,distance_matrix = load_data_simu_plda_feats_knn(args.dataset_str,filename,n_clusters,scale=scale)
            # scale=1
            # _,distance_matrix = load_data_simu_plda(args.dataset_str,filename,n_clusters,scale=scale)
            affinity = distance_matrix
            # features = features/np.linalg.norm(features,axis=1).reshape(-1,1)
            
            # adj_plda_no_diagonal = adj_plda - sp.dia_matrix((adj_plda.diagonal()[np.newaxis, :], [0]), shape=adj_plda.shape)
            
            # # # # # # self.model.load_state_dict(torch.load(f'{args.savedmodel}/{filename}.pth'))
            # adj_norm = preprocess_graph(adj_plda_no_diagonal)
            # emb = self.model(features,adj_norm)
            # hidden_emb = emb.data.cpu().numpy()
            # # affinity = np.dot(hidden_emb, hidden_emb.T)
            # affinity = hidden_emb

            # os.system(f'mkdir -p {args.outf}/score_affine')
            # filepath = f'{args.outf}/score_affine/{filename}.npy'
            # np.save(filepath,affinity)
            # bp()

            print(np.min(affinity),np.max(affinity))
            
            # scale=10
            # affinity = expit(affinity)
           
            # affinity[affinity<0] = 0.0
            # affinity =  (affinity+1)/2
            # if np.min(affinity)<0:
            #     affinity = (affinity - np.min(affinity))/(np.max(affinity)-np.min(affinity))
            # affinity = affinity * distance_matrix
            # affinity = distance_matrix + affinity
            # beta = 0.7
            # affinity = affinity * (distance_matrix + affinity)
            # affinity = affinity/np.max(affinity)
            repeat = 0
            if repeat:
                adj_plda_no_diagonal = get_knn(affinity)
                # adj_plda_no_diagonal = adj_plda - sp.dia_matrix((adj_plda.diagonal()[np.newaxis, :], [0]), shape=adj_plda.shape)
                # # # # # self.model.load_state_dict(torch.load(f'{args.savedmodel}/{filename}.pth'))
                adj_norm = preprocess_graph(adj_plda_no_diagonal)
                emb = self.model(features,adj_norm)
                hidden_emb = emb.data.cpu().numpy()
                # affinity = np.dot(hidden_emb, hidden_emb.T)
                affinity = hidden_emb
                affinity = expit(affinity)
                affinity = affinity * (distance_matrix + affinity)
                affinity = affinity/np.max(affinity)
                



            rttm_gndfile = f'{self.rttm_gnd_val}/{filename}.rttm'
            reco2utt = pair_list[idx]
            print('ground clusters:', n_clusters)
            print(f'File: {filename}\n',flush=True)
            print('\n---------------Baseline ----------------------------\n')
            scoretype = 'laplacian'
            overlap = 1                
            pref ='_baseline_'
            # pref = f'_threshold{args.threshold}_baseline_'
            # _ = validate_spectral_clustering_threshold(filename,distance_matrix,reco2utt,1,1,threshold=args.threshold,pref=pref,rttm_gndfile=rttm_gndfile)
            
            rttm_basefile = self.out_file_base + '/'+filename+'.rttm'
            outpath_base = self.out_file_base + '/val_der_'+filename
            try:
                base_der = compute_score(rttm_gndfile,rttm_basefile,outpath_base,0)
                if overlap:
                    overlap_base_der = compute_score(rttm_gndfile,rttm_basefile,outpath_base,overlap)
                    print("\noverlap DER: %.2f" % (overlap_base_der))
                print("\nDER: %.2f" % (base_der))
            except:
                flag=2
                
                overlap_der = validate_path_integral_ami(args,filename,distance_matrix,reco2utt,flag,1,n_clusters,pref=pref,rttm_gndfile=rttm_gndfile)
            # continue
            print('\n---------------after training ----------------------------\n')
            # scoretype = 'laplacian'
            if args.threshold is None:
            #     # pref = f'_spectralthreshold0.4init_sigmoid'
            #     # pref = f'_spectralthreshold0.5init_'
            #     # pref='_baseline_scaled10_'
                # pref='_sup_'
            #     # pref='_sup_scaled10_'
                # pref='_sup_affine_'
                # pref='_sup_affine_sigmoid_'
                # pref=f'_sup_affine_relu_K30_Model{args.epochs}_nb2'
                # pref=f'_sup_affine_relu_K50_Model{args.epochs}'
                K=30
                flag=2
                # pref=f'_knnpldainit50_sup_affine_K{K}_Model{args.epochs}'
                # pref=f'_knnpldainit_repeat_sup_affine_K{K}_Model{args.epochs}'
                # pref=f'_knnpldainit_sup_affine_K{K}_Model{args.epochs}'
                # pref=f'_knnpldainit_sup_affine_mult_K{K}_Model{args.epochs}'
                # pref=f'_knnpldainit_sup_affine_mult_K{K}_Model{args.epochs}'
                # pref=f'_sup_affine_gnd_K{K}_z{args.z}_beta{args.beta}_flag{flag}_no_overlap_Model{args.epochs}'
                pref='_baseline_widePLDA_'
                # pref=f'_sup_affine_gnd_K{K}_z{args.z}_beta{args.beta}_flag{flag}_weighted0.5_overlap_Model{args.epochs}'

                # pref=f'_knnpldainit_sup_affine_sum_K{K}_Model{args.epochs}'
                # pref=f'_knnpldainit_sup_affine_alone_K{K}_Model{args.epochs}'
                # pref=f'_knnpldainit_sup_affine_K{K}_weighted{beta}_Model{args.epochs}'
                # pref=f'_knnpldainit_sup_affine_sigmoid_K30_Model{args.epochs}'
                # pref=f'_sup_relu_K30_Model{args.epochs}'
                # pref='_sup_affine_relu_adjKNN30_'
                # pref='_sup_affine_1_2shiftscale_'
                # pref='_sup_affine_picorg_'
                # pref='_sup_affine_scaled10_'
                # pref='_sup_affine_scaledadj_'
                # pref='_gnd_plda_no_overlap_K40'
                # pref='_weightedgnd_plda_no_overlap_'
                # pref='_gnd_no_overlap_'
            #     pref='_gnd_'

                if args.clustering == 'spectral':
                    overlap_der = validate_spectral_clustering(filename,affinity,reco2utt,1,1,n_clusters,pref=pref,rttm_gndfile=rttm_gndfile,scoretype=scoretype)
                else:
                    overlap_der = validate_path_integral_ami(args,filename,affinity,reco2utt,flag,1,n_clusters,pref=pref,rttm_gndfile=rttm_gndfile,K=K)
                    # overlap_der = validate_path_integral_ami_gnd(args,filename,affinity,reco2utt,flag,1,n_clusters,pref=pref,rttm_gndfile=rttm_gndfile,K=K,clean_ind=clean_ind)
                    # overlap_der = validate_gnd(args,filename,affinity,reco2utt,flag,1,n_clusters,pref=pref,rttm_gndfile=rttm_gndfile)
    
    def gae_spectral_clustering_angleproto(self,args,datalist):
        reco2num = open(args.reco2num_list).readlines()
        pair_list = open(args.reco2utt_list).readlines()
        
        for nidx in range(len(datalist)):
            idx = datalist[nidx]
            reco2utt = pair_list[idx]
            
            n_clusters = int(reco2num[idx].rsplit()[1])
            
            filename = reco2utt.split()[0]
           
            scale = 1
            # adj, features,clean_ind,ovpmax2_ind = load_data_simu(args.dataset_str,filename,set='val')
            # # adj_label = adj + sp.eye(adj.shape[0])
            # # affinity = adj_label.toarray()
            # overlap_ind = np.arange(features.shape[0])
           
            # overlap_ind[clean_ind] = -1
            # overlap_ind = overlap_ind[overlap_ind>=0]
            # adj = adj.toarray()
            # adj[np.ix_(overlap_ind,clean_ind)] = 0.5
            # adj[np.ix_(clean_ind,overlap_ind)] = 0.5
            # adj[np.ix_(overlap_ind,overlap_ind)] = 0.5
            # adj_plda,features,distance_matrix = load_data_simu_plda_feats_knn(args.dataset_str,filename,n_clusters,scale=scale,myK=50)
            adj_plda,features,distance_matrix = load_data_simu_plda_ami_feats_knn(args.dataset_str,filename,n_clusters,scale=scale,myK=50)
            adj_plda = sparse.csr_matrix(distance_matrix)
            # # bp()
            # # adj = adj.toarray()
            # adj = (distance_matrix+adj)*(adj)
            # adj = adj/adj.max()
            # # adj_plda = sparse.csr_matrix(adj)
            # adj_plda = get_knn(adj,myK=30)
            # adj_plda = adj
            
            # scale=10
            # adj_plda,features,distance_matrix = load_data_simu_plda_feats(args.dataset_str,filename,n_clusters,scale=scale)
            # scale=1
            # _,distance_matrix = load_data_simu_plda(args.dataset_str,filename,n_clusters,scale=scale)
            # affinity = distance_matrix
            features = features/np.linalg.norm(features,axis=1).reshape(-1,1)
            
            adj_plda_no_diagonal = adj_plda - sp.dia_matrix((adj_plda.diagonal()[np.newaxis, :], [0]), shape=adj_plda.shape)
            
            # # # # # self.model.load_state_dict(torch.load(f'{args.savedmodel}/{filename}.pth'))
            adj_norm = preprocess_graph(adj_plda_no_diagonal)
            emb = self.model.cos_sigmoid(features,adj_norm)
            hidden_emb = emb.data.cpu().numpy()
            # affinity = np.dot(hidden_emb, hidden_emb.T)
            affinity = hidden_emb

            # os.system(f'mkdir -p {args.outf}/score_affine_knn50_epochs{args.epochs}')
            # filepath = f'{args.outf}/score_affine_knn50_epochs{args.epochs}/{filename}.npy'
            # np.save(filepath,affinity)
            # bp()
            # continue

            print(np.min(affinity),np.max(affinity))
            
            # scale=10
            affinity = expit(affinity)
            # affinity = affinity *expit(logit(distance_matrix)*10)
            # affinity[affinity<0.6] = 0
          
            # N = affinity.shape[0]
            # neb = args.nb
            # beta1 = args.beta
            # toep = np.abs(np.arange(N).reshape(N,1)-np.arange(N).reshape(1,N))
            # toep[toep>neb] = neb
            # weighting = beta1**(toep)
            # affinity = weighting*affinity
            # affinity[affinity<0] = 0.0
            # affinity =  (affinity+1)/2
            # if np.min(affinity)<0:
            #     affinity = (affinity - np.min(affinity))/(np.max(affinity)-np.min(affinity))
            # affinity = affinity * distance_matrix
            repeat = 0
            if repeat:
                # affinity_n =  affinity
                affinity_n = affinity*(affinity+distance_matrix)
                affinity_n = affinity_n/affinity_n.max()
                adj_plda_no_diagonal = affinity_n - np.eye(affinity_n.shape[0])
                # adj_plda_no_diagonal = get_knn(affinity_n,myK=70)
                # adj_plda_no_diagonal = adj_plda - sp.dia_matrix((adj_plda.diagonal()[np.newaxis, :], [0]), shape=adj_plda.shape)
                # # # # # self.model.load_state_dict(torch.load(f'{args.savedmodel}/{filename}.pth'))
                adj_norm = preprocess_graph(adj_plda_no_diagonal)
                emb = self.model.cos_sigmoid(features,adj_norm)
                # emb = self.model(features,adj_norm)
                hidden_emb = emb.data.cpu().numpy()
                # affinity = np.dot(hidden_emb, hidden_emb.T)
                affinity = hidden_emb
                affinity = expit(affinity)
            repeatwithcluster = 0
            if repeatwithcluster:
                adj_plda_no_diagonal = get_clusterinit_both(affinity,distance_matrix,n_clusters=n_clusters,filename = filename)
                # adj_plda_no_diagonal = adj_plda - sp.dia_matrix((adj_plda.diagonal()[np.newaxis, :], [0]), shape=adj_plda.shape)
                # # # # # self.model.load_state_dict(torch.load(f'{args.savedmodel}/{filename}.pth'))
                adj_norm = preprocess_graph(adj_plda_no_diagonal)
                emb = self.model.cos_sigmoid(features,adj_norm)
                # emb = self.model(features,adj_norm)
                hidden_emb = emb.data.cpu().numpy()
                # affinity = np.dot(hidden_emb, hidden_emb.T)
                affinity = hidden_emb
                affinity = expit(affinity)


            rttm_gndfile = f'{self.rttm_gnd_val}/{filename}.rttm'
            reco2utt = pair_list[idx]
            print('ground clusters:', n_clusters)
            print(f'File: {filename}\n',flush=True)
            print('\n---------------Baseline ----------------------------\n')
            scoretype = 'laplacian'
            overlap = 1                
            pref ='_baseline_'
            # pref = f'_threshold{args.threshold}_baseline_'
            # _ = validate_spectral_clustering_threshold(filename,distance_matrix,reco2utt,1,1,threshold=args.threshold,pref=pref,rttm_gndfile=rttm_gndfile)
            
            rttm_basefile = self.out_file_base + '/'+filename+'.rttm'
            outpath_base = self.out_file_base + '/val_der_'+filename
            try:
                base_der = compute_score(rttm_gndfile,rttm_basefile,outpath_base,0)
                if overlap:
                    overlap_base_der = compute_score(rttm_gndfile,rttm_basefile,outpath_base,overlap)
                    print("\noverlap DER: %.2f" % (overlap_base_der))
                print("\nDER: %.2f" % (base_der))
            except:
                flag=2
                
                overlap_der = validate_path_integral_ami(args,filename,distance_matrix,reco2utt,flag,1,n_clusters,pref=pref,rttm_gndfile=rttm_gndfile)
            # continue
            print('\n---------------after training ----------------------------\n')
            scoretype = 'laplacian'
            if args.threshold is None:
            #     # pref = f'_spectralthreshold0.4init_sigmoid'
            #     # pref = f'_spectralthreshold0.5init_'
            #     # pref='_baseline_scaled10_'
                # pref='_sup_'
            #     # pref='_sup_scaled10_'
                # pref='_sup_affine_'
                # pref='_sup_affine_sigmoid_'
                # pref=f'_sup_affine_relu_K30_Model{args.epochs}_nb2'
                # pref=f'_sup_affine_relu_K50_Model{args.epochs}'
                K=30
                
                # pref=f'_sup_affine_adjscaled_Model{args.epochs}'
                # pref=f'_knnpldainit50_sup_affine_Model{args.epochs}'
                # pref=f'_knnpldainit30plusgndadj_sup_affine_Model{args.epochs}'
                pref=args.pref
                # pref=f'_pldaplusgndadj_sup_affine_Model{args.epochs}'
                # pref=f'_knnpldainit50_sup_affine_temp_Model{args.epochs}'
                # pref=f'_knnpldainit50_repeat_sup_affine_Model{args.epochs}'
                # pref=f'_knnpldainit50_repeat_withpiclabels_sup_affine_Model{args.epochs}'
                # pref=f'_knnpldainit50_repeat_withbothlabels_sup_affine_Model{args.epochs}'

                flag=1
                
                # bp()
                if args.clustering == 'spectral':
                    overlap_der = validate_spectral_clustering(filename,affinity,reco2utt,1,1,n_clusters,pref=pref,rttm_gndfile=rttm_gndfile,scoretype=scoretype)
                else:
                    overlap_der = validate_path_integral_ami(args,filename,affinity,reco2utt,flag,1,n_clusters,pref=pref,rttm_gndfile=rttm_gndfile,K=K)
                    # overlap_der = validate_path_integral_ami_gnd(args,filename,affinity,reco2utt,flag,1,n_clusters,pref=pref,rttm_gndfile=rttm_gndfile,K=K,clean_ind=clean_ind)
                    # overlap_der = validate_gnd(args,filename,affinity,reco2utt,flag,1,n_clusters,pref=pref,rttm_gndfile=rttm_gndfile)
            else:
                pref=f'_knnpldainit50_sup_affine_threshold{args.threshold}_Model{args.epochs}'
                _ = validate_spectral_clustering_threshold(filename,affinity,reco2utt,1,1,threshold=args.threshold,pref=pref,rttm_gndfile=rttm_gndfile)

       
    def gae_ahc_clustering(self,args,datalist):
        reco2num = open(args.reco2num_list).readlines()
        pair_list = open(args.reco2utt_list).readlines()
        
        for nidx in range(len(datalist)):
            idx = datalist[nidx]
            reco2utt = pair_list[idx]
            
            n_clusters = int(reco2num[idx].rsplit()[1])
            
            filename = reco2utt.split()[0]
            # bp()
            scale = args.scale
            if args.PLDA == 'dihard':
                adj_plda,distance_matrix = load_data_simu_plda(args.dataset_str,filename,n_clusters,scale=scale,clustering=args.clustering)
            elif args.PLDA == 'ami':
                adj_plda,distance_matrix = load_data_simu_plda_ami(args.dataset_str,filename,n_clusters,scale=scale,clustering=args.clustering)
            elif args.PLDA == 'libvox':
                    adj_plda,distance_matrix = load_data_simu_plda_libvox(args.dataset_str,filename,n_clusters,scale=scale,clustering=args.clustering)
            else:
                print('check PLDA argument')
            # bp()
            # affinity = logit(distance_matrix)
            affinity = distance_matrix
            
            rttm_gndfile = f'{self.rttm_gnd_val}/{filename}.rttm'
            reco2utt = pair_list[idx]
            print('ground clusters:', n_clusters)
            print(f'File: {filename}\n',flush=True)
            print('\n---------------Baseline ----------------------------\n')
            # overlap = 1                
            # pref = f'_threshold{args.threshold}_baseline_'
            # _ = validate_spectral_clustering_threshold(filename,distance_matrix,reco2utt,1,1,threshold=args.threshold,pref=pref,rttm_gndfile=rttm_gndfile)
            # print('\n---------------after training ----------------------------\n')
            pref=args.pref
            if args.threshold is None:
                print(pref)
                # pref='_baseline_'
                # pref='_baseline_AmiPLDA_'
                # _= validate_ahc(args,filename,affinity,reco2utt,0,1,n_clusters,pref=pref,rttm_gndfile=rttm_gndfile,overlap=1)
            else:
            #    pref=f'_baseline_AmiPLDA_threshold{args.threshold}'
               n_clusters = 1
            _= validate_ahc(args,filename,affinity,reco2utt,0,1,n_clusters,pref=pref,rttm_gndfile=rttm_gndfile,overlap=1)


    def gae_ahc_clustering_angleproto(self,args,datalist):
        reco2num = open(args.reco2num_list).readlines()
        pair_list = open(args.reco2utt_list).readlines()
        
        for nidx in range(len(datalist)):
            idx = datalist[nidx]
            reco2utt = pair_list[idx]
            
            n_clusters = int(reco2num[idx].rsplit()[1])
            
            filename = reco2utt.split()[0]
           
            scale = 1
            # adj_plda,features,distance_matrix = load_data_simu_plda(args.dataset_str,filename,n_clusters,scale=scale)
            # adj_plda,distance_matrix = load_data_simu_plda(args.dataset_str,filename,n_clusters,scale=scale)
            # bp()
            adj_plda,features,distance_matrix = load_data_simu_plda_feats_knn(args.dataset_str,filename,n_clusters,scale=scale,myK=50)

            # affinity = logit(distance_matrix)
            features = features/np.linalg.norm(features,axis=1).reshape(-1,1)
            
            adj_plda_no_diagonal = adj_plda - sp.dia_matrix((adj_plda.diagonal()[np.newaxis, :], [0]), shape=adj_plda.shape)
            
            # # self.model.load_state_dict(torch.load(f'{args.savedmodel}/{filename}.pth'))
            adj_norm = preprocess_graph(adj_plda_no_diagonal)
            emb = self.model.cos_sigmoid(features,adj_norm)

            hidden_emb = emb.data.cpu().numpy()

            affinity = hidden_emb
            
            affinity = expit(affinity)
            affinity = affinity * (distance_matrix + affinity)
            # affinity = affinity * distance_matrix
            # affinity = affinity/np.max(affinity)
            # emb = self.model(features,adj_norm)
            # hidden_emb = emb.data.cpu().numpy()
            # affinity = np.dot(hidden_emb, hidden_emb.T)
            # affinity = expit(affinity)
            # affinity = affinity - np.min(affinity)
            # affinity = affinity/np.max(affinity)
            
            rttm_gndfile = f'{self.rttm_gnd_val}/{filename}.rttm'
            reco2utt = pair_list[idx]
            print('ground clusters:', n_clusters)
            print(f'File: {filename}\n',flush=True)
            # print('\n---------------Baseline ----------------------------\n')
            # overlap = 1                
            # pref = f'_threshold{args.threshold}_baseline_'
            # _ = validate_spectral_clustering_threshold(filename,distance_matrix,reco2utt,1,1,threshold=args.threshold,pref=pref,rttm_gndfile=rttm_gndfile)
            print('\n---------------after training ----------------------------\n')

            if args.threshold is None:

                # pref='_baseline_'
                # pref=f'_knnpldainit50_sup_affine_mult_Model{args.epochs}'
                pref=f'_knnpldainit50_sup_affine_sigmoid_Model{args.epochs}'
                
                _= validate_ahc(args,filename,affinity,reco2utt,0,1,n_clusters,pref=pref,rttm_gndfile=rttm_gndfile,overlap=1)
            else:
                pref = f'_threshold{args.threshold}_sigmoid'
                _ = validate_spectral_clustering_threshold(filename,affinity,reco2utt,1,1,threshold=args.threshold,pref=pref,rttm_gndfile=rttm_gndfile)


    
    def validate_spectral_clustering_val(self,fname,output_new,reco2utt,flag,final,n_clusters=None,pref='_'):
        f = fname
        # print('Spectral Clustering')
        overlap =1
        
    
        forcing_label = 0
        results_dict = defaultdict(np.array)

        # threshold = None
        labelfull=np.arange(output_new.shape[0])
        clusterlen=[1]*len(labelfull)    

        nframe = output_new.shape[0]
        # distance_matrix = (output_new+1)/2
        # output_new = output_new/np.max(abs(output_new))
        # distance_matrix = 1/(1+np.exp(-output_new))
        distance_matrix = output_new
        
        if flag:
            if n_clusters is not None:
                minK = n_clusters
                maxK = n_clusters
                th = 1e-2
            else:
                minK = 1
                maxK = 10
                th = 0.5
        # custom_dist='cosine',scoretype='laplacian',
        labelfull = do_spectral_clustering(distance_matrix,
                                            gauss_blur=0.1,
                                            p_percentile=0.95,
                                            minclusters=minK,
                                            maxclusters=maxK,
                                            truek=4,custom_dist='cosine',
                                            scoretype='laplacian',
                                            stop_eigenvalue=th)
    
        uniq_labels,labelfull=unique(labelfull,True)
        # uniq_labels = np.unique(labelfull)
        n_clusters=len(uniq_labels)
        clusterlen = []
        for lab in uniq_labels:
            clusterlen.append(len(np.where(labelfull==lab)[0]))
        # bp()
        print('clusterlen:',clusterlen,' n_clusters:',n_clusters)
        results_dict[f]=labelfull
        if final:
            out_file=args.outf+'/'+'final_spectral{}rttms/'.format(pref)
            rttm_valfile = out_file+'/valrttm'
            # out_file=args.outf+'/'+'final_spectral_test_rttms/'
        else:
            out_file=args.outf+'/'+'rttms{}/'.format(pref)
            rttm_valfile = out_file+'/valbaserttm'

        mkdir_p(out_file)
        # if not os.path.isdir(out_file):
        #     os.makedirs(out_file)
        # outpath=out_file +'/'+f
        rttm_newfile=out_file+'/'+f+'.rttm'
        
        # rttm_gndfile = args.rttm_ground_path+'/'+f+'.rttm'
        
        outpath = out_file +'/val_der'
        write_results_dict(fname, out_file, results_dict, reco2utt)
        # self.write_results_dict(out_file)
            
        bashCommand="cat {} >> {}".format(rttm_newfile,rttm_valfile)
        os.system(bashCommand)
        # subprocess.check_output(bashCommand,shell=False)
        
        return rttm_valfile, outpath
    
    def find_val_feats(self,args):    
        epoch_val_loss = 0.0
        epoch_roc_curr = 0.0
        epoch_ap_curr = 0.0
        epoch_ovp_roc_curr = 0.0
        base_der = 0
        val_der = 0
        overlap = 1
        total_val_len = torch.tensor(0).to(device)
        self.pair_list = open(args.reco2utt_list).readlines()
        
        # bp()
        
        for idx, data in enumerate(tqdm(self.valloader)):
            
            val_features,adj_val_label,adj_val_norm,n_clusters,n_nodes,temp_mask,ovp_labels = data
            
            adj_val_norm = adj_val_norm.to_sparse()
            temp_mask = temp_mask.to_sparse()

            adj_val = adj_val_label.numpy() - np.eye(adj_val_label.shape[0])
            adj_val = sp.coo_matrix(adj_val)

            val_features = val_features.to(device)
            adj_val_label = adj_val_label.to(device)
            adj_val_norm = adj_val_norm.to(device)
            n_nodes = n_nodes.to(device)
            temp_mask = temp_mask.to(device)
            ovp_labels = ovp_labels.to(device)
            # base_der += validate_spectral_clustering(filename,output_new,reco2utt,1,0,n_clusters)
            # rttm_basefile, outpath = validate_spectral_clustering_val(filename,output_new,reco2utt,1,1,n_clusters,pref='_baseline_')

            with torch.no_grad():
               emb,z = self.model(val_features, adj_val_norm)
            # recovered_up = recovered[np.triu_indices(n_nodes,k=1)]
            val_loss = self.custom_loss(emb, mask=adj_val_label,temp_mask = temp_mask)
            bceloss = self.bceloss(z,ovp_labels)

            total_loss = self.beta*val_loss + (1-self.beta)*bceloss
            total_mean = total_loss/n_nodes.sum()
            epoch_val_loss +=total_loss.item()

            hidden_emb = emb.data.cpu().numpy()
            label_z = expit(z.data.cpu().numpy())
            true_labels = ovp_labels.data.cpu().numpy()
            
            # if n_clusters > 1:
            try:
                roc_curr, ap_curr,recall = get_roc_score_modified(hidden_emb, adj_val)
                ovp_roc_curr, ovp_ap_curr,ovp_recall = get_roc_score_overlap(label_z,true_labels)
            except:
                roc_curr = 1.0
                ap_curr = 1.0
                recall = 1.0
            epoch_roc_curr +=roc_curr
            epoch_ap_curr +=ap_curr
            epoch_ovp_roc_curr +=ovp_roc_curr
            print('Validation loss: {:.5f} roc_score: {:.2f} precision: {:.2f} recall {:.2f}'.format(total_mean.item(),roc_curr*100,ap_curr*100,recall*100))
            print('ovp_roc_score: {:.2f} ovp_precision: {:.2f} ovp_recall {:.2f}'.format(ovp_roc_curr*100,ovp_ap_curr*100,ovp_recall*100))
            
            adj_rec = np.dot(hidden_emb, hidden_emb.T)
            # bp()
            output_new =expit(adj_rec)

            filecount = 0
            curr_batchsize = len(n_nodes)
            total_val_len += n_nodes.sum()

            start = 0
            # bp()
            while filecount < curr_batchsize:
                index = idx * self.val_batchsize + filecount
                new_index = self.val_list[index]
                reco2utt = self.pair_list[new_index]
                filename = reco2utt.split()[0]
                print('validation filename: ',filename)
                adj_plda,output_plda = load_data_dihard_plda(args.dataset_str,filename,device,n_clusters[filecount])
            
                if n_clusters[filecount] > 1:
                    roc_curr, ap_curr,recall = get_roc_score_modified(adj_plda, adj_val,baseline=1)
                else:
                    roc_curr = 1.0
                    ap_curr = 1.0
                    recall = 1.0
                print('Baseline ROC acc: {:.2f} Precision: {:.2f} Recall {:.2f}'.format(roc_curr*100,ap_curr*100,recall*100))

                end = start + n_nodes[filecount]
                affinity = output_new[start:end,start:end]
                rttm_valfile, outpath = self.validate_spectral_clustering_val(filename,affinity,reco2utt,1,1,n_clusters[filecount])
                start = end
                filecount += 1
            del emb
            del z
            torch.cuda.empty_cache()
        print('baseline DER')
        
        # bashCommand="cat {}.txt | grep OVERALL |awk '{{print $4}}'".format(outpath)
        # output=subprocess.check_output(bashCommand,shell=True)
        # base_der = float(output.decode('utf-8').rstrip())
        
        rttm_basefile = self.out_file_base + '/valrttm'
        outpath_base = self.out_file_base + '/val_der'
        base_der = compute_score(self.rttm_gnd_val,rttm_basefile,outpath_base,0)

        if overlap:
            overlap_base_der = compute_score(self.rttm_gnd_val,rttm_basefile,outpath_base,overlap)
            print("\noverlap DER: %.2f" % (overlap_base_der))
        print("\nDER: %.2f" % (base_der))
        
        print('validation DER')
        try:
            val_der = compute_score(self.rttm_gnd_val,rttm_valfile,outpath,0)
            if overlap:
                overlap_val_der = compute_score(self.rttm_gnd_val,rttm_valfile,outpath,overlap)
                print("\noverlap DER: %.2f" % (overlap_val_der))
            print("\nDER: %.2f" % (val_der))
        except:
            val_der = 100
            overlap_val_der = 100
            
        epoch_roc_curr /=len(self.valloader)
        epoch_ap_curr /=len(self.valloader)

        epoch_val_loss /=total_val_len
        epoch_ovp_roc_curr /= len(self.valloader)
        # base_der /=len(val_list)
        # val_der /=len(val_list)
        os.remove(rttm_valfile)
       
        self.results['val_roc'].append(epoch_roc_curr)
        self.results['val_ap'].append(epoch_ap_curr)
        self.results['val_loss'].append(epoch_val_loss)
        self.results['baseline_DER'].append(overlap_base_der)
        self.results['val_DER'].append(overlap_val_der)
        self.results['val_ovp_roc'].append(epoch_ovp_roc_curr)
        # results['val_filename'].append(filename)
        # return results

def generate_rttms(args):
    
    # rttm_gndfile_val = args.outf+'/final_spectral_rttms/ref_val.rttm'
    # rttm_gndfile_val= 'lists/dihard_dev_2020_track1_fbank_jhu/rttm_val'
    epoch=49
    feat_dim = 512
    outfilepath =  f'{args.outf}/outscores/'
    cmd = f'mkdir -p {outfilepath}'
    os.system(cmd)


    if args.model == 'gcn_ae_norm':
        # model = GCNModelAE_norm_overlap(feat_dim, args.hidden1, args.hidden2, args.dropout)
        model = GCNModelAE_norm(feat_dim, args.hidden1, args.hidden2, args.dropout)
    else:
        model = GCNModelAE(feat_dim, args.hidden1, args.hidden2, args.dropout, device = device )

    model = model.to(device)
    cpupath = args.outf+'/models/modelcpu_snapshot_{}.pth'.format(epoch+1)
    if os.path.isfile(cpupath):
        torchmodel = torch.load(cpupath)
        model.load_state_dict(torchmodel)
        model = model.to(device)
    else:
        torchmodel = torch.load(args.outf+'/models/model_snapshot_{}.pth'.format(epoch+1))
        model.load_state_dict(torchmodel)
        model = model.to(device)
        torch.save(model.state_dict(),cpupath)
        return
   

    model.eval()
    pair_list = open(args.reco2utt_list).readlines()

    nfile = len(pair_list)
    # filepath = 'lists/{}/val.list'.format(args.dataset_str)
    filepath = 'lists/{}/{}'.format(args.dataset_str,args.splitpath)
    # bp()
    if os.path.isfile(filepath):
        val_list = np.genfromtxt(filepath,dtype=float).astype(int).reshape(-1,)
        # bp()
    # bp()
    base_der = 0
    val_der = 0
    overlap = 1
    weighted = 1
    for filecount in val_list:
        reco2utt = pair_list[filecount]
        reco2num = open(args.reco2num_list).readlines()
        nprime = int(reco2num[filecount].rsplit()[1])
        # filename = 'DH_DEV_0002'
        filename = reco2utt.split()[0]
        print(' filename: ',filename)
        n_clusters = nprime
        
        adj_plda,output_new = load_data_dihard_plda(args.dataset_str,filename,device,n_clusters)
        adj_val, val_features, clean_ind, _ = load_data_dihard_val(args.dataset_str,filename,device)
        n_nodes, feat_dim = val_features.shape
        val_features = val_features.to(device)
        # continue
        # Store original adjacency matrix (without diagonal entries) for later
        adj_orig = adj_plda
        adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
        adj_orig.eliminate_zeros()

        adj_val_norm = preprocess_graph(adj_orig).to(device)
        
        # base_der += validate_spectral_clustering(filename,output_new,reco2utt,1,0,n_clusters)
        # rttm_basefile, outpath = validate_spectral_clustering_val(filename,output_new,reco2utt,1,1,n_clusters,pref='_baseline_train_')
        if weighted:
            scoretype='softkmeans'
            # scoretype='cmeans'
          
            # pref='_ground{}centrality_val_'.format(scoretype)
            pref='_baseline{}_val_'.format(scoretype)

            overlap_th = 0.8
            pref = f'{pref}{overlap_th}mx_'
            # pref = f'{pref}{overlap_th}mx_th{args.threshold}'
            print('groundtruth speakers: ',n_clusters)

            # output_new = output_new * adj_val.toarray()
            # rttm_basefile, outpath = validate_spectral_clustering_weighted_val(filename,output_new,reco2utt,1,1,n_clusters,pref=pref,scoretype=scoretype,overlap_th=overlap_th)
            
            # using cent (clustering coefficient*(1-betweeness)) higher value shows densely connected node used for overlap prediction
            # rttm_basefile, outpath = validate_spectral_clustering_weighted_val(filename,adj_val.toarray(),reco2utt,1,1,n_clusters,pref=pref,scoretype=scoretype,overlap_th=overlap_th,cent=cent)

            # use threshold
            # rttm_basefile, outpath = validate_spectral_clustering_weighted_val(filename,adj_val.toarray(),reco2utt,1,1,pref=pref,scoretype=scoretype,overlap_th=overlap_th)

        else:
            # pref='_plda_knn_valspectralcentrality_val_'
            # pref='_plda_knn_valspectralcentralityth1.0_val_'
            # pref = f'_ground_valspectral_th{args.threshold}'
            
            # pref = f'_groundmaskedbaseline_valspectral_'
            # pref = f'_pldamaskedbaseline_valspectral_'

            print('groundtruth speakers: ',n_clusters)
            # output_new = output_new * adj_val.toarray()
            # output_new = adj_plda.toarray() 

            # rttm_basefile, outpath = validate_spectral_clustering_val(filename,adj_val.toarray(),reco2utt,1,1,n_clusters,pref=pref)
            
            # use threshold
            # rttm_basefile, outpath = validate_spectral_clustering_val(filename,adj_val.toarray(),reco2utt,1,1,pref=pref)

        # continue
        
        if 0: #args.model == 'gcn_ae_norm':
            emb, z = model(val_features, adj_val_norm)

            hidden_emb = emb.data.cpu().numpy()
            label_z = expit(z.data.cpu().numpy())
        else:
            emb = model(val_features, adj_val_norm)

            hidden_emb = emb.data.cpu().numpy()

        
        adj_rec = np.dot(hidden_emb, hidden_emb.T)
       
        output_new =expit(adj_rec)
       
        # output_new = adj_rec

        # output_new = 0.7*output_new + 0.3*adj_plda.toarray()
        # th = np.mean(output_new)
        # th = 0.5
        overlap_th = 0.5
        scoretype='softkmeans'
        pref = '_val{}_overlapth{}'.format(scoretype,overlap_th)
        # pref = '_train{}_overlapth{}'.format(scoretype,overlap_th)
        # pref = '_valgnn_using_sigpldaKNN_init_'
        # print(pref)
        # output_new = output_new * adj_val.toarray()
        
        # output_new = KNN(output,K=30)
        # output_new = (output_new + output_new.T)/2

        # G = nx.from_numpy_matrix(output_new)
        # cent=nx.clustering(G)
        # bet=nx.betweenness_centrality(G)
        with open(f'{outfilepath}/{filename}.pkl','wb') as fadj:
            pkl.dump(output_new,fadj)
        flag = 1
        if args.clustering == 'spectral':
            
            # rttm_valfile, outpath = validate_spectral_clustering_val(filename,output_new,reco2utt,1,1,n_clusters,pref=pref)
            
            # cent = 1 - label_z
            cent = None
            rttm_valfile, outpath = validate_spectral_clustering_weighted_val(filename,output_new,reco2utt,1,1,n_clusters,pref=pref,scoretype=scoretype,overlap_th=overlap_th,cent=cent)
        elif args.clustering == 'pic' :
            rttm_valfile, outpath = validate_path_integral(filename,output_new,reco2utt,flag,1,n_clusters,pref=pref)
        else:
            rttm_valfile, outpath = validate_ahc(filename,output_new,reco2utt,flag,1,n_clusters,pref=pref)
    
    
    # print('baseline DER')
    # base_der = compute_score(rttm_gndfile_val,rttm_basefile,outpath,0)

    # if overlap:
    #     overlap_base_der = compute_score(rttm_gndfile_val,rttm_basefile,outpath,overlap)
    #     print("\noverlap DER: %.2f" % (overlap_base_der))
    # print("\nDER: %.2f" % (base_der))
    
    # print('validation DER')
          
    # val_der = compute_score(rttm_gndfile_val,rttm_valfile,outpath,0)
    # if overlap:
    #     overlap_val_der = compute_score(rttm_gndfile_val,rttm_valfile,outpath,overlap)
    #     print("\noverlap DER: %.2f" % (overlap_val_der))
    # print("\nDER: %.2f" % (val_der))

    # os.remove(rttm_valfile)
    # os.remove(rttm_basefile)

def generate_rttms_weighted(args):
    
    # rttm_gndfile_val = args.outf+'/final_spectral_rttms/ref_val.rttm'
    # rttm_gndfile_val= 'lists/dihard_dev_2020_track1_fbank_jhu/rttm_val'
    epoch=3
    feat_dim = 512
    outfilepath =  f'{args.outf}/outscores/'
    cmd = f'mkdir -p {outfilepath}'
    os.system(cmd)


    if args.model == 'gcn_ae_norm':
        model = GCNModelAE_norm_overlap(feat_dim, args.hidden1, args.hidden2, args.dropout)
    else:
        model = GCNModelAE(feat_dim, args.hidden1, args.hidden2, args.dropout, device = device )

    model = model.to(device)
    cpupath = args.outf+'/models/modelcpu_snapshot_{}.pth'.format(epoch+1)
    if os.path.isfile(cpupath):
        torchmodel = torch.load(cpupath)
        model.load_state_dict(torchmodel)
        model = model.to(device)
    else:
        torchmodel = torch.load(args.outf+'/models/model_snapshot_{}.pth'.format(epoch+1))
        model.load_state_dict(torchmodel)
        model = model.to(device)
        torch.save(model.state_dict(),cpupath)
        return
   

    model.eval()
    pair_list = open(args.reco2utt_list).readlines()

    nfile = len(pair_list)
    # filepath = 'lists/{}/val.list'.format(args.dataset_str)
    filepath = 'lists/{}/{}'.format(args.dataset_str,args.splitpath)
    # bp()
    if os.path.isfile(filepath):
        val_list = np.genfromtxt(filepath,dtype=float).astype(int).reshape(-1,)
        # bp()
    # bp()
    base_der = 0
    val_der = 0
    overlap = 1
    weighted = 0
    for filecount in val_list:
        reco2utt = pair_list[filecount]
        reco2num = open(args.reco2num_list).readlines()
        nprime = int(reco2num[filecount].rsplit()[1])
        # filename = 'DH_DEV_0002'
        filename = reco2utt.split()[0]
        print(' filename: ',filename)
        n_clusters = nprime
        # bp()
        adj_plda,output_new = load_data_dihard_plda(args.dataset_str,filename,device,n_clusters)
        adj_val, val_features, clean_ind, org_cent = load_data_dihard_val_weightadj(args.dataset_str,filename,device)
        n_nodes, feat_dim = val_features.shape
        print('#overlaps:',n_nodes-len(clean_ind) )
        val_features = val_features.to(device)
        # continue
        # Store original adjacency matrix (without diagonal entries) for later
        adj_orig = adj_plda
        adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
        adj_orig.eliminate_zeros()

        adj_val_norm = preprocess_graph(adj_orig).to(device)
        
        # base_der += validate_spectral_clustering(filename,output_new,reco2utt,1,0,n_clusters)
        # rttm_basefile, outpath = validate_spectral_clustering_val(filename,output_new,reco2utt,1,1,n_clusters,pref='_baseline_train_')
        if weighted:
            scoretype='softkmeans'
            # scoretype='cmeans'
          
            # pref='_ground{}centrality_val_'.format(scoretype)
            pref='_groundweighted{}centrality_val_'.format(scoretype)
            # pref='_baseline{}_val_'.format(scoretype)

            overlap_th = 0.8
            print('unique cent:',np.unique(org_cent))

            pref = f'{pref}{overlap_th}mx_'
            # pref = f'{pref}{overlap_th}mx_th{args.threshold}'
            print('groundtruth speakers: ',n_clusters)

            # output_new = output_new * adj_val.toarray()
            # rttm_basefile, outpath = validate_spectral_clustering_weighted_val(filename,output_new,reco2utt,1,1,n_clusters,pref=pref,scoretype=scoretype,overlap_th=overlap_th)
            
            # using cent (clustering coefficient*(1-betweeness)) higher value shows densely connected node used for overlap prediction
            # rttm_basefile, outpath = validate_spectral_clustering_weighted_val(filename,adj_val.toarray(),reco2utt,1,1,n_clusters,pref=pref,scoretype=scoretype,overlap_th=overlap_th,cent=org_cent)

            # use threshold
            # rttm_basefile, outpath = validate_spectral_clustering_weighted_val(filename,adj_val.toarray(),reco2utt,1,1,pref=pref,scoretype=scoretype,overlap_th=overlap_th)

        else:
            # pref='_plda_knn_valspectralcentrality_val_'
            # pref='_plda_knn_valspectralcentralityth1.0_val_'
            # pref = f'_ground_valspectral_th{args.threshold}'
            
            # pref = f'_groundmaskedbaseline_valspectral_'
            # pref = f'_pldamaskedbaseline_valspectral_'
            

            print('groundtruth speakers: ',n_clusters)
            # output_new = output_new * adj_val.toarray()
            # output_new = adj_plda.toarray() 
            
            # rttm_basefile, outpath = validate_spectral_clustering_val(filename,adj_val.toarray(),reco2utt,1,1,n_clusters,pref=pref)
            rttm_gnd_val = '/home/prachis/Dihard_2020/LDC2020E12_Third_DIHARD_Challenge_Development_Data/data/rttm'
            rttm_gndfile = f'{rttm_gnd_val}/{filename}.rttm'
            pref=f'_val_ground_weighted_'
            flag = 1
            _ = validate_spectral_clustering(filename,adj_val.toarray(),reco2utt,flag,1,n_clusters,pref=pref,rttm_gndfile=rttm_gndfile)
            flag = 2
            _ = validate_path_integral(args,filename,adj_val.toarray(),reco2utt,flag,1,n_clusters,pref=pref,rttm_gndfile=rttm_gndfile)

            # use threshold
            # rttm_basefile, outpath = validate_spectral_clustering_val(filename,adj_val.toarray(),reco2utt,1,1,pref=pref)

        continue
        
        if args.model == 'gcn_ae_norm':
            emb, z = model(val_features, adj_val_norm)

            hidden_emb = emb.data.cpu().numpy()
            label_z = expit(z.data.cpu().numpy())
        else:
            recovered,emb = model(val_features, adj_val_norm)

            hidden_emb = emb.data.cpu().numpy()

        
        adj_rec = np.dot(hidden_emb, hidden_emb.T)
       
        output_new =expit(adj_rec)
       
        # output_new = adj_rec

        # output_new = 0.7*output_new + 0.3*adj_plda.toarray()
        # th = np.mean(output_new)
        # th = 0.5
        overlap_th = 0.4
        scoretype='softkmeans'
        pref = '_val{}_overlapth{}'.format(scoretype,overlap_th)
        # pref = '_train{}_overlapth{}'.format(scoretype,overlap_th)
        # pref = '_valgnn_using_sigpldaKNN_init_'
        # print(pref)
        # output_new = output_new * adj_val.toarray()
        
        # output_new = KNN(output,K=30)
        # output_new = (output_new + output_new.T)/2

        # G = nx.from_numpy_matrix(output_new)
        # cent=nx.clustering(G)
        # bet=nx.betweenness_centrality(G)
        with open(f'{outfilepath}/{filename}.pkl','wb') as fadj:
            pkl.dump(output_new,fadj)
        flag = 1
        if args.clustering == 'spectral':
            
            # rttm_valfile, outpath = validate_spectral_clustering_val(filename,output_new,reco2utt,1,1,n_clusters,pref=pref)
            
            # cent = 1 - label_z
            cent = None
            rttm_valfile, outpath = validate_spectral_clustering_weighted_val(filename,output_new,reco2utt,1,1,n_clusters,pref=pref,scoretype=scoretype,overlap_th=overlap_th,cent=cent)
        elif args.clustering == 'pic' :
            rttm_valfile, outpath = validate_path_integral(filename,output_new,reco2utt,flag,1,n_clusters,pref=pref)
        else:
            rttm_valfile, outpath = validate_ahc(filename,output_new,reco2utt,flag,1,n_clusters,pref=pref)
    
    
    # print('baseline DER')
    # base_der = compute_score(rttm_gndfile_val,rttm_basefile,outpath,0)

    # if overlap:
    #     overlap_base_der = compute_score(rttm_gndfile_val,rttm_basefile,outpath,overlap)
    #     print("\noverlap DER: %.2f" % (overlap_base_der))
    # print("\nDER: %.2f" % (base_der))
    
    # print('validation DER')
          
    # val_der = compute_score(rttm_gndfile_val,rttm_valfile,outpath,0)
    # if overlap:
    #     overlap_val_der = compute_score(rttm_gndfile_val,rttm_valfile,outpath,overlap)
    #     print("\noverlap DER: %.2f" % (overlap_val_der))
    # print("\nDER: %.2f" % (val_der))

    # os.remove(rttm_valfile)
    # os.remove(rttm_basefile)

def validate_spectral_clustering(fname,output_new,reco2utt,flag,final,n_clusters=None,pref='_',rttm_gndfile=None,overlap=1,scoretype='laplacian'):
    f = fname
    print('Spectral Clustering')
    # overlap =1

    results_dict = defaultdict(np.array)

    # threshold = None
    labelfull=np.arange(output_new.shape[0])
    clusterlen=[1]*len(labelfull)    

    nframe = output_new.shape[0]
    # distance_matrix = (output_new+1)/2
    # output_new = output_new/np.max(abs(output_new))
    # distance_matrix = 1/(1+np.exp(-output_new))
    distance_matrix = output_new
    
    if flag:
        if n_clusters is not None:
            minK = n_clusters
            maxK = n_clusters
            th = 1e-2
        else:
            minK = 1
            maxK = 10
            th = 0.5
    
    
    if scoretype is None:
        # custom_dist='cosine'
        labelfull = do_spectral_clustering(distance_matrix,
                                            gauss_blur=0.1,
                                            p_percentile=0.95,
                                            minclusters=minK,
                                            maxclusters=maxK,
                                            truek=4,custom_dist='cosine',
                                            stop_eigenvalue=th)
    else:
        # custom_dist='cosine',scoretype='laplacian',
        labelfull = do_spectral_clustering(distance_matrix,
                                            gauss_blur=0.1,
                                            p_percentile=0.95,
                                            minclusters=minK,
                                            maxclusters=maxK,
                                            truek=4,
                                            scoretype=scoretype,
                                            custom_dist='cosine',
                                            stop_eigenvalue=th)
   
    uniq_labels,labelfull=unique(labelfull,True)

    labelfull = labelfull.reshape(-1,1)
    adj_bool = labelfull == labelfull.T
    adj_label = np.ones((nframe,nframe))*adj_bool
    # plt.imshow(adj_label)
    # plt.title('{}_spec_trained'.format(f))
    # plt.savefig('{}/testpic_images/{}/spec_trained.png'.format(args.outf,f))

    # uniq_labels = np.unique(labelfull)
    n_clusters=len(uniq_labels)
    clusterlen = []
    for lab in uniq_labels:
        clusterlen.append(len(np.where(labelfull==lab)[0]))
    # bp()
    print('clusterlen:',clusterlen,' n_clusters:',n_clusters)
    results_dict[f]=labelfull
    if final:
        out_file=args.outf+'/'+'final_spectral{}rttms/'.format(pref)
        # out_file=args.outf+'/'+'final_spectral_test_rttms/'
    else:
        out_file=args.outf+'/'+'rttms/'
    # if not os.path.isdir(out_file):
    #     os.makedirs(out_file)
    mkdir_p(out_file)
    outpath=out_file +'/'+f
    rttm_newfile=out_file+'/'+f+'.rttm'
    
    # rttm_gndfile = args.rttm_ground_path+'/'+f+'.rttm'
    write_results_dict(fname, out_file, results_dict, reco2utt)
    # self.write_results_dict(out_file)
    # bp()
    der = compute_score(rttm_gndfile,rttm_newfile,outpath,0)
    if overlap:
        overlap_der = compute_score(rttm_gndfile,rttm_newfile,outpath,1)
        print("\n%s  overlap DER: %.2f" % (fname, overlap_der))
    print("\n%s DER: %.2f" % (fname, der))
    return overlap_der

def validate_spectral_clustering_threshold(fname,output_new,reco2utt,flag,final,threshold=None,pref='_',rttm_gndfile=None,overlap=1,scoretype='laplacian'):
    f = fname
    print('Spectral Clustering')
    # overlap =1

    results_dict = defaultdict(np.array)

    # threshold = None
    labelfull=np.arange(output_new.shape[0])
    clusterlen=[1]*len(labelfull)    

    nframe = output_new.shape[0]
    # distance_matrix = (output_new+1)/2
    # output_new = output_new/np.max(abs(output_new))
    # distance_matrix = 1/(1+np.exp(-output_new))
    distance_matrix = output_new
    print(f'threshold:{threshold}')
    if flag:
        minK = 1
        maxK = 10
        th = threshold
    # custom_dist='cosine',scoretype='laplacian',
    if scoretype is None:
        # custom_dist='cosine'
        labelfull = do_spectral_clustering(distance_matrix,
                                            gauss_blur=0.1,
                                            p_percentile=0.95,
                                            minclusters=minK,
                                            maxclusters=maxK,
                                            truek=4,custom_dist='cosine',
                                            stop_eigenvalue=th)
    else:
        # custom_dist='cosine',scoretype='laplacian',
        labelfull = do_spectral_clustering(distance_matrix,
                                            gauss_blur=0.1,
                                            p_percentile=0.95,
                                            minclusters=minK,
                                            maxclusters=maxK,
                                            truek=4,
                                            scoretype=scoretype,
                                            custom_dist='cosine',
                                            stop_eigenvalue=th)
   
    uniq_labels,labelfull=unique(labelfull,True)

    labelfull = labelfull.reshape(-1,1)
    adj_bool = labelfull == labelfull.T
    adj_label = np.ones((nframe,nframe))*adj_bool
    # plt.imshow(adj_label)
    # plt.title('{}_spec_trained'.format(f))
    # plt.savefig('{}/testpic_images/{}/spec_trained.png'.format(args.outf,f))

    # uniq_labels = np.unique(labelfull)
    n_clusters=len(uniq_labels)
    clusterlen = []
    for lab in uniq_labels:
        clusterlen.append(len(np.where(labelfull==lab)[0]))
    # bp()
    print('clusterlen:',clusterlen,' n_clusters:',n_clusters)
    results_dict[f]=labelfull.reshape(-1,)
    if final:
        out_file=args.outf+'/'+'final_spectral{}rttms/'.format(pref)
        # out_file=args.outf+'/'+'final_spectral_test_rttms/'
    else:
        out_file=args.outf+'/'+'rttms/'
    # if not os.path.isdir(out_file):
    #     os.makedirs(out_file)
    mkdir_p(out_file)
    outpath=out_file +'/'+f
    rttm_newfile=out_file+'/'+f+'.rttm'
    
    # rttm_gndfile = args.rttm_ground_path+'/'+f+'.rttm'
    write_results_dict(fname, out_file, results_dict, reco2utt)
    # self.write_results_dict(out_file)
    # bp()
    der = compute_score(rttm_gndfile,rttm_newfile,outpath,0)
    if overlap:
        overlap_der = compute_score(rttm_gndfile,rttm_newfile,outpath,1)
        print("\n%s  overlap DER: %.2f" % (fname, overlap_der))
    print("\n%s DER: %.2f" % (fname, der))
    return overlap_der

def generate_adjacency_matrix(args):
    epoch=3
    feat_dim = 512
    # Imagefold = f'{args.outf}/Imagedir_separatecolorbar'
    Imagefold = f'{args.outf}/Imagedir_separatecolorbar_pldath0.5_v5'
    # Imagefold = f'{args.outf}/Imagedir_withPLDA'
    # Imagefold = f'{args.outf}/Imagedir_binarize'

    os.system('mkdir -p {}'.format(Imagefold))

    if args.model == 'gcn_vae':
        model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
    elif args.model == 'gcn_ae_linear':
        model = GCNModelAE_linear(feat_dim, args.hidden1, args.hidden2, args.dropout, device = device )
    else:
        model = GCNModelAE_norm_overlap(feat_dim, args.hidden1, args.hidden2, args.dropout)

    model = model.to(device)
    # model.load_state_dict(torch.load(args.outf+'/models/modelcpu_snapshot_{}.pth'.format(epoch+1)))
    # model.load_state_dict(torch.load(args.savedmodel))
    model.eval()
    pair_list = open(args.reco2utt_list).readlines()

    nfile = len(pair_list)
    filepath = 'lists/{}/val.list'.format(args.dataset_str)
    # bp()
    if os.path.isfile(filepath):
        val_list = np.genfromtxt(filepath,dtype=float).astype(int)

    # num_train = int(nfile *0.7)
    # train_list = nfile_list[:num_train]
    # val_list = nfile_list[num_train:]
    filecount = random.choice(val_list)
    # val_list =  np.array([157,145,186,244])
    # val_list = np.array([143,118,234,157,145,186,244])
    val_list = np.array([143,123,163,151,54,234])
    for filecount in tqdm(val_list):
        reco2utt = pair_list[filecount]
        reco2num = open(args.reco2num_list).readlines()
        nprime = int(reco2num[filecount].rsplit()[1])
        filename = reco2utt.split()[0]
        model.load_state_dict(torch.load(f'{args.savedmodel}/{filename}.pth'))
        # if filename != 'DH_DEV_0199':
        #     continue
        print('validation filename: {} # spks: {}'.format(filename,nprime))
        # continue
        n_clusters = nprime
        # graphdatapath = 'exp/graphs_overlap_v2/{}.pkl'.format(filename)

        # output_new here is baseline plda after sigmoid
        adj_val, val_features,_,_ = load_data_dihard_val_weightadj(args.dataset_str,filename,set='val')
       
        adj_plda,output_plda = load_data_dihard_plda(args.dataset_str,filename,device,n_clusters)
        # adj_plda, val_features,output_plda = load_data_dihard_plda_org(args.dataset_str,filename,device,n_clusters)

        # adj_val, _, _ = load_data_dihard(args.dataset_str,filename,device)
        n_nodes, feat_dim = val_features.shape
        val_features = val_features.to(device)
        
        # Store original adjacency matrix (without diagonal entries) for later

        adj_orig = adj_plda
        adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
        adj_orig.eliminate_zeros()
        # adj_cluster = adj_plda + sp.eye(adj_plda.shape[0])
        # output_new =adj_cluster.toarray() 
        
        # adj_triu = sp.triu(adj)
        # adj_tuple = sparse_to_tuple(adj_triu)
        # val_edges = adj_tuple[0]
        adj_val_norm = preprocess_graph(adj_orig).to(device)
               
        # Reference
        adj_val_label = adj_val + sp.eye(adj_val.shape[0])
        
        # adj_label = sparse_to_tuple(adj_label)
        # adj_val_label = torch.FloatTensor(adj_val_label.toarray()).to(device)
       
        emb,_ = model(val_features, adj_val_norm)
        # emb,_ = model(val_features, sparse_mx_to_torch_sparse_tensor(adj_plda)) # v4

        hidden_emb = emb.data.cpu().numpy()
       
        adj_rec = np.dot(hidden_emb, hidden_emb.T)
        # bp()
        # model output
        output_new =expit(adj_rec)
        imagepath=f'{Imagefold}/{filename}.png'
        # bp()
        # save_images(output_plda,adj_val_label.toarray(), output_new,imagepath)
        # save_images(adj_val_norm.to_dense().numpy(),adj_val_label.toarray(), output_new,imagepath) # v2
        # save_images(adj_plda.toarray(),adj_val_label.toarray(), output_new,imagepath) # v3
        output_old = output_new.copy()
        output_new = gaussian_filter(output_new, sigma=5.0)
        diff = output_old - output_new
       
        save_images(adj_plda.toarray(),adj_val_label.toarray(), output_new,imagepath) # v5
        
def save_images(Image1,Image2,Image3,imagepath):
    # create figure
    # fig = plt.figure()
    
    # fig = plt.figure(figsize=(8, 8))
    rows=1
    columns=3
    
    fig, axes = plt.subplots(nrows=rows, ncols=columns)
    ax = axes.flat[0]
    
    im = ax.imshow(Image1)
    ax.set_title("PLDA sigmoid")
    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.36, 0.02, 0.3])
    fig.colorbar(im, ax=ax)

    ax = axes.flat[1]
    im = ax.imshow(Image2)
    ax.set_yticklabels([])
    ax.set_title("Groundtruth")

    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.36, 0.02, 0.3])
    fig.colorbar(im, ax=ax)

    ax = axes.flat[2]
    # im = ax.imshow(Image3, vmin=0, vmax=1)
    im = ax.imshow(Image3)
    ax.set_yticklabels([])
    ax.set_title("GNN")
    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.36, 0.02, 0.3])
    fig.colorbar(im, ax=ax)
    # plt.show()

    fig.savefig(imagepath)

def get_features_filewise(feats_fname,reco2utt):
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

def get_features(feats_fname):
    # if args.dataset_str == "vox_diar" or args.dataset_str == "lib_vox_cv_all":
    if args.dataset_str == "lib_vox_cv_all":
        prefix = '/data1/prachis/Dihard_2020/gae-pytorch/gae/tools_diar/'
    else:
        prefix = ''
    featsdict = {}
    with open(args.xvecpath ) as fpath:
        for line in fpath: 
            key, value = line.split(" ",1)
          
            featsdict[key] = value.rsplit()[0]

    reco2utt_list = open(args.reco2utt_list).readlines()
    reco2utt_dict = {}
    for line in reco2utt_list:
        rec, utt = line.split(" ",1)
        reco2utt_dict[rec] = utt
    
    reco2utt = reco2utt_dict[feats_fname]

    utts = reco2utt.rstrip().split()
    feats_list = []
    
    for j,utt in enumerate(utts):
        features = read_vec_flt(f'{prefix}{featsdict[utt]}')
        feats_list.append(features)
    
    features = np.array(feats_list)
    return features


def extract_processed_xvectors():
    # DISPLACE analysis
    
    device = 'cpu'
    dataset = args.dataset_str
    filename = args.filename

    kaldi_recipe_path='/data1/prachis/Dihard_2020/Dihard_2020_track1'
    pldadataset = 'displace_dev_fbank_seg_0.75s'
    pldamodel= 'lists/{0}/plda_displace_dev_fbank_0.75s.pkl'.format(pldadataset)
    pldamodel = pkl.load(open(pldamodel,'rb'))
    # dataset = 'dihard_dev_2020_track1_fbank_jhu'

    if args.xvecpath is None:
        xvecpath = 'tools_diar/xvectors_0.25s_npy/{}/'.format(dataset)

    try:
        X = np.load('{}/{}.npy'.format(args.xvecpath,filename))
    except:
        
        X = get_features(filename)
        # X = []
        # for f in filename:
        #     X.append(np.load('{}/{}.npy'.format(xvecpath,f)))
        # X = np.concatenate(X,axis=0)
    features = torch.FloatTensor(X).to(device)
   
    xvecD = X.shape[1]
    pca_dim = 30
    target_energy = 1 # use the energy based PCA

    inpdata = features[np.newaxis]
    net_init = weight_initialization(pldamodel,dimension=xvecD,pca_dimension=pca_dim,device=device)
    model_init = net_init.to(device)
    affinity_init,_,_,xnew = model_init.compute_plda_affinity_matrix(pldamodel,inpdata,target=target_energy) # original filewise PCA transform
    # output_model = affinity_init.detach().cpu().numpy()[0]
    xnew = xnew.detach().cpu().numpy()[0]

    np.save(f'{args.outf}/{filename}.npy',xnew)
    
def main_spec_clustering():

    test_freq = 1 
    feat_dim = 512

    val_batch = 8
    print("Using {} dataset".format(args.dataset_str))
    
    filepath = 'lists/{}/{}'.format(args.dataset_str,args.splitpath)
    
    if os.path.isfile(filepath):
        val_list = np.genfromtxt(filepath,dtype=float).astype(int).reshape(-1,)
    else:
        print('file does not exist')
        return
    pair_list = np.array(open(args.reco2utt_list).readlines())

    # rttm_gndfile_val= 'lists/dihard_dev_2020_track1_fbank_jhu/rttm_val'
    if args.dataset_str=='dihard_dev_2020_track1_fbank_jhu':
        rttm_gndfile_val = '/home/prachis/Dihard_2020/LDC2020E12_Third_DIHARD_Challenge_Development_Data/data/rttm'
    else:
        # rttm_gndfile_val = '/data1/prachis/Dihard_2020/Dihard_2020_track1/data/dihard_eval_2020_track1/filewise_rttms'
        rttm_gndfile_val = f'lists/{args.dataset_str}/filewise_rttms'

    if args.clustering == 'spectral':
        base_fold = f'exp/results_pldaSpectral_baseline/{args.dataset_str}/'
        out_file=base_fold+'/'+'final_spectral{}rttms/'.format(args.pref)
        os.system('mkdir -p {}'.format(out_file))
    else:
        pref='_baseline_widePLDA_'
        base_fold = f'exp/results_pldaPIC_baseline/{args.dataset_str}/'
        
        out_file=base_fold+'/'+'final_pic{}rttms/'.format(pref)
        os.system('mkdir -p {}'.format(out_file))
        
    params = {}
    params['test_freq'] = test_freq
    params['feat_dim'] = feat_dim
    params['rttm_gndval'] = rttm_gndfile_val
    params['out_file_base'] = out_file
    params['val_list'] = val_list
    params['val_batch'] = val_batch

    runner = baseline_clustering(**params)
    if args.clustering == 'spectral':
        runner.gae_spectral_clustering(args,val_list)
        
    elif args.clustering == 'pic':
        # runner.gae_pic_clustering(args,val_list)
        runner.gae_pic_clustering_amiplda(args,val_list)
    else:
        runner.gae_ahc_clustering(args,val_list)
    
def main_spec_clustering_wespk(pldamodel=None):

    test_freq = 1 
    feat_dim = 256
    
    val_batch = 8
    print("Using {} dataset".format(args.dataset_str))
    
    filepath = 'lists_wespk/{}/{}'.format(args.dataset_str,args.splitpath)
    
    if os.path.isfile(filepath):
        val_list = np.genfromtxt(filepath,dtype=float).astype(int).reshape(-1,)
    else:
        print('file does not exist')
        return
    pair_list = np.array(open(args.reco2utt_list).readlines())

    # rttm_gndfile_val= 'lists/dihard_dev_2020_track1_fbank_jhu/rttm_val'
    if args.dataset_str=='dihard_dev_2020_track1_fbank_jhu':
        rttm_gndfile_val = '/home/prachis/Dihard_2020/LDC2020E12_Third_DIHARD_Challenge_Development_Data/data/rttm'
    else:
        # rttm_gndfile_val = '/data1/prachis/Dihard_2020/Dihard_2020_track1/data/dihard_eval_2020_track1/filewise_rttms'
        rttm_gndfile_val = f'lists_wespk/{args.dataset_str}/filewise_rttms'

    if args.clustering == 'spectral':
        # base_fold = f'exp/results_pldaSpectral_baseline/{args.dataset_str}/'
        # out_file=base_fold+'/'+'final_spectral{}rttms/'.format(args.pref)
        out_file=args.outf
        os.system('mkdir -p {}'.format(out_file))
    else:
        pref='_baseline_widePLDA_'
        base_fold = f'exp/results_pldaPIC_baseline/{args.dataset_str}/'
        
        out_file=base_fold+'/'+'final_pic{}rttms/'.format(pref)
        os.system('mkdir -p {}'.format(out_file))
        
    params = {}
    params['test_freq'] = test_freq
    params['feat_dim'] = feat_dim
    params['rttm_gndval'] = rttm_gndfile_val
    params['out_file_base'] = out_file
    params['val_list'] = val_list
    params['val_batch'] = val_batch

    runner = baseline_clustering(**params)
    if args.clustering == 'spectral':
        runner.gae_spectral_clustering(args,val_list,pldamodel)
        
    elif args.clustering == 'pic':
        # runner.gae_pic_clustering(args,val_list)
        runner.gae_pic_clustering_amiplda(args,val_list)
    else:
        runner.gae_ahc_clustering(args,val_list)
    

if __name__ == '__main__':
    if args.extract_feats:
        extract_processed_xvectors()
    else:
        # main_spec_clustering() # for ETDNN
        main_spec_clustering_wespk(args.pldamodel) # for Wespk
    

