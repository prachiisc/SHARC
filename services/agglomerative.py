#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 15:51:43 2019

@author: prachi singh 
@email: prachisingh@iisc.ac.in 
"""

import numpy as np
import argparse
import sys
import os
import matplotlib.pyplot as plt
import pickle
from pdb import set_trace as bp
import subprocess
import scipy.io as sio
from scipy.sparse import coo_matrix
# import path_integral_clustering as mypic
# import eend_cosine_embeddings as myeend
import torch
import torch.nn as nn
import itertools


sys.path.insert(0,os.getcwd())
sys.path.insert(0,os.getcwd()+'/../SSC')
print(os.getcwd())

# from models_train_ssc import weight_initialization,Deep_Ahc_model
device='cpu'
#updating
def setup():
    """Get cmds and setup directories."""
    cmdparser = argparse.ArgumentParser(description='Do speaker clsutering based on'\
                                                    'my ahc',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cmdparser.add_argument('--threshold', help='threshold for clustering',
                            type=float, default=None)
    cmdparser.add_argument('--lamda', help='lamda for clustering',
                            type=float, default=0)
    # cmdparser.add_argument('--custom-dist', help='e.g. euclidean, cosine', type=str, default=None)
    cmdparser.add_argument('--reco2utt', help='spk2utt to create labels', default='../swbd_diar/exp/callhome1/spk2utt')
    cmdparser.add_argument('--reco2num', help='reco2num_spk to get true speakers', default='None')
    cmdparser.add_argument('--label-out', dest='out_file',
                           help='output file used for storing labels', default='../generated_rttm_new/rttm_callhome_my_clustering/cosine/labels')
    # cmdparser.add_argument('--minMaxK', nargs=2, default=[2, 10])
    cmdparser.add_argument('--score_file', help='file containing list of score matrices', type=str,default='../lists/callhome1/callhome1.list')
    cmdparser.add_argument('--score_path', help='path of scores', type=str,default='../scores_cosine/callhome1_scores')
    cmdparser.add_argument('--using_init', help='if initialisation is needed', type=int,default=0)
    cmdparser.add_argument('--dataset', help='dataset name', type=str, default="callhome1")
    cmdparser.add_argument('--k', type=float, default=30)
    cmdparser.add_argument('--z', type=float, default=0.1)
    cmdparser.add_argument('--clustering', type=str, default='PIC')
    # cmdparser.add_argument('--out_path', help='path of output scores', type=str, default=None)
    cmdparser.add_argument('--weight', help='weight for fusion',
                            type=float, default=1.0)
    cmdargs = cmdparser.parse_args()
    return cmdargs

def compute_affinity_matrix(X):
        """Compute the affinity matrix from data.

        Note that the range of affinity is [0,1].

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            affinity: numpy array of shape (n_samples, n_samples)
        """
        # Normalize the data.
        l2_norms = np.linalg.norm(X, axis=1)
        X_normalized = X / l2_norms[:, None]
        # Compute cosine similarities. Range is [-1,1].
        cosine_similarities = np.matmul(X_normalized, np.transpose(X_normalized))
        # Compute the affinity. Range is [0,1].
        # Note that this step is not mentioned in the paper!
        affinity = cosine_similarities

        # affinity = (cosine_similarities + 1.0) / 2.0
        return affinity


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

def AHC(sim_mx, threshold=None,nspeaker=1,maxspeaker=10):
    """ Performs UPGMA variant (wikipedia.org/wiki/UPGMA) of Agglomerative
    Hierarchical Clustering using the input pairwise similarity matrix.
    Input:
        sim_mx    - NxN pairwise similarity matrix
        threshold - threshold for stopping the clustering algorithm
                    (see function twoGMMcalib_lin for its estimation)
    Output:
        cluster labels stored in an array of length N containing (integers in
        the range from 0 to C-1, where C is the number of dicovered clusters)
    """
    dist = -sim_mx
    dist[np.diag_indices_from(dist)] = np.inf
    clsts = [[i] for i in range(len(dist))]
    clst_count = len(dist)
    print('start speaker count: ',clst_count)
    while True:
        mi, mj = np.sort(np.unravel_index(dist.argmin(), dist.shape))
        if threshold is None:
            if clst_count==nspeaker:
                print('nspeaker: ',clst_count)
                break
        else:
            if dist[mi, mj] > -threshold and clst_count<=maxspeaker:
                break
        dist[:, mi] = dist[mi,:] = (dist[mi,:]*len(clsts[mi])+dist[mj,:]*len(clsts[mj]))/(len(clsts[mi])+len(clsts[mj]))
        dist[:, mj] = dist[mj,:] = np.inf
        clsts[mi].extend(clsts[mj])
        clsts[mj] = None
        clst_count = clst_count - 1
    labs= np.empty(len(dist), dtype=int)
    for i, c in enumerate([e for e in clsts if e]):
        labs[c] = i
    return labs

class clustering:
    def __init__(self,n_clusters,clusterlen,labelfull,dist=None,lamda=0):
        self.n_clusters = n_clusters
        self.labelfull = labelfull.copy()
        self.mergeind = []
        self.eta = 0.1
        self.kc = 2
        self.max_10per_scores = 5
        self.lamda = lamda
        self.clusterlen = clusterlen.copy()
       
        # self.clusterlen=[1]*len(labelfull)
        self.dist = dist
        self.minloss_current = 1000

    def initialize_clusters(self,A):
        sampleNum = len(A)
        NNIndex = np.argsort(A)[:,::-1]
        clusterLabels = np.ones((sampleNum, 1),dtype=int)*(-1)
        counter = 0
        for i in range(sampleNum):
            idx = NNIndex[i,:2]
            assignedCluster = clusterLabels[idx]
            assignedCluster = np.unique(assignedCluster[assignedCluster >= 0])
            if len(assignedCluster) == 0:
                clusterLabels[idx] = counter
                counter = counter + 1
            elif len(assignedCluster) == 1:
                clusterLabels[idx] = assignedCluster
            else:
                clusterLabels[idx] = assignedCluster[0];            
                for j in range(1,len(assignedCluster)):
                    clusterLabels[clusterLabels == assignedCluster[j]] = assignedCluster[0]
            
        uniqueLabels = np.unique(clusterLabels)
        clusterNumber = len(uniqueLabels)
        
        self.labelfull = clusterLabels[:,0].astype(int)
        initialClusters = []
        output_new = A.copy()
        clusterlist=[]
        for i,lab in enumerate(uniqueLabels):
            ind=np.where(clusterLabels==lab)[0]
            cluster_count = len(ind)
            initialClusters.append(cluster_count)
            clusterlist.append(ind[0])
            avg=np.sum(output_new[ind],axis=0)
            output_new[ind[0]]=avg
            output_new[:,ind[0]]=avg
        #     initialClusters{i} = find(clusterLabels(:) == uniqueLabels(i));
        self.clusterlen = initialClusters
        output_new = output_new[np.ix_(clusterlist,clusterlist)]
        return self.labelfull,self.clusterlen,output_new  

    def compute_distance(self):
        colvec = np.array(self.clusterlen).reshape(-1,1)
        tmp_mat = np.dot(colvec,colvec.T)
        return (1/tmp_mat)

    def Ahc_full(self,A):
        self.A = A.copy()
        while 1:        
            B = self.A.copy()
            tmp_mat=self.compute_distance()
            self.A = self.A*tmp_mat # all elementwise operation
            self.A = np.triu(self.A,k=1)
            cur_samp = self.A.shape[0]
            minA = np.min(self.A)
            self.A[np.tril_indices(cur_samp)]=-abs(minA)*100
            if cur_samp < 20:
                min_len =  min(20,int(0.1*len(self.labelfull)))
                predicted_clusters =len(np.array(self.clusterlen)[np.array(self.clusterlen)>=min_len])

            if self.n_clusters != None:
                if cur_samp == self.n_clusters:
                    return self.labelfull,self.clusterlen,self.mergeind
                if self.dist!=None:
                   if ((self.A<self.dist).all() or cur_samp==1):
                        if predicted_clusters >= cur_samp:
                            print('predicted_clusters:',predicted_clusters)
                            return self.labelfull,self.clusterlen,self.mergeind
            else:
                if (self.A<self.dist).all() or cur_samp==1:
                    if predicted_clusters >= cur_samp:
                        print('predicted_clusters:',predicted_clusters)
                        return self.labelfull,self.clusterlen,self.mergeind
           
            ind = np.where(self.A==np.amax(self.A))
            minind = min(ind[0][0],ind[1][0])
            maxind = max(ind[0][0],ind[1][0])
            trackind = [list(np.where(self.labelfull==minind)[0])]
            trackind.extend(np.where(self.labelfull==maxind)[0])
            if minind == maxind:
                print(minind,maxind)
            self.clusterlen[minind] +=self.clusterlen[maxind]
            self.clusterlen.pop(maxind)
            self.labelfull[np.where(self.labelfull==maxind)[0]]=minind
            unifull = list(np.unique(self.labelfull))
            labelfullnew = np.zeros(self.labelfull.shape).astype(int)
            for i in range(len(self.labelfull)):
                labelfullnew[i]=unifull.index(self.labelfull[i])
            self.labelfull = labelfullnew
            self.mergeind.append(trackind)
            newsamp = cur_samp -1
            # recomputation
            B[:,minind] =B[:,minind]+B[:,maxind]
            B[minind] = B[:,minind]
            B = np.delete(B,maxind,1)
            B = np.delete(B,maxind,0)
            B[np.diag_indices(newsamp)]=np.min(B)
            B[np.diag_indices(newsamp)] = np.max(B,axis=1)
            self.A = B.copy()
        return self.labelfull,self.clusterlen,self.mergeind


    def get_params(self):
        return self.labelfull, self.mergeind


def write_results_dict(results_dict, output_file,reco2utt):
    """Writes the results in label file"""

    output_label = open(output_file,'w')
    reco2utt = open(reco2utt,'r').readlines()
    i=0

    for meeting_name, hypothesis in results_dict.items():
        
        reco = reco2utt[i].split()[0]
        utts = reco2utt[i].rstrip().split()[1:]
        if reco == meeting_name:
            for j,utt in enumerate(utts):
                if np.isscalar(hypothesis[j]): 
                    towrite = utt +' '+str(hypothesis[j])+'\n'
                else:
                    if hypothesis[j,1]==-1:
                        towrite = utt +'\t'+str(hypothesis[j,0])+'\n'
                    else: 
                        towrite = utt +'\t'+str(hypothesis[j,0])+' '+str(hypothesis[j,1])+'\n'
                output_label.writelines(towrite)     
        else:
            print('reco mismatch!')
            
             
        i=i+1
        
def GDL_clustering_withlabels():
    args = setup()
    fold = args.score_path
    file_list = open(args.score_file,'r').readlines()
    out_file = args.out_file
    reco2utt = args.reco2utt
    reco2num = args.reco2num
    threshold=args.threshold
    neb = 2
    beta1 = 0.95
    per_k=args.k
    k= int(args.k)
    z=args.z
    
    print(threshold)
    if reco2num != 'None':
        reco2num_spk = open(args.reco2num).readlines()
    results_dict ={}
    for i,fn in enumerate(file_list):
        f=fn.rsplit()[0]
        print("filename:",f)

        labelfull = np.squeeze(np.load(fold+'/'+f+'.npy').astype(int))

        uni1,method1=unique(labelfull,True)
        results_dict[f]=method1
    write_results_dict(results_dict, out_file,reco2utt)
    
def PIC_clustering_modified_short():
    args = setup()
    fold = args.score_path
    file_list = open(args.score_file,'r').readlines()
    out_file = args.out_file
    reco2utt = args.reco2utt
    reco2num = args.reco2num
    threshold=args.threshold
    dataset=args.dataset
    xvecpath = 'xvectors_npy/{}_xvec0_5s/'.format(dataset)
    pca_dim = 30

    neb = 2
    beta1 = 0.95
    k=args.k
    z=args.z
    overlap_th = 0.01
    if z> 0.2:
        overlap_th = 0.2
    print('k:{}, z:{}'.format(k,z))
    print(threshold)
    if os.path.isfile(out_file):
        return
    if reco2num != 'None':
        reco2num_spk = open(args.reco2num).readlines()
    results_dict ={}
    for i,fn in enumerate(file_list):
        f=fn.rsplit()[0]
        print("filename:",f)
        modelpath1 = fold+'/../models/'+f+'.pth'
        n_clusters = int(reco2num_spk[i].split()[1])
       
        pldamodelpath = '../SSC/lists/{0}/plda_{0}.pkl'.format(dataset)
        pldamodel = pickle.load(open(pldamodelpath,'rb'))
        system = np.load(xvecpath+'/'+f+'.npy')
        x1_array=system[np.newaxis]
        data = torch.from_numpy(x1_array).float()
        inpdata =  data.float().to(device)
        xvecD = inpdata.shape[2]
        net = Deep_Ahc_model(pldamodel,dimension=xvecD,red_dimension=pca_dim,device=device)
        model = net.to(device)
        if n_clusters > 1:
             modelpath = torch.load(modelpath1)
             model.load_state_dict(modelpath)
        model.eval()
        output,_= model(inpdata)
        b=output.detach().cpu().numpy()[0]
        ground_labels=open('../SSC/ALL_GROUND_LABELS/'+args.dataset+'_xvec0_5s/threshold_0.25/labels_'+f).readlines()
        full_gndlist=[g.split()[1:] for g in ground_labels]
        gnd_list = np.array([g[0] for g in full_gndlist])
        uni_gnd_letter = np.unique(gnd_list)
        uni_gnd = np.arange(len(uni_gnd_letter))
        nframe=len(full_gndlist)
        
        clusterlen_gnd=[]
        speaker_dict={}
        for ind,uni in enumerate(uni_gnd_letter):
            myindex=np.where(gnd_list==uni)[0]
            speaker_dict[uni]=ind
            gnd_list[myindex]=ind
            clusterlen_gnd.append(len(myindex))
        gnd_list = gnd_list.astype(int)
        speaker_count = len(uni_gnd_letter)
        clean_list = np.array([len(f) for f in full_gndlist])
        clean_ind =np.where(clean_list == 1)[0]
        overlap_ind =np.where(clean_list >1)[0]
        overlap_list = np.empty((len(overlap_ind),1),dtype=int)
        for j,ind in enumerate(overlap_ind):
            key = full_gndlist[ind][1]
            if key in speaker_dict.keys():
                overlap_list[j]=speaker_dict[key]
            else:
                speaker_dict[key]=speaker_count
                overlap_list[j] = speaker_dict[key]
                speaker_count +=1
        # overlap_list = np.array([g[1] for g in full_gndlist])
        # overlap_list = overlap_list[overlap_ind]
        uni_overlap_letter = np.unique(overlap_list)
        # uni_gnd = np.arange(len(uni_gnd_letter))
        
        clusterlen_gnd = np.array(clusterlen_gnd)
        for uni in uni_overlap_letter:
            # overlap_list[overlap_list==uni]=ind
            # ind = np.where(uni_gnd_letter==uni)[0]
            # ind = speaker_dict[uni]
            if uni < len(clusterlen_gnd):
                clusterlen_gnd[uni]= clusterlen_gnd[uni] + len(np.where(overlap_list==uni)[0])
            else:
                clusterlen_gnd = np.vstack((clusterlen_gnd.reshape(-1,1),len(np.where(overlap_list==uni)[0])))
        print("----------------------------------------------------------------------")
        print('filename: {} overlap_percentage: {}' .format(f,100*len(overlap_ind)/nframe))
        print("clusterlen_gnd",clusterlen_gnd)
        print("-----------------------------------------------------------------------------")
        if "baseline" in fold :
            b = np.load(fold+'/'+f+'.npy')
            b = (b+1)/2
        else:
            # deepahcmodel = pickle.load(open(fold+'/'+f+'.pkl','rb'))
            # b = deepahcmodel['output']
            b = (b+1)/2

            # weighting for temporal weightage
            N= b.shape[0]
            toep = np.abs(np.arange(N).reshape(N,1)-np.arange(N).reshape(1,N))
            toep[toep>neb] = neb
            weighting = beta1**(toep)
            b = weighting*b

        clusterlen = [1]*b.shape[0]
        labels = np.arange(b.shape[0])
        filelength = len(b)
        k=min(k,len(b)-1)
        print('filelength:',len(b))
        if reco2num != 'None':
            n_clusters = int(reco2num_spk[i].split()[1])      
            n_clusters = min(n_clusters,len(clusterlen)) # whichever is minimum
            if f!=reco2num_spk[i].split()[0]:
                print('file mismatch',f,reco2num_spk[i].split()[0])
            threshold = None
            affinity = b.copy()
            
            clus = mypic.PIC_dihard_threshold(n_clusters,clusterlen,labels,affinity,K=k,z=z)
            if filelength > 500:
                if "baseline" in fold:
                    labelfull,clusterlen,W= clus.gacCluster_oracle_org()
                else:
                    labelfull,clusterlen,W= clus.gacCluster_oracle()
            else:
                k = min(k,len(b)-1)
                labelfull,clusterlen,W= clus.gacCluster_oracle()
            Wnew = W/np.sum(W,axis=1).reshape(-1,1)
            if len(overlap_ind)>0:
                accuracy = 100*len(np.where(np.sum(Wnew[overlap_ind]>overlap_th,axis=1)>1)[0])/len(overlap_ind)
            else:
                accuracy=None
            false_alarm = 100*len(np.where(np.sum(Wnew[clean_ind]>overlap_th,axis=1)>1)[0])/len(clean_ind)
            predicted_overlap_percentage=100*len(np.where(np.sum(Wnew>overlap_th,axis=1)>1)[0])/len(W)
            print("------------------------------------------------------------------------")
            print("filename: {} n_clusters:{} clusterlen:{} accuracy:{} false_alarm:{} \n predicted_overlap_percentage:{}\n".format(f,n_clusters,clusterlen,accuracy,false_alarm,predicted_overlap_percentage))
            print("-------------------------------------------------------------------------------")
            
            label_withoverlap=np.ones((filelength,2),dtype=int)*(-1)
            pred_overlap_ind=np.where(np.sum(Wnew>overlap_th,axis=1)>1)[0]
            label_withoverlap[:,0]=labelfull
            if len(pred_overlap_ind)>0:
                sort_ind = np.argsort(Wnew[pred_overlap_ind],axis=1)[:,::-1]
                label_withoverlap[pred_overlap_ind,1]=sort_ind[:,1]
            labelfull = label_withoverlap
            
        else:
            
            affinity = b.copy()
            n_clusters = 1  # atleast 1 speaker
            clus = mypic.PIC_dihard_threshold(n_clusters,clusterlen,labels,affinity,threshold,K=k,z=z)
            if filelength > 500:
                if "baseline" in fold:
                    labelfull,clusterlen,W= clus.gacCluster_org()                    
                else:
                    labelfull,clusterlen,W= clus.gacCluster()
            else:
                labelfull,clusterlen,W= clus.gacCluster_org()
            n_clusters = len(clusterlen)
            
            Wnew = W/np.sum(W,axis=1).reshape(-1,1)
            if len(overlap_ind)>0:
                accuracy = 100*len(np.where(np.sum(Wnew[overlap_ind]>overlap_th,axis=1)>1)[0])/len(overlap_ind)
            else:
                accuracy=None
            false_alarm = 100*len(np.where(np.sum(Wnew[clean_ind]>overlap_th,axis=1)>1)[0])/len(clean_ind)
            predicted_overlap_percentage=100*len(np.where(np.sum(Wnew>overlap_th,axis=1)>1)[0])/len(W)
            print("------------------------------------------------------------------------")
            print("filename: {} n_clusters:{} clusterlen:{}  accuracy:{} false_alarm:{} \n predicted_overlap_percentage:{}\n".format(f,n_clusters,clusterlen,accuracy,false_alarm,predicted_overlap_percentage))
            print("-------------------------------------------------------------------------------")

        # uni1,method1=unique(labelfull,True)
        results_dict[f]=labelfull
    write_results_dict(results_dict, out_file,reco2utt)
    


def PIC_clustering_modified_withahcinit_finecoarse():
    args = setup()
    fold = args.score_path
    file_list = open(args.score_file,'r').readlines()
    out_file = args.out_file
    reco2utt = args.reco2utt
    reco2num = args.reco2num
    threshold=args.threshold
    neb = 2
    beta1 = 0.95
    k=args.k
    z=args.z
    overlap_th = 0.01
    if z> 0.2:
        overlap_th = 0.5
    print('k:{}, z:{}'.format(k,z))
    print(threshold)
    if reco2num != 'None':
        reco2num_spk = open(args.reco2num).readlines()
    results_dict ={}
    for i,fn in enumerate(file_list):
        f=fn.rsplit()[0]
        print("filename:",f)
        # if f!="DH_DEV_0003":
        #     continue
        ground_labels=open('../SSC/ALL_GROUND_LABELS/'+args.dataset+'/threshold_0.75/labels_'+f).readlines()
        # mapping_utts=np.genfromtxt('../SSC/ALL_GROUND_LABELS/'+args.dataset+'/avg_mapping/fine2coarse_'+f,dtype=str)
        # mapping_utts = mapping_utts[:,1]
        
        # uniq_coarse,mapped_utts = unique(mapping_utts,True)
        
        full_gndlist=[g.split()[1:] for g in ground_labels]
        gnd_list = np.array([g[0] for g in full_gndlist])
        uni_gnd_letter = np.unique(gnd_list)
        uni_gnd = np.arange(len(uni_gnd_letter))
        nframe=len(full_gndlist)
        
        clusterlen_gnd=[]
        speaker_dict={}
        for ind,uni in enumerate(uni_gnd_letter):
            myindex=np.where(gnd_list==uni)[0]
            speaker_dict[uni]=ind
            gnd_list[myindex]=ind
            clusterlen_gnd.append(len(myindex))
        gnd_list = gnd_list.astype(int)
        speaker_count = len(uni_gnd_letter)
        clean_list = np.array([len(f) for f in full_gndlist])
        clean_ind =np.where(clean_list == 1)[0]
        overlap_ind =np.where(clean_list >1)[0]
        overlap_list = np.empty((len(overlap_ind),1),dtype=int)
        for j,ind in enumerate(overlap_ind):
            key = full_gndlist[ind][1]
            if key in speaker_dict.keys():
                overlap_list[j]=speaker_dict[key]
            else:
                speaker_dict[key]=speaker_count
                overlap_list[j] = speaker_dict[key]
                speaker_count +=1
        # overlap_list = np.array([g[1] for g in full_gndlist])
        # overlap_list = overlap_list[overlap_ind]
        uni_overlap_letter = np.unique(overlap_list)
        # uni_gnd = np.arange(len(uni_gnd_letter))
        
        clusterlen_gnd = np.array(clusterlen_gnd)
        for uni in uni_overlap_letter:
            # overlap_list[overlap_list==uni]=ind
            # ind = np.where(uni_gnd_letter==uni)[0]
            # ind = speaker_dict[uni]
            if uni < len(clusterlen_gnd):
                clusterlen_gnd[uni]= clusterlen_gnd[uni] + len(np.where(overlap_list==uni)[0])
            else:
                clusterlen_gnd = np.vstack((clusterlen_gnd.reshape(-1,1),len(np.where(overlap_list==uni)[0])))
        print("----------------------------------------------------------------------")
        print('filename: {} overlap_percentage: {}' .format(f,100*len(overlap_ind)/nframe))
        print("clusterlen_gnd",clusterlen_gnd)
        print("-----------------------------------------------------------------------------")
        if "baseline" in fold :
            b_coarse = np.load(fold+'../../'+args.dataset.split('_xvec')[0]+'_scores/cosine_scores/'+f+'.npy')
            b_coarse_ahc = b_coarse.copy()
            # b_coarse = np.load(fold+'/'+f+'.npy')
            b_coarse = (b_coarse+1)/2
            # b_fine = np.load(fold+'../../'+args.dataset+'_xvec_0_5s_scores/cosine_scores/'+f+'.npy')
            # b_fine = np.load(fold+'/'+f+'.npy')
            # b_fine = (b_fine+1)/2
            # b= (b_fine + b_coarse[np.ix_(mapped_utts,mapped_utts)])/2
            b = b_coarse.copy()
            # bp()
        else:
            deepahcmodel = pickle.load(open(fold+'/'+f+'.pkl','rb'))
            b = deepahcmodel['output']
            b = (b+1)/2

            # weighting for temporal weightage
            N= b.shape[0]
            toep = np.abs(np.arange(N).reshape(N,1)-np.arange(N).reshape(1,N))
            toep[toep>neb] = neb
            weighting = beta1**(toep)
            b = weighting*b

        clusterlen = [1]*b_coarse.shape[0]
        labels = np.arange(b_coarse.shape[0])
        filelength = len(b)
        if k>=filelength:
            # k=int(0.5*filelength)
            k = int(filelength-1)
        else:
            k = int(k)
        print('filelength:',filelength)
        if reco2num != 'None':
            n_clusters = int(reco2num_spk[i].split()[1])      
            n_clusters = min(n_clusters,len(clusterlen)) # whichever is minimum
            if f!=reco2num_spk[i].split()[0]:
                print('file mismatch',f,reco2num_spk[i].split()[0])
            threshold = None
            if n_clusters > 1:
                # first AHC till 50 clusters or 50% of filelength whichever is minimum
                affinity = b_coarse_ahc.copy()
                n_clusters_50 = min(50,int(0.5*filelength))
                
                # perform ahc
                clus_ahc =clustering(n_clusters_50,clusterlen,0,labels,dist=threshold)
                labels,clusterlen,_=clus_ahc.Ahc_full(b)
                # perform pic
                
                n_clusters_10 = n_clusters
            
                clus = mypic.PIC_dihard_threshold(n_clusters_10,clusterlen,labels,affinity,K=k,z=z)
                if filelength > 0:
                    if "baseline" in fold:
                        # labelfull,clusterlen,W= clus.gacCluster_withoverlap_new_oracle()
                         labelfull,clusterlen,W= clus.gacCluster_oracle_org()
                    else:
                        labelfull,clusterlen,W= clus.gacCluster_oracle()
                else:
                    
                    # labelfull,clusterlen,W= clus.gacCluster_withoverlap_new_oracle()
                    labelfull,clusterlen,W= clus.gacCluster_oracle_org()
                
                # labelfull_fine = labelfull[mapped_utts]
                
                # affinity = b.copy()
                
                # clus_fine = mypic.PIC_dihard_threshold(n_clusters,clusterlen,labelfull_fine,affinity,K=k,z=z)
                # labelfull,clusterlen,W= clus_fine.gacCluster_withoverlap_new_oracle()
                Wnew = W/np.sum(W,axis=1).reshape(-1,1)
                # Wnew = W
                
                mx = overlap_th*np.max(Wnew,axis=1,keepdims=True)
                mask_overlap= (Wnew>=mx)
                if len(overlap_ind)>0:
                    accuracy = 100*len(np.where(np.sum(mask_overlap[overlap_ind],axis=1)>1)[0])/len(overlap_ind)
                else:
                    accuracy=None
                false_alarm = 100*len(np.where(np.sum(mask_overlap[clean_ind],axis=1)>1)[0])/len(clean_ind)
                predicted_overlap_percentage=100*len(np.where(np.sum(mask_overlap,axis=1)>1)[0])/len(W)
                print("------------------------------------------------------------------------")
                print("filename: {} n_clusters:{} clusterlen:{} accuracy:{} false_alarm:{} \n predicted_overlap_percentage:{}\n".format(f,n_clusters,clusterlen,accuracy,false_alarm,predicted_overlap_percentage))
                print("-------------------------------------------------------------------------------")
               
                # label_withoverlap=np.ones((filelength,2),dtype=int)*(-1)
                # pred_overlap_ind=np.where(np.sum(mask_overlap,axis=1)>1)[0]
                # label_withoverlap[:,0]=labelfull
                # if len(pred_overlap_ind)>0:
                #     sort_ind = np.argsort(Wnew[pred_overlap_ind],axis=1)[:,::-1]
                #     label_withoverlap[pred_overlap_ind,0]=sort_ind[:,0]
                #     label_withoverlap[pred_overlap_ind,1]=sort_ind[:,1]
                    
                # labelfull = label_withoverlap
                
                # Groundtruth overlap
                # if len(overlap_ind)>0:
                #     sort_ind = np.argsort(Wnew[overlap_ind],axis=1)[:,::-1]
                #     label_withoverlap[overlap_ind,0]=sort_ind[:,0]
                #     label_withoverlap[overlap_ind,1]=sort_ind[:,1]
                    
                # labelfull = label_withoverlap
            else:
                labelfull = np.zeros((nframe,),dtype=int)
            # bp()
            
        else:
            
            affinity = b.copy()
            n_clusters = 1  # atleast 1 speaker
            clus = mypic.PIC_dihard_threshold(n_clusters,clusterlen,labels,affinity,threshold,K=k,z=z)
            if filelength > 500:
                if "baseline" in fold:
                    labelfull,clusterlen,W= clus.gacCluster_org()                    
                else:
                    labelfull,clusterlen,W= clus.gacCluster()
            else:
                labelfull,clusterlen,W= clus.gacCluster_org()
            n_clusters = len(clusterlen)
            
            Wnew = W/np.sum(W,axis=1).reshape(-1,1)
            if len(overlap_ind)>0:
                accuracy = 100*len(np.where(np.sum(Wnew[overlap_ind]>overlap_th,axis=1)>1)[0])/len(overlap_ind)
            else:
                accuracy=None
            false_alarm = 100*len(np.where(np.sum(Wnew[clean_ind]>overlap_th,axis=1)>1)[0])/len(clean_ind)
            predicted_overlap_percentage=100*len(np.where(np.sum(Wnew>overlap_th,axis=1)>1)[0])/len(W)
            print("------------------------------------------------------------------------")
            print("filename: {} n_clusters:{} clusterlen:{}  accuracy:{} false_alarm:{} \n predicted_overlap_percentage:{}\n".format(f,n_clusters,clusterlen,accuracy,false_alarm,predicted_overlap_percentage))
            print("-------------------------------------------------------------------------------")

        # uni1,method1=unique(labelfull,True)
        results_dict[f]=labelfull
    write_results_dict(results_dict, out_file,reco2utt)

def PIC_clustering_modified_withfinecoarse():
    args = setup()
    fold = args.score_path
    file_list = open(args.score_file,'r').readlines()
    out_file = args.out_file
    reco2utt = args.reco2utt
    reco2num = args.reco2num
    threshold=args.threshold
    neb = 2
    beta1 = 0.95
    init_k=args.k
    z=args.z
    overlap_th = 0.01
    if z> 0.2:
        overlap_th = 0.5
    print('k:{}, z:{}'.format(init_k,z))
    print(threshold)
    if reco2num != 'None':
        reco2num_spk = open(args.reco2num).readlines()
    results_dict ={}
    for i,fn in enumerate(file_list):
        f=fn.rsplit()[0]
        print("filename:",f)
        # if f!="DH_DEV_0003":
        #     continue
        ground_labels=open('../SSC/ALL_GROUND_LABELS/'+args.dataset+'/threshold_0.25/labels_'+f).readlines()
        mapping_utts=np.genfromtxt('../SSC/ALL_GROUND_LABELS/'+args.dataset+'/avg_mapping/fine2coarse_'+f,dtype=str)
        mapping_utts = mapping_utts[:,1]
        
        uniq_coarse,mapped_utts = unique(mapping_utts,True)
        
        full_gndlist=[g.split()[1:] for g in ground_labels]
        gnd_list = np.array([g[0] for g in full_gndlist])
        uni_gnd_letter = np.unique(gnd_list)
        uni_gnd = np.arange(len(uni_gnd_letter))
        nframe=len(full_gndlist)
        
        clusterlen_gnd=[]
        speaker_dict={}
        for ind,uni in enumerate(uni_gnd_letter):
            myindex=np.where(gnd_list==uni)[0]
            speaker_dict[uni]=ind
            gnd_list[myindex]=ind
            clusterlen_gnd.append(len(myindex))
        gnd_list = gnd_list.astype(int)
        speaker_count = len(uni_gnd_letter)
        clean_list = np.array([len(f) for f in full_gndlist])
        clean_ind =np.where(clean_list == 1)[0]
        overlap_ind =np.where(clean_list >1)[0]
        overlap_list = np.empty((len(overlap_ind),1),dtype=int)
        for j,ind in enumerate(overlap_ind):
            key = full_gndlist[ind][1]
            if key in speaker_dict.keys():
                overlap_list[j]=speaker_dict[key]
            else:
                speaker_dict[key]=speaker_count
                overlap_list[j] = speaker_dict[key]
                speaker_count +=1
        # overlap_list = np.array([g[1] for g in full_gndlist])
        # overlap_list = overlap_list[overlap_ind]
        uni_overlap_letter = np.unique(overlap_list)
        # uni_gnd = np.arange(len(uni_gnd_letter))
        
        clusterlen_gnd = np.array(clusterlen_gnd)
        for uni in uni_overlap_letter:
            # overlap_list[overlap_list==uni]=ind
            # ind = np.where(uni_gnd_letter==uni)[0]
            # ind = speaker_dict[uni]
            if uni < len(clusterlen_gnd):
                clusterlen_gnd[uni]= clusterlen_gnd[uni] + len(np.where(overlap_list==uni)[0])
            else:
                clusterlen_gnd = np.vstack((clusterlen_gnd.reshape(-1,1),len(np.where(overlap_list==uni)[0])))
        print("----------------------------------------------------------------------")
        print('filename: {} overlap_percentage: {}' .format(f,100*len(overlap_ind)/nframe))
        print("clusterlen_gnd",clusterlen_gnd)
        print("-----------------------------------------------------------------------------")
        if "baseline" in fold :
            b_coarse = np.load(fold+'../../'+args.dataset.split('_xvec')[0]+'_scores/cosine_scores/'+f+'.npy')
            # b_coarse = np.load(fold+'/'+f+'.npy')
            b_coarse = (b_coarse+1)/2
            # b_fine = np.load(fold+'../../'+args.dataset+'_xvec_0_5s_scores/cosine_scores/'+f+'.npy')
            b_fine = np.load(fold+'/'+f+'.npy')
            b_fine = (b_fine+1)/2
            b= 0.5*b_fine + 0.5*b_coarse[np.ix_(mapped_utts,mapped_utts)]
            # bp()
        else:
            deepahcmodel = pickle.load(open(fold+'/'+f+'.pkl','rb'))
            b = deepahcmodel['output']
            b = (b+1)/2

            # weighting for temporal weightage
            N= b.shape[0]
            toep = np.abs(np.arange(N).reshape(N,1)-np.arange(N).reshape(1,N))
            toep[toep>neb] = neb
            weighting = beta1**(toep)
            b = weighting*b

        clusterlen = [1]*b_coarse.shape[0]
        labels = np.arange(b_coarse.shape[0])
        filelength = len(b)
        k_10 = int(init_k)
        if init_k>=filelength:
            # k=int(0.5*filelength)
            k = int(filelength-1)
        else:
            k = int(init_k)
        print('filelength:',filelength)
        if reco2num != 'None':
            n_clusters = int(reco2num_spk[i].split()[1])      
            n_clusters = min(n_clusters,len(clusterlen)) # whichever is minimum
            if f!=reco2num_spk[i].split()[0]:
                print('file mismatch',f,reco2num_spk[i].split()[0])
            threshold = None
            affinity = b_coarse.copy()
            n_clusters_10 = n_clusters + 5
            k_10 = min(int(k_10),len(b_coarse)-1)
            if n_clusters > 1:
                clus = mypic.PIC_dihard_threshold(n_clusters_10,clusterlen,labels,affinity,K=k_10,z=z)
                if filelength > 0:
                    if "baseline" in fold:
                        # labelfull,clusterlen,W= clus.gacCluster_withoverlap_new_oracle()
                         labelfull,clusterlen,W= clus.gacCluster_oracle_org()
                    else:
                        labelfull,clusterlen,W= clus.gacCluster_oracle()
                else:
                    
                    # labelfull,clusterlen,W= clus.gacCluster_withoverlap_new_oracle()
                    labelfull,clusterlen,W= clus.gacCluster_oracle_org()
                
                labelfull_fine = labelfull[mapped_utts]
                
                affinity = b.copy()
                z_fine = 0.5
                clus_fine = mypic.PIC_dihard_threshold(n_clusters,clusterlen,labelfull_fine,affinity,K=k,z=z_fine)
                labelfull,clusterlen,W= clus_fine.gacCluster_withoverlap_new_oracle()
                Wnew = W/np.sum(W,axis=1).reshape(-1,1)
                # Wnew = W
                
                mx = overlap_th*np.max(Wnew,axis=1,keepdims=True)
                mask_overlap= (Wnew>=mx)
                if len(overlap_ind)>0:
                    accuracy = 100*len(np.where(np.sum(mask_overlap[overlap_ind],axis=1)>1)[0])/len(overlap_ind)
                else:
                    accuracy=None
                false_alarm = 100*len(np.where(np.sum(mask_overlap[clean_ind],axis=1)>1)[0])/len(clean_ind)
                predicted_overlap_percentage=100*len(np.where(np.sum(mask_overlap,axis=1)>1)[0])/len(W)
                print("------------------------------------------------------------------------")
                print("filename: {} n_clusters:{} clusterlen:{} accuracy:{} false_alarm:{} \n predicted_overlap_percentage:{}\n".format(f,n_clusters,clusterlen,accuracy,false_alarm,predicted_overlap_percentage))
                print("-------------------------------------------------------------------------------")
               
                # label_withoverlap=np.ones((filelength,2),dtype=int)*(-1)
                # pred_overlap_ind=np.where(np.sum(mask_overlap,axis=1)>1)[0]
                # label_withoverlap[:,0]=labelfull
                # if len(pred_overlap_ind)>0:
                #     sort_ind = np.argsort(Wnew[pred_overlap_ind],axis=1)[:,::-1]
                #     label_withoverlap[pred_overlap_ind,0]=sort_ind[:,0]
                #     label_withoverlap[pred_overlap_ind,1]=sort_ind[:,1]
                    
                # labelfull = label_withoverlap
                
                # Groundtruth overlap
                # if len(overlap_ind)>0:
                #     sort_ind = np.argsort(Wnew[overlap_ind],axis=1)[:,::-1]
                #     label_withoverlap[overlap_ind,0]=sort_ind[:,0]
                #     label_withoverlap[overlap_ind,1]=sort_ind[:,1]
                    
                # labelfull = label_withoverlap
            else:
                labelfull = np.zeros((nframe,),dtype=int)
            # bp()
            
        else:
            
            affinity = b.copy()
            n_clusters = 1  # atleast 1 speaker
            clus = mypic.PIC_dihard_threshold(n_clusters,clusterlen,labels,affinity,threshold,K=k,z=z)
            if filelength > 500:
                if "baseline" in fold:
                    labelfull,clusterlen,W= clus.gacCluster_org()                    
                else:
                    labelfull,clusterlen,W= clus.gacCluster()
            else:
                labelfull,clusterlen,W= clus.gacCluster_org()
            n_clusters = len(clusterlen)
            
            Wnew = W/np.sum(W,axis=1).reshape(-1,1)
            if len(overlap_ind)>0:
                accuracy = 100*len(np.where(np.sum(Wnew[overlap_ind]>overlap_th,axis=1)>1)[0])/len(overlap_ind)
            else:
                accuracy=None
            false_alarm = 100*len(np.where(np.sum(Wnew[clean_ind]>overlap_th,axis=1)>1)[0])/len(clean_ind)
            predicted_overlap_percentage=100*len(np.where(np.sum(Wnew>overlap_th,axis=1)>1)[0])/len(W)
            print("------------------------------------------------------------------------")
            print("filename: {} n_clusters:{} clusterlen:{}  accuracy:{} false_alarm:{} \n predicted_overlap_percentage:{}\n".format(f,n_clusters,clusterlen,accuracy,false_alarm,predicted_overlap_percentage))
            print("-------------------------------------------------------------------------------")

        # uni1,method1=unique(labelfull,True)
        results_dict[f]=labelfull
    write_results_dict(results_dict, out_file,reco2utt)

def PIC_clustering_modified():
    args = setup()
    fold = args.score_path
    file_list = open(args.score_file,'r').readlines()
    out_file = args.out_file
    reco2utt = args.reco2utt
    reco2num = args.reco2num
    threshold=args.threshold
    neb = 2
    beta1 = 0.95
    k=args.k
    z=args.z
    overlap_th = 0.01
    if z> 0.2:
        overlap_th = 0.5
    print('k:{}, z:{}'.format(k,z))
    print(threshold)
    if reco2num != 'None':
        reco2num_spk = open(args.reco2num).readlines()
    results_dict ={}
    for i,fn in enumerate(file_list):
        f=fn.rsplit()[0]
        print("filename:",f)
        ground_labels=open('../SSC/ALL_GROUND_LABELS/'+args.dataset.split('_wide')[0]+'/threshold_0.75/labels_'+f).readlines()
        
        full_gndlist=[g.split()[1:] for g in ground_labels]
        gnd_list = np.array([g[0] for g in full_gndlist])
        uni_gnd_letter = np.unique(gnd_list)
        uni_gnd = np.arange(len(uni_gnd_letter))
        nframe=len(full_gndlist)
        
        clusterlen_gnd=[]
        speaker_dict={}
        for ind,uni in enumerate(uni_gnd_letter):
            myindex=np.where(gnd_list==uni)[0]
            speaker_dict[uni]=ind
            gnd_list[myindex]=ind
            clusterlen_gnd.append(len(myindex))
        gnd_list = gnd_list.astype(int)
        speaker_count = len(uni_gnd_letter)
        clean_list = np.array([len(f) for f in full_gndlist])
        clean_ind =np.where(clean_list == 1)[0]
        overlap_ind =np.where(clean_list >1)[0]
        overlap_list = np.empty((len(overlap_ind),1),dtype=int)
        for j,ind in enumerate(overlap_ind):
            key = full_gndlist[ind][1]
            if key in speaker_dict.keys():
                overlap_list[j]=speaker_dict[key]
            else:
                speaker_dict[key]=speaker_count
                overlap_list[j] = speaker_dict[key]
                speaker_count +=1
        # overlap_list = np.array([g[1] for g in full_gndlist])
        # overlap_list = overlap_list[overlap_ind]
        uni_overlap_letter = np.unique(overlap_list)
        # uni_gnd = np.arange(len(uni_gnd_letter))
        
        clusterlen_gnd = np.array(clusterlen_gnd)
        for uni in uni_overlap_letter:
            # overlap_list[overlap_list==uni]=ind
            # ind = np.where(uni_gnd_letter==uni)[0]
            # ind = speaker_dict[uni]
            if uni < len(clusterlen_gnd):
                clusterlen_gnd[uni]= clusterlen_gnd[uni] + len(np.where(overlap_list==uni)[0])
            else:
                clusterlen_gnd = np.vstack((clusterlen_gnd.reshape(-1,1),len(np.where(overlap_list==uni)[0])))
        print("----------------------------------------------------------------------")
        print('filename: {} overlap_percentage: {}' .format(f,100*len(overlap_ind)/nframe))
        print("clusterlen_gnd",clusterlen_gnd)
        print("-----------------------------------------------------------------------------")
        if "baseline" in fold :
            b = np.load(fold+'/'+f+'.npy')
            b = (b+1)/2
        else:
            deepahcmodel = pickle.load(open(fold+'/'+f+'.pkl','rb'))
            b = deepahcmodel['output']
            # b = (b+1)/2
            b = 1/(1+np.exp(-b))
            # weighting for temporal weightage
            N= b.shape[0]
            toep = np.abs(np.arange(N).reshape(N,1)-np.arange(N).reshape(1,N))
            toep[toep>neb] = neb
            weighting = beta1**(toep)
            b = weighting*b

        clusterlen = [1]*b.shape[0]
        labels = np.arange(b.shape[0])
        filelength = len(b)
        if k>=filelength:
            # k=int(0.5*filelength)
            k = int(filelength-1)
        else:
            k = int(k)
        print('filelength:',filelength)
        if reco2num != 'None':
            n_clusters = int(reco2num_spk[i].split()[1])      
            n_clusters = min(n_clusters,len(clusterlen)) # whichever is minimum
            if f!=reco2num_spk[i].split()[0]:
                print('file mismatch',f,reco2num_spk[i].split()[0])
            threshold = None
            affinity = b.copy()
            
            clus = mypic.PIC_dihard_threshold(n_clusters,clusterlen,labels,affinity,K=k,z=z)
            if filelength < 0:
                if "baseline" in fold:
                    labelfull,clusterlen,W= clus.gacCluster_withoverlap_new_oracle()
                else:
                    labelfull,clusterlen,W= clus.gacCluster_oracle()
            else:
                
                labelfull,clusterlen,W= clus.gacCluster_withoverlap_new_oracle()
            # bp()
            Wnew = W/np.sum(W,axis=1).reshape(-1,1)
            # Wnew = W
            
            mx = overlap_th*np.max(Wnew,axis=1,keepdims=True)
            mask_overlap= (Wnew>=mx)
            if len(overlap_ind)>0:
                accuracy = 100*len(np.where(np.sum(mask_overlap[overlap_ind],axis=1)>1)[0])/len(overlap_ind)
            else:
                accuracy=None
            false_alarm = 100*len(np.where(np.sum(mask_overlap[clean_ind],axis=1)>1)[0])/len(clean_ind)
            predicted_overlap_percentage=100*len(np.where(np.sum(mask_overlap,axis=1)>1)[0])/len(W)
            print("------------------------------------------------------------------------")
            print("filename: {} n_clusters:{} clusterlen:{} accuracy:{} false_alarm:{} \n predicted_overlap_percentage:{}\n".format(f,n_clusters,clusterlen,accuracy,false_alarm,predicted_overlap_percentage))
            print("-------------------------------------------------------------------------------")
           
            # label_withoverlap=np.ones((filelength,2),dtype=int)*(-1)
            # pred_overlap_ind=np.where(np.sum(mask_overlap,axis=1)>1)[0]
            # label_withoverlap[:,0]=labelfull
            # if len(pred_overlap_ind)>0:
            #     sort_ind = np.argsort(Wnew[pred_overlap_ind],axis=1)[:,::-1]
            #     label_withoverlap[pred_overlap_ind,0]=sort_ind[:,0]
            #     label_withoverlap[pred_overlap_ind,1]=sort_ind[:,1]
                
            # labelfull = label_withoverlap
            
            # Groundtruth overlap
            # if len(overlap_ind)>0:
            #     sort_ind = np.argsort(Wnew[overlap_ind],axis=1)[:,::-1]
            #     label_withoverlap[overlap_ind,0]=sort_ind[:,0]
            #     label_withoverlap[overlap_ind,1]=sort_ind[:,1]
                
            # labelfull = label_withoverlap
            # bp()
            
        else:
            
            affinity = b.copy()
            n_clusters = 1  # atleast 1 speaker
            clus = mypic.PIC_dihard_threshold(n_clusters,clusterlen,labels,affinity,threshold,K=k,z=z)
            if filelength > 0:
                if "baseline" in fold:
                    labelfull,clusterlen,W= clus.gacCluster_org()                    
                else:
                    labelfull,clusterlen,W= clus.gacCluster()
            else:
                labelfull,clusterlen,W= clus.gacCluster_org()
            n_clusters = len(clusterlen)
            
            Wnew = W/np.sum(W,axis=1).reshape(-1,1)
            if len(overlap_ind)>0:
                accuracy = 100*len(np.where(np.sum(Wnew[overlap_ind]>overlap_th,axis=1)>1)[0])/len(overlap_ind)
            else:
                accuracy=None
            false_alarm = 100*len(np.where(np.sum(Wnew[clean_ind]>overlap_th,axis=1)>1)[0])/len(clean_ind)
            predicted_overlap_percentage=100*len(np.where(np.sum(Wnew>overlap_th,axis=1)>1)[0])/len(W)
            print("------------------------------------------------------------------------")
            print("filename: {} n_clusters:{} clusterlen:{}  accuracy:{} false_alarm:{} \n predicted_overlap_percentage:{}\n".format(f,n_clusters,clusterlen,accuracy,false_alarm,predicted_overlap_percentage))
            print("-------------------------------------------------------------------------------")

        # uni1,method1=unique(labelfull,True)
        results_dict[f]=labelfull
    write_results_dict(results_dict, out_file,reco2utt)

def PIC_clustering_fine():
    args = setup()
    fold = args.score_path
    file_list = open(args.score_file,'r').readlines()
    out_file = args.out_file
    reco2utt = args.reco2utt
    reco2num = args.reco2num
    threshold=args.threshold
    neb = 2
    beta1 = 0.95
    per_k=args.k
    k= int(args.k)
    z=args.z
    
    print(threshold)
    if reco2num != 'None':
        reco2num_spk = open(args.reco2num).readlines()
    results_dict ={}
    for i,fn in enumerate(file_list):
        f=fn.rsplit()[0]
        print("filename:",f)

        if "baseline" in fold :
            b = np.load(fold+'/'+f+'.npy')
            # b = 1/(1+np.exp(-b))
            b = (b+1)/2
        else:
            # deepahcmodel = pickle.load(open(fold+'/'+f+'.pkl','rb'))
            # b = deepahcmodel['output']
            b = np.load(fold+'/'+f+'.npy')
            b = (b+1)/2
            nframe = b.shape[0]
            # # weighting for temporal weightage
            # N= b.shape[0]
            # toep = np.abs(np.arange(N).reshape(N,1)-np.arange(N).reshape(1,N))
            # toep[toep>neb] = neb
            # weighting = beta1**(toep)
            # b = weighting*b
            window_size = 2
            subsample_again = window_size
            mod_i = 0
            moving_averages = []
            b_new = b.copy()
            # b_new = b[::subsample_again]
            # b_new = b_new[:,::subsample_again]
           
            while mod_i < int(len(b)):        
                this_window = b[mod_i : mod_i + window_size]        
                window_average = np.mean(this_window,axis=0)       
                b_new[mod_i] = window_average
                this_window = b_new[:,mod_i : mod_i + window_size]        
                window_average = np.mean(this_window,axis=1)       
                b_new[:,mod_i] = window_average
                mod_i +=subsample_again
            b_new = b[::subsample_again]
            b_new = b_new[:,::subsample_again]
        b = b_new.copy()
        clusterlen = [1]*b.shape[0]
        labels = np.arange(b.shape[0])
        filelength = len(b)
        # if filelength <= k:
        #     k= int(0.5*filelength)
        
        # k = int(max(1,per_k*filelength))
        k = min(k,filelength-1)
        print('filelength:',len(b))
        print('k:{}, z:{}'.format(k,z))
        # bp()
        if reco2num != 'None':
            n_clusters = int(reco2num_spk[i].split()[1])      
            n_clusters = min(n_clusters,len(clusterlen)) # whichever is minimum
            if f!=reco2num_spk[i].split()[0]:
                print('file mismatch',f,reco2num_spk[i].split()[0])
            threshold = None
            affinity = b.copy()

            if filelength > 0:
                if "baseline" in fold:
                    clus = mypic.PIC_org(n_clusters,clusterlen,labels,affinity,K=k,z=z)
                else:
                    clus = mypic.PIC_ami(n_clusters,clusterlen,labels,affinity,K=k,z=z)
            else:
                clus = mypic.PIC_callhome(n_clusters,clusterlen,labels,affinity,K=k,z=z)
               
            labelfull,clusterlen= clus.gacCluster()
            print("filename: {} n_clusters:{} clusterlen:{}".format(f,n_clusters,clusterlen))

        else:
            affinity = b.copy()
            n_clusters = 1  # atleast 1 speaker

            if filelength > 1000:
                if "baseline" in fold:
                    clus = mypic.PIC_org_threshold(n_clusters,clusterlen,labels,affinity,threshold,K=k,z=z)
                else:
                    clus = mypic.PIC_ami_threshold(n_clusters,clusterlen,labels,affinity,threshold,K=k,z=z)
            else:
                clus = mypic.PIC_callhome_threshold(n_clusters,clusterlen,labels,affinity,threshold,K=k,z=z)
            labelfull,clusterlen= clus.gacCluster()
            n_clusters = len(clusterlen)
            print("filename: {} n_clusters:{} clusterlen:{}".format(f,n_clusters,clusterlen))
        res = list(itertools.chain.from_iterable(itertools.repeat(i, subsample_again) 
                                           for i in labelfull)) 

        labelfull = np.array(res[:nframe])
        uni1,method1=unique(labelfull,True)
        results_dict[f]=method1
    write_results_dict(results_dict, out_file,reco2utt)
    
def PIC_clustering_fusion():
    args = setup()
    fold = args.score_path
    file_list = open(args.score_file,'r').readlines()
    out_file = args.out_file
    reco2utt = args.reco2utt
    reco2num = args.reco2num
    threshold=args.threshold
    weight = args.weight
    neb = 2
    beta1 = 0.95
    per_k=args.k
    k= int(args.k)
    z=args.z
    
    print(threshold)
    if reco2num != 'None':
        reco2num_spk = open(args.reco2num).readlines()
    results_dict ={}
    for i,fn in enumerate(file_list):
        f=fn.rsplit()[0]
        print("filename:",f)
        out_affinity = None
        # out_affinity =  os.path.dirname(out_file)+'/pic_affinity_10/'+f+'.npy'
        if "baseline" in fold :
            b = np.load(fold+'/'+f+'.npy')
            similarity='plda'
            fold_plda='/data1/prachis/Dihard_2020/SSC/{}_pca_baseline/{}_scores/{}_scores'.format(similarity,args.dataset,similarity)
            b_plda = np.load(fold_plda+'/'+f+'.npy')
            # b_plda = 1/(1+np.exp(-b_plda))
            b_plda = b_plda/np.max(abs(b_plda))
            
            b= b/np.max(abs(b))
            
            # b_plda = b_plda - np.mean(b_plda)
            # b_plda = b_plda/np.std(b_plda)
            # b = b-np.mean(b)
            # b = b/np.std(b)
            
            b = weight*(b)+(1-weight)*b_plda
            b = 1/(1+np.exp(-b))
            # b = (b+1)/2
        else:
            if os.path.isfile(fold+'/'+f+'.pkl'):
                deepahcmodel = pickle.load(open(fold+'/'+f+'.pkl','rb'))
                b = deepahcmodel['output']
                # if "DH_DEV_0001" in f and b.shape[0]!=924:
                #     b2 = b[1:]
                #     b = b2[:,1:]
            else:
                b = np.load(fold+'/'+f+'.npy')
            fold_plda='/data1/prachis/Dihard_2020/SSC/plda_pca_baseline/dihard_eval_2020_track1_wide_scores/plda_scores'
            b_plda = np.load(fold_plda+'/'+f+'.npy')
            # b_plda = 1/(1+np.exp(-b_plda))
            # b_plda = b_plda-np.min(b_plda)
            # b_plda /=np.max(b_plda)
            # b = (b+b.T)/2
            # b /=np.max(b)
            # b = 1-b
            b = weight*(b)+(1-weight)*b_plda
            b = 1/(1+np.exp(-b))
            # b = b-np.min(b)
            # b /=np.max(b)
            # b = (b+1)/2
            # nframe = b.shape[0]
            # # # weighting for temporal weightage
            # N= b.shape[0]
            # toep = np.abs(np.arange(N).reshape(N,1)-np.arange(N).reshape(1,N))
            # toep[toep>neb] = neb
            # weighting = beta1**(toep)
            # b = weighting*b
            
        
        clusterlen = [1]*b.shape[0]
        labels = np.arange(b.shape[0])
        filelength = len(b)
        # if filelength <= k:
        #     k= int(0.5*filelength)
        
        # k = int(max(1,per_k*filelength))
        k = min(k,filelength-1)
        print('filelength:',len(b))
        print('k:{}, z:{}'.format(k,z))
        # bp()
        if reco2num != 'None':
            try:
                n_clusters = int(reco2num_spk[i].split()[1])      
            except:
                n_clusters = 2
            n_clusters = min(n_clusters,len(clusterlen)) # whichever is minimum
            if f!=reco2num_spk[i].split()[0]:
                print('file mismatch',f,reco2num_spk[i].split()[0])
            threshold = None
            affinity = b.copy()
            
            if filelength > 0:
                if "baseline" in fold:
                    clus = mypic.PIC_org(n_clusters,clusterlen,labels,affinity,K=k,z=z,path=out_affinity)
                else:
                    clus = mypic.PIC_ami(n_clusters,clusterlen,labels,affinity,K=k,z=z)
            else:
                clus = mypic.PIC_callhome(n_clusters,clusterlen,labels,affinity,K=k,z=z)
               
            labelfull,clusterlen= clus.gacCluster()
            n_clusters = len(clusterlen)
            print("filename: {} n_clusters:{} clusterlen:{}".format(f,n_clusters,clusterlen))

        else:
            affinity = b.copy()
            n_clusters = 1  # atleast 1 speaker

            if filelength > 0:
                if "baseline" in fold:
                    clus = mypic.PIC_org_threshold(n_clusters,clusterlen,labels,affinity,threshold,K=k,z=z)
                else:
                    clus = mypic.PIC_ami_threshold(n_clusters,clusterlen,labels,affinity,threshold,K=k,z=z)
            else:
                clus = mypic.PIC_callhome_threshold(n_clusters,clusterlen,labels,affinity,threshold,K=k,z=z)
            labelfull,clusterlen= clus.gacCluster()
            n_clusters = len(clusterlen)
            print("filename: {} n_clusters:{} clusterlen:{}".format(f,n_clusters,clusterlen))
       
        uni1,method1=unique(labelfull,True)
        results_dict[f]=method1
    write_results_dict(results_dict, out_file,reco2utt)
    
def PIC_clustering():
    args = setup()
    fold = args.score_path
    file_list = open(args.score_file,'r').readlines()
    out_file = args.out_file
    reco2utt = args.reco2utt
    reco2num = args.reco2num
    threshold=args.threshold
    neb = 2
    beta1 = 0.95
    per_k=args.k
    k= int(args.k)
    z=args.z
    
    print(threshold)
    if reco2num != 'None':
        reco2num_spk = open(args.reco2num).readlines()
    results_dict ={}
    for i,fn in enumerate(file_list):
        f=fn.rsplit()[0]
        print("filename:",f)

        if "baseline" in fold :
            b = np.load(fold+'/'+f+'.npy')
            b = 1/(1+np.exp(-b))
            # b = (b+1)/2
        else:
            if os.path.isfile(fold+'/'+f+'.pkl'):
                deepahcmodel = pickle.load(open(fold+'/'+f+'.pkl','rb'))
                b = deepahcmodel['output']
                # if "DH_DEV_0001" in f and b.shape[0]!=924:
                #     b2 = b[1:]
                #     b = b2[:,1:]
            else:
                b = np.load(fold+'/'+f+'.npy')
            # fold_plda='/data1/prachis/Dihard_2020/SSC/plda_pca_baseline/dihard_dev_2020_track1_scores/plda_scores'
            # b_plda = np.load(fold_plda+'/'+f+'.npy')
            # b = (b+b_plda)/2
            b = b/np.max(abs(b))
            b = 1/(1+np.exp(-b))
            # b = b-np.min(b)
            # b /=np.max(b)
            # b = (b+1)/2
            # nframe = b.shape[0]
            # # # weighting for temporal weightage
            # N= b.shape[0]
            # toep = np.abs(np.arange(N).reshape(N,1)-np.arange(N).reshape(1,N))
            # toep[toep>neb] = neb
            # weighting = beta1**(toep)
            # b = weighting*b
            
            
        clusterlen = [1]*b.shape[0]
        labels = np.arange(b.shape[0])
        filelength = len(b)
        # if filelength <= k:
        #     k= int(0.5*filelength)
        
        # k = int(max(1,per_k*filelength))
        k = min(k,filelength-1)
        print('filelength:',len(b))
        print('k:{}, z:{}'.format(k,z))
        # bp()
        if reco2num != 'None':
            try:
                n_clusters = int(reco2num_spk[i].split()[1])      
            except:
                n_clusters = 2
            n_clusters = min(n_clusters,len(clusterlen)) # whichever is minimum
            if f!=reco2num_spk[i].split()[0]:
                print('file mismatch',f,reco2num_spk[i].split()[0])
            threshold = None
            affinity = b.copy()

            if filelength > 0:
                if "baseline" in fold:
                    clus = mypic.PIC_org(n_clusters,clusterlen,labels,affinity,K=k,z=z)
                else:
                    clus = mypic.PIC_ami(n_clusters,clusterlen,labels,affinity,K=k,z=z)
            else:
                clus = mypic.PIC_callhome(n_clusters,clusterlen,labels,affinity,K=k,z=z)
               
            labelfull,clusterlen= clus.gacCluster()
            n_clusters = len(clusterlen)
            print("filename: {} n_clusters:{} clusterlen:{}".format(f,n_clusters,clusterlen))

        else:
            affinity = b.copy()
            n_clusters = 1  # atleast 1 speaker

            if filelength > 0:
                if "baseline" in fold:
                    clus = mypic.PIC_org_threshold(n_clusters,clusterlen,labels,affinity,threshold,K=k,z=z)
                else:
                    clus = mypic.PIC_ami_threshold(n_clusters,clusterlen,labels,affinity,threshold,K=k,z=z)
            else:
                clus = mypic.PIC_callhome_threshold(n_clusters,clusterlen,labels,affinity,threshold,K=k,z=z)
            labelfull,clusterlen= clus.gacCluster()
            n_clusters = len(clusterlen)
            print("filename: {} n_clusters:{} clusterlen:{}".format(f,n_clusters,clusterlen))
       
        uni1,method1=unique(labelfull,True)
        results_dict[f]=method1
    write_results_dict(results_dict, out_file,reco2utt)
    
    
def AHC_clustering_fine():
    args = setup()
    fold = args.score_path
    file_list = np.genfromtxt(args.score_file,dtype=str)
    out_file = args.out_file
    reco2utt = args.reco2utt
    reco2num = args.reco2num
    threshold=args.threshold
    lamda = args.lamda
    dataset = fold.split('/')[-3]
    print(threshold)
    if reco2num != 'None':
        reco2num_spk = open(args.reco2num).readlines()
    results_dict ={}
    for i,f in enumerate(file_list):
        print(f)

        if "baseline" in fold:
            b = np.load(fold+'/'+f+'.npy')
        else:
            # deepahcmodel = pickle.load(open(fold+'/'+f+'.pkl','rb'))
            # b = deepahcmodel['output']
            b = np.load(fold+'/'+f+'.npy')
            nframe  = b.shape[0]
            
            window_size = 2
            subsample_again = window_size
            mod_i = 0
            moving_averages = []
            b_new = b.copy()
            # b_new = b[::subsample_again]
            # b_new = b_new[:,::subsample_again]
           
            while mod_i < int(len(b)):        
                this_window = b[mod_i : mod_i + window_size]        
                window_average = np.mean(this_window,axis=0)       
                b_new[mod_i] = window_average
                this_window = b_new[:,mod_i : mod_i + window_size]        
                window_average = np.mean(this_window,axis=1)       
                b_new[:,mod_i] = window_average
                mod_i +=subsample_again
            b_new = b[::subsample_again]
            b_new = b_new[:,::subsample_again]
        b = b_new.copy()
        clusterlen = [1]*b.shape[0]
        labels = np.arange(b.shape[0])
       
        if reco2num != 'None':
            n_clusters = int(reco2num_spk[i].split()[1])      
            n_clusters = min(n_clusters,len(clusterlen)) # whichever is minimum
            if f!=reco2num_spk[i].split()[0]:
                print('file mismatch',f,reco2num_spk[i].split()[0])
            threshold = None
        else:
            n_clusters = None         
        clus =clustering(n_clusters,clusterlen,lamda,labels,dist=threshold)
        _,_,output_new = clus.initialize_clusters(b)
        
        labelfull,clusterlen,_=clus.Ahc_full(output_new)
        n_clusters = len(clusterlen)
        print("filename: {} n_clusters:{} clusterlen:{}".format(f,n_clusters,clusterlen))
        res = list(itertools.chain.from_iterable(itertools.repeat(i, subsample_again) 
                                           for i in labelfull)) 

        labelfull = np.array(res[:nframe])
        
        uni1,method1=unique(labelfull,True)
        results_dict[f]=method1     

    write_results_dict(results_dict, out_file,reco2utt)
    
def AHC_clustering_fusion():
    args = setup()
    fold = args.score_path
    file_list = np.genfromtxt(args.score_file,dtype=str)
    out_file = args.out_file
    reco2utt = args.reco2utt
    reco2num = args.reco2num
    threshold=args.threshold
    lamda = args.lamda
    weight = args.weight
    dataset = fold.split('/')[-3]
    print(threshold)
    if reco2num != 'None':
        reco2num_spk = open(args.reco2num).readlines()
    results_dict ={}
    for i,f in enumerate(file_list):
        print(f)

        if "baseline" in fold:
            b_inp = np.load(fold+'/'+f+'.npy')
        else:
            if os.path.isfile(fold+'/'+f+'.pkl'):
                deepahcmodel = pickle.load(open(fold+'/'+f+'.pkl','rb'))
                b = deepahcmodel['output']
               
            else:
                b = np.load(fold+'/'+f+'.npy')
        fold_plda='/data1/prachis/Dihard_2020/SSC/plda_pca_baseline/dihard_dev_2020_track1_wide_scores/plda_scores'
        b_plda = np.load(fold_plda+'/'+f+'.npy')
        if weight == 1.0:
            b = b_inp.copy()
        else:
            b = weight *b_inp+(1-weight)*b_plda
        clusterlen = [1]*b.shape[0]
        labels = np.arange(b.shape[0])
       
        if reco2num != 'None':
            n_clusters = int(reco2num_spk[i].split()[1])      
            n_clusters = min(n_clusters,len(clusterlen)) # whichever is minimum
            if f!=reco2num_spk[i].split()[0]:
                print('file mismatch',f,reco2num_spk[i].split()[0])
            threshold = None
        else:
            n_clusters = None         
        clus =clustering(n_clusters,clusterlen,lamda,labels,dist=threshold)
        # _,_,output_new = clus.initialize_clusters(b)
        
        labelfull,_,mergeind=clus.Ahc_full(b)
        uni1,method1=unique(labelfull,True)
        results_dict[f]=method1     

    write_results_dict(results_dict, out_file,reco2utt)

def AHC_clustering():
    args = setup()
    fold = args.score_path
    file_list = np.genfromtxt(args.score_file,dtype=str)
    out_file = args.out_file
    reco2utt = args.reco2utt
    reco2num = args.reco2num
    threshold=args.threshold
    lamda = args.lamda
    dataset = fold.split('/')[-3]
    print(threshold)
    if reco2num != 'None':
        reco2num_spk = open(args.reco2num).readlines()
    results_dict ={}
    for i,f in enumerate(file_list):
        print(f)

        if "baseline" in fold:
            b = np.load(fold+'/'+f+'.npy')
        else:
            if os.path.isfile(fold+'/'+f+'.pkl'):
                deepahcmodel = pickle.load(open(fold+'/'+f+'.pkl','rb'))
                b = deepahcmodel['output']   
            else:
                b = np.load(fold+'/'+f+'.npy')
        # fold_plda='/data1/prachis/Dihard_2020/SSC/plda_pca_baseline/dihard_dev_2020_track1_scores/plda_scores'
        # b_plda = np.load(fold_plda+'/'+f+'.npy')
        # b = (b+b_plda)/2
        clusterlen = [1]*b.shape[0]
        labels = np.arange(b.shape[0])
       
        if reco2num != 'None':
            n_clusters = int(reco2num_spk[i].split()[1])      
            n_clusters = min(n_clusters,len(clusterlen)) # whichever is minimum
            if f!=reco2num_spk[i].split()[0]:
                print('file mismatch',f,reco2num_spk[i].split()[0])
            threshold = None
        else:
            n_clusters = None         
        clus =clustering(n_clusters,clusterlen,labels,dist=threshold)
        # _,_,output_new = clus.initialize_clusters(b)
        
        labelfull,_,mergeind=clus.Ahc_full(b)
        uni1,method1=unique(labelfull,True)
        results_dict[f]=method1     

    write_results_dict(results_dict, out_file,reco2utt)


if __name__ == "__main__":
    args = setup()
    fold = args.score_path
    if args.clustering == "PIC":
        # PIC_clustering_modified()
        # PIC_clustering_fine()
        PIC_clustering()
        # PIC_clustering_fusion()
        # PIC_clustering_modified_withfinecoarse()
        # PIC_clustering_modified_withahcinit_finecoarse()
    else:
        # GDL_clustering_withlabels()
        # AHC_clustering_fusion()
        AHC_clustering()
        # AHC_clustering_fine()
        


