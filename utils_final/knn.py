#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file re-uses implementation from https://github.com/yl-1993/learn-to-cluster
"""

import os
import math
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

from utils_final import Timer
from .faiss_search import faiss_search_knn
from pdb import set_trace as bp

__all__ = [
    'knn_faiss', 'knn_faiss_gpu',
    'fast_knns2spmat', 'build_knns',
    'knns2ordered_nbrs','knns_from_graph',
    'get_cosine_mat', 'get_PLDA_mat' 
]

def knns2ordered_nbrs(knns, sort=True):
    if isinstance(knns, list):
        knns = np.array(knns)
    nbrs = knns[:, 0, :].astype(np.int32)
    dists = knns[:, 1, :]
    if sort:
        # sort dists from low to high
        nb_idx = np.argsort(dists, axis=1)
        idxs = np.arange(nb_idx.shape[0]).reshape(-1, 1)
        dists = dists[idxs, nb_idx]
        nbrs = nbrs[idxs, nb_idx]
    return dists, nbrs

def fast_knns2spmat(knns, k, th_sim=0, use_sim=True, fill_value=None):
    # convert knns to symmetric sparse matrix
    from scipy.sparse import csr_matrix
    eps = 1e-5
    n = len(knns)
    if isinstance(knns, list):
        knns = np.array(knns)
    if len(knns.shape) == 2:
        # knns saved by hnsw has different shape
        n = len(knns)
        ndarr = np.ones([n, 2, k])
        ndarr[:, 0, :] = -1  # assign unknown dist to 1 and nbr to -1
        for i, (nbr, dist) in enumerate(knns):
            size = len(nbr)
            assert size == len(dist)
            ndarr[i, 0, :size] = nbr[:size]
            ndarr[i, 1, :size] = dist[:size]
        knns = ndarr
    nbrs = knns[:, 0, :]
    dists = knns[:, 1, :]
    assert -eps <= dists.min() <= dists.max(
    ) <= 1 + eps, "min: {}, max: {}".format(dists.min(), dists.max())
    if use_sim:
        sims = 1. - dists
    else:
        sims = dists
    if fill_value is not None:
        print('[fast_knns2spmat] edge fill value:', fill_value)
        sims.fill(fill_value)
    row, col = np.where(sims >= th_sim)
    # remove the self-loop
    idxs = np.where(row != nbrs[row, col])
    row = row[idxs]
    col = col[idxs]
    data = sims[row, col]
    col = nbrs[row, col]  # convert to absolute column
    assert len(row) == len(col) == len(data)
    spmat = csr_matrix((data, (row, col)), shape=(n, n))
    return spmat

def get_PLDA_mat(feats,pldamodel=None,target=0,**kwargs):
    from models_train_ssc_plda import weight_initialization
    import pickle as pkl
    import torch
    # final=1,lnorm=1,filepca=1,ldatransform=None,temporal=0
    device='cpu'
    inpdata = feats[np.newaxis]
    inpdata = torch.from_numpy(inpdata)
    if pldamodel is None:
        pldamodel = "/data1/prachis/Dihard_2020/gae-pytorch/gae/lists/lib_vox_tr_all_gnd_0.75s/plda_lib_vox_tr_all_gnd_0.75s.pkl"

    pldamodel = pkl.load(open(pldamodel,'rb'))
    
    pca_dim=30
    xvecD=feats.shape[1]
    net_init = weight_initialization(pldamodel,pca_dimension=pca_dim,device=device,**kwargs)
    model_init = net_init.to(device)
    
    affinity_init,_,_,inpdata = model_init.compute_plda_affinity_matrix(pldamodel,inpdata,target,**kwargs)
    output_model = affinity_init.detach().cpu().numpy()[0]

    sig_v = np.vectorize(sigmoid)
    PLDA_mat = sig_v(output_model)
    PLDA_mat = PLDA_mat / np.max(PLDA_mat) 
    # weighting for temporal weightage
    
    if 'temporal' in kwargs:
        temp_param = kwargs['temp_param']
        if temp_param is None:
            neb = 5
            beta1 = 0.95
        else:
            temp_param = kwargs['temp_param']
            neb,beta1 = temp_param.split(',')
            neb = int(neb)
            beta1 = float(beta1)
        N = PLDA_mat.shape[0]
        if 'temp_labels' in kwargs:
            temp_labels = kwargs['temp_labels']
            if temp_labels is None:
                temp_labels = np.arange(N)
            toep = np.abs(temp_labels.reshape(N,1)-temp_labels.reshape(1,N))
        else:
            toep = np.abs(np.arange(N).reshape(N,1)-np.arange(N).reshape(1,N))
        toep[toep>neb] = neb
        weighting = beta1**(toep)
        PLDA_mat = weighting*PLDA_mat
    feats = inpdata.detach().cpu().numpy()[0]
    return PLDA_mat, feats

def sigmoid(x):
    return 1/(1+np.exp(-x))

def PLDA_knn(PLDA_mat, k, mode, labels=None,temp_param=None):
    if "x4" in mode:
        PLDA_mat = 4 * PLDA_mat
    elif "x2" in mode:
        PLDA_mat = 2 * PLDA_mat
    #print(np.max(PLDA_mat),np.min(PLDA_mat))
    # sig_v = np.vectorize(sigmoid)
    # PLDA_mat = sig_v(output_model)
    # PLDA_mat = PLDA_mat / np.max(PLDA_mat) 

    if "GT" in mode:
        gt_mat = np.reshape([int(l1 == l2)  for l1 in labels for l2 in labels],(len(labels),len(labels)))  
        PLDA_mat = (PLDA_mat + gt_mat)/2     
    PLDA_mat = 1-PLDA_mat
    #print(np.max(PLDA_mat),np.min(PLDA_mat))
    PLDA_mat[np.diag_indices(PLDA_mat.shape[0])] = 0
    dists = np.sort(PLDA_mat,axis=1)
    nbrs = np.argsort(PLDA_mat,axis=1)
    dists = dists[:,:k]
    nbrs = nbrs[:,:k]
    knn = [(np.array(nbr, dtype=np.int32),
                          np.array(dist, dtype=np.float32))
                         for nbr, dist in zip(nbrs, dists)]
    return knn

def get_cosine_mat(feats,pldamodel=None,target=0,**kwargs):
    from models_train_ssc_plda import weight_initialization
    import pickle as pkl
    import torch
    device='cpu'
    inpdata = feats[np.newaxis]
    inpdata = torch.from_numpy(inpdata)
    #pldadataset='dihard_dev_2020_track1_fbank_jhu_wide'
    #pldamodel= '/data1/prachis/Dihard_2020/SSC/lists/{0}/plda_{0}.pkl'.format(pldadataset)
    #pldamodel = 'plda_{0}.pkl'.format(pldadataset)
    if pldamodel is None:
        pldamodel = "/data1/prachis/Dihard_2020/gae-pytorch/gae/lists/ami_sdm_train_gnd/plda_ami_sdm_train_gnd.pkl"
    
    pldamodel = pkl.load(open(pldamodel,'rb'))
    pca_dim=30
    xvecD=feats.shape[1]
    net_init = weight_initialization(pldamodel,dimension=xvecD,pca_dimension=pca_dim,device=device,**kwargs)
    model_init = net_init.to(device)
    affinity_init, inpdata = model_init.compute_affinity_matrix(inpdata)
    output_model = affinity_init.detach().cpu().numpy()[0]
   

    if 'temporal' in kwargs:
        temp_param = kwargs['temp_param']
        if temp_param is None:
            neb = 5
            beta1 = 0.95
        else:
            temp_param = kwargs['temp_param']
            neb,beta1 = temp_param.split(',')
            neb = int(neb)
            beta1 = float(beta1)
        N= output_model.shape[0]
        toep = np.abs(np.arange(N).reshape(N,1)-np.arange(N).reshape(1,N))
        toep[toep>neb] = neb
        weighting = beta1**(toep)
        output_model = weighting*output_model
    feats = inpdata.detach().cpu().numpy()[0]
    return output_model, feats

def get_cosine_mat_overlap(feats,pldamodel=None,target=0,**kwargs):
    from models_train_ssc_plda import weight_initialization
    import pickle as pkl
    import torch
    # device='cpu'
    # inpdata = feats[np.newaxis]
    # inpdata = torch.from_numpy(inpdata)

    if pldamodel is None:
        l2_norms = np.norm(feats, axis=1,keepdim=True)
        X_normalized = inpdata / l2_norms
        cosine_similarities = X_normalized @ X_normalized.T
        output_model = (cosine_similarities + 1.0) / 2.0
    else:
        pldamodel = pkl.load(open(pldamodel,'rb'))
        pca_dim=30
        xvecD=feats.shape[1]
        net_init = weight_initialization(pldamodel,dimension=xvecD,pca_dimension=pca_dim,device=device,**kwargs)
        model_init = net_init.to(device)
        affinity_init, inpdata = model_init.compute_affinity_matrix(inpdata)
        output_model = affinity_init.detach().cpu().numpy()[0]

    if 'temporal' in kwargs:
        temp_param = kwargs['temp_param']
        if temp_param is None:
            neb = 5
            beta1 = 0.95
        else:
            temp_param = kwargs['temp_param']
            neb,beta1 = temp_param.split(',')
            neb = int(neb)
            beta1 = float(beta1)
        N= output_model.shape[0]
        toep = np.abs(np.arange(N).reshape(N,1)-np.arange(N).reshape(1,N))
        toep[toep>neb] = neb
        weighting = beta1**(toep)
        output_model = weighting*output_model
    return output_model


def cosine_knn(cosine_sim_mat, k, mode, labels=None):
     
    #if np.min(cosine_sim_mat) < 0:
    #    cosine_sim_mat = (cosine_sim_mat - np.min(cosine_sim_mat)) / (np.max(cosine_sim_mat) - np.min(cosine_sim_mat))
    #cosine_sim_mat = (cosine_sim_mat + 1)/2
    #cosine_sim_mat[cosine_sim_mat>1] = 1
    if "GT" in mode:
        gt_array = np.reshape([int(l1 == l2)  for l1 in labels for l2 in labels],(len(labels),len(labels)))
        cosine_sim_mat = (cosine_sim_mat+ind_array) / 2
    #sims_new = np.sort(ind_array,axis=1)[:,::-1]
    #nbrs_new = np.argsort(ind_array,axis=1)[:,::-1]
    #sims_new = sims_new[:,:k]
    #nbrs_new = nbrs_new[:,:k]
    #sims = [cosine_sim_mat[i,nbrs_new[i]] for i in range(nbrs_new.shape[0])]
    sims = np.sort(cosine_sim_mat,axis=1)[:,::-1]
    nbrs = np.argsort(cosine_sim_mat,axis=1)[:,::-1]
    sims = sims[:,:k]
    nbrs = nbrs[:,:k]
    #if 0np.min(sims) < 0:
    #    sims = (sims - np.min(sims)) / (np.max(sims) - np.min(sims))


    knn = [(np.array(nbr, dtype=np.int32),
                          1 - np.array(sim, dtype=np.float32))
                         for nbr, sim in zip(nbrs, sims)]

    #knn_new = [(np.array(nbr, dtype=np.int32),
    #                      1 - np.array(sim, dtype=np.float32))
    #                     for nbr, sim in zip(nbrs_new, sims_new)]
    return knn

def build_knns(feats,
               k,
               knn_method,
               mode,
               labels,
               dump=True,
               temp_param=None):
    with Timer('build index'):
        if knn_method == 'faiss':
            index = knn_faiss(feats, k, omp_num_threads=None)
        elif knn_method == 'faiss_gpu':
            index = knn_faiss_gpu(feats, k)
        elif knn_method == 'PLDA':
            knns = PLDA_knn(feats, k, mode, labels,temp_param)
            return knns
        elif knn_method == 'cosine':
            knns = cosine_knn(feats, k, mode, labels)
            return knns
        else:
            raise KeyError(
                'Only support faiss and faiss_gpu currently ({}).'.format(knn_method))
        knns = index.get_knns()
    return knns


def knns_from_graph(bipartite, k):
    col,row = bipartite.edges() 
    data = 1 - ((bipartite.edata["prob_conn"][:,1] +  bipartite.edata["raw_affine"][:]) / 2)
    col = col.detach().numpy()
    row = row.detach().numpy()
    data = data.detach().numpy()
    idx = np.argsort(row)
    col = col[idx]
    data = data[idx]
    nbrs = np.reshape(col,(-1,k-1))
    dists = np.reshape(data,(-1,k-1))
    idx = np.argsort(dists,1)
    for i in range(nbrs.shape[0]):  
        nbrs[i] = nbrs[i,idx[i]]
    dists = np.sort(dists,1)
    nbrs = np.c_[np.arange(nbrs.shape[0]),nbrs]
    dists = np.c_[np.zeros(dists.shape[0]),dists]
    k = int(np.ceil(0.5*k))
    dists = dists[:,:k]
    nbrs = nbrs[:,:k]
    knn = [(np.array(nbr, dtype=np.int32),
                          np.array(dist, dtype=np.float32))
                         for nbr, dist in zip(nbrs, dists)]
    return knn


class knn():
    def __init__(self, feats, k, index_path='', verbose=True):
        pass

    def filter_by_th(self, i):
        th_nbrs = []
        th_dists = []
        nbrs, dists = self.knns[i]
        for n, dist in zip(nbrs, dists):
            if 1 - dist < self.th:
                continue
            th_nbrs.append(n)
            th_dists.append(dist)
        th_nbrs = np.array(th_nbrs)
        th_dists = np.array(th_dists)
        return (th_nbrs, th_dists)

    def get_knns(self, th=None):
        if th is None or th <= 0.:
            return self.knns
        # TODO: optimize the filtering process by numpy
        # nproc = mp.cpu_count()
        nproc = 1
        with Timer('filter edges by th {} (CPU={})'.format(th, nproc),
                   self.verbose):
            self.th = th
            self.th_knns = []
            tot = len(self.knns)
            if nproc > 1:
                pool = mp.Pool(nproc)
                th_knns = list(
                    tqdm(pool.imap(self.filter_by_th, range(tot)), total=tot))
                pool.close()
            else:
                th_knns = [self.filter_by_th(i) for i in range(tot)]
            return th_knns

class knn_faiss(knn):
    def __init__(self,
                 feats,
                 k,
                 nprobe=128,
                 omp_num_threads=None,
                 rebuild_index=True,
                 verbose=True,
                 **kwargs):
        import faiss
        if omp_num_threads is not None:
            faiss.omp_set_num_threads(omp_num_threads)
        self.verbose = verbose
        with Timer('[faiss] build index', verbose):
            feats = feats.astype('float32')
            size, dim = feats.shape
            index = faiss.IndexFlatIP(dim)
            index.add(feats)
        with Timer('[faiss] query topk {}'.format(k), verbose):
            sims, nbrs = index.search(feats, k=k)
            if np.min(sims) < 0:
                sims = (sims - np.min(sims)) / (np.max(sims) - np.min(sims))
            self.knns = [(np.array(nbr, dtype=np.int32),
                          1 - np.array(sim, dtype=np.float32))
                         for nbr, sim in zip(nbrs, sims)]

class knn_faiss_gpu(knn):
    def __init__(self,
                 feats,
                 k,
                 nprobe=128,
                 num_process=4,
                 is_precise=True,
                 sort=True,
                 verbose=True,
                 **kwargs):
        with Timer('[faiss_gpu] query topk {}'.format(k), verbose):
            dists, nbrs = faiss_search_knn(feats,
                                           k=k,
                                           nprobe=nprobe,
                                           num_process=num_process,
                                           is_precise=is_precise,
                                           sort=sort,
                                           verbose=verbose)
            sims = 1 - dists
            if np.min(sims) < 0:
                sims = (sims - np.min(sims)) / (np.max(sims) - np.min(sims))
            self.knns = [(np.array(nbr, dtype=np.int32),
                          1 - np.array(sim, dtype=np.float32))
                         for nbr, sim in zip(nbrs, sims)]
