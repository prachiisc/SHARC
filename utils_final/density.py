#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file re-uses implementation from https://github.com/yl-1993/learn-to-cluster
"""

import numpy as np
from tqdm import tqdm
from itertools import groupby
import torch
from pdb import set_trace as bp
__all__ = ['density_estimation', 'density_to_peaks', 'density_to_peaks_vectorize','update_clusters_density_estimation','density_estimation_overlap']

def density_estimation(dists, nbrs, labels, **kwargs):
    ''' use supervised density defined on neigborhood
    '''
    num, k_knn = dists.shape
    conf = np.ones((num, ), dtype=np.float32)
    
    if len(labels.shape) > 1:
        
        ind_array = labels[:,0][nbrs] == np.expand_dims(labels[:,0], 1).repeat(k_knn, 1)
        for i in range(labels.shape[1]):
            for j in range(labels.shape[1]):
                if i==j:
                    if i==0:
                        continue
                    else:
                        l1 = labels[:,i].copy()
                        l1[l1==-1] = -100 # change the labels if the second entry is -1 which indicates no second speaker
                                            # so that two labels with -1 value doesn't get similarity 1
                        ind_array += l1[nbrs] == np.expand_dims(labels[:,j], 1).repeat(k_knn, 1)
                else:
                    if i>0 and j>0:
                        l1 = labels[:,i].copy()
                        l1[l1==-1] = -100 # change the labels if the second entry is -1 which indicates no second speaker
                                            # so that two labels with -1 value doesn't get similarity 1
                        ind_array += l1[nbrs] == np.expand_dims(labels[:,j], 1).repeat(k_knn, 1)
                    else:
                        ind_array += labels[:,i][nbrs] == np.expand_dims(labels[:,j], 1).repeat(k_knn, 1)
        
    else:
        ind_array = labels[nbrs] == np.expand_dims(labels, 1).repeat(k_knn, 1) # orginal case
    
    pos = ((1-dists[:,1:]) * ind_array[:,1:]).sum(1)
    neg = ((1-dists[:,1:]) * (1-ind_array[:,1:])).sum(1)
    conf = (pos - neg) * conf

    if len(labels.shape) > 1 :
        if kwargs['density_weighting'] is True:
            overlap_ind = np.where(labels[:,1]!=-1)[0] # overlapping segments
            conf[overlap_ind] /=2
    conf /= (k_knn - 1)
    return conf

def density_estimation_overlap(dists, nbrs, labels, **kwargs):
    ''' use supervised density defined on neigborhood
    '''
    num, k_knn = dists.shape
    conf = np.ones((num, ), dtype=np.float32)
    
    if len(labels.shape) > 1:
        
        ind_array = labels[:,0][nbrs] == np.expand_dims(labels[:,0], 1).repeat(k_knn, 1)
        for i in range(labels.shape[1]):
            for j in range(labels.shape[1]):
                if i==j:
                    if i==0:
                        continue
                    else:
                        l1 = labels[:,i].copy()
                        l1[l1==-1] = -100 # change the labels if the second entry is -1 which indicates no second speaker
                                            # so that two labels with -1 value doesn't get similarity 1
                        ind_array += l1[nbrs] == np.expand_dims(labels[:,j], 1).repeat(k_knn, 1)
                else:
                    if i>0 and j>0:
                        l1 = labels[:,i].copy()
                        l1[l1==-1] = -100 # change the labels if the second entry is -1 which indicates no second speaker
                                            # so that two labels with -1 value doesn't get similarity 1
                        ind_array += l1[nbrs] == np.expand_dims(labels[:,j], 1).repeat(k_knn, 1)
                    else:
                        ind_array += labels[:,i][nbrs] == np.expand_dims(labels[:,j], 1).repeat(k_knn, 1)
        
    else:
        ind_array = labels[nbrs] == np.expand_dims(labels, 1).repeat(k_knn, 1) # orginal case
    
    pos = ((1-dists[:,1:]) * ind_array[:,1:]).sum(1)
    neg = ((1-dists[:,1:]) * (1-ind_array[:,1:])).sum(1)
    conf = (pos - neg) * conf
    if len(labels.shape) > 1:
        overlap_ind = np.where(labels[:,1]!=-1)[0] # overlapping segments
        conf[overlap_ind] /=2
    conf /= (k_knn - 1)
    return conf,ind_array


def update_clusters_density_estimation(dists, edge_predictions, cluster_matrices):
        ''' use supervised density defined on neigborhood
        use the cluster information to compute cluster specific density for each node
        '''
        
        num, k_knn = dists.shape        
        conf_full = []
        for cluster in cluster_matrices:
            conf = np.ones((num, ), dtype=np.float32)
            # nbrs_c = nbrs[nbrs==cluster]
            # ind_array = labels[:,0][nbrs_c] == np.expand_dims(labels[:,0], 1).repeat(k_knn, 1)
        
            pos = ((1-dists[:,1:]) * edge_predictions*cluster).sum(1)
            neg = ((1-dists[:,1:]) * (1-edge_predictions)*cluster).sum(1)
            conf = (pos - neg) * conf
            conf /= (k_knn - 1)
            conf_full.append(conf)
            # prob = edges.data['prob_conn']
            # res = edges.data['raw_affine'] * (prob[:, 1] - prob[:, 0])
        conf_full = np.array(conf_full).T
        return conf_full

def density_to_peaks_vectorize(dists, nbrs, density, max_conn=1, name = ''):
    # just calculate 1 connectivity
    assert dists.shape[0] == density.shape[0]
    assert dists.shape == nbrs.shape

    num, k = dists.shape

    if name == 'gcn_feat':
        include_mask = nbrs != np.arange(0, num).reshape(-1, 1)
        secondary_mask = np.sum(include_mask, axis = 1) == k # TODO: the condition == k should not happen as distance to the node self should be smallest, check for numerical stability; TODO: make top M instead of only supporting top 1
        include_mask[secondary_mask, -1] = False
        nbrs_exclude_self = nbrs[include_mask].reshape(-1, k-1) # (V, 79)
        dists_exclude_self = dists[include_mask].reshape(-1, k-1) # (V, 79)
    else:
        include_mask = nbrs != np.arange(0, num).reshape(-1, 1)
        nbrs_exclude_self = nbrs[include_mask].reshape(-1, k-1) # (V, 79)
        dists_exclude_self = dists[include_mask].reshape(-1, k-1) # (V, 79)

    compare_map = density[nbrs_exclude_self] > density.reshape(-1, 1)
    peak_index = np.argmax(np.where(compare_map, 1, 0), axis = 1) # (V,)
    compare_map_sum = np.sum(compare_map.cpu().data.numpy(), axis=1) # (V,)

    dist2peak = {i: [] if compare_map_sum[i] == 0 else [dists_exclude_self[i, peak_index[i]]] for i in range(num)}
    peaks = {i: [] if compare_map_sum[i] == 0 else [nbrs_exclude_self[i, peak_index[i]]] for i in range(num)}

    return dist2peak, peaks

def density_to_peaks(dists, nbrs, density, max_conn=1, sort='dist'):
    # Note that dists has been sorted in ascending order
    assert dists.shape[0] == density.shape[0]
    assert dists.shape == nbrs.shape

    num, _ = dists.shape
    dist2peak = {i: [] for i in range(num)}
    peaks = {i: [] for i in range(num)}

    for i, nbr in tqdm(enumerate(nbrs)):
        nbr_conf = density[nbr]
        for j, c in enumerate(nbr_conf):
            nbr_idx = nbr[j]
            if i == nbr_idx or c <= density[i]:
                continue
            dist2peak[i].append(dists[i, j])
            peaks[i].append(nbr_idx)
            if len(dist2peak[i]) >= max_conn:
                break

    return dist2peak, peaks
