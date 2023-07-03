#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file re-uses implementation from https://github.com/yl-1993/learn-to-cluster
"""

import os
import time
import json
import pickle
import random
import numpy as np

class TextColors:
    HEADER = '\033[35m'
    OKBLUE = '\033[34m'
    OKGREEN = '\033[32m'
    WARNING = '\033[33m'
    FATAL = '\033[31m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Timer():
    def __init__(self, name='task', verbose=True):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose:
            print('[Time] {} consumes {:.4f} s'.format(
                self.name,
                time.time() - self.start))
        return exc_type is None

def set_random_seed(seed, cuda=False):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

def l2norm(vec):
    vec /= np.linalg.norm(vec, axis=1).reshape(-1, 1)
    return vec

def is_l2norm(features, size):
    rand_i = random.choice(range(size))
    norm_ = np.dot(features[rand_i, :], features[rand_i, :])
    return abs(norm_ - 1) < 1e-6

def is_spmat_eq(a, b):
    return (a != b).nnz == 0

def aggregate(features, adj, times):
    dtype = features.dtype
    for i in range(times):
        features = adj * features
    return features.astype(dtype)

def mkdir_if_no_exists(path, subdirs=[''], is_folder=False):
    if path == '':
        return
    for sd in subdirs:
        if sd != '' or is_folder:
            d = os.path.dirname(os.path.join(path, sd))
        else:
            d = os.path.dirname(path)
        if not os.path.exists(d):
            os.makedirs(d)

def stop_iterating(current_l, total_l, early_stop, num_edges_add_this_level, num_edges_add_last_level, knn_k):
    # Stopping rule 1: run all levels
    if current_l == total_l - 1:
        return True
    # Stopping rule 2: no new edges
    if num_edges_add_this_level == 0:
        return True
    # Stopping rule 3: early stopping, two levels start to produce similar numbers of edges
    if early_stop and float(num_edges_add_last_level) / num_edges_add_this_level < knn_k - 1:
        return True
    return False

from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info

class save_dgl_graphs():
    def __init__(self, save_path):
        # self.graphs = graphs
        self.save_path = save_path

    def save(self,recid,graphs):
        # save graphs and labels
        graph_path = os.path.join(self.save_path,recid + '_dgl_graph.bin')
        # save_graphs(graph_path, self.graphs, {'labels': self.labels})
        save_graphs(graph_path,graphs)
        # save other information in python dict
        # info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
        # save_info(info_path, {'num_classes': self.num_classes})

    def load(self,recid):
        # load processed data from directory `self.save_path`
        graph_path = os.path.join(self.save_path, recid + '_dgl_graph.bin')
        # self.graphs, label_dict = load_graphs(graph_path)
        graphs = load_graphs(graph_path)
        # self.labels = label_dict['labels']
        # info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
        # self.num_classes = load_info(info_path)['num_classes']
        return graphs
        
    def has_cache(self,recid):
        # check whether there are processed data in `self.save_path`
        graph_path = os.path.join(self.save_path, recid + '_dgl_graph.bin')
        # info_path = os.path.join(self.save_path, recid + '_info.pkl')
        # return os.path.exists(graph_path) and os.path.exists(info_path)
        return os.path.exists(graph_path)