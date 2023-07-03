import numpy as np
import pickle

import dgl
import torch
from pdb import set_trace as bp

# from utils_final import (build_knns, fast_knns2spmat, row_normalize, knns2ordered_nbrs,
#                    density_estimation, sparse_mx_to_indices_values, l2norm,
#                    decode, build_next_level, get_cosine_mat, get_PLDA_mat, build_next_level_e2e)

from utils_final import *


class SHARCDataset(object):
    def __init__(self, features, labels, mode, cluster_features=None, k=10, levels=1, affinity_mat=None,
                 must_run=False, feats_norm = 0,
                 pldamodel=None,ldatransform=None, # must_run=True return graph no matter edges are present or not
                 temp_param = None,
                 temp_labels=None, maxfeatureshape = 20
                ):
        
        self.k = k
        self.gs = []
        self.nbrs = []
        self.dists = []
        self.levels = levels
        self.numspk = np.unique(labels).size
        self.pldamodel = pldamodel
        if 'target' in mode:
            target=1
        else:
            target = 0
        
        # Initialize features and labels
        if "rec_aff" in mode:
            if "PLDA" in mode or "cosine" in mode:
                copy_feats = features.copy()
        
        if affinity_mat is None:
            if "PLDA" in mode:
                if "proc_feats" in mode:
                    affinity_mat, features = get_PLDA_mat(features,pldamodel=self.pldamodel)
                elif "proc_feats_whiten" in mode:
                    affinity_mat, features = get_PLDA_mat(features,pldamodel=self.pldamodel,final=0)
                    # bp()
                elif "proc_feats_whiten_no_norm" in mode:
                    affinity_mat, features = get_PLDA_mat(features,pldamodel=self.pldamodel,final=0,lnorm=0)
                    # bp()
                elif "proc_feats_PLDA" in mode:
                    # bp()
                    affinity_mat, features = get_PLDA_mat(features,pldamodel=self.pldamodel,final=2)
                elif "proc_feats_PLDA_nofilepca" in mode:
                        # bp()
                    affinity_mat, features = get_PLDA_mat(features,pldamodel=self.pldamodel,final=2,filepca=0)
                elif "proc_feats_LDA_nofilepca" in mode:
                   
                    affinity_mat, features = get_PLDA_mat(features,pldamodel=self.pldamodel,final=0,filepca=0,ldatransform=ldatransform)
                else:
                    
                    if 'temporal' in mode:
                        affinity_mat, _ = get_PLDA_mat(features,pldamodel=self.pldamodel,target=target,temporal=1,temp_param=temp_param,temp_labels=temp_labels)
                    else:
                        affinity_mat, _ = get_PLDA_mat(features,pldamodel=self.pldamodel,target=target)

            elif "cosine" in mode:
                if "proc_feats" in mode:
                    affinity_mat, features = get_cosine_mat(features)
                else:
                    if 'temporal' in mode:
                        affinity_mat, _ = get_cosine_mat(features,temporal=1,temp_param=temp_param)
                    else:
                        affinity_mat, _  = get_cosine_mat(features)

            #features = l2norm(features.astype('float32'))
        
        if feats_norm and ("proc_feats" not in mode or "proc_feats_whiten" not in mode):
            features = l2norm(features.astype('float32'))        
        
        if cluster_features is None:
            cluster_features = features
            
        global_features = features.copy()
        global_num_nodes = features.shape[0]
        global_edges = ([], [])
        global_peaks = np.array([], dtype=np.long)
        ids = np.arange(global_num_nodes)
        prev_num_nodes = global_num_nodes
        
        # Recursive graph construction
        for lvl in range(self.levels):
            if "test" in mode:
                if "heir" in mode:
                    if features.shape[0] >= 10:
                        if np.ceil(0.3*features.shape[0]) <= self.k:
                            self.k = int(np.ceil(0.3*features.shape[0]))
                    else:
                        break
                else:
                    if features.shape[0] <= self.k:
                        if features.shape[0] > maxfeatureshape or must_run:
                            self.k = features.shape[0]
                        else:
                            break
            elif "train" in mode:    
                if "heir" in mode:
                    if np.ceil(0.3*features.shape[0]) <= self.k:
                        self.k = int(np.ceil(0.3*features.shape[0]))
                else:
                    if features.shape[0] <= self.k:
                        self.k = features.shape[0]
            
            if "PLDA" in mode:
                knns = build_knns(affinity_mat, self.k, 'PLDA', mode, labels)
            elif "cosine" in mode:
                knns = build_knns(affinity_mat, self.k, 'cosine', mode, labels)
            else:
                knns = build_knns(features, self.k, 'faiss', mode, labels)
            
            dists, nbrs = knns2ordered_nbrs(knns)
            self.nbrs.append(nbrs)
            self.dists.append(dists)
            density = density_estimation(dists, nbrs, labels)
            

            g = self._build_graph(features, cluster_features, labels, density, knns)
            
            if g.number_of_edges() > 0 or must_run:
                self.gs.append(g)
            
            if lvl >= self.levels - 1:
                break
            if "train" in mode:
                if features.shape[0] <= self.numspk:
                    break
                
            print('spks: ',self.numspk,'features shape: ',features.shape[0])
            # Decode peak nodes
            new_pred_labels, peaks,\
                global_edges, global_pred_labels, global_peaks = decode(g, 0, 'sim', True, mode,
                                                                        ids, global_edges, global_num_nodes,
                                                                        global_peaks,device='cuda')
            if prev_num_nodes == len(peaks):
                print("Exiting due to failure in final cluster formation")
                break
            prev_num_nodes = len(peaks)
            ids = ids[peaks]
            
            features, labels, cluster_features = build_next_level(features, labels, peaks,
                                                                  global_features, global_pred_labels, global_peaks)
            if "rec_aff" not in mode:
                if "PLDA" in mode or "cosine" in mode:
                    affinity_mat = affinity_mat[peaks][:,peaks]
            else:
                if "PLDA" in mode or "cosine" in mode:
                    copy_feats = copy_feats[peaks]
                    if "PLDA" in mode:
                        if "proc_feats" in mode:
                            affinity_mat, features = get_PLDA_mat(copy_feats,self.pldamodel)
                            #features = l2norm(features.astype('float32'))
                        else:
                            # if 'temporal' in mode:
                            #     affinity_mat, _ = get_PLDA_mat(copy_feats,pldamodel=self.pldamodel,target=target,temporal=1,temp_param=temp_param)
                            # else:
                            affinity_mat, _ = get_PLDA_mat(copy_feats,self.pldamodel,target=target) # no temporal continuity in second level onwards
                    elif "cosine" in mode:
                        if "proc_feats" in mode:
                            affinity_mat, features = get_cosine_mat(copy_feats)
                            #features = l2norm(features.astype('float32'))
                        else:
                            affinity_mat, _ = get_cosine_mat(copy_feats)


    def _build_graph(self, features, cluster_features, labels, density, knns):
        adj = fast_knns2spmat(knns, self.k)
        # bp()
        adj, adj_row_sum = row_normalize(adj)
        indices, values, shape = sparse_mx_to_indices_values(adj)

        #adj_new = fast_knns2spmat(knns_new, self.k)
        #adj_new, adj_row_sum_new = row_normalize(adj_new)
        #indices_new, values_new, shape_new = sparse_mx_to_indices_values(adj_new)
        
        g = dgl.graph((indices[1], indices[0]),num_nodes=features.shape[0])
        g.ndata['features'] = torch.FloatTensor(features)
        g.ndata['cluster_features'] = torch.FloatTensor(cluster_features)
        g.ndata['labels'] = torch.LongTensor(labels)
        g.ndata['density'] = torch.FloatTensor(density)
        g.edata['affine'] = torch.FloatTensor(values)
        
        #g.edata['affine_new'] = torch.FloatTensor(values_new)
        # A Bipartite from DGL sampler will not store global eid, so we explicitly save it here
        g.edata['global_eid'] = g.edges(form='eid')
        g.ndata['norm'] = torch.FloatTensor(adj_row_sum)
        #g.ndata['norm_new'] = torch.FloatTensor(adj_row_sum_new)
        g.apply_edges(lambda edges: {'raw_affine': edges.data['affine'] / edges.dst['norm']})
        #g.apply_edges(lambda edges: {'raw_affine_new': edges.data['affine_new'] / edges.dst['norm_new']})
        g.apply_edges(lambda edges: {'labels_conn': (edges.src['labels'] == edges.dst['labels']).long()})
        g.apply_edges(lambda edges: {'mask_conn': (edges.src['density'] > edges.dst['density']).bool()})
        return g

    def __getitem__(self, index):
        assert index < len(self.gs)
        return self.gs[index]

    def __len__(self):
        return len(self.gs)

    def clusterwise_density_estimation(self,g,labels,clusters_ids=None):
        # affinity = g.adj(etype='affine')
        
        dists = self.dists[0]
        nbrs = self.nbrs[0]
        N, knn = dists.shape
        rows_nbrs = np.arange(N).repeat(knn)
        edge_pred = g.edata['prob_conn']
        
        # edge_pred_full = np.zeros(dists.shape)
        # edge_pred_full[:,1:] = edge_pred[:,1].reshape(N,knn-1) # only positive softmax prob
        edge_pred_full = edge_pred[:,1].reshape(N,knn-1)# only positive softmax prob
        edge_pred_full = edge_pred_full.numpy()
        
        if clusters_ids is None:
            clusters_ids = np.unique(labels)
        cluster_matrices = []
        for cluster_id in clusters_ids:
            cluster_labels = np.zeros((N,N))
            # cluster_labels_nbrs = np.zeros((N,knn-1))
            ids = np.where(labels==cluster_id)[0]
            
            cluster_labels[:,ids] = 1
            cluster_labels_nbrs  = cluster_labels[rows_nbrs,nbrs.reshape(-1,)].reshape(N,knn)
            cluster_labels_nbrs = cluster_labels_nbrs[:,1:]
            # cluster_labels_nbrs = cluster_labels[g.edges()[0],g.edges()[1]].reshape(N,knn-1)
            cluster_matrices.append(cluster_labels_nbrs) 
            
        # output_bipartite.apply_edges(self.pred_conn)
        
        # output_bipartite.edata['prob_conn'] = F.softmax(output_bipartite.edata['pred_conn'], dim=1)
        # output_bipartite.update_all(self.update_clusters_density_estimation, fn.mean('pred_den_msg', 'pred_den'))
        # edge_pred = 
        cluster_density = update_clusters_density_estimation(dists, edge_pred_full, cluster_matrices)
        g.ndata['cluster_density'] = torch.from_numpy(cluster_density)
        return g

 
    def clusterwise_gnddensity_estimation(self,g,labels):
        # affinity = g.adj(etype='affine')
        
        dists = self.dists[0]
        nbrs = self.nbrs[0]
        
        N, knn = dists.shape
        rows_nbrs = np.arange(N).repeat(knn)
        
        # labels  = g.ndata['labels']
        # edge_pred = g.edata['prob_conn']
        
        if len(labels.shape) > 1 or len(np.where(labels[:,1]>-1)[0])>1 :
            ind_array = np.expand_dims(labels[:,0], 1).repeat(knn, 1) == labels[:,0][nbrs]
            for i in range(labels.shape[1]):
                for j in range(labels.shape[1]):
                    if i==j:
                        if i==0:
                            continue
                        else:
                            l1 = labels[:,i].copy()
                            l1[l1==-1] = -100 # change the labels if the second entry is -1 which indicates no second speaker
                                                # so that two labels with -1 value doesn't get similarity 1
                            ind_array += np.expand_dims(labels[:,j], 1).repeat(knn, 1) == l1[nbrs]
                    else:
                        if i>0 and j>0:
                            l1 = labels[:,i].copy()
                            l1[l1==-1] = -100 # change the labels if the second entry is -1 which indicates no second speaker
                                                # so that two labels with -1 value doesn't get similarity 1
                            ind_array += l1[nbrs] == np.expand_dims(labels[:,j], 1).repeat(knn, 1)
                        else:
                            ind_array += labels[:,i][nbrs] == np.expand_dims(labels[:,j], 1).repeat(knn, 1)
            
        else:
            ind_array = labels[nbrs] == np.expand_dims(labels, 1).repeat(knn, 1) # orginal case
    
        
        clusters_ids = np.unique(labels[:,0])
        # rg = dgl.reorder_graph(g, edge_permute_algo='dst')
        cluster_matrices = []
        for cluster_id in clusters_ids:
            cluster_labels = np.zeros((N,N))
            # cluster_labels_nbrs = np.zeros((N,knn-1))
            ids = np.where(labels==cluster_id)[0]
            
            cluster_labels[:,ids] = 1
            cluster_labels_nbrs  = cluster_labels[rows_nbrs,nbrs.reshape(-1,)].reshape(N,knn)
            cluster_labels_nbrs = cluster_labels_nbrs[:,1:]
            # cluster_labels_nbrs = cluster_labels[g.edges()[0],g.edges()[1]].reshape(N,knn-1)
            cluster_matrices.append(cluster_labels_nbrs) 
            
        # output_bipartite.apply_edges(self.pred_conn)
        
        # output_bipartite.edata['prob_conn'] = F.softmax(output_bipartite.edata['pred_conn'], dim=1)
        # output_bipartite.update_all(self.update_clusters_density_estimation, fn.mean('pred_den_msg', 'pred_den'))
        
        cluster_density = update_clusters_density_estimation(dists, ind_array[:,1:], cluster_matrices)
        
        g.ndata['gnd_cluster_density'] = torch.from_numpy(cluster_density)
        return g


class SHARCDataset_withmfcc(object):
    def __init__(self, mfcc_xvec_features, features, labels, 
                mode, cluster_features=None, k=10, levels=1, affinity_mat=None, must_run=False,
                feats_norm = 0, 
                pldamodel=None,
                device='cpu',
                temp_param = None,
                temp_labels=None,maxfeatureshape=20
                ):
        self.k = k
        self.gs = []
        self.nbrs = []
        self.dists = []
        self.levels = levels
        self.numspk = np.unique(labels).size
        self.device = device
        self.pldamodel = pldamodel
        if 'target' in mode:
            target=1
        else:
            target = 0
        # Initialize features and labels
        if "rec_aff" in mode:
            if "PLDA" in mode or "cosine" in mode:
                copy_feats = features.copy()
        
        if affinity_mat is None:
            if "PLDA" in mode:
                if "proc_feats" in mode:
                    affinity_mat, features = get_PLDA_mat(features,self.pldamodel)
                else:
                    if 'temporal' in mode:
                        affinity_mat, _ = get_PLDA_mat(features,pldamodel=self.pldamodel,target=target,temporal=1,temp_param=temp_param,temp_labels=temp_labels)
                    else:
                        affinity_mat, _ = get_PLDA_mat(features,pldamodel=self.pldamodel,target=target)

            elif "cosine" in mode:
                if "proc_feats" in mode:
                    affinity_mat, features = get_cosine_mat(features)
                else:
                    affinity_mat, _ = get_cosine_mat(features)
            #features = l2norm(features.astype('float32'))
                
        if feats_norm:
            mfcc_xvec_features = mfcc_xvec_features/torch.norm(mfcc_xvec_features,dim=1).reshape(-1,1)
        if cluster_features is None:
            cluster_features = mfcc_xvec_features
        global_features = mfcc_xvec_features
        global_num_nodes = features.shape[0]
        global_edges = ([], [])
        global_peaks = np.array([], dtype=np.long)
        ids = np.arange(global_num_nodes)
        prev_num_nodes = global_num_nodes
        
        # Recursive graph construction
        for lvl in range(self.levels):
            if "test" in mode:
                if "heir" in mode:
                    if features.shape[0] >= 10:
                        if np.ceil(0.3*features.shape[0]) <= self.k:
                            self.k = int(np.ceil(0.3*features.shape[0]))
                    else:
                        break
                else:
                    if features.shape[0] <= self.k:
                        if features.shape[0] > maxfeatureshape or must_run:
                            self.k = features.shape[0]
                        else:
                            break
            elif "train" in mode:    
                if "heir" in mode:
                    if np.ceil(0.3*features.shape[0]) <= self.k:
                        self.k = int(np.ceil(0.3*features.shape[0]))
                else:
                    if features.shape[0] <= self.k:
                        self.k = features.shape[0]
            
            if "PLDA" in mode:
                knns = build_knns(affinity_mat, self.k, 'PLDA', mode, labels)
            elif "cosine" in mode:
                knns = build_knns(affinity_mat, self.k, 'cosine', mode, labels)
            else:
                knns = build_knns(features, self.k, 'faiss', mode, labels)
            
            dists, nbrs = knns2ordered_nbrs(knns)
            self.nbrs.append(nbrs)
            self.dists.append(dists)
            density = density_estimation(dists, nbrs, labels)

            g = self._build_graph(mfcc_xvec_features, cluster_features, labels, density, knns)
            if g.number_of_edges() > 0 or must_run:
                self.gs.append(g)

            if lvl >= self.levels - 1:
                break
            if "train" in mode:
                if mfcc_xvec_features.shape[0] <= self.numspk:
                    break

            print('spks: ',self.numspk,'features shape: ',features.shape[0])
            # Decode peak nodes
            new_pred_labels, peaks,\
                global_edges, global_pred_labels, global_peaks = decode(g, 0, 'sim', True, mode,
                                                                        ids, global_edges, global_num_nodes,
                                                                        global_peaks,device='cuda')
            
            if prev_num_nodes == len(peaks):
                print("Exiting due to failure in final cluster formation")
                break
            prev_num_nodes = len(peaks)
            ids = ids[peaks]
            
            mfcc_xvec_features, features, labels, cluster_features = build_next_level_e2e(mfcc_xvec_features, features,labels, peaks,
                                                                  global_features, global_pred_labels, global_peaks)
            if "rec_aff" not in mode:
                if "PLDA" in mode or "cosine" in mode:
                    affinity_mat = affinity_mat[peaks][:,peaks]
            else:
                if "PLDA" in mode or "cosine" in mode:
                    copy_feats = copy_feats[peaks]
                    if "PLDA" in mode:
                        if "proc_feats" in mode:
                            affinity_mat, features = get_PLDA_mat(copy_feats,self.pldamodel)
                            #features = l2norm(features.astype('float32'))
                        else:
                            affinity_mat, _ = get_PLDA_mat(copy_feats,self.pldamodel)
                    elif "cosine" in mode:
                        if "proc_feats" in mode:
                            affinity_mat, features = get_cosine_mat(copy_feats)
                            #features = l2norm(features.astype('float32'))
                        else:
                            affinity_mat, _ = get_cosine_mat(copy_feats)


    def _build_graph(self, features, cluster_features, labels, density, knns):
        
        adj = fast_knns2spmat(knns, self.k)
        adj, adj_row_sum = row_normalize(adj)
        indices, values, shape = sparse_mx_to_indices_values(adj)
        
        #adj_new = fast_knns2spmat(knns_new, self.k)
        #adj_new, adj_row_sum_new = row_normalize(adj_new)
        #indices_new, values_new, shape_new = sparse_mx_to_indices_values(adj_new)
        g = dgl.graph((indices[1], indices[0]),num_nodes=features.shape[0])
        g.ndata['labels'] = torch.LongTensor(labels)
        g.ndata['density'] = torch.FloatTensor(density)
        g.edata['affine'] = torch.FloatTensor(values)
        #g.edata['affine_new'] = torch.FloatTensor(values_new)
        # A Bipartite from DGL sampler will not store global eid, so we explicitly save it here
        g.edata['global_eid'] = g.edges(form='eid')
        g.ndata['norm'] = torch.FloatTensor(adj_row_sum)
        #g.ndata['norm_new'] = torch.FloatTensor(adj_row_sum_new)
        g.apply_edges(lambda edges: {'raw_affine': edges.data['affine'] / edges.dst['norm']})
        #g.apply_edges(lambda edges: {'raw_affine_new': edges.data['affine_new'] / edges.dst['norm_new']})
        g.apply_edges(lambda edges: {'labels_conn': (edges.src['labels'] == edges.dst['labels']).long()})
        g.apply_edges(lambda edges: {'mask_conn': (edges.src['density'] > edges.dst['density']).bool()})
        
        g = g.to(self.device)
        g.ndata['features'] = features
        g.ndata['cluster_features'] = cluster_features
        return g

    def __getitem__(self, index):
        assert index < len(self.gs)
        return self.gs[index]

    def __len__(self):
        return len(self.gs)
