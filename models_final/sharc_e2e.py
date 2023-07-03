#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn

from .graphconv import GraphConv
from .focal_loss import FocalLoss
from .model_tdnn import *


class SHARC(nn.Module):
    def __init__(self, feature_dim, nhid, num_conv=4, dropout=0,
                 use_GAT=True, K=1, balance=False,
                 use_cluster_feat = True, use_focal_loss = True,beta=1.0, **kwargs):
        super(SHARC, self).__init__()
        nhid_half = int(nhid / 2)
        self.use_cluster_feat = use_cluster_feat
        self.use_focal_loss = use_focal_loss
        self.beta = beta
        #out_feat_dim = 256
        
        if self.use_cluster_feat:
            self.feature_dim = feature_dim * 2
        else:
            self.feature_dim = feature_dim

        input_dim = (feature_dim, nhid, nhid, nhid_half)
        output_dim = (nhid, nhid, nhid_half, nhid_half)
        self.conv = nn.ModuleList()
        self.conv.append(GraphConv(self.feature_dim, nhid, dropout, use_GAT, K))
        for i in range(1, num_conv):
            self.conv.append(GraphConv(input_dim[i], output_dim[i], dropout, use_GAT, K))

        self.src_mlp = nn.Linear(output_dim[num_conv - 1], nhid_half)
        self.dst_mlp = nn.Linear(output_dim[num_conv - 1], nhid_half)

        self.classifier_conn = nn.Sequential(nn.PReLU(nhid_half),
                                          nn.Linear(nhid_half, nhid_half),
                                          nn.PReLU(nhid_half),
                                          nn.Linear(nhid_half, 2))

        if self.use_focal_loss:
            self.loss_conn = FocalLoss(2)
        else:
            self.loss_conn = nn.CrossEntropyLoss()
        self.loss_den = nn.MSELoss()

        self.balance = balance

    def pred_conn(self, edges):
        src_feat = self.src_mlp(edges.src['conv_features'])
        dst_feat = self.dst_mlp(edges.dst['conv_features'])
        pred_conn = self.classifier_conn(src_feat + dst_feat)
        return {'pred_conn': pred_conn}

    def pred_den_msg(self, edges):
        prob = edges.data['prob_conn']
        res = edges.data['raw_affine'] * (prob[:, 1] - prob[:, 0])
        return {'pred_den_msg': res}

    def forward(self, bipartites):
       
        if isinstance(bipartites, dgl.DGLGraph):
            bipartites = [bipartites] * len(self.conv)
            if self.use_cluster_feat:
                neighbor_x = torch.cat([bipartites[0].ndata['features'], bipartites[0].ndata['cluster_features']], axis=1)
            else:
                neighbor_x = bipartites[0].ndata['features']

            for i in range(len(self.conv)):
                neighbor_x = self.conv[i](bipartites[i], neighbor_x)

            output_bipartite = bipartites[-1]
            output_bipartite.ndata['conv_features'] = neighbor_x
        else:
            if self.use_cluster_feat:
                neighbor_x_src = torch.cat([bipartites[0].srcdata['features'], bipartites[0].srcdata['cluster_features']], axis=1)
                center_x_src = torch.cat([bipartites[1].srcdata['features'], bipartites[1].srcdata['cluster_features']], axis=1)
            else:
                neighbor_x_src = bipartites[0].srcdata['features']
                center_x_src = bipartites[1].srcdata['features']

            for i in range(len(self.conv)):
                neighbor_x_dst = neighbor_x_src[:bipartites[i].num_dst_nodes()]
                neighbor_x_src = self.conv[i](bipartites[i], (neighbor_x_src, neighbor_x_dst))
                center_x_dst = center_x_src[:bipartites[i+1].num_dst_nodes()]
                center_x_src = self.conv[i](bipartites[i+1], (center_x_src, center_x_dst))

            output_bipartite = bipartites[-1]
            output_bipartite.srcdata['conv_features'] = neighbor_x_src
            output_bipartite.dstdata['conv_features'] = center_x_src

        output_bipartite.apply_edges(self.pred_conn)
        torch.cuda.empty_cache()
        output_bipartite.edata['prob_conn'] = F.softmax(output_bipartite.edata['pred_conn'], dim=1)
        output_bipartite.update_all(self.pred_den_msg, fn.mean('pred_den_msg', 'pred_den'))
        #print("Memory Reserved= %f"%(torch.cuda.memory_reserved()))
        #print("Memory Allocated= %f"%(torch.cuda.memory_allocated()))
        return output_bipartite

    def compute_loss(self, bipartite):
        pred_den = bipartite.dstdata['pred_den']
        loss_den = self.loss_den(pred_den, bipartite.dstdata['density'])


        labels_conn = bipartite.edata['labels_conn']
        mask_conn = bipartite.edata['mask_conn']

        if self.balance:
            labels_conn = bipartite.edata['labels_conn']
            neg_check = torch.logical_and(bipartite.edata['labels_conn'] == 0, mask_conn)
            num_neg = torch.sum(neg_check).item()
            neg_indices = torch.where(neg_check)[0]
            pos_check = torch.logical_and(bipartite.edata['labels_conn'] == 1, mask_conn)
            num_pos = torch.sum(pos_check).item()
            pos_indices = torch.where(pos_check)[0]
            if num_pos > num_neg:
                mask_conn[pos_indices[np.random.choice(num_pos, num_pos - num_neg, replace = False)]] = 0
            elif num_pos < num_neg:
                mask_conn[neg_indices[np.random.choice(num_neg, num_neg - num_pos, replace = False)]] = 0

        # In subgraph training, it may happen that all edges are masked in a batch
        if mask_conn.sum() > 0:
            loss_conn = self.loss_conn(bipartite.edata['pred_conn'][mask_conn], labels_conn[mask_conn])
            if self.beta < 1:
                loss = (1-self.beta)*loss_den + self.beta*loss_conn
            else:
                loss = loss_den + self.beta*loss_conn
            loss_den_val = loss_den.item()
            loss_conn_val = loss_conn.item()
        else:
            loss = loss_den
            loss_den_val = loss_den.item()
            loss_conn_val = 0
        
        return loss, loss_den_val, loss_conn_val

    def get_clustergnnfeats(self,bipartites):
        # bp()
        if isinstance(bipartites, dgl.DGLGraph):
            bipartites = [bipartites] * len(self.conv)
            if self.use_cluster_feat:
                neighbor_x = torch.cat([bipartites[0].ndata['features'], bipartites[0].ndata['cluster_features']], axis=1)
            else:
                neighbor_x = bipartites[0].ndata['features']

            for i in range(len(self.conv)):
                neighbor_x = self.conv[i](bipartites[i], neighbor_x)

            output_bipartite = bipartites[-1]
            output_bipartite.ndata['conv_features'] = neighbor_x
        return output_bipartite.ndata['conv_features']

class e2e_diarization_nosilence(nn.Module): # remove silence from features before feeding 
    def __init__(self,xvec_dim=512,dropout=0,xvecmodelpath=None,device='cpu',fulltrain=0):
        super(e2e_diarization_nosilence, self).__init__()
        self.xvec_dim =xvec_dim
        self.pooling_function = torch.std
        self.xvector_extractor = XVectorNet_ETDNN_12Layer(pooling_function = self.pooling_function)
       
        if xvecmodelpath is not None:
            self.xvector_extractor.LoadFromKaldi(xvecmodelpath)
        if not fulltrain:
            self.xvector_extractor.tdnn1.requires_grad = False
            self.xvector_extractor.tdnn1a.requires_grad = False
            self.xvector_extractor.tdnn2.requires_grad = False
            self.xvector_extractor.tdnn2a.requires_grad = False
            self.xvector_extractor.tdnn3.requires_grad = False
            self.xvector_extractor.tdnn3a.requires_grad = False
            self.xvector_extractor.tdnn4.requires_grad = False
            
        # self.xvector_extractor.eval()
        # self.xvector_extractor_org = XVectorNet_ETDNN_12Layer(pooling_function = self.pooling_function)
        # self.xvector_extractor_org.LoadFromKaldi(xvecmodelpath)
        
        self.device = device
        
    def train1(self):
        self.train()
        self.xvector_extractor.tdnn1.bn.training = False
        self.xvector_extractor.tdnn1a.bn.training = False
        self.xvector_extractor.tdnn2.bn.trainsing = False
        self.xvector_extractor.tdnn2a.bn.training = False
        self.xvector_extractor.tdnn3.bn.training = False
        self.xvector_extractor.tdnn3a.bn.training = False
        self.xvector_extractor.tdnn4.bn.training = False
        self.xvector_extractor.tdnn4a.bn.training = False
        self.xvector_extractor.tdnn5.bn.training = False
       
    def mfcc_full_xvec_segmented(self,inp,adj,segmentsfile,org_xvec,filenames):
        # use full MFCCs without silence and create segments using segments file
        subsample = 1 # subsampling factor for x-vector 
        n_frames = inp.shape[1]
        x_4,x = self.xvector_extractor.extract_modified(inp.unsqueeze(0)) #batch,N,D

        win = 150 #frames
        shift = 25 #frames

        # read x-vector segments file here
        frame_step=0.010
        
        #   recording_id, n_frames = seg.strip().split()
        #   n_frames = int(n_frames)
        # generate xvectors.
        count = 0
        for idx,recording_id in enumerate(filenames):
        #   if recording_id in speech_segments:
            segmentation = segmentsfile[idx]
            xvecs = []

            segend = 0
            for onset, offset in zip(
                    segmentation.onsets, segmentation.offsets):
                bp()
                segdiff = offset-onset
                start = segend
                end = start + win
                diff = offset-onset
                if diff < 150:
                    if diff > 50:
                        
                        end  = min(end,segend+segdiff)
                        if count % subsample ==0:
                            seg1 = x_4[:,start:end]
                            seg2 = x[:,start:end]
                            pooledout1 = self.xvector_extractor.statspooling(seg1)
                            pooledout2 = self.xvector_extractor.statspooling(seg2)
                            pooledout = torch.cat((pooledout1,pooledout2),-1)
                            xvecs.append(self.xvector_extractor.lin6.forward(pooledout))
                        count = count + 1
                else:
                    
                    while end < segend + segdiff:
                        # if  segend + segdiff - end <= offset: 
                        #     end = segend + segdiff
                        if count % subsample ==0:
                            seg1 = x_4[:,start:end]
                            seg2 = x[:,start:end]
                            pooledout1 = self.xvector_extractor.statspooling(seg1)
                            pooledout2 = self.xvector_extractor.statspooling(seg2)

                            pooledout = torch.cat((pooledout1,pooledout2),-1)
                            xvecs.append(self.xvector_extractor.lin6.forward(pooledout))
                            
                        start = start + shift
                        end = start + win
                        count = count + 1
                    if end >= segend + segdiff:
                        end =  segend + segdiff
                        start = end - win
                        if count % subsample ==0:
                            seg1 = x_4[:,start:end]
                            seg2 = x[:,start:end]
                            pooledout1 = self.xvector_extractor.statspooling(seg1)
                            pooledout2 = self.xvector_extractor.statspooling(seg2)

                            pooledout = torch.cat((pooledout1,pooledout2),-1)
                            xvecs.append(self.xvector_extractor.lin6.forward(pooledout))
                        count = count + 1
                
                segend = segend + segdiff
            
            xvecs = torch.cat(xvecs,dim=0)
            
            x_new = 0.5*(xvecs + org_xvec)
            z = self.GCN_layer(x_new,adj)
        
        return z
    

    def mfcc_full_moving_avg(self,inp,filenames):
        # use full MFCCs without silence and use moving average window function calculate stats and xvectors
        
        subsample = 1 # subsampling factor for x-vector 
        n_frames = inp.shape[1]
        x_4,x = self.xvector_extractor.extract_modified(inp.unsqueeze(0)) #batch,N,D
        win = 150 #frames
        shift = 75 #frames
        xvecs = self.xvector_extractor.segmented_extraction(x_4,x)
        return xvecs


    def mfcc_segmented(self,inp,segmentsfile=None,filenames=None):
        # use MFCCs without silence in batches already
        fullbatchsize,n_frames,D = inp.shape
        batchsize = min(1000,fullbatchsize) 
        batch_count = int(fullbatchsize/batchsize)
        cur_batch = batchsize*batch_count
        remainder = fullbatchsize - cur_batch
        xvecs = []
       
        inp_new = inp[:batchsize*batch_count].reshape(batch_count,batchsize,n_frames,D)
        
        for i in range(batch_count):
            xvecs.append(self.xvector_extractor.extract(inp_new[i])) #batch,N,D
            # print('count:',i)
            
        if remainder > 0:
            xvecs.append(self.xvector_extractor.extract(inp[cur_batch:]))
        
        xvecs = torch.cat(xvecs,dim=0)
        
        return xvecs


class main_model(nn.Module):
    def __init__(self,feature_dim,nhid,num_conv=4, dropout=0,use_GAT=True, K=1, 
                 balance=False, use_cluster_feat = True, use_focal_loss = True,
                 xvecmodelpath=None, model_filename=None, device='cpu',fulltrain=0,xvecbeta=0.5,withcheckpoint=0):
         super(main_model, self).__init__()
         self.xvecbeta = xvecbeta
         self.e2e_xvecs = e2e_diarization_nosilence(xvec_dim=feature_dim,xvecmodelpath=xvecmodelpath,device=device,fulltrain=fulltrain)
         self.mylander = SHARC(feature_dim=feature_dim, nhid=nhid,
                        num_conv=num_conv, dropout=dropout,
                        use_GAT=use_GAT, K=K,
                        balance=balance,
                        use_cluster_feat=use_cluster_feat,
                        use_focal_loss=use_focal_loss)
         if model_filename is not None:
            if not withcheckpoint:
                self.mylander.load_state_dict(torch.load(model_filename, map_location=device))
            else:
                checkpoints = torch.load(model_filename, map_location=device)
                self.mylander.load_state_dict(checkpoints['model'])
                

    def train1(self):
        self.train()
        self.e2e_xvecs.train1()

    def compute_loss(self,bipartite):
        return self.mylander.compute_loss(bipartite)

    def extract_xvecs(self,inp,org_xvec):
        xvecs = self.e2e_xvecs.mfcc_segmented(inp)
        w = self.xvecbeta
        new_xvecs = w*xvecs+(1-w)*org_xvec
        return new_xvecs
        
    def forward(self,bipartites):
        return self.mylander(bipartites)
    
    def get_clustergnnfeats(self,bipartites):
        return self.mylander.get_clustergnnfeats(bipartites)
        
