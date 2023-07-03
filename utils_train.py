import pickle as pkl
import itertools
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import sys
import errno, os
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score
from models_train_ssc_plda import weight_initialization
from scipy.special import expit
sys.path.insert(0,'services/')
import pic_dihard as pic
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy import sparse
# from run_spectralclustering import do_spectral_clustering
import matplotlib as mat
from services.kaldi_io import *
mat.use('Agg')
from pdb import set_trace as bp
import random

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise
def DER(ref,system):
    spks = ref.shape[-1]
    ref_time = 0.0
    sys_error = 0.0
    for spk in range(spks):
        idx = np.where(ref[:,spk]==1)[0]
        ref_time = ref_time + len(idx)
        sys_error = sys_error + len(idx) - np.sum(system[idx,spk]) # sum where not 1 
    der = sys_error/len(ref)
    return der

def load_data_dihard(dataset,filename,groundpath,featsdict,device='cpu',useoverlap=1):
    # load the data: x, tx, allx, graph
    # bp()
    filepath = f'exp/{dataset}/ground_adj_cent_e2e/{filename}.pkl'
    cmd = f'mkdir -p exp/{dataset}/ground_adj_cent/'
    os.system(cmd)
    
    kaldi_recipe_path='/data1/prachis/Dihard_2020/Dihard_2020_track1'
    # dataset = 'dihard_dev_2020_track1_fbank_jhu'
    xvecpath = '{}/xvectors_npy/{}/'.format(kaldi_recipe_path,dataset)

    features = read_mat(featsdict[filename])
    features = torch.FloatTensor(features).to(device)
    # print('filename:',filename)
    X = np.load('{}/{}.npy'.format(xvecpath,filename))
    xvecs = torch.FloatTensor(X).to(device)
    ground_labels=open('{}/{}/threshold_0.5_avg/labels_{}'.format(groundpath,dataset.split('_fbank')[0],filename)).readlines()
    full_gndlist=[g.split()[1:] for g in ground_labels]
    gnd_list = np.array([g[0] for g in full_gndlist])
    uni_gnd_letter = np.unique(gnd_list)

    clean_list = np.array([len(f) for f in full_gndlist])
    clean_ind =np.where(clean_list ==1)[0]

    if os.path.isfile(filepath):
        with open(filepath,'rb') as fb:
            mydict = pkl.load(fb)
        adj = mydict['adj_overlap']
        # final_cent = mydict['cent']
        return adj, xvecs, clean_ind, features

    
    overlap_ind = np.where(clean_list > 1.0)[0]
    
    overlap_gnd_list = np.array([full_gndlist[oi][1] for oi in overlap_ind])
    uni_overlap_letter = np.unique(overlap_gnd_list)
    uni_gnd_letter = np.unique(np.hstack((uni_gnd_letter,uni_overlap_letter)))
  
    # plt.figure()
    # G = nx.Graph()
    G = nx.OrderedGraph()
    pos = nx.spring_layout(G)
    # bp()
    # plt.subplot(311)
    # print("The original Graph:")
    # nx.draw_networkx(G)
    N = len(gnd_list)
    L = np.arange(N)
    G.add_nodes_from(L)
    # color_list = ['blue','green','black','yellow','magenta']
    # color_map = L.astype(str)
    
    if useoverlap:
        for ind in uni_gnd_letter:
            L = np.where(gnd_list==ind)[0] 
            # edges = itertools.combinations(L,2)
            # G.add_edges_from(list(edges))
            # color_map[L] = color_list[ind]
            L2 = np.where(overlap_gnd_list==ind)[0]
            overlap_L =  overlap_ind[L2]
            
            # color_map[overlap_L] = 'red'
            
            full_L = np.unique(np.hstack((L,overlap_L)))
            edges = itertools.combinations(full_L,2)
            G.add_edges_from(edges)
    else:
        gnd_clean_ind = gnd_list[clean_ind]
        for ind in uni_gnd_letter:
            L = np.where(gnd_clean_ind==ind)[0] 
            clean_L = clean_ind[L]
            edges = itertools.combinations(clean_L,2)
            G.add_edges_from(list(edges))
   
    
    
    adj = nx.adjacency_matrix(G)
    # mydict = {}
    # mydict['adj_overlap']= adj
    # with open(filepath,'wb') as fb:
    #     pkl.dump(mydict,fb)
    # bp()
    return adj, xvecs, clean_ind,features
    # return adj, features, overlap_ind
    
def load_mfcc_feats_nosilence(args,filename,featsdict,reco2utt,device='cpu',useoverlap=1,featsbatch=0):
    # load the data: x, tx, allx, graph
    
    dataset = args.dataset_str
    
    segments_xvec = f'{args.segments_list}/{filename}.segments' 
    
    utts = reco2utt.split() # all utterances in a recording stored as dict
    # bp()
    # diff = segments.offsets - segments.onsets 
    # idx = np.where(diff>50)[0]  # discard segments <=0.5sec for training
    seg_xvec = np.genfromtxt(segments_xvec,dtype=float)[:,2:]
    diff_xvec = seg_xvec[:,1]-seg_xvec[:,0]
    diff_xvec = np.round(diff_xvec,2)
    # idx_xvec = np.where(diff_xvec>=1.5)[0]  # discard segments <=1.5sec for training
    idx_xvec = np.where(diff_xvec>=0.0)[0] 
    subsample = 1
    idx_xvec = idx_xvec[::subsample]
    
    if not featsbatch:
        features = []
        for utt in utts:
            val = featsdict[utt]
            features.append(read_mat(val))
        features = np.concatenate(features,axis=0)
        features = torch.FloatTensor(features).to(device)
    else:
        win = 150
        features = []
        utts = np.asarray(utts)
        for uid,utt in enumerate(utts[idx_xvec]):
            val = featsdict[utt]
            valsplit = val.split('[')[0]
            start = int(val.split('[')[1].split(':')[0])
            end =  int(val.split('[')[1].split(':')[1].split(']')[0])
            if end+1-start >= win:
                feats = read_mat(valsplit)[start:start+win]
                features.append(feats)
            else:
                diff = end+1-start
                # print('diff short:',diff)
                low = int(np.floor((win-diff)/2))
                high = int(np.ceil((win-diff)/2))
                
                feats = read_mat(valsplit)[start:start+win]
                # featspad = np.pad(feats,((low,high),(0,0)),'symmetric')
                featspad = np.pad(feats,((0,win-diff),(0,0)),'wrap')
                features.append(featspad)
            #     print(start,end,end+1-start,utt,round(seg_xvec[uid,1]-seg_xvec[uid,0],2),uid)
    
    features = np.array(features)
    # features = features[::subsample]
    # features = torch.FloatTensor(features).to(device)
    
    return features,idx_xvec
    

def load_mfcc_feats_nosilence_sub(args,filename,featsdict,reco2utt,start,end,device='cpu',featsbatch=0,clean_ind=None):
    # load the data: x, tx, allx, graph
    
    utts = reco2utt.split() # all utterances in a recording stored as dict
    utts = np.array(utts)
    if clean_ind is not None:
        utts = utts[clean_ind]
    utts = utts[start:end]
    if not featsbatch:
        features = []
        for utt in utts:
            val = featsdict[utt]
            features.append(read_mat(val))
        features = np.concatenate(features,axis=0)
        features = torch.FloatTensor(features).to(device)
    else:
        win = 150
        features = []
        utts = np.asarray(utts)
        for uid,utt in enumerate(utts):
            val = featsdict[utt]
            valsplit = val.split('[')[0]
            start = int(val.split('[')[1].split(':')[0])
            end =  int(val.split('[')[1].split(':')[1].split(']')[0])
            if end+1-start >= win:
                feats = read_mat(valsplit)[start:start+win]
                features.append(feats)
            else:
                diff = end+1-start
                # print('diff short:',diff)
                low = int(np.floor((win-diff)/2))
                high = int(np.ceil((win-diff)/2))
                
                feats = read_mat(valsplit)[start:start+win]
                # featspad = np.pad(feats,((low,high),(0,0)),'symmetric')
                featspad = np.pad(feats,((0,win-diff),(0,0)),'wrap')
                features.append(featspad)
            #     print(start,end,end+1-start,utt,round(seg_xvec[uid,1]-seg_xvec[uid,0],2),uid)
    
    features = np.array(features)
    # features = features[::subsample]
    # features = torch.FloatTensor(features).to(device)
    
    return features
   
def load_data_dihard_nosilence(args,filename,featsdict,reco2utt,device='cpu',useoverlap=1,n_clusters=None):
    # load the data: x, tx, allx, graph
    
    dataset = args.dataset_str
    groundpath = args.groundlabels
    segments_xvec = f'{args.segments}/{filename}.segments' 
    filepath = f'exp/{dataset}/ground_adj_e2e/{filename}.pkl'
    cmd = f'mkdir -p exp/{dataset}/ground_adj_e2e/'
    os.system(cmd)
    
    kaldi_recipe_path='/data1/prachis/Dihard_2020/Dihard_2020_track1'
    # dataset = 'dihard_dev_2020_track1_fbank_jhu'
    xvecpath = '{}/xvectors_npy/{}/'.format(kaldi_recipe_path,dataset)
    utts = reco2utt.split()[1:]
    idx_xvec = None
    # diff = segments.offsets - segments.onsets 
    # idx = np.where(diff>50)[0]  # discard segments <=0.5sec for training
    # seg_xvec = np.genfromtxt(segments_xvec,dtype=float)[:,2:]
    # diff_xvec = seg_xvec[:,1]-seg_xvec[:,0]
    # diff_xvec = np.round(diff_xvec,2)
    # idx_xvec = np.where(diff_xvec>=0.0)[0]  # discard segments <0sec for training
    # subsample = 1
    # idx_xvec = idx_xvec[::subsample]
    
    # features = []
    # for utt in utts:
    #     val = featsdict[utt]
    #     features.append(read_mat(val))
    # features = np.concatenate(features,axis=0)
    # features = torch.FloatTensor(features).to(device)
    win = 150
    features = []
    utts = np.asarray(utts)
    for uid,utt in enumerate(utts):
        val = featsdict[utt]
        valsplit = val.split('[')[0]
        start = int(val.split('[')[1].split(':')[0])
        end =  int(val.split('[')[1].split(':')[1].split(']')[0])
        if end+1-start >= win:
            feats = read_mat(valsplit)[start:start+win]
            features.append(feats)
        else:
            diff = end+1-start
            # print('diff short:',diff)
            low = int(np.floor((win-diff)/2))
            high = int(np.ceil((win-diff)/2))
            
            feats = read_mat(valsplit)[start:start+win]
            # featspad = np.pad(feats,((low,high),(0,0)),'symmetric')
            featspad = np.pad(feats,((0,win-diff),(0,0)),'wrap')
            features.append(featspad)
            
        #     print(start,end,end+1-start,utt,round(seg_xvec[uid,1]-seg_xvec[uid,0],2),uid)
   
    features = np.array(features)
    # features = features[::subsample]
    features = torch.FloatTensor(features).to(device)
    # print('filename:',filename)
    X = np.load('{}/{}.npy'.format(xvecpath,filename))
    # X = X[idx_xvec]
    xvecs = torch.FloatTensor(X).to(device)
    ground_labels=open('{}/{}/threshold_0.5_avg/labels_{}'.format(groundpath,dataset.split('_fbank')[0],filename)).readlines()
    
    full_gndlist=[g.split()[1:] for g in ground_labels]
    
    # full_gndlist=[ground_labels[g].split()[1:] for g in idx_xvec]
    
    gnd_list = np.array([g[0] for g in full_gndlist])
    
    
    uni_gnd_letter = np.unique(gnd_list)

    clean_list = np.array([len(f) for f in full_gndlist])
    clean_ind =np.where(clean_list ==1)[0]

    if os.path.isfile(filepath):
        with open(filepath,'rb') as fb:
            mydict = pkl.load(fb)
        adj = mydict['adj_overlap']
        # final_cent = mydict['cent']
        return adj, xvecs, clean_ind, features,idx_xvec
    if n_clusters == 1:
        n_nodes  = X.shape[0]
        
        adj = sparse.csr_matrix(np.ones((n_nodes,n_nodes)))
        return adj, xvecs, clean_ind, features,idx_xvec
    
    overlap_ind = np.where(clean_list > 1.0)[0]
    
    overlap_gnd_list = np.array([full_gndlist[oi][1] for oi in overlap_ind])
    uni_overlap_letter = np.unique(overlap_gnd_list)
    uni_gnd_letter = np.unique(np.hstack((uni_gnd_letter,uni_overlap_letter)))
  
    # plt.figure()
    # G = nx.Graph()
    G = nx.OrderedGraph()
    pos = nx.spring_layout(G)
    # bp()
    # plt.subplot(311)
    # print("The original Graph:")
    # nx.draw_networkx(G)
    N = len(gnd_list)
    L = np.arange(N)
    G.add_nodes_from(L)
    # color_list = ['blue','green','black','yellow','magenta']
    # color_map = L.astype(str)
    
    if useoverlap and overlap_gnd_list.shape[0]>0:
        for ind in uni_gnd_letter:
            L = np.where(gnd_list==ind)[0] 
            # edges = itertools.combinations(L,2)
            # G.add_edges_from(list(edges))
            # color_map[L] = color_list[ind]
            L2 = np.where(overlap_gnd_list==ind)[0]
            overlap_L =  overlap_ind[L2]
            
            # color_map[overlap_L] = 'red'
            
            full_L = np.unique(np.hstack((L,overlap_L)))
            edges = itertools.combinations(full_L,2)
            G.add_edges_from(edges)
    else:
        gnd_clean_ind = gnd_list[clean_ind]
        for ind in uni_gnd_letter:
            L = np.where(gnd_clean_ind==ind)[0] 
            clean_L = clean_ind[L]
            edges = itertools.combinations(clean_L,2)
            G.add_edges_from(list(edges))
   
    
    
    adj = nx.adjacency_matrix(G)
    mydict = {}
    mydict['adj_overlap']= adj
    with open(filepath,'wb') as fb:
        pkl.dump(mydict,fb)
    # bp()
    return adj, xvecs, clean_ind,features,idx_xvec
    # return adj, features, overlap_ind



def load_data_dihard_val(dataset,filename,device='cpu',useoverlap=1,outf=None):
    # load the data: x, tx, allx, graph

    filepath = f'exp/{dataset}/ground_adj_cent/{filename}.pkl'
    cmd = f'mkdir -p exp/{dataset}/ground_adj_cent/'
    os.system(cmd)
    
    kaldi_recipe_path='/data1/prachis/Dihard_2020/Dihard_2020_track1'
    # dataset = 'dihard_dev_2020_track1_fbank_jhu'
    xvecpath = '{}/xvectors_npy/{}/'.format(kaldi_recipe_path,dataset)

    # print('filename:',filename)
    X = np.load('{}/{}.npy'.format(xvecpath,filename))
    features = torch.FloatTensor(X).to(device)
    ground_labels=open('/data1/prachis/Dihard_2020/SSC/ALL_GROUND_LABELS/'+dataset.split('_fbank')[0]+'/threshold_0.5_avg/labels_'+filename).readlines()
    full_gndlist=[g.split()[1:] for g in ground_labels]
    gnd_list = np.array([g[0] for g in full_gndlist])
    uni_gnd_letter = np.unique(gnd_list)

    clean_list = np.array([len(f) for f in full_gndlist])
    clean_ind =np.where(clean_list ==1)[0]

    if os.path.isfile(filepath):
        with open(filepath,'rb') as fb:
            mydict = pkl.load(fb)
        adj = mydict['adj_overlap']
        try:
            final_cent = mydict['cent']
        except:
            final_cent = 0 
        return adj, features, clean_ind, final_cent

    
    overlap_ind = np.where(clean_list > 1.0)[0]
   
    

    overlap_gnd_list = np.array([full_gndlist[oi][1] for oi in overlap_ind])
    uni_overlap_letter = np.unique(overlap_gnd_list)
        
    clusterlen_gnd = []
    for ind,uni in enumerate(uni_gnd_letter):
        gnd_list[gnd_list==uni]=ind
        
    gnd_list = gnd_list.astype(int)
    
    for ind,uni in enumerate(uni_overlap_letter):
        overlap_gnd_list[overlap_gnd_list==uni]=ind
        
    uni_gnd_letter = np.unique(np.hstack((uni_gnd_letter,uni_overlap_letter)))

    uni_gnd = np.arange(len(uni_gnd_letter))

    overlap_gnd_list = overlap_gnd_list.astype(int)
    
    # plt.figure()
    # G = nx.Graph()
    G = nx.OrderedGraph()
    pos = nx.spring_layout(G)
    # bp()
    # plt.subplot(311)
    # print("The original Graph:")
    # nx.draw_networkx(G)
    N = len(gnd_list)
    L = np.arange(N)
    G.add_nodes_from(L)
    # color_list = ['blue','green','black','yellow','magenta']
    # color_map = L.astype(str)
    
    if useoverlap:
        for ind in uni_gnd:
            L = np.where(gnd_list==ind)[0] 
            # edges = itertools.combinations(L,2)
            # G.add_edges_from(list(edges))
            # color_map[L] = color_list[ind]
            L2 = np.where(overlap_gnd_list==ind)[0]
            overlap_L =  overlap_ind[L2]
            
            # color_map[overlap_L] = 'red'
            
            full_L = np.unique(np.hstack((L,overlap_L)))
            edges = itertools.combinations(full_L,2)
            G.add_edges_from(edges)
    else:
        gnd_clean_ind = gnd_list[clean_ind]
        for ind in uni_gnd:
            L = np.where(gnd_clean_ind==ind)[0] 
            clean_L = clean_ind[L]
            edges = itertools.combinations(clean_L,2)
            G.add_edges_from(list(edges))
    # bp()
    if X.shape[0]< 1500:    
        cc=nx.average_clustering(G)
        cent=nx.clustering(G)
        bet=nx.betweenness_centrality(G)
        final_cent = np.array(list(cent.values()))*(1-np.array(list(bet.values())))
        bp()
   
    
    adj = nx.adjacency_matrix(G)
    mydict = {}
    mydict['adj_overlap']= adj
    # mydict['cent'] = final_cent
    with open(filepath,'wb') as fb:
        pkl.dump(mydict,fb)
    # bp()
    return adj, features, clean_ind, 0
    # return adj, features, overlap_ind

def load_data_dihard_val_weightadj(dataset,filename,device='cpu',useoverlap=1,outf=None,set='val'):
    # load the data: x, tx, allx, graph

    filepath = f'exp/{dataset}/ground_adj_cent_weighted/{filename}.pkl'
    cmd = f'mkdir -p exp/{dataset}/ground_adj_cent_weighted/'
    os.system(cmd)
    
    kaldi_recipe_path='/data1/prachis/Dihard_2020/Dihard_2020_track1'
    # dataset = 'dihard_dev_2020_track1_fbank_jhu'
    xvecpath = '{}/xvectors_npy/{}/'.format(kaldi_recipe_path,dataset)

    # print('filename:',filename)
    X = np.load('{}/{}.npy'.format(xvecpath,filename))
    features = torch.FloatTensor(X).to(device)
    ground_labels=open('/data1/prachis/Dihard_2020/SSC/ALL_GROUND_LABELS/'+dataset.split('_fbank')[0]+'/threshold_0.5_avg/labels_'+filename).readlines()
    full_gndlist=[g.split()[1:] for g in ground_labels]
    gnd_list = np.array([g[0] for g in full_gndlist])
    uni_gnd_letter = np.unique(gnd_list)

    clean_list = np.array([len(f) for f in full_gndlist])
    clean_ind =np.where(clean_list ==1)[0]
    # bp()
    if os.path.isfile(filepath):
        with open(filepath,'rb') as fb:
            mydict = pkl.load(fb)
        adj = mydict['adj_overlap']
        if set == 'val':
            try:
                final_cent = mydict['cent']
                return adj, features, clean_ind, final_cent
            except:
                return adj, features, clean_ind
        else:
            return adj, features, clean_ind

    overlap_ind = np.where(clean_list > 1.0)[0]
    # pred_overlap_ind = np.where(final_cent<1)[0]
    # intersection = list(set(overlap_ind).intersection(pred_overlap_ind))
    # print('intersection: ',len(intersection))
    # if len(intersection)!=0:
    #     print('overlap precision: ', len(intersection)/len(pred_overlap_ind),'overlap recall: ', len(intersection)/len(overlap_ind))
    # # bp()
    # if len(intersection)==0 or len(intersection)/len(overlap_ind) < 1:
    #     print(np.unique(final_cent))
        # bp()
    # return adj, features, clean_ind, 0
    

    overlap_gnd_list = np.array([full_gndlist[oi][1] for oi in overlap_ind])
    uni_overlap_letter = np.unique(overlap_gnd_list)
        
  
    uni_gnd_letter = np.unique(np.hstack((uni_gnd_letter,uni_overlap_letter)))

    
    G = nx.OrderedGraph()
    pos = nx.spring_layout(G)
   
    N = len(gnd_list)
    L = np.arange(N)
    G.add_nodes_from(L)
    
    if useoverlap:
        for ind in uni_gnd_letter:
            L = np.where(gnd_list[clean_ind]==ind)[0]
            clean_L = clean_ind[L]
            edges = itertools.combinations(clean_L,2)
            G.add_edges_from(list(edges))
            # color_map[L] = color_list[ind]
            
            L2 = np.where(overlap_gnd_list==ind)[0]
            overlap_L =  overlap_ind[L2]
            
            L3 = np.where(gnd_list[overlap_ind]==ind)[0]
            gn_L =  overlap_ind[L3]
            # color_map[overlap_L] = 'red'
            
            full_overlap_L = np.unique(np.hstack((overlap_L,gn_L)))
            
            edges = itertools.combinations(full_overlap_L,2)
            G.add_edges_from(list(edges),weight=0.5)
            
            for cl in clean_L:
                for ol in full_overlap_L:
                    G.add_edge(cl,ol,weight=0.5)


    else:
        gnd_clean_ind = gnd_list[clean_ind]
        for ind in uni_gnd_letter:
            L = np.where(gnd_clean_ind==ind)[0] 
            clean_L = clean_ind[L]
            edges = itertools.combinations(clean_L,2)
            G.add_edges_from(list(edges))
    # bp()
    # if X.shape[0]< 1500:    
    adj = nx.adjacency_matrix(G)
    mydict = {}
    mydict['adj_overlap']= adj
    if set=='val':
        cent=nx.clustering(G)
        bet=nx.betweenness_centrality(G)
        final_cent = np.array(list(cent.values()))*(1-np.array(list(bet.values())))
            
        pred_overlap_ind = np.where(final_cent<1)[0]
        intersection = list(set(overlap_ind).intersection(pred_overlap_ind))
        print('intersection: ',len(intersection))
        if len(intersection)!=0:
            print('overlap precision: ', len(intersection)/len(pred_overlap_ind),'overlap recall: ', len(intersection)/len(overlap_ind))
        mydict['cent'] = final_cent
        with open(filepath,'wb') as fb:
            pkl.dump(mydict,fb)
        return adj, features, clean_ind, final_cent
    else:
        return adj, features, clean_ind

    # return adj, features, overlap_ind

def load_data_dihard_pic(dataset,filename,device):
    # load the data: x, tx, allx, graph

    kaldi_recipe_path='/data1/prachis/Dihard_2020/Dihard_2020_track1'
    # dataset = 'dihard_dev_2020_track1_fbank_jhu'
    xvecpath = '{}/xvectors_npy/{}/'.format(kaldi_recipe_path,dataset)

    # print('filename:',filename)
    X = np.load('{}/{}.npy'.format(xvecpath,filename))
    ground_labels=open('/data1/prachis/Dihard_2020/SSC/ALL_GROUND_LABELS/'+dataset.split('_fbank')[0]+'/threshold_0.5_avg/labels_'+filename).readlines()
    # ground_labels=open('/data1/prachis/Dihard_2020/SSC/ALL_GROUND_LABELS/{}'.format(dataset)+'/threshold_0.25/labels_'+filename).readlines()

    full_gndlist=[g.split()[1:] for g in ground_labels]
    gnd_list = np.array([g[0] for g in full_gndlist])
    uni_gnd_letter = np.unique(gnd_list)

    clean_list = np.array([len(f) for f in full_gndlist])
    clean_ind =np.where(clean_list ==1)[0]
    overlap_ind = np.where(clean_list > 1)[0]

    overlap_gnd_list = np.array([full_gndlist[oi][1] for oi in overlap_ind])
    uni_overlap_letter = np.unique(overlap_gnd_list)
        
    clusterlen_gnd = []
    for ind,uni in enumerate(uni_gnd_letter):
        gnd_list[gnd_list==uni]=ind
        
    gnd_list = gnd_list.astype(int)
    
    
    for ind,uni in enumerate(uni_overlap_letter):
        overlap_gnd_list[overlap_gnd_list==uni]=ind
        
    uni_gnd_letter = np.unique(np.hstack((uni_gnd_letter,uni_overlap_letter)))

    uni_gnd = np.arange(len(uni_gnd_letter))

    overlap_gnd_list = overlap_gnd_list.astype(int)
    # bp()
    # plt.figure()
    # G = nx.Graph()
    G = nx.OrderedGraph()
    pos = nx.spring_layout(G)
    # bp()
    # plt.subplot(311)
    # print("The original Graph:")
    # nx.draw_networkx(G)
    N = len(gnd_list)
    L = np.arange(N)
    G.add_nodes_from(L)
    # color_list = ['blue','green','black','yellow','magenta','brown']
    # color_map = L.astype(str)
    for ind in uni_gnd:
        L = np.where(gnd_list[clean_ind]==ind)[0]
        clean_L = clean_ind[L]
        edges = itertools.combinations(clean_L,2)
        G.add_edges_from(list(edges))
        # color_map[L] = color_list[ind]
        
        L2 = np.where(overlap_gnd_list==ind)[0]
        overlap_L =  overlap_ind[L2]
        
        L3 = np.where(gnd_list[overlap_ind]==ind)[0]
        gn_L =  overlap_ind[L3]
        # color_map[overlap_L] = 'red'
        
        full_overlap_L = np.unique(np.hstack((overlap_L,gn_L)))
        
        edges = itertools.combinations(full_overlap_L,2)
        G.add_edges_from(list(edges),weight=0.5)
        
        for cl in clean_L:
            for ol in full_overlap_L:
                G.add_edge(cl,ol,weight=0.5)
        
        
        
    # plt.figure()
    # nx.draw_networkx(G,node_color=color_map,node_size=20, with_labels=False)

    # plt.savefig('exp/overlap_graphs/graph_{}.png'.format(filename))
    
    features = torch.FloatTensor(X).to(device)
    adj = nx.adjacency_matrix(G)
    # bp()
    return adj, features

def load_data_dihard_plda(dataset,filename,n_clusters,device='cpu',threshold=None,set='val',scale=1):
    # load the data: x, tx, allx, graph
# load the data: x, tx, allx, graph
    # scale = 1
    if scale> 1:
        scale_str =f'_scale{scale}'
    else:
        scale_str=''
    if set == 'train':
        filepath = f'exp/{dataset}/plda_adj_A_0.75s{scale_str}/{filename}.pkl'
        cmd = f'mkdir -p exp/{dataset}/plda_adj_A_0.75s{scale_str}/'
    else:
        filepath = f'exp/{dataset}/plda_adj_A{scale_str}/{filename}.pkl'
        cmd = f'mkdir -p exp/{dataset}/plda_adj_A{scale_str}/'
    os.system(cmd)
    if os.path.isfile(filepath):
        with open(filepath,'rb') as fb:
            A = pkl.load(fb)
    else:
        device = 'cpu'
        kaldi_recipe_path='/data1/prachis/Dihard_2020/Dihard_2020_track1'
        pldamodel= '/data1/prachis/Dihard_2020/SSC/lists/{0}/plda_{0}.pkl'.format(dataset)
        pldamodel = pkl.load(open(pldamodel,'rb'))
        # dataset = 'dihard_dev_2020_track1_fbank_jhu'
        if set=='train':
            xvecpath = 'tools_diar/xvectors_0.75s_npy/{}/'.format(dataset)
        else:
            xvecpath = '{}/xvectors_npy/{}/'.format(kaldi_recipe_path,dataset)
        # filename  = 'DH_DEV_0001'
        # print('filename:',filename)
        X = np.load('{}/{}.npy'.format(xvecpath,filename))
        features = torch.FloatTensor(X).to(device)

        xvecD = X.shape[1]
        pca_dim = 30
        inpdata = features[np.newaxis]
        net_init = weight_initialization(pldamodel,dimension=xvecD,pca_dimension=pca_dim,device=device)
        model_init = net_init.to(device)
        affinity_init,_,_ = model_init.compute_plda_affinity_matrix(pldamodel,inpdata) # original filewise PCA transform
        output_model = affinity_init.detach().cpu().numpy()[0]
        # bp()
        A = expit(output_model*scale)
        with open(filepath,'wb') as fb:
            pkl.dump(A,fb)

    # directly feeding adjacency matrix
    B = A.copy()
    B[B<=0.5] =0
    G = nx.from_numpy_matrix(B)
    
    adj = nx.adjacency_matrix(G)
    
    return adj, A

def load_data_dihard_plda_spectral(dataset,filename,n_clusters,device='cpu'):
    # load the data: x, tx, allx, graph
    kaldi_recipe_path='/data1/prachis/Dihard_2020/Dihard_2020_track1'
    xvecpath = '{}/xvectors_npy/{}/'.format(kaldi_recipe_path,dataset)
    X = np.load('{}/{}.npy'.format(xvecpath,filename))
    features = torch.FloatTensor(X).to(device)
    filepath = f'exp/{dataset}/plda_adj_A/{filename}.pkl'
    cmd = f'mkdir -p exp/{dataset}/plda_adj_A/'
    os.system(cmd)
    if os.path.isfile(filepath):
        with open(filepath,'rb') as fb:
            A = pkl.load(fb)
    else:
        plda_dataset = 'dihard_dev_2020_track1_fbank_jhu'
        pldamodel= '/data1/prachis/Dihard_2020/SSC/lists/{0}/plda_{0}.pkl'.format(plda_dataset)
        pldamodel = pkl.load(open(pldamodel,'rb'))
        # dataset = 'dihard_dev_2020_track1_fbank_jhu'
        xvecD = X.shape[1]
        pca_dim = 30
        inpdata = features[np.newaxis]
        net_init = weight_initialization(pldamodel,dimension=xvecD,pca_dimension=pca_dim,device=device)
        model_init = net_init.to(device)
        affinity_init,_,_ = model_init.compute_plda_affinity_matrix(pldamodel,inpdata) # original filewise PCA transform
        output_model = affinity_init.detach().cpu().numpy()[0]
        # bp()
        A = expit(output_model)
        with open(filepath,'wb') as fb:
            pkl.dump(A,fb)

    # directly feeding adjacency matrix
    B = A.copy()
    N = B.shape[0]
    B[B<=0.5] =0
    # G = nx.from_numpy_matrix(B)
    # adj = nx.adjacency_matrix(G)
    # KNN
    # B = -A.copy()
    # K = 30
    z = 0.8
    # K = min(K,N-1)
    # B[np.diag_indices(N)] = -np.inf
    # sortedDist = np.sort(B,axis=1)
    # NNIndex = np.argsort(B,axis=1)
    # NNIndex = NNIndex[:,:K+1]
    # ND = -sortedDist[:, 1:K+1].copy()
    # NI = NNIndex[:, 1:K+1].copy()
    # XI = np.dot(np.arange(N).reshape(-1,1),np.ones((1,K),dtype=int))
    # B = coo_matrix((ND.reshape(-1), (XI.reshape(-1),  NI.reshape(-1))), shape=(N, N)).toarray()
    # B[np.diag_indices(N)]=1
    # B = B/np.sum(B,axis=1).reshape(N,1) # Transition matrix
    # bp()
    # B = np.linalg.inv(np.eye(N)-z*B)
    # B[B<=1e-4] = 0
    adj_input = sp.csr_matrix(B)
    # labelfull=np.arange(A.shape[0])
   
    # clusterlen=[1]*len(labelfull)    
    # N = len(labelfull)
    # # plda scores
    # z=0.5
    # K = 30
    # final_k = min(K, N - 1) 
    # mypic =pic.PIC_dihard_threshold(n_clusters,clusterlen,labelfull,A.copy(),threshold,K=final_k,z=z) 
    
    # if threshold == None:
    #     labelfull,clusterlen = mypic.gacCluster_oracle_org()
    # else:
    #     labelfull,clusterlen = mypic.gacCluster_org()
    # # bp()
    # uni_label = np.unique(labelfull)
    # G = nx.OrderedGraph()

    # L = np.arange(N)
    # G.add_nodes_from(L)
    # for ind in uni_label:
    #     L = np.where(labelfull==ind)[0] 
    #     edges = itertools.combinations(L,2)
    #     G.add_edges_from(list(edges))
    # cent=nx.clustering(G)
    # bet=nx.betweenness_centrality(G)
    # bp()

    return adj_input, features, A

def load_data_dihard_pldaSpecc(dataset,filename,n_clusters,threshold=None):
    # load the data: x, tx, allx, graph
    device = 'cpu'
    kaldi_recipe_path='/data1/prachis/Dihard_2020/Dihard_2020_track1'
    pldamodel= '/data1/prachis/Dihard_2020/SSC/lists/{0}/plda_{0}.pkl'.format(dataset)
    pldamodel = pkl.load(open(pldamodel,'rb'))
    # dataset = 'dihard_dev_2020_track1_fbank_jhu'
    xvecpath = '{}/xvectors_npy/{}/'.format(kaldi_recipe_path,dataset)
    # filename  = 'DH_DEV_0001'
    print('filename:',filename)
    X = np.load('{}/{}.npy'.format(xvecpath,filename))
    features = torch.FloatTensor(X)
    xvecD = X.shape[1]
    pca_dim = 30
    inpdata = features[np.newaxis]
    net_init = weight_initialization(pldamodel,dimension=xvecD,pca_dimension=pca_dim,device=device)
    model_init = net_init.to(device)
    affinity_init,_,_ = model_init.compute_plda_affinity_matrix(pldamodel,inpdata) # original filewise PCA transform
    output_model = affinity_init.detach().cpu().numpy()[0]
    A = expit(output_model)
    return A
    
def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.
    
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        # print('checking shape...',np.round(a - b[:, None], tol).shape)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])
    # bp()
    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])
    # bp()
    # assert ~ismember(test_edges_false, edges_all)
    # assert ~ismember(val_edges_false, edges_all)
    # assert ~ismember(val_edges, train_edges)
    # assert ~ismember(test_edges, train_edges)
    # assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

def mask_val_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.
    
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
   
    num_val = int(np.floor(edges.shape[0] / 100.))
    # bp()
    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
   
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        # print('checking shape...',np.round(a - b[:, None], tol).shape)
        return np.any(rows_close)

   
    # bp()
    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])


    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false

def mask_val_edges_simplified(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.
    
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    # bp()
    edges_all = sparse_to_tuple(adj)[0]
   
    num_val = int(np.floor(edges.shape[0] / 100.))
    # bp()
    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
   
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([val_edge_idx]), axis=0)
    train_edge_weights = np.delete(adj_tuple[1], np.hstack([val_edge_idx]), axis=0)
    
    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        # print('checking shape...',np.round(a - b[:, None], tol).shape)
        return np.any(rows_close)

   
    # bp()
    val_edges_false = []
    adj_arr = adj.toarray()
    N = adj_arr.shape[0]
    tr_indices = np.triu_indices(N,k=1)
    tr_flattened = tr_indices[0]*N+tr_indices[1]
    # bp()
    indices = np.where(adj_arr[tr_indices]==0)[0]
    random.shuffle(indices)
    negatives = tr_flattened[indices]
    c = 0
    while len(val_edges_false) < len(val_edges):
        idx_i = int(negatives[c]/N)
        idx_j = int(negatives[c] % N)
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])
        c +=1


    # data = np.ones(train_edges.shape[0])
    data = train_edge_weights
    # bp()
    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])

    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(expit(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(expit(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def get_roc_score_modified(emb, adj_orig,baseline=None):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    if baseline is None:
        N= emb.shape[0]
        adj_rec =expit( np.dot(emb, emb.T))
    else:
        adj_rec = emb.toarray()
        N= adj_rec.shape[0]
        
    tr_indices = np.triu_indices(N,k=1)
    adj_triu = adj_orig.toarray()[tr_indices]
    # bp()
    # Predict on test set of edges
    
    adj_rec_triu = adj_rec[tr_indices]
    
    preds = adj_rec_triu[np.where(adj_triu==1)[0]]
    preds_neg = adj_rec_triu[np.where(adj_triu==0)[0]]
    # preds = []
    # pos = []
    # for e in edges_pos:
    #     preds.append(expit(adj_rec[e[0], e[1]]))
    #     pos.append(adj_orig[e[0], e[1]])

    # preds_neg = []
    # neg = []
    # for e in edges_neg:
    #     preds_neg.append(expit(adj_rec[e[0], e[1]]))
    #     neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    if len(np.unique(labels_all))>1:
        roc_score = roc_auc_score(labels_all, preds_all)
    else:
        roc_score=1
    ap_score = average_precision_score(labels_all, preds_all)
    preds_all[preds_all>0.5] = 1
    preds_all[preds_all<=0.5] = 0
    
    r_score = recall_score(labels_all,preds_all)
    return roc_score, ap_score,r_score

def get_roc_score_overlap(preds_all,labels_all):
    
    if len(np.unique(labels_all))>1:
        roc_score = roc_auc_score(labels_all, preds_all)
    else:
        roc_score=1
    ap_score = average_precision_score(labels_all, preds_all)
    preds_all[preds_all>0.5] = 1
    preds_all[preds_all<=0.5] = 0
    
    r_score = recall_score(labels_all,preds_all)
    return roc_score, ap_score,r_score


def get_roc_score_kfold(emb, adj_orig,baseline=None):
    if baseline is None:
        N= emb.shape[0]
        adj_rec =expit( np.dot(emb, emb.T))
    else:
        adj_rec = emb.toarray()
        N = adj_rec.shape[0]
        
    tr_indices = np.triu_indices(N,k=1)
    adj_triu = adj_orig[tr_indices]
    # bp()
    # Predict on test set of edges
    
    adj_rec_triu = adj_rec[tr_indices]
    
    preds = adj_rec_triu[np.where(adj_triu==1)[0]]
    preds_neg = adj_rec_triu[np.where(adj_triu==0)[0]]

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    
    return roc_score, ap_score

def plot_results(results, test_freq,filename, path='exp/results'):
    # Init
    plt.close('all')
    fig = plt.figure(figsize=(8, 8))

    x_axis_train = range(len(results['train_loss']))
    x_axis_test = range(0, len(x_axis_train), test_freq)
    # Elbo
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(x_axis_train, results['train_loss'])
    ax.set_ylabel('Loss ')
    ax.set_title('Loss ')
    ax.legend(['Train'], loc='upper right')

    # Accuracy
    # ax = fig.add_subplot(2, 2, 2)
    # ax.plot(x_axis_train, results['accuracy_train'])
    # ax.plot(x_axis_test, results['accuracy_test'])
    # ax.set_ylabel('Accuracy')
    # ax.set_title('Accuracy')
    # ax.legend(['Train', 'Test'], loc='lower right')

    # ROC
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(x_axis_train, results['roc_train'])
    # ax.plot(x_axis_test, results['roc_test'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('ROC AUC')
    ax.set_title('ROC AUC')
    ax.legend(['Train', 'Test'], loc='lower right')

    # Precision
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(x_axis_train, results['ap_train'])
    # ax.plot(x_axis_test, results['ap_test'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Precision')
    ax.set_title('Precision')
    ax.legend(['Train', 'Test'], loc='lower right')

    # Save
    fig.tight_layout()
    fig.savefig('{}/{}.png'.format(path,filename))
    
def plot_results_full(results, test_freq,filename, path='exp/results',epoch=1):
    # Init
    plt.close('all')
    fig = plt.figure(figsize=(8, 8))

    x_axis_train = range(len(results['train_loss']))
    x_axis_test = range(0, len(x_axis_train), test_freq)
    # Elbo
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(x_axis_train, results['train_loss'])
    ax.plot(x_axis_test, results['val_loss'])
    ax.set_ylabel('Loss ')
   
    ax.set_title('Loss till epoch {}'.format(epoch))
    ax.legend(['Train','Val'], loc='upper right')

    # Accuracy
    # ax = fig.add_subplot(2, 2, 2)
    # ax.plot(x_axis_train, results['accuracy_train'])
    # ax.plot(x_axis_test, results['accuracy_test'])
    # ax.set_ylabel('Accuracy')
    # ax.set_title('Accuracy')
    # ax.legend(['Train', 'Test'], loc='lower right')
    # DER
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(x_axis_test, results['baseline_DER'])
    ax.plot(x_axis_test, results['val_DER'])
    ax.set_ylabel('Avg. DER')
    ax.set_title('DER for Validation')
    ax.legend(['baseline', 'Val'], loc='lower right')

    # ROC
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(x_axis_train, results['roc_train'])
    ax.plot(x_axis_test, results['val_roc'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('ROC AUC')
    ax.set_title('ROC AUC')
    ax.legend(['Train', 'Val'], loc='lower right')

    # Precision
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(x_axis_train, results['ap_train'])
    ax.plot(x_axis_test, results['val_ap'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Precision')
    ax.set_title('Precision')
    ax.legend(['Train', 'Val'], loc='lower right')

    # Save
    fig.tight_layout()
    fig.savefig('{}/{}.png'.format(path,filename))

def plot_results_full_withoverlaplabels(results, test_freq,filename, path='exp/results'):
    # Init
    plt.close('all')
    fig = plt.figure(figsize=(8, 8))

    x_axis_train = range(len(results['train_loss']))
    x_axis_test = range(0, len(x_axis_train), test_freq)
    # Elbo
    ax = fig.add_subplot(3, 2, 1)
    ax.plot(x_axis_train, results['train_loss'])
    ax.plot(x_axis_test, results['val_loss'])
    ax.set_ylabel('Loss ')
    ax.set_title('Loss ')
    ax.legend(['Train','Val'], loc='upper right')

    # Accuracy
    # ax = fig.add_subplot(2, 2, 2)
    # ax.plot(x_axis_train, results['accuracy_train'])
    # ax.plot(x_axis_test, results['accuracy_test'])
    # ax.set_ylabel('Accuracy')
    # ax.set_title('Accuracy')
    # ax.legend(['Train', 'Test'], loc='lower right')
    # DER
    ax = fig.add_subplot(3, 2, 2)
    ax.plot(x_axis_test, results['baseline_DER'])
    ax.plot(x_axis_test, results['val_DER'])
    ax.set_ylabel('Avg. DER')
    ax.set_title('DER for Validation')
    ax.legend(['baseline', 'Val'], loc='lower right')

    # ROC
    ax = fig.add_subplot(3, 2, 3)
    ax.plot(x_axis_train, results['roc_train'])
    ax.plot(x_axis_test, results['val_roc'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('ROC AUC')
    ax.set_title('ROC AUC')
    ax.legend(['Train', 'Val'], loc='lower right')

    # Precision
    ax = fig.add_subplot(3, 2, 4)
    ax.plot(x_axis_train, results['ap_train'])
    ax.plot(x_axis_test, results['val_ap'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Precision')
    ax.set_title('Precision')
    ax.legend(['Train', 'Val'], loc='lower right')

    # overlap ROC
    ax = fig.add_subplot(3, 2, 5)
    ax.plot(x_axis_train, results['ovp_roc_train'])
    ax.plot(x_axis_test, results['val_ovp_roc'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('ROC AUC')
    ax.set_title('Overlap detection ROC AUC')
    ax.legend(['Train', 'Val'], loc='lower right')

    # Save
    fig.tight_layout()
    fig.savefig('{}/{}.png'.format(path,filename))
