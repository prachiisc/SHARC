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
# import pic_dihard as pic
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy import sparse
# from run_spectralclustering import do_spectral_clustering
import matplotlib as mat
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
def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        '''
        fix Pickle incompatibility of numpy arrays between Python 2 and 3
        https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3
        '''
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as rf:
            u = pkl._Unpickler(rf)
            u.encoding = 'latin1'
            cur_data = u.load()
            objects.append(cur_data)
        # objects.append(
        #     pkl.load(open("data/ind.{}.{}".format(dataset, names[i]), 'rb')))
    x, tx, allx, graph = tuple(objects)
    # bp()
    test_idx_reorder = parse_index_file(
        "data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = torch.FloatTensor(np.array(features.todense()))
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features

def load_data_simu(dataset,filename,device='cpu',useoverlap=1,set='val',batch=None):
    # load the data: x, tx, allx, graph
    # bp()
    if batch is None:
        if set=='train':
            if useoverlap:
                filepath = f'exp/{dataset}/ground_adj_cent_max2overlap/{filename}.pkl'
                cmd = f'mkdir -p exp/{dataset}/ground_adj_cent_max2overlap/'
            else:
                filepath = f'exp/{dataset}/ground_adj_cent_clean/{filename}.pkl'
                cmd = f'mkdir -p exp/{dataset}/ground_adj_cent_clean/'
            ground_labels=open('/data1/prachis/Dihard_2020/gae-pytorch/gae/tools_diar/ALL_GROUND_LABELS/'+dataset+'/threshold_0.5/labels_'+filename).readlines()

        else:
            filepath = f'exp/{dataset}/ground_adj_cent/{filename}.pkl'
            cmd = f'mkdir -p exp/{dataset}/ground_adj_cent/'
            ground_labels=open('/data1/prachis/Dihard_2020/gae-pytorch/gae/tools_diar/ALL_GROUND_LABELS/'+dataset+'/threshold_0.5_avg/labels_'+filename).readlines()

        xvecpath = '/data1/prachis/Dihard_2020/gae-pytorch/gae/tools_diar/xvectors_0.75s_npy/{}/'.format(dataset)
        # ground_labels=open('/data1/prachis/Dihard_2020/gae-pytorch/gae/tools_diar/ALL_GROUND_LABELS/'+dataset+'/threshold_0.5_avg/labels_'+filename).readlines()
        
        os.system(cmd)
        
        # dataset = 'dihard_dev_2020_track1_fbank_jhu'
        # print('filename:',filename)
        X = np.load('{}/{}.npy'.format(xvecpath,filename))
        features = torch.FloatTensor(X).to(device)

    else:
        filepath = f'exp/{dataset}/ground_adj_cent_max2overlap_batch/{filename[0]}.pkl'
        cmd = f'mkdir -p exp/{dataset}/ground_adj_cent_max2overlap_batch/'
        os.system(cmd)
        
        X = []
        ground_labels = []
        for f in filename:
            xvecpath = 'tools_diar/xvectors_0.75s_npy/{}/'.format(dataset)
            ground_label=open('tools_diar/ALL_GROUND_LABELS/'+dataset+'/threshold_0.5_avg/labels_'+f).readlines()
            ground_labels.extend(ground_label)
            X.append(np.load('{}/{}.npy'.format(xvecpath,f)))
        
        X = np.concatenate(X,axis=0)
        
        features = torch.FloatTensor(X).to(device)
    
    full_gndlist=[g.split()[1:] for g in ground_labels]
    gnd_list = np.array([g[0] for g in full_gndlist])
    uni_gnd_letter = np.unique(gnd_list)

    clean_list = np.array([len(f) for f in full_gndlist])
    clean_ind =np.where(clean_list ==1)[0]
    max2ovp_ind = np.where(clean_list <=2 )[0]
    
    if os.path.isfile(filepath):
        with open(filepath,'rb') as fb:
            mydict = pkl.load(fb)
        adj = mydict['adj_overlap']
        # final_cent = mydict['cent']
        return adj, features, clean_ind, max2ovp_ind
    if set == 'train':
        overlap_ind = np.where(clean_list ==2 )[0] # only 2 speakers overlap
    else:
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
   
    # plt.figure()
    # nx.draw_networkx(G,node_color=color_map,node_size=20, with_labels=False)
    # plt.subplot(211)
    # print("The added Graph:")
    # nx.draw_networkx(G)
    
    # plt.subplot(212)
    # print("The sub Graph:")
    # H = G.subgraph([0,1, 2, 3, 4,5,6,7,8,9,10])
    
    # nx.draw_networkx(H)
    
    # plt.savefig('exp/overlap_graphs/graph_{}.png'.format(filename))
    
    # features = sp.vstack((allx, tx)).tolil()
    # features[test_idx_reorder, :] = features[test_idx_range, :]
    
    adj = nx.adjacency_matrix(G)
    mydict = {}
    mydict['adj_overlap']= adj
    with open(filepath,'wb') as fb:
        pkl.dump(mydict,fb)
    # bp()
    return adj, features, clean_ind, max2ovp_ind
    # return adj, features, overlap_ind


def load_data_simu_labels(dataset,filename,device='cpu',useoverlap=1,set='val',batch=None):
    # load the data: x, tx, allx, graph
    # bp()
    if batch is None:
        if set=='train':
            if useoverlap:
                filepath = f'exp/{dataset}/ground_adj_cent_max2overlap/{filename}.pkl'
                cmd = f'mkdir -p exp/{dataset}/ground_adj_cent_max2overlap/'
            else:
                filepath = f'exp/{dataset}/ground_adj_cent_clean/{filename}.pkl'
                cmd = f'mkdir -p exp/{dataset}/ground_adj_cent_clean/'
            ground_labels=open('/data1/prachis/Dihard_2020/gae-pytorch/gae/tools_diar/ALL_GROUND_LABELS/'+dataset+'/threshold_0.5/labels_'+filename).readlines()

        else:
            filepath = f'exp/{dataset}/ground_adj_cent/{filename}.pkl'
            cmd = f'mkdir -p exp/{dataset}/ground_adj_cent/'
            ground_labels=open('/data1/prachis/Dihard_2020/gae-pytorch/gae/tools_diar/ALL_GROUND_LABELS/'+dataset+'/threshold_0.5_avg/labels_'+filename).readlines()

        xvecpath = '/data1/prachis/Dihard_2020/gae-pytorch/gae/tools_diar/xvectors_0.75s_npy/{}/'.format(dataset)
        # ground_labels=open('/data1/prachis/Dihard_2020/gae-pytorch/gae/tools_diar/ALL_GROUND_LABELS/'+dataset+'/threshold_0.5_avg/labels_'+filename).readlines()
        
        os.system(cmd)
        
        # dataset = 'dihard_dev_2020_track1_fbank_jhu'
        # print('filename:',filename)
        X = np.load('{}/{}.npy'.format(xvecpath,filename))
        features = torch.FloatTensor(X).to(device)

    else:
        filepath = f'exp/{dataset}/ground_adj_cent_max2overlap_batch/{filename[0]}.pkl'
        cmd = f'mkdir -p exp/{dataset}/ground_adj_cent_max2overlap_batch/'
        os.system(cmd)
        
        X = []
        ground_labels = []
        for f in filename:
            xvecpath = 'tools_diar/xvectors_0.75s_npy/{}/'.format(dataset)
            ground_label=open('tools_diar/ALL_GROUND_LABELS/'+dataset+'/threshold_0.5_avg/labels_'+f).readlines()
            ground_labels.extend(ground_label)
            X.append(np.load('{}/{}.npy'.format(xvecpath,f)))
        
        X = np.concatenate(X,axis=0)
        
        features = torch.FloatTensor(X).to(device)
    
    full_gndlist=[g.split()[1:] for g in ground_labels]
    gnd_list = np.array([g[0] for g in full_gndlist])
    uni_gnd_letter = np.unique(gnd_list)

    clean_list = np.array([len(f) for f in full_gndlist])
    clean_ind =np.where(clean_list ==1)[0]
    max2ovp_ind = np.where(clean_list <=2 )[0]
    
    if os.path.isfile(filepath):
        with open(filepath,'rb') as fb:
            mydict = pkl.load(fb)
        adj = mydict['adj_overlap']
        # final_cent = mydict['cent']
        return adj, features, clean_ind, max2ovp_ind,gnd_list
    if set == 'train':
        overlap_ind = np.where(clean_list ==2 )[0] # only 2 speakers overlap
    else:
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
   
    # plt.figure()
    # nx.draw_networkx(G,node_color=color_map,node_size=20, with_labels=False)
    # plt.subplot(211)
    # print("The added Graph:")
    # nx.draw_networkx(G)
    
    # plt.subplot(212)
    # print("The sub Graph:")
    # H = G.subgraph([0,1, 2, 3, 4,5,6,7,8,9,10])
    
    # nx.draw_networkx(H)
    
    # plt.savefig('exp/overlap_graphs/graph_{}.png'.format(filename))
    
    # features = sp.vstack((allx, tx)).tolil()
    # features[test_idx_reorder, :] = features[test_idx_range, :]
    
    adj = nx.adjacency_matrix(G)
    mydict = {}
    mydict['adj_overlap']= adj
    with open(filepath,'wb') as fb:
        pkl.dump(mydict,fb)
    # bp()
    return adj, features, clean_ind, max2ovp_ind,gnd_list


def load_data_simu_weightedlabels_avg(dataset,filename,device='cpu',useoverlap=1,set='val',batch=None):
    # load the data: x, tx, allx, graph
    # bp()
    if batch is None:
        if set=='train':
            if useoverlap:
                filepath = f'exp/{dataset}/ground_adj_cent_max2overlap_avg_weighted/{filename}.pkl'
                cmd = f'mkdir -p exp/{dataset}/ground_adj_cent_max2overlap_avg_weighted/'
            else:
                filepath = f'exp/{dataset}/ground_adj_cent_clean/{filename}.pkl'
                cmd = f'mkdir -p exp/{dataset}/ground_adj_cent_clean/'
            ground_labels=open('/data1/prachis/Dihard_2020/gae-pytorch/gae/tools_diar/ALL_GROUND_LABELS/'+dataset+'/threshold_0.5_avg/labels_'+filename).readlines()

        else:
            filepath = f'exp/{dataset}/ground_adj_cent_weighted/{filename}.pkl'
            cmd = f'mkdir -p exp/{dataset}/ground_adj_cent_weighted/'
            ground_labels=open('/data1/prachis/Dihard_2020/gae-pytorch/gae/tools_diar/ALL_GROUND_LABELS/'+dataset+'/threshold_0.5_avg/labels_'+filename).readlines()

        xvecpath = '/data1/prachis/Dihard_2020/gae-pytorch/gae/tools_diar/xvectors_0.75s_npy/{}/'.format(dataset)
        # ground_labels=open('/data1/prachis/Dihard_2020/gae-pytorch/gae/tools_diar/ALL_GROUND_LABELS/'+dataset+'/threshold_0.5_avg/labels_'+filename).readlines()
        
        os.system(cmd)
        
        # dataset = 'dihard_dev_2020_track1_fbank_jhu'
        # print('filename:',filename)
        X = np.load('{}/{}.npy'.format(xvecpath,filename))
        features = torch.FloatTensor(X).to(device)

    else:
        filepath = f'exp/{dataset}/ground_adj_cent_max2overlap_batch/{filename[0]}.pkl'
        cmd = f'mkdir -p exp/{dataset}/ground_adj_cent_max2overlap_batch/'
        os.system(cmd)
        
        X = []
        ground_labels = []
        for f in filename:
            xvecpath = 'tools_diar/xvectors_0.75s_npy/{}/'.format(dataset)
            ground_label=open('tools_diar/ALL_GROUND_LABELS/'+dataset+'/threshold_0.5_avg/labels_'+f).readlines()
            ground_labels.extend(ground_label)
            X.append(np.load('{}/{}.npy'.format(xvecpath,f)))
        
        X = np.concatenate(X,axis=0)
        
        features = torch.FloatTensor(X).to(device)
    
    full_gndlist=[g.split()[1:] for g in ground_labels]
    gnd_list = np.array([g[0] for g in full_gndlist])
    uni_gnd_letter = np.unique(gnd_list)

    clean_list = np.array([len(f) for f in full_gndlist])
    clean_ind =np.where(clean_list ==1)[0]
    max2ovp_ind = np.where(clean_list <=2 )[0]
    clean_list_sub = clean_list[max2ovp_ind]
    clean_ind_sub = np.where(clean_list_sub ==1)[0]
    if set == 'train':
        overlap_ind = np.where(clean_list ==2 )[0] # only 2 speakers overlap
    else:
        overlap_ind = np.where(clean_list > 1.0)[0]
    max_overlap = len(np.unique(clean_list))
    N = gnd_list.shape[0]
    if len(uni_gnd_letter) == 1:
        W_gnd = np.ones((N,1))
        # return W,full_overlap
    
    nframe = len(gnd_list)
    label_withoverlap=np.ones((nframe,max_overlap),dtype=int)*(-1)
    label_withoverlap = label_withoverlap.astype(str)
    
    overlap_gnd_list = np.array([full_gndlist[oi][1] for oi in overlap_ind])
    uni_overlap_letter = np.unique(overlap_gnd_list)
    uni_gnd_letter = np.unique(np.hstack((uni_gnd_letter,uni_overlap_letter)))
    uni_letter = uni_gnd_letter
    
    # max_overlap=2
    if max_overlap > 1: 
        # overlap_ind = []
        # overlap_gnd_list = []
        for count_ovp in range(2,max_overlap+1):
            overlap_ind=np.where(clean_list==count_ovp)[0]
        
            overlap_gnd_list = np.array([full_gndlist[oi][count_ovp-1] for oi in overlap_ind])
            uni_overlap_letter = np.unique(overlap_gnd_list)
            uni_letter = np.concatenate((uni_letter,uni_overlap_letter))
            # for ind,uni in enumerate(uni_overlap_letter):
            #     overlap_gnd_list[overlap_gnd_list==uni]=ind
            # overlap_gnd_list = overlap_gnd_list.astype(int)

            label_withoverlap[overlap_ind,count_ovp-1] = overlap_gnd_list
    
    label_withoverlap[:,0]= gnd_list
    
    uni_letter = np.unique(uni_letter)
    
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OneHotEncoder
    # bp()
    # integer encode
    label_encoder = LabelEncoder()
    label_encoder.fit(np.concatenate((uni_letter,np.array(['-1']))))
    integer_encoded1 = label_encoder.transform(np.concatenate((uni_letter,np.array(['-1']))))
    integer_encoded = label_encoder.transform(label_withoverlap.reshape(-1,))
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    # integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoder.fit(integer_encoded1.reshape(-1,1))
    onehot_encoded = onehot_encoder.transform(integer_encoded.reshape(-1,1))
    W = onehot_encoded[:,1:]
    N = label_withoverlap.shape[0]
    spks = W.shape[1]
    W = W.reshape(N,-1,spks)
    W_gnd = W.sum(axis=1)
    # W = W/W.sum(axis=1).reshape(-1,1)
    
    
    if 0: #os.path.isfile(filepath):
        with open(filepath,'rb') as fb:
            mydict = pkl.load(fb)
        adj = mydict['adj_overlap']
        return adj, features,clean_ind, max2ovp_ind,W_gnd, clean_ind_sub
    
    G = nx.OrderedGraph()
    N = len(gnd_list)
    L = np.arange(N)
    G.add_nodes_from(L)
    # color_list = ['blue','green','black','yellow','magenta']
    # color_map = L.astype(str)
    
    if useoverlap and overlap_gnd_list.shape[0]>0:
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
   
  
    
    adj = nx.adjacency_matrix(G)
    mydict = {}
    mydict['adj_overlap']= adj
    with open(filepath,'wb') as fb:
        pkl.dump(mydict,fb)
    # bp()
    return adj, features, clean_ind, max2ovp_ind,W_gnd,clean_ind_sub


def load_data_dihard(dataset,filename,device='cpu',useoverlap=1,outf=None,set='val'):
    # load the data: x, tx, allx, graph
    kaldi_recipe_path='/data1/prachis/Dihard_2020/Dihard_2020_track1'
    
    if set == 'train':
        filepath = f'exp/{dataset}/ground_adj_cent_max2overlap/{filename}.pkl'
        cmd = f'mkdir -p exp/{dataset}/ground_adj_cent_max2overlap/'
        xvecpath = 'tools_diar/xvectors_0.75s_npy/{}/'.format(dataset)
        ground_labels=open('/data1/prachis/Dihard_2020/SSC/ALL_GROUND_LABELS/'+dataset+'_0.75s'+'/threshold_0.5_avg/labels_'+filename).readlines()

    else:
        filepath = f'exp/{dataset}/ground_adj_val/{filename}.pkl'
        cmd = f'mkdir -p exp/{dataset}/ground_adj_val/'
        xvecpath = '{}/xvectors_npy/{}/'.format(kaldi_recipe_path,dataset)
        ground_labels=open('/data1/prachis/Dihard_2020/SSC/ALL_GROUND_LABELS/'+dataset.split('_fbank')[0]+'/threshold_0.5_avg/labels_'+filename).readlines()

    os.system(cmd)
    
    
    # dataset = 'dihard_dev_2020_track1_fbank_jhu'
   

    # print('filename:',filename)
    X = np.load('{}/{}.npy'.format(xvecpath,filename))
    features = torch.FloatTensor(X).to(device)
    
    full_gndlist=[g.split()[1:] for g in ground_labels]
    gnd_list = np.array([g[0] for g in full_gndlist])
    uni_gnd_letter = np.unique(gnd_list)

    clean_list = np.array([len(f) for f in full_gndlist])
    clean_ind =np.where(clean_list ==1)[0]
    max2ovp_ind = np.where(clean_list <=2 )[0]
    
    if os.path.isfile(filepath):
        with open(filepath,'rb') as fb:
            mydict = pkl.load(fb)
        adj = mydict['adj_overlap']
        # final_cent = mydict['cent']
        return adj, features, clean_ind, max2ovp_ind
    
    if set == 'train':
        overlap_ind = np.where(clean_list ==2 )[0] # only 2 speakers overlap
    else:
        overlap_ind = np.where(clean_list > 1.0)[0]
    
    overlap_gnd_list = np.array([full_gndlist[oi][1] for oi in overlap_ind])
    uni_overlap_letter = np.unique(overlap_gnd_list)
    uni_gnd_letter = np.unique(np.hstack((uni_gnd_letter,uni_overlap_letter)))
  

    # for ind,uni in enumerate(uni_gnd_letter):
    #     gnd_list[gnd_list==uni]=ind
        
    # gnd_list = gnd_list.astype(int)
    # for ind,uni in enumerate(uni_gnd_letter):
    #     overlap_gnd_list[overlap_gnd_list==uni]=ind

    # will result in different index to overlapping speakers
    # for ind,uni in enumerate(uni_overlap_letter):
    #     overlap_gnd_list[overlap_gnd_list==uni]=ind
        
    
    # uni_gnd = np.arange(len(uni_gnd_letter))

    # overlap_gnd_list = overlap_gnd_list.astype(int)
    
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
    
    if useoverlap and len(overlap_ind)>0:
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
   
    # plt.figure()
    # nx.draw_networkx(G,node_color=color_map,node_size=20, with_labels=False)
    # plt.subplot(211)
    # print("The added Graph:")
    # nx.draw_networkx(G)
    
    # plt.subplot(212)
    # print("The sub Graph:")
    # H = G.subgraph([0,1, 2, 3, 4,5,6,7,8,9,10])
    
    # nx.draw_networkx(H)
    
    # plt.savefig('exp/overlap_graphs/graph_{}.png'.format(filename))
    
    # features = sp.vstack((allx, tx)).tolil()
    # features[test_idx_reorder, :] = features[test_idx_range, :]
    
    adj = nx.adjacency_matrix(G)
    mydict = {}
    mydict['adj_overlap']= adj
    with open(filepath,'wb') as fb:
        pkl.dump(mydict,fb)
    # bp()
    return adj, features, clean_ind, max2ovp_ind
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
    # bp()
    # plt.figure()
    # nx.draw_networkx(G,node_color=color_map,node_size=20, with_labels=False)
    # plt.subplot(211)
    # print("The added Graph:")
    # nx.draw_networkx(G)
    
    # plt.subplot(212)
    # print("The sub Graph:")
    # H = G.subgraph([0,1, 2, 3, 4,5,6,7,8,9,10])
    
    # nx.draw_networkx(H)
    
    # plt.savefig('exp/overlap_graphs/graph_{}.png'.format(filename))
    
    # features = sp.vstack((allx, tx)).tolil()
    # features[test_idx_reorder, :] = features[test_idx_range, :]
    
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
    kaldi_recipe_path='/data1/prachis/Dihard_2020/Dihard_2020_track1'
    
    if set == 'train':
        filepath = f'exp/{dataset}/ground_adj_cent_max2overlap_weighted/{filename}.pkl'
        cmd = f'mkdir -p exp/{dataset}/ground_adj_cent_max2overlap_weighted/'
        xvecpath = 'tools_diar/xvectors_0.75s_npy/{}/'.format(dataset)
        ground_labels=open('/data1/prachis/Dihard_2020/SSC/ALL_GROUND_LABELS/'+dataset+'_0.75s'+'/threshold_0.5_avg/labels_'+filename).readlines()

    else:
        filepath = f'exp/{dataset}/ground_adj_cent_weighted/{filename}.pkl'
        cmd = f'mkdir -p exp/{dataset}/ground_adj_cent_weighted/'
        xvecpath = '{}/xvectors_npy/{}/'.format(kaldi_recipe_path,dataset)
        ground_labels=open('/data1/prachis/Dihard_2020/SSC/ALL_GROUND_LABELS/'+dataset.split('_fbank')[0]+'/threshold_0.5_avg/labels_'+filename).readlines()

    os.system(cmd)
    
    # print('filename:',filename)
    X = np.load('{}/{}.npy'.format(xvecpath,filename))
    features = torch.FloatTensor(X).to(device)
    full_gndlist=[g.split()[1:] for g in ground_labels]
    gnd_list = np.array([g[0] for g in full_gndlist])
    uni_gnd_letter = np.unique(gnd_list)

    clean_list = np.array([len(f) for f in full_gndlist])
    clean_ind =np.where(clean_list ==1)[0]
    max2ovp_ind = np.where(clean_list <=2 )[0]
    
    if os.path.isfile(filepath):
        with open(filepath,'rb') as fb:
            mydict = pkl.load(fb)
        adj = mydict['adj_overlap']
        # if set == 'val':
        #     # try:
        #     #     final_cent = mydict['cent']
        #     #     return adj, features, clean_ind, final_cent
        #     # except:
        #         return adj, features, clean_ind
        # else:
            # return adj, features, clean_ind
        return adj, features, clean_ind, max2ovp_ind

    # overlap_ind = np.where(clean_list > 1)[0]
    overlap_ind = np.where(clean_list == 2)[0] # segments with only 2 speaker overlaps
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
    
    if useoverlap and overlap_gnd_list.shape[0]>0:
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
    # if set=='val':
    #     cent=nx.clustering(G)
    #     bet=nx.betweenness_centrality(G)
    #     final_cent = np.array(list(cent.values()))*(1-np.array(list(bet.values())))
            
    #     pred_overlap_ind = np.where(final_cent<1)[0]
    #     intersection = list(set(overlap_ind).intersection(pred_overlap_ind))
    #     print('intersection: ',len(intersection))
    #     if len(intersection)!=0:
    #         print('overlap precision: ', len(intersection)/len(pred_overlap_ind),'overlap recall: ', len(intersection)/len(overlap_ind))
    #     mydict['cent'] = final_cent
    #     with open(filepath,'wb') as fb:
    #         pkl.dump(mydict,fb)
    #     return adj, features, clean_ind, final_cent
    # else:
    #     return adj, features, clean_ind
    
    return adj, features, clean_ind, max2ovp_ind

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

def load_data_simu_plda_feats(dataset,filename,n_clusters,device='cpu',threshold=None,set='val',scale=1,batch=None):
    # load the data: x, tx, allx, graph
    # scale = 1
   
    if scale> 1:
        scale_str =f'_scale{scale}'
    else:
        scale_str=''
    if batch is None:
        filepath = f'exp/{dataset}/plda_adj_A_0.75s{scale_str}/{filename}.pkl'
        cmd = f'mkdir -p exp/{dataset}/plda_adj_A_0.75s{scale_str}/'
        os.system(cmd)
    else:
        filepath = f'exp/{dataset}/plda_adj_A_0.75s{scale_str}_batch/{filename[0]}.pkl'
        cmd = f'mkdir -p exp/{dataset}/plda_adj_A_0.75s{scale_str}_batch/'
        os.system(cmd)
        
    xvecpath = 'tools_diar/xvectors_0.75s_npy/{}/'.format(dataset)
    
    if os.path.isfile(filepath):
        with open(filepath,'rb') as fb:
            A = pkl.load(fb)
        if batch is None:
            X = np.load('{}/{}.npy'.format(xvecpath,filename))
            features = torch.FloatTensor(X).to(device) 
        if batch is not None and set == 'train':
            return A,None
    else:
        device = 'cpu'
        kaldi_recipe_path='/data1/prachis/Dihard_2020/Dihard_2020_track1'
        pldadataset = 'dihard_dev_2020_track1_fbank_jhu_wide'
        pldamodel= '/data1/prachis/Dihard_2020/SSC/lists/{0}/plda_{0}.pkl'.format(pldadataset)
        pldamodel = pkl.load(open(pldamodel,'rb'))
        # dataset = 'dihard_dev_2020_track1_fbank_jhu'

        xvecpath = 'tools_diar/xvectors_0.75s_npy/{}/'.format(dataset)
        if batch is None:
            X = np.load('{}/{}.npy'.format(xvecpath,filename))
        else:
            X = []
            for f in filename:
                X.append(np.load('{}/{}.npy'.format(xvecpath,f)))
            X = np.concatenate(X,axis=0)
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
        if batch is None:
            with open(filepath,'wb') as fb:
                pkl.dump(A,fb)

    # directly feeding adjacency matrix
    B = A.copy()
    B[B<=0.5] =0
    adj = sparse.csr_matrix(B)
    # G = nx.from_numpy_matrix(B)
   
    # adj = nx.adjacency_matrix(G)
    
    # KNN
    # B = -A.copy()
    # K = 30
    # N = X.shape[0]
    # K = min(K,N-1)
    # B[np.diag_indices(N)] = -np.inf
    # sortedDist = np.sort(B,axis=1)
    # NNIndex = np.argsort(B,axis=1)
    # NNIndex = NNIndex[:,:K+1]
    # ND = -sortedDist[:, 1:K+1].copy()
    # NI = NNIndex[:, 1:K+1].copy()
    # XI = np.dot(np.arange(N).reshape(-1,1),np.ones((1,K),dtype=int))
    # graphW = coo_matrix((ND.reshape(-1), (XI.reshape(-1),  NI.reshape(-1))), shape=(N, N)).toarray()
    # graphW[np.diag_indices(N)]=1
    
    # adj = graphW
    
    # G = nx.from_numpy_matrix(graphW)
    
    if batch is not None and set == 'train':
            with open(filepath,'wb') as fb:
                pkl.dump(adj,fb)
    return adj,features, A


def load_data_simu_plda_ami_feats(dataset,filename,n_clusters,device='cpu',threshold=None,set='val',scale=1,batch=None,th=0.5):
    # load the data: x, tx, allx, graph
    # scale = 1
   
    if scale> 1:
        scale_str =f'_scale{scale}'
    else:
        scale_str=''
    if batch is None:
        filepath = f'exp/{dataset}/plda_ami_adj_A_0.75s{scale_str}/{filename}.pkl'
        cmd = f'mkdir -p exp/{dataset}/plda_ami_adj_A_0.75s{scale_str}/'
        os.system(cmd)
    else:
        filepath = f'exp/{dataset}/plda_ami_adj_A_0.75s{scale_str}_batch/{filename[0]}.pkl'
        cmd = f'mkdir -p exp/{dataset}/plda_ami_adj_A_0.75s{scale_str}_batch/'
        os.system(cmd)
        
    xvecpath = 'tools_diar/xvectors_0.75s_npy/{}/'.format(dataset)
    
    if os.path.isfile(filepath):
        with open(filepath,'rb') as fb:
            A = pkl.load(fb)
        if batch is None:
            X = np.load('{}/{}.npy'.format(xvecpath,filename))
            features = torch.FloatTensor(X).to(device) 
        if batch is not None and set == 'train':
            return A,None
    else:
        device = 'cpu'
        
        pldadataset = 'ami_sdm_train_gnd'
        pldamodel= 'lists/{0}/plda_{0}.pkl'.format(pldadataset)

        pldamodel = pkl.load(open(pldamodel,'rb'))
        # dataset = 'dihard_dev_2020_track1_fbank_jhu'

        xvecpath = 'tools_diar/xvectors_0.75s_npy/{}/'.format(dataset)
        if batch is None:
            X = np.load('{}/{}.npy'.format(xvecpath,filename))
        else:
            X = []
            for f in filename:
                X.append(np.load('{}/{}.npy'.format(xvecpath,f)))
            X = np.concatenate(X,axis=0)
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
        if batch is None:
            with open(filepath,'wb') as fb:
                pkl.dump(A,fb)

    # directly feeding adjacency matrix
    B = A.copy()
    B[B<=th] =0
    adj = sparse.csr_matrix(B)
    # G = nx.from_numpy_matrix(B)
   
    # adj = nx.adjacency_matrix(G)
    
    # KNN
    # B = -A.copy()
    # K = 30
    # N = X.shape[0]
    # K = min(K,N-1)
    # B[np.diag_indices(N)] = -np.inf
    # sortedDist = np.sort(B,axis=1)
    # NNIndex = np.argsort(B,axis=1)
    # NNIndex = NNIndex[:,:K+1]
    # ND = -sortedDist[:, 1:K+1].copy()
    # NI = NNIndex[:, 1:K+1].copy()
    # XI = np.dot(np.arange(N).reshape(-1,1),np.ones((1,K),dtype=int))
    # graphW = coo_matrix((ND.reshape(-1), (XI.reshape(-1),  NI.reshape(-1))), shape=(N, N)).toarray()
    # graphW[np.diag_indices(N)]=1
    
    # adj = graphW
    
    # G = nx.from_numpy_matrix(graphW)
    
    if batch is not None and set == 'train':
            with open(filepath,'wb') as fb:
                pkl.dump(adj,fb)
    return adj,features, A


def load_data_simu_plda_feats_knn(dataset,filename,n_clusters,device='cpu',threshold=None,set='val',scale=1,batch=None,myK=50):
    # load the data: x, tx, allx, graph
    # scale = 1
   
    if scale> 1:
        scale_str =f'_scale{scale}'
    else:
        scale_str=''
    if batch is None:
        filepath = f'exp/{dataset}/plda_adj_A_0.75s{scale_str}/{filename}.pkl'
        cmd = f'mkdir -p exp/{dataset}/plda_adj_A_0.75s{scale_str}/'
        os.system(cmd)
    else:
        filepath = f'exp/{dataset}/plda_adj_A_0.75s{scale_str}_batch/{filename[0]}.pkl'
        cmd = f'mkdir -p exp/{dataset}/plda_adj_A_0.75s{scale_str}_batch/'
        os.system(cmd)
        
    xvecpath = 'tools_diar/xvectors_0.75s_npy/{}/'.format(dataset)
    
    if os.path.isfile(filepath):
        with open(filepath,'rb') as fb:
            A = pkl.load(fb)
        if batch is None:
            X = np.load('{}/{}.npy'.format(xvecpath,filename))
            features = torch.FloatTensor(X).to(device) 
        if batch is not None and set == 'train':
            return A,None
    else:
        device = 'cpu'
        kaldi_recipe_path='/data1/prachis/Dihard_2020/Dihard_2020_track1'
        pldadataset = 'dihard_dev_2020_track1_fbank_jhu_wide'
        pldamodel= '/data1/prachis/Dihard_2020/SSC/lists/{0}/plda_{0}.pkl'.format(pldadataset)
        pldamodel = pkl.load(open(pldamodel,'rb'))
        # dataset = 'dihard_dev_2020_track1_fbank_jhu'

        xvecpath = 'tools_diar/xvectors_0.75s_npy/{}/'.format(dataset)
        if batch is None:
            X = np.load('{}/{}.npy'.format(xvecpath,filename))
        else:
            X = []
            for f in filename:
                X.append(np.load('{}/{}.npy'.format(xvecpath,f)))
            X = np.concatenate(X,axis=0)
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
        if batch is None:
            with open(filepath,'wb') as fb:
                pkl.dump(A,fb)

    # directly feeding adjacency matrix
    B = A.copy()
    # B[B<=0.5] =0
    # adj = sparse.csr_matrix(B)
    # G = nx.from_numpy_matrix(B)
   
    # adj = nx.adjacency_matrix(G)
    
    # KNN
    B = -A.copy()
    K = myK
    N = X.shape[0]
    K = min(K,N-1)
    B[np.diag_indices(N)] = -np.inf
    sortedDist = np.sort(B,axis=1)
    NNIndex = np.argsort(B,axis=1)
    NNIndex = NNIndex[:,:K+1]
    ND = -sortedDist[:, 1:K+1].copy()
    NI = NNIndex[:, 1:K+1].copy()
    XI = np.dot(np.arange(N).reshape(-1,1),np.ones((1,K),dtype=int))
    graphW = coo_matrix((ND.reshape(-1), (XI.reshape(-1),  NI.reshape(-1))), shape=(N, N)).toarray()
    graphW[np.diag_indices(N)]=1
    
    adj = graphW
    
    # G = nx.from_numpy_matrix(graphW)
    
    if batch is not None and set == 'train':
            with open(filepath,'wb') as fb:
                pkl.dump(adj,fb)
    return adj,features, A

def load_data_simu_cosine_ami(dataset,filename,n_clusters,device='cpu',threshold=None,set='val',scale=1,batch=None,myK=50,filepca=1):
    xvecpath = 'tools_diar/xvectors_0.75s_npy/{}/'.format(dataset)
    X = np.load('{}/{}.npy'.format(xvecpath,filename))
    xvecD = X.shape[1]
    features = torch.FloatTensor(X).to(device)
    pca_dim = 30
    inpdata = features[np.newaxis]
    # kaldi_recipe_path='/data1/prachis/Dihard_2020/Dihard_2020_track1'
    # pldadataset = 'dihard_dev_2020_track1_fbank_jhu_wide'
    pldadataset = 'ami_sdm_train_gnd'
    pldamodel= 'lists/{0}/plda_{0}.pkl'.format(pldadataset)
    pldamodel = pkl.load(open(pldamodel,'rb'))

    net_init = weight_initialization(pldamodel,dimension=xvecD,pca_dimension=pca_dim,device=device)
    model_init = net_init.to(device)

    affinity_init= model_init.compute_affinity_matrix(inpdata,filepca=filepca) # original filewise PCA transform
    output_model = affinity_init.detach().cpu().numpy()[0]

    return output_model

def load_data_simu_cosine_ami_knn(dataset,filename,device='cpu',threshold=None,set='val',scale=1,myK=50,filepca=1):
    xvecpath = 'tools_diar/xvectors_0.75s_npy/{}/'.format(dataset)
    X = np.load('{}/{}.npy'.format(xvecpath,filename))
    xvecD = X.shape[1]
    features = torch.FloatTensor(X).to(device)
    pca_dim = 30
    inpdata = features[np.newaxis]
    # kaldi_recipe_path='/data1/prachis/Dihard_2020/Dihard_2020_track1'
    # pldadataset = 'dihard_dev_2020_track1_fbank_jhu_wide'
    pldadataset = 'ami_sdm_train_gnd'
    pldamodel= 'lists/{0}/plda_{0}.pkl'.format(pldadataset)
    pldamodel = pkl.load(open(pldamodel,'rb'))

    net_init = weight_initialization(pldamodel,dimension=xvecD,pca_dimension=pca_dim,device=device)
    model_init = net_init.to(device)

    affinity_init,_= model_init.compute_affinity_matrix(inpdata,filepca=filepca) # original filewise PCA transform
    output_model = affinity_init.detach().cpu().numpy()[0]
    A = (output_model+1.0)/2.0

    B = -A.copy()
    K = myK
    N = X.shape[0]
    K = min(K,N-1)
    B[np.diag_indices(N)] = -np.inf
    sortedDist = np.sort(B,axis=1)
    NNIndex = np.argsort(B,axis=1)
    NNIndex = NNIndex[:,:K+1]
    ND = -sortedDist[:, 1:K+1].copy()
    NI = NNIndex[:, 1:K+1].copy()
    XI = np.dot(np.arange(N).reshape(-1,1),np.ones((1,K),dtype=int))
    graphW = coo_matrix((ND.reshape(-1), (XI.reshape(-1),  NI.reshape(-1))), shape=(N, N)).toarray()
    graphW[np.diag_indices(N)]=1
    
    adj = graphW
    return adj,A

def load_data_simu_cosine_ami_withfeats(dataset,filename,device='cpu',set='val',scale=1,filepca=1,type=None):
    xvecpath = 'tools_diar/xvectors_0.75s_npy/{}/'.format(dataset)
    X = np.load('{}/{}.npy'.format(xvecpath,filename))
    xvecD = X.shape[1]
    # bp()
    features = torch.FloatTensor(X).to(device)
    pca_dim = 30
    inpdata = features[np.newaxis]
    # kaldi_recipe_path='/data1/prachis/Dihard_2020/Dihard_2020_track1'
    # pldadataset = 'dihard_dev_2020_track1_fbank_jhu_wide'
    pldadataset = 'ami_sdm_train_gnd'
    pldamodel= 'lists/{0}/plda_{0}.pkl'.format(pldadataset)
    pldamodel = pkl.load(open(pldamodel,'rb'))

    net_init = weight_initialization(pldamodel,dimension=xvecD,pca_dimension=pca_dim,device=device)
    model_init = net_init.to(device)
    # bp()
    affinity_init,X_trans= model_init.compute_affinity_matrix(inpdata,filepca=filepca,type=type) # original filewise PCA transform
    output_model = affinity_init.detach().cpu().numpy()[0]
    features = X_trans[0]
    A = (output_model+1.0)/2.0
    return A,features

def compute_global_PCAdimreduce(dataset,device='cpu',set='val'):
    xvecpath = 'tools_diar/xvectors_0.75s_npy/{}/'.format(dataset)
    import glob
    files = glob.glob(f'{xvecpath}/*.npy')
    X = []
    for filen in files:
        X.append(np.load(filen))
    X = np.concatenate(X,0)
    xvecD = X.shape[1]
    features = torch.FloatTensor(X).to(device)
    pca_dim = 30
    inpdata = features[np.newaxis]
    pldadataset = 'ami_sdm_train_gnd'
    pldamodel= 'lists/{0}/plda_{0}.pkl'.format(pldadataset)
    pldamodel = pkl.load(open(pldamodel,'rb'))

    net_init = weight_initialization_withoutfilePCA(pldamodel,dimension=xvecD,pca_dimension=pca_dim,device=device)
    model_init = net_init.to(device)
    PCA = model_init.compute_PCAglobal_dimreduce(pldamodel,inpdata)
    PCA_transform = PCA.detach().cpu().numpy()
    np.save(f'lists/{pldadataset}/global_pca_30d.npy',PCA_transform)

def load_data_simu_plda_withoutfilepca_ami(dataset,filename,n_clusters,device='cpu',threshold=None,set='val',scale=1,batch=None,myK=50,th=0.5):
    xvecpath = 'tools_diar/xvectors_0.75s_npy/{}/'.format(dataset)
    X = np.load('{}/{}.npy'.format(xvecpath,filename))
    xvecD = X.shape[1]
    features = torch.FloatTensor(X).to(device)
    pca_dim = 30
    inpdata = features[np.newaxis]
    # kaldi_recipe_path='/data1/prachis/Dihard_2020/Dihard_2020_track1'
    # pldadataset = 'dihard_dev_2020_track1_fbank_jhu_wide'
    pldadataset = 'ami_sdm_train_gnd'
    pldamodel= 'lists/{0}/plda_{0}.pkl'.format(pldadataset)
    pldamodel = pkl.load(open(pldamodel,'rb'))

    net_init = weight_initialization_withoutfilePCA(pldamodel,dimension=xvecD,pca_dimension=pca_dim,device=device)
    model_init = net_init.to(device)
    # PCA_transform = np.load(f'lists/{pldadataset}/global_pca_30d.npy')
    # PCA_transform = torch.FloatTensor(PCA_transform).to(device)
    affinity_init= model_init.compute_plda_affinity_matrix(pldamodel,inpdata) # no filewise PCA transform
    # affinity_init= model_init.compute_plda_affinity_matrix_dimreduce(pldamodel,inpdata,PCA_transform)
    output_model = affinity_init.detach().cpu().numpy()[0]
    A = expit(output_model*scale)

    # B = A.copy()
    # B[B<=th] = 0
    # adj = sparse.csr_matrix(B)

    # KNN
    B = -A.copy()
    K = myK
    N = X.shape[0]
    K = min(K,N-1)
    B[np.diag_indices(N)] = -np.inf
    sortedDist = np.sort(B,axis=1)
    NNIndex = np.argsort(B,axis=1)
    NNIndex = NNIndex[:,:K+1]
    ND = -sortedDist[:, 1:K+1].copy()
    NI = NNIndex[:, 1:K+1].copy()
    XI = np.dot(np.arange(N).reshape(-1,1),np.ones((1,K),dtype=int))
    graphW = coo_matrix((ND.reshape(-1), (XI.reshape(-1),  NI.reshape(-1))), shape=(N, N)).toarray()
    graphW[np.diag_indices(N)]=1
    
    adj = graphW

    return adj,A

def load_data_simu_plda_ami_feats_knn(dataset,filename,n_clusters,device='cpu',threshold=None,set='val',scale=1,batch=None,myK=50):
    # load the data: x, tx, allx, graph
    # scale = 1
   
    if scale> 1:
        scale_str =f'_scale{scale}'
    else:
        scale_str=''
    if batch is None:
        filepath = f'exp/{dataset}/plda_ami_adj_A_0.75s{scale_str}/{filename}.pkl'
        cmd = f'mkdir -p exp/{dataset}/plda_ami_adj_A_0.75s{scale_str}/'
        os.system(cmd)
    else:
        filepath = f'exp/{dataset}/plda_ami_adj_A_0.75s{scale_str}_batch/{filename[0]}.pkl'
        cmd = f'mkdir -p exp/{dataset}/plda_ami_adj_A_0.75s{scale_str}_batch/'
        os.system(cmd)
        
    xvecpath = 'tools_diar/xvectors_0.75s_npy/{}/'.format(dataset)
    
    if os.path.isfile(filepath):
        with open(filepath,'rb') as fb:
            A = pkl.load(fb)
        if batch is None:
            X = np.load('{}/{}.npy'.format(xvecpath,filename))
            features = torch.FloatTensor(X).to(device) 
        if batch is not None and set == 'train':
            return A,None
    else:
        device = 'cpu'
        kaldi_recipe_path='/data1/prachis/Dihard_2020/Dihard_2020_track1'
        # pldadataset = 'dihard_dev_2020_track1_fbank_jhu_wide'
        pldadataset = 'ami_sdm_train_gnd'
        pldamodel= 'lists/{0}/plda_{0}.pkl'.format(pldadataset)
        pldamodel = pkl.load(open(pldamodel,'rb'))
        # dataset = 'dihard_dev_2020_track1_fbank_jhu'

        xvecpath = 'tools_diar/xvectors_0.75s_npy/{}/'.format(dataset)
        if batch is None:
            X = np.load('{}/{}.npy'.format(xvecpath,filename))
        else:
            X = []
            for f in filename:
                X.append(np.load('{}/{}.npy'.format(xvecpath,f)))
            X = np.concatenate(X,axis=0)
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
        if batch is None:
            with open(filepath,'wb') as fb:
                pkl.dump(A,fb)

    # directly feeding adjacency matrix
    B = A.copy()
    # B[B<=0.5] =0
    # adj = sparse.csr_matrix(B)
    # G = nx.from_numpy_matrix(B)
   
    # adj = nx.adjacency_matrix(G)
    
    # KNN
    B = -A.copy()
    K = myK
    N = X.shape[0]
    K = min(K,N-1)
    B[np.diag_indices(N)] = -np.inf
    sortedDist = np.sort(B,axis=1)
    NNIndex = np.argsort(B,axis=1)
    NNIndex = NNIndex[:,:K+1]
    ND = -sortedDist[:, 1:K+1].copy()
    NI = NNIndex[:, 1:K+1].copy()
    XI = np.dot(np.arange(N).reshape(-1,1),np.ones((1,K),dtype=int))
    graphW = coo_matrix((ND.reshape(-1), (XI.reshape(-1),  NI.reshape(-1))), shape=(N, N)).toarray()
    graphW[np.diag_indices(N)]=1
    
    adj = graphW
    
    # G = nx.from_numpy_matrix(graphW)
    adj = sparse.csr_matrix(adj)
    
    if batch is not None and set == 'train':
            with open(filepath,'wb') as fb:
                pkl.dump(adj,fb)
    return adj,features, A



def load_data_simu_plda(dataset,filename,n_clusters,device='cpu',threshold=None,set='val',scale=1,batch=None,clustering='spectral'):
    # load the data: x, tx, allx, graph
    # scale = 1
    
    if scale> 1:
        scale_str =f'_scale{scale}'
    else:
        scale_str=''
    
    if batch is None:
        if clustering == 'ahc':
            strclustering = '_ahc'
        else:
            strclustering = ''
        filepath = f'exp/{dataset}/plda_adj_A_0.75s{scale_str}{strclustering}/{filename}.pkl'
        cmd = f'mkdir -p exp/{dataset}/plda_adj_A_0.75s{scale_str}{strclustering}/'
        os.system(cmd)
    else:
        filepath = f'exp/{dataset}/plda_adj_A_0.75s{scale_str}_batch/{filename[0]}.pkl'
        cmd = f'mkdir -p exp/{dataset}/plda_adj_A_0.75s{scale_str}_batch/'
        os.system(cmd)
    if os.path.isfile(filepath):
            with open(filepath,'rb') as fb:
                A = pkl.load(fb)
            if batch is not None and set == 'train':
                return A,None
    else:
        device = 'cpu'
        kaldi_recipe_path='/data1/prachis/Dihard_2020/Dihard_2020_track1'
        pldadataset = 'dihard_dev_2020_track1_fbank_jhu_wide'
        pldamodel= '/data1/prachis/Dihard_2020/SSC/lists/{0}/plda_{0}.pkl'.format(pldadataset)
        pldamodel = pkl.load(open(pldamodel,'rb'))
        # dataset = 'dihard_dev_2020_track1_fbank_jhu'

        xvecpath = 'tools_diar/xvectors_0.75s_npy/{}/'.format(dataset)
        if batch is None:
            X = np.load('{}/{}.npy'.format(xvecpath,filename))
        else:
            X = []
            for f in filename:
                X.append(np.load('{}/{}.npy'.format(xvecpath,f)))
            X = np.concatenate(X,axis=0)
        features = torch.FloatTensor(X).to(device)
       
        xvecD = X.shape[1]
        pca_dim = 30
        inpdata = features[np.newaxis]
        net_init = weight_initialization(pldamodel,dimension=xvecD,pca_dimension=pca_dim,device=device)
        model_init = net_init.to(device)
        affinity_init,_,_ = model_init.compute_plda_affinity_matrix(pldamodel,inpdata) # original filewise PCA transform
        output_model = affinity_init.detach().cpu().numpy()[0]
        # bp()
        if clustering == 'ahc':
            A = output_model*scale
        else:
            A = expit(output_model*scale)
        if batch is None:
            with open(filepath,'wb') as fb:
                pkl.dump(A,fb)

    # directly feeding adjacency matrix
    if clustering == 'ahc':
        B = expit(A).copy()
    else:
        B = A.copy()
    B[B<=0.5] =0
    G = nx.from_numpy_matrix(B)
   
    adj = nx.adjacency_matrix(G)
    if batch is not None and set == 'train':
            with open(filepath,'wb') as fb:
                pkl.dump(adj,fb)
    return adj, A


def load_data_simu_plda_ami(dataset,filename,n_clusters,device='cpu',threshold=None,set='val',scale=1,\
                            batch=None,th=0.5,clustering='spectral',\
                                xvecpath=None,isadj=None,pldamodel=None,X=None):
    # load the data: x, tx, allx, graph
    # scale = 1
    
    if scale> 1:
        scale_str =f'_scale{scale}'
    else:
        scale_str=''
    if batch is None:
        if clustering == 'ahc':
            strclustering = '_ahc'
        else:
            strclustering = ''
        filepath = f'exp/{dataset}/plda_ami_adj_A_0.75s{scale_str}{strclustering}/{filename}.pkl'
        cmd = f'mkdir -p exp/{dataset}/plda_ami_adj_A_0.75s{scale_str}{strclustering}/'
        os.system(cmd)
    else:
        filepath = f'exp/{dataset}/plda_ami_adj_A_0.75s{scale_str}_batch/{filename[0]}.pkl'
        cmd = f'mkdir -p exp/{dataset}/plda_ami_adj_A_0.75s{scale_str}_batch/'
        os.system(cmd)
    if os.path.isfile(filepath):
            with open(filepath,'rb') as fb:
                A = pkl.load(fb)
            if batch is not None and set == 'train':
                return A,None
    else:
        device = 'cpu'
        kaldi_recipe_path='/data1/prachis/Dihard_2020/Dihard_2020_track1'
        if pldamodel is None:
            pldadataset = 'ami_sdm_train_gnd'
            pldamodel= 'lists/{0}/plda_{0}.pkl'.format(pldadataset)
        pldamodel = pkl.load(open(pldamodel,'rb'))
        # dataset = 'dihard_dev_2020_track1_fbank_jhu'

        if xvecpath is None:
            xvecpath = 'tools_diar/xvectors_0.75s_npy/{}/'.format(dataset)
        if X is None:
            if batch is None:
                X = np.load('{}/{}.npy'.format(xvecpath,filename))
            else:
                X = []
                for f in filename:
                    X.append(np.load('{}/{}.npy'.format(xvecpath,f)))
                X = np.concatenate(X,axis=0)
        features = torch.FloatTensor(X).to(device)
       
       
        xvecD = X.shape[1]
        pca_dim = 30
        inpdata = features[np.newaxis]
        net_init = weight_initialization(pldamodel,dimension=xvecD,pca_dimension=pca_dim,device=device)
        model_init = net_init.to(device)
        affinity_init,_,_,_ = model_init.compute_plda_affinity_matrix(pldamodel,inpdata) # original filewise PCA transform
        output_model = affinity_init.detach().cpu().numpy()[0]
        # bp()
        if clustering == 'ahc':
            A = output_model*scale
            B = expit(A).copy()
        else:
            A = expit(output_model*scale)
            B = A.copy()
        
        if batch is None:
            with open(filepath,'wb') as fb:
                pkl.dump(A,fb)

    # directly feeding adjacency matrix
    if clustering == 'ahc':
        B = expit(A).copy()
    else:
        B = A.copy()

    if isadj is None:
        adj=None
    else:
        B[B<=th] =0
        G = nx.from_numpy_matrix(B)
        
        adj = nx.adjacency_matrix(G)
        if batch is not None and set == 'train':
                with open(filepath,'wb') as fb:
                    pkl.dump(adj,fb)
    return adj, A

def load_data_simu_plda_libvox(dataset,filename,n_clusters,device='cpu',threshold=None,set='val',scale=1,batch=None,th=0.5,clustering='spectral'):
    # load the data: x, tx, allx, graph
    # scale = 1
    
    if scale> 1:
        scale_str =f'_scale{scale}'
    else:
        scale_str=''
    if batch is None:
        if clustering == 'ahc':
            strclustering = '_ahc'
        else:
            strclustering = ''
        filepath = f'exp/{dataset}/plda_libvox_adj_A_0.75s{scale_str}{strclustering}/{filename}.pkl'
        cmd = f'mkdir -p exp/{dataset}/plda_libvox_adj_A_0.75s{scale_str}{strclustering}/'
        os.system(cmd)
    
    if os.path.isfile(filepath):
            with open(filepath,'rb') as fb:
                A = pkl.load(fb)
            
    else:
        device = 'cpu'
        kaldi_recipe_path='/data1/prachis/Dihard_2020/Dihard_2020_track1'
        pldadataset = 'lib_vox_tr_all_gnd_0.75s'
        pldamodel= 'lists/{0}/plda_{0}.pkl'.format(pldadataset)
        pldamodel = pkl.load(open(pldamodel,'rb'))
        # dataset = 'dihard_dev_2020_track1_fbank_jhu'

        xvecpath = 'tools_diar/xvectors_0.75s_npy/{}/'.format(dataset)
        if batch is None:
            X = np.load('{}/{}.npy'.format(xvecpath,filename))
        else:
            X = []
            for f in filename:
                X.append(np.load('{}/{}.npy'.format(xvecpath,f)))
            X = np.concatenate(X,axis=0)
        features = torch.FloatTensor(X).to(device)
       
        xvecD = X.shape[1]
        pca_dim = 30
        inpdata = features[np.newaxis]
        net_init = weight_initialization(pldamodel,dimension=xvecD,pca_dimension=pca_dim,device=device)
        model_init = net_init.to(device)
        affinity_init,_,_ = model_init.compute_plda_affinity_matrix(pldamodel,inpdata) # original filewise PCA transform
        output_model = affinity_init.detach().cpu().numpy()[0]
        # bp()
        if clustering == 'ahc':
            A = output_model*scale
        else:
            A = expit(output_model*scale)
        
        if batch is None:
            with open(filepath,'wb') as fb:
                pkl.dump(A,fb)

    # directly feeding adjacency matrix
    if clustering == 'ahc':
        B = expit(A).copy()
    else:
        B = A.copy()
    B[B<=th] =0
    G = nx.from_numpy_matrix(B)
   
    adj = nx.adjacency_matrix(G)
    if batch is not None and set == 'train':
            with open(filepath,'wb') as fb:
                pkl.dump(adj,fb)
    return adj, A

def load_data_simu_plda_ami_withparam(dataset,filename,n_clusters,device='cpu',threshold=None,set='val',scale=1,batch=None,th=0.5):
    # load the data: x, tx, allx, graph
    # scale = 1
    
    if scale> 1:
        scale_str =f'_scale{scale}'
    else:
        scale_str=''
    if batch is None:
        filepath = f'exp/{dataset}/plda_ami_adj_A_0.75s{scale_str}/{filename}.pkl'
        cmd = f'mkdir -p exp/{dataset}/plda_ami_adj_A_0.75s{scale_str}/'
        os.system(cmd)
    else:
        filepath = f'exp/{dataset}/plda_ami_adj_A_0.75s{scale_str}_batch/{filename[0]}.pkl'
        cmd = f'mkdir -p exp/{dataset}/plda_ami_adj_A_0.75s{scale_str}_batch/'
        os.system(cmd)
    device = 'cpu'
    kaldi_recipe_path='/data1/prachis/Dihard_2020/Dihard_2020_track1'
    pldadataset = 'ami_sdm_train_gnd'
    pldamodel= 'lists/{0}/plda_{0}.pkl'.format(pldadataset)
    pldamodel = pkl.load(open(pldamodel,'rb'))
    if 0:#os.path.isfile(filepath):
            with open(filepath,'rb') as fb:
                A = pkl.load(fb)
            if batch is not None and set == 'train':
                return A,None
    else:
        
        # dataset = 'dihard_dev_2020_track1_fbank_jhu'

        xvecpath = 'tools_diar/xvectors_0.75s_npy/{}/'.format(dataset)
        if batch is None:
            X = np.load('{}/{}.npy'.format(xvecpath,filename))
        else:
            X = []
            for f in filename:
                X.append(np.load('{}/{}.npy'.format(xvecpath,f)))
            X = np.concatenate(X,axis=0)
        features = torch.FloatTensor(X).to(device)
       
        xvecD = X.shape[1]
        pca_dim = 30
        inpdata = features[np.newaxis]
        net_init = weight_initialization(pldamodel,dimension=xvecD,pca_dimension=pca_dim,device=device)
        model_init = net_init.to(device)
        affinity_init,_,X1 = model_init.compute_plda_affinity_matrix(pldamodel,inpdata) # original filewise PCA transform
        output_model = affinity_init.detach().cpu().numpy()[0]
        # bp()
        A = expit(output_model*scale)
        if batch is None:
            with open(filepath,'wb') as fb:
                pkl.dump(A,fb)

    # directly feeding adjacency matrix
    B = A.copy()
    B[B<=th] =0
    G = nx.from_numpy_matrix(B)
   
    adj = nx.adjacency_matrix(G)
    if batch is not None and set == 'train':
            with open(filepath,'wb') as fb:
                pkl.dump(adj,fb)
    # bp()
    return adj, A, pldamodel['mean_vec']

def load_data_simu_plda_ami_featspreprocess(dataset,filename,n_clusters,device='cpu',threshold=None,set='val',scale=1,batch=None,th=0.5):
    # load the data: x, tx, allx, graph
    # scale = 1
    
    if scale> 1:
        scale_str =f'_scale{scale}'
    else:
        scale_str=''
    if batch is None:
        filepath = f'exp/{dataset}/plda_ami_adj_A_0.75s{scale_str}/{filename}.pkl'
        cmd = f'mkdir -p exp/{dataset}/plda_ami_adj_A_0.75s{scale_str}/'
        os.system(cmd)
    else:
        filepath = f'exp/{dataset}/plda_ami_adj_A_0.75s{scale_str}_batch/{filename[0]}.pkl'
        cmd = f'mkdir -p exp/{dataset}/plda_ami_adj_A_0.75s{scale_str}_batch/'
        os.system(cmd)
    device = 'cpu'
    kaldi_recipe_path='/data1/prachis/Dihard_2020/Dihard_2020_track1'
    pldadataset = 'ami_sdm_train_gnd'
    pldamodel= 'lists/{0}/plda_{0}.pkl'.format(pldadataset)
    pldamodel = pkl.load(open(pldamodel,'rb'))
    if 0:#os.path.isfile(filepath):
            with open(filepath,'rb') as fb:
                A = pkl.load(fb)
            if batch is not None and set == 'train':
                return A,None
    else:
        
        # dataset = 'dihard_dev_2020_track1_fbank_jhu'

        xvecpath = 'tools_diar/xvectors_0.75s_npy/{}/'.format(dataset)
        if batch is None:
            X = np.load('{}/{}.npy'.format(xvecpath,filename))
        else:
            X = []
            for f in filename:
                X.append(np.load('{}/{}.npy'.format(xvecpath,f)))
            X = np.concatenate(X,axis=0)
        features = torch.FloatTensor(X).to(device)
       
        xvecD = X.shape[1]
        pca_dim = 30
        inpdata = features[np.newaxis]
        net_init = weight_initialization(pldamodel,dimension=xvecD,pca_dimension=pca_dim,device=device)
        model_init = net_init.to(device)
        affinity_init,_,X1 = model_init.compute_plda_affinity_matrix(pldamodel,inpdata) # original filewise PCA transform
        output_model = affinity_init.detach().cpu().numpy()[0]
        # bp()
        A = expit(output_model*scale)
        if batch is None:
            with open(filepath,'wb') as fb:
                pkl.dump(A,fb)

    # directly feeding adjacency matrix
    B = A.copy()
    B[B<=th] =0
    G = nx.from_numpy_matrix(B)
   
    adj = nx.adjacency_matrix(G)
    if batch is not None and set == 'train':
            with open(filepath,'wb') as fb:
                pkl.dump(adj,fb)
    # bp()
    feats = X1[0]
    return adj, A, feats



def load_data_simu_plda_knn(dataset,filename,n_clusters,device='cpu',threshold=None,set='val',scale=1,batch=None,myK=50):
    # load the data: x, tx, allx, graph
    # scale = 1
    
    if scale> 1:
        scale_str =f'_scale{scale}'
    else:
        scale_str=''
    if batch is None:
        filepath = f'exp/{dataset}/plda_adj_A_0.75s{scale_str}/{filename}.pkl'
        cmd = f'mkdir -p exp/{dataset}/plda_adj_A_0.75s{scale_str}/'
        os.system(cmd)
    else:
        filepath = f'exp/{dataset}/plda_adj_A_0.75s{scale_str}_batch/{filename[0]}.pkl'
        cmd = f'mkdir -p exp/{dataset}/plda_adj_A_0.75s{scale_str}_batch/'
        os.system(cmd)
    if 0:#os.path.isfile(filepath):
            with open(filepath,'rb') as fb:
                A = pkl.load(fb)
            if batch is not None and set == 'train':
                return A,None
    else:
        device = 'cpu'
        kaldi_recipe_path='/data1/prachis/Dihard_2020/Dihard_2020_track1'
        pldadataset = 'dihard_dev_2020_track1_fbank_jhu_wide'
        pldamodel= '/data1/prachis/Dihard_2020/SSC/lists/{0}/plda_{0}.pkl'.format(pldadataset)
        pldamodel = pkl.load(open(pldamodel,'rb'))
        # dataset = 'dihard_dev_2020_track1_fbank_jhu'

        xvecpath = 'tools_diar/xvectors_0.75s_npy/{}/'.format(dataset)
        if batch is None:
            X = np.load('{}/{}.npy'.format(xvecpath,filename))
        else:
            X = []
            for f in filename:
                X.append(np.load('{}/{}.npy'.format(xvecpath,f)))
            X = np.concatenate(X,axis=0)
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
        if batch is None:
            with open(filepath,'wb') as fb:
                pkl.dump(A,fb)


    # KNN
    B = -A.copy()
    K = myK
    N = X.shape[0]
    K = min(K,N-1)
    B[np.diag_indices(N)] = -np.inf
    sortedDist = np.sort(B,axis=1)
    NNIndex = np.argsort(B,axis=1)
    NNIndex = NNIndex[:,:K+1]
    ND = -sortedDist[:, 1:K+1].copy()
    NI = NNIndex[:, 1:K+1].copy()
    XI = np.dot(np.arange(N).reshape(-1,1),np.ones((1,K),dtype=int))
    graphW = coo_matrix((ND.reshape(-1), (XI.reshape(-1),  NI.reshape(-1))), shape=(N, N)).toarray()
    graphW[np.diag_indices(N)]=1
    
    adj = graphW
    adj = sparse.csr_matrix(adj)
    # adj = nx.adjacency_matrix(G)
    if batch is not None and set == 'train':
            with open(filepath,'wb') as fb:
                pkl.dump(adj,fb)
    return adj, A


def load_data_dihard_plda(dataset,filename,n_clusters,device='cpu',threshold=None,set='val',scale=1):
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
        pldadataset = 'dihard_dev_2020_track1_fbank_jhu'
        kaldi_recipe_path='/data1/prachis/Dihard_2020/Dihard_2020_track1'
        pldamodel= '/data1/prachis/Dihard_2020/SSC/lists/{0}/plda_{0}.pkl'.format(pldadataset)
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
    # KNN
    # B = -A
    # K = 30
    # N = X.shape[0]
    # K = min(K,N-1)
    # B[np.diag_indices(N)] = -np.inf
    # sortedDist = np.sort(B,axis=1)
    # NNIndex = np.argsort(B,axis=1)
    # NNIndex = NNIndex[:,:K+1]
    # ND = -sortedDist[:, 1:K+1].copy()
    # NI = NNIndex[:, 1:K+1].copy()
    # XI = np.dot(np.arange(N).reshape(-1,1),np.ones((1,K),dtype=int))
    # graphW = coo_matrix((ND.reshape(-1), (XI.reshape(-1),  NI.reshape(-1))), shape=(N, N)).toarray()
    # graphW[np.diag_indices(N)]=1
    
    # G = nx.from_numpy_matrix(graphW)

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
    adj = nx.adjacency_matrix(G)
    
    return adj, A

def load_data_dihard_plda_spectral(dataset,filename,n_clusters,device='cpu',scale=1):
    # load the data: x, tx, allx, graph
    # scale = 1
    if scale> 1:
        scale_str =f'_scale{scale}'
    else:
        scale_str=''
    kaldi_recipe_path='/data1/prachis/Dihard_2020/Dihard_2020_track1'
    xvecpath = '{}/xvectors_npy/{}/'.format(kaldi_recipe_path,dataset)
    X = np.load('{}/{}.npy'.format(xvecpath,filename))
    features = torch.FloatTensor(X).to(device)
    filepath = f'exp/{dataset}/plda_adj_A{scale_str}/{filename}.pkl'
    cmd = f'mkdir -p exp/{dataset}/plda_adj_A{scale_str}/'
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
        A = expit(output_model*scale)
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

def preprocess_graph(adj,selfloop=1):
    adj = sp.coo_matrix(adj)
    if selfloop:
        adj_ = adj + sp.eye(adj.shape[0])
    else:
        adj_ = adj

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

def get_roc_score_adj(S, adj_orig,mask_indices,baseline=None,flag=None):
    # bp()
    mask_indices = mask_indices.data.cpu().numpy()
    if flag is not None:
        S = S[mask_indices[:,0]!=mask_indices[:,1]]
        mask_indices = mask_indices[mask_indices[:,0]!=mask_indices[:,1]]
       
    adj_rec_triu = expit(S).reshape(-1,)
    # N= adj_orig.shape[0]
    # tr_indices = np.triu_indices(N,k=1)
    # adj_triu = adj_orig.toarray()[tr_indices]
    # bp()
    # Predict on test set of edges
    
    # adj_rec_triu = adj_rec[tr_indices]
    adj_triu = adj_orig.toarray()[mask_indices[:,0],mask_indices[:,1]]
    
    preds = adj_rec_triu[np.where(adj_triu==1)[0]]
    preds_neg = adj_rec_triu[np.where(adj_triu==0)[0]]
   
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

def get_roc_score_modified(emb, adj_orig,baseline=None,mask = None):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    if baseline is None:
        N = emb.shape[0]
        adj_rec =expit( np.dot(emb, emb.T))

        # adj_rec =np.dot(emb, emb.T)
        # if np.min(adj_rec)<0:
        #     adj_rec = (adj_rec -np.min(adj_rec))/(np.max(adj_rec)-np.min(adj_rec))
    else:
        if baseline ==1:
            adj_rec = emb.toarray()
        else:
            adj_rec = emb
        N = adj_rec.shape[0]
    if mask is not None:
        adj_rec = adj_rec*mask
        adj_orig = adj_orig.toarray()*mask
    else:
        adj_orig = adj_orig.toarray()
    
    tr_indices = np.triu_indices(N,k=1)
    adj_triu = adj_orig[tr_indices]
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
    # plt.figure()
    # plt.scatter(results['train_loss'], results['baseline_DER'])
    # plt.show()
    # plt.savefig('{}/{}_scatter.png'.format(path,filename))
    
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
