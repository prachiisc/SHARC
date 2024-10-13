#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 15:18:06 2020

@author: prachi singh
@email: prachisingh@iisc.ac.in
"""
# import os
# import argparse
# import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
from pdb import set_trace as bp
import pickle as pkl

# value given in https://kaldi-asr.org/doc/kaldi-math_8h_source.html
M_LOG_2PI = 1.8378770664093454835606594728112


class Deep_Ahc_model(nn.Module):
    def __init__(self,plda_org,plda,dimension=128,red_dimension=10,device=torch.device('cuda')):
        super(Deep_Ahc_model, self).__init__()        
        
       
        self.dimension = dimension
        self.red_dimension = red_dimension
        # Global PCA 
        self.globalpca_out = nn.Linear(dimension,dimension,bias=False)
        # Filewise PCA 
        self.filepca_out = nn.Linear(dimension,red_dimension,bias=False)
       
        
        self.device = device
        self.plda_init = plda.copy()
        self.plda_org = plda_org.copy()
        
        # Diagonalizing transform
        self.Dnew = Parameter(torch.from_numpy(plda['diagonalizing_transform'].T[np.newaxis]).float().to(device))
        # self.Dnew = Parameter(torch.randn(red_dimension,red_dimension))
        # Diagonal covariance 
        # self.psi =Parameter(torch.randn(red_dimension))
        self.psi = Parameter(torch.from_numpy(plda['Psi_across_covar_diag']).float().to(device))
        
        self.mean_vec = torch.from_numpy(plda['mean_vec']).float().to(device)
        self.transform_mat = torch.from_numpy(plda['transform_mat']).float().to(device)
        self.filewise_transform = torch.rand((red_dimension,dimension))
        self.target_energy = 0.3

    def init_weights(self,filewise_PCA):

        self.globalpca_out.weight.data = self.transform_mat
        # self.globalpca_out.bias.data = torch.zeros((self.dimension,1))
        self.filewise_transform = filewise_PCA        
        # self.red_dimension = filewise_PCA.shape[0] 
        self.filepca_out.weight.data = self.filewise_transform
        # self.filepca_out.bias.data = torch.zeros((self.red_dimension,1))

    
    def preprocessing(self,X):
        """
        Perform mean_subtraction using mean.vec-> apply transform.mat -> input length norm

        Parameters
        ----------
        X : Xvectors 1 X N X D

        Returns
        -------
        transformed xvectors 1 X N X D

        """
        
        # xvecs = torch.transpose(X,1,2) # 1 X DX N
        xvecs = X # 1X N X D
        dim = xvecs.shape[2]
        # preprocessing
        # mean subtraction
        xvecs = xvecs - self.mean_vec[np.newaxis,np.newaxis,:]
        # PCA transform        
        # xvecs = torch.bmm(self.transform_mat[np.newaxis],xvecs)
        # bp()
        xvecs = self.globalpca_out(xvecs)
        l2_norm = torch.norm(xvecs, dim=2, keepdim=True)
        l2_norm = l2_norm/math.sqrt(dim)
        # l2_norm = np.linalg.norm(xvecs,axis=1,keepdims=True)
        xvecsnew = xvecs/l2_norm
        
        return xvecsnew #torch.transpose(xvecsnew,1,2)

    def compute_PCA_transform(self,X):
        """
        Computes filewise PCA
        given in https://kaldi-asr.org/doc/ivector-plda-scoring-dense_8cc_source.html
        Apply transform on mean shifted xvectors

        Parameters
        ----------
        Xvectors

        Returns
        ----------
        new xvectors and transform

        """
        # xvec = X[0].cpu().detach().numpy() #N X D
        xvec = X[0]
        num_rows = xvec.shape[0]
        num_cols = xvec.shape[1]
        mean = torch.mean(xvec,0,keepdim=True)
        # xvec = xvec - mean
        S = torch.transpose(xvec,0,1) @ xvec
        # S = torch.matmul(xvec.T,xvec)
        # S = np.cov(xvec.T)
        S = S/num_rows
        
        S = S - torch.transpose(mean,0,1) @ mean
        # ev_s , eig_s = np.linalg.eig(S)
        # ev_s, eig_s , _ = np.linalg.svd(S,full_matrices=True)
        
        
        try:
            ev_s,eig_s,_ = torch.svd(S)
        except:
            ValueError(S)
        total_energy = torch.sum(eig_s)
        energy =0.0
        dim=1
        while energy/total_energy <= self.target_energy:
            energy += eig_s[dim-1]
            dim +=1
        transform = ev_s[:,:dim]
        # transform[:,1] = -1 *transform[:,1]
        transxvec = xvec @ transform
        newX = transxvec[np.newaxis]
        # newX = torch.from_numpy(transxvec[np.newaxis]).float().to(self.device)
        # transform = torch.from_numpy(ev_s[:,:dim])

        return newX, torch.transpose(transform,0,1)

    
    def compute_affinity_matrix(self,X,attention=False):
        """Compute the affinity matrix from data.

        Note that the range of affinity is [0,1].

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            affinity: numpy array of shape (n_samples, n_samples)
        """
        X = self.preprocessing(X) #output -N X D
        X = self.filepca_out(X)
        
        # X = X - torch.mean(X,dim=1,keepdim=True)        
        
        if type(attention)!=bool:            
            X = torch.bmm(attention,X)
            
        # Normalize the data.
        
        l2_norms = torch.norm(X, dim=2,keepdim=True)
        X_normalized = X / l2_norms
        # Compute cosine similarities. Range is [-1,1].

        cosine_similarities = torch.bmm(X_normalized, torch.transpose(X_normalized,1,2))
       
        affinity = cosine_similarities
        
        return affinity,X_normalized
    
    def transformXvectors(self,X):
        """
        Apply plda mean and diagonalizing transform to xvectors for scoring

        Parameters
        ----------
        X : TYPE
           Xvectors 1 X N X D

        Returns
        -------
        X_new : TYPE
            transformed x-vectors

        """
        
        mean = torch.from_numpy(self.plda_org['plda_mean']).float().to(self.device)
        mean = torch.transpose(mean.reshape(-1,1)[np.newaxis],1,2)
        mean_new = self.filepca_out(mean)
       
        # offset = torch.from_numpy(self.plda['offset']).float().to(self.device)
        # offset = torch.transpose(offset[np.newaxis],1,2)
        # mean = mean.type(torch.float64)
        # D = torch.from_numpy(self.plda['diagonalizing_transform']).float().to(self.device)
        # Dnew = torch.transpose(D[np.newaxis],1,2)
        # X = X - mean
        
        offset = -torch.bmm(mean_new,self.Dnew)
        X_new = torch.bmm(X,self.Dnew) 
        X_new = X_new + offset
        # Get normalizing factor
        # Defaults : normalize_length(true), simple_length_norm(false)
        X_new_sq = X_new**2
        # psi = torch.from_numpy(self.plda['Psi_across_covar_diag']).float().to(self.device)
        inv_covar = (1.0/(1.0+self.psi)).reshape(-1,1)
        dot_prod = torch.matmul(X_new_sq[0],inv_covar) # N X 1
        Dim = self.Dnew.shape[2]
        normfactor = torch.sqrt(Dim/dot_prod)
        X_new = X_new*normfactor[np.newaxis]
        return X_new
        
    def compute_plda_score(self,X):
        """
        Computes plda affinity matrix using Loglikelihood function

        Parameters
        ----------
        X : TYPE
            X-vectors 1 X N X D

        Returns
        -------
        Affinity matrix TYPE
            1 X N X N 

        """
        
        psi = self.psi
        mean = psi/(psi+1.0)
        mean = mean.reshape(1,-1)*X[0] # N X D , X[0]- Train xvectors
        
        # given class computation
        variance_given = 1.0 + psi/(psi+1.0)
        logdet_given = torch.sum(torch.log(variance_given))
        variance_given = 1.0/variance_given
        
        # without class computation
        variance_without =1.0 + psi
        logdet_without = torch.sum(torch.log(variance_without))
        variance_without = 1.0/variance_without
        
        sqdiff = X[0] #---- Test x-vectors
        nframe = X.shape[1]
        dim = X.shape[2]
        loglike_given_class = torch.zeros((nframe,nframe)).float().to(self.device)
        for i in range(nframe):
            sqdiff_given = sqdiff - mean[i]
            sqdiff_given  =  sqdiff_given**2
            
            loglike_given_class[:,i] = -0.5 * (logdet_given + M_LOG_2PI * dim + \
                                   torch.matmul(sqdiff_given, variance_given))
        sqdiff_without = sqdiff**2
        loglike_without_class = -0.5 * (logdet_without + M_LOG_2PI * dim + \
                                     torch.matmul(sqdiff_without, variance_without));
        loglike_without_class = loglike_without_class.reshape(-1,1) 
        # loglike_given_class - N X N, loglike_without_class - N X1
        loglike_ratio = loglike_given_class - loglike_without_class  # N X N
        
        return loglike_ratio[np.newaxis]
    
    def compute_plda_affinity_matrix(self,X,S=None):
        """Compute the plda_affinity matrix from data.
        plda functions given in https://kaldi-asr.org/doc/classkaldi_1_1Plda.html#afda9c0178f439b40698914f237adef81
        Args:
            X: numpy array of shape (1,n_samples, n_features)

        Returns:
            affinity: numpy array of shape (n_samples, n_samples)
        """
        nframe = X.shape[1]
        
        X = self.preprocessing(X) #output - 1 X N X D
        if S is not None:
            X = torch.bmm(S,X)
        X = self.filepca_out(X)
       
        X = self.transformXvectors(X)
        affinity = self.compute_plda_score(X)
        # affinity = affinity/torch.max(torch.abs(affinity))
        # affinity = 1/(1+torch.exp(-affinity))
        #Shreyas affinity
        # affinity2 = self.shreyas_affinity_matrix(X)
        
        return affinity
        
    def forward(self, X,S=None):        
        #output = x.copy()
        if S is not None:
            X = torch.bmm(S,X)
            
        affinity = self.compute_plda_affinity_matrix(X)
        return affinity

class weight_initialization(nn.Module):
    def __init__(self,plda,dimension=128,pca_dimension=10,device='cpu',**kwargs):
        super(weight_initialization, self).__init__()

        self.device = device
        self.plda = plda
        self.mean_vec = torch.from_numpy(plda['mean_vec']).float().to(device)
        self.transform_mat = torch.from_numpy(plda['transform_mat']).float().to(device)
        self.target_energy = 0.3
        self.pca_dimension = pca_dimension
        self.dimension =dimension
        if 'ldatransform' in kwargs:
            self.ldatransform = kwargs['ldatransform']
        else:
            self.ldatransform = None
        self.device = device

    def preprocessing(self,X,lnorm=1):
        """
        Perform mean_subtraction using mean.vec-> apply transform.mat -> input length norm

        Parameters
        ----------
        X : Xvectors N X d

        Returns
        -------
        transformed xvectors N X D

        """
        
        xvecs = torch.transpose(X,1,2) # DX N
        dim = xvecs.shape[1]
        # preprocessing
        # mean subtraction
        xvecs = xvecs - self.mean_vec[np.newaxis,:,np.newaxis]
        # PCA transform
        xvecs = torch.bmm(self.transform_mat[np.newaxis],xvecs)
        l2_norm = torch.norm(xvecs, dim=1, keepdim=True)
        l2_norm = l2_norm/math.sqrt(dim)
        # l2_norm = np.linalg.norm(xvecs,axis=1,keepdims=True)
        xvecsnew = xvecs/l2_norm
        if lnorm:
            return torch.transpose(xvecsnew,1,2)
        else:
            return torch.transpose(xvecsnew,1,2), torch.transpose(xvecs,1,2)

    
    def preprocessing_lda(self,X):
        """
        Perform mean_subtraction using mean.vec-> apply transform.mat -> input length norm

        Parameters
        ----------
        X : Xvectors N X d

        Returns
        -------
        transformed xvectors N X D

        """
        self.ldatransform = pkl.load(open(self.ldatransform,'rb'))
        ldatransform_mat = torch.from_numpy(self.ldatransform['transform_mat']).float().to(self.device)
        offset = ldatransform_mat[:,-1].reshape(-1,1)
        ldatransform_mat = ldatransform_mat[:,:-1]
        
        xvecs = torch.transpose(X,1,2) # DX N
        dim = xvecs.shape[1]
        # preprocessing
        # mean subtraction
        xvecs = xvecs - self.mean_vec[np.newaxis,:,np.newaxis]
        # LDA transform
        xvecs = torch.bmm(ldatransform_mat[np.newaxis],xvecs)
        xvecs = xvecs + offset[np.newaxis]
        return torch.transpose(xvecs,1,2)
      


    def compute_PCA_transform(self,X,dim=None):
        """
        Computes filewise PCA
        given in https://kaldi-asr.org/doc/ivector-plda-scoring-dense_8cc_source.html
        Apply transform on mean shifted xvectors

        Parameters
        ----------
        Xvectors

        Returns
        ----------
        new xvectors and transform

        """
        # xvec = X[0].cpu().detach().numpy() #N X D
        xvec = X[0]
        num_rows = xvec.shape[0]
        num_cols = xvec.shape[1]
        mean = torch.mean(xvec,0,keepdim=True)
        # xvec = xvec - mean
        S = torch.transpose(xvec,0,1) @ xvec
        # S = torch.matmul(xvec.T,xvec)
        # S = np.cov(xvec.T)
        S = S/num_rows
        
        S = S - torch.transpose(mean,0,1) @ mean
        # ev_s , eig_s = np.linalg.eig(S)
        # ev_s, eig_s , _ = np.linalg.svd(S,full_matrices=True
        
        if dim is None:
            dim = self.pca_dimension
       
        try:
            ev_s,eig_s,_ = torch.svd(S)
        except:
            print('SVD_error')
        
        # below code is for selecting dimension using target energy 
        # total_energy = torch.sum(eig_s)
        
        # energy =0.0
        # dim=1
        # while energy/total_energy <= self.target_energy:
        #     energy += eig_s[dim-1]
        #     dim +=1
        # print('pca_dim computed: ',dim)
        transform = ev_s[:,:dim]
       
        transxvec = xvec @ transform
        newX = transxvec[np.newaxis]
        
        return newX, torch.transpose(transform,0,1)

    def compute_PCA_transform_withtarget(self,X):
        """
        Computes filewise PCA
        given in https://kaldi-asr.org/doc/ivector-plda-scoring-dense_8cc_source.html
        Apply transform on mean shifted xvectors

        Parameters
        ----------
        Xvectors

        Returns
        ----------
        new xvectors and transform

        """
        # xvec = X[0].cpu().detach().numpy() #N X D
        xvec = X[0]
        num_rows = xvec.shape[0]
        num_cols = xvec.shape[1]
        mean = torch.mean(xvec,0,keepdim=True)
      
        S = torch.transpose(xvec,0,1) @ xvec
        S = S/num_rows
        
        S = S - torch.transpose(mean,0,1) @ mean
        # ev_s , eig_s = np.linalg.eig(S)
        # ev_s, eig_s , _ = np.linalg.svd(S,full_matrices=True)
        
        try:
            ev_s,eig_s,_ = torch.svd(S)
        except:
            print('SVD_error')
        # dim = self.pca_dimension
        total_energy = torch.sum(eig_s)
        energy =0.0
        dim=1
        while energy/total_energy <= self.target_energy:
            energy += eig_s[dim-1]
            dim +=1
        print('pca_dim computed: ',dim)
        transform = ev_s[:,:dim]
        # transform[:,1] = -1 *transform[:,1]
        transxvec = xvec @ transform
        newX = transxvec[np.newaxis]
        # newX = torch.from_numpy(transxvec[np.newaxis]).float().to(self.device)
        # transform = torch.from_numpy(ev_s[:,:dim])
        return newX, torch.transpose(transform,0,1)
    
    def applytransform_plda(self,transform_in):
        """
        Apply PCA filewise transform on PLDA parameters
        details are given in : https://kaldi-asr.org/doc/classkaldi_1_1Plda.html#afda9c0178f439b40698914f237adef81

        Parameters
        ----------
        transform_in : numpy  D X dim
           PCA filewise transform

        """
        transform_in = transform_in.cpu().detach().numpy()
        mean_plda = self.plda['plda_mean']
        #transfomed mean vector
        new_mean = transform_in @ mean_plda[:,np.newaxis]
        D = self.plda['diagonalizing_transform']
        psi = self.plda['Psi_across_covar_diag']
        D_inv = np.linalg.inv(D)
        # within class and between class covarinace
        phi_b=  (D_inv * psi.reshape(1,-1)) @ D_inv.T
        phi_w = D_inv @ D_inv.T
        # transformed with class and between class covariance
        new_phi_b = transform_in @ phi_b @ transform_in.T
        new_phi_w = transform_in @ phi_w @ transform_in.T
        ev_w, eig_w,_ =np.linalg.svd(new_phi_w)
        eig_w_inv = 1/np.sqrt(eig_w)
        Dnew = eig_w_inv.reshape(-1,1)*ev_w.T
        new_phi_b_proj = Dnew @ new_phi_b @ Dnew.T
        ev_b, eig_b,_ = np.linalg.svd(new_phi_b_proj)
        psi_new = eig_b

        Dnew = ev_b.T @ Dnew
        self.plda['plda_mean'] = new_mean
        self.plda['diagonalizing_transform'] = Dnew
        self.plda['Psi_across_covar_diag'] = psi_new
        self.plda['offset'] = -Dnew @ new_mean.reshape(-1,1)
        # ac = res['Psi_across_covar_diag']
        tot = 1 + psi_new
        self.plda['diagP'] = psi_new/(tot*(tot-psi_new*psi_new/tot))
        self.plda['diagQ'] = (1/tot) - 1/(tot - psi_new*psi_new/tot)
        
    def transformXvectors(self,X,dim=None):
        """
        Apply plda mean and diagonalizing transform to xvectors for scoring

        Parameters
        ----------
        X : TYPE
           Xvectors 1 X N X D

        Returns
        -------
        X_new : TYPE
            transformed x-vectors

        """
        
        # mean = torch.from_numpy(self.plda['plda_mean']).float().to(self.device)
        # mean = torch.transpose(mean[np.newaxis],1,2)

        # mean = mean.type(torch.float64)
        D = torch.from_numpy(self.plda['diagonalizing_transform']).float().to(self.device)
        Dnew = torch.transpose(D[np.newaxis],1,2)
        # X = X - mean
        
        if dim is None:
            offset = torch.from_numpy(self.plda['offset']).float().to(self.device)
        else:
            # bp()
            Dnew = Dnew[:,:,:dim]
            mean = torch.from_numpy(self.plda['plda_mean']).float().to(self.device)
            # mean = torch.transpose(mean[np.newaxis],1,2)
            offset = -D @ mean.reshape(-1,1)
            offset = offset[:dim]
            D = D[:dim]
             
        offset = torch.transpose(offset[np.newaxis],1,2)
        X_new = torch.bmm(X,Dnew) 
        X_new = X_new + offset
        # Get normalizing factor
        # Defaults : normalize_length(true), simple_length_norm(false)
        X_new_sq = X_new**2
        psi = torch.from_numpy(self.plda['Psi_across_covar_diag']).float().to(self.device)
        if dim is not None:
            psi = psi[:dim]
        inv_covar = (1.0/(1.0+psi)).reshape(-1,1)
        dot_prod = torch.matmul(X_new_sq[0],inv_covar) # N X 1
        Dim = D.shape[0]
        normfactor = torch.sqrt(Dim/dot_prod)
        X_new = X_new*normfactor[np.newaxis]
        
        return X_new
    
    def compute_plda_score(self,X):
        """
        Computes plda affinity matrix using Loglikelihood function

        Parameters
        ----------
        X : TYPE
            X-vectors 1 X N X D

        Returns
        -------
        Affinity matrix TYPE
            1 X N X N 

        """
        
        psi = torch.from_numpy(self.plda['Psi_across_covar_diag']).float().to(self.device)
        mean = psi/(psi+1.0)
        mean = mean.reshape(1,-1)*X[0] # N X D , X[0]- Train xvectors
        
        # given class computation
        variance_given = 1.0 + psi/(psi+1.0)
        logdet_given = torch.sum(torch.log(variance_given))
        variance_given = 1.0/variance_given
        
        # without class computation
        variance_without =1.0 + psi
        logdet_without = torch.sum(torch.log(variance_without))
        variance_without = 1.0/variance_without
        
        sqdiff = X[0] #---- Test x-vectors
        nframe = X.shape[1]
        dim = X.shape[2]
        loglike_given_class = torch.zeros((nframe,nframe)).float().to(self.device)
        for i in range(nframe):
            sqdiff_given = sqdiff - mean[i]
            sqdiff_given  =  sqdiff_given**2
            
            loglike_given_class[:,i] = -0.5 * (logdet_given + M_LOG_2PI * dim + \
                                   torch.matmul(sqdiff_given, variance_given))
        sqdiff_without = sqdiff**2
        loglike_without_class = -0.5 * (logdet_without + M_LOG_2PI * dim + \
                                     torch.matmul(sqdiff_without, variance_without));
        loglike_without_class = loglike_without_class.reshape(-1,1) 
        # loglike_given_class - N X N, loglike_without_class - N X1
        loglike_ratio = loglike_given_class - loglike_without_class  # N X N
        
        return loglike_ratio[np.newaxis]
    
    def compute_filewise_PCAtransform_withtarget(self,plda,X):
        nframe = X.shape[1]
        self.plda = plda.copy()
        
        X = self.preprocessing(X) #output -N X D
        
        _, PCA_transform = self.compute_PCA_transform_withtarget(X)
        
        return PCA_transform
    
    def compute_filewise_PCAtransform(self,plda,X):
        nframe = X.shape[1]
        self.plda = plda.copy()
        
        X = self.preprocessing(X) #output -N X D
        
        _, PCA_transform = self.compute_PCA_transform(X)
        
        return PCA_transform
        
    def compute_plda_affinity_matrix(self,plda,X,target=0,**kwargs):
        # final=1,lnorm=1,filepca=1
        """Compute the plda_affinity matrix from data.
        plda functions given in https://kaldi-asr.org/doc/classkaldi_1_1Plda.html#afda9c0178f439b40698914f237adef81
        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            affinity: numpy array of shape (n_samples, n_samples)
        """
        if 'final' in kwargs:
            final = kwargs['final']
        else:
            final = 1
        nframe = X.shape[1]
        self.plda = plda.copy()
       
        if self.ldatransform is None:
            if 'lnorm' in kwargs:
                X = self.preprocessing(X) #output - N X D
                proc_feats = X
            else:
                X, proc_feats = self.preprocessing(X,lnorm=0) #output - N X D
        else:
            proc_feats = self.preprocessing_lda(X)
        # if final==0:
        if target==0:
            X, PCA_transform = self.compute_PCA_transform(X)
        else:
            X, PCA_transform = self.compute_PCA_transform_withtarget(X)
        
        self.applytransform_plda(PCA_transform)
        X = self.transformXvectors(X)
 
        affinity = self.compute_plda_score(X)
        plda_backup = self.plda
        if final==1:
            # with PCA dimension
            proc_feats = X
        elif final==2:
            self.plda = plda
            # LDA -128 dimension
            if 'filepca' in kwargs:
                proc_feats, PCA_transform_2 = self.compute_PCA_transform(proc_feats,dim=self.dimension)
                self.applytransform_plda(PCA_transform_2)
                proc_feats = self.transformXvectors(proc_feats)
            else:
                proc_feats = self.transformXvectors(proc_feats,dim=self.dimension)
        #Shreyas affinity
        # affinity2 = self.shreyas_affinity_matrix(X)
        return affinity, plda_backup, PCA_transform, proc_feats


    def compute_affinity_matrix(self,X, target=0, filepca=1):
        """Compute the affinity matrix from data.

        Note that the range of affinity is [-1,1].

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            affinity: numpy array of shape (n_samples, n_samples)
        """
        # Normalize the data.
        X = self.preprocessing(X) #output -N X D
        if filepca:
            if target==0:
                X, PCA_transform = self.compute_PCA_transform(X)
            else:
                X, PCA_transform = self.compute_PCA_transform_withtarget(X)
        l2_norms = torch.norm(X, dim=2,keepdim=True)
        X_normalized = X / l2_norms
        # Compute cosine similarities. Range is [-1,1].
        proc_feats = X_normalized

        cosine_similarities = torch.bmm(X_normalized, torch.transpose(X_normalized,1,2))
        # Compute the affinity. Range is [0,1].
        # Note that this step is not mentioned in the paper!
        # affinity = cosine_similarities
        #affinity = cosine_similarities
        affinity = (cosine_similarities + 1.0) / 2.0

        return affinity, proc_feats

