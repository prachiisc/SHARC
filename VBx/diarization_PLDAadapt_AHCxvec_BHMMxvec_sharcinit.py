#!/usr/bin/env python

# Copyright 2019 Lukas Burget (burget@fit.vutbr.cz)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Recipe for doing diarization on data from The Second DIHARD Diarization Challenge
# https://coml.lscp.ens.fr/dihard/index.html
# The recipe consists in doing Agglomerative Hierachical Clustering on
# x-vectors in a first step. Then, Variational Bayes HMM over x-vectors
# is applied using the AHC output as initialization.
# 
# The BUT submission for the challenge is presented in
# F. Landini, S. Wang, M. Diez, L. Burget et al.
# BUT System for the Second DIHARD Speech Diarization Challenge, ICASSP 2020
# and a more detailed analysis of this approach is presented in 
# M. Diez, L. Burget, F. Landini, S. Wang, J. \v{C}ernock\'{y}
# Optimizing Bayesian HMM based x-vector clustering for the second DIHARD speech 
# diarization challenge, ICASSP 2020

# A more thorough description and study of the VB-HMM with eigen-voice priors 
# approach for diarization is presented in 
# M. Diez, L. Burget, F. Landini, J. \v{C}ernock\'{y}
# Analysis of Speaker Diarization based on Bayesian HMM with Eigenvoice Priors, 
# IEEE Transactions on Audio, Speech and Language Processing, 2019

# This recipe differs from our submission to the challenge in that
# VB resegmentation and overlapped speech post-processing are not applied.
# These two steps are not presented for producing small improvements
# but adding considerably more complicated processing to the recipe.

import sys
import numpy as np
import itertools
import kaldi_io
from diarization_lib import *
import VB_diarization
import time
import argparse
from scipy.special import softmax
# sys.path.insert(0,'/data1/prachis/SRE_19/Self_supervised_clustering/services/')
# import path_integral_clustering as pic
from pdb import set_trace as bp

# out_rttm_dir  =       sys.argv[1]   # Directory to store output rttm files
# xvec_ark_file =       sys.argv[2]   # Kaldi ark file with x-vectors from one or more input recordings 
#                                     # Attention: all x-vectors from one recording must be in one ark file
# segments_file =       sys.argv[3]   # File with x-vector timing info (see diarization_lib.read_xvector_timing_dict)
# mean_vec_file =       sys.argv[4]   # File with mean vector in Kaldi format for x-vector centering
# tran_mat_file =       sys.argv[5]   # File with linear transformation matrix in Kaldi format for x-vector whitening
# plda_file     =       sys.argv[6]   # File with PLDA model in Kaldi format used for AHC and VB-HMM x-vector clustering
# plda_adapt_file=      sys.argv[7]   # Another PLDA model in Kaldi format which is interpolated with the previous one

# alpha         = float(sys.argv[8])  # Interpolation weight between 0 and 1 for mixing the two PLDA model alpha=0 corresponds to plda_adapt
# threshold     = float(sys.argv[9])  # Threshold (bias) used for AHC
# target_energy = float(sys.argv[10]) # Parameter affecting AHC. (see diarization_lib.kaldi_ivector_plda_scoring_dense)
# init_smoothing= float(sys.argv[11]) # AHC produces hard assignments of x-vetors to speakers. These are "smoothed" to
                                    # soft assignments as the initialization for VB-HMM. This parameter controls the amount
                                    # of smoothing. Not so important, high value (e.g. 10) is OK  => keeping hard assigment
# lda_dim       =   int(sys.argv[12]) # For VB-HMM, x-vectors are reduced to this dimensionality using LDA
# Fa            = float(sys.argv[13]) # Parameter of VB-HMM (see VB_diarization.VB_diarization)
# Fb            = float(sys.argv[14]) # Parameter of VB-HMM (see VB_diarization.VB_diarization)
# LoopP         = float(sys.argv[15]) # Parameter of VB-HMM (see VB_diarization.VB_diarization)
# max_iters     = int(sys.argv[16]) # maximum no. of iterations
# labelspath = sys.argv[17] # path of labels file

# use_PIC       = int(sys.argv[17])
use_PIC = 0
# pca_dim       = int(sys.argv[18])
pca_dim       = None
frm_shift = 0.01 # frame rate of MFCC features
use_VB    = True                # False for using only AHC
###########
# ArgParser
def arguments():
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--alpha', type=float )
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--target_energy', type=float, required=True)
    parser.add_argument('--init_smoothing', type=float, required=True)
    parser.add_argument('--lda_dim', type=int, required=True)
    parser.add_argument('--Fa', type=float, required=True)
    parser.add_argument('--Fb', type=float, required=True)
    parser.add_argument('--LoopP', type=float, required=True)
    parser.add_argument('--max_iters',type=int, required=True)
    parser.add_argument('--use_VB_withoverlap', action='store_true')
    parser.add_argument('--knn_k', type=int, required=True)
    parser.add_argument('--tau', type=float, required=True)
    parser.add_argument('--labelspath', type=str, required=True,help="initialization labels filename")
    parser.add_argument('--overlap_filename', type=str,help="pyannote detector filename")
    
    parser.add_argument('out_rttm_dir', type=str)
    parser.add_argument('xvec_ark_file', type=str)
    parser.add_argument('segments_file', type=str)
    parser.add_argument('mean_vec_file', type=str)
    parser.add_argument('tran_mat_file', type=str)
    parser.add_argument('plda_file', type=str)
    parser.add_argument('plda_adapt_file', type=str)

    
    args = parser.parse_args()
    
    return args

args = arguments()

out_rttm_dir = args.out_rttm_dir
xvec_ark_file = args.xvec_ark_file
segments_file = args.segments_file
mean_vec_file = args.mean_vec_file
tran_mat_file = args.tran_mat_file
plda_file = args.plda_file
plda_adapt_file = args.plda_adapt_file
overlap_filename = args.overlap_filename

def load_rttm(rttmfile,step=100):
    # loading rttm when performing overlap detection and speaker assignment
    rttm = np.genfromtxt(rttmfile, dtype='str')
    rttm_idx = np.asarray([False,False,False,True,True,False,False,True,False, False])
    rttm = rttm[:,rttm_idx] 
    
    # unique_spks,spks = np.unique(rttm[:,-1],return_index=True)
    spks = rttm[:,-1]
    rttm = rttm[:,:2].astype(float)
    rttm[:,1] = rttm[:,0]+rttm[:,1] # convert duration to end point
    rttm = rttm *step # convert to frames
    rttm = rttm.astype(int)
    n_frames = rttm[-1,1]+1
    ref = np.ones(n_frames,dtype=int)*-1
    ref = ref.astype(str)
    for i, line in enumerate(rttm):
        start = line[0]
        end = line[1]
        ref[start:end+1] = spks[i] 
    return ref

def get_overlap_labels(recid, n_frames,step=100,mode=['pyannote'],overlap_filename=None):

    if 'pyannote' in mode:
        if overlap_filename is not None:
            overlap_labels = np.zeros((n_frames,),dtype=int)
            overlap_filename = f'{overlap_filename}/{recid}.txt'
            if os.path.isfile(overlap_filename):
                overlap_idx = np.genfromtxt(overlap_filename,dtype=float)
                if len(overlap_idx) == 0:
                    print(recid,": not predicted overlaps")
                    return overlap_labels
                overlap_idx = overlap_idx.reshape(-1,4) # 4 columns
                
                overlap_idx = overlap_idx[:,[1,2]]
                overlap_idx = overlap_idx*step
            overlap_idx= overlap_idx.astype(int)
           
            for start,end in overlap_idx:
                overlap_labels[start:end+1] = 1
        else:
           print('error loading Overlap labels file')
    else:
        overlap_labels = load_gnd_overlap(recid)
    
    return overlap_labels

def write_rttm_file(rttm_path_org, rttm_path, labels_org, recid,channel=1, step=0.01, precision=2):
    """Write RTTM file.

    Parameters
    ----------
    rttm_path : Path
        Path to output RTTM file.

    labels : ndarray, (n_frames,)
        Array of predicted speaker labels. See ``get_labels`` for explanation.

    channel : int, optional
        Channel (0-indexed) to output segments for.
        (Default: 0)

    step : float, optional
        Duration in seconds between onsets of frames.
        (Default: 0.01)

    precision : int, optional
        Output ``precision`` digits.
        (Default: 2)
    """
    
    # Determine indices of onsets/offsets of speaker turns.

    unique_spks,labels = np.unique(labels_org,return_inverse=True)
    
    is_cp = np.diff(labels, n=1, prepend=-999, append=-999) != 0
    cp_inds = np.nonzero(is_cp)[0]
    bis = cp_inds[:-1]  # Last changepoint is "fake".
    eis = cp_inds[1:] -1
    
    N = len(bis)
    # rttm_idx = np.asarray([False,False,False,True,True,False,False,True,False, False])
    rttm = np.empty((N,10)).astype(str)
    
    rttm[:,0] =  "SPEAKER"
    rttm[:,1] = recid
    rttm[:,2] = channel
    rttm[:,3] = np.round(bis * step,2).astype(str)
    rttm[:,4] = np.round((eis-bis) * step,2).astype(str)
    rttm[:,5] = "<NA>"
    rttm[:,6] = "<NA>"
    rttm[:,7] = labels_org[bis]
    rttm[:,8] = "<NA>"
    rttm[:,9] = "<NA>"

    
    # remove the silence regions, labels_org=-1
    speech_ind = np.where(labels_org[bis]!='-1')[0]
    rttm = rttm[speech_ind]

    rttm_org = np.genfromtxt(rttm_path_org,dtype=str)
    rttm = np.concatenate((rttm_org,rttm),axis=0)
    np.savetxt(rttm_path,rttm, fmt='%s')



glob_tran = kaldi_io.read_mat(tran_mat_file)           # x-vector whitening transformation
glob_mean = kaldi_io.read_vec_flt(mean_vec_file)       # x-vector centering vector
kaldi_plda_train = kaldi_io.read_plda(plda_file)       # out-of-domain PLDA model
kaldi_plda_adapt = kaldi_io.read_plda(plda_adapt_file) # in-domain "adaptation" PLDA model
segs_dict = read_xvector_timing_dict(segments_file)    # segments file with x-vector timing information

plda_train_mu, plda_train_tr, plda_train_psi = kaldi_plda_train
plda_adapt_mu, plda_adapt_tr, plda_adapt_psi = kaldi_plda_adapt

# Interpolate across-class, within-class and means of the two PLDA models with interpolation factor "alpha"
plda_mu = args.alpha*plda_train_mu + (1.0-args.alpha)*plda_adapt_mu
W_train = np.linalg.inv(plda_train_tr.T.dot(plda_train_tr))
B_train = np.linalg.inv((plda_train_tr.T/plda_train_psi).dot(plda_train_tr))
W_adapt = np.linalg.inv(plda_adapt_tr.T.dot(plda_adapt_tr))
B_adapt = np.linalg.inv((plda_adapt_tr.T/plda_adapt_psi).dot(plda_adapt_tr))
W = args.alpha * W_train + (1.0-args.alpha) * W_adapt
B = args.alpha * B_train + (1.0-args.alpha) * B_adapt
acvar, wccn = spl.eigh(B,  W)
plda_psi = acvar[::-1]
plda_tr = wccn.T[::-1]

# Prepare model for VB-HMM clustering (see the comment on "fea" variable below)
ubmWeights = np.array([1.0])
ubmMeans = np.zeros((1,args.lda_dim))
invSigma= np.ones((1,args.lda_dim))
V=np.diag(np.sqrt(plda_psi[:args.lda_dim]))[:,np.newaxis,:]
VtinvSigmaV = VB_diarization.precalculate_VtinvSigmaV(V, invSigma)

# Open ark file with x-vectors and in each iteration of the following for-loop
# read a batch of x-vectors corresponding to one recording
arkit = kaldi_io.read_vec_flt_ark(xvec_ark_file)
# for AMI
if 'ami' in args.labelspath or 'displace' in args.labelspath:
  recit = itertools.groupby(arkit, lambda e: e[0].rsplit('-')[0]) # group xvectors in ark by recording name
else:
  recit = itertools.groupby(arkit, lambda e: e[0].rsplit('_', 1)[0]) # group xvectors in ark by recording name
for file_name, segs in recit:
    print(file_name)

    seg_names, xvecs = zip(*segs)
    x = np.array(xvecs) # matrix of all x-vectors corresponding to recording "file_name"

    #hac_start = time.time()
    # Kaldi-like global norm and lenth-norm
    x = (x-glob_mean).dot(glob_tran.T)
    x *= np.sqrt(x.shape[1] / (x**2).sum(axis=1)[:,np.newaxis])

    # Kaldi-like AHC of x-vectors (scr_mx is matrix of pairwise similarities between all x-vectors)
    # scr_mx = kaldi_ivector_plda_scoring_dense((plda_mu, plda_tr, plda_psi), x, target_energy=target_energy,pca_dim=pca_dim)
    # thr, junk = twoGMMcalib_lin(scr_mx.ravel()) # Optionally, figure out utterance specific threshold for AHC.
    # labels = AHC(scr_mx, thr+threshold) # output "labels" is integer vector of speaker (cluster) ids
    
    labels = np.genfromtxt(f'{args.labelspath}/{file_name}.labels',dtype=str)[:,1]
    _,labels = np.unique(labels,return_inverse=True)
    if use_PIC:
      N= scr_mx.shape[0]
      distance_matrix = 1/(1+np.exp(-scr_mx))
      # for threshold PIC
      n_clusters = 1
      # for oracle number of speakers/AHC PIC
      #threshold=None
      #n_clusters = args.n_speakers
      #n_clusters = len(np.unique(labelsAHC))
      final_k = min(40,N-1)
      z = 0.5
      labelfull=np.arange(N)
      clusterlen_org=[1]*len(labelfull)
      # mypic =pic.PIC_org(n_clusters,clusterlen_org,labelfull,distance_matrix.copy(),threshold,K=final_k,z=z)
      if threshold is None:
            mypic =pic.PIC_org(n_clusters,clusterlen_org,labelfull,distance_matrix.copy(),K=final_k,z=z)
            labels,clusterlen = mypic.gacCluster()
      else:
            mypic =pic.PIC_org_threshold(n_clusters,clusterlen_org,labelfull,distance_matrix.copy(),threshold,K=final_k,z=z)
            labels,clusterlen = mypic.gacCluster()
      #labels1st = np.genfromtxt(args.labelsPIC,dtype=int)[:,1]
      labelsAHC = labels.copy()
    #hac_time = time.time()-hac_start
    if use_VB:
        #vbx_start = time.time()

        # Smooth the hard labels obtained from AHC to soft assignments of x-vectors to speakers
        q_init = np.zeros((len(labels), np.max(labels)+1))
        q_init[range(len(labels)), labels] = 1.0
        q_init = softmax(q_init*args.init_smoothing, axis=1)

        # Transform x-vectors to LDA space and reduce its dimensionality
        # Now, mean is 0, within-class covariance identity and across-class covariance  diagonal (plda_psi)
        fea = (x-plda_mu).dot(plda_tr.T)[:,:args.lda_dim]

        # Use VB-HMM for x-vector clustering. Instead of i-vector extractor model, we use PLDA
        # => GMM with only 1 component, V derived accross-class covariance, and invSigma is inverse within-class covariance (i.e. identity)
        q, sp, L = VB_diarization.VB_diarization(fea, ubmMeans, invSigma, ubmWeights, V, pi=None, gamma=q_init, maxSpeakers=q_init.shape[1], maxIters=args.max_iters, VtinvSigmaV=VtinvSigmaV,
                                        downsample=None, sparsityThr=0.001, epsilon=1e-6, loopProb=args.LoopP, Fa=args.Fa, Fb=args.Fb)

        #vbx_time = time.time()-vbx_start
        labels = np.unique(q.argmax(1), return_inverse=True)[1] 
        

    assert(np.all(segs_dict[file_name][0] == np.array(seg_names)))
    start, end = segs_dict[file_name][1].T
    starts, ends, out_labels  = merge_adjacent_labels(start, end, labels)

#######################################################################################
    if args.use_VB_withoverlap:
    
      if len(np.unique(labels)) > 1:
        pref_org = f'k{args.knn_k}_tau{args.tau}_VBx'
      else:
        pref_org = f'k{args.knn_k}_tau{args.tau}_withoverlap_VBx'
      rttm_org_fold = out_rttm_dir+'/'+'final_{}rttms/'.format(pref_org)
      mkdir_p(rttm_org_fold)

      rttm_file_org = f'{rttm_org_fold}/{file_name}.rttm'
      with open(rttm_file_org, 'w') as fp:
        for l, s, e in zip(out_labels, starts, ends):
          fp.write("SPEAKER %s 1 %.3f %.3f <NA> <NA> %d <NA> <NA>\n" % (file_name, s, e-s, l+1))

      if len(np.unique(labels)) == 1:
          continue
    else:
      mkdir_p(out_rttm_dir)
      with open(out_rttm_dir+'/'+file_name+'.rttm', 'w') as fp:
        for l, s, e in zip(out_labels, starts, ends):
          fp.write("SPEAKER %s 1 %.3f %.3f <NA> <NA> %d <NA> <NA>\n" % (file_name, s, e-s, l+1))

    #with open(out_rttm_dir+'/'+file_name+'.time', 'w') as fp:
    #  fp.write("%f %f %f %f %f\n" % (np.sum(ends-starts), ends[-1],  hac_time, vbx_time, vbf_time))

    if args.use_VB_withoverlap:
        
        ovplabels = np.argsort(q,axis=1)[:,::-1]
        ovplabels = ovplabels[:,1] # second highest speaker
        

        assert(np.all(segs_dict[file_name][0] == np.array(seg_names)))
        start, end = segs_dict[file_name][1].T
        ovpstarts, ovpends, ovpout_labels  = merge_adjacent_labels(start, end, ovplabels)

    #######################################################################################
        pref = f'k{args.knn_k}_tau{args.tau}_overlaponly_VBx'

        outpath = out_rttm_dir+'/'+'final_{}rttms/'.format(pref)
        mkdir_p(outpath)
        with open(outpath+'/'+file_name+'.rttm', 'w') as fp:
          for l, s, e in zip(ovpout_labels, ovpstarts, ovpends):
            fp.write("SPEAKER %s 1 %.3f %.3f <NA> <NA> %d <NA> <NA>\n" % (file_name, s, e-s, l+1))

        # load this rttm
        rttm_overlap = outpath+'/'+file_name+'.rttm'
        ref = load_rttm(rttm_overlap,step=100)
        n_frames = len(ref)

        # load pyannote labels
        overlap_labels = get_overlap_labels(file_name, n_frames,overlap_filename=overlap_filename)
        n_frames = min(n_frames,len(overlap_labels))

        # remove regions which are not overlapping 
        overlap_labels = overlap_labels[:n_frames]
        ref = ref[:n_frames]
        ref[overlap_labels!=1] = '-1'
       
        pref_overlap = f'k{args.knn_k}_tau{args.tau}_withoverlap_VBx'
    
        outpath = out_rttm_dir+'/'+'final_{}rttms/'.format(pref_overlap)
        rttm_newfile = out_rttm_dir+'/'+'final_{}rttms/{}.rttm'.format(pref_overlap,file_name)
        mkdir_p(outpath)
        
        write_rttm_file(rttm_file_org, rttm_newfile,ref,file_name)
        
        # rttm_gndfile = args.rttm_ground_path+'/'+recid+'.rttm'
        # outpath=outpath +'/'+recid
        # create a new rttm with clean one 
        # mkdir_p(out_withoverlap_rttm_dir)
        # with open(out_withoverlap_rttm_dir+'/'+file_name+'.rttm', 'w') as fp:
        #   for l, s, e in zip(out_labels, starts, ends): # clean
        #     fp.write("SPEAKER %s 1 %.3f %.3f <NA> <NA> %d <NA> <NA>\n" % (file_name, s, e-s, l+1))

        #   for l, s, e in zip(ovpout_labels, ovpstarts, ovpends): # overlapping
        #     fp.write("SPEAKER %s 1 %.3f %.3f <NA> <NA> %d <NA> <NA>\n" % (file_name, s, e-s, l+1))

