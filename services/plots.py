import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np
import os

from pdb import set_trace as bp
mp.use('Agg')

def train_vs_validation_loss_2ndpass():
    resultspath = '/data1/prachis/Amrit_sharc/exp_sharc/results_with_libriplda_trained_libri/pretrain_sharcagain_lr0.001_fix_filecount_2ndpass_newsplit/'
    train = np.genfromtxt(f'{resultspath}/fix_filecount1_train_loss_weight1.0.1.log',dtype=str)
    val = np.genfromtxt(f'{resultspath}/fix_filecount1_val_loss_weight1.0.1.log',dtype=str)

    # plt.figure()
    # plt.plot(train[:,1].astype(int),train[:,3].astype(float))
    # plt.title('BCE train loss vs epochs')
    # plt.savefig('train_loss_2ndpass_newsplit.png')

    # plt.figure()
    # plt.plot(val[:,1].astype(int),val[:,3].astype(float))
    # plt.title('BCE val loss vs epochs')
    # # plt.legend(['train','val'])
    # plt.savefig('val_loss_2ndpass_newsplit.png')

    plt.figure()
    plt.plot(train[:,1].astype(int),train[:,3].astype(float))
    plt.plot(val[:,1].astype(int),val[:,3].astype(float))
    plt.title('BCE loss vs epochs')
    plt.legend(['train','val'])
    plt.savefig('train_val_loss_2ndpass_newsplit.png')

def train_vs_validation_loss():
    import os
    # resultspath = 'exp_sharc/results_with_libriplda_trained_libri/pretrain_sharcagain_lr0.001_fix_gat_head4/'
    # resultspath = '/data1/prachis/Amrit_sharc/exp_sharc/results_with_ami_sdm_train/pretrain_sharcagain_lr0.001_gat_head1_swa/log/'
    # resultspath = '/data1/prachis/Amrit_sharc/exp_sharc/results_with_ami_sdm_train/pretrain_sharcagain_lr0.001_gat_head1/log/'
    # resultspath = '/data1/prachis/Amrit_sharc/exp_sharc/results_with_ami_sdm_train/pretrain_sharcagain_k60_lr0.001_gat_head1_swa/log'
    # resultspath = '/data1/prachis/Amrit_sharc/exp_sharc/results_with_libriplda_trained_libri/pretrain_sharcagain_lr0.001_fix_gat_head4_swa/'
    
    # resultspath = '/data1/prachis/Amrit_sharc/exp_sharc/results_with_ami_sdm_train/pretrain_sharcagain_k60_lr0.001_gat_head1None//log'
    # resultspath = '/data1/prachis/Amrit_sharc/exp_sharc/results_with_ami_sdm_train/pretrain_sharcagain_k60_lr0.001_gat_head1_swa2//log'
    # resultspath = '/data1/prachis/Amrit_sharc/exp_sharc/results_with_ami_sdm_train/pretrain_sharcagain_k60_lr0.001_gat_head1_normNone//log'
    # resultspath = '/data1/prachis/Amrit_sharc/exp_sharc/results_with_ami_sdm_train/pretrain_sharcagain_k60_lr0.001_gat_head1_norm_swa1//log'
    # resultspath = '/data1/prachis/Amrit_sharc/exp_sharc/results_with_ami_sdm_train/pretrain_sharcagain_k60_lr0.001_gat_head1_norm_swa2//log'

    # resultspath = '/data1/prachis/Amrit_sharc/exp_sharc/results_with_ami_sdm_train/pretrain_sharcagain_lr0.001_k60_v2_dropout0.2_gat_temporal//log'
    # resultspath = '/data1/prachis/Amrit_sharc/exp_sharc/results_with_ami_sdm_train/pretrain_sharcagain_lr0.001_k60_v2_dropout0.2_gat_head4_temporal/log'

    # resultspath = '/data1/prachis/Amrit_sharc/exp_sharc/results_with_ami_sdm_train/pretrain_sharcagain_lr0.001_k60_v2_dropout0.0_LDA128_nofilepca//log/'
    # resultspath = '/data1/prachis/Amrit_sharc/exp_sharc/results_with_ami_sdm_train/pretrain_sharcagain_lr0.001_k60_v2_dropout0.2_gat//log'
    # resultspath = '/data1/prachis/Amrit_sharc/exp_sharc/results_with_ami_sdm_train/pretrain_sharcagain_lr0.001_k60_v2_dropout0.2_temporal/log'
    # resultspath = '/data1/prachis/Amrit_sharc/exp_sharc/ami_sdm_train_mixorg_sim/pretrain_sharcagain/lr0.001_fix_filecount1_withoutinit_2ndpass_overlap_temporal3_0.95_v4approach_usingmodel1_lr0.0001_k_2ndpass30_testorg/'
    resultspath = '/data1/prachis/Amrit_sharc/exp_sharc/ami_sdm_train/pretrain_sharcagain/lr0.001_fix_filecount1_withoutinit_2ndpass_overlap_temporal3_0.95_v4_fineresolution_approach_usingmodel1_lr0.0001_k_2ndpass30/'
    os.system(f'grep Final_epoch {resultspath}/*.log | awk "NR%2==1" > {resultspath}/train_loss.txt')
    os.system(f'grep Final_epoch {resultspath}/*.log | awk "NR%2==0" > {resultspath}/val_loss.txt')
    
    train = np.genfromtxt(f'{resultspath}/train_loss.txt',dtype=str)
    val = np.genfromtxt(f'{resultspath}/val_loss.txt',dtype=str)
    
    plt.figure()
    # epochs=500
    epochs=100
    plt.plot(train[:epochs,1].astype(int),train[:epochs,3].astype(float))
    plt.plot(val[:epochs,1].astype(int),val[:epochs,3].astype(float))
    plt.title('BCE loss vs epochs')
    plt.legend(['train','val'])
    plt.savefig(f'{resultspath}/train_val_loss_2ndpass_newsplit_{epochs}.png')

def validation_der():
    import re
    resultspath = 'logdir/log_ami_sharc_proc_temporal_epochwise.out'
    resultsfile = open(resultspath).readlines()
    pattern = 'OVERALL'
    DER = []
  
    for line in resultsfile:
        if re.search(pattern, line):
            der_val = float(line.split()[3])
            DER.append(der_val)
    DER = np.array(DER)[1::2]
    N = len(DER)
    epochs = np.arange(10,N*10+1,10)
    plt.figure()
    plt.plot(epochs,DER)
    plt.title('DER vs epochs')
    plt.savefig(f'{resultspath}.png')

# plot density histogram
def density_histogram(resultspath):
    import glob 
    files = glob.glob(f'{resultspath}/density_*')
    density_full_clean = []
    density_full_ovp = []
    for file in files:
        densitylist = np.genfromtxt(file,dtype=str)
        labels = densitylist[:,1].astype(float).astype(int)
        clean_ind = np.where(labels==0)[0]
        ovp_ind = np.where(labels==1)[0]
        density_full_clean.append(densitylist[clean_ind,0].astype(float))
        density_full_ovp.append(densitylist[ovp_ind,0].astype(float))
    
    # bp()
    density_full_clean = np.concatenate(density_full_clean,axis=0)
    density_full_ovp = np.concatenate(density_full_ovp, axis=0)
    print(f'overlap percentage:{len(density_full_ovp)/len(density_full_clean)*100}')
    plt.figure()
    plt.subplot(121)
    plt.hist(density_full_clean)
    plt.title('clean')
    # plt.figure()
    plt.subplot(122)
    plt.hist(density_full_ovp)
    plt.title('overlap')
    # plt.legend(['clean','overlap'])
    plt.savefig(f'{resultspath}/histogramplot.png')

def density_paths():
    # vox_diar, k=30, tau=0.8, gnd labels with avg
    # resultspath = '/data1/prachis/Amrit_sharc/exp_sharc/results_with_libriplda_trained_libri_2ndpass/vox_diar/labels_wihtoutglobalfeats_norm_full_40_filegroupcount1_getdensity4/density/'

    # vox_diar, k=30, tau=0.8, gnd labels with org
    # resultspath = '/data1/prachis/Amrit_sharc/exp_sharc/results_with_libriplda_trained_libri_2ndpass/vox_diar/labels_org_wihtoutglobalfeats_norm_full_40_filegroupcount1_getdensity4/density_30'

    # vox_diar, k=60, tau=0.8, gnd labels with org
    # resultspath = '/data1/prachis/Amrit_sharc/exp_sharc/results_with_libriplda_trained_libri_2ndpass/vox_diar/labels_org_wihtoutglobalfeats_norm_full_40_filegroupcount1_getdensity4/density_60'

    # vox_diar, k=100, tau=0.5 gnd labels with org
    # resultspath = '/data1/prachis/Amrit_sharc/exp_sharc/results_with_libriplda_trained_libri_2ndpass/vox_diar/labels_org_wihtoutglobalfeats_norm_full_40_filegroupcount1_getdensity4/density_100'

    # libvox_val_newsplit, k=60, tau=0.5, gnd labels org
    resultspath = '/data1/prachis/Amrit_sharc/exp_sharc/results_with_libriplda_trained_libri_2ndpass/lib_vox_val_newsplit/labels_wihtoutglobalfeats_norm_full_40_filegroupcount1_getdensity5/density'

    # libvox_val_newsplit, k=30, tau=0.8, gnd labels org
    # resultspath = '/data1/prachis/Amrit_sharc/exp_sharc/results_with_libriplda_trained_libri_2ndpass/lib_vox_val_newsplit/labels_wihtoutglobalfeats_norm_full_40_filegroupcount1_getdensity5/density_30'

    density_histogram(resultspath)

def miss_fa_overlap(resultspath,filelist,dataset,threshold=0.5):
    from sklearn.metrics import det_curve
    filelist = np.genfromtxt(filelist,dtype=str)
    # dataset = 'vox_diar'
    try:
        ref = f'/data1/prachis/Dihard_2020/gae-pytorch/gae/tools_diar/ALL_GROUND_LABELS/{dataset}/threshold_{threshold}_avg/'
    except:
        ref = f'tools_diar/ALL_GROUND_LABELS/{dataset}/threshold_{threshold}_avg/'

    reflabels_ind_full = []
    syslabels_ind_full = []
    # bp()
    for filename in filelist:
        refname = f'{ref}/labels_{filename}'
        sysname = f'{resultspath}/{filename}.labels' #luvfz_pred_labels_k30_tau0.8.txt
        
        reflabels = open(refname,'r').readlines()
        syslabels = open(sysname,'r').readlines()
        if len(reflabels) != len(syslabels):
            print(filename)
            bp()

        reflabels_ind = np.zeros((len(reflabels),))
        syslabels_ind = np.zeros((len(syslabels),))
        for idx,rl in enumerate(reflabels):
            rl = rl.split()
            if len(rl) >2: 
                reflabels_ind[idx] = 1
               
        for idx,sl in enumerate(syslabels):
            sl = sl.split()
            if len(sl) >2: 
                syslabels_ind[idx] = 1
                
        reflabels_ind_full.append(reflabels_ind)
        syslabels_ind_full.append(syslabels_ind)
    
    reflabels_ind_full = np.concatenate(reflabels_ind_full,axis=0)
    syslabels_ind_full = np.concatenate(syslabels_ind_full,axis=0)

    fpr, fnr, thresholds = det_curve(reflabels_ind_full, syslabels_ind_full)
    print(f'Miss rate: {fnr[1]} False alarm: {fpr[1]} thresholds: {thresholds[1]}')
    # print(f'Miss rate: {fnr} False alarm: {fpr} thresholds: {thresholds}')



def miss_fa_overlap_detector(resultspath,filelist,dataset,threshold=0.5,shift=None,full=1,avg=1):
    from sklearn.metrics import det_curve
    filelist = np.genfromtxt(filelist,dtype=str)
    # dataset = 'vox_diar'
    if avg==1:
        ref = f'/data1/prachis/Dihard_2020/gae-pytorch/gae/tools_diar/ALL_GROUND_LABELS/{dataset}/threshold_{threshold}_avg/'
        if not os.path.exists(ref):
            if shift is None:
                ref = f'tools_diar/ALL_GROUND_LABELS/{dataset}/threshold_{threshold}_avg/'
            else:
                ref = f'tools_diar/ALL_GROUND_LABELS/{dataset}/threshold_{threshold}_avg_shift/'
    else:
        ref = f'tools_diar/ALL_GROUND_LABELS/{dataset}/threshold_{threshold}/'

    reflabels_ind_full = []
    syslabels_ind_full = []
    # bp()
    for filename in filelist:
        refname = f'{ref}/labels_{filename}'
        sysname = f'{resultspath}/labels_{filename}' #luvfz_pred_labels_k30_tau0.8.txt
        
        reflabels = open(refname,'r').readlines()
        syslabels = np.genfromtxt(sysname,dtype=str)
        if len(reflabels) != len(syslabels):
            print(filename)
            bp()

        reflabels_ind = np.zeros((len(reflabels),))
        
        syslabels_ind = np.array(syslabels[:,1]).astype(int)
        # syslabels_ind = syslabels_ind.astype(int)
        atmax_2spk_ind = []
        for idx,rl in enumerate(reflabels):
            rl = rl.split()
            if len(rl) < 4:
                atmax_2spk_ind.append(idx)
            if len(rl) >2: 
                reflabels_ind[idx] = 1
        atmax_2spk_ind = np.array(atmax_2spk_ind)  

        if full:
            reflabels_ind_full.append(reflabels_ind)
            syslabels_ind_full.append(syslabels_ind)
        else:
            # max 2 spks
            reflabels_ind_full.append(reflabels_ind[atmax_2spk_ind])
            syslabels_ind_full.append(syslabels_ind[atmax_2spk_ind])
        
    reflabels_ind_full = np.concatenate(reflabels_ind_full,axis=0)
    syslabels_ind_full = np.concatenate(syslabels_ind_full,axis=0)

    fpr, fnr, thresholds = det_curve(reflabels_ind_full, syslabels_ind_full)
    print(f'Miss rate: {fnr[1]:.2f} False alarm: {fpr[1]:.2f} thresholds: {thresholds[1]}')
    # print(f'Miss rate: {fnr} False alarm: {fpr} thresholds: {thresholds}')


def miss_fa_overlap_libvox(resultspath):
    from sklearn.metrics import det_curve
    filelist = np.genfromtxt('lists/lib_vox_val_newsplit/lib_vox_val_newsplit.list',dtype=str)
    dataset = 'lib_vox_tr_all'
    ref = f'/data1/prachis/Dihard_2020/gae-pytorch/gae/tools_diar/ALL_GROUND_LABELS/{dataset}/threshold_0.5/'
    reflabels_ind_full = []
    syslabels_ind_full = []
    # bp()
    for filename in filelist:
        refname = f'{ref}/labels_{filename}'
        sysname = f'{resultspath}/{filename}.labels' #luvfz_pred_labels_k30_tau0.8.txt
        
        reflabels = open(refname,'r').readlines()
        syslabels = open(sysname,'r').readlines()
        if len(reflabels) != len(syslabels):
            print(filename)
            bp()

        reflabels_ind = np.zeros((len(reflabels),))
        syslabels_ind = np.zeros((len(syslabels),))
        for idx,rl in enumerate(reflabels):
            rl = rl.split()
            if len(rl) >2: 
                reflabels_ind[idx] = 1
               
        for idx,sl in enumerate(syslabels):
            sl = sl.split()
            if len(sl) >2: 
                syslabels_ind[idx] = 1
                
        reflabels_ind_full.append(reflabels_ind)
        syslabels_ind_full.append(syslabels_ind)
    
    reflabels_ind_full = np.concatenate(reflabels_ind_full,axis=0)
    syslabels_ind_full = np.concatenate(syslabels_ind_full,axis=0)

    fpr, fnr, thresholds = det_curve(reflabels_ind_full, syslabels_ind_full)
    print(f'Miss rate: {fnr} False alarm: {fpr} thresholds: {thresholds}')

def miss_fa_overlap_paths():
    # dataset = 'vox_diar' 
    # ami_dev_fbank_0.75s, ami_sdm_train_sub_0.75s
    dataset='ami_dev_fbank_0.75s'
    filelist = f'lists/{dataset}/{dataset}.list'
    # resultspath = f'exp_sharc/results_with_ami_sdm_train/{dataset}/labels_withoutglobalfeats_norm_full_100_lr0.001_traink60_modified/dropout0.0_temporal_beta0.95_neb3_without_init_2ndpass_overlap_myapproach_withprobpred_v3/trained_withv3approach_pyannoteovpdetectionavg_nodensity/final_k60_tau0.0_ovpth0.0_2ndpassk30_density_gap0.0_overlaprttms'
    # resultspath = f'exp_sharc/results_with_ami_sdm_train/ami_dev_fbank_0.75s/labels_withoutglobalfeats_norm_full_50_lr0.001_traink60_modified/dropout0.0_temporal_beta0.95_neb3_without_init_2ndpass_overlap_myapproach_withprobpred_v4/trained_withv4approach_membership/final_k60_tau0.0_ovpth0.0_2ndpassk30_density_gap0.0_overlaprttms'
    # resultspath = f'exp_sharc/results_with_ami_sdm_train/ami_dev_fbank_0.75s/labels_withoutglobalfeats_norm_full_50_lr0.001_traink60_modified/dropout0.0_temporal_beta0.95_neb3_without_init_2ndpass_overlap_myapproach_withprobpred_v4/trained_withv4approach_avglabels_usingmodel1_membership/final_k60_tau0.0_ovpth0.0_2ndpassk30_density_gap0.0_overlaprttms'
    # resultspath = f'exp_sharc/results_with_ami_sdm_train/ami_dev_fbank_0.75s/labels_withoutglobalfeats_norm_full_50_lr0.001_traink60_modified/dropout0.0_temporal_beta0.95_neb3_without_init_2ndpass_overlap_myapproach_withprobpred_v4/trained_withv4approach_membership/final_k60_tau0.0_ovpth0.5_2ndpassk30_density_gap0.0_overlaprttms'
    # resultspath = f'exp_sharc/results_with_ami_sdm_train/ami_dev_fbank_0.75s/labels_withoutglobalfeats_norm_full_50_lr0.001_traink60_modified/dropout0.0_temporal_beta0.95_neb3_without_init_2ndpass_overlap_myapproach_withprobpred_v4/trained_withv4approach_avglabels_membership/final_k60_tau0.0_ovpth0.5_2ndpassk30_density_gap0.0_overlaprttms' #21.15
    # resultspath = f'exp_sharc/results_with_ami_sdm_train/ami_dev_fbank_0.75s/labels_withoutglobalfeats_norm_full_50_lr0.001_traink60_modified/dropout0.0_temporal_beta0.95_neb3_without_init_2ndpass_overlap_myapproach_withprobpred_v4/trained_withv4approach_avglabels_usingmodel1_membership/final_k60_tau0.0_ovpth0.5_2ndpassk30_density_gap0.0_overlaprttms' #21.44
    # resultspath = f'exp_sharc/results_with_ami_sdm_train/ami_dev_fbank_0.75s/labels_withoutglobalfeats_norm_full_50_lr0.001_traink60_modified/dropout0.0_temporal_beta0.95_neb3_without_init_2ndpass_overlap_myapproach_withprobpred_v4/trained_withv4approach_usingmodel1_membership/final_k60_tau0.0_ovpth0.0_2ndpassk30_density_gap0.0_overlaprttms'
    # resultspath = f'exp_sharc/results_with_ami_sdm_train_simulated_0.75s_mix/ami_dev_fbank_0.75s/labels_withoutglobalfeats_norm_full_50_lr0.001_traink60_modified/dropout0.0_temporal_beta0.95_neb3_without_init_2ndpass_overlap_myapproach_withprobpred_v4/trained_withv4approach_usingmodel1_membership/final_k60_tau0.0_ovpth0.5_2ndpassk30_density_gap0.0_overlaprttms'
    # resultspath = f'exp_sharc/results_with_ami_sdm_train_simulated_0.75s_mix/ami_dev_fbank_0.75s/labels_withoutglobalfeats_norm_full_100_lr0.001_traink60_modified/dropout0.0_temporal_beta0.95_neb3_without_init_2ndpass_overlap_myapproach_withprobpred_v4/trained_withv4approach_usingmodel1/final_k60_tau0.0_ovpth0.5_2ndpassk30_density_gap0.0_overlaprttms'
    # resultspath = f'exp_sharc/results_with_ami_sdm_train_simulated_0.75s_mix/ami_dev_fbank_0.75s/labels_withoutglobalfeats_norm_full_100_lr0.001_traink60_modified/dropout0.0_temporal_beta0.95_neb3_without_init_2ndpass_overlap_myapproach_withprobpred_v4/trained_withv4approach_usingmodel1_intracluster_per0.5/final_k60_tau0.0_ovpth0.5_2ndpassk30_density_gap0.0_overlaprttms'
    # resultspath = f'exp_sharc/results_with_ami_sdm_train_simulated_0.75s_mix/ami_dev_fbank_0.75s/labels_withoutglobalfeats_norm_full_50_lr0.001_traink60_modified/dropout0.0_temporal_beta0.95_neb3_without_init_2ndpass_overlap_myapproach_withprobpred_v4/trained_withv4approach_usingmodel1_intracluster_per0.5_no_density_gap/final_k60_tau0.0_ovpth0.5_2ndpassk30_density_gap0.0_overlaprttms'
    # resultspath = f'exp_sharc/results_with_ami_sdm_train_simulated_0.75s_mix/ami_dev_fbank_0.75s/labels_withoutglobalfeats_norm_full_100_lr0.001_traink60_modified/dropout0.0_temporal_beta0.95_neb3_without_init_2ndpass_overlap_myapproach_withprobpred_v4/trained_withv4approach_usingmodel1_intracluster_per0.5_no_density_gap/final_k60_tau0.0_ovpth0.6_2ndpassk30_density_gap0.0_overlaprttms'
    # miss_fa_overlap(resultspath,filelist,dataset,threshold=0.5)

    # dataset='ami_dev_fbank_win1_0.5s'
    # dataset='ami_dev_fbank_win1_0.25s'
    dataset='ami_dev_fbank_win0.025_0.010s'
   
    # filelist = f'lists/{dataset}/{dataset}.list'
    threshold=0.025
    # resultspath = f'exp_sharc/results_with_ami_sdm_train/ami_dev_fbank_0.75s/labels_withoutglobalfeats_norm_full_50_lr0.001_traink60_modified/dropout0.0_temporal_beta0.95_neb3_without_init_2ndpass_overlap_myapproach_withprobpred_v4/trained_withv4approach_fineresolution_usingmodel1_membership_intracluster_per0.5/final_k60_tau0.0_ovpth0.5_2ndpassk30_density_gap0.0_overlaprttms'
    # resultspath = f'exp_sharc/results_with_ami_sdm_train/ami_dev_fbank_0.75s/labels_withoutglobalfeats_norm_full_50_lr0.001_traink60_modified/dropout0.0_temporal_beta0.95_neb3_without_init_2ndpass_overlap_myapproach_withprobpred_v4/trained_withv4approach_fineresolution_usingmodel1_membership/final_k60_tau0.0_ovpth0.5_2ndpassk30_density_gap0.0_overlaprttms'
    # resultspath = f'exp_sharc/results_with_ami_sdm_train/ami_dev_fbank_0.75s/labels_withoutglobalfeats_norm_full_50_lr0.001_traink60_modified/dropout0.0_temporal_beta0.95_neb3_without_init_2ndpass_overlap_myapproach_withprobpred_v4/trained_withv4approach_fineresolution_usingmodel1_membership_intracluster_per0.5_no_density_gap/final_k60_tau0.0_ovpth0.5_2ndpassk30_density_gap0.0_overlaprttms'
    
    # resultspath = f'tools_diar/PYANNOTE_OVERLAP_LABELS/{dataset}/threshold_0.25_avg'
    resultspath = f'tools_diar/PYANNOTE_OVERLAP_LABELS/{dataset}/threshold_{threshold}'
    # resultspath = f'exp_sharc/ami_sdm_train/overlap_detection/withoutinit_overlap_3linlayer_temporal3_0.95_fineresolution_lr0.001_k_2ndpass20_avglabels/results_on_ami_dev_fbank_win1_0.5s/gnn_overlap_th0.5'
    # resultspath = f'exp_sharc/ami_sdm_train/overlap_detection/withoutinit_overlap_3linlayer_temporal3_0.95_fineresolution_lr0.001_k_2ndpass20_avglabels/results_on_ami_dev_fbank_win1_0.5s_epochs100/gnn_overlap_th0.5_k10'
    miss_fa_overlap_detector(resultspath,filelist,dataset,threshold=threshold,full=1,avg=0)
    epochs=100
    k=10
    # resultspath = f'exp_sharc/ami_sdm_train/overlap_detection/withoutinit_overlap_3linlayer_temporal3_0.95_fineresolution_lr0.001_k_2ndpass20_avglabels/results_on_ami_dev_fbank_win1_0.5s/gnn_overlap_th0.5_k{k}'

    # resultspath = f'exp_sharc/ami_sdm_train/overlap_detection/withoutinit_overlap_3linlayer_temporal3_0.95_fineresolution_lr0.001_k_2ndpass20_avglabels/results_on_ami_dev_fbank_win1_0.5s_epochs{epochs}/gnn_overlap_th0.5'
    # resultspath = f'exp_sharc/ami_sdm_train/overlap_detection/withoutinit_overlap_3linlayer_temporal3_0.95_fineresolution_lr0.001_k_2ndpass20_avglabels/results_on_ami_dev_fbank_win1_0.5s_epochs{epochs}/gnn_overlap_th0.5_k{k}'

    # miss_fa_overlap_detector(resultspath,filelist,dataset,threshold=threshold,full=0)

    # resultspath = f'tools_diar/PYANNOTE_OVERLAP_LABELS/{dataset}/threshold_0.25_avg_shift'
    # miss_fa_overlap_detector(resultspath,filelist,dataset,threshold=0.25,shift=1,full=0)


# for vox_diar
# resultspath = '/data1/prachis/Amrit_sharc/exp_sharc/results_with_libriplda_trained_libri_2ndpass/vox_diar/labels_wihtoutglobalfeats_norm_full_40_filegroupcount1_strategy3/final_k30_tau0.8_ovpth0.9_2ndpassrttms/'
# density_th = 0.5
# resultspath = f'/data1/prachis/Amrit_sharc/exp_sharc/results_with_libriplda_trained_libri_2ndpass/vox_diar/labels_wihtoutglobalfeats_norm_full_40_filegroupcount1_strategy3/density_th{density_th}/final_k30_tau0.8_ovpth0.9_2ndpassrttms/'
# resultspath = f'/data1/prachis/Amrit_sharc/exp_sharc/results_with_libriplda_trained_libri_2ndpass/vox_diar/labels_wihtoutglobalfeats_norm_full_5_filegroupcount1_strategy2/final_k30_tau0.8_ovpth0.9_2ndpassrttms/'
# resultspath = f'/data1/prachis/Amrit_sharc/exp_sharc/results_with_libriplda_trained_libri_2ndpass/vox_diar/labels_wihtoutglobalfeats_norm_full_5_filegroupcount1_strategy2/density_th{density_th}/final_k30_tau0.8_ovpth0.9_2ndpassrttms/'
# density_th = 0.0
# k=30
# strategy='3'
# epochs=40
# ovpth=0.9
# # resultspath = f'exp_sharc/results_with_libriplda_trained_libri_2ndpass/vox_diar/labels_wihtoutglobalfeats_norm_full_{epochs}_filegroupcount1_strategy{strategy}_newsplit/density_th{density_th}_correctmapping/final_k{k}_tau0.8_ovpth{ovpth}_2ndpassrttms/'
# resultspath = f'exp_sharc/results_with_libriplda_trained_libri_2ndpass/vox_diar/labels_wihtoutglobalfeats_norm_full_{epochs}_filegroupcount1_strategy{strategy}/density_th{density_th}_correctmapping/final_k{k}_tau0.8_ovpth{ovpth}_2ndpassrttms/'

# fulltrain_2ndpass
# epochs=20
# strategy=3
# density_th=0.7
# k=30
# ovpth=0.9
# batchsize=2
# resultspath = f'exp_sharc/fulltrain_2ndpass/results_with_libriplda_trained_libri_2ndpass_batch_size{batchsize}/vox_diar/labels_wihtoutglobalfeats_norm_full_{epochs}_ovpepochs5_filegroupcount1_strategy{strategy}/density_th{density_th}_correctmapping/final_k{k}_tau0.8_ovpth{ovpth}_2ndpassrttms/'

#only sharc, no 2ndpass training
# density_th=0.0
# for density_th in [-0.5,0.0,0.3,0.5,0.7]:
# k=30
# strategy='3_v3'

# ovpth=0.0
# resultspath = f'exp_sharc/results_with_libriplda_trained_libri_2ndpass_usingonlysharc/vox_diar/labels_wihtoutglobalfeats_norm_full_strategy{strategy}/density_th{density_th}_correctmapping/final_k{k}_tau0.8_ovpth{ovpth}_2ndpassrttms/'

# resultspath = f'exp_sharc/results_with_libriplda_trained_libri_2ndpass/vox_diar/labels_wihtoutglobalfeats_norm_full_40_filegroupcount1_strategy{strategy}/density_th{density_th}_gndovp_correctmapping/final_k{k}_tau0.8_ovpth{ovpth}_2ndpassrttms/'
# resultspath = f'exp_sharc/results_with_libriplda_trained_libri_2ndpass/vox_diar/labels_wihtoutglobalfeats_norm_full_40_filegroupcount1_strategy{strategy}/density_th{density_th}_gndovp_correctmapping_k2_20_tau2_0.5/final_k{k}_tau0.8_ovpth{ovpth}_2ndpassrttms/'

# resultspath = f'exp_sharc/results_with_libriplda_trained_libri_2ndpass/vox_diar/labels_wihtoutglobalfeats_norm_full_40_filegroupcount1_strategy{strategy}/density_th{density_th}_gndovp_correctmapping_onlysharc/final_k{k}_tau0.8_ovpth{ovpth}_2ndpassrttms/'

# resultspath = f'exp_sharc/results_with_libriplda_trained_libri_2ndpass/vox_diar/labels_wihtoutglobalfeats_norm_full_40_filegroupcount1_strategy{strategy}/density_th{density_th}_gndovp_correctmapping_orglabels/final_k{k}_tau0.8_ovpth{ovpth}_2ndpassrttms/'

# print('')
# print(resultspath)
# miss_fa_overlap(resultspath)

# for libvox val simulated
# density_th = 0.0
# strategy = '1newsplit'
# epochs=5
# k=100
# tau=0.5
# resultspath = f'/data1/prachis/Amrit_sharc/exp_sharc/results_with_libriplda_trained_libri_2ndpass/lib_vox_val_newsplit/labels_wihtoutglobalfeats_norm_full_{epochs}_filegroupcount1_strategy{strategy}_newsplit/density_th{density_th}_correctmapping/final_k{k}_tau{tau}_ovpth0.9_2ndpassrttms/'
# miss_fa_overlap_libvox(resultspath)

# train_vs_validation_loss()
# validation_der()
miss_fa_overlap_paths()