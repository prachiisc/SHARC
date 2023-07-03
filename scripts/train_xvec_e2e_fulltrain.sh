. ./cmd.sh
. ./path.sh

# dataset to train and test on
stage=Vox # AMI 
python=/home/amritk/.conda/envs/Hilander1/bin/python

# Voxconverse
if [ $stage == "Vox" ]; then
    # training data
    # xvector model
    xvecmodelpath_pkl=xvector_model/fbank_jhu_etdnn.pkl
    dataset=lib_vox_tr_all
    segments_list=lists//${dataset}/segments_xvec
    reco2utt_list=lists/${dataset}/tmp/spk2utt
    featspath=tools_diar/exp_xvec/xvectors_${dataset}_0.75s/subsegments_data/feats.scp
    
    #initialization
    model_filename=checkpoint_amrit/librivox_nonoverlap_sampler_6_PLDA.pth
    xvecpath=/data1/prachis/Dihard_2020/gae-pytorch/gae/tools_diar/xvectors_0.75s_npy/lib_vox_tr_all/
    labelspath=tools_diar/ALL_GROUND_LABELS/lib_vox_tr_all/threshold_0.5/
    pldamodel=plda_models/lib_vox_tr_all_gnd_0.75s/plda_lib_vox_tr_all_gnd_0.75s.pkl

    # mapping between recording name and recording id
    file_pairs=/data1/prachis/Amrit_sharc/lists/lib_vox_tr_all/file_pairs.list
    
    model_savepath=checkpoint/librivox_nonoverlap_sampler_6_PLDA_e2e_fulltrain.pth
    outpath=exp_sharc/results_with_libriplda_trained_libri_e2e/fulltrain/

    mkdir -p $outpath

    $exec_cmd JOB=1 $outpath/e2e.JOB.log \
    $python train_subg_final_e2e.py --mode train,PLDA,rec_aff \
    --model_filename  $model_filename \
    --featspath $featspath \
    --reco2utt_list $reco2utt_list \
    --segments_list $segments_list \
    --dataset_str $dataset \
    --xvec_model_weight_path $xvecmodelpath_pkl \
    --knn_k 60 --levels 15 --hidden 2048 \
    --epochs 50 --lr 0.001 --batch_size 5 \
    --num_conv 1 \
    --balance --use_cluster_feat \
    --xvecpath $xvecpath \
    --labelspath $labelspath \
    --file_pairs $file_pairs \
    --model_savepath $model_savepath \
    --ngpu 1 \
    --pldamodel $pldamodel \
    --fulltrain 1
fi


if [ $stage == "AMI" ];then
    # xvector model
    xvecmodelpath_pkl=xvector_model/fbank_jhu_etdnn.pkl
    # train dataset
    dataset=ami_sdm_train
    segments_list=lists//${dataset}/segments_xvec
    reco2utt_list=lists/${dataset}/tmp/spk2utt
    featspath=tools_diar/exp_xvec/xvectors_${dataset}_0.75s/subsegments_data/feats.scp

    # initialization from SHARC
    model_filename=checkpoint_pre_trained/ami/sharc/ami_sdm_train_model_best.pth

    xvecpath=tools_diar/exp_xvec/xvectors_${dataset}_0.75s/xvector.scp
    labelspath=tools_diar/ALL_GROUND_LABELS/${dataset}/threshold_0.25/
    #PLDA model
    pldamodel=plda_models/ami_sdm_train_gnd/plda_ami_sdm_train_gnd.pkl
    filegroupcount=1
    modified=3
    batch_size=2
    lr=0.001
    k=30

    # output model
    model_savepath=checkpoint/${dataset}/${dataset}_nonoverlap_sampler_3_PLDA_e2e_fulltrain_modified${modified}_nonorm_filecount${filegroupcount}_batchsize${batch_size}/sharcinitk60_lr0.001/
    # output log files
    outpath=exp_sharc/results_with_${dataset}_e2e_fulltrain_modified${modified}_nonorm_filecount${filegroupcount}_batchsize${batch_size}_traink${k}/sharcinitk60_lr0.001/

    mkdir -p $model_savepat
    mkdir -p $outpath

    $train_cmd JOB=1 $outpath/e2e.JOB.log \
    $python train_subg_final_e2e.py \
    --mode train,PLDA,rec_aff \
    --model_filename  $model_filename \
    --featspath $featspath \
    --filegroupcount $filegroupcount \
    --reco2utt_list $reco2utt_list \
    --segments_list $segments_list \
    --dataset_str $dataset \
    --xvec_model_weight_path $xvecmodelpath_pkl \
    --knn_k $k --levels 15 --hidden 2048 \
    --epochs 30 --lr $lr --batch_size $batch_size \
    --num_conv 1 \
    --balance --use_cluster_feat \
    --xvecpath $xvecpath \
    --labelspath $labelspath \
    --pldamodel $pldamodel \
    --model_savepath $model_savepath \
    --ngpu 1 \
    --fulltrain 1 \
    --modified $modified \
    --withcheckpoint 1 
fi
