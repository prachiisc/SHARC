. ./cmd.sh
. ./path.sh

# which dataset to train on
stage=$1 # Vox,AMI


if [ $stage == "Vox" ]; then
    loss_weight=1.0
    model_savepath=checkpoint/pretrain_sharcagain/librivox_nonoverlap_sampler_6_PLDA_fixfilecount_loss_weight${loss_weight}/
    mkdir -p $model_savepath
    dataset=lib_vox_tr_all #training set
    reco2utt_list=lists/${dataset}/tmp/spk2utt
    labelspath=tools_diar/ALL_GROUND_LABELS/lib_vox_tr_all/threshold_0.5/
    file_pairs=lists/lib_vox_tr_all/file_pairs.list
    xvecpath=tools_diar/exp_xvec/xvectors_lib_vox_tr_all_gnd_0.75s/xvector.scp
    outpath=exp_sharc/results_with_libriplda_trained_libri/pretrain_sharcagain/
    pldamodel=plda_models/lib_vox_tr_all_gnd_0.75s/plda_lib_vox_tr_all_gnd_0.75s.pkl


    logfile=fixfilecount_loss_weight${loss_weight}
    mkdir -p $outpath
    epochs=500
    lr=0.01
    python=/home/amritk/.conda/envs/Hilander1/bin/python
    $exec_cmd JOB=1 $outpath/$logfile.JOB.log \
    $python train_subg_final.py \
    --mode train,PLDA,rec_aff \
    --train_files train_files_librivox \
    --model_savepath $model_savepath \
    --dataset_str $dataset \
    --labelspath $labelspath \
    --file_pairs $file_pairs \
    --reco2utt_list $reco2utt_list \
    --xvecpath $xvecpath \
    --ngpu 0 \
    --knn_k 60 \
    --levels 15 \
    --hidden 2048 \
    --epochs $epochs \
    --lr $lr \
    --batch_size 5 \
    --num_conv 1 \
    --balance --use_cluster_feat \
    --pldamodel $pldamodel \
    --fixfilecount
fi

if [ $stage == "AMI" ];then
    loss_weight=1.0

    dataset=ami_sdm_train
    # dataset=ami_sdm_train_mdm
    reco2utt_list=lists/${dataset}/tmp/spk2utt
    labelspath=tools_diar/ALL_GROUND_LABELS/${dataset}/threshold_0.25/
    xvecpath=tools_diar/exp_xvec/xvectors_${dataset}_0.75s/xvector.scp
    pldamodel=plda_models/${dataset}_gnd/plda_${dataset}_gnd.pkl
    k=60
    knn_k_val=60
    lr=0.001
    dropout=0.0
    
    test=ami_dev_fbank_0.75s
    reco2utt_list_test=lists/${test}/tmp/spk2utt
    labelspath_test=/data1/prachis/Dihard_2020/gae-pytorch/gae/tools_diar/ALL_GROUND_LABELS/${test}/threshold_0.5/
    xvecpath_test=/data1/prachis/SRE_19/Self_supervised_clustering/tools_diar/exp/xvector_nnet_1a_tdnn_fbank/xvectors_${test}/xvector.scp

    outpath=exp_sharc/results_with_${dataset}/pretrain_sharcagain_lr${lr}_k${k}_v2_dropout${dropout}/
    model_savepath=checkpoint/${dataset}/pretrain_sharcagain/${dataset}_nonoverlap_sampler_6_PLDA_lr${lr}_dropout${dropout}/

    mkdir -p $model_savepath

    logfile=$outpath/log
    mkdir -p $outpath
    mkdir -p $logfile
    echo $logfile

    epochs=500

    python=/home/amritk/.conda/envs/Hilander1/bin/python
    $train_cmd_node5 JOB=1 $logfile/sharc.JOB.log \
    $python train_subg_final_ami.py \
    --mode train,rec_aff,PLDA \
    --train_files train_files \
    --model_savepath $model_savepath \
    --dataset_str $dataset \
    --labelspath $labelspath \
    --reco2utt_list $reco2utt_list \
    --xvecpath $xvecpath \
    --filegroupcount 1 \
    --ngpu 3 \
    --knn_k $k \
    --levels 15 \
    --hidden 2048 \
    --epochs $epochs \
    --lr $lr \
    --batch_size 5 \
    --num_conv 1 \
    --balance --use_cluster_feat \
    --fixfilecount \
    --pldamodel $pldamodel \
    --reco2utt_list_test $reco2utt_list_test \
    --labelspath_test $labelspath_test \
    --xvecpath_test $xvecpath_test \
    --knn_k_val $knn_k_val \
    --dropout $dropout
done