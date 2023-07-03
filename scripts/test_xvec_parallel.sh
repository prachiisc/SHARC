. ./cmd.sh
. ./path.sh

# dataset to test on
stage=Vox # AMI

python=python

# Voxconverse
if [ $stage == "Vox" ]; then

    for dataset in vox_diar vox_diar_test; do
        segments_list=lists//${dataset}/segments_xvec
        reco2utt_list=lists/${dataset}/tmp/spk2utt
        
        xvecpath=tools_diar/xvectors_0.75s_npy/${dataset}/
        labelspath=tools_diar/ALL_GROUND_LABELS/${dataset}/threshold_0.5/
        pldamodel=plda_models/lib_vox_tr_all_gnd_0.75s/plda_lib_vox_tr_all_gnd_0.75s.pkl

        filelist=lists/${dataset}/${dataset}.list
        rttm_ground_path=lists/${dataset}/filewise_rttms/
        segmentspath=lists/${dataset}/segments_xvec/
        nj=40
        splitname=lists/${dataset}/split$nj
        period=0.75
        model_filename=checkpoint_pre_trained/voxconverse/sharc/librivox_nonoverlap_sampler_6_PLDA.pth
        out_path=exp_sharc/results_with_libriplda_trained_libri/${dataset}/labels_withoutglobalfeats_norm_full

        log_path=$out_path/log
        mkdir -p $out_path
        mkdir -p $log_path
        echo $log_path
        JOB=2
        for k in 30 50; do
            for tau in 0.5; do
                echo "tau=$tau k=$k"
                echo "##################################"
                $exec_cmd_med JOB=1:$nj $log_path/log.JOB.tau${tau}_k${k}.txt \
                    $python test_subg_final.py \
                    --mode "test,PLDA,rec_aff" \
                    --labelspath ${labelspath} \
                    --feats_file $filelist \
                    --out_path $out_path \
                    --knn_k $k \
                    --tau $tau --level 15 \
                    --threshold prob --hidden 2048 --num_conv 1 \
                    --batch_size 4096 --use_cluster_feat \
                    --reco2utt_list $reco2utt_list \
                    --segments_list $segments_list \
                    --dataset_str $dataset \
                    --xvecpath $xvecpath \
                    --model_filename $model_filename \
                    --splitlist $splitname/JOB/full.list \
                    --which_python $python \
                    --rttm_ground_path $rttm_ground_path \
                    --segments $segmentspath \
                    --pldamodel $pldamodel
                    
                bash score.sh $out_path/final_k${k}_tau${tau}rttms/ $dataset
            done
        done
    done
fi

if [ $stage == "AMI" ];then
    for dataset in  ami_dev_fbank_0.75s ;do
    segments_list=lists/${dataset}/segments_xvec
    reco2utt_list=lists/${dataset}/tmp/spk2utt
    lr=0.001
    traink=60
    trainset=ami_sdm_train

    xvecpath=/data1/prachis/Dihard_2020/gae-pytorch/gae/tools_diar/xvectors_0.75s_npy/${dataset}/
    labelspath=tools_diar/ALL_GROUND_LABELS/${dataset}/threshold_0.5/
    pldamodel=plda_models/${trainset}_gnd/plda_${trainset}_gnd.pkl

    filelist=lists/${dataset}/${dataset}.list
    rttm_ground_path=lists/${dataset}/filewise_rttms/
    segmentspath=lists/${dataset}/segments_xvec/
    nj=15
    splitname=lists/${dataset}/split$nj
   
    epochs=400
    dropout=0.0

    model_filename=checkpoint_pre_trained/ami/sharc/ami_sdm_train_model_best.pth
    out_path=exp_sharc/results_with_${trainset}/${dataset}/labels_withoutglobalfeats_norm_full_${epochs}_lr${lr}_traink${traink}_proc/dropout${dropout}
    log_path=$out_path/log
    mkdir -p $out_path
    mkdir -p $log_path
    echo $log_path

    JOB=1
    echo "epochs=$epochs"

    for k in 60; do
        for tau in 0.0; do
            echo "tau=$tau k=$k"
            echo "##################################"
            $exec_cmd_long JOB=1:$nj $log_path/log.JOB.tau${tau}_k${k}.txt \
                $python test_subg_final.py \
                --mode "test,PLDA,rec_aff" \
                --labelspath ${labelspath} \
                --feats_file $filelist \
                --out_path $out_path \
                --knn_k $k \
                --tau $tau --level 15 \
                --threshold prob --hidden 2048 --num_conv 1 \
                --batch_size 4096 --use_cluster_feat \
                --reco2utt_list $reco2utt_list \
                --segments_list $segments_list \
                --dataset_str $dataset \
                --xvecpath $xvecpath \
                --model_filename $model_filename \
                --splitlist $splitname/JOB/full.list \
                --which_python $python \
                --rttm_ground_path $rttm_ground_path \
                --segments $segmentspath \
                --pldamodel $pldamodel \
                --withcheckpoint 
            
            # bash score_noTNO.sh $out_path/final_k${k}_tau${tau}rttms/ $dataset
            bash score.sh $out_path/final_k${k}_tau${tau}rttms/ $dataset
        done
    done
    done
fi