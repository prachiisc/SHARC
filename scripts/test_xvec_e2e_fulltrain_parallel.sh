. ./cmd.sh
. ./path.sh

# dataset to test on
# stage=Vox # E-SHARC clustering, # AMI
stage=AMI_extract_xvec
python=/home/prachis/.conda/envs/Hilander1/bin/python # path of python containing required libraries

if [ $stage == "Vox" ]; then
    xvecmodelpath_pkl=xvector_model/fbank_jhu_etdnn.pkl

    for dataset in vox_diar vox_diar_test;do
    for epochs in 20;do
    segments_list=lists/${dataset}/segments_xvec
    reco2utt_list=lists/${dataset}/tmp/spk2utt

    featspath=tools_diar/exp_xvec/xvectors_${dataset}_0.75s/subsegments_data/feats.scp
    model_filename=checkpoint_pre_trained/voxconverse/sharc/librivox_nonoverlap_sampler_6_PLDA.pth
    xvecpath=tools_diar/exp_xvec/xvectors_${dataset}_0.75s/
    labelspath=tools_diar/ALL_GROUND_LABELS/${dataset}/threshold_0.5/
    pldamodel=plda_models/lib_vox_tr_all_gnd_0.75s/plda_lib_vox_tr_all_gnd_0.75s.pkl
       
    filelist=lists/${dataset}/${dataset}.list
    rttm_ground_path=lists/${dataset}/filewise_rttms/
    segmentspath=lists/${dataset}/segments_xvec/
    echo $rttm_ground_path

    traink=60
    model_savepath=checkpoint_pre_trained/voxconverse/e_sharc/librivox_nonoverlap_sampler_6_PLDA_e2e_${epochs}_snapshot.pth
    out_path=exp_sharc/results_with_${dataset}_e2e/fulltrain_epochs${epochs}/labels_withoutglobalfeats_norm_${epochs}

    log_path=$out_path/log
    
    nj=40
    splitname=lists/${dataset}/split$nj

    mkdir -p $out_path
    mkdir -p $log_path
    echo $log_path

    for k in 30; do
        for tau in 0.8; do
            echo "tau=$tau k=$k"
            echo "##################################"
            $exec_cmd_long JOB=1:$nj $log_path/teste2e.JOB.tau${tau}_k${k}.log \
                $python test_subg_final_e2e.py \
                --mode "test,PLDA,rec_aff" \
                --featspath ${featspath} \
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
                --model_savepath $model_savepath  \
                --splitlist $splitname/JOB/full.list \
                --rttm_ground_path $rttm_ground_path \
                --segments $segmentspath \
                --which_python $python \
                --fulltrain 1 \
                --pldamodel $pldamodel

            bash score.sh $out_path/final_k${k}_tau${tau}rttms/ $dataset
        done
    done

    done
    done
fi


if [ $stage == "AMI" ]; then
    # E2E_SHARC without temporal continuity

    xvecmodelpath_pkl=xvector_model/fbank_jhu_etdnn.pkl

    for dataset in ami_dev_fbank_0.75s ami_eval_fbank_0.75s; do

    segments_list=lists/${dataset}/segments_xvec
    reco2utt_list=lists/${dataset}/tmp/spk2utt
    featspath=tools_diar/exp_xvec/xvectors_${dataset}/subsegments_data/feats.scp
    xvecpath=tools_diar/exp_xvec/xvectors_${dataset}/
    labelspath=tools_diar/ALL_GROUND_LABELS/${dataset}/threshold_0.5/
    pldamodel=plda_models/ami_sdm_train_gnd/plda_ami_sdm_train_gnd.pkl
    
    filegroupcount=1
    batch_size=2

    for epoch in 25;do
    traindataset=ami_sdm_train
   
    filelist=lists/${dataset}/${dataset}.list
    rttm_ground_path=lists/${dataset}/filewise_rttms/
    segmentspath=lists/${dataset}/segments_xvec/
    echo $rttm_ground_path
    traink=30
    
    model_savepath=checkpoint_pre_trained/ami/e_sharc/sharcinitk60_lr0.001_model_25_snapshot.pth
    out_path=exp_sharc/results_with_${traindataset}_e2e_fulltrain_sharcinitk60_lr0.001_k${traink}/withnonorm/$dataset/labels_withoutglobalfeats_norm_${epoch}
    log_path=$out_path/log
   
    nj=15
    splitname=lists/${dataset}/split$nj

    mkdir -p $out_path
    mkdir -p $log_path
    echo $log_path

    JOB=4
    for k in 50; do
        for tau in 0.0; do
            echo "tau=$tau k=$k"
            echo "##################################"
             $exec_cmd_long JOB=1:$nj $log_path/log.JOB.tau${tau}_k${k}.txt \
                $python test_subg_final_e2e.py \
                --mode "test,PLDA,rec_aff" \
                --featspath ${featspath} \
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
                --model_savepath $model_savepath  \
                --splitlist $splitname/JOB/full.list \
                --rttm_ground_path $rttm_ground_path \
                --segments $segmentspath \
                --which_python $python \
                --pldamodel $pldamodel \
                --fulltrain 1  
            
            # bash score_noTNO.sh $out_path/final_k${k}_tau${tau}rttms/ $dataset

            bash score.sh $out_path/final_k${k}_tau${tau}rttms/ $dataset
        done
    done

    done
    done
fi

#Extract ESHARC feats 
if [ $stage == "AMI_extract_xvec" ]; then
    # E2E_SHARC without temporal continuity

    xvecmodelpath_pkl=xvector_model/fbank_jhu_etdnn.pkl
    period=0.75
    # ami_dev_0.75s
    for dataset in ami_eval_0.75s; do
    # for dataset in ami_sdm_train; do
    nj=15
    segments_list=lists/${dataset}/segments_xvec
    reco2utt_list=lists/${dataset}/tmp/spk2utt
    # featspath=tools_diar/exp_xvec/xvectors_${dataset}_${period}s/subsegments_data/feats.scp
    # xvecpath=tools_diar/exp_xvec/xvectors_${dataset}_${period}s/
    # labelspath=tools_diar/ALL_GROUND_LABELS/${dataset}_${period}s/threshold_0.25/

    featspath=tools_diar/exp_xvec/xvectors_${dataset}/subsegments_data/feats.scp
    xvecpath=tools_diar/exp_xvec/xvectors_${dataset}/
    labelspath=tools_diar/ALL_GROUND_LABELS/${dataset}/threshold_0.5/
    pldamodel=plda_models/ami_sdm_train_gnd/plda_ami_sdm_train_gnd.pkl
    
    filegroupcount=1
    batch_size=2

    for epoch in 20;do
    traindataset=ami_sdm_train
   
    filelist=lists/${dataset}/${dataset}.list
    rttm_ground_path=lists/${dataset}/filewise_rttms/
    segmentspath=lists/${dataset}/segments_xvec/
    echo $rttm_ground_path
    traink=30
    model_savepath=checkpoint/${traindataset}/${traindataset}_nonoverlap_sampler_3_PLDA_e2e_fulltrain_nonorm_filecount1_batchsize2/sharcinitk60_lr0.001/model_${epoch}_snapshot.pth

    # model_savepath=checkpoint_pre_trained/ami/e_sharc/sharcinitk60_lr0.001_model_25_snapshot.pth
    out_path=exp_sharc/results_with_${traindataset}_e2e_fulltrain_sharcinitk60_lr0.001_k${traink}/withnonorm/$dataset/labels_withoutglobalfeats_norm_${epoch}
    xvecpath_out=exp_sharc/results_with_${traindataset}_e2e_fulltrain_sharcinitk60_lr0.001_k${traink}/withnonorm/$dataset/esharc_xvectors_${epoch}/
    log_path=$out_path/log
   
    
    splitname=lists/${dataset}/split$nj
    mkdir -p $xvecpath_out
    mkdir -p $out_path
    mkdir -p $log_path
    echo $log_path
    # 1,15,19-23:2,34,39
    JOB=1
    for k in 50; do
        for tau in 0.0; do
            echo "tau=$tau k=$k"
            echo "##################################"
             $exec_cmd JOB=2:7 $log_path/log.JOB.tau${tau}_k${k}.txt \
                $python test_subg_final_e2e.py \
                --mode "test,PLDA,rec_aff" \
                --mymode "extract_xvec" \
                --featspath ${featspath} \
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
                --model_savepath $model_savepath  \
                --splitlist $splitname/JOB/full.list \
                --rttm_ground_path $rttm_ground_path \
                --segments $segmentspath \
                --which_python $python \
                --pldamodel $pldamodel \
                --fulltrain 1  \
                --xvecpath_out $xvecpath_out
            
            
            # bash score_noTNO.sh $out_path/final_k${k}_tau${tau}rttms/ $dataset

            # bash score.sh $out_path/final_k${k}_tau${tau}rttms/ $dataset
        done
    done

    done
    done
fi


#Extract ESHARC feats 
if [ $stage == "AMI_extract_pldafeats" ]; then
    # E2E_SHARC without temporal continuity

    xvecmodelpath_pkl=xvector_model/fbank_jhu_etdnn.pkl
    period=0.75
    # for dataset in ami_dev_fbank_0.75s ami_eval_fbank_0.75s; do
    for dataset in ami_sdm_train; do
    nj=40
    segments_list=lists/${dataset}/segments_xvec
    reco2utt_list=lists/${dataset}/tmp/spk2utt
    featspath=tools_diar/exp_xvec/xvectors_${dataset}_${period}s/subsegments_data/feats.scp
    xvecpath=tools_diar/exp_xvec/xvectors_${dataset}_${period}s/
    labelspath=tools_diar/ALL_GROUND_LABELS/${dataset}_${period}s/threshold_0.25/
    pldamodel=plda_models/ami_sdm_train_gnd/plda_ami_sdm_train_gnd.pkl
    
    filegroupcount=1
    batch_size=2

    for epoch in 20;do
    traindataset=ami_sdm_train
   
    filelist=lists/${dataset}/${dataset}.list
    rttm_ground_path=lists/${dataset}/filewise_rttms/
    segmentspath=lists/${dataset}/segments_xvec/
    echo $rttm_ground_path
    traink=30
    model_savepath=checkpoint/${traindataset}/${traindataset}_nonoverlap_sampler_3_PLDA_e2e_fulltrain_nonorm_filecount1_batchsize2/sharcinitk60_lr0.001/model_${epoch}_snapshot.pth

    # model_savepath=checkpoint_pre_trained/ami/e_sharc/sharcinitk60_lr0.001_model_25_snapshot.pth
    out_path=exp_sharc/results_with_${traindataset}_e2e_fulltrain_sharcinitk60_lr0.001_k${traink}/withnonorm/$dataset/labels_withoutglobalfeats_norm_${epoch}
    xvecpath_out=exp_sharc/results_with_${traindataset}_e2e_fulltrain_sharcinitk60_lr0.001_k${traink}/withnonorm/$dataset/plda_xvectors_${epoch}/
    log_path=$out_path/log_v3
   
    
    splitname=lists/${dataset}/split$nj
    mkdir -p $xvecpath_out
    mkdir -p $out_path
    mkdir -p $log_path
    echo $log_path
    # 1,15,19-23:2,34,39
    JOB=38
    for k in 50; do
        for tau in 0.0; do
            echo "tau=$tau k=$k"
            echo "##################################"
             $exec_cmd JOB=1:$nj $log_path/log.JOB.tau${tau}_k${k}.txt \
                $python test_subg_final_e2e.py \
                --mode "test,PLDA,rec_aff" \
                --mymode "extract_pldafeats" \
                --featspath ${featspath} \
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
                --model_savepath $model_savepath  \
                --splitlist $splitname/JOB/full.list \
                --rttm_ground_path $rttm_ground_path \
                --segments $segmentspath \
                --which_python $python \
                --pldamodel $pldamodel \
                --fulltrain 1  \
                --xvecpath_out $xvecpath_out
            
            # bash score_noTNO.sh $out_path/final_k${k}_tau${tau}rttms/ $dataset

            # bash score.sh $out_path/final_k${k}_tau${tau}rttms/ $dataset
        done
    done

    done
    done
fi
