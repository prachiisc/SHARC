#/bin/bash
. ./cmd.sh
. ./path.sh

stage=21 #SC
# stage=32 # AHC
python=/home/prachis/.conda/envs/Hilander1/bin/python

# Second pass , SC, AMI
if [ $stage -eq 11 ]; then
    clustering=spectral
    # dataset=vox_diar
    # dataset=vox_diar_test
    # displace_eval_fbank_29thmay_seg_0.75s
    # dataset=ami_eval_fbank_0.75s
    for dataset in ami_eval_fbank_0.75s; do
    nj=15

    segments_list=lists/${dataset}/segments_xvec
    reco2utt_list=lists/${dataset}/tmp/spk2utt
    model_filename=checkpoint_amrit/librivox_nonoverlap_sampler_6_PLDA.pth
    xvecpath=tools_diar/exp_xvec/xvectors_${dataset}/
    
    if grep -q "dev" <<< "$dataset"; then
        #dev
        labelspath=tools_diar/ALL_GROUND_LABELS/${dataset}/threshold_0.5_avg/
    else
        #eval
        labelspath=None
    fi
    k_2ndpass=30
    overlap_filename=pyannote_overlap/$dataset/
    for modestat in 15 30; do
    density_gap=0.0
    overlap_th=0.0

    filelist=lists/${dataset}/${dataset}.list
    rttm_ground_path=lists/${dataset}/filewise_rttms/
    segmentspath=lists/${dataset}/segments_xvec/
    labels_dir=/data1/prachis/Amrit_sharc/exp/results_pldaami_spectral_baseline/$dataset/final_spectral_baseline_scaled10_widePLDA_threshold0.7rttms/
    
    echo $rttm_ground_path
    out_path=exp_sharc/results_with_${dataset}/$clustering/labels_2ndpass_modestat${modestat}_pyannoteoptimize
    
    log_path=$out_path/log

    splitname=lists/${dataset}/split$nj

    mkdir -p $out_path
    mkdir -p $log_path
    echo $log_path
    # rm $log_path/log_*
    JOB=1
    k=60
    tau=0.0

    echo "tau=$tau k=$k"
    echo "##################################"
    $exec_cmd_long JOB=1:$nj $log_path/testcluster_ovp.JOB.log \
        $python test_baseline_clustering_ovp.py \
        --mode "test,PLDA,rec_aff,pyannote" \
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
        --splitlist $splitname/JOB/full.list \
        --rttm_ground_path $rttm_ground_path \
        --segments $segmentspath \
        --which_python $python \
        --k_2ndpass $k_2ndpass \
        --overlap_filename $overlap_filename \
        --modestat $modestat \
        --overlap_th $overlap_th \
        --density_gap $density_gap \
        --labels_dir $labels_dir

    # bash score.sh $out_path/final_k${k}_tau${tau}rttms/ $dataset
    bash score_nocollar.sh $out_path/final_2ndpassk${k_2ndpass}_density_gap${density_gap}_overlaprttms/ $dataset

    done
    done
fi



# Second pass , AHC, AMI
if [ $stage -eq 12 ]; then
    clustering=AHC
    # dataset=vox_diar
    # dataset=vox_diar_test
    # displace_eval_fbank_29thmay_seg_0.75s
    # dataset=ami_eval_fbank_0.75s
    for dataset in ami_dev_fbank_0.75s ami_eval_fbank_0.75s; do
    nj=15

    segments_list=lists/${dataset}/segments_xvec
    reco2utt_list=lists/${dataset}/tmp/spk2utt
    model_filename=checkpoint_amrit/librivox_nonoverlap_sampler_6_PLDA.pth
    xvecpath=tools_diar/exp_xvec/xvectors_${dataset}/
    
    if grep -q "dev" <<< "$dataset"; then
        #dev
        labelspath=tools_diar/ALL_GROUND_LABELS/${dataset}/threshold_0.5_avg/
    else
        #eval
        labelspath=None
    fi
    for k_2ndpass in 30 60; do
    overlap_filename=pyannote_overlap/$dataset/
    for modestat in 15 30; do
    density_gap=0.0
    overlap_th=0.0

   
    filelist=lists/${dataset}/${dataset}.list
    rttm_ground_path=lists/${dataset}/filewise_rttms/
    segmentspath=lists/${dataset}/segments_xvec/
    labels_dir=/data1/prachis/Dihard_2020/gae-pytorch/gae/exp_baselines/results_pldaAhc_baseline/$dataset/final_ahc_baseline_AmiPLDA_threshold0.25rttms
    
    echo $rttm_ground_path
    out_path=exp_sharc/results_with_${dataset}/$clustering/labels_2ndpass_modestat${modestat}_pyannoteoptimize
    
    log_path=$out_path/log

    splitname=lists/${dataset}/split$nj

    mkdir -p $out_path
    mkdir -p $log_path
    echo $log_path
    # rm $log_path/log_*
    JOB=1
    k=60
    tau=0.0

    echo "tau=$tau k=$k"
    echo "##################################"
    $exec_cmd_long JOB=1:$nj $log_path/teste2e.JOB.log \
        $python test_baseline_clustering_ovp.py \
        --mode "test,PLDA,rec_aff,pyannote" \
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
        --splitlist $splitname/JOB/full.list \
        --rttm_ground_path $rttm_ground_path \
        --segments $segmentspath \
        --which_python $python \
        --k_2ndpass $k_2ndpass \
        --overlap_filename $overlap_filename \
        --modestat $modestat \
        --overlap_th $overlap_th \
        --density_gap $density_gap \
        --labels_dir $labels_dir

    # bash score.sh $out_path/final_k${k}_tau${tau}rttms/ $dataset
    bash score_nocollar.sh $out_path/final_2ndpassk${k_2ndpass}_density_gap${density_gap}_overlaprttms/ $dataset

    done
    done
    done
fi



# Second pass , SC, Vox
if [ $stage -eq 21 ]; then
    clustering=spectral
    # dataset=vox_diar
    # dataset=vox_diar_test
    # displace_eval_fbank_29thmay_seg_0.75s
    # dataset=ami_eval_fbank_0.75s
    period=0.75
    for dataset in vox_diar vox_diar_test; do
    nj=40

    segments_list=lists/${dataset}/segments_xvec
    reco2utt_list=lists/${dataset}/tmp/spk2utt
    model_filename=checkpoint_amrit/librivox_nonoverlap_sampler_6_PLDA.pth
    xvecpath=tools_diar/exp_xvec/xvectors_${dataset}_${period}s/
    
    if grep -q "dev" <<< "$dataset"; then
        #dev
        labelspath=tools_diar/ALL_GROUND_LABELS/${dataset}/threshold_0.5_avg/
    else
        #eval
        labelspath=None
    fi
    k_2ndpass=30
    overlap_filename=pyannote_overlap/$dataset/
    for modestat in  30; do
    density_gap=0.0
    overlap_th=0.0
    thr=0.8 # 0.7

   
    filelist=lists/${dataset}/${dataset}.list
    rttm_ground_path=lists/${dataset}/filewise_rttms/
    segmentspath=lists/${dataset}/segments_xvec/
    labels_dir=/data1/prachis/Dihard_2020/gae-pytorch/gae/exp_baselines/results_pldalibvox_spectral_baseline/$dataset/final_spectral_baseline_scaled10_widePLDA_threshold${thr}rttms
    echo $rttm_ground_path
    out_path=exp_sharc/results_with_${dataset}/$clustering/labels_init_threshold${thr}_2ndpass_modestat${modestat}_pyannoteoptimize
    
    log_path=$out_path/log

    splitname=lists/${dataset}/split$nj

    mkdir -p $out_path
    mkdir -p $log_path
    echo $log_path
    # rm $log_path/log_*
    JOB=1
    k=60
    tau=0.0

    echo "tau=$tau k=$k"
    echo "##################################"
    $exec_cmd_long JOB=1:$nj $log_path/testcluster_ovp.JOB.log \
        $python test_baseline_clustering_ovp.py \
        --mode "test,PLDA,rec_aff,pyannote" \
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
        --splitlist $splitname/JOB/full.list \
        --rttm_ground_path $rttm_ground_path \
        --segments $segmentspath \
        --which_python $python \
        --k_2ndpass $k_2ndpass \
        --overlap_filename $overlap_filename \
        --modestat $modestat \
        --overlap_th $overlap_th \
        --density_gap $density_gap \
        --labels_dir $labels_dir

    # bash score.sh $out_path/final_k${k}_tau${tau}rttms/ $dataset
    bash score_nocollar.sh $out_path/final_2ndpassk${k_2ndpass}_density_gap${density_gap}_overlaprttms/ $dataset

    done
    done
fi


# Second pass , AHC, Vox
if [ $stage -eq 22 ]; then
    clustering=AHC
    period=0.75
    # dataset=vox_diar
    # dataset=vox_diar_test
    # displace_eval_fbank_29thmay_seg_0.75s
    # dataset=ami_eval_fbank_0.75s
    for dataset in vox_diar vox_diar_test; do
    nj=40

    segments_list=lists/${dataset}/segments_xvec
    reco2utt_list=lists/${dataset}/tmp/spk2utt
    model_filename=checkpoint_amrit/librivox_nonoverlap_sampler_6_PLDA.pth
    xvecpath=tools_diar/exp_xvec/xvectors_${dataset}_${period}s/
    
    if grep -q "dev" <<< "$dataset"; then
        #dev
        labelspath=tools_diar/ALL_GROUND_LABELS/${dataset}/threshold_0.5_avg/
    else
        #eval
        labelspath=None
    fi
    k_2ndpass=60
    overlap_filename=pyannote_overlap/$dataset/
    for modestat in 15 30 45 60; do
    density_gap=0.0
    overlap_th=0.0

   
    filelist=lists/${dataset}/${dataset}.list
    rttm_ground_path=lists/${dataset}/filewise_rttms/
    segmentspath=lists/${dataset}/segments_xvec/
    labels_dir=/data1/prachis/Dihard_2020/gae-pytorch/gae/exp_baselines/results_pldalibvox_ahc_baseline/$dataset/final_ahc_baseline_widePLDA_threshold0.4rttms/

    echo $rttm_ground_path
    out_path=exp_sharc/results_with_${dataset}/$clustering/labels_2ndpass_modestat${modestat}_pyannoteoptimize
    
    log_path=$out_path/log

    splitname=lists/${dataset}/split$nj

    mkdir -p $out_path
    mkdir -p $log_path
    echo $log_path
    # rm $log_path/log_*
    JOB=1
    k=60
    tau=0.0

    echo "tau=$tau k=$k"
    echo "##################################"
    $exec_cmd_long JOB=1:$nj $log_path/teste2e.JOB.log \
        $python test_baseline_clustering_ovp.py \
        --mode "test,PLDA,rec_aff,pyannote" \
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
        --splitlist $splitname/JOB/full.list \
        --rttm_ground_path $rttm_ground_path \
        --segments $segmentspath \
        --which_python $python \
        --k_2ndpass $k_2ndpass \
        --overlap_filename $overlap_filename \
        --modestat $modestat \
        --overlap_th $overlap_th \
        --density_gap $density_gap \
        --labels_dir $labels_dir

    # bash score.sh $out_path/final_k${k}_tau${tau}rttms/ $dataset
    bash score_nocollar.sh $out_path/final_2ndpassk${k_2ndpass}_density_gap${density_gap}_overlaprttms/ $dataset

    done
    done
fi


# Second pass , SC, Displace
if [ $stage -eq 31 ]; then
    clustering=spectral
    # dataset=vox_diar
    # dataset=vox_diar_test
    # displace_eval_fbank_29thmay_seg_0.75s
    # dataset=ami_eval_fbank_0.75s
    period=0.75
    for dataset in displace_eval_fbank_29thmay_seg_0.75s; do
    nj=20

    segments_list=lists/${dataset}/segments_xvec
    reco2utt_list=lists/${dataset}/tmp/spk2utt
    model_filename=checkpoint_amrit/librivox_nonoverlap_sampler_6_PLDA.pth
    xvecpath=tools_diar/exp_xvec/xvectors_${dataset}/
    
    if grep -q "dev" <<< "$dataset"; then
        #dev
        labelspath=tools_diar/ALL_GROUND_LABELS/${dataset}/threshold_0.5_avg/
    else
        #eval
        labelspath=None
    fi
    k_2ndpass=60

    overlap_filename=pyannote_overlap_optimize/$dataset/
    for modestat in 15 30 45 60; do
    density_gap=0.0
    overlap_th=0.0

   
    filelist=lists/${dataset}/${dataset}.list
    rttm_ground_path=lists/${dataset}/filewise_rttms/
    segmentspath=lists/${dataset}/segments_xvec/
    labels_dir=/data1/prachis/Amrit_sharc/tools_diar/exp_xvec/xvectors_$dataset/tuning_0.75_$clustering/per_file_labels/
    echo $rttm_ground_path
    out_path=exp_sharc/results_with_${dataset}/$clustering/labels_2ndpass_modestat${modestat}_pyannoteoptimize
    
    log_path=$out_path/log

    splitname=lists/${dataset}/split$nj

    mkdir -p $out_path
    mkdir -p $log_path
    echo $log_path
    # rm $log_path/log_*
    JOB=1
    k=60
    tau=0.0

    echo "tau=$tau k=$k"
    echo "k_2ndpass=$k_2ndpass"
    echo "##################################"
    $exec_cmd_long JOB=1:$nj $log_path/testcluster_ovp.JOB.log \
        $python test_baseline_clustering_ovp.py \
        --mode "test,PLDA,rec_aff,pyannote" \
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
        --splitlist $splitname/JOB/full.list \
        --rttm_ground_path $rttm_ground_path \
        --segments $segmentspath \
        --which_python $python \
        --k_2ndpass $k_2ndpass \
        --overlap_filename $overlap_filename \
        --modestat $modestat \
        --overlap_th $overlap_th \
        --density_gap $density_gap \
        --labels_dir $labels_dir

    # bash score.sh $out_path/final_k${k}_tau${tau}rttms/ $dataset
    bash score_nocollar.sh $out_path/final_2ndpassk${k_2ndpass}_density_gap${density_gap}_overlaprttms/ $dataset
    
    done
    done
fi


# Second pass , AHC, Displace
if [ $stage -eq 32 ]; then
    clustering=AHC
    period=0.75

    # displace_eval_fbank_29thmay_seg_0.75s
    # dataset=ami_eval_fbank_0.75s
    for dataset in displace_eval_fbank_29thmay_seg_0.75s; do
    nj=20

    segments_list=lists/${dataset}/segments_xvec
    reco2utt_list=lists/${dataset}/tmp/spk2utt
    model_filename=checkpoint_amrit/librivox_nonoverlap_sampler_6_PLDA.pth
    xvecpath=tools_diar/exp_xvec/xvectors_${dataset}/
    
    if grep -q "dev" <<< "$dataset"; then
        #dev
        labelspath=tools_diar/ALL_GROUND_LABELS/${dataset}/threshold_0.5_avg/
    else
        #eval
        labelspath=None
    fi
    k_2ndpass=60

    overlap_filename=pyannote_overlap_optimize/$dataset/
    for modestat in 15 30 45 60; do
    density_gap=0.0
    overlap_th=0.0

   
    filelist=lists/${dataset}/${dataset}.list
    rttm_ground_path=lists/${dataset}/filewise_rttms/
    segmentspath=lists/${dataset}/segments_xvec/
    labels_dir=/data1/prachis/Amrit_sharc/tools_diar/exp_xvec/xvectors_$dataset/tuning_0.75_$clustering/per_file_labels/

    echo $rttm_ground_path
    out_path=exp_sharc/results_with_${dataset}/$clustering/labels_2ndpass_modestat${modestat}_pyannoteoptimize
    
    log_path=$out_path/log

    splitname=lists/${dataset}/split$nj

    mkdir -p $out_path
    mkdir -p $log_path
    echo $log_path
    # rm $log_path/log_*
    JOB=1
    k=60
    tau=0.0

    echo "tau=$tau k=$k"
    echo "k_2ndpass=$k_2ndpass"
    echo "##################################"
    $exec_cmd_long JOB=1:$nj $log_path/teste2e.JOB.log \
        $python test_baseline_clustering_ovp.py \
        --mode "test,PLDA,rec_aff,pyannote" \
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
        --splitlist $splitname/JOB/full.list \
        --rttm_ground_path $rttm_ground_path \
        --segments $segmentspath \
        --which_python $python \
        --k_2ndpass $k_2ndpass \
        --overlap_filename $overlap_filename \
        --modestat $modestat \
        --overlap_th $overlap_th \
        --density_gap $density_gap \
        --labels_dir $labels_dir

    # bash score.sh $out_path/final_k${k}_tau${tau}rttms/ $dataset
    bash score_nocollar.sh $out_path/final_2ndpassk${k_2ndpass}_density_gap${density_gap}_overlaprttms/ $dataset
    
    done
    done
fi