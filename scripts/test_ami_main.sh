#bin/bash
. ./cmd.sh
. ./path.sh

stage=2
stop_stage=2

issad=0 # computing sad 

overlap=1 # pyannote overlap output is available
# extract x-vectors
stage_extract=1
stop_stage_extract=7

python=/home/prachis/.conda/envs/Hilander1/bin/python
pyannote_pretrained_model=vad_benchmarking/VAD_model/pytorch_model.bin
#0.15 is the best with DER 28.22 for ESHARC
# 0.25 0.2 0.15 0.1 0.05 #0.15 is the best
# for sadthreshold in 0.15 0.2 0.25; do #pyannote onset and offset
sadthreshold=0.15
dataset_org=ami_eval
dataset=ami_eval_seg_th${sadthreshold}
data=tools_diar/data
ovpdataset=ami_eval_0.75s

nj=15

. utils/parse_options.sh || exit 1;

# Extract xvectors
win=1.5
period=0.75
if [ $win == "1.5" ]; then
  DEV_XVEC_DIR=tools_diar/exp_xvec/xvectors_${dataset}_${period}s
else
  DEV_XVEC_DIR=tools_diar/exp_xvec/xvectors_${dataset}_win${win}_${period}s
fi


sad_type=pyannote #silero #pyannote
start=`date +%s`
if [ $issad -eq 1 ];then
  if [ $sad_type == "pyannote" ];then
    if [ ! -f $data/${dataset}/segments ];then
    #if segments is not present
    for onset in $sadthreshold; do
      offset=$onset
    #   sad_exp=tools_diar/exp_sad/hyper_${dataset}_onset${onset}_offset${offset}_min_duration_on0.0554_min_duration_off0.0979_seg
      
      sad_decode_stage=1
      sad_python=/home/prachis/.conda/envs/pyannote/bin/python
      sad_dir=$data/${dataset}
      echo "$0: Applying SAD model to DEV/EVAL..."
      sad_model=$pyannote_pretrained_model
     
      vad_benchmarking/run_pyannote_SAD.sh \
        --nj $nj --stage $sad_decode_stage \
        --PYTHON $sad_python --eval_sad true \
        --onset $onset --offset $offset \
        $data/$dataset_org $sad_dir \
        $sad_model 
    done
    fi
  elif [ $sad_type == "silero" ];then
    sad_exp=tools_diar/exp_sad/${dataset}_silero_seg

    sad_decode_stage=4
    sad_python=python
    sad_dir=$data/${dataset}_silero_seg

     vad_benchmarking/run_silero_SAD.sh \
      --nj $njobs --stage $sad_decode_stage \
      --PYTHON $sad_python --eval_sad true \
      $data/$dataset $sad_exp 
  else
    echo "None of the condition met"
  fi
  # dataset=${dataset}_seg
fi

end=`date +%s`
echo pyannote SAD extraction Execution time was `expr $end - $start` seconds.

start=`date +%s`
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    services/test_extract_xvec.sh --dataset $dataset --data $data --njobs $nj --python $python \
    --stage $stage_extract --stop_stage $stop_stage_extract --win $win --period $period
fi
end=`date +%s`
echo xvector extraction Execution time was `expr $end - $start` seconds.
dataset=${dataset}_0.75s

if [ $overlap -eq 0 ];then
    # E2E_SHARC 
    start=`date +%s`
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        xvecmodelpath_pkl=/data1/prachis/Dihard_2020/gae-pytorch/gae/xvector_models/fbank_jhu_etdnn.pkl
        # ami_dev_fbank_0.75s
    
        segments_list=lists/${dataset}/segments_xvec
        reco2utt_list=lists/${dataset}/tmp/spk2utt
        featspath=$DEV_XVEC_DIR/subsegments_data/feats.scp
        xvecpath=$DEV_XVEC_DIR/
        labelspath=/data1/prachis/Dihard_2020/gae-pytorch/gae/tools_diar/ALL_GROUND_LABELS/${dataset}/threshold_0.5_avg/
        pldamodel=/data1/prachis/Dihard_2020/gae-pytorch/gae/lists/ami_sdm_train_gnd/plda_ami_sdm_train_gnd.pkl
        filegroupcount=1
        batch_size=4
        for epoch in 20;do
        traindataset=ami_sdm_train
        model_savepath=checkpoint/${traindataset}/${traindataset}_nonoverlap_sampler_3_PLDA_e2e_fulltrain_nonorm_filecount1_batchsize2/sharcinitk60_lr0.001/model_${epoch}_snapshot.pth

        filelist=lists/${dataset}/${dataset}.list
        rttm_ground_path=lists/${dataset}/filewise_rttms/
        segmentspath=lists/${dataset}/segments_xvec/
        echo $rttm_ground_path
        traink=30
        
        out_path=exp_sharc/results_with_${traindataset}_e2e_fulltrain_sharcinitk60_lr0.001_k${traink}/withnonorm/$dataset/labels_withoutglobalfeats_norm_${epoch}

        log_path=$out_path/log
        splitname=lists/${dataset}/split$nj

        mkdir -p $out_path
        mkdir -p $log_path
        echo $log_path

        JOB=14
        for k in 50; do
            for tau in 0.0; do
                echo "tau=$tau k=$k"
                echo "##################################"
                # $exec_cmd JOB=1:$nj $log_path/log.JOB.tau${tau}_k${k}.txt \
                #     $python test_subg_final_e2e.py \
                #     --mode "test,PLDA,rec_aff" \
                #     --featspath ${featspath} \
                #     --labelspath ${labelspath} \
                #     --feats_file $filelist \
                #     --out_path $out_path \
                #     --knn_k $k \
                #     --tau $tau --level 15 \
                #     --threshold prob --hidden 2048 --num_conv 1 \
                #     --batch_size 4096 --use_cluster_feat \
                #     --reco2utt_list $reco2utt_list \
                #     --segments_list $segments_list \
                #     --dataset_str $dataset \
                #     --xvecpath $xvecpath \
                #     --model_savepath $model_savepath  \
                #     --splitlist $splitname/JOB/full.list \
                #     --rttm_ground_path $rttm_ground_path \
                #     --segments $segmentspath \
                #     --which_python $python \
                #     --pldamodel $pldamodel \
                #     --fulltrain 1  
                # grep '#gt clusters' $log_path/log_tau${tau}_k${k}.txt
                # bash score.sh $out_path/final_k${k}_tau${tau}rttms/ lists/$dataset/rttm_val $python
                bash score_collar.sh $out_path/final_k${k}_tau${tau}rttms/ lists/$dataset/rttm_val $python
                # services/dscore-master/scorelib/md-eval.pl -s $out_path/final_k${k}_tau${tau}rttms/valrttm -r lists/$dataset/rttm_val
            done
        done
        done
        
    fi
    end=`date +%s`
    echo Esharc Execution time was `expr $end - $start` seconds.
else
    # E2E_SHARC-Ovp
    start=`date +%s`
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        
        xvecmodelpath_pkl=/data1/prachis/Dihard_2020/gae-pytorch/gae/xvector_models/fbank_jhu_etdnn.pkl
        segments_list=lists//${dataset}/segments_xvec
        reco2utt_list=lists/${dataset}/tmp/spk2utt
        featspath=$DEV_XVEC_DIR/subsegments_data/feats.scp
        xvecpath=$DEV_XVEC_DIR/
        # labelspath=/data1/prachis/Dihard_2020/gae-pytorch/gae/tools_diar/ALL_GROUND_LABELS/${dataset}/threshold_0.5_avg/
        labelspath=None
        pldamodel=/data1/prachis/Dihard_2020/gae-pytorch/gae/lists/ami_sdm_train_gnd/plda_ami_sdm_train_gnd.pkl 
        k_2ndpass=30

        # overlap file name obtained from pyannote model
        overlap_filename=pyannote_overlap/$ovpdataset/
        modestat=30
    
        filegroupcount=1
        batch_size=4
        overlap_th=0.0
        density_gap=0.0

        for epoch in 20;do
        traindataset=ami_sdm_train
    
        model_savepath=checkpoint/${traindataset}/${traindataset}_nonoverlap_sampler_3_PLDA_e2e_fulltrain_nonorm_filecount1_batchsize2/sharcinitk60_lr0.001/model_${epoch}_snapshot.pth

        filelist=lists/${dataset}/${dataset}.list
        rttm_ground_path=lists/${dataset}/filewise_rttms/
        segmentspath=lists/${dataset}/segments_xvec/
        echo $rttm_ground_path
        traink=30
        out_path=exp_sharc/results_with_${traindataset}_e2e_fulltrain_sharcinitk60_lr0.001_k${traink}/withnonorm/$dataset/labels_withoutglobalfeats_norm_${epoch}_2ndpass_modestat${modestat}_pyannote
        log_path=$out_path/log
        splitname=lists/${dataset}/split$nj

        mkdir -p $out_path
        mkdir -p $log_path
        echo $log_path
        JOB=4
        startfull=`date +%s`
        for k in 50; do
            for tau in 0.0; do
                echo "tau=$tau k=$k"
                echo "##################################"
                 $exec_cmd2 JOB=1:$nj $log_path/log.JOB.tau${tau}_k${k}.txt \
                    $python test_subg_final_prachi_sharc_e2e_2ndpass_overlap.py \
                    --mode "test,PLDA,rec_aff,v4_overlaponly,pyannote" \
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
                    --overlap_filename $overlap_filename \
                    --k_2ndpass $k_2ndpass \
                    --modestat $modestat 

                # bash score_noTNO.sh $out_path/final_k${k}_tau${tau}rttms/ $dataset
                bash score.sh $out_path/final_k${k}_tau${tau}_ovpth${overlap_th}_2ndpassk${k_2ndpass}_density_gap${density_gap}_overlaprttms/ lists/$dataset/rttm_val $python
                services/dscore-master/scorelib/md-eval.pl -s $out_path/final_k${k}_tau${tau}_ovpth${overlap_th}_2ndpassk${k_2ndpass}_density_gap${density_gap}_overlaprttms/valrttm -r lists/$dataset/rttm_val

            done
        done

        done
        end=`date +%s`
        echo E-SHARC-Ovp Execution time was `expr $end - $start` seconds.
    fi

fi

start=`date +%s`
# E2E_SHARC-Ovp VBx
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # generate rttm to labels
    threshold=0.5
    k=50
    tau=0.0
    traink=30
    traindataset=ami_sdm_train
    modestat=30
    epoch=20
    k_2ndpass=30
    overlap_th=0.0
    density_gap=0.0

    # E-SHARC-Ovp
    out_path=exp_sharc/results_with_${traindataset}_e2e_fulltrain_sharcinitk60_lr0.001_k${traink}/withnonorm/$dataset/labels_withoutglobalfeats_norm_${epoch}_2ndpass_modestat${modestat}_pyannote
    gt_rttm=$out_path/final_k${k}_tau${tau}_ovpth${overlap_th}_2ndpassk${k_2ndpass}_density_gap${density_gap}_overlaprttms/valrttm
    model=esharc
    modelth=$tau

    # Needed for VBx
    labels_dir=tools_diar/SHARC_models/$model/${dataset}/best_threshold${modelth}/threshold_${threshold}_avg
    overlap_filename=pyannote_overlap/$ovpdataset/
    # # avg segments
    segments=$DEV_XVEC_DIR/avg_segments
    if [ ! -d $labels_dir ];then
      python services/generate_groundtruth_label_sequence.py \
      --segmentsfile $segments \
      --labelsfiledir $labels_dir \
      --ground_truth_rttm $gt_rttm \
      --threshold $threshold
    fi
    # perform VBx
    PWD=`pwd`
    echo $exec_cmd_med2
    VBx/run_recipe_ami_final.sh --SET $dataset \
    --basedir $PWD --PYTHON $python \
    --TMP_DIR $DEV_XVEC_DIR \
    --labelspath $labels_dir \
    --overlap_filename $overlap_filename \
    --uem lists/$ovpdataset/all.uem \
    --cmd "$exec_cmd_med2" 
   
fi
end=`date +%s`
echo E-SHARC-Ovp+VBx Execution time was `expr $end - $start` seconds.
# done