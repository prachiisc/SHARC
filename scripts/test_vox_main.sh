#bin/bash
. ./cmd.sh
. ./path.sh

# This script performs following for voxconverse dev/eval sets dataset
# VAD using pyannote when issad=1,
# The following are the stages
# stage=0,stop_stage=0 performs x-vector extraction. don't repeat this step again once extracted.
# overlap=0, stage=1, stop_stage=1: performs ESHARC (no overlap handling)
# overlap=1, stage=1, stop_stage=1: performs ESHARC+ ESHARC-Ovp (overlap handling using pyannote)
# stage=2, stop_stage=2: performs VBx using ESHARC-Ovp labels

stage=0
stop_stage=2
issad=1 # computing sad 
overlap=1 # pyannote overlap output is available

ovpdataset=vox_diar_test
overlapversion=_2.1
# extract x-vectors
stage_extract=1
stop_stage_extract=7

python=/home/prachis/.conda/envs/Hilander1/bin/python
pyannote_pretrained_model=vad_benchmarking/VAD_model/pytorch_model.bin
# 0.6 0.5 0.8 0.9 0.4 0.3 done
# 0.5 is the best
# false should be less than or equal to miss
# pyannote 2.1 onset=0.767
for sadthreshold in 0.767; do #pyannote onset and offset
dataset_org=vox_diar_test
dataset=vox_diar_test_seg_th${sadthreshold}
data=tools_diar/data

nj=40

. utils/parse_options.sh || exit 1;


# sad_type=pyannote #silero #pyannote
sad_type="pyannote_2_1"

start=`date +%s`
if [ $issad -eq 1 ];then
  if [ $sad_type == "pyannote" ];then
    if [[ (! -d $data/${dataset}) || (! -f $data/${dataset}/segments) ]];then
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
  elif [ $sad_type == "pyannote_2_1" ];then
    echo "2.1"
    #if segments is not present
    min_duration_on=0.182 
    min_duration_off=0.501
    onset=$sadthreshold
    offset=0.713
    dataset=${dataset}_off${offset}
    
    if [[ (! -d $data/${dataset}) || (! -f $data/${dataset}/segments) ]];then
        sad_decode_stage=3
        sad_python=/home/prachis/.conda/envs/pyannote/bin/python
        sad_dir=$data/${dataset}
        echo "$0: Applying SAD model to DEV/EVAL..."
        sad_model=$pyannote_pretrained_model
        
        vad_benchmarking/run_pyannote_SAD.sh \
            --nj $nj --stage $sad_decode_stage \
            --PYTHON $sad_python --eval_sad true \
            --onset $onset --offset $offset \
            --min_duration_on $min_duration_on \
            --min_duration_off $min_duration_off \
            --outputdir "pyannote_vad_2.1" \
            $data/$dataset_org $sad_dir \
            $sad_model 
    fi
  else
    echo "None of the condition met"
  fi
  # dataset=${dataset}_seg
fi


end=`date +%s`
echo pyannote SAD extraction Execution time was `expr $end - $start` seconds.

# Extract xvectors
win=1.5
period=0.75
if [ $win == "1.5" ]; then
  DEV_XVEC_DIR=tools_diar/exp_xvec/xvectors_${dataset}_${period}s
else
  DEV_XVEC_DIR=tools_diar/exp_xvec/xvectors_${dataset}_win${win}_${period}s
fi
start=`date +%s`
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    services/test_extract_xvec.sh --dataset $dataset --data $data --njobs $nj --python $python \
    --stage $stage_extract --stop_stage $stop_stage_extract --win $win --period $period
fi
end=`date +%s`
echo xvector extraction Execution time was `expr $end - $start` seconds.

dataset=${dataset}_0.75s

if [ $overlap -eq 0 ];then
    # E-SHARC : no overlap handling
    start=`date +%s`
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        xvecmodelpath_pkl=xvector_model/fbank_jhu_etdnn.pkl
        traindataset=librivox
        segments_list=lists/${dataset}/segments_xvec
        reco2utt_list=lists/${dataset}/tmp/spk2utt
        featspath=$DEV_XVEC_DIR/subsegments_data/feats.scp
        xvecpath=$DEV_XVEC_DIR/
        labelspath=tools_diar/ALL_GROUND_LABELS/${dataset}/threshold_0.5_avg/
        pldamodel=plda_models/lib_vox_tr_all_gnd_0.75s/plda_lib_vox_tr_all_gnd_0.75s.pkl

        filegroupcount=1
        batch_size=4
      
        epochs=20
        model_savepath=checkpoint_pre_trained/voxconverse/esharc/${traindataset}_nonoverlap_sampler_6_PLDA_e2e_fulltrain.pth_${epochs}_snapshot.pth
        filelist=lists/${dataset}/${dataset}.list
        rttm_ground_path=lists/${dataset}/filewise_rttms/
        segmentspath=lists/${dataset}/segments_xvec/
        echo $rttm_ground_path
        traink=60
        
        out_path=exp_sharc/results_with_${dataset}_e2e/fulltrain_epochs${epochs}/labels_${epochs}

        log_path=$out_path/log
        splitname=lists/${dataset}/split$nj

        mkdir -p $out_path
        mkdir -p $log_path
        echo $log_path

        JOB=1
        for k in 30; do
            for tau in 0.8; do
                echo "tau=$tau k=$k"
                echo "##################################"
                $exec_cmd JOB=1:$nj $log_path/log.JOB.tau${tau}_k${k}.txt \
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
                
                bash score_collar.sh $out_path/final_k${k}_tau${tau}rttms/ lists/$dataset/rttm_val $python
                # services/dscore-master/scorelib/md-eval.pl -s $out_path/final_k${k}_tau${tau}rttms/valrttm -r lists/$dataset/rttm_val

            done
        done

        
        
    fi
    end=`date +%s`
    echo Esharc Execution time was `expr $end - $start` seconds.
else
    start=`date +%s`
    # E-SHARC-Ovp: with overlap handling
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        traindataset=librivox
        xvecmodelpath_pkl=xvector_model/fbank_jhu_etdnn.pkl
        segments_list=lists/${dataset}/segments_xvec
        reco2utt_list=lists/${dataset}/tmp/spk2utt
        featspath=$DEV_XVEC_DIR/subsegments_data/feats.scp
        xvecpath=$DEV_XVEC_DIR/
        # model_filename=checkpoint_amrit/librivox_nonoverlap_sampler_6_PLDA.pth
        # labelspath=/data1/prachis/Dihard_2020/gae-pytorch/gae/tools_diar/ALL_GROUND_LABELS/${dataset}/threshold_0.5_avg/
        labelspath=None
        pldamodel=plda_models/lib_vox_tr_all_gnd_0.75s/plda_lib_vox_tr_all_gnd_0.75s.pkl
        k_2ndpass=30

        # overlap file name obtained from pyannote model
        overlap_filename=pyannote_overlap$overlapversion/$ovpdataset/
        modestat=30
    
        filegroupcount=1
        batch_size=4
        overlap_th=0.0
        density_gap=0.0
        epochs=20
        
        model_savepath=checkpoint_pre_trained/voxconverse/esharc/${traindataset}_nonoverlap_sampler_6_PLDA_e2e_fulltrain.pth_${epochs}_snapshot.pth
        filelist=lists/${dataset}/${dataset}.list
        rttm_ground_path=lists/${dataset}/filewise_rttms/
        segmentspath=lists/${dataset}/segments_xvec/
        echo $rttm_ground_path
        traink=60
        out_path=exp_sharc/results_with_${traindataset}_e2e_fulltrain_sharcinitk60_lr0.001_k${traink}/withnonorm/$dataset/labels_withoutglobalfeats_norm_${epochs}_2ndpass_modestat${modestat}_pyannote$overlapversion
        log_path=$out_path/log
        splitname=lists/${dataset}/split$nj

        mkdir -p $out_path
        mkdir -p $log_path
        echo $log_path
        JOB=16
        startfull=`date +%s`
        for k in 30; do
            for tau in 0.8; do
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

                
                # echo "with overlap collar"
                bash score_collar.sh $out_path/final_k${k}_tau${tau}_ovpth${overlap_th}_2ndpassk${k_2ndpass}_density_gap${density_gap}_overlaprttms/ lists/$dataset/rttm_val $python
                # services/dscore-master/scorelib/md-eval.pl -s $out_path/final_k${k}_tau${tau}_ovpth${overlap_th}_2ndpassk${k_2ndpass}_density_gap${density_gap}_overlaprttms/valrttm -r lists/$dataset/rttm_val 
            done
        done

        
        end=`date +%s`
        echo Total Execution time was `expr $end - $startfull` seconds.
    fi
    end=`date +%s`
    echo Esharc overlap Execution time was `expr $end - $start` seconds.


fi
done
# E2E_SHARC-Ovp VBx
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # generate rttm to labels
    threshold=0.5
    k=30
    tau=0.8
    traink=60
    traindataset=librivox
    modestat=30
    epochs=20
    k_2ndpass=30
    overlap_th=0.0
    density_gap=0.0

    # E-SHARC-Ovp
    out_path=exp_sharc/results_with_${traindataset}_e2e_fulltrain_sharcinitk60_lr0.001_k${traink}/withnonorm/$dataset/labels_withoutglobalfeats_norm_${epochs}_2ndpass_modestat${modestat}_pyannote
    gt_rttm=$out_path/final_k${k}_tau${tau}_ovpth${overlap_th}_2ndpassk${k_2ndpass}_density_gap${density_gap}_overlaprttms/valrttm
    
    model=esharc_overlap_pyannote$overlapversion
    modelth=$tau

    # Needed for VBx
    labels_dir=tools_diar/SHARC_models/$model/${dataset}/best_threshold${modelth}/threshold_${threshold}_avg
    overlap_filename=pyannote_overlap$overlapversion/$ovpdataset/
    # # avg segments
    #generate labels
    segments=$DEV_XVEC_DIR/avg_segments
    if [ ! -d $labels_dir ];then
    # if 1;then
        python services/generate_groundtruth_label_sequence.py \
        --segmentsfile $segments \
        --labelsfiledir $labels_dir \
        --ground_truth_rttm $gt_rttm \
        --threshold $threshold
    fi
    # perform VBx
    PWD=`pwd`
    echo $exec_cmd_med2
    VBx/run_recipe_vox_final.sh --SET $dataset \
    --basedir $PWD --PYTHON $python \
    --TMP_DIR $DEV_XVEC_DIR \
    --labelspath $labels_dir \
    --overlap_filename $overlap_filename \
    --cmd "$exec_cmd_med2" 
   
fi