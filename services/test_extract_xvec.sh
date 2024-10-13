#/bin/bash
. ./cmd.sh
. ./path.sh


dataset=ami_eval_fbank
njobs=15

data=tools_diar/data

################
win=1.5
period=0.75
################

fullpath=/data1/prachis/Dihard_2020/gae-pytorch/gae/tools_diar/
fbankdir=tools_diar/fbank

nnet_dir=$fullpath/exp/xvector_nnet_1a_tdnn_fbank

########### STAGE ##############################
stage=1
stop_stage=100

python=/home/prachis/.conda/envs/Hilander1/bin/python

. utils/parse_options.sh || exit 1;
echo $python
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # feats extraction
  for name in $dataset; do
    steps/make_fbank.sh --fbank-config conf/fbank_16k.conf --nj $njobs \
      --cmd "$exec_cmd_med" --write-utt2num-frames true \
      $data/$name tools_diar/exp_feats/make_fbank $fbankdir
    # cp $data_dir/rttm data/${name}/rttm
    utils/fix_data_dir.sh $data/$name
  done
fi
echo stage=$stage
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then

  for name in $dataset; do
   local/nnet3/xvector/prepare_feats.sh \
        --nj $njobs --cmd "$exec_cmd_med" \
        $data/$name $data/${name}_cmn tools_diar/exp_xvec/${name}_cmn
    if [ -f $data/$name/vad.scp ]; then
        echo "vad.scp found .. copying it"
        cp $data/$name/vad.scp $data/${name}_cmn/
    fi
    if [ -f $data/$name/segments ]; then
        echo "Segments found .. copying it"
        cp $data/$name/segments $data/${name}_cmn/
    fi
    utils/fix_data_dir.sh $data/${name}_cmn
  done
fi

if [ $win == "1.5" ]; then
  DEV_XVEC_DIR=tools_diar/exp_xvec/xvectors_${dataset}_${period}s
else
  DEV_XVEC_DIR=tools_diar/exp_xvec/xvectors_${dataset}_win${win}_${period}s
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Extracting x-vectors..."
  cmn_dir=$data/${dataset}_cmn
  with_gpu=false
  if $with_gpu; then
      diarization/nnet3/xvector/extract_xvectors.sh \
    --cmd "$train_cmd" --nj $njobs --use-gpu true \
    --window $win --period $period --apply-cmn false \
    $nnet_dir \
    $cmn_dir $DEV_XVEC_DIR
  else
    diarization/nnet3/xvector/extract_xvectors.sh \
    --cmd "$exec_cmd_med" --nj $njobs \
    --window $win --period $period --apply-cmn false \
    $nnet_dir \
    $cmn_dir $DEV_XVEC_DIR
  fi
  echo "X-vector extraction finished for DEV. See $DEV_XVEC_DIR/log for logs."
 
fi

if [ $win == "1.5" ];then
  dataset2=${dataset}_${period}s
else
  dataset2=${dataset}_win${win}_${period}s
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    SSC_fold=./
    
    rm -f $data/$dataset/reco2num_spk
    ## cat $data/$dataset/${dataset}.list | while read i; do
    awk '{print $1}' $data/$dataset/wav.scp | while read i; do
      numspk=`grep $i $data/$dataset/rttm | awk '{print $8}' | sort | uniq | wc -l`
      echo "$i $numspk" >> $data/$dataset/reco2num_spk
    done

    # copy spk2utt,utt2spk, segments in lists folder required for training
    srcdir=$DEV_XVEC_DIR  # path of xvectors.scp
    mkdir -p $SSC_fold/lists/$dataset2/tmp
    cp $srcdir/spk2utt $SSC_fold/lists/$dataset2/tmp/spk2utt
    cp $srcdir/utt2spk $SSC_fold/lists/$dataset2/tmp/utt2spk
    cp $srcdir/segments $SSC_fold/lists/$dataset2/tmp/segments
    cp $data/$dataset/reco2num_spk $SSC_fold/lists/$dataset2/reco2num_spk
    cp $data/$dataset/reco2num_spk $SSC_fold/lists/$dataset2/tmp/reco2num_spk

    awk '{print $1}' $srcdir/spk2utt > $SSC_fold/lists/$dataset2/${dataset2}.list
    cp $SSC_fold/lists/$dataset2/$dataset2.list $SSC_fold/lists/$dataset2/tmp/dataset.list
    # store segments filewise in folder segments_xvec
    mkdir -p $SSC_fold/lists/$dataset2/segments_xvec
    cat $SSC_fold/lists/$dataset2/${dataset2}.list | while read i; do
        grep $i $SSC_fold/lists/$dataset2/tmp/segments > $SSC_fold/lists/$dataset2/segments_xvec/${i}.segments
    done    
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  python services/make_contiguous_segments.py $DEV_XVEC_DIR/segments $DEV_XVEC_DIR/avg_segments
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  threshold=0.5
  gt_rttm=$data/$dataset/rttm
  # gt_rttm=lists/$dataset2/rttm

  segments=$DEV_XVEC_DIR/segments
  labels_dir=tools_diar/ALL_GROUND_LABELS/${dataset2}/threshold_${threshold}

  # avg segments
  segments=$DEV_XVEC_DIR/avg_segments
  labels_dir=tools_diar/ALL_GROUND_LABELS/${dataset2}/threshold_${threshold}_avg
  python services/generate_groundtruth_label_sequence.py \
  --segmentsfile $segments \
  --labelsfiledir $labels_dir \
  --ground_truth_rttm $gt_rttm \
  --threshold $threshold

fi


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    SSC_fold=./
    nj=$njobs

    totalfiles=`cat $SSC_fold/lists/$dataset2/${dataset2}.list | wc -l`
    filelen=`expr $totalfiles - 1`
    for i in `seq 0 $filelen`; do
        echo $i >>  $SSC_fold/lists/$dataset2/full.list
    done
    $python services/split_list.py $SSC_fold/lists/$dataset2/ $totalfiles $nj
    sort -R $SSC_fold/lists/$dataset2/full.list > $SSC_fold/lists/$dataset2/shuffled_list.txt
    cp $data/$dataset/rttm $SSC_fold/lists/$dataset2/rttm_val

    mkdir -p $SSC_fold/lists/$dataset2/filewise_rttms
    cat $SSC_fold/lists/$dataset2/${dataset2}.list | while read i; do
        grep $i $SSC_fold/lists/$dataset2/rttm_val > $SSC_fold/lists/$dataset2/filewise_rttms/${i}.rttm
    done

    awk '{print $8}' $SSC_fold/lists/$dataset2/rttm_val | sort | uniq >  $SSC_fold/lists/$dataset2/spkall.list
fi



# # generate labels for AMI SHARC/E-SHARC-OVERLAP rttm
# if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then

#   threshold=0.5

#   # E-SHARC
#   gt_rttm=/data1/prachis/Amrit_sharc/exp_sharc/results_with_ami_sdm_train_e2e_fulltrain_sharcinitk60_lr0.001_k30/withnonorm/$dataset2/labels_withoutglobalfeats_norm_20_2ndpass_modestat30_pyannote/final_k50_tau0.0_ovpth0.0_2ndpassk30_density_gap0.0_overlaprttms/valrttm
#   model=esharc
#   modelth=0.0
#   labels_dir=tools_diar/SHARC_models/$model/${dataset2}/best_threshold${modelth}/threshold_${threshold}_avg

#   # avg segments
#   segments=$DEV_XVEC_DIR/avg_segments
#   python services/generate_groundtruth_label_sequence.py \
#   --segmentsfile $segments \
#   --labelsfiledir $labels_dir \
#   --ground_truth_rttm $gt_rttm \
#   --threshold $threshold

# fi
