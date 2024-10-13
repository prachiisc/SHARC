. ./cmd.sh
. ./path.sh

# dataset=ami_sdm_eval

#DEV set
# dataset=ami_sdm_dev
# dataset=ami_dev_fbank
# kaldi_dataset=ami_dev_fbank_seg

dataset=ami_eval_fbank
kaldi_dataset=ami_eval_fbank_seg

eval_sad=true

nj=15
stage=2
# default
min_duration_on=0.0554
min_duration_off=0.0979
echo "dataset=$dataset, kaldi_dataset=$kaldi_dataset"
if [ $stage -eq 1 ]; then
    # onset=0.3, offset=0.3
    # onset=0.4, offset=0.4
    # for audio in $(ls /home1/somil/language_diarization/Displace_data/eval_1/*/*.wav);  
    for onset in 0.3 0.4; do
    #  for offset in 0.4 0.5 0.6; do
    #    for min_duration_on in 0.05 0.10 0.25 0.30; do
    #        for min_duration_off in 0.25 0.30; do
                offset=$onset
                hyper=hyper_onset${onset}_offset${offset}_min_duration_on${min_duration_on}_min_duration_off${min_duration_off}
                outputdir=pyannote_vad/${dataset}/${hyper} 
                mkdir -p $outputdir

                JOB=2
                # awk '{print $2}' /data1/prachis/Amrit_sharc/tools_diar/data/$dataset/split$nj/JOB/wav.scp | while read audio
                # do 
                echo "######################################################################"
                # echo "audio=$audio"
                $exec_cmd_med JOB=1:$nj $outputdir/log/sad.JOB.log \
                    python /data1/prachis/Amrit_sharc/vad_benchmarking/VAD.py \
                    --in-audio=/data1/prachis/Amrit_sharc/tools_diar/data/$dataset/split$nj/JOB/wav.scp \
                    --in-VAD=Pyannote_VAD \
                    --dataset $dataset \
                    --tuning \
                    --onset $onset \
                    --offset $offset \
                    --min_duration_on $min_duration_on \
                    --min_duration_off $min_duration_off
                    
                # done
                echo "######################################################################"
                    
                
    #  done
     
    done
fi

#####################################
# Evaluate SAD output.
#####################################
if [ $stage -le 2  -a  $eval_sad = "true" ]; then
  echo "$0: Scoring SAD output on FULL set using uem ..."
  for onset in 0.3 0.4; do
#      for offset in 0.2 0.3 0.4 0.5 0.6; do
    #    for min_duration_on in 0.05 0.10 0.25 0.30; do
    #        for min_duration_off in 0.25 0.30; do
                offset=$onset
                hyper=hyper_onset${onset}_offset${offset}_min_duration_on${min_duration_on}_min_duration_off${min_duration_off}
                outputdir=pyannote_vad/${dataset}/${hyper} 
                path_new_kaldi_segs=/data1/prachis/Amrit_sharc/tools_diar/data/$kaldi_dataset/$hyper
                echo $hyper

                # generate pyannote segments 
                python vad_benchmarking/utils.py --vad_dir_path $outputdir
                # convert to kaldi style segments
                vad_benchmarking/run_kaldi_seg.sh $outputdir $kaldi_dataset $hyper
                cat $path_new_kaldi_segs/filewise_segments/*.segments > $path_new_kaldi_segs/segments
            
                services/score_sad.py \
                    --n-jobs $nj --collar 0.0 \
                    -u tools_diar/data/$dataset/all.uem \
                    tools_diar/data/$dataset/segments \
                    $path_new_kaldi_segs/segments \
                    tools_diar/data/$dataset/recordings.tbl
                echo ""
                
      
      
  done  
fi


# default parameters
 hyper=hyper_AMIdefault
 echo $hyper
if [ $stage -eq 3 ]; then
    
    outputdir=pyannote_vad/${dataset}/${hyper} 
    mkdir -p $outputdir

    JOB=2
    # awk '{print $2}' /data1/prachis/Amrit_sharc/tools_diar/data/$dataset/split$nj/JOB/wav.scp | while read audio
    # do 
    echo "######################################################################"
    # echo "audio=$audio"
    $exec_cmd_med JOB=1:$nj $outputdir/log/sad.JOB.log \
        python /data1/prachis/Amrit_sharc/vad_benchmarking/VAD.py \
        --in-audio=/data1/prachis/Amrit_sharc/tools_diar/data/$dataset/split$nj/JOB/wav.scp \
        --in-VAD=Pyannote_VAD \
        --dataset $dataset
        
    # done
    echo "######################################################################"
    
fi

#####################################
# Evaluate SAD output.
#####################################
if [ $stage -eq 4  -a  $eval_sad = "true" ]; then
  echo "$0: Scoring SAD output on FULL set..."
    outputdir=pyannote_vad/${dataset}/${hyper} 
    path_new_kaldi_segs=/data1/prachis/Amrit_sharc/tools_diar/data/$kaldi_dataset/$hyper
    # generate pyannote segments 
    python vad_benchmarking/utils.py --vad_dir_path $outputdir
    # convert to kaldi style segments
    vad_benchmarking/run_kaldi_seg.sh $outputdir $kaldi_dataset $hyper
    cat $path_new_kaldi_segs/filewise_segments/*.segments > $path_new_kaldi_segs/segments

    services/score_sad.py \
        --n-jobs $nj --collar 0.0 \
        -u tools_diar/data/$dataset/all.uem \
        tools_diar/data/$dataset/segments \
        $path_new_kaldi_segs/segments \
        tools_diar/data/$dataset/recordings.tbl
    echo ""

fi