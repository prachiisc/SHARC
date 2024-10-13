. ./cmd.sh
. ./path.sh

eval_sad=false
PYTHON=python

nj=27
stage=1
# default
min_duration_on=0.0554
min_duration_off=0.0979
onset=0.5
offset=0.5

. utils/parse_options.sh



datasetpath=$1 #/data1/shareefb/track2_cluster/data/$dataset
path_new_kaldi_segs=$2 #/data1/shareefb/track2_cluster/data/$kaldi_dataset
dataset=`basename $datasetpath` #displace_dev_fbank
kaldi_dataset=`basename $path_new_kaldi_segs` #displace_pyannote_dev_fbank_seg
pyannote_pretrained_model=$3 #vad_benchmarking/VAD_model/pytorch_model.bin
echo $path_new_kaldi_segs, $kaldi_dataset


utils/split_data.sh $datasetpath $nj
outputdir=silero_vad/${dataset}/
mkdir -p $outputdir
echo "dataset=$dataset, kaldi_dataset=$kaldi_dataset"
if [ $stage -le 1 ]; then
    JOB=1
    echo "######################################################################"
    $exec_cmd_med2 JOB=1:$nj $outputdir/log/sad.JOB.log \
        $PYTHON vad_benchmarking/VAD.py \
        --in-audio=$datasetpath/split$nj/JOB/wav.scp \
        --in-VAD=silero_VAD \
        --dataset $dataset \
        --tuning \
        --outputpath $outputdir 

    echo "######################################################################"
                    
fi

#####################################
# Generate SAD output in segments format.
#####################################
if [ $stage -le 2 ]; then
    echo "$0: convert to kaldi style segments ..."

    # generate pyannote segments 
    echo utils.py --vad_dir_path $outputdir --vad_type silero
    $PYTHON vad_benchmarking/utils.py --vad_dir_path $outputdir --vad_type silero

    # convert to kaldi style segments
    vad_benchmarking/run_kaldi_seg.sh $outputdir $kaldi_dataset $path_new_kaldi_segs/filewise_segments
    cat $path_new_kaldi_segs/filewise_segments/*.segments > $path_new_kaldi_segs/segments
             
fi
####################################################
echo copying wav.scp and creating utt2spk and spk2utt from segments folder
####################################################
if [ $stage -le 3 ]; then
    cp $datasetpath/wav.scp $path_new_kaldi_segs/.
    cp $datasetpath/rttm $path_new_kaldi_segs/
    awk '{print $1,$2}'  $path_new_kaldi_segs/segments >  $path_new_kaldi_segs/utt2spk
    utils/utt2spk_to_spk2utt.pl $path_new_kaldi_segs/utt2spk > $path_new_kaldi_segs/spk2utt
fi

#####################################
# Evaluate SAD output.
#####################################
if [ $stage -eq 4  -a  $eval_sad = "true" ]; then
    # if [ ! -f $datasetpath/recordings.tbl ];then
    #     echo "uri\tdomain" > $datasetpath/recordings.tbl
    #     awk '{print $1,"meeting"}' $datasetpath/$dataset.list >> $datasetpath/recordings.tbl
    # fi
    #  -u tools_diar/data/$dataset/all.uem \
    $PYTHON services/score_sad.py \
        --n-jobs $nj --collar 0.0 \
        $datasetpath/segments \
        $path_new_kaldi_segs/segments \
        $datasetpath/recordings.tbl
    echo ""
fi