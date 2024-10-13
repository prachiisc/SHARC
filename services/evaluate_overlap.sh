. ./cmd.sh
. ./path.sh
stage=6
eval_sad=true
dset=ami_dev_fbank_0.75s
nj=15

#####################################
# Evaluate SAD output.
#####################################
if [ $stage -eq 3  -a  $eval_sad = "true" ]; then
  echo "$0: Scoring SAD output on FULL DEV set..."
  services/score_sad.py \
    --n-jobs $nj --collar 0.0 \
    -u lists/$dset/overlaps/all.uem \
    lists/$dset/overlaps/segments \
    /data1/prachis/Amrit_sharc/pyannote_overlap/$dset/segments \
    lists/$dset/overlaps/recordings.tbl
  echo ""
fi

if [ $stage -eq 4  -a  $eval_sad = "true" ]; then
  echo "$0: Scoring SAD output on FULL EVAL set..."
  local/segmentation/score_sad.py \
    --n-jobs $nj --collar 0.0 \
    data/displace_$dset/segments \
    data/displace_${dset}_seg/segments \
    $evaldatapath/docs/recordings.tbl
  echo ""
fi

# Voxconverse set
nj=40
dset=vox_diar
if [ $stage -eq 5  -a  $eval_sad = "true" ]; then
  echo "$0: Scoring SAD output on FULL DEV set..."
  services/score_sad.py \
    --n-jobs $nj --collar 0.0 \
    -u lists/$dset/overlaps/all.uem \
    lists/$dset/overlaps/segments \
    /data1/prachis/Amrit_sharc/pyannote_overlap/$dset/segments \
    lists/$dset/overlaps/recordings.tbl
  echo ""
fi

dset=vox_diar
# considering only more than 1 spk files
if [ $stage -eq 6  -a  $eval_sad = "true" ]; then
  echo "$0: Scoring SAD output on More than 1 spk DEV set..."
  services/score_sad.py \
    --n-jobs $nj --collar 0.0 \
    -u lists/$dset/overlaps/sub.uem \
    lists/$dset/overlaps/segments_sub \
    /data1/prachis/Amrit_sharc/pyannote_overlap/$dset/segments_sub \
    lists/$dset/overlaps/recordings_sub.tbl
  echo ""
fi

