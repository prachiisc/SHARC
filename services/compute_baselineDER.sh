#/bin/bash
. ./cmd.sh
# . ./cmd_gpu.sh
# . ./cmd_long.sh
. ./path.sh

stage=1

if [ $stage -eq 1 ];then
  clustering=spectral
  # clustering=ahc
  # dataset=vox_diar
  # pldatype=libvox #ami, dihard, libvox
  # nj=40

  # dataset=ami_dev_fbank_0.75s
  # dataset=ami_dev_0.75s
  dataset=ami_eval_0.75s

  nj=15
  pldatype=ami
  # outputfold=exp/results_plda${pldatype}_${clustering}_baseline/$dataset/
  # xvecpath=/data1/prachis/Dihard_2020/gae-pytorch/gae/tools_diar/xvectors_0.75s_npy/$dataset
  
  outputfold=tools_wespk/exp_wespk/results_plda${pldatype}_${clustering}_baseline/$dataset/
  xvecpath=tools_wespk/exp_wespk/xvectors_${dataset}/
  
  mkdir -p $outputfold
  echo $outputfold
  epochs=40
  log=log


  overlap=1

  which_python=/home/prachis/.conda/envs/Hilander1/bin/python
  which_file=compute_baseline_clustering_results.py
  pldamodel=lists_wespk/ami_sdm_train_gnd/plda_ami_sdm_train_gnd.pkl
  lists=lists_wespk
  # pref='_baseline_widePLDA_'
  scale=10
  pref='_baseline_scaled10_amiPLDA_'
 
  for th in  0.7; do
  pref=_baseline_scaled10_amiPLDA_threshold${th}
  # pref=_baseline_widePLDA_threshold${th}
  JOB=1

  $exec_cmd JOB=1:$nj $outputfold/$log/GAE.JOB.log \
    $which_python $which_file \
      --outf $outputfold \
      --ngpu -1 \
      --dataset_str $dataset \
      --clustering $clustering \
      --useoverlap $overlap \
      --splitpath split$nj/JOB/full.list \
      --reco2utt_list $lists/$dataset/tmp/spk2utt \
      --reco2num_list $lists/$dataset/tmp/reco2num_spk \
      --segments $lists/$dataset/segments_xvec/ \
      --pref $pref \
      --scale $scale \
      --th $th \
      --PLDA $pldatype \
      --xvecpath $xvecpath \
      --which_python $which_python \
      --pldamodel $pldamodel

    echo $pref $th
    bash score.sh $outputfold/final_${clustering}${pref}rttms/ $lists/$dataset/rttm $which_python
  done
fi

if [ $stage -eq 2 ];then

  dataset=displace_dev_fbank_29thmay_seg_0.25s
  xvecpath=tools_diar/exp_xvec/xvectors_baselinedisplace_diarization_nnet_1a_dev_fbank_spectral_laplacian_scaling10/xvectors_0.25/xvector.scp
  outputfold=tools_diar/exp_xvec/xvectors_baselinedisplace_diarization_nnet_1a_dev_fbank_spectral_laplacian_scaling10/processed_xvectorfeats/
  # filename=B021
  # filename=B022
  # filename=M032
  # filename=M037
  
  which_python=/home/prachis/.conda/envs/Hilander1/bin/python
  which_file=compute_baseline_clustering_results.py
  
  mkdir -p $outputfold
  $which_python $which_file \
  --outf $outputfold \
  --ngpu -1 \
  --dataset_str $dataset \
  --reco2utt_list lists/$dataset/tmp/spk2utt \
  --reco2num_list lists/$dataset/tmp/reco2num_spk \
  --segments lists/$dataset/segments_xvec/ \
  --xvecpath $xvecpath \
  --which_python $which_python \
  --filename $filename \
  --extract_feats

fi
