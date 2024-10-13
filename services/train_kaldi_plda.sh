#/bin/bash
. ./path.sh
. ./cmd.sh


stage=4
# xvectorscp_dir=tools_wespk/exp_wespk/xvectors_ami_sdm_train_gnd_0.75s
xvectorscp_dir=exp_sharc/results_with_ami_sdm_train_e2e_fulltrain_sharcinitk60_lr0.001_k30/withnonorm/ami_sdm_train/plda_xvectors_20
if [ $stage -le 1 ]; then
  echo "$0: Computing mean of xvectors"
  $exec_cmd_med $xvectorscp_dir/log/mean.log \
    ivector-mean scp:$xvectorscp_dir/xvector.scp $xvectorscp_dir/mean.vec || exit 1;
fi

if [ $stage -le 2 ]; then
  if [ -z "$pca_dim" ]; then
    pca_dim=-1
  fi
  echo "$0: Computing whitening transform"
  $exec_cmd_med $xvectorscp_dir/log/transform.log \
    est-pca --read-vectors=true --normalize-mean=false \
      --normalize-variance=true --dim=$pca_dim \
      scp:$xvectorscp_dir/xvector.scp $xvectorscp_dir/transform.mat || exit 1;
fi

# Train PLDA models
if [ $stage -le 3 ]; then
  transform_dir=$xvectorscp_dir
  plda_dir=$xvectorscp_dir
  # Train a PLDA model on VoxCeleb/AMI train, using DIHARD 2018 development set to whiten.
  echo "training PLDA model"
  $exec_cmd $plda_dir/log/plda.log \
    ivector-compute-plda ark:$xvectorscp_dir/spk2utt \
      "ark:ivector-subtract-global-mean \
      scp:$xvectorscp_dir/xvector.scp ark:- \
      | transform-vec $transform_dir/transform.mat ark:- ark:- \
      | ivector-normalize-length ark:- ark:- |" \
    $plda_dir/plda || exit 1;

    # copying to a common directory
    # cp $plda_dir/plda plda_models_wespk/ami_sdm_train_gnd/plda
    mkdir -p plda_models_esharc/ami_sdm_train_gnd/
    cp $plda_dir/plda plda_models_esharc/ami_sdm_train_gnd/plda
fi


if [ $stage -le 4 ]; then
  
    # convert to kaldi plda to pkl format
    kaldi_feats_path=$xvectorscp_dir
    # pldamodel=lists_wespk/ami_sdm_train_gnd/plda_ami_sdm_train_gnd.pkl
    pldamodel=lists_esharc/ami_sdm_train_gnd/plda_ami_sdm_train_gnd.pkl
    dataset=ami_sdm_train_gnd
    mkdir -p lists_esharc/ami_sdm_train_gnd
    python services/convert_kaldi_to_pkl.py \
    --kaldi_feats_path $kaldi_feats_path \
    --dataset $dataset \
    --output_dir ./ \
    --lists lists_esharc

  # cp $pldamodel plda_models_wespk/ami_sdm_train_gnd/
  mkdir -p plda_models_esharc/ami_sdm_train_gnd/
  cp $pldamodel plda_models_esharc/ami_sdm_train_gnd/

fi