# run from /data1/prachis/Amrit_sharc/lists/database_pynannote

dataset=AMI
database_yml=/data1/prachis/Amrit_sharc/lists/database_pynannote/database.yml
# for python
# import os
# os.environ["PYANNOTE_DATABASE_CONFIG"] = "/data1/prachis/pyannote-audio/tutorials/AMI-diarization-setup/pyannote/database.yml"

# for shell
export PYANNOTE_DATABASE_CONFIG="/data1/prachis/Amrit_sharc/lists/database_pynannote/database.yml"

if [ $dataset == "AMI" ]; then
    # AMI
    hypothesis=../../exp_sharc/results_with_ami_sdm_train/ami_dev_fbank_0.75s/labels_withoutglobalfeats_norm_full_400_lr0.001_traink60/final_k60_tau0.0rttms/valrttm
    outpath=ami_sharc_gcn

    epochs=400
    #hypothesis=../../exp_sharc/results_with_ami_sdm_train/ami_dev_fbank_0.75s/labels_withoutglobalfeats_norm_full_${epochs}_lr0.001_traink60_proc/dropout0.2_temporal_beta0.95_neb5/final_k70_tau0.0rttms/valrttm
    #outpath=amisharc_gcn_proc0.2_temporalnb5_0.95_k70


    #hypothesis=../../exp_sharc/results_with_ami_sdm_train/ami_dev_fbank_0.75s/labels_withoutglobalfeats_norm_full_100_lr0.001_traink60_modified/dropout0.0_temporal_beta0.95_neb3_without_init_2ndpass_overlap_myapproach_withprobpred_v3/trained_withv3approach_pyannote_frameoverlaps/final_k60_tau0.0_ovpth0.0_2ndpassk30_density_gap0.0_overlaprttms/valrttm
    #outpath=amisharc_2ndpass_v3pyannote_framelevel.der
    # pyannote-metrics diarization --subset development AMI.SpeakerDiarization.SDM $hypothesis > results_ICASSP2023/$outpath
    
    #pyannote-metrics diarization --subset development AMI.SpeakerDiarization.SDM $hypothesis > results/$outpath
    
    # Baseline AHC
    # hypothesis=/data1/prachis/Dihard_2020/gae-pytorch/gae/exp_baselines/results_pldaAhc_baseline/ami_dev_fbank_0.75s/final_ahc_baseline_AmiPLDA_threshold0.25rttms/valrttm
    # outpath=ami_ahc_baseline.der
    # outpath_novp=ami_ahc_baseline.der_novp
    
    # Baseline SC
    # hypothesis=/data1/prachis/Dihard_2020/gae-pytorch/gae/exp_baselines/results_pldaSpectral_baseline/ami_dev_fbank_0.75s/final_spectral_baseline_scaled10_AmiPLDA_threshold0.7_rttms/valrttm
    # outpath=ami_spec_baseline.der
    # outpath_novp=ami_spec_baseline.der_novp
    

    # SHARC
    hypothesis=/data1/prachis/Dihard_2020/gae-pytorch/gae/exp_sharc/results_ami/ami_dev_0.75s/valrttm
    outpath=ami_sharc.der
    outpath_novp=ami_sharc.der_novp

    pyannote-metrics diarization --subset development AMI.SpeakerDiarization.SDM $hypothesis > results_ICASSP2023/$outpath
    
    pyannote-metrics diarization --subset development  --collar=0.25 --skip-overlap AMI.SpeakerDiarization.SDM $hypothesis > results_ICASSP2023/$outpath_novp


elif [ $dataset == "Voxconverse" ]; then
    ####################################################################################################################################################################
    # Voxconverse

    # baseline AHC 
    hypothesis=/data1/prachis/Dihard_2020/gae-pytorch/gae/exp_baselines/results_pldalibvox_ahc_baseline/vox_diar/final_ahc_baseline_widePLDA_threshold0.4rttms/valrttm
    outpath=vox_diar_ahc_baseline.der
    outpath_novp=vox_diar_ahc_baseline.der_novp

    # baseline spectral
    # hypothesis=/data1/prachis/Dihard_2020/gae-pytorch/gae/exp_baselines/results_pldalibvox_spectral_baseline/vox_diar/final_spectral_baseline_scaled10_widePLDA_threshold0.7rttms/valrttm
    # outpath=vox_diar_spec_baseline.der
    # outpath_novp=vox_diar_spec_baseline.der_novp

    #SHARC
    hypothesis=/data1/prachis/Amrit_sharc/exp_sharc/results_with_libriplda_trained_libri/vox_diar/labels_wihtoutglobalfeats_norm_full/final_k30_tau0.5rttms/valrttm
    outpath=vox_diar_sharc.der
    outpath_novp=vox_diar_sharc.der_novp

    # E2E-SHARC
    # hypothesis=/data1/prachis/Amrit_sharc/exp_sharc/results_with_vox_diar_e2e/fulltrain/labels_wihtoutglobalfeats_norm_numpyxvec_epochs20/final_k30_tau0.9rttms/valrttm
    # outpath=vox_diar_e2e_sharc.der
    # outpath_novp=vox_diar_e2e_sharc.der_novp

    mkdir -p results_ICASSP2023

    # without collar and with overlap
    pyannote-metrics diarization --subset development Voxconverse.SpeakerDiarization.vox $hypothesis > results_ICASSP2023/$outpath
    
    # with collar and without overlap
    pyannote-metrics diarization --subset development  --collar=0.25 --skip-overlap Voxconverse.SpeakerDiarization.vox $hypothesis > results_ICASSP2023/$outpath_novp

else
    echo "$dataset not available"
fi
