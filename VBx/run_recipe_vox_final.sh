. ./cmd.sh
. ./path.sh

# INSTRUCTION=VBx_withoverlap # all or features or xvectors or VBx or score
# INSTRUCTION=VBx_withoverlap_sharc

# Intialize with SHARC/E-SHARC-Overlap
# INSTRUCTION=VBx_sharc_withsharcoverlap
INSTRUCTION=VBx_withesharcoverlap

# INSTRUCTION=VBx_withoverlap_displace
# INSTRUCTION=VBx_withoverlap_displace_sharc
# INSTRUCTION=VBx_withoverlap_displace_sharc_withsharcoverlap
# INSTRUCTION=VBx_withesharcoverlap_displace

#SET=dev # dev or eval
# vox_diar_test
SET=vox_diar_test

DSET=$SET
period=0.75
SCORE_DIR=/data1/prachis/Amrit_sharc/services/dscore-master

# TMP_DIR=/data1/prachis/Dihard_2020/gae-pytorch/gae/tools_diar/exp/xvector_nnet_1a_tdnn_fbank/xvectors_${SET}_${period}s
TMP_DIR=/data1/prachis/Amrit_sharc/tools_diar/exp_xvec/xvectors_${SET}_${period}s/

PLDA_FOLD=fbank_jhu_plda_libvoxtrain
kaldi_recipe_path=/data1/prachis/SRE_19/Diarization_scores/swbd_diar/
basedir=/data1/prachis/Amrit_sharc/github/SHARC_check
labelspath=$basedir/tools_diar/SHARC_models/esharc/$SET/threshold_0.5_avg/
PYTHON=/home/prachis/.conda/envs/Hilander1/bin/python

overlap_filename=pyannote_overlap/$SET
cmd=$exec_cmd_med
uem=

. utils/parse_options.sh || exit 1;


echo uem=$uem
echo cmd=$cmd
DSET=$SET
data_dir=$basedir/lists/

# E-SHARC 
if [[ $INSTRUCTION = "all" ]] || [[ $INSTRUCTION = "VBx_withoverlap" ]]; then
	alpha=1.0
	nj=40
	thr=0.0
	tareng=0.3
	smooth=5.0
	lda_dim=220
	
	Fa=0.4
	Fb=11
	JOB=1                          # best till Fa=0.4, Fb=11, thr=0.8/0.9,loop=0.99,iters=10
	k=30

 	overlap_filename=/data1/prachis/Amrit_sharc/pyannote_overlap/$SET
	JOB=1
 	# 0.5 0.6 0.7 0.8
	for loopP in 0.9 0.99; do
    for max_iters in 5; do 
	for thr in 0.9; do
    tau=$thr
	labelspath=$basedir/exp_sharc/results_with_${SET}_e2e/fulltrain/labels_wihtoutglobalfeats_norm_numpyxvec_epochs20/final_k${k}_tau${thr}rttms
	OUT_DIR=${OUT_main}/VBx_withoverlap_rttms_th${thr}_Fa${Fa}_Fb${Fb}_loopP${loopP}_maxiters${max_iters}
	mkdir -p $OUT_DIR 
    $train_cmd JOB=1:$nj $OUT_DIR/log/VBx.JOB.log \
	$PYTHON diarization_PLDAadapt_AHCxvec_BHMMxvec_sharcinit.py \
	 					--alpha $alpha \
						--threshold $thr \
						--target_energy $tareng \
						--init_smoothing $smooth \
						--lda_dim $lda_dim \
						--Fa $Fa \
						--Fb $Fb \
						--LoopP $loopP \
						--max_iters $max_iters \
						--use_VB_withoverlap \
						--knn_k $k \
						--tau $tau \
						--overlap_filename $overlap_filename \
						--labelspath $labelspath \
	 					$OUT_DIR \
	 					$TMP_DIR/xvector.JOB.ark \
	 					$TMP_DIR/segments \
                 	 	$PLDA_FOLD/mean.vec \
	 					$PLDA_FOLD/transform.mat \
	 					$PLDA_FOLD/plda \
	 					$PLDA_FOLD/plda 

	# python $SCORE_DIR/score.py \
	#     --collar 0.25 --ignore_overlaps \
	#     -r $data_dir/$DSET/filewise_rttms/*.rttm \
	#     -s $OUT_DIR/*.rttm >  $OUT_DIR/der.txt 2>err.txt
	#     echo "$OUT_DIR"
	#      grep  OVERALL $OUT_DIR/der.txt

	python $SCORE_DIR/score.py \
			-r $data_dir/$DSET/filewise_rttms/*.rttm \
			-s $OUT_DIR/final_k${k}_tau${tau}_withoverlap_VBxrttms/*.rttm >  $OUT_DIR/final_k${k}_tau${tau}_withoverlap_VBxrttms/der_ovp.txt 2>err.txt
			echo "$OUT_DIR/final_k${k}_tau${tau}_withoverlap_VBxrttms/"
			grep  OVERALL $OUT_DIR/final_k${k}_tau${tau}_withoverlap_VBxrttms/der_ovp.txt

	#exit
	done
	done
	done
fi


# E-SHARC-overlap init 
if [[ $INSTRUCTION = "all" ]] || [[ $INSTRUCTION = "VBx_withesharcoverlap" ]]; then
	OUT_main=out_dir_sharc_e2e_fulltrain_withsharck60init/${SET}_${period}s
	alpha=1.0
	nj=40
	thr=0.0
	tareng=0.3
	smooth=5.0
	lda_dim=220
	
	Fa=0.4
	Fb=11
	JOB=1                          # best till Fa=0.4, Fb=11, thr=0.8/0.9,loop=0.99,iters=10
	k=30

 	
	JOB=1
 	# 0.5 0.6 0.7 0.8
	for loopP in  0.8; do
    for max_iters in 2; do 
	for thr in 0.8; do
    tau=$thr

	OUT_DIR=${OUT_main}/VBx_with_esharcoverlapinit_rttms_th${thr}_Fa${Fa}_Fb${Fb}_loopP${loopP}_maxiters${max_iters}
	mkdir -p $OUT_DIR 
    $cmd JOB=1:$nj $OUT_DIR/log/VBx.JOB.log \
	$PYTHON VBx/diarization_PLDAadapt_AHCxvec_BHMMxvec_sharcOverlapinit.py \
	 					--alpha $alpha \
						--threshold $thr \
						--target_energy $tareng \
						--init_smoothing $smooth \
						--lda_dim $lda_dim \
						--Fa $Fa \
						--Fb $Fb \
						--LoopP $loopP \
						--max_iters $max_iters \
						--use_VB_withoverlap \
						--knn_k $k \
						--tau $tau \
						--overlap_filename $overlap_filename \
						--labelspath $labelspath \
	 					$OUT_DIR \
	 					$TMP_DIR/xvector.JOB.ark \
	 					$TMP_DIR/segments \
                 	 	$PLDA_FOLD/mean.vec \
	 					$PLDA_FOLD/transform.mat \
	 					$PLDA_FOLD/plda \
	 					$PLDA_FOLD/plda 

    echo "no collar + no overlaps"		
    python $SCORE_DIR/score.py \
			-r $data_dir/$DSET/filewise_rttms/*.rttm \
			-s $OUT_DIR/final_k${k}_tau${tau}_withoverlap_VBxrttms/*.rttm >  $OUT_DIR/final_k${k}_tau${tau}_withoverlap_VBxrttms/der_ovp.txt 2>err.txt
			echo "$OUT_DIR/final_k${k}_tau${tau}_withoverlap_VBxrttms/"
	
			
			grep  OVERALL $OUT_DIR/final_k${k}_tau${tau}_withoverlap_VBxrttms/der_ovp.txt
    
	echo "collar + overlaps"
	python $SCORE_DIR/score.py \
	    --collar 0.25  \
	    -r $data_dir/$DSET/filewise_rttms/*.rttm \
	    -s $OUT_DIR/final_k${k}_tau${tau}_withoverlap_VBxrttms/*.rttm >  $OUT_DIR/final_k${k}_tau${tau}_withoverlap_VBxrttms/der_collar_ovp.txt 2>err.txt
	    echo "$OUT_DIR"
		
	    grep  OVERALL $OUT_DIR/final_k${k}_tau${tau}_withoverlap_VBxrttms/der_collar_ovp.txt
    
	echo "collar + no overlaps"
	python $SCORE_DIR/score.py \
	    --collar 0.25 --ignore_overlaps \
	    -r $data_dir/$DSET/filewise_rttms/*.rttm \
	    -s $OUT_DIR/final_k${k}_tau${tau}_withoverlap_VBxrttms/*.rttm >  $OUT_DIR/final_k${k}_tau${tau}_withoverlap_VBxrttms/der_collar_novp.txt 2>err.txt
	    echo "$OUT_DIR"
	    grep  OVERALL $OUT_DIR/final_k${k}_tau${tau}_withoverlap_VBxrttms/der_collar_novp.txt

	
	#exit
	done
	done
	done
fi


#SHARC
if [[ $INSTRUCTION = "all" ]] || [[ $INSTRUCTION = "VBx_withoverlap_sharc" ]]; then
	OUT_main=out_dir_sharc_fulltrain_withsharck60init/${SET}_${period}s
	alpha=1.0
	nj=40
	thr=0.0
	tareng=0.3
	smooth=5.0
	lda_dim=220
	
	Fa=0.4
	Fb=11
	JOB=1                          # best till Fa=0.4, Fb=11, thr=0.8/0.9,loop=0.99,iters=10
	k=30

 	overlap_filename=/data1/prachis/Amrit_sharc/pyannote_overlap/$SET
	JOB=1
 	# 0.5 0.6 0.7 0.8
	for loopP in 0.9; do
    for max_iters in 10; do 
	for thr in 0.5 0.8; do
    tau=$thr
	# SHARC

	labelspath=$basedir/exp_sharc/results_with_libriplda_trained_libri/${SET}/labels_withoutglobalfeats_norm_full_repeat9_nochange_tau_dataset_mustrunmodified/final_k${k}_tau${thr}rttms
	OUT_DIR=${OUT_main}/VBx_withoverlap_rttms_th${thr}_Fa${Fa}_Fb${Fb}_loopP${loopP}_maxiters${max_iters}
	mkdir -p $OUT_DIR 
    $train_cmd JOB=1:$nj $OUT_DIR/log/VBx.JOB.log \
	$PYTHON diarization_PLDAadapt_AHCxvec_BHMMxvec_sharcinit.py \
	 					--alpha $alpha \
						--threshold $thr \
						--target_energy $tareng \
						--init_smoothing $smooth \
						--lda_dim $lda_dim \
						--Fa $Fa \
						--Fb $Fb \
						--LoopP $loopP \
						--max_iters $max_iters \
						--use_VB_withoverlap \
						--knn_k $k \
						--tau $tau \
						--overlap_filename $overlap_filename \
						--labelspath $labelspath \
	 					$OUT_DIR \
	 					$TMP_DIR/xvector.JOB.ark \
	 					$TMP_DIR/segments \
                 	 	$PLDA_FOLD/mean.vec \
	 					$PLDA_FOLD/transform.mat \
	 					$PLDA_FOLD/plda \
	 					$PLDA_FOLD/plda 

	# python $SCORE_DIR/score.py \
	#     --collar 0.25 --ignore_overlaps \
	#     -r $data_dir/$DSET/filewise_rttms/*.rttm \
	#     -s $OUT_DIR/*.rttm >  $OUT_DIR/der.txt 2>err.txt
	#     echo "$OUT_DIR"
	#      grep  OVERALL $OUT_DIR/der.txt

	python $SCORE_DIR/score.py \
			-r $data_dir/$DSET/filewise_rttms/*.rttm \
			-s $OUT_DIR/final_k${k}_tau${tau}_withoverlap_VBxrttms/*.rttm >  $OUT_DIR/final_k${k}_tau${tau}_withoverlap_VBxrttms/der_ovp.txt 2>err.txt
			echo "$OUT_DIR/final_k${k}_tau${tau}_withoverlap_VBxrttms/"
			grep  OVERALL $OUT_DIR/final_k${k}_tau${tau}_withoverlap_VBxrttms/der_ovp.txt

	#exit
	done
	done
	done
fi

#SHARC-overlap init 
if [[ $INSTRUCTION = "all" ]] || [[ $INSTRUCTION = "VBx_sharc_withsharcoverlap" ]]; then 
	OUT_main=out_dir_sharc_fulltrain_withsharck60init/${SET}_${period}s
	alpha=1.0
	nj=40
	thr=0.0
	tareng=0.3
	smooth=5.0
	lda_dim=220
	
	Fa=0.4
	Fb=11
	JOB=1                          # best till Fa=0.4, Fb=11, thr=0.8/0.9,loop=0.99,iters=10
	k=30

 	overlap_filename=/data1/prachis/Amrit_sharc/pyannote_overlap/$SET
	JOB=1
 	# 0.5 0.6 0.7 0.8
	for loopP in 0.5 0.6 0.7 0.8 0.9; do
    for max_iters in 1 2 3 4 6 7 8 9 10; do 
	for thr in 0.5; do
    tau=$thr
	# SHARC-overlap init

	# labelspath=$basedir/exp_sharc/results_with_lib_vox_tr_all/lib_vox_cv_all/vox_diar_test_labels_withoutglobalfeats_norm_full_120_lr0.01_traink60_modified/dropout0.0_without_init_2ndpass_overlap_myapproach_withprobpred_v4/trained_withv4approach_withavglabels_usingmodel1_intracluster_per0.1_pyannote_framewise_mode15/final_k${k}_tau${thr}_ovpth0.0_2ndpassk30_density_gap0.0_overlaprttms
	
	labelspath=$basedir/tools_diar/SHARC_models/sharc/${SET}_${period}s/threshold_0.5_avg
	OUT_DIR=${OUT_main}/VBx_with_sharcoverlapinit_rttms_th${thr}_Fa${Fa}_Fb${Fb}_loopP${loopP}_maxiters${max_iters}
	
	mkdir -p $OUT_DIR 
    $train_cmd JOB=1:$nj $OUT_DIR/log/VBx.JOB.log \
	$PYTHON diarization_PLDAadapt_AHCxvec_BHMMxvec_sharcOverlapinit.py \
	 					--alpha $alpha \
						--threshold $thr \
						--target_energy $tareng \
						--init_smoothing $smooth \
						--lda_dim $lda_dim \
						--Fa $Fa \
						--Fb $Fb \
						--LoopP $loopP \
						--max_iters $max_iters \
						--use_VB_withoverlap \
						--knn_k $k \
						--tau $tau \
						--overlap_filename $overlap_filename \
						--labelspath $labelspath \
	 					$OUT_DIR \
	 					$TMP_DIR/xvector.JOB.ark \
	 					$TMP_DIR/segments \
                 	 	$PLDA_FOLD/mean.vec \
	 					$PLDA_FOLD/transform.mat \
	 					$PLDA_FOLD/plda \
	 					$PLDA_FOLD/plda 

	# python $SCORE_DIR/score.py \
	#     --collar 0.25 --ignore_overlaps \
	#     -r $data_dir/$DSET/filewise_rttms/*.rttm \
	#     -s $OUT_DIR/*.rttm >  $OUT_DIR/der.txt 2>err.txt
	#     echo "$OUT_DIR"
	#      grep  OVERALL $OUT_DIR/der.txt

	python $SCORE_DIR/score.py \
			-r $data_dir/$DSET/filewise_rttms/*.rttm \
			-s $OUT_DIR/final_k${k}_tau${tau}_withoverlap_VBxrttms/*.rttm >  $OUT_DIR/final_k${k}_tau${tau}_withoverlap_VBxrttms/der_ovp.txt 2>err.txt
			echo "$OUT_DIR/final_k${k}_tau${tau}_withoverlap_VBxrttms/"
			grep  OVERALL $OUT_DIR/final_k${k}_tau${tau}_withoverlap_VBxrttms/der_ovp.txt

	#exit
	done
	done
	done
fi



if [[ $INSTRUCTION = "all" ]] || [[ $INSTRUCTION = "score" ]]; then
	SCORE_DIR=dscore-master # directory with scoring tool: https://github.com/nryant/dscore
	Fa=0.3
	Fb=10
	
	for loopP in 0.0; do
    for max_iters in 40; do
    for thr in -0.5 ; do
    OUT_DIR=${OUT_main}/rttms_th${thr}_Fa${Fa}_Fb${Fb}_loopP${loopP}_maxiters${max_iters}
	python $SCORE_DIR/score.py \
		--collar 0.0 \
		-u $DATA_DIR/data/uem_scoring/full/all.uem \
		-r $DATA_DIR/data/rttm/*.rttm \
		-s $OUT_DIR/*.rttm >  $OUT_DIR/der.txt 2>err.txt
	echo "$OUT_DIR"
	grep  OVERALL $OUT_DIR/der.txt
done
done
done

fi
