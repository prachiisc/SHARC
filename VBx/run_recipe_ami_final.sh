#bin/bash!
# . ./cmd.sh
. ./path.sh


INSTRUCTION=VBx_withesharcoverlap
# INSTRUCTION=VBx_sharc_withsharcoverlap
trainset=ami_sdm_train

SCORE_DIR=/data1/prachis/Amrit_sharc/services/dscore-master
# ami_dev_fbank_0.75s
SET=ami_eval_fbank_0.75s

# TMP_DIR=/data1/prachis/Dihard_2020/VBx/JHU_TDNN/VBx/xvectors/xvectors_$SET
TMP_DIR=/data1/prachis/Amrit_sharc/tools_diar/exp_xvec/xvectors_$SET

PLDA_FOLD=fbank_jhu_plda_amitrain
kaldi_recipe_path=/data1/prachis/SRE_19/Diarization_scores/swbd_diar/
basedir=/data1/prachis/Amrit_sharc/github/SHARC_check
labelspath=$basedir/tools_diar/SHARC_models/esharc/$SET/threshold_0.5_avg/
overlap_filename=/data1/prachis/Amrit_sharc/pyannote_overlap/$SET
cmd=$exec_cmd_med
uem=
PYTHON=/home/prachis/.conda/envs/Hilander1/bin/python

. utils/parse_options.sh || exit 1;

echo uem=$uem
echo cmd=$cmd
DSET=$SET
data_dir=$basedir/lists/

#SHARC
if [[ $INSTRUCTION = "all" ]] || [[ $INSTRUCTION = "VBx_sharc_withsharcoverlap" ]]; then
	OUT_main=out_dir_sharc_fulltrain_withsharck60init/$SET

	alpha=1.0
	nj=15
	thr=0.0
	tareng=0.3
	smooth=5.0
	lda_dim=220
	
	Fa=0.4
	Fb=11
	JOB=1                          # best till Fa=0.4, Fb=11, thr=0.0,loop=0.8,iters=5
	for k in 60;do

 	# overlap_filename=/data1/prachis/Amrit_sharc/pyannote_overlap/$SET
	traink=60
	trainset=ami_sdm_train

	for loopP in 0.8; do
    for max_iters in 1 ; do 
	thr=0.0
	tau=$thr

	OUT_DIR=${OUT_main}/VBx_with_sharcoverlapinit_rttms_th${thr}_Fa${Fa}_Fb${Fb}_loopP${loopP}_maxiters${max_iters}
    # $cmd JOB=1:$nj $OUT_DIR/log/VBx.JOB.log \
	# $PYTHON VBx/diarization_PLDAadapt_AHCxvec_BHMMxvec_sharcOverlapinit.py \
	# 					--alpha $alpha \
	# 					--threshold $thr \
	# 					--target_energy $tareng \
	# 					--init_smoothing $smooth \
	# 					--lda_dim $lda_dim \
	# 					--Fa $Fa \
	# 					--Fb $Fb \
	# 					--LoopP $loopP \
	# 					--max_iters $max_iters \
	# 					--use_VB_withoverlap \
	# 					--knn_k $k \
	# 					--tau $tau \
	# 					--overlap_filename $overlap_filename \
	# 					--labelspath $labelspath \
	#  					$OUT_DIR \
	#  					$TMP_DIR/xvector.JOB.ark \
	#  					$TMP_DIR/segments \
    #              	 	$PLDA_FOLD/mean.vec \
	#  					$PLDA_FOLD/transform.mat \
	#  					$PLDA_FOLD/plda \
	#  					$PLDA_FOLD/plda 

    echo "no collar +  overlaps"
	python $SCORE_DIR/score.py \
		-u $data_dir/$DSET/all.uem \
		-r $data_dir/$DSET/filewise_rttms/*.rttm \
		-s $OUT_DIR/final_k${k}_tau${tau}_withoverlap_VBxrttms/*.rttm >  $OUT_DIR/final_k${k}_tau${tau}_withoverlap_VBxrttms/der_ovp.txt 2>err.txt
		echo "$OUT_DIR/final_k${k}_tau${tau}_withoverlap_VBxrttms/"
		grep  OVERALL $OUT_DIR/final_k${k}_tau${tau}_withoverlap_VBxrttms/der_ovp.txt
    
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


# E-SHARC-OVERLAP init
if [[ $INSTRUCTION = "all" ]] || [[ $INSTRUCTION = "VBx_withesharcoverlap" ]]; then
	OUT_main=out_dir_sharc_e2e_fulltrain_withsharck60init/$SET

	alpha=1.0
	nj=15
	thr=0.0
	tareng=0.3
	smooth=5.0
	lda_dim=220
	
	Fa=0.4
	Fb=11
	JOB=1                          # best till Fa=0.4, Fb=11, thr=0.0,loop=0.8,iters=5
	k=50
	tau=0.0
 	# overlap_filename=/data1/prachis/Amrit_sharc/pyannote_overlap/$SET

	for loopP in 0.6 0.8; do
    for max_iters in 1; do 
	for thr in 0.0; do
	tau=$thr
	JOB=12
	
	OUT_DIR=${OUT_main}/VBx_with_esharcoverlapinit_rttms_th${thr}_Fa${Fa}_Fb${Fb}_loopP${loopP}_maxiters${max_iters}
    # $cmd JOB=1:$nj $OUT_DIR/log/VBx.JOB.log \
	# $PYTHON VBx/diarization_PLDAadapt_AHCxvec_BHMMxvec_sharcOverlapinit.py \
	# 					--alpha $alpha \
	# 					--threshold $thr \
	# 					--target_energy $tareng \
	# 					--init_smoothing $smooth \
	# 					--lda_dim $lda_dim \
	# 					--Fa $Fa \
	# 					--Fb $Fb \
	# 					--LoopP $loopP \
	# 					--max_iters $max_iters \
	# 					--use_VB_withoverlap \
	# 					--knn_k $k \
	# 					--tau $tau \
	# 					--overlap_filename $overlap_filename \
	# 					--labelspath $labelspath \
	#  					$OUT_DIR \
	#  					$TMP_DIR/xvector.JOB.ark \
	#  					$TMP_DIR/segments \
    #              	 	$PLDA_FOLD/mean.vec \
	#  					$PLDA_FOLD/transform.mat \
	#  					$PLDA_FOLD/plda \
	#  					$PLDA_FOLD/plda 


    echo "no collar + overlaps"
	python $SCORE_DIR/score.py \
	    -u $uem \
		-r $data_dir/$DSET/filewise_rttms/*.rttm \
		-s $OUT_DIR/final_k${k}_tau${tau}_withoverlap_VBxrttms/*.rttm >  $OUT_DIR/final_k${k}_tau${tau}_withoverlap_VBxrttms/der_ovp.txt 2>err.txt
		echo "$OUT_DIR/final_k${k}_tau${tau}_withoverlap_VBxrttms/"
		grep  OVERALL $OUT_DIR/final_k${k}_tau${tau}_withoverlap_VBxrttms/der_ovp.txt
    
	echo "collar + no overlaps"
	python $SCORE_DIR/score.py \
	    --collar 0.25 --ignore_overlaps \
	    -r $data_dir/$DSET/filewise_rttms/*.rttm \
	    -s $OUT_DIR/final_k${k}_tau${tau}_withoverlap_VBxrttms/*.rttm >  $OUT_DIR/final_k${k}_tau${tau}_withoverlap_VBxrttms/der_collar_novp.txt 2>err.txt
	    echo "$OUT_DIR"
	    grep  OVERALL $OUT_DIR/final_k${k}_tau${tau}_withoverlap_VBxrttms/der_collar_novp.txt

	# cat $OUT_DIR/final_k${k}_tau${tau}_withoverlap_VBxrttms/*.rttm > $OUT_DIR/final_k${k}_tau${tau}_withoverlap_VBxrttms/valrttm
	# $SCORE_DIR/scorelib/md-eval.pl -s $OUT_DIR/final_k${k}_tau${tau}_withoverlap_VBxrttms/valrttm -r lists/$DSET/rttm_val

	#exit
	done
	done
	done

fi
