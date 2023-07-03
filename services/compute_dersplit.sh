#dataset=ami_dev_fbank_0.75s
dataset=$2
ref=/data1/prachis/Amrit_sharc/lists/$dataset/rttm_val.rttm
sys=$1/valrttm
outpath=$1
perl diarization/md-eval.pl -r $ref -s $sys > $outpath/der.txt
cat $outpath/der.txt


#ref=/data1/prachis/Amrit_sharc/lists/ami_dev_fbank_0.75s/filewise_rttms/AMI_ES2011b_SDM.rttm
#sys=$1/AMI_ES2011b_SDM.rttm
#outpath=$1
#perl diarization/md-eval.pl -v -r $ref -s $sys -m > $outpath/der1.txt
#cat $outpath/der1.txt
