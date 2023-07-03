#ref=/data1/prachis/Amrit_sharc/lists/ami_dev_fbank_0.75s/rttm_val.rttm
#sys=$1/valrttm
#outpath=$1
#perl diarization/md-eval.pl -r $ref -s $sys > $outpath/sad.txt
#cat $outpath/sad.txt


ref=/data1/prachis/Amrit_sharc/lists/ami_dev_fbank_0.75s/overlaprttm
sys=$1/valrttm
outpath=$1
uem=lists/ami_dev_fbank_0.75s/overlaps/all.uem
perl diarization/md-eval.pl -r $ref -s $sys -u $uem > $outpath/overlap.txt
cat $outpath/overlap.txt

#ref=/data1/prachis/Amrit_sharc/lists/ami_dev_fbank_0.75s/overlaps_rttms/rttm
#sys=$1/valrttm
#outpath=$1
#perl diarization/md-eval.pl -r $ref -s $sys > $outpath/overlap.txt
#cat $outpath/overlap.txt





