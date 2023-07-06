path=$1
dataset=$2
groundpath=lists/$dataset/rttm
#rm -rf $path/val.rttm
cat $path/*.rttm > $path/valrttm
python services/dscore-master/score.py -r $groundpath -s $path/valrttm > $path/der.txt 2> err.txt
grep OVERALL $path/der.txt

python services/dscore-master/score.py --ignore_overlaps --collar 0.25 -r $groundpath -s $path/valrttm > $path/der_novp.txt 2> err.txt
grep OVERALL $path/der_novp.txt
