path=$1
dataset=$2
groundpath=lists/$dataset/rttm
#rm -rf $path/val.rttm
cat lists/$dataset/${dataset}_noTNO.list | while read i; do
    cat $path/${i}.rttm >> $path/valrttm_noTNO
    #grep $i $groundpath >> ${groundpath}_noTNO
done
python services/dscore-master/score.py -r ${groundpath}_noTNO -s $path/valrttm_noTNO > $path/der_noTNO.txt 2> err.txt
grep OVERALL $path/der_noTNO.txt

python services/dscore-master/score.py --ignore_overlaps --collar 0.25 -r ${groundpath}_noTNO -s $path/valrttm_noTNO > $path/der_novp_noTNO.txt 2> err.txt
grep OVERALL $path/der_novp_noTNO.txt
