# dataset=ami_dev_fbank_0.75s
dataset=$1
#path=/data1/prachis/Dihard_2020/gae-pytorch/gae/exp_march/results_spec_sup_ae_norm_cosinesoftmaxloss_angleproto_clean_0_xvec0.75shift_norm_PLDA_scaled/_avg_accumgradient_use_gnd_adj_withadjplda/ami_dev_fbank_0.75s/results_sup_pic_widePLDA/final_pic_knnpldainit50_sup_affine_threshold0.7_K30_z0.1_nb5_beta0.95_Model40rttms
path=$2
#cat $path/*.rttm > $path/valrttm

rm -f $path/reco2num_spk
cat lists/$dataset/$dataset.list | while read i; do
    numspk=`grep $i $path/${i}.rttm | awk '{print $8}' | sort | uniq | wc -l`
    echo "$i $numspk">> $path/reco2num_spk
done
