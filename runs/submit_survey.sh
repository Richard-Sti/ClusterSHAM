#!/bin/bash

nthreads=14
memory=4
node="berg"
fpath="/usr/bin/python3 run_survey.py"
path="NSA_SERSIC_survey.toml"
#path="NSA_ELPETRO_survey.toml"
conversion=-0.8120578088224437
#conversion=0.32482312352897746

for sub_id in "N2" "N3" "N4"
#for sub_id in "N1"
do
    string="addqueue -q $node -n 1x$nthreads -s -m $memory $fpath --path $path --sub_id $sub_id --nthreads $nthreads --conversion $conversion"
    $string
    echo $string
    sleep 3s
done

# for nd_type in "BMF"
# do
#     for i in 3
#     do
#         string="addqueue -q berg -n $nthreads -m $memory $path --name $name --config $config --nd_type $nd_type --bin_index $i --file_index $file_index"
#         $string &
#         echo $string
#         sleep 3s
#     done
# done




# python3 run_survey.py --path NSA_ELPETRO_ABSMAG_survey.toml --sub_id 'N1' --nthreads 2 --conversion -0.8120578088224437
