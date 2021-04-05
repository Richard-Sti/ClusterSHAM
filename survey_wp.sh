#!/bin/bash

nthreads=28
memory=7
#name="matched"
#config="matched_config.toml"

#for nd_type in "BMF" "HIMF" "SMF" "LF"
#do
#    for i in 0 1 2 3
#    do
#        string="addqueue -s -q berg -n 1x$nthreads -m $memory /usr/bin/python3 survey_wp.py --name $name --config $config --nd_type $nd_type --bin_index $i --nthreads $nthreads"
#        $string
#        echo $string
#        sleep 5s
#    done
#done

name="NSAmatch_ELPETRO"
config="NYUconfig.toml"

for nd_type in "LF"
do
    for scope in "-21.0 -20.0" "-20.0 -19.0" "-19.0 -18.0"
    do
        string="addqueue -s -q berg -n 1x$nthreads -m $memory /usr/bin/python3 survey_wp.py --name $name --config $config --nd_type $nd_type --scope $scope --nthreads $nthreads"
        $string
        echo $string
        sleep 5s
    done
done
