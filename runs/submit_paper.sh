#!/bin/bash
nthreads=12
memory=7
node="cmb"
fpath="/usr/bin/python3 run_paper.py"
sample="NYU"
attr="ABSMAG"

path="${sample}_${attr}_AM.toml"
beta=0.1
Ninit=1
Nmcmc=20
checkpoint_kind=1

for sub_id in "N2"
do
    if [ $checkpoint_kind -eq 0 ]
    then
       comm="addqueue -s -q $node -n 1x$nthreads -m $memory $fpath --path $path --sub_id $sub_id --Ninit $Ninit --Nmcmc $Nmcmc --nthreads $nthreads --beta $beta"
       echo $comm
       $comm
    elif [ $checkpoint_kind -eq 1 ]
    then
       checkpoint="./out/${attr}_${sub_id}/checkpoint.z"
       comm="addqueue -s -q $node -n 1x$nthreads -m $memory $fpath --path $path --sub_id $sub_id --Ninit $Ninit --Nmcmc $Nmcmc --nthreads $nthreads --checkpoint $checkpoint --beta $beta"
       echo $comm
       $comm
    elif [ $checkpoint_kind -eq 2 ]
    then
       checkpoint="./temp/checkpoint_${attr}_${sub_id}.z"
       comm="addqueue -s -q $node -n 1x$nthreads -m $memory $fpath --path $path --sub_id $sub_id --Ninit $Ninit --Nmcmc $Nmcmc --nthreads $nthreads --checkpoint $checkpoint --beta $beta"
       echo $comm
       $comm
    else
       echo "Invalid checkpoint for ${sub_Id}"
       break
    fi
    sleep 3s
done
