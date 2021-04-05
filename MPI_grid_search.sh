#!/bin/bash

nthreads=100
memory=10
file_index=0
name="matched"
config="matched_config.toml"
path="/usr/bin/python3 MPI_grid_search.py"

for nd_type in "BMF"
do
    for i in 3
    do
        string="addqueue -q berg -n $nthreads -m $memory $path --name $name --config $config --nd_type $nd_type --bin_index $i --file_index $file_index"
        $string &
        echo $string
        sleep 3s
    done
done
