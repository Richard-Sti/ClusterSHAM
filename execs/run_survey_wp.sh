#!/bin/bash

echo "#cores to request"
read ncores

string="addqueue -s -q berg -n 1x$ncores -m 4 /usr/bin/python3 run_survey_wp.py"

$string
