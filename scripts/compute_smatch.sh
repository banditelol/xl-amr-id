#!/bin/bash

set -e
lang=$1
model_dir=$2
pred=${model_dir}/test_output/test_amr.${lang}.pred.txt.postproc
gold=data/AMR/amr_2.0/test_${lang}.txt.features

cp $pred $gold tools/amr-evaluation-tool-enhanced
printf "Computing Finegrained Smatch\n\n"
cd tools/amr-evaluation-tool-enhanced
python2 smatch/smatch.py --pr --ms -f test_amr.${lang}.pred.txt.postproc  test_${lang}.txt.features > ../../${model_dir}/test_output/finegrained_${lang}.txt
python2 smatch/smatch.py --raw --ms -f test_amr.${lang}.pred.txt.postproc  test_${lang}.txt.features > ../../${model_dir}/test_output/finegrained_raw_${lang}.txt
printf "Finished, Computing Overall Smatch\n\n"
./evaluation.sh test_amr.${lang}.pred.txt.postproc  test_${lang}.txt.features | tee ../../${model_dir}/test_output/summary_${lang}.txt
