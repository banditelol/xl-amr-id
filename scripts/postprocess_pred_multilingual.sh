#!/usr/bin/env bash

set -e

# Directory where intermediate utils will be saved to speed up processing.
util_dir=data/AMR/en_ms_utils

lang=$1
model_dir=$2

spotlight_path=babelfy/ms_test_babelfy_wiki

# ========== Set the above variables correctly ==========
declare -a StringArray=("test_sent_A"  "test_sent_B"  "train_sent_A"  "train_sent_B"  "valid_sent_A"  "valid_sent_B")
util_dir=data/AMR/en_ms_utils

for files in ${StringArray[@]}; do
  echo $files
  pred_data=${model_dir}/test_output/${files}.pred.txt

  if [[ -f "$pred_data" ]]; then
    printf "Frame lookup...`date`\n"
    python -u -m xlamr_stog.data.dataset_readers.amr_parsing.postprocess.node_restore \
        --amr_files ${pred_data} \
        --util_dir ${util_dir}
    printf "Done.`date`\n\n"

    printf "Wikification...`date`\n"
    python -u -m xlamr_stog.data.dataset_readers.amr_parsing.postprocess.wikification \
        --amr_files ${pred_data}.frame \
        --util_dir ${util_dir}\
        --spotlight_wiki ${spotlight_path}\
        --exclude_spotlight\
        --lang ${lang}

    printf "Done.`date`\n\n"

    printf "Expanding nodes...`date`\n"
    python -u -m xlamr_stog.data.dataset_readers.amr_parsing.postprocess.expander \
        --amr_files ${pred_data}.frame.wiki \
        --util_dir ${util_dir} \
        --u_pos True \
        --lang ${lang}

    printf "Done.`date`\n\n"

    mv ${pred_data}.frame.wiki.expand ${pred_data}.postproc

  fi

done