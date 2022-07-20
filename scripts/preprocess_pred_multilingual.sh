#!/usr/bin/env bash

set -e
lang=$1
data_dir=data/AMR/wrete
declare -a StringArray=("test_sent_A.txt.features"  "test_sent_B.txt.features"  "train_sent_A.txt.features"  "train_sent_B.txt.features"  "valid_sent_A.txt.features"  "valid_sent_B.txt.features")
util_dir=data/AMR/en_ms_utils

for files in ${StringArray[@]}; do
  echo $files
  pred_data=${data_dir}/${files}

  if [[ -f "$pred_data" ]]; then
    printf "Copying to input_clean ...`date`\n"
    cp ${pred_data} ${pred_data}.input_clean
    printf "Done. `date`\n\n"

    printf "Anonymizing NM ...`date`\n"
    python -u -m xlamr_stog.data.dataset_readers.amr_parsing.preprocess.text_anonymizor \
        --util_dir ${util_dir} \
        --amr_file ${pred_data}.input_clean \
        --lang ${lang}

    printf "Done.`date`\n\n"

    printf "Removing senses...`date`\n"
    python -u -m xlamr_stog.data.dataset_readers.amr_parsing.preprocess.sense_remover \
        --util_dir ${util_dir} \
        --amr_files ${pred_data}.input_clean.recategorize \

    printf "Done.`date`\n\n"

    printf "Renaming preprocessed files...`date`\n"
    mv ${pred_data}.input_clean.recategorize.nosense ${pred_data}.recat
    rm ${data_dir}/*.input_clean*

  fi

done