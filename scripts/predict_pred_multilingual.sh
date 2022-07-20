#!/usr/bin/env bash
cuda=0
lang=$1
model_dir=$2
data_dir=data/AMR/wrete
declare -a StringArray=("test_sent_A"  "test_sent_B"  "train_sent_A"  "train_sent_B"  "valid_sent_A"  "valid_sent_B")
util_dir=data/AMR/en_ms_utils

for files in ${StringArray[@]}; do
  echo $files
  input_file=${data_dir}/${files}.txt.features.recat

  if [[ -f "$input_file" ]]; then
    printf "Predicting...`date`\n"
    python -u -m xlamr_stog.commands.predict --archive-file ${model_dir}/ \
    --weights-file ${model_dir}/best.th \
    --input-file $lang $input_file \
    --batch-size 32 \
    --use-dataset-reader \
    --cuda-device $cuda \
    --output-file ${model_dir}/test_output/${files}.pred.txt \
    --silent \
    --beam-size 5 \
    --predictor STOG

    printf "Done.`date`\n\n"

  fi

done