lang=$1
util_dir=data/AMR/en_ms_utils

mkdir ${util_dir}/babelfy
data_dir=data/AMR/wrete
declare -a StringArray=("test_sent_A"  "test_sent_B"  "train_sent_A"  "train_sent_B"  "valid_sent_A"  "valid_sent_B")

for files in ${StringArray[@]}; do
  echo $files
  input_file=${data_dir}/${files}.txt.features.recat
  babelfy_path=babelfy/${files}_babelfy_wiki_wrete

  if [[ -f "$input_file" ]]; then
    printf "Wikification...`date`\n"
    python -u -m xlamr_stog.data.dataset_readers.amr_parsing.postprocess.wikification \
        --amr_files ${input_file} \
        --util_dir ${util_dir} \
        --babelfy_wiki ${babelfy_path} \
        --lang ${lang} \
        --dump_babelfy_wiki
    printf "Done.`date`\n\n"

  fi

done