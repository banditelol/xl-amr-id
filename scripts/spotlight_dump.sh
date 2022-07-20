lang=$1
util_dir=data/AMR/en_ms_utils
test_data=data/AMR/amr_2.0/test_${lang}.txt.features

mkdir ${util_dir}/babelfy
babelfy_path=babelfy/${lang}_test_babelfy_wiki
printf "Wikification...`date`\n"
python -u -m xlamr_stog.data.dataset_readers.amr_parsing.postprocess.wikification \
    --amr_files ${test_data} \
    --util_dir ${util_dir}\
    --babelfy_wiki $babelfy_path\
    --lang ${lang}\
    --dump_babelfy_wiki
printf "Done.`date`\n\n"

