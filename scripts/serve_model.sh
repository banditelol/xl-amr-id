#!/usr/bin/env bash
cuda=0
lang=$1
model_dir=$2
util_dir=data/AMR/en_ms_utils

printf "Predicting...`date`\n"
python -u -m xlamr_stog.commands.serve --archive-file ${model_dir}/ \
--weights-file ${model_dir}/best.th \
--util-dir ${util_dir} \
--cuda-device $cuda \
--beam-size 5 \
--lang ${lang} \
--predictor STOG \
--raw-text '# ::id nw.chtb_0302.8 ::date 2012-11-28T21:52:48 ::annotator SDL-AMR-09 ::preferred\n# ::snt Tamat\n# ::tokens ["Tamat"]\n# ::lemmas ["tamat"]\n# ::pos_tags ["PROPN"]\n# ::ner_tags ["O"]\n# ::tok-en ( End )\n(e / end-01)'

printf "Done.`date`\n\n"