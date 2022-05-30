#!/usr/bin/env bash

set -e

echo "Downloading embeddings."
mkdir -p data/bert-base-multilingual-cased
curl -O https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz
tar -xzvf bert-base-multilingual-cased.tar.gz -C data/bert-base-multilingual-cased
curl -o data/bert-base-multilingual-cased/bert-base-multilingual-cased-vocab.txt \
    https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt
rm bert-base-multilingual-cased.tar.gz

echo "Downloading base mt5"
mkdir -p data/mt5-base
curl -o data/mt5-base/config.json -L https://huggingface.co/google/mt5-base/resolve/main/config.json
curl -o data/mt5-base/pytorch_model.bin -L https://huggingface.co/google/mt5-base/resolve/main/pytorch_model.bin
curl -o data/mt5-base/spiece.model -L https://huggingface.co/google/mt5-base/resolve/main/spiece.model

echo "Downloading base xlm-r"
mkdir -p data/xlm-roberta-base
curl -o data/xlm-roberta-base/config.json -L https://huggingface.co/xlm-roberta-base/resolve/main/config.json
curl -o data/xlm-roberta-base/pytorch_model.bin -L https://huggingface.co/xlm-roberta-base/resolve/main/pytorch_model.bin
curl -o data/xlm-roberta-base/sentencepiece.bpe.model -L https://huggingface.co/xlm-roberta-base/resolve/main/sentencepiece.bpe.model
curl -o data/xlm-roberta-base/tokenizer.json -L https://huggingface.co/xlm-roberta-base/resolve/main/tokenizer.json


echo "Downloading tools."
mkdir -p tools
git clone https://github.com/ChunchuanLv/amr-evaluation-tool-enhanced.git tools/amr-evaluation-tool-enhanced
