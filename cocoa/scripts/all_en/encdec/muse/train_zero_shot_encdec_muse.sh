#!/usr/bin/env bash

cs_mode="All_EN"
nn_model="encdec"
home_dir="/home1/mmhamdi/"
data_path=$home_dir"Datasets/CS/"
results_path=$home_dir"Results/cs-aware-colab/"$cs_mode"/"$nn_model"/RANDOM/"
embed_type="MUSE"

# Metadata file
schema_path=$data_path"metadata/schema_sp_hi_en.json"
scenarios_path=$data_path"metadata/scenarios.json"
stop_words_path=$data_path"metadata/common_words_en_es.txt" # TODO to be replaced to fit hindi

echo "Train and fine tune on English with no CS data"
PYTHONPATH=. python main.py \
    --train-examples-paths $data_path"splits/"$cs_mode"/train.json" \
    --test-examples-paths $data_path"splits/"$cs_mode"/dev.json" \
    --schema-path $schema_path\
    --scenarios-path $scenarios_path \
    --stop-words $stop_words_path \
    --word-embed-type $embed_type \
    --rnn-type lstm \
    --min-epochs 10 \
    --learning-rate 0.5 \
    --optimizer adagrad \
    --print-every 50 \
    --rnn-size 100 \
    --grad-clip 0 \
    --num-items 12 \
    --batch-size 32 \
    --entity-encoding-form type \
    --entity-decoding-form type \
    --node-embed-in-rnn-inputs \
    --msg-aggregation max \
    --word-embed-size 300 \
    --node-embed-size 50 \
    --entity-hist-len -1 \
    --gpu 1 \
    --learned-utterance-decay \
    --checkpoint $results_path"checkpoints/" \
    --stats-file $results_path"stats.json" \
    --log-file $results_path"logs/train.txt"
