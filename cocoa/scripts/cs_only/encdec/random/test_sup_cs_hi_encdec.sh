#!/usr/bin/env bash

cs_mode="CS/CS_HI"
nn_model="encdec"
home_dir="/home1/mmhamdi/"
data_path=$home_dir"Datasets/CS/"
results_path=$home_dir"Results/cs-aware-colab/"$cs_mode"/"$nn_model"/"

# Metadata file
schema_path=$data_path"metadata/schema_sp_hi_en.json"
scenarios_path=$data_path"metadata/scenarios.json"
stop_words_path=$data_path"metadata/common_words_en_es.txt" # TODO to be replaced to fit hindi

echo "Test on all chats in Hinglish separately"
PYTHONPATH=. python main.py \
    --test-examples-paths $data_path"splits/"$cs_mode"/test.json" \
    --schema-path $schema_path\
    --scenarios-path $scenarios_path \
    --stop-words $stop_words_path \
    --test \
    --best \
    --batch-size 32 \
    --decoding sample 0 \
    --init-from $results_path"checkpoints/" \
    --stats-file $results_path"stats.json" \
    --log-file $results_path"logs/test_commondost_all_chats.txt"