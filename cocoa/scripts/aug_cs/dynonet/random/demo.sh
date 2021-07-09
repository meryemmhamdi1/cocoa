#!/usr/bin/env bash
home_dir="/home1/mmhamdi/"
checkpoint_path=$home_dir"Results/cs-aware-colab/All_EN/dynonet/checkpoints/"
data_path=$home_dir"Datasets/CS/"
schema_path=$data_path"metadata/schema_sp_hi_en.json"
scenarios_path=$data_path"metadata/scenarios.json"
stop_words_path=$data_path"metadata/common_words_en_es.txt" # TODO to be replaced to fit hindi

PYTHONPATH=. python generate_dataset.py\
                    --max-turns 46 \
                    --schema-path $schema_path \
                    --scenarios-path $scenarios_path \
                    --stop-words $stop_words_path \
                    --test-examples-paths $home_dir"cocoa/test_demo.json" \
                    --train-max-examples 0 \
                    --agents neural neural \
                    --model-path $checkpoint_path \
                    --decoding sample 0.5 select