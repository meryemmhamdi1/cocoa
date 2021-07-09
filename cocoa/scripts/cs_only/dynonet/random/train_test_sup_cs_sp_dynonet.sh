#!/usr/bin/env bash

cs_mode="CS/CS_SP_HI"
nn_model="dynonet"
dynot_model="attn-copy-encdec"
home_dir="/home1/mmhamdi/"
data_path=$home_dir"Datasets/CS/"
results_path=$home_dir"Results/cs-aware-colab/"$cs_mode"/"$nn_model"/"

# Metadata file
schema_path=$data_path"metadata/schema_sp_hi_en.json"
scenarios_path=$data_path"metadata/scenarios.json"
stop_words_path=$data_path"metadata/common_words_en_es.txt" # TODO to be replaced to fit hindi

echo "Train on CS_SP only"
PYTHONPATH=. python main.py \
    --train-examples-paths $data_path"splits/"$cs_mode"/train.json" \
    --test-examples-paths $data_path"splits/"$cs_mode"/dev.json" \
    --schema-path $schema_path\
    --scenarios-path $scenarios_path \
    --stop-words $stop_words_path \
    --model $dynot_model \
    --min-epochs 10 \
    --rnn-type lstm \
    --learning-rate 0.5 \
    --optimizer adagrad \
    --print-every 50 \
    --gpu 1 \
    --rnn-size 100 \
    --grad-clip 0 \
    --num-items 12 \
    --batch-size 32 \
    --entity-encoding-form type \
    --entity-decoding-form type \
    --node-embed-in-rnn-inputs \
    --msg-aggregation max \
    --word-embed-size 100 \
    --node-embed-size 50 \
    --entity-hist-len -1 \
    --learned-utterance-decay \
    --checkpoint $results_path"checkpoints/" \
    --stats-file $results_path"stats.json" \
    --log-file $results_path"logs/train.txt"

echo "Test on English only"
PYTHONPATH=. python main.py \
    --test-examples-paths $data_path"All_EN/test.json" \
    --schema-path $schema_path\
    --scenarios-path $scenarios_path \
    --stop-words $stop_words_path \
    --init-from results_path"checkpoints/"$cs_mode"/"$nn_model"/checkpoint" \
    --test \
    --batch-size 32 \
    --best \
    --decoding sample 0\
    --init-from $results_path"checkpoints/" \
    --stats-file $results_path"stats.json" \
    --log-file $results_path"logs/test_mutual_friends.txt"

echo "Test on each Pattern in Spanglish separately"
lang="sp"
data_name="commonamigos"

for pattern in "en_ins_"$lang  "en_ins_"$lang"_inf"  $lang"_ins_en"  $lang"_ins_en_inf" "en_alt_"$lang  "en_alt_"$lang"_inf"  $lang"_alt_en"  $lang"_alt_en_inf" "en_mono" $lang"_mono" "random"
do
    PYTHONPATH=. python main.py \
        --test-examples-paths $data_path"splits/Patterns/"$data_name"/"$pattern"/test.json" \
        --schema-path $schema_path \
        --scenarios-path $scenarios_path \
        --stop-words $stop_words_path \
        --test \
        --best \
        --batch-size 32 \
        --decoding sample 0 \
        --init-from $results_path"checkpoints/" \
        --stats-file $results_path"stats.json" \
        --log-file $results_path"logs/test_"$data_name"_"$pattern+".txt"
done

echo "Test on each Pattern in Hinglish separately"
lang="hi"
data_name="commondost"
for pattern in "en_ins_"$lang  "en_ins_"$lang"_inf"  $lang"_ins_en"  $lang"_ins_en_inf" "en_alt_"$lang  "en_alt_"$lang"_inf"  $lang"_alt_en"  $lang"_alt_en_inf" "en_mono" $lang"_mono" "random"
do
    PYTHONPATH=. python main.py \
        --test-examples-paths $data_path"splits/Patterns/"$data_name"/"$pattern"/test.json" \
        --schema-path $schema_path \
        --scenarios-path $scenarios_path \
        --stop-words $stop_words_path \
        --test \
        --best \
        --batch-size 32 \
        --decoding sample 0 \
        --init-from $results_path"checkpoints/" \
        --stats-file $results_path"stats.json" \
        --log-file $results_path"logs/test_"$data_name"_"$pattern+".txt"
done