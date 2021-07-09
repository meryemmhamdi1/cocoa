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

echo "Test on all chats in Spanglish separately"
PYTHONPATH=. python main.py \
    --test-examples-paths $data_path"splits/CS/CS_SP/test.json" \
    --schema-path $schema_path\
    --scenarios-path $scenarios_path \
    --stop-words $stop_words_path \
    --word-embed-type $embed_type \
    --test \
    --best \
    --batch-size 32 \
    --decoding sample 0 \
    --init-from $results_path"checkpoints/" \
    --stats-file $results_path"stats.json" \
    --log-file $results_path"logs/test_commonamigos_all_chats.txt"


echo "Test on all chats in Hinglish separately"
PYTHONPATH=. python main.py \
    --test-examples-paths $data_path"splits/CS/CS_HI/test.json" \
    --schema-path $schema_path\
    --scenarios-path $scenarios_path \
    --stop-words $stop_words_path \
    --word-embed-type $embed_type \
    --test \
    --best \
    --batch-size 32 \
    --decoding sample 0 \
    --init-from $results_path"checkpoints/" \
    --stats-file $results_path"stats.json" \
    --log-file $results_path"logs/test_commondost_all_chats.txt"

##############################
echo "Test on Mutual Friends only"
PYTHONPATH=. python main.py \
    --test-examples-paths $data_path"MutualFriends/test.json" \
    --schema-path $schema_path \
    --scenarios-path $scenarios_path \
    --stop-words $stop_words_path \
    --word-embed-type $embed_type \
    --test \
    --best \
    --batch-size 32 \
    --decoding sample 0 \
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
        --word-embed-type $embed_type \
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
        --word-embed-type $embed_type \
        --test \
        --best \
        --batch-size 32 \
        --decoding sample 0 \
        --init-from $results_path"checkpoints/" \
        --stats-file $results_path"stats.json" \
        --log-file $results_path"logs/test_"$data_name"_"$pattern+".txt"
done
