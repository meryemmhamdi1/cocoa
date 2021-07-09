#!/usr/bin/env bash

. scripts/hyper.config
. scripts/paths.config
cs_mode="All_EN"
nn_model="dynonet"
emb_model="RANDOM"
dynot_model="attn-copy-encdec"
results_path=$results_dir$cs_mode"/"$nn_model"/"$emb_model"/"

for data_split in "splits/CS/CS_SP" "splits/CS/CS_HI" "MutualFriends"
do
    if [$data_split == "splits/CS/CS_SP"]
    then
        name="commonamigos"
    elif [$data_split == "splits/CS/CS_HI"]
    then
        name="commondost"
    else
        name="mutualfriends"
    fi

    echo "Test on all chats in "$name" separately"
    PYTHONPATH=. python main.py \
        --test-examples-paths $data_path$data_split"/test.json" \
        --schema-path $schema_path\
        --scenarios-path $scenarios_path \
        --stop-words $stop_words_path \
        --word-embed-type $emb_model \
        --test \
        --best \
        --batch-size $batch_size \
        --decoding sample $decoding_sample \
        --init-from $results_path"checkpoints/" \
        --stats-file $results_path"stats.json" \
        --log-file $results_path"logs/test_"$name"_all_chats.txt"
done

echo "Test on each Pattern in Spanglish and Hinglish separately"
for lang in "sp" "hi"
do
    if [$lang == "sp"]
    then
        data_name="commonamigos"
    else
        data_name="commondost"

    fi

    for pattern in "en_ins_"$lang  "en_ins_"$lang"_inf"  $lang"_ins_en"  $lang"_ins_en_inf" "en_alt_"$lang  "en_alt_"$lang"_inf"  $lang"_alt_en"  $lang"_alt_en_inf" "en_mono" $lang"_mono" "random"
    do
        PYTHONPATH=. python main.py \
            --test-examples-paths $data_path"splits/Patterns/"$data_name"/"$pattern"/test.json" \
            --schema-path $schema_path \
            --scenarios-path $scenarios_path \
            --stop-words $stop_words_path \
            --word-embed-type $emb_model \
            --test \
            --best \
            --batch-size $batch_size \
            --decoding sample 0 \
            --init-from $results_path"checkpoints/" \
            --stats-file $results_path"stats.json" \
            --log-file $results_path"logs/test_"$data_name"_"$pattern+".txt"
    done
done

