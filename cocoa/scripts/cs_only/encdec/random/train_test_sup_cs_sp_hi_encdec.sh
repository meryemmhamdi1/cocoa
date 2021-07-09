#!/usr/bin/env bash

cs_mode="cs_sp_hi"
nn_model="encdec"
data_path="/home1/mmhamdi/Datasets/CS/splits/" ## ADD THE ONE IN THE CLUSTER
results_path="/home1/mmhamdi/Results/cs-aware-colab" ### ADD THE ONE IN THE CLUSTER

echo "Train on CS_SP_HI only"
PYTHONPATH=. python src/main_cs.py \
    --train-examples-paths data_path"CS/CS_SP_HI/train.json" \
    --test-examples-paths data_path"CS/CS_SP_HI/dev.json" \
    --schema-path data_path"schema_sp_hi_en.json"\
    --scenarios-path data_path"scenarios.json" \
    --stop-words "data/common_words_en_es.txt" \
    --min-epochs 10 \
    --checkpoint results_path"checkpoints/"$cs_mode"/"$nn_model"/checkpoint" \
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
    --stats-file results_path"checkpoints/"$cs_mode"/"$nn_model"/stats.json" \
    --log-file results_path"checkpoints/"$cs_mode"/"$nn_model"/logs/"$cs_mode"/"$nn_model"/train.txt"

echo "Test on English only"
PYTHONPATH=. python src/main.py \
    --test-examples-paths data_path"All_EN/test.json" \
    --schema-path data_path"schema_sp_hi_en.json"\
    --scenarios-path data_path"scenarios.json" \
    --stop-words data_path"common_words_en_es.txt"\
    --init-from results_path"checkpoints/"$cs_mode"/"$nn_model"/checkpoint" \
    --test \
    --batch-size 32 \
    --best \
    --decoding sample 0\
    --stats-file results_path"checkpoints/"$cs_mode"/"$nn_model"/stats.json" \
    --log-file results_path"checkpoints/"$cs_mode"/"$nn_model"/logs/"$cs_mode"/"$nn_model"/test_en.txt"

echo "Test on each pattern on each Pattern in Spanglish separately"
for pattern in "en_alt_sp" "en_alt_sp_inf" "en_ins_sp" "en_ins_sp_inf" "en_mono" "random" "sp_alt_en" "sp_alt_en_inf" "sp_ins_en" "sp_ins_en_inf" "sp_mono"
do
    PYTHONPATH=. python src/main.py \
        --test-examples-paths data_path"Patterns/commonamigos/"pattern"/test.json" \
        --schema-path data_path"schema_sp_hi_en.json"\
        --scenarios-path data_path"scenarios.json" \
        --stop-words data_path"common_words_en_es.txt" \
        --init-from results_path"checkpoints/"$cs_mode"/"$nn_model"/checkpoint" \
        --test \
        --batch-size 32 \
        --best \
        --decoding sample 0 \
        --stats-file results_path"checkpoints/"$cs_mode"/"$nn_model"/stats.json" \
        --log-file results_path"checkpoints/"$cs_mode"/"$nn_model"/logs/"$cs_mode"/"$nn_model"/test_commonamigos_"+pattern+".txt"
done

echo "Test on each pattern on each Pattern in Hinglish separately"
for pattern in "en_alt_hi" "en_alt_hi_inf" "en_ins_hi" "en_ins_hi_inf" "en_mono" "random" "hi_alt_en" "hi_alt_en_inf" "hi_ins_en" "hi_ins_en_inf" "hi_mono"
do
    PYTHONPATH=. python src/main.py \
        --test-examples-paths data_path"Patterns/commondost/"pattern"/test.json" \
        --schema-path data_path"schema_sp_hi_en.json"\
        --scenarios-path data_path"scenarios.json" \
        --stop-words data_path"common_words_en_es.txt" \
        --init-from results_path"checkpoints/"$cs_mode"/"$nn_model"/checkpoint" \
        --test \
        --batch-size 32 \
        --best \
        --decoding sample 0 \
        --stats-file results_path"checkpoints/"$cs_mode"/"$nn_model"/stats.json" \
        --log-file results_path"checkpoints/"$cs_mode"/"$nn_model"/logs/"$cs_mode"/"$nn_model"/test_commondost_"+pattern+".txt"
done