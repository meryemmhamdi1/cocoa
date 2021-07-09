#!/usr/bin/env bash

cs_mode="sup_cs"
nn_model="encdec_bert"
dynot_model="encdec"

for i in "0" #"1" "2" "3" "4" "5" "6" "7" "8" "9"
do
    echo "Train only on CS data"
        PYTHONPATH=. python src/main_cs.py --schema-path "cross-val/schema_cs.json" --scenarios-path "data/scenarios.json" \
        --train-examples-paths "cross-val/train_"$i".json" --word-embed-type "BERT" --word-embed-size 768  \
        --test-examples-paths "cross-val/test_small.json" --stop-words "data/common_words_en_es.txt" \
        --min-epochs 10 --checkpoint "checkpoints/"$cs_mode"/"$nn_model"/checkpoint_"$i --rnn-type lstm --learning-rate 0.5 \
        --optimizer adagrad --print-every 50 --model $dynot_model --gpu 1 --rnn-size 384 --grad-clip 0 --num-items 12 \
        --batch-size 32 --stats-file stats.json --entity-encoding-form type --entity-decoding-form type \
        --node-embed-in-rnn-inputs --msg-aggregation max --node-embed-size 50 \
        --entity-hist-len -1 --learned-utterance-decay --log-file "logs/"$cs_mode"/"$nn_model"/train_"$i".txt"

done

