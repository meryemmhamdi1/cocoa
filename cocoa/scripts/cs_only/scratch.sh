#!/usr/bin/env bash

echo "Test on CS Dataset"
    PYTHONPATH=. python src/main.py --schema-path "cross-val/schema_cs.json" --scenarios-path "data/scenarios.json" \
    --stop-words "data/common_words_en_es.txt" --init-from  "checkpoints/"$cs_mode"/"$nn_model"/checkpoint_"$i \
     --word-embed-size 300 --word-embed-type "MUSE" \
    --test-examples-paths "cross-val/test_"$i".json" --test --batch-size 32 --best --stats-file "stats.json" --decoding sample 0\
    --log-file "logs/"$cs_mode"/"$nn_model"/test_cs_"$i".txt"


echo "Train only on CS data"
    PYTHONPATH=. python src/main.py --schema-path "cross-val/schema_cs.json" --scenarios-path "data/scenarios.json" \
    --train-examples-paths "cross-val/train_"$i".json" \
    --test-examples-paths "cross-val/test_"$i".json" --stop-words "data/common_words_en_es.txt" \
    --min-epochs 10 --checkpoint "checkpoints/"$cs_mode"/"$nn_model"/checkpoint_"$i --rnn-type lstm --learning-rate 0.5 \
    --optimizer adagrad --print-every 50 --model $dynot_model --gpu 1 --rnn-size 100 --grad-clip 0 --num-items 12 \
    --batch-size 32 --stats-file stats.json --entity-encoding-form type --entity-decoding-form type \
    --node-embed-in-rnn-inputs --msg-aggregation max --word-embed-size 100 --node-embed-size 50 \
    --entity-hist-len -1 --learned-utterance-decay --log-file "logs/"$cs_mode"/"$nn_model"/train_"$i".txt"