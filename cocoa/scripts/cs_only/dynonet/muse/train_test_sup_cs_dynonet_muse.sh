#!/usr/bin/env bash

cs_mode="sup_cs"
nn_model="dynonet_muse"
dynot_model="attn-copy-encdec"

for i in "9" #"0" "1" "2" "3" "4" "5" "6" "7" "8" "9"
do

    echo "Test on CS Dataset"
    PYTHONPATH=. python src/main.py --schema-path "cross-val/schema_cs.json" --scenarios-path "data/scenarios.json" \
    --stop-words "data/common_words_en_es.txt" --init-from  "checkpoints/"$cs_mode"/"$nn_model"/checkpoint_"$i \
    --word-embed-size 300 --word-embed-type "MUSE" --test-examples-paths "cross-val/test_"$i".json" --test \
    --batch-size 32 --best --stats-file "stats.json" --decoding sample 0 --log-file "logs/"$cs_mode"/"$nn_model"/test_cs_"$i".txt"

done