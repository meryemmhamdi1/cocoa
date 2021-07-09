#!/usr/bin/env bash

cs_mode="joint" # sup_cs #zero_shot
nn_model="encdec" #dynonet


i="0"


echo "Test on CS Dataset"
    PYTHONPATH=. python src/main.py --schema-path "data/schema_cs_en.json" --scenarios-path "data/scenarios.json" \
    --stop-words "data/common_words_en_es.txt" --init-from  "checkpoints/"$cs_mode"/"$nn_model"/checkpoint_"$i \
    --test-examples-paths "cross-val/test_"$i".json" --test --batch-size 32 --best --stats-file "stats.json" --decoding sample 0\
    --log-file "logs/"$cs_mode"/"$nn_model"/test_cs_"$i".txt"