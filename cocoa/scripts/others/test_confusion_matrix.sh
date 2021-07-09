#!/usr/bin/env bash

cs_mode="sup_cs"
nn_model="dynonet"
dynot_model="attn-copy-encdec"

i="0"

echo "Test on CS Dataset"
PYTHONPATH=. python src/main_cs.py --schema-path "cross-val/schema_cs.json" --scenarios-path "data/scenarios.json" \
--stop-words "data/common_words_en_es.txt" --init-from  "checkpoints/"$cs_mode"/"$nn_model"/checkpoint_"$i \
--test-examples-paths "cross-val/all_cs_dialogue.json" --test --batch-size 32 --best --stats-file "stats.json" --decoding sample 0