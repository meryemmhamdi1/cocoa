#!/usr/bin/env bash

. scripts/hyper.config
. scripts/paths.config
cs_mode="All_EN"
nn_model="dynonet"
emb_model="MUSE"
dynot_model="attn-copy-encdec"
results_path=$results_dir$cs_mode"/"$nn_model"/"$emb_model"/"

echo "Train and fine tune on English with no CS data"
PYTHONPATH=. python main.py \
    --train-examples-paths $data_path"splits/"$cs_mode"/train.json" \
    --test-examples-paths $data_path"splits/"$cs_mode"/dev.json" \
    --schema-path $schema_path\
    --scenarios-path $scenarios_path \
    --stop-words $stop_words_path \
    --model $dynot_model \
    --word-embed-type $emb_model \
    --rnn-type $rnn_type \
    --min-epochs $min_epochs \
    --learning-rate $learning_rate \
    --optimizer $optimizer \
    --print-every $print_every \
    --rnn-size $rnn_size \
    --grad-clip $grad_clip \
    --num-items $num_items \
    --batch-size $batch_size \
    --entity-encoding-form $entity_encoding_form \
    --entity-decoding-form $entity_decoding_form \
    --node-embed-in-rnn-inputs \
    --msg-aggregation $msg_aggregation \
    --word-embed-size $word_embed_size \
    --node-embed-size $node_embed_size \
    --entity-hist-len $entity_hist_len \
    --gpu 1 \
    --learned-utterance-decay \
    --checkpoint $results_path"checkpoints/" \
    --stats-file $results_path"stats.json" \
    --log-file $results_path"logs/train.txt"
