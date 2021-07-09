#!/usr/bin/env bash
PYTHONPATH=. python src/scripts/get_data_statistics.py --transcripts data/test.json --schema-path data/schema.json \
--scenarios-path data/scenarios_new.json --stop-words data/common_words.txt --stats-output dialogue_stats.json
