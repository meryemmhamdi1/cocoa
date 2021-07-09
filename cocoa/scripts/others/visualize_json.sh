#!/usr/bin/env bash
PYTHONPATH=. python src/scripts/visualize_data.py --transcripts data/test.json --schema-path data/schema.json \
--scenarios-path data/scenarios.json --html-output chat.html
