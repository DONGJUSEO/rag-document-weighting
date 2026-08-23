#!/bin/bash
set -e
cd "$(dirname "$0")"
export LLM_BACKEND="openai"
export LLM_MODEL="gpt-4.1-mini"
export OPENAI_API_KEY="${OPENAI_API_KEY}"  # Set via environment variable

for ev in cross_encoder embedding_stability nli; do
  for ds in nq triviaqa popqa; do
    echo "=== GPT-4.1-mini: $ds × $ev ==="
    python3 run_main.py --dataset $ds --evidence $ev --aggregation voting
  done
done
echo "=== GPT-4.1-mini ALL DONE ==="
