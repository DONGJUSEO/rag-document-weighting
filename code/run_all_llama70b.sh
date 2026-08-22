#!/bin/bash
cd "$(dirname "$0")"
export LLM_BACKEND="together"
export LLM_MODEL="meta-llama/Llama-3.3-70B-Instruct-Turbo"

for ev in cross_encoder embedding_stability nli; do
  for ds in nq triviaqa popqa; do
    echo "=== Llama-3.3-70B: $ds × $ev ==="
    python3 run_main.py --dataset $ds --evidence $ev --aggregation voting
  done
done
echo "=== Llama-3.3-70B ALL DONE ==="
