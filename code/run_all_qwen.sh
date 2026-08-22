#!/bin/bash
cd "$(dirname "$0")"

for ev in cross_encoder embedding_stability nli; do
  for ds in nq triviaqa popqa; do
    echo "=== Qwen: $ds × $ev ==="
    python3 run_main.py --dataset $ds --evidence $ev --aggregation voting
  done
done
echo "=== Qwen ALL DONE ==="
