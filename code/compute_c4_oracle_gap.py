"""
C4: Oracle gap closure

Definition: How much of the gap between Naive and Generation Oracle does Dir-CE close?

  closure = (EM_DirCE - EM_Naive) / (EM_Oracle - EM_Naive)

Generation Oracle = if ANY of the 10 per-document answers matches gold, count 1.
This is the upper bound achievable via re-weighting alone (assuming ideal weights).

Useful framing for the paper: small absolute EM gains (+0.5-4.1%p) become more
meaningful when expressed as fraction of oracle headroom recovered.

Usage:
    python3 compute_c4_oracle_gap.py

Outputs:
    results/c4_oracle_gap.json
"""

import json
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CODE_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Sources
LLM_FILES = {
    "qwen2_5_7b": "qwen2_5_7b_instruct_turbo",
    "gpt_4_1_mini": "gpt_4_1_mini",
    "llama_3_3_70b": "llama_3_3_70b_instruct_turbo",
}

DATASETS = ["nq", "triviaqa", "popqa"]


def load_em_values():
    """Load EM values for Naive, Dir-CE, and Oracle from existing result files."""
    em = {}

    for llm_key, llm_short in LLM_FILES.items():
        em[llm_key] = {}

        # 1. Oracle from additional_baselines
        ab_file = os.path.join(RESULTS_DIR, f"additional_baselines_{llm_short}.json")
        with open(ab_file, "r", encoding="utf-8") as f:
            ab = json.load(f)

        # 2. Naive and Dir-CE from cross_encoder_voting
        for ds in DATASETS:
            ce_file = os.path.join(RESULTS_DIR, f"{ds}_cross_encoder_voting_{llm_short}.json")
            with open(ce_file, "r", encoding="utf-8") as f:
                ce = json.load(f)

            # All EM values stored as fractions [0, 1]; convert to percent for display
            naive_em = ce["naive"]["EM"] * 100
            dir_ce_em = ce["dirichlet_b0.5_l30.0_cross_encoder"]["EM"] * 100
            oracle_em = ab[ds]["oracle"]["rate"] * 100

            em[llm_key][ds] = {
                "naive": naive_em,
                "dir_ce": dir_ce_em,
                "oracle": oracle_em,
            }

    return em


def main():
    output = os.path.join(RESULTS_DIR, "c4_oracle_gap.json")
    print(f"=== C4: Oracle gap closure ===")
    print(f"Output: {output}\n")

    em = load_em_values()
    closures = {}

    for llm_key in LLM_FILES:
        closures[llm_key] = {}
        for ds in DATASETS:
            n = em[llm_key][ds]["naive"]
            d = em[llm_key][ds]["dir_ce"]
            o = em[llm_key][ds]["oracle"]
            gap = o - n
            improvement = d - n
            closure = improvement / gap if gap > 1e-9 else 0.0

            closures[llm_key][ds] = {
                "naive": n,
                "dir_ce": d,
                "oracle": o,
                "improvement_pp": improvement,
                "oracle_gap_pp": gap,
                "closure_fraction": closure,
                "closure_percent": closure * 100,
            }

    closures["_meta"] = {
        "definition": "(Dir-CE - Naive) / (Oracle - Naive)",
        "oracle": "Generation Oracle: any per-doc answer matches gold",
        "timestamp": datetime.now().isoformat(),
    }

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(closures, f, indent=2, ensure_ascii=False)

    print(f"=== Saved: {output} ===\n")
    print(f"=== Oracle Gap Closure (% of oracle headroom recovered by Dir-CE) ===")
    print(f"{'LLM':<15} {'Dataset':<10} {'Naive':<7} {'DirCE':<7} {'Oracle':<7} {'ΔEM':<7} {'Gap':<7} {'Closure':<10}")
    print("-" * 75)

    all_closures = []
    for llm_key in LLM_FILES:
        for ds in DATASETS:
            c = closures[llm_key][ds]
            all_closures.append(c["closure_fraction"])
            print(f"{llm_key:<15} {ds:<10} {c['naive']:.2f}   {c['dir_ce']:.2f}   "
                  f"{c['oracle']:.2f}   +{c['improvement_pp']:.2f}   "
                  f"{c['oracle_gap_pp']:.2f}    {c['closure_percent']:.1f}%")
    print("-" * 75)
    print(f"  Average closure: {np.mean(all_closures) * 100:.1f}%")
    print(f"  Range: {min(all_closures) * 100:.1f}% – {max(all_closures) * 100:.1f}%")


if __name__ == "__main__":
    main()
