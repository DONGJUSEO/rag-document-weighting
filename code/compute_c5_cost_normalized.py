"""
C5: Cost-normalized gain

Definition: ΔEM (vs Naive) per millisecond of evidence-computation latency.
Quantifies the "training-free, low-overhead" advantage of CE.

  cost_norm = mean(ΔEM_method) / latency_method (in %p per ms)

Latency values (from paper Appendix J):
  CE: ~20 ms/query
  ES: ~160 ms/query
  NLI: ~110 ms/query
  weighting: <1 ms (negligible)

ΔEM averages from main paper:
  CE: +2.19 (Dir-CE vs Naive, 9 cells)
  ES: +0.82
  NLI: -1.38

Usage:
    python3 compute_c5_cost_normalized.py

Outputs:
    results/c5_cost_normalized.json
"""

import json
import os
import sys
import time
from datetime import datetime

import numpy as np

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CODE_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, "results")

LLM_FILES = {
    "qwen2_5_7b": "qwen2_5_7b_instruct_turbo",
    "gpt_4_1_mini": "gpt_4_1_mini",
    "llama_3_3_70b": "llama_3_3_70b_instruct_turbo",
}
DATASETS = ["nq", "triviaqa", "popqa"]

# Latency (ms/query) — from paper Appendix J (Implementation Details)
LATENCY_MS = {
    "ce": 20,
    "es": 160,
    "nli": 110,
    "weighting": 1,  # < 1 ms, treat as 1 for division safety
}

# Per-call API cost estimate ($)
# Total ~$100 across all experiments (per paper); ~31K queries × 10 docs
# CE/ES/NLI are local — no API cost
# Per-query LLM cost is the main API cost; here we focus on evidence overhead
PER_QUERY_API_COST = {
    "ce": 0.0,   # local
    "es": 0.0,   # local
    "nli": 0.0,  # local
}


def load_em_diffs():
    """Load Naive vs Dir-* EM for all 9 cells × 3 evidences (CE/ES/NLI)."""
    diffs = {"ce": [], "es": [], "nli": []}
    em_data = {"ce": [], "es": [], "nli": []}

    evidence_files = {
        "ce": "cross_encoder",
        "es": "embedding_stability",
        "nli": "nli",
    }
    config_keys = {
        "ce": "dirichlet_b0.5_l30.0_cross_encoder",
        "es": "dirichlet_b0.5_l30.0_embedding_stability",
        "nli": "dirichlet_b0.5_l30.0_nli",
    }

    for evid_short, evid_full in evidence_files.items():
        for llm_key, llm_short in LLM_FILES.items():
            for ds in DATASETS:
                fn = os.path.join(RESULTS_DIR, f"{ds}_{evid_full}_voting_{llm_short}.json")
                if not os.path.exists(fn):
                    print(f"  WARN: missing {fn}")
                    continue
                with open(fn, "r", encoding="utf-8") as f:
                    d = json.load(f)
                naive = d["naive"]["EM"] * 100
                cfg_key = config_keys[evid_short]
                if cfg_key not in d:
                    print(f"  WARN: {cfg_key} missing in {fn}")
                    continue
                dir_em = d[cfg_key]["EM"] * 100
                diffs[evid_short].append(dir_em - naive)
                em_data[evid_short].append({
                    "llm": llm_key, "dataset": ds,
                    "naive": naive, "dir": dir_em,
                    "delta": dir_em - naive,
                })

    return diffs, em_data


def main():
    output = os.path.join(RESULTS_DIR, "c5_cost_normalized.json")
    print(f"=== C5: Cost-normalized gain ===")
    print(f"Output: {output}\n")

    diffs, em_data = load_em_diffs()
    n_expected = len(LLM_FILES) * len(DATASETS)
    for evid, cells in diffs.items():
        if len(cells) != n_expected:
            raise SystemExit(f"C5: {evid.upper()} has {len(cells)}/{n_expected} cells (missing result files); "
                             "refusing to write a partial average")

    results = {
        "_meta": {
            "definition": "(mean ΔEM vs Naive across 9 cells) / (latency ms/query)",
            "unit": "%p per ms",
            "latency_source": "paper Appendix J (Implementation Details)",
            "timestamp": datetime.now().isoformat(),
        },
        "latency_ms": LATENCY_MS,
        "evidence": {},
    }

    print(f"{'Evidence':<10} {'Mean ΔEM':<12} {'Latency':<12} {'%p/ms':<10} {'%p/sec':<10}")
    print("-" * 55)

    for evid in ["ce", "es", "nli"]:
        if not diffs[evid]:
            continue
        mean_delta = float(np.mean(diffs[evid]))
        std_delta = float(np.std(diffs[evid], ddof=0))
        latency = LATENCY_MS[evid]
        cost_norm = mean_delta / latency
        cost_norm_per_sec = mean_delta / (latency / 1000.0)

        results["evidence"][evid] = {
            "mean_delta_em_pp": mean_delta,
            "std_delta_em_pp": std_delta,
            "n_cells": len(diffs[evid]),
            "latency_ms": latency,
            "cost_normalized_pp_per_ms": cost_norm,
            "cost_normalized_pp_per_sec": cost_norm_per_sec,
        }

        print(f"{evid.upper():<10} {mean_delta:+.2f}        {latency:>3} ms       "
              f"{cost_norm:+.4f}    {cost_norm_per_sec:+.2f}")

    # Compare CE vs ES vs NLI
    if all(e in results["evidence"] for e in ["ce", "es", "nli"]):
        print(f"\n=== Per-method advantage ===")
        ce = results["evidence"]["ce"]
        es = results["evidence"]["es"]
        nli = results["evidence"]["nli"]

        print(f"CE/ES ratio (cost-normalized): {ce['cost_normalized_pp_per_ms'] / max(es['cost_normalized_pp_per_ms'], 1e-6):.1f}×")
        print(f"  CE delivers {ce['cost_normalized_pp_per_ms'] / max(es['cost_normalized_pp_per_ms'], 1e-6):.1f}× more EM gain per ms than ES")
        print(f"  NLI is {-nli['cost_normalized_pp_per_ms']:+.4f} %p/ms (negative — actively harmful)")

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n=== Saved: {output} ===")


if __name__ == "__main__":
    main()
