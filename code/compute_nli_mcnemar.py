"""Per-cell McNemar test: Dir-NLI (beta=0.5, lambda=30, NLI evidence) vs Naive.

Reads the per-query EM vectors stored by run_main.py under
results/{ds}_nli_voting_{llm}.json["_per_question"] (full retrieval set;
PopQA n=14,267, as in results/mcnemar_results.json for Dir-CE) and writes
results/dir_nli_vs_naive_mcnemar.json in the same format as
results/dir_es_vs_naive_mcnemar.json.

Also sweeps the full 24-point (beta, lambda) NLI grid in every cell and
counts configurations that are Bonferroni-significant (alpha/27)
improvements / decreases over Naive, which backs the main-text statement
that no NLI configuration yields a significant improvement in any cell.

McNemar with Yates continuity correction, p via chi2.sf (see
regen_mcnemar_stats.py for the rationale).

Run from the package root:  python code/compute_nli_mcnemar.py
"""
import json
import os
import sys

import numpy as np
from scipy.stats import chi2

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "..", "results")
ALPHA_27 = 0.05 / 27

DATASETS = ["nq", "triviaqa", "popqa"]
LLMS = [("qwen", "qwen2_5_7b_instruct_turbo"),
        ("gpt", "gpt_4_1_mini"),
        ("llama", "llama_3_3_70b_instruct_turbo")]
DEFAULT_KEY = "dirichlet_b0.5_l30.0_nli"


def mcnemar_yates(a, b):
    """a, b: 0/1 vectors (naive, variant). Returns n01, n10, chi2, p."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    assert a.shape == b.shape
    n01 = int(np.sum(~a & b))   # naive wrong, variant right
    n10 = int(np.sum(a & ~b))   # naive right, variant wrong
    n = n01 + n10
    if n == 0:
        return n01, n10, 0.0, 1.0
    stat = max(0.0, abs(n01 - n10) - 1) ** 2 / n  # Yates; clamp matters only when n01 == n10
    return n01, n10, float(stat), float(chi2.sf(stat, df=1))


def main():
    out = {"_meta": {
        "comparison": "Dir-NLI (beta=0.5, lambda=30, NLI evidence) vs Naive; "
                      "full retrieval set (PopQA n=14,267); McNemar with Yates "
                      "continuity correction, p via chi2.sf",
        "alpha_27": ALPHA_27,
        "source": "results/{ds}_nli_voting_{llm}.json['_per_question'] (run_main.py)",
        "n01_definition": "Naive wrong, Dir-NLI correct (paired)",
        "n10_definition": "Naive correct, Dir-NLI wrong (paired)",
        "generated_by": "code/compute_nli_mcnemar.py",
    }}
    grid_summary = {}
    sig_improve_total = sig_decrease_total = 0
    for ds in DATASETS:
        for short, slug in LLMS:
            path = os.path.join(RESULTS_DIR, f"{ds}_nli_voting_{slug}.json")
            with open(path) as f:
                res = json.load(f)
            pq = res["_per_question"]
            naive = pq["naive"]
            default = pq[DEFAULT_KEY]
            # Gate: per-query means must reproduce the stored EM fields.
            em_naive_pq = 100.0 * float(np.mean(naive))
            em_dir_pq = 100.0 * float(np.mean(default))
            em_naive_json = 100.0 * res["naive"]["EM"]
            em_dir_json = 100.0 * res[DEFAULT_KEY]["EM"]
            assert abs(em_naive_pq - em_naive_json) < 1e-6, (ds, short, em_naive_pq, em_naive_json)
            assert abs(em_dir_pq - em_dir_json) < 1e-6, (ds, short, em_dir_pq, em_dir_json)

            n01, n10, stat, p = mcnemar_yates(naive, default)
            cell = {
                "n": len(naive),
                "EM_naive": round(em_naive_pq, 4),
                "EM_variant": round(em_dir_pq, 4),
                "dEM_vs_naive": round(em_dir_pq - em_naive_pq, 4),
                "n01_naive_wrong_variant_right": n01,
                "n10_naive_right_variant_wrong": n10,
                "chi2_yates": round(stat, 4),
                "p_value": p,
                "sig_alpha27": bool(p < ALPHA_27),
                "direction": ("improve" if n01 > n10 else "decrease" if n10 > n01 else "tie"),
            }
            out[f"{short}_{ds}"] = cell

            # Full grid sweep (24 dirichlet configs).
            keys = [k for k in pq if k.startswith("dirichlet_b") and k.endswith("_nli")]
            assert len(keys) == 24, (ds, short, len(keys))
            n_imp = n_dec = 0
            best_key, best_d = None, -1e9
            for k in keys:
                a01, a10, _, pk = mcnemar_yates(naive, pq[k])
                d = 100.0 * (np.mean(pq[k]) - np.mean(naive))
                if d > best_d:
                    best_d, best_key = d, k
                if pk < ALPHA_27:
                    if a01 > a10:
                        n_imp += 1
                    else:
                        n_dec += 1
            sig_improve_total += n_imp
            sig_decrease_total += n_dec
            grid_summary[f"{short}_{ds}"] = {
                "n_configs": len(keys),
                "sig_improvements_alpha27": n_imp,
                "sig_decreases_alpha27": n_dec,
                "best_config": best_key,
                "best_dEM_vs_naive": round(float(best_d), 4),
            }
    out["_grid_sweep"] = {
        "description": "All 24 (beta, lambda) NLI configurations x 9 cells = 216 runs; "
                       "Bonferroni-significant (alpha/27) improvements/decreases vs Naive",
        "total_runs": 9 * 24,
        "total_sig_improvements": sig_improve_total,
        "total_sig_decreases": sig_decrease_total,
        "cells": grid_summary,
    }
    # Default-config tallies.
    cells = [v for k, v in out.items() if not k.startswith("_")]
    out["_meta"]["default_summary"] = {
        "cells": len(cells),
        "n_positive_dEM": sum(c["dEM_vs_naive"] > 0 for c in cells),
        "n_sig_improve": sum(c["sig_alpha27"] and c["direction"] == "improve" for c in cells),
        "n_sig_decrease": sum(c["sig_alpha27"] and c["direction"] == "decrease" for c in cells),
        "mean_dEM": round(float(np.mean([c["dEM_vs_naive"] for c in cells])), 4),
    }
    out_path = os.path.join(RESULTS_DIR, "dir_nli_vs_naive_mcnemar.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}")
    print(json.dumps(out["_meta"]["default_summary"], indent=1))
    print("grid:", {k: v for k, v in out["_grid_sweep"].items() if k != "cells"})
    for k, c in out.items():
        if k.startswith("_"):
            continue
        print(f"{k:15s} n={c['n']:6d} EM {c['EM_naive']:6.2f}->{c['EM_variant']:6.2f} "
              f"d={c['dEM_vs_naive']:+6.2f} n01={c['n01_naive_wrong_variant_right']:4d} "
              f"n10={c['n10_naive_right_variant_wrong']:4d} chi2={c['chi2_yates']:8.2f} "
              f"p={c['p_value']:.2e} sig={c['sig_alpha27']} {c['direction']}")
    for k, g in grid_summary.items():
        print(f"  grid {k:15s} imp={g['sig_improvements_alpha27']} dec={g['sig_decreases_alpha27']} best={g['best_config']} ({g['best_dEM_vs_naive']:+.2f})")


if __name__ == "__main__":
    sys.exit(main())
