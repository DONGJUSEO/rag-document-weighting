"""
Compute and archive average weight entropy for Dir-CE at default config (beta=0.5, lambda=30, CE evidence).
Referenced in main.tex §5.4: NQ 2.073, TQA 1.989, PopQA 2.087.
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATASETS, DATA_DIR, RESULTS_DIR
from weighting import dirichlet_weights

EVIDENCE_CACHE = os.path.join(DATA_DIR, "evidence_cache")
DATASETS_LIST = ["nq", "triviaqa", "popqa"]
BETA = 0.5
LAMBDA = 30
EPS = 1e-12


def shannon_entropy(w):
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    return float(-np.sum(w * np.log(w + EPS)))


def main():
    results = {"config": {"beta": BETA, "lambda": LAMBDA, "evidence": "cross_encoder"},
               "uniform_entropy": float(np.log(10)),  # 2.3026
               "per_dataset": {}}

    for ds in DATASETS_LIST:
        with open(DATASETS[ds], "r", encoding="utf-8") as f:
            data = json.load(f)
        with open(os.path.join(EVIDENCE_CACHE, f"{ds}_cross_encoder.json"), "r") as f:
            evidence_cache = json.load(f)

        entropies = []
        skipped = 0
        # evidence_cache is list-of-lists, aligned with data by index
        for idx, sample in enumerate(data):
            sims = [d["score"] for d in sample["retrieved_docs"]]
            if idx >= len(evidence_cache):
                skipped += 1
                continue
            evs = evidence_cache[idx]
            if isinstance(evs, list) and len(evs) == len(sims):
                w = dirichlet_weights(sims, evs, beta=BETA, lam=LAMBDA)
                entropies.append(shannon_entropy(w))
            else:
                skipped += 1

        mean_H = float(np.mean(entropies)) if entropies else None
        reduction_pct = (np.log(10) - mean_H) / np.log(10) * 100 if mean_H else None
        results["per_dataset"][ds] = {
            "mean_entropy": mean_H,
            "reduction_vs_uniform_pct": reduction_pct,
            "n_queries": len(entropies),
            "n_skipped": skipped,
        }
        print(f"{ds}: mean H = {mean_H:.3f}, reduction vs uniform = {reduction_pct:.1f}% (n={len(entropies)}, skipped={skipped})")

    out = os.path.join(RESULTS_DIR, "weight_entropy.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
