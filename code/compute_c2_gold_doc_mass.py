"""
C2: Gold-doc weight mass

Definition: For each query, compute the sum of weights assigned to documents
labeled as 'gold' in the retrieval set. The 'gold' label is the dataset-native
annotation (at most one gold document per query; the same label as the paper's
has-gold split, Sec. 5.2) -- stricter than the answer-containment ratio of Sec. 4.1.

  mass = sum_{i : docs[i]['type'] == 'gold'} w_i

Complementary to C1 (answer-support mass):
- C1 measures answer-LEVEL alignment (LLM produces gold answer).
- C2 measures retrieval-LEVEL alignment (weight mass on the annotated gold doc).

If Dir-CE > Naive in BOTH C1 and C2, the framework operates at multiple levels.

NO new LLM inference (uses retrieval labels + similarities + evidence cache).

Usage:
    cd code/
    python3 compute_c2_gold_doc_mass.py

Outputs:
    results/c2_gold_doc_mass.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weighting import naive_weights, replug_weights, dirichlet_weights

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CODE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
EVIDENCE_DIR = os.path.join(DATA_DIR, "evidence_cache")

DATASET_CONFIG = {
    "nq": {
        "test_file": os.path.join(DATA_DIR, "nq_cosine.json"),
        "evidence_file": os.path.join(EVIDENCE_DIR, "nq_cross_encoder.json"),
    },
    "triviaqa": {
        "test_file": os.path.join(DATA_DIR, "triviaqa_cosine.json"),
        "evidence_file": os.path.join(EVIDENCE_DIR, "triviaqa_cross_encoder.json"),
    },
    "popqa": {
        "test_file": os.path.join(DATA_DIR, "popqa_contriever.json"),
        "evidence_file": os.path.join(EVIDENCE_DIR, "popqa_cross_encoder.json"),
    },
}

BETA = 0.5
LAMBDA_DIR_CE = 30.0


def evidence_only_weights(evidence_scores):
    total = sum(evidence_scores)
    if total <= 0:
        k = len(evidence_scores)
        return [1.0 / k] * k
    return [e / total for e in evidence_scores]


def compute_cell(dataset_key, verbose=True):
    ds_cfg = DATASET_CONFIG[dataset_key]

    with open(ds_cfg["test_file"], "r", encoding="utf-8") as f:
        test_data = json.load(f)
    n_total = len(test_data)

    with open(ds_cfg["evidence_file"], "r", encoding="utf-8") as f:
        evidence_cache = json.load(f)
    assert len(evidence_cache) == n_total

    masses = {"naive": [], "simw": [], "dir_ce": [], "eo_ce": []}
    n_skipped = 0
    gold_count_per_query = []

    for q_idx, sample in enumerate(test_data):
        docs = sample.get("retrieved_docs", [])
        if len(docs) != 10:
            n_skipped += 1
            continue

        # Mark gold positions
        gold_mask = [d.get("type") == "gold" for d in docs]
        gold_count_per_query.append(sum(gold_mask))

        sims = [float(d["score"]) for d in docs]
        evs = [float(e) for e in evidence_cache[q_idx]]

        weights_dict = {
            "naive": naive_weights(10),
            "simw": replug_weights(sims, beta=BETA),
            "dir_ce": dirichlet_weights(sims, evs, beta=BETA, lam=LAMBDA_DIR_CE),
            "eo_ce": evidence_only_weights(evs),
        }

        for method, w in weights_dict.items():
            assert abs(sum(w) - 1.0) < 1e-6
            mass = sum(w[i] for i in range(10) if gold_mask[i])
            assert 0.0 <= mass <= 1.0 + 1e-6
            masses[method].append(mass)

    n_used = len(masses["naive"])
    gold_arr = np.array(gold_count_per_query)

    result = {}
    for method, vals in masses.items():
        arr = np.array(vals)
        result[method] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "median": float(np.median(arr)),
            "n": int(len(arr)),
        }

    result["n_skipped"] = n_skipped
    result["n_total"] = n_total
    result["n_used"] = n_used
    result["gold_doc_ratio"] = float(gold_arr.mean() / 10.0)  # avg fraction of gold docs in top-10

    if verbose:
        print(f"  Dataset: {dataset_key}, n={n_used}/{n_total}, gold_ratio={result['gold_doc_ratio']:.3f}")
        print(f"  Mass means: ", end="")
        print(", ".join(f"{m}={result[m]['mean']:.4f}" for m in ["naive", "simw", "dir_ce", "eo_ce"]))

    return result


def main():
    output = os.path.join(RESULTS_DIR, "c2_gold_doc_mass.json")
    print(f"=== C2: Gold-doc weight mass ===")
    print(f"Output: {output}\n")

    results = {}
    t_start = time.time()

    for ds_key in ["nq", "triviaqa", "popqa"]:
        print(f"--- {ds_key} ---")
        results[ds_key] = compute_cell(ds_key, verbose=True)
        print()

    results["_meta"] = {
        "evidence": "cross_encoder",
        "beta": BETA,
        "lambda_dir_ce": LAMBDA_DIR_CE,
        "k": 10,
        "note": "C2 is LLM-independent (only uses retrieval labels + similarities + evidence)",
        "timestamp": datetime.now().isoformat(),
        "elapsed_sec": time.time() - t_start,
    }

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"=== Saved: {output} ===")

    print(f"\n=== Summary (mean gold-doc mass per dataset) ===")
    print(f"{'Dataset':<12} {'GoldRatio':<12} {'Naive':<8} {'SimW':<8} {'Dir-CE':<8} {'EO-CE':<8}")
    print("-" * 60)
    for ds_key in ["nq", "triviaqa", "popqa"]:
        cell = results[ds_key]
        gr = cell["gold_doc_ratio"]
        row = f"{ds_key:<12} {gr:.3f}        "
        for m in ["naive", "simw", "dir_ce", "eo_ce"]:
            row += f"{cell[m]['mean']:.4f}  "
        print(row)


if __name__ == "__main__":
    main()
