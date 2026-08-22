"""
C1: Answer-support weight mass

Definition: For each query, compute the sum of weights assigned to documents
whose generated answer matches the gold answer (after normalization).

  mass = sum_{i : normalize(a_i) in gold_set} w_i

Computed for 4 methods across 9 (LLM, dataset) cells, using cached LLM answers
and pre-computed evidence/similarity scores. NO new LLM inference.

Mechanism interpretation: if Dir-CE > Naive in mass, then CE evidence redirects
weight toward answer-bearing documents, supporting the framework's mechanism
claim (D1 attack defense).

Usage:
    cd code/
    python3 compute_c1_answer_support_mass.py [--llm qwen|gpt|llama|all] [--dataset nq|tqa|popqa|all]

Outputs:
    results/c1_answer_support_mass.json
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

# Local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import normalize_answer
from weighting import naive_weights, replug_weights, dirichlet_weights

# ============================================================
# Paths and configuration
# ============================================================

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CODE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
EVIDENCE_DIR = os.path.join(DATA_DIR, "evidence_cache")

# LLM cache + cache key model name (used in MD5)
LLM_CONFIG = {
    "qwen2_5_7b": {
        "cache_file": os.path.join(DATA_DIR, "llm_cache_qwen2_5_7b_instruct_turbo.json"),
        "model_name": "Qwen/Qwen2.5-7B-Instruct-Turbo",
    },
    "gpt_4_1_mini": {
        "cache_file": os.path.join(DATA_DIR, "llm_cache_gpt_4_1_mini.json"),
        "model_name": "gpt-4.1-mini",
    },
    "llama_3_3_70b": {
        "cache_file": os.path.join(DATA_DIR, "llm_cache_llama_3_3_70b_instruct_turbo.json"),
        "model_name": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    },
}

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

# Default hyperparameters (matching main paper)
BETA = 0.5
LAMBDA_DIR_CE = 30.0


# ============================================================
# Cache key (must match generation.py exactly)
# ============================================================

def cache_key(model_name, question, doc_text):
    """MD5 hash matching generation._cache_key."""
    raw = f"{model_name}|||{question}|||{doc_text[:800]}"
    return hashlib.md5(raw.encode()).hexdigest()


# ============================================================
# Weight functions
# ============================================================

def evidence_only_weights(evidence_scores):
    """
    Evidence-only weighting: w_i = e_i / sum_j e_j.
    This is the lambda -> infinity limit of dirichlet_weights.
    """
    total = sum(evidence_scores)
    if total <= 0:
        # fall back to uniform if all evidence is zero
        k = len(evidence_scores)
        return [1.0 / k] * k
    return [e / total for e in evidence_scores]


# ============================================================
# Per-cell computation
# ============================================================

def compute_cell(llm_key, dataset_key, verbose=True):
    """
    Compute answer-support mass for one (LLM, dataset) cell across 4 methods.

    Returns:
        dict: {
            'naive': {'mean': ..., 'std': ..., 'n': ...},
            'simw': {...},
            'dir_ce': {...},
            'eo_ce': {...},
            'cache_hit_rate': float,
            'n_skipped': int,
        }
    """
    llm_cfg = LLM_CONFIG[llm_key]
    ds_cfg = DATASET_CONFIG[dataset_key]

    # 1. Load test data
    with open(ds_cfg["test_file"], "r", encoding="utf-8") as f:
        test_data = json.load(f)
    n_total = len(test_data)
    if verbose:
        print(f"  Test data: {n_total} queries")

    # 2. Load evidence (CE) cache
    with open(ds_cfg["evidence_file"], "r", encoding="utf-8") as f:
        evidence_cache = json.load(f)
    assert len(evidence_cache) == n_total, \
        f"evidence size {len(evidence_cache)} != test size {n_total}"

    # 3. Load LLM cache
    with open(llm_cfg["cache_file"], "r", encoding="utf-8") as f:
        llm_cache = json.load(f)
    if verbose:
        print(f"  LLM cache: {len(llm_cache)} entries")

    model_name = llm_cfg["model_name"]

    # 4. Per-query computation
    masses = {"naive": [], "simw": [], "dir_ce": [], "eo_ce": []}
    n_cache_hits = 0
    n_lookups = 0
    n_skipped = 0

    # PopQA: filter queries with k != 10 (matching paper's k=10 setting)
    for q_idx, sample in enumerate(test_data):
        question = sample["question"]
        gold_answers = sample.get("answers", [])
        docs = sample.get("retrieved_docs", [])

        if len(docs) != 10:
            # PopQA has some queries with <10 docs; paper excludes these at k=10
            n_skipped += 1
            continue

        if not gold_answers:
            n_skipped += 1
            continue

        # Lookup answers from cache
        cached_answers = []
        cache_miss = False
        for doc in docs:
            key = cache_key(model_name, question, doc["text"])
            n_lookups += 1
            if key in llm_cache:
                n_cache_hits += 1
                entry = llm_cache[key]
                if isinstance(entry, str):
                    answer_text = entry
                else:
                    answer_text = entry.get("answer_text", "")
                cached_answers.append(answer_text)
            else:
                # Cache miss → cannot compute, skip query
                cache_miss = True
                break

        if cache_miss:
            n_skipped += 1
            continue

        # Normalize gold answers
        gold_norm = {normalize_answer(g) for g in gold_answers if g}

        # Compute weights for 4 methods
        sims = [float(d["score"]) for d in docs]
        evs = [float(e) for e in evidence_cache[q_idx]]

        weights_dict = {
            "naive": naive_weights(10),
            "simw": replug_weights(sims, beta=BETA),
            "dir_ce": dirichlet_weights(sims, evs, beta=BETA, lam=LAMBDA_DIR_CE),
            "eo_ce": evidence_only_weights(evs),
        }

        # For each method, compute mass
        for method, w in weights_dict.items():
            # Sanity: weights sum to 1
            assert abs(sum(w) - 1.0) < 1e-6, f"weights sum != 1 for {method}"

            mass = 0.0
            for i in range(10):
                ans_norm = normalize_answer(cached_answers[i])
                if ans_norm in gold_norm:
                    mass += w[i]
            assert 0.0 <= mass <= 1.0 + 1e-6, f"mass {mass} out of [0, 1]"
            masses[method].append(mass)

    # 5. Aggregate
    cache_hit_rate = n_cache_hits / max(n_lookups, 1)
    n_used = len(masses["naive"])

    result = {}
    for method, vals in masses.items():
        if vals:
            arr = np.array(vals)
            result[method] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)),
                "median": float(np.median(arr)),
                "n": int(len(arr)),
            }
        else:
            result[method] = {"mean": None, "std": None, "median": None, "n": 0}

    result["cache_hit_rate"] = cache_hit_rate
    result["n_skipped"] = n_skipped
    result["n_total"] = n_total
    result["n_used"] = n_used

    if verbose:
        print(f"  Cache hit rate: {cache_hit_rate:.4f}")
        print(f"  Used: {n_used}/{n_total} (skipped: {n_skipped})")
        print(f"  Mass means: ", end="")
        print(", ".join(f"{m}={result[m]['mean']:.4f}" for m in ["naive", "simw", "dir_ce", "eo_ce"]))

    return result


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", default="all", choices=["qwen", "gpt", "llama", "all"])
    parser.add_argument("--dataset", default="all", choices=["nq", "tqa", "popqa", "all"])
    parser.add_argument("--output", default=os.path.join(RESULTS_DIR, "c1_answer_support_mass.json"))
    args = parser.parse_args()

    llm_map = {
        "qwen": ["qwen2_5_7b"],
        "gpt": ["gpt_4_1_mini"],
        "llama": ["llama_3_3_70b"],
        "all": ["qwen2_5_7b", "gpt_4_1_mini", "llama_3_3_70b"],
    }
    ds_map = {
        "nq": ["nq"],
        "tqa": ["triviaqa"],
        "popqa": ["popqa"],
        "all": ["nq", "triviaqa", "popqa"],
    }

    llms = llm_map[args.llm]
    datasets = ds_map[args.dataset]

    print(f"=== C1: Answer-support weight mass ===")
    print(f"LLMs: {llms}")
    print(f"Datasets: {datasets}")
    print(f"Output: {args.output}")
    print()

    results = {}
    t_start = time.time()

    for llm_key in llms:
        results[llm_key] = {}
        for ds_key in datasets:
            print(f"--- {llm_key} / {ds_key} ---")
            t0 = time.time()
            cell = compute_cell(llm_key, ds_key, verbose=True)
            t1 = time.time()
            print(f"  ({t1 - t0:.1f}s)")
            print()
            results[llm_key][ds_key] = cell

    results["_meta"] = {
        "evidence": "cross_encoder",
        "beta": BETA,
        "lambda_dir_ce": LAMBDA_DIR_CE,
        "eo_ce": "direct_normalize (e_i / sum e_j)",
        "k": 10,
        "timestamp": datetime.now().isoformat(),
        "elapsed_sec": time.time() - t_start,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"=== Saved: {args.output} ===")
    print(f"Total elapsed: {time.time() - t_start:.1f}s")

    # Print summary table
    print()
    print("=== Summary (mean mass per cell) ===")
    print(f"{'LLM':<15} {'Dataset':<10} {'Naive':<8} {'SimW':<8} {'Dir-CE':<8} {'EO-CE':<8}")
    print("-" * 60)
    for llm_key in llms:
        for ds_key in datasets:
            cell = results[llm_key][ds_key]
            row = f"{llm_key:<15} {ds_key:<10}"
            for m in ["naive", "simw", "dir_ce", "eo_ce"]:
                v = cell[m]["mean"]
                row += f" {v:.4f} " if v is not None else " ---    "
            print(row)


if __name__ == "__main__":
    main()
