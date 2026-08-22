"""
Recompute PopQA voting metrics on k=10 strict subset (13,659 queries).

Issue: Original voting code processed all 14,267 PopQA queries with variable k
(some queries had k=8 or k=9). Paper claims "13,659 queries used at k=10",
but result JSON shows 14,267 num_total. This script fixes the inconsistency
by re-evaluating voting only on the 13,659 queries with exactly 10 docs.

NO new LLM inference (uses existing per-doc cache).
NO new evidence inference (uses existing CE/ES/NLI cache).

Output: results/popqa_k10_strict_voting_{LLM}.json (fresh files)

Usage:
    cd code/
    python3 recompute_popqa_k10.py
"""

import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import normalize_answer, exact_match
from weighting import naive_weights, replug_weights, dirichlet_weights

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CODE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
EVIDENCE_DIR = os.path.join(DATA_DIR, "evidence_cache")

LLM_CONFIG = {
    "qwen2_5_7b": {
        "cache_file": os.path.join(DATA_DIR, "llm_cache_qwen2_5_7b_instruct_turbo.json"),
        "model_name": "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "short": "qwen2_5_7b_instruct_turbo",
    },
    "gpt_4_1_mini": {
        "cache_file": os.path.join(DATA_DIR, "llm_cache_gpt_4_1_mini.json"),
        "model_name": "gpt-4.1-mini",
        "short": "gpt_4_1_mini",
    },
    "llama_3_3_70b": {
        "cache_file": os.path.join(DATA_DIR, "llm_cache_llama_3_3_70b_instruct_turbo.json"),
        "model_name": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "short": "llama_3_3_70b_instruct_turbo",
    },
}

EVIDENCE_FILES = {
    "cross_encoder": "popqa_cross_encoder.json",
    "embedding_stability": "popqa_embedding_stability.json",
    "nli": "popqa_nli.json",
}

POPQA_FILE = os.path.join(DATA_DIR, "popqa_contriever.json")
NUM_DOCS = 10


def cache_key(model_name, question, doc_text):
    raw = f"{model_name}|||{question}|||{doc_text[:800]}"
    return hashlib.md5(raw.encode()).hexdigest()


def evidence_only(evs):
    s = sum(evs)
    if s <= 0:
        return [1.0 / len(evs)] * len(evs)
    return [e / s for e in evs]


def vote(answers, weights):
    vmass = defaultdict(float)
    for a, w in zip(answers, weights):
        vmass[normalize_answer(a)] += w
    if not vmass:
        return "", 0.0
    pred = max(vmass, key=vmass.get)
    return pred, vmass[pred]


def compute_cell(llm_key, evidence_type):
    """Compute EM for one (LLM, evidence) cell on PopQA k=10 only."""
    print(f"\n--- {llm_key} / {evidence_type} ---")
    cfg = LLM_CONFIG[llm_key]

    # Load PopQA + filter to k=10
    with open(POPQA_FILE, "r", encoding="utf-8") as f:
        popqa = json.load(f)
    print(f"  Total queries: {len(popqa)}")

    # Indices into the original list, keeping only k=10
    valid_indices = [i for i, s in enumerate(popqa) if len(s.get("retrieved_docs", [])) == 10]
    print(f"  k=10 queries: {len(valid_indices)}")

    # Load evidence cache (aligned with original test order)
    evidence_path = os.path.join(EVIDENCE_DIR, EVIDENCE_FILES[evidence_type])
    with open(evidence_path, "r", encoding="utf-8") as f:
        evidence_cache = json.load(f)
    assert len(evidence_cache) == len(popqa), \
        f"Evidence size {len(evidence_cache)} != popqa size {len(popqa)}"

    # Load LLM cache
    with open(cfg["cache_file"], "r", encoding="utf-8") as f:
        llm_cache = json.load(f)
    print(f"  LLM cache: {len(llm_cache)} entries")

    model_name = cfg["model_name"]

    # Methods to compute
    methods_to_compute = ["naive", "simw", f"dir_{evidence_type[:2]}", f"eo_{evidence_type[:2]}"]
    em_counts = {m: 0 for m in methods_to_compute}

    # Statistical: per-query correctness for McNemar
    per_query_correct = {m: [] for m in methods_to_compute}

    n_eval = 0
    cache_miss = 0

    for q_idx in valid_indices:
        sample = popqa[q_idx]
        question = sample["question"]
        gold = sample.get("answers", [])
        docs = sample["retrieved_docs"]

        if not gold:
            continue

        # Lookup answers
        cached_answers = []
        miss = False
        for d in docs:
            key = cache_key(model_name, question, d["text"])
            if key not in llm_cache:
                miss = True
                break
            entry = llm_cache[key]
            ans = entry if isinstance(entry, str) else entry.get("answer_text", "")
            cached_answers.append(ans)

        if miss:
            cache_miss += 1
            continue

        sims = [float(d["score"]) for d in docs]
        evs = [float(e) for e in evidence_cache[q_idx]]
        gold_norm = {normalize_answer(g) for g in gold if g}

        # 4 methods
        weights_dict = {
            "naive": naive_weights(NUM_DOCS),
            "simw": replug_weights(sims, beta=0.5),
            f"dir_{evidence_type[:2]}": dirichlet_weights(sims, evs, beta=0.5, lam=30.0),
            f"eo_{evidence_type[:2]}": evidence_only(evs),
        }

        for method, w in weights_dict.items():
            pred, _ = vote(cached_answers, w)
            is_correct = int(pred in gold_norm)
            em_counts[method] += is_correct
            per_query_correct[method].append({"q_idx": q_idx, "correct": is_correct})

        n_eval += 1

    em = {m: em_counts[m] / max(n_eval, 1) for m in em_counts}

    print(f"  n_eval: {n_eval} (cache_miss skipped: {cache_miss})")
    for m in methods_to_compute:
        print(f"    {m}: EM={em[m]:.4f} ({em_counts[m]}/{n_eval})")

    return {
        "llm": llm_key,
        "llm_short": cfg["short"],
        "evidence": evidence_type,
        "n_total_popqa": len(popqa),
        "n_k10": len(valid_indices),
        "n_evaluated": n_eval,
        "cache_miss": cache_miss,
        "em_counts": em_counts,
        "EM": em,
        "per_query_correct": per_query_correct,
        "config": {"beta": 0.5, "lambda": 30.0, "k": 10},
        "timestamp": datetime.now().isoformat(),
    }


def main():
    print("=== PopQA k=10 strict re-evaluation ===")
    print("Goal: align voting num_total with paper's stated 13,659 (k=10 only)")
    print()

    all_results = {}

    for llm_key in ["qwen2_5_7b", "gpt_4_1_mini", "llama_3_3_70b"]:
        all_results[llm_key] = {}
        for evidence in ["cross_encoder", "embedding_stability", "nli"]:
            cell = compute_cell(llm_key, evidence)
            all_results[llm_key][evidence] = cell

    # Save
    out = os.path.join(RESULTS_DIR, "popqa_k10_strict_voting.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n=== Saved: {out} ===")

    # Summary
    print(f"\n=== Summary: PopQA k=10 strict (n=13659) ===")
    print(f"{'LLM':<22} {'Evid':<6} {'Naive':<7} {'SimW':<7} {'Dir':<7} {'EO':<7}")
    print('-' * 60)
    for llm_key in ["qwen2_5_7b", "gpt_4_1_mini", "llama_3_3_70b"]:
        for evid in ["cross_encoder", "embedding_stability", "nli"]:
            r = all_results[llm_key][evid]
            ev_short = evid[:2]
            em = r["EM"]
            row = f"{llm_key:<22} {ev_short:<6}"
            row += f" {em['naive']*100:>5.2f} "
            row += f" {em['simw']*100:>5.2f} "
            row += f" {em[f'dir_{ev_short}']*100:>5.2f} "
            row += f" {em[f'eo_{ev_short}']*100:>5.2f}"
            print(row)


if __name__ == "__main__":
    main()
