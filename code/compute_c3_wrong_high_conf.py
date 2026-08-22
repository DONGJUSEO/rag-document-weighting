"""
C3: Wrong high-confidence rate

Definition: Fraction of queries where the model is WRONG but assigns HIGH
confidence (max vote mass > threshold).

  rate = P(prediction != gold AND confidence > tau)

Lower is better. Measures dangerous overconfidence — wrong answers given
with high vote concentration. Useful for selective prediction discussion.

Confidence = max_y (sum_{i: a_i = y} w_i) — the vote mass for the winning answer.

Usage:
    cd code/
    python3 compute_c3_wrong_high_conf.py

Outputs:
    results/c3_wrong_high_conf.json
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import normalize_answer
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

BETA = 0.5
LAMBDA_DIR_CE = 30.0
CONFIDENCE_THRESHOLD = 0.5  # high confidence = vote mass > 0.5


def cache_key(model_name, question, doc_text):
    raw = f"{model_name}|||{question}|||{doc_text[:800]}"
    return hashlib.md5(raw.encode()).hexdigest()


def evidence_only_weights(evidence_scores):
    total = sum(evidence_scores)
    if total <= 0:
        k = len(evidence_scores)
        return [1.0 / k] * k
    return [e / total for e in evidence_scores]


def vote(answers, weights):
    """Weighted majority vote.
    Returns (winning_answer, confidence) where confidence = winning vote mass.
    """
    vote_mass = defaultdict(float)
    for ans, w in zip(answers, weights):
        vote_mass[normalize_answer(ans)] += w
    if not vote_mass:
        return "", 0.0
    winner = max(vote_mass, key=vote_mass.get)
    return winner, vote_mass[winner]


def compute_cell(llm_key, dataset_key, verbose=True):
    llm_cfg = LLM_CONFIG[llm_key]
    ds_cfg = DATASET_CONFIG[dataset_key]

    with open(ds_cfg["test_file"], "r", encoding="utf-8") as f:
        test_data = json.load(f)

    with open(ds_cfg["evidence_file"], "r", encoding="utf-8") as f:
        evidence_cache = json.load(f)

    with open(llm_cfg["cache_file"], "r", encoding="utf-8") as f:
        llm_cache = json.load(f)

    model_name = llm_cfg["model_name"]
    n_total = len(test_data)

    # Per-method tracking
    methods = ["naive", "simw", "dir_ce", "eo_ce"]
    n_wrong_high_conf = {m: 0 for m in methods}
    n_correct_high_conf = {m: 0 for m in methods}
    n_total_used = 0
    confidences = {m: [] for m in methods}
    em_count = {m: 0 for m in methods}

    for q_idx, sample in enumerate(test_data):
        question = sample["question"]
        gold_answers = sample.get("answers", [])
        docs = sample.get("retrieved_docs", [])

        if len(docs) != 10 or not gold_answers:
            continue

        # Cache lookup
        cached_answers = []
        miss = False
        for doc in docs:
            key = cache_key(model_name, question, doc["text"])
            if key in llm_cache:
                entry = llm_cache[key]
                ans = entry if isinstance(entry, str) else entry.get("answer_text", "")
                cached_answers.append(ans)
            else:
                miss = True
                break
        if miss:
            continue

        gold_norm = {normalize_answer(g) for g in gold_answers if g}
        if not gold_norm:
            continue

        sims = [float(d["score"]) for d in docs]
        evs = [float(e) for e in evidence_cache[q_idx]]

        weights_dict = {
            "naive": naive_weights(10),
            "simw": replug_weights(sims, beta=BETA),
            "dir_ce": dirichlet_weights(sims, evs, beta=BETA, lam=LAMBDA_DIR_CE),
            "eo_ce": evidence_only_weights(evs),
        }

        n_total_used += 1
        for method, w in weights_dict.items():
            pred, conf = vote(cached_answers, w)
            confidences[method].append(conf)
            is_correct = pred in gold_norm
            if is_correct:
                em_count[method] += 1
                if conf > CONFIDENCE_THRESHOLD:
                    n_correct_high_conf[method] += 1
            else:
                if conf > CONFIDENCE_THRESHOLD:
                    n_wrong_high_conf[method] += 1

    result = {
        "n_total": n_total,
        "n_used": n_total_used,
        "threshold": CONFIDENCE_THRESHOLD,
    }
    for method in methods:
        n = n_total_used
        result[method] = {
            "wrong_high_conf_rate": n_wrong_high_conf[method] / max(n, 1),
            "correct_high_conf_rate": n_correct_high_conf[method] / max(n, 1),
            "em": em_count[method] / max(n, 1),
            "mean_confidence": float(np.mean(confidences[method])) if confidences[method] else None,
        }

    if verbose:
        print(f"  n={n_total_used}/{n_total}, threshold={CONFIDENCE_THRESHOLD}")
        for m in methods:
            r = result[m]
            print(f"    {m:<8} EM={r['em']:.3f}  wrong-high-conf={r['wrong_high_conf_rate']:.3f}  conf={r['mean_confidence']:.3f}")

    return result


def main():
    output = os.path.join(RESULTS_DIR, "c3_wrong_high_conf.json")
    print(f"=== C3: Wrong high-confidence rate (threshold > {CONFIDENCE_THRESHOLD}) ===")
    print(f"Output: {output}\n")

    results = {}
    t_start = time.time()

    for llm_key in ["qwen2_5_7b", "gpt_4_1_mini", "llama_3_3_70b"]:
        results[llm_key] = {}
        for ds_key in ["nq", "triviaqa", "popqa"]:
            print(f"--- {llm_key} / {ds_key} ---")
            results[llm_key][ds_key] = compute_cell(llm_key, ds_key, verbose=True)
            print()

    results["_meta"] = {
        "evidence": "cross_encoder",
        "beta": BETA,
        "lambda_dir_ce": LAMBDA_DIR_CE,
        "threshold": CONFIDENCE_THRESHOLD,
        "k": 10,
        "timestamp": datetime.now().isoformat(),
        "elapsed_sec": time.time() - t_start,
    }

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"=== Saved: {output} ===")

    # Aggregated summary across cells
    print(f"\n=== Wrong-high-conf rate (avg across 9 cells) ===")
    print(f"{'Method':<10} {'Avg wrong-hc':<14} {'Avg correct-hc':<16} {'Avg conf':<10}")
    print("-" * 55)
    for method in ["naive", "simw", "dir_ce", "eo_ce"]:
        wrongs = []
        corrects = []
        confs = []
        for llm_key in ["qwen2_5_7b", "gpt_4_1_mini", "llama_3_3_70b"]:
            for ds_key in ["nq", "triviaqa", "popqa"]:
                r = results[llm_key][ds_key][method]
                wrongs.append(r["wrong_high_conf_rate"])
                corrects.append(r["correct_high_conf_rate"])
                confs.append(r["mean_confidence"])
        print(f"{method:<10} {np.mean(wrongs):.4f}        {np.mean(corrects):.4f}          {np.mean(confs):.4f}")


if __name__ == "__main__":
    main()
