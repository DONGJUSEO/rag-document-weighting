"""
Compute four additional baselines (cache-only; no new API calls).
=================================================================
1. Evidence-only: w_i = e_i / sum_j e_j (no retrieval prior).
2. CE-Rerank + Uniform: hard top-k by cross-encoder, uniform weights.
3. Random weights: Dirichlet(1, ..., 1) weights averaged over 100 trials.
4. Oracle: fraction of queries whose top-10 per-document answers contain the gold.

Usage: python3 run_additional_baselines.py --model qwen
"""
import json
import os
import sys
import numpy as np
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATASETS, RESULTS_DIR, DATA_DIR, SEED
from generation import load_cache, generate_all_doc_answers
from weighting import naive_weights
from metrics import (
    exact_match, f1_score, expected_calibration_error,
    normalize_answer, compute_aurc, risk_at_coverage,
)

np.random.seed(SEED)

EVIDENCE_CACHE_DIR = os.path.join(DATA_DIR, "evidence_cache")


def load_evidence_cache(dataset, method):
    cache_file = os.path.join(EVIDENCE_CACHE_DIR, f"{dataset}_{method}.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)
    return None


def aggregate_vote(cached_entries, weights):
    answers = [e["answer_text"] for e in cached_entries]
    total_weight = float(sum(weights))
    answer_scores = {}
    for answer, weight in zip(answers, weights):
        normed = normalize_answer(answer)
        if normed and normed != "unknown":
            answer_scores[normed] = answer_scores.get(normed, 0.0) + weight
    if not answer_scores:
        return "", 0.0
    best_normed = max(answer_scores, key=answer_scores.get)
    confidence = answer_scores[best_normed] / total_weight if total_weight > 0 else 0.0
    for answer in answers:
        if normalize_answer(answer) == best_normed:
            return answer, confidence
    return best_normed, confidence


def compute_metrics(preds, ground_truths, confs):
    from sklearn.metrics import roc_auc_score
    ems = [exact_match(p, g) for p, g in zip(preds, ground_truths)]
    f1s = [f1_score(p, g) for p, g in zip(preds, ground_truths)]
    avg_em = float(np.mean(ems))
    avg_f1 = float(np.mean(f1s))
    conf_arr = np.clip(np.array(confs, dtype=float), 0.0, 1.0)
    ece = expected_calibration_error(conf_arr.tolist(), ems)
    auroc = float(roc_auc_score(ems, conf_arr.tolist())) if len(set(ems)) >= 2 else 0.5
    aurc = compute_aurc(conf_arr.tolist(), ems)
    r08 = risk_at_coverage(conf_arr.tolist(), ems, 0.8)
    r09 = risk_at_coverage(conf_arr.tolist(), ems, 0.9)
    brier = float(np.mean((conf_arr - np.array(ems, dtype=float)) ** 2))
    return {
        "EM": avg_em, "F1": avg_f1, "ECE": ece, "AUROC": auroc,
        "AURC": aurc, "Risk@0.8": r08, "Risk@0.9": r09, "Brier": brier,
        "conf_mean": float(conf_arr.mean()), "conf_std": float(conf_arr.std()),
    }


def evidence_only_weights(evidence_scores):
    total = sum(evidence_scores)
    if total <= 0:
        return [1.0 / len(evidence_scores)] * len(evidence_scores)
    return [e / total for e in evidence_scores]


def ce_rerank_uniform_weights(evidence_scores, top_k=5):
    """Uniform weights on the top-k cross-encoder documents, zero elsewhere."""
    k = len(evidence_scores)
    top_k = min(top_k, k)
    indices = sorted(range(k), key=lambda i: evidence_scores[i], reverse=True)[:top_k]
    weights = [0.0] * k
    for idx in indices:
        weights[idx] = 1.0 / top_k
    return weights


def run_baselines(dataset_name, model_tag):
    print(f"\n{'='*70}")
    print(f"[{datetime.now():%H:%M:%S}] {dataset_name} — Additional Baselines ({model_tag})")
    print(f"{'='*70}")

    with open(DATASETS[dataset_name], "r", encoding="utf-8") as f:
        data = json.load(f)
    n = len(data)
    print(f"  Loaded {n} questions")

    load_cache()

    print("  Loading cached LLM answers...")
    all_cached = []
    for sample in tqdm(data, desc="Cache", leave=False):
        cached = generate_all_doc_answers(sample["question"], sample["retrieved_docs"])
        all_cached.append(cached)

    ground_truths = [s["answers"] for s in data]
    results = {}

    # ============================================================
    # 1. Oracle
    # ============================================================
    print("\n  [1] Oracle upper bound...")
    oracle_count = 0
    for i, sample in enumerate(data):
        cached = all_cached[i]
        for entry in cached:
            ans = entry["answer_text"]
            if exact_match(ans, ground_truths[i]):
                oracle_count += 1
                break
    oracle_rate = oracle_count / n
    results["oracle"] = {"rate": oracle_rate, "count": oracle_count, "total": n}
    print(f"    Oracle@any: {oracle_rate:.4f} ({oracle_count}/{n})")

    # ============================================================
    # 2. Random weights (100 trials)
    # ============================================================
    print("\n  [2] Random weights (100 trials)...")
    rng = np.random.default_rng(SEED)
    random_ems, random_eces, random_briers = [], [], []

    for trial in range(100):
        preds, confs = [], []
        for i, sample in enumerate(data):
            k = len(sample["retrieved_docs"])
            w = rng.dirichlet(np.ones(k)).tolist()
            ans, conf = aggregate_vote(all_cached[i], w)
            preds.append(ans)
            confs.append(conf)
        ems_trial = [exact_match(p, g) for p, g in zip(preds, ground_truths)]
        random_ems.append(float(np.mean(ems_trial)))
        random_eces.append(expected_calibration_error(confs, ems_trial))
        random_briers.append(float(np.mean((np.array(confs) - np.array(ems_trial, dtype=float)) ** 2)))

    results["random"] = {
        "EM_mean": float(np.mean(random_ems)), "EM_std": float(np.std(random_ems)),
        "ECE_mean": float(np.mean(random_eces)), "ECE_std": float(np.std(random_eces)),
        "Brier_mean": float(np.mean(random_briers)), "Brier_std": float(np.std(random_briers)),
    }
    print(f"    EM={np.mean(random_ems):.4f}±{np.std(random_ems):.4f}")
    print(f"    ECE={np.mean(random_eces):.4f}±{np.std(random_eces):.4f}")

    # ============================================================
    # 3. Evidence-only (3 evidence types)
    # ============================================================
    for ev_method in ["cross_encoder", "embedding_stability", "nli"]:
        print(f"\n  [3] Evidence-only ({ev_method})...")
        evidence = load_evidence_cache(dataset_name, ev_method)
        if not evidence:
            # Fall back to the main result file when the evidence cache is missing.
            ev_file = os.path.join(EVIDENCE_CACHE_DIR, f"{dataset_name}_{ev_method}.json")
            if not os.path.exists(ev_file):
                print(f"    Evidence cache not found, skipping")
                continue
            with open(ev_file) as ef:
                evidence = json.load(ef)

        preds, confs = [], []
        for i, sample in enumerate(data):
            w = evidence_only_weights(evidence[i])
            ans, conf = aggregate_vote(all_cached[i], w)
            preds.append(ans)
            confs.append(conf)
        m = compute_metrics(preds, ground_truths, confs)
        results[f"evidence_only_{ev_method}"] = m
        print(f"    EM={m['EM']:.4f}  F1={m['F1']:.4f}  ECE={m['ECE']:.4f}  AUROC={m['AUROC']:.4f}  Brier={m['Brier']:.4f}")

    # ============================================================
    # 4. CE Rerank + Uniform (top-5)
    # ============================================================
    print(f"\n  [4] CE Rerank + Uniform (top-5)...")
    ce_evidence = load_evidence_cache(dataset_name, "cross_encoder")
    if ce_evidence:
        preds, confs = [], []
        for i, sample in enumerate(data):
            w = ce_rerank_uniform_weights(ce_evidence[i], top_k=5)
            ans, conf = aggregate_vote(all_cached[i], w)
            preds.append(ans)
            confs.append(conf)
        m = compute_metrics(preds, ground_truths, confs)
        results["ce_rerank_uniform_top5"] = m
        print(f"    EM={m['EM']:.4f}  F1={m['F1']:.4f}  ECE={m['ECE']:.4f}  AUROC={m['AUROC']:.4f}  Brier={m['Brier']:.4f}")
    else:
        print(f"    CE evidence cache not found, skipping")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen", choices=["qwen", "llama", "llama70b", "gpt4o"])
    args = parser.parse_args()

    model_configs = {
        "qwen": {"env": {"LLM_BACKEND": "together", "LLM_MODEL": "Qwen/Qwen2.5-7B-Instruct-Turbo"}, "tag": "qwen2_5_7b_instruct_turbo"},
        "llama": {"env": {"LLM_BACKEND": "together", "LLM_MODEL": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"}, "tag": "meta_llama_3_1_8b_instruct_turbo"},
        "llama70b": {"env": {"LLM_BACKEND": "together", "LLM_MODEL": "meta-llama/Llama-3.3-70B-Instruct-Turbo"}, "tag": "llama_3_3_70b_instruct_turbo"},
        "gpt4o": {"env": {"LLM_BACKEND": "openai", "LLM_MODEL": "gpt-4.1-mini"}, "tag": "gpt_4_1_mini"},
    }

    cfg = model_configs[args.model]
    # Set env vars first, then reload config/generation so they pick up the change.
    for k, v in cfg.get("env", {}).items():
        os.environ[k] = v

    import importlib
    import config
    importlib.reload(config)
    import generation
    importlib.reload(generation)

    print(f"{'='*70}")
    print(f"Additional baselines - {args.model}")
    print(f"{'='*70}")

    all_results = {}
    for ds in ["nq", "triviaqa", "popqa"]:
        all_results[ds] = run_baselines(ds, cfg["tag"])

    # Save
    output_file = os.path.join(RESULTS_DIR, f"additional_baselines_{cfg['tag']}.json")

    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=convert, ensure_ascii=False)

    print(f"\nSaved: {output_file}")
    print("Done!")


if __name__ == "__main__":
    main()
