"""
3-seed repeated split for Naive and SimW baselines.
Uses identical seeds (42, 123, 456) and split logic as run_phase2_analysis.run_repeated_split
so results align cell-by-cell with the Dir-CE/Dir-ES splits.

Usage (run once per LLM; set LLM_MODEL so the cache key matches the cache file):
  LLM_BACKEND=together LLM_CACHE_FILE=llm_cache_qwen2_5_7b_instruct_turbo.json \
      python3 compute_naive_simw_splits.py qwen2_5_7b_instruct_turbo
  LLM_BACKEND=openai LLM_MODEL=gpt-4.1-mini \
      LLM_CACHE_FILE=llm_cache_gpt_4_1_mini.json \
      python3 compute_naive_simw_splits.py gpt_4_1_mini
  LLM_BACKEND=together LLM_MODEL=meta-llama/Llama-3.3-70B-Instruct-Turbo \
      LLM_CACHE_FILE=llm_cache_llama_3_3_70b_instruct_turbo.json \
      python3 compute_naive_simw_splits.py llama_3_3_70b_instruct_turbo
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATASETS, RESULTS_DIR
from generation import load_cache, generate_all_doc_answers
from weighting import naive_weights, replug_weights
from run_phase2_analysis import aggregate_vote, compute_all_metrics

SEEDS = [42, 123, 456]
DATASETS_LIST = ["nq", "triviaqa", "popqa"]
SIMW_BETA = 0.5


_CACHE_LOADED = False


def load_data_and_cache_light(dataset_name):
    """Load data + fill answer cache only (no LLM calls — cache must already exist).
    Cache is loaded once per process, shared across datasets."""
    global _CACHE_LOADED
    with open(DATASETS[dataset_name], "r", encoding="utf-8") as f:
        data = json.load(f)
    if not _CACHE_LOADED:
        load_cache()
        _CACHE_LOADED = True
    all_cached = []
    for sample in data:
        all_cached.append(
            generate_all_doc_answers(sample["question"], sample["retrieved_docs"])
        )
    ground_truths = [s["answers"] for s in data]
    return data, all_cached, ground_truths


def run_split(data, all_cached, ground_truths, dataset_name):
    n = len(data)
    results = {"naive": [], "simw": []}

    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        indices = np.arange(n)
        rng.shuffle(indices)
        test_idx = indices[n // 2:]
        test_gts = [ground_truths[i] for i in test_idx]

        # Naive (uniform)
        n_preds, n_confs = [], []
        for idx in test_idx:
            k = len(data[idx]["retrieved_docs"])
            w = naive_weights(k)
            a, c = aggregate_vote(all_cached[idx], w)
            n_preds.append(a); n_confs.append(c)
        m_n = compute_all_metrics(n_preds, test_gts, n_confs)
        m_n["seed"] = seed
        del m_n["per_em"]; del m_n["per_conf"]
        results["naive"].append(m_n)

        # SimW (REPLUG-style, beta=0.5, lambda=0)
        s_preds, s_confs = [], []
        for idx in test_idx:
            sims = [d["score"] for d in data[idx]["retrieved_docs"]]
            w = replug_weights(sims, beta=SIMW_BETA)
            a, c = aggregate_vote(all_cached[idx], w)
            s_preds.append(a); s_confs.append(c)
        m_s = compute_all_metrics(s_preds, test_gts, s_confs)
        m_s["seed"] = seed
        del m_s["per_em"]; del m_s["per_conf"]
        results["simw"].append(m_s)

        print(f"  [{dataset_name}] seed={seed}: Naive EM={m_n['EM']:.4f} | SimW EM={m_s['EM']:.4f}")

    # Mean and std across 3 seeds
    averages = {}
    for baseline in ["naive", "simw"]:
        avg = {}
        for metric in ["EM", "F1", "ECE", "AUROC", "AURC", "Brier"]:
            vals = [r[metric] for r in results[baseline]]
            avg[f"{metric}_mean"] = float(np.mean(vals))
            avg[f"{metric}_std"] = float(np.std(vals))
        averages[baseline] = avg
    return {"per_seed": results, "average": averages}


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 compute_naive_simw_splits.py <llm_key>")
        print("  llm_key e.g., qwen2_5_7b_instruct_turbo, gpt_4_1_mini, llama_3_3_70b_instruct_turbo")
        sys.exit(1)

    llm_key = sys.argv[1]
    all_results = {}
    for dataset_name in DATASETS_LIST:
        print(f"\n=== {llm_key} / {dataset_name} ===")
        data, all_cached, ground_truths = load_data_and_cache_light(dataset_name)
        result = run_split(data, all_cached, ground_truths, dataset_name)
        all_results[dataset_name] = result

    out_path = os.path.join(RESULTS_DIR, f"naive_simw_splits_{llm_key}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
