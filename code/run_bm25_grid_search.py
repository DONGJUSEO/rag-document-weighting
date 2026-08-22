"""
E6-C: BM25 grid search over (beta, lambda).

Reuses existing caches (LLM answers, CE evidence, BM25 retrieval) — no new
inference. Computes EM for 4 beta × 6 lambda + REPLUG (lam=0) + EO-CE (lam=inf)
across 3 datasets, all on Qwen-7B.

Goal: identify if a non-default (beta, lambda) recovers Dir-CE > EO-CE on BM25,
or whether EO-CE remains the BM25-optimal regime.

Usage:
    cd code/
    python3 run_bm25_grid_search.py

Output:
    results/bm25_grid_search.json
"""

import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import normalize_answer
from weighting import naive_weights, replug_weights, dirichlet_weights

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CODE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

INDEX_DIR = os.path.join(DATA_DIR, "bm25_pyserini_index")
BM25_LLM_CACHE = os.path.join(DATA_DIR, "llm_cache_bm25_qwen.json")
BM25_CE_CACHE = os.path.join(DATA_DIR, "ce_cache_bm25.json")
BM25_RETRIEVAL_CACHE = os.path.join(DATA_DIR, "bm25_retrieval_cache.json")

LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct-Turbo"
DOC_TRUNC_LLM = 800
DOC_TRUNC_CE = 1000
NUM_DOCS = 10

DATASET_FILES = {
    "nq": "nq_test.json",
    "triviaqa": "triviaqa_test.json",
    "popqa": "popqa_contriever.json",
}

BETAS = [0.5, 1.0, 2.0, 4.0]
LAMBDAS = [0.1, 0.5, 1.0, 3.0, 10.0, 30.0]


def llm_cache_key(question, doc_text):
    raw = f"{LLM_MODEL}|||BM25|||{question}|||{doc_text[:DOC_TRUNC_LLM]}"
    return hashlib.md5(raw.encode()).hexdigest()


def ce_cache_key(question, doc_text):
    raw = f"{question}|||{doc_text[:DOC_TRUNC_CE]}"
    return hashlib.md5(raw.encode()).hexdigest()


_searcher = None


def get_searcher():
    global _searcher
    if _searcher is None:
        from pyserini.search.lucene import LuceneSearcher
        _searcher = LuceneSearcher(INDEX_DIR)
        _searcher.set_bm25(k1=0.9, b=0.4)
    return _searcher


def fetch_doc_text(docid):
    searcher = get_searcher()
    raw = searcher.doc(docid).raw()
    obj = json.loads(raw)
    return obj.get("contents", "")


def vote_em(answers, weights, gold_norm):
    vote = defaultdict(float)
    for a, w in zip(answers, weights):
        vote[normalize_answer(a)] += w
    if not vote:
        return 0
    pred = max(vote, key=vote.get)
    return int(pred in gold_norm)


def evidence_only(evs):
    s = sum(evs)
    if s <= 0:
        return [1.0 / len(evs)] * len(evs)
    return [e / s for e in evs]


def run_grid(dataset):
    print(f"\n=== Grid search: {dataset} ===", flush=True)

    # Load data
    test_file = os.path.join(DATA_DIR, DATASET_FILES[dataset])
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"  Queries: {len(test_data)}", flush=True)

    # Load caches
    with open(BM25_RETRIEVAL_CACHE, "r", encoding="utf-8") as f:
        retrieval_cache = json.load(f)
    with open(BM25_LLM_CACHE, "r", encoding="utf-8") as f:
        llm_cache = json.load(f)
    with open(BM25_CE_CACHE, "r", encoding="utf-8") as f:
        ce_cache = json.load(f)
    print(f"  LLM cache: {len(llm_cache):,} | CE cache: {len(ce_cache):,}", flush=True)

    # Pre-compute per-query: BM25 sims, evs, cached answers, gold
    print("  Building per-query data (lazy doc fetch)...", flush=True)
    queries = []
    miss = 0
    for s in test_data:
        q = s["question"]
        gold = s.get("answers", [])
        if not gold:
            continue
        q_key = hashlib.md5(q.encode()).hexdigest()
        hits = retrieval_cache.get(q_key, [])
        if len(hits) != NUM_DOCS:
            continue
        # Fetch doc texts
        docs_text = []
        sims_raw = []
        ok = True
        for h in hits:
            try:
                t = fetch_doc_text(h["id"])
            except Exception:
                ok = False
                break
            docs_text.append(t)
            sims_raw.append(float(h["score"]))
        if not ok:
            continue
        # Cached answers
        answers = []
        ce_evs = []
        ok = True
        for t in docs_text:
            k_llm = llm_cache_key(q, t)
            if k_llm not in llm_cache:
                ok = False
                break
            answers.append(llm_cache[k_llm])
            k_ce = ce_cache_key(q, t)
            if k_ce not in ce_cache:
                ok = False
                break
            ce_evs.append(ce_cache[k_ce])
        if not ok:
            miss += 1
            continue
        # Min-max normalize sims
        smin, smax = min(sims_raw), max(sims_raw)
        if smax > smin:
            sims = [(x - smin) / (smax - smin) for x in sims_raw]
        else:
            sims = [0.5] * len(sims_raw)
        gold_norm = {normalize_answer(g) for g in gold if g}
        queries.append({
            "answers": answers,
            "sims": sims,
            "evs": ce_evs,
            "gold_norm": gold_norm,
        })
    print(f"  Eval queries: {len(queries)} (miss: {miss})", flush=True)

    # Compute EM for each (beta, lambda) + REPLUG (lam=0) + EO-CE
    results = {}

    # Naive (β-,λ- independent)
    em_naive = sum(vote_em(q["answers"], naive_weights(NUM_DOCS), q["gold_norm"]) for q in queries) / max(len(queries), 1)
    results["naive"] = em_naive
    print(f"  Naive: {em_naive:.4f}", flush=True)

    # EO-CE (λ→∞)
    em_eo = sum(vote_em(q["answers"], evidence_only(q["evs"]), q["gold_norm"]) for q in queries) / max(len(queries), 1)
    results["eo_ce"] = em_eo
    print(f"  EO-CE: {em_eo:.4f}", flush=True)

    # SimW per beta (λ=0, REPLUG)
    for beta in BETAS:
        em = sum(vote_em(q["answers"], replug_weights(q["sims"], beta=beta), q["gold_norm"]) for q in queries) / max(len(queries), 1)
        results[f"simw_b{beta}"] = em

    # Dirichlet grid
    for beta in BETAS:
        for lam in LAMBDAS:
            em = sum(
                vote_em(q["answers"], dirichlet_weights(q["sims"], q["evs"], beta=beta, lam=lam), q["gold_norm"])
                for q in queries
            ) / max(len(queries), 1)
            results[f"dir_b{beta}_l{lam}"] = em
        print(f"  beta={beta} done", flush=True)

    return {
        "dataset": dataset,
        "n_eval": len(queries),
        "results": results,
        "betas": BETAS,
        "lambdas": LAMBDAS,
    }


def main():
    all_results = {}
    for ds in ["nq", "triviaqa", "popqa"]:
        all_results[ds] = run_grid(ds)

    out = os.path.join(RESULTS_DIR, "bm25_grid_search.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "all_results": all_results,
            "betas": BETAS,
            "lambdas": LAMBDAS,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "k": NUM_DOCS,
                "bm25_normalize": "min-max [0,1]",
                "ce_evidence": "cross-encoder/ms-marco-MiniLM-L-6-v2 sigmoid",
                "llm": LLM_MODEL,
            },
        }, f, indent=2, ensure_ascii=False)
    print(f"\n=== Saved: {out} ===", flush=True)

    # Summary table
    print("\n=== Summary: Best Dir-CE per dataset ===", flush=True)
    for ds in ["nq", "triviaqa", "popqa"]:
        r = all_results[ds]["results"]
        naive = r["naive"]
        eo = r["eo_ce"]
        # Best dir
        dir_keys = [k for k in r if k.startswith("dir_")]
        best_k = max(dir_keys, key=lambda k: r[k])
        best_em = r[best_k]
        print(f"  {ds}: Naive={naive:.4f}, EO-CE={eo:.4f}, BestDir={best_em:.4f} ({best_k})")


if __name__ == "__main__":
    main()
