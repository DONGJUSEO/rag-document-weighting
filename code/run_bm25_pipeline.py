"""
E6: BM25 retriever robustness pipeline (Qwen × 3 datasets)

Workflow:
  1. Load BM25 Pyserini index (Lucene backend)
  2. Retrieve top-10 docs per query (with per-query cache)
  3. Lazy doc text lookup via searcher.doc(docid).raw() — no 24GB lookup dict
  4. Generate Qwen-7B answer per (q, doc) — cached separately from main
  5. CE evidence per (q, doc) — batched per query (10 pairs at once)
  6. BM25 score min-max normalized to [0,1] for compatibility with replug/dirichlet
  7. Apply framework: Naive / SimW / Dir-CE / EO-CE
  8. Compute EM

Important design choices:
  - BM25 scores → min-max normalized so SimW (REPLUG-style) is meaningful
    (raw BM25 unbounded scores cause weight to collapse on top-1 doc)
  - psg_lookup removed → lazy lookup via Lucene index (saves ~24 GB RAM)
  - CE batched per query (10 pairs) → ~5x speedup vs. one-pair calls
  - API key validated before expensive operations
  - Atomic write for retrieval cache (.tmp + os.replace)
  - Skip-count logged transparently

Usage:
    cd code/
    export TOGETHER_API_KEY=...
    python3 run_bm25_pipeline.py --dataset all
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import requests as http_requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import normalize_answer, exact_match
from weighting import naive_weights, replug_weights, dirichlet_weights

from config import hf_revision  # recorded HF commits (PIN_MODEL_REVISIONS=1)

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CODE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

INDEX_DIR = os.path.join(DATA_DIR, "bm25_pyserini_index")

TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "")
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct-Turbo"

BM25_LLM_CACHE = os.path.join(DATA_DIR, "llm_cache_bm25_qwen.json")
BM25_CE_CACHE = os.path.join(DATA_DIR, "ce_cache_bm25.json")
BM25_RETRIEVAL_CACHE = os.path.join(DATA_DIR, "bm25_retrieval_cache.json")

NUM_DOCS = 10
DOC_TRUNC_LLM = 800
DOC_TRUNC_CE = 1000
BETA = 0.5
LAMBDA_DIR_CE = 30.0
NUM_PARALLEL = 8

DATASET_FILES = {
    "nq": "nq_test.json",
    "triviaqa": "triviaqa_test.json",
    "popqa": "popqa_contriever.json",
}


# ============================================================
# Atomic JSON write helper
# ============================================================

def atomic_json_dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


# ============================================================
# Pyserini retrieval (with caching + lazy doc lookup)
# ============================================================

_searcher = None


def get_searcher():
    global _searcher
    if _searcher is None:
        from pyserini.search.lucene import LuceneSearcher
        print(f"  [pyserini] Loading index: {INDEX_DIR}", flush=True)
        _searcher = LuceneSearcher(INDEX_DIR)
        _searcher.set_bm25(k1=0.9, b=0.4)  # Lucene/Pyserini defaults
    return _searcher


def fetch_doc_text(docid):
    """Lazy lookup of doc text via Lucene index. Avoids 24GB psg_lookup dict."""
    searcher = get_searcher()
    raw = searcher.doc(docid).raw()
    obj = json.loads(raw)
    # JSONL was built with {"id": ..., "contents": "title text"}
    return obj.get("contents", "")


def bm25_retrieve_all(questions, k=NUM_DOCS):
    """Retrieve top-k for all queries via Pyserini, with on-disk cache."""
    searcher = get_searcher()

    cache = {}
    if os.path.exists(BM25_RETRIEVAL_CACHE):
        with open(BM25_RETRIEVAL_CACHE, "r", encoding="utf-8") as f:
            cache = json.load(f)

    results = {}
    new_count = 0
    t0 = time.time()
    for i, q in enumerate(questions):
        q_key = hashlib.md5(q.encode()).hexdigest()
        if q_key in cache:
            results[q] = cache[q_key]
            continue
        hits = searcher.search(q, k=k)
        results[q] = [{"id": h.docid, "score": float(h.score)} for h in hits]
        cache[q_key] = results[q]
        new_count += 1
        if new_count % 500 == 0:
            atomic_json_dump(cache, BM25_RETRIEVAL_CACHE)
            elapsed = time.time() - t0
            rate = new_count / max(elapsed, 0.01)
            print(f"    [retrieve] {new_count} new, total cached={len(cache)}, {rate:.1f}/s", flush=True)

    if new_count > 0:
        atomic_json_dump(cache, BM25_RETRIEVAL_CACHE)
    print(f"  Retrieval done: {len(questions)} queries ({new_count} new) in {(time.time()-t0)/60:.1f}min", flush=True)
    return results


# ============================================================
# LLM cache (separate from main)
# ============================================================

_llm_cache = None
_llm_cache_dirty = False


def load_llm_cache():
    global _llm_cache
    if os.path.exists(BM25_LLM_CACHE):
        with open(BM25_LLM_CACHE, "r", encoding="utf-8") as f:
            _llm_cache = json.load(f)
    else:
        _llm_cache = {}


def save_llm_cache():
    global _llm_cache_dirty
    if not _llm_cache_dirty:
        return
    snapshot = dict(_llm_cache)  # snapshot to avoid race during dump
    atomic_json_dump(snapshot, BM25_LLM_CACHE)
    _llm_cache_dirty = False


def llm_cache_key(question, doc_text):
    raw = f"{LLM_MODEL}|||BM25|||{question}|||{doc_text[:DOC_TRUNC_LLM]}"
    return hashlib.md5(raw.encode()).hexdigest()


def call_qwen(question, doc_text):
    prompt = (
        f"Based on the following context, answer the question in a few words.\n\n"
        f"Context: {doc_text[:DOC_TRUNC_LLM]}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 50,
        "temperature": 0,
    }
    for attempt in range(4):
        try:
            resp = http_requests.post(
                TOGETHER_API_URL,
                headers={"Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json"},
                json=payload, timeout=60,
            )
            if resp.status_code == 429:
                time.sleep(30 * (attempt + 1))
                continue
            result = resp.json()
            if "error" in result:
                msg = result["error"].get("message", "")
                if "rate" in msg.lower():
                    time.sleep(30 * (attempt + 1))
                    continue
                time.sleep(2 ** attempt)
                continue
            ans = result["choices"][0]["message"]["content"].strip().split("\n")[0].strip()
            return ans
        except Exception:
            time.sleep(2 ** attempt)
    return "unknown"


def get_llm_answer(question, doc_text):
    global _llm_cache, _llm_cache_dirty
    key = llm_cache_key(question, doc_text)
    if key in _llm_cache:
        return _llm_cache[key]
    ans = call_qwen(question, doc_text)
    _llm_cache[key] = ans
    _llm_cache_dirty = True
    return ans


# ============================================================
# CE evidence (batched per query)
# ============================================================

_ce_model = None


def load_ce_model():
    global _ce_model
    if _ce_model is None:
        from sentence_transformers import CrossEncoder
        _ce_model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            revision=hf_revision("cross-encoder/ms-marco-MiniLM-L-6-v2"),  # recorded commit when PIN_MODEL_REVISIONS=1
        )
    return _ce_model


def ce_score_batch(question, doc_texts):
    """Compute CE scores for one query and 10 docs in a single batch.

    Returns: list[float] of sigmoid-applied scores in [0, 1].
    """
    model = load_ce_model()
    pairs = [(question, dt[:DOC_TRUNC_CE]) for dt in doc_texts]
    raw = model.predict(pairs)
    # Apply sigmoid; raw is np.ndarray
    return [1.0 / (1.0 + math.exp(-float(s))) for s in raw]


def ce_cache_key(question, doc_text):
    raw = f"{question}|||{doc_text[:DOC_TRUNC_CE]}"
    return hashlib.md5(raw.encode()).hexdigest()


# ============================================================
# Pipeline
# ============================================================

def run_dataset(dataset, save_every=200):
    print(f"\n=== BM25 pipeline: {dataset} ===")

    # Load test data
    test_file = os.path.join(DATA_DIR, DATASET_FILES[dataset])
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    questions = [s["question"] for s in test_data]
    print(f"  Queries: {len(questions)}")

    # 1. BM25 retrieval (with lazy doc lookup later)
    retrieval = bm25_retrieve_all(questions, k=NUM_DOCS)

    # Stats
    n_under_k = sum(1 for hits in retrieval.values() if len(hits) < NUM_DOCS)
    if n_under_k:
        print(f"  WARNING: {n_under_k} queries have < {NUM_DOCS} BM25 hits (excluded from EM)")

    # 2. Build augmented samples (text fetched lazily per (q, doc) pair when needed)
    augmented = []
    for s in test_data:
        q = s["question"]
        hits = retrieval.get(q, [])
        docs = []
        for h in hits[:NUM_DOCS]:
            docs.append({"id": h["id"], "score": float(h["score"])})  # text deferred
        augmented.append({
            "question": q,
            "answers": s.get("answers", []),
            "docs": docs,
        })

    # 3. Build text cache for unique doc IDs (still memory-bounded; only retrieved docs)
    print("  Lazy fetching unique doc texts via Lucene...")
    unique_ids = sorted({d["id"] for s in augmented for d in s["docs"]})
    print(f"    Unique doc ids: {len(unique_ids):,}")
    text_cache = {}
    for i, did in enumerate(unique_ids):
        text_cache[did] = fetch_doc_text(did)
        if (i + 1) % 50000 == 0:
            print(f"    fetched {i+1}/{len(unique_ids)}", flush=True)
    print(f"  Doc texts loaded: {len(text_cache):,}")

    # 4. LLM answers (parallel)
    load_llm_cache()
    pairs_to_gen = []
    for sample in augmented:
        q = sample["question"]
        for d in sample["docs"]:
            text = text_cache.get(d["id"], "")
            key = llm_cache_key(q, text)
            if key not in _llm_cache:
                pairs_to_gen.append((q, text))
    n_total_pairs = sum(len(s["docs"]) for s in augmented)
    print(f"  LLM gen: {len(pairs_to_gen)}/{n_total_pairs} new (cached: {n_total_pairs - len(pairs_to_gen)})")

    if pairs_to_gen:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=NUM_PARALLEL) as pool:
            futs = {pool.submit(get_llm_answer, q, t): i for i, (q, t) in enumerate(pairs_to_gen)}
            done = 0
            for fut in as_completed(futs):
                try: fut.result()
                except Exception: pass
                done += 1
                if done % 500 == 0:
                    elapsed = time.time() - t0
                    rate = done / max(elapsed, 0.01)
                    eta = (len(pairs_to_gen) - done) / max(rate, 0.01)
                    print(f"    LLM {done}/{len(pairs_to_gen)} | rate={rate:.1f}/s | ETA={eta/60:.1f}min", flush=True)
                if done % save_every == 0:
                    save_llm_cache()
        save_llm_cache()
        print(f"  LLM gen done in {(time.time()-t0)/60:.1f}min")

    # 5. CE evidence (batched per query)
    print("  CE scoring (batched per query)...")
    ce_cache = {}
    if os.path.exists(BM25_CE_CACHE):
        with open(BM25_CE_CACHE, "r", encoding="utf-8") as f:
            ce_cache = json.load(f)

    ce_by_q = {}
    new_ce = 0
    t0 = time.time()
    for q_idx, sample in enumerate(augmented):
        q = sample["question"]
        doc_texts = [text_cache.get(d["id"], "") for d in sample["docs"]]
        # Check cache
        keys = [ce_cache_key(q, t) for t in doc_texts]
        if all(k in ce_cache for k in keys):
            ce_by_q[q] = [ce_cache[k] for k in keys]
            continue
        # Compute batch
        scores = ce_score_batch(q, doc_texts)
        for k, s in zip(keys, scores):
            ce_cache[k] = s
        ce_by_q[q] = scores
        new_ce += len(scores)
        if (q_idx + 1) % 500 == 0:
            atomic_json_dump(ce_cache, BM25_CE_CACHE)
            elapsed = time.time() - t0
            rate = (q_idx + 1) / max(elapsed, 0.01)
            print(f"    CE {q_idx+1}/{len(augmented)} | rate={rate:.1f} q/s", flush=True)

    if new_ce > 0:
        atomic_json_dump(ce_cache, BM25_CE_CACHE)
    print(f"  CE done in {(time.time()-t0)/60:.1f}min, {new_ce} new pairs")

    # 6. Compute EM with 4 methods
    print("  Computing EM for 4 methods...")
    em_counts = {"naive": 0, "simw": 0, "dir_ce": 0, "eo_ce": 0}
    n_eval = 0
    skipped_no_gold = 0
    skipped_under_k = 0
    per_query = []

    for sample in augmented:
        q = sample["question"]
        gold = sample.get("answers", [])
        if not gold:
            skipped_no_gold += 1
            continue
        if len(sample["docs"]) != NUM_DOCS:
            skipped_under_k += 1
            continue

        # BM25 raw scores
        sims_raw = [d["score"] for d in sample["docs"]]
        # Min-max normalize to [0,1] for compatibility with exp(beta * s) prior
        smin, smax = min(sims_raw), max(sims_raw)
        if smax > smin:
            sims = [(s - smin) / (smax - smin) for s in sims_raw]
        else:
            sims = [0.5] * len(sims_raw)
        evs = ce_by_q.get(q, [0.5] * NUM_DOCS)

        weights = {
            "naive": naive_weights(NUM_DOCS),
            "simw": replug_weights(sims, beta=BETA),
            "dir_ce": dirichlet_weights(sims, evs, beta=BETA, lam=LAMBDA_DIR_CE),
            "eo_ce": [e / max(sum(evs), 1e-9) for e in evs],
        }

        # Cached LLM answers
        cached_answers = []
        for d in sample["docs"]:
            t = text_cache.get(d["id"], "")
            ans = _llm_cache.get(llm_cache_key(q, t), "unknown")
            cached_answers.append(ans)

        gold_norm = {normalize_answer(g) for g in gold if g}

        # Vote per method
        method_preds = {}
        for method, w in weights.items():
            vote = defaultdict(float)
            for i, ans in enumerate(cached_answers):
                normed = normalize_answer(ans)
                if normed and normed != "unknown":  # same rule as generation.aggregate_vote: abstentions carry no vote
                    vote[normed] += w[i]
            if not vote:
                method_preds[method] = ""  # all documents abstained: counted as incorrect
                continue
            pred = max(vote, key=vote.get)  # ties: first answer in retrieval order
            method_preds[method] = pred
            if pred in gold_norm:
                em_counts[method] += 1

        n_eval += 1
        per_query.append({
            "question": q, "gold": gold, "predictions": method_preds,
            "n_unique_answers": len({normalize_answer(a) for a in cached_answers}),
        })

    em = {m: em_counts[m] / max(n_eval, 1) for m in em_counts}

    result = {
        "dataset": dataset,
        "retriever": "BM25 (Pyserini, Lucene)",
        "llm": LLM_MODEL,
        "n_evaluated": n_eval,
        "n_total": len(test_data),
        "n_skipped_no_gold": skipped_no_gold,
        "n_skipped_under_k": skipped_under_k,
        "EM": em,
        "k": NUM_DOCS,
        "beta": BETA,
        "lambda_dir_ce": LAMBDA_DIR_CE,
        "bm25_normalize": "min-max to [0,1] before exp(beta*s)",
        "timestamp": datetime.now().isoformat(),
        "per_query": per_query[:100],  # save first 100 only to keep file small
    }

    print(f"  EM: " + ", ".join(f"{m}={em[m]:.4f}" for m in ["naive", "simw", "dir_ce", "eo_ce"]))
    print(f"  Used {n_eval}/{len(test_data)} (skipped: no_gold={skipped_no_gold}, under_k={skipped_under_k})")

    out = os.path.join(RESULTS_DIR, f"bm25_pipeline_qwen_{dataset}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    atomic_json_dump(result, out)
    print(f"  Saved: {out}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="nq", choices=["nq", "triviaqa", "popqa", "all"])
    args = parser.parse_args()

    # Validate API key BEFORE expensive operations
    if not TOGETHER_API_KEY:
        sys.exit("ERROR: TOGETHER_API_KEY not set. Aborting before expensive operations.")

    # Validate index dir
    if not os.path.isdir(INDEX_DIR):
        sys.exit(f"ERROR: BM25 index not found at {INDEX_DIR}. Build it first.")

    datasets = ["nq", "triviaqa", "popqa"] if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        try:
            run_dataset(ds)
        except KeyboardInterrupt:
            save_llm_cache()
            sys.exit(1)


if __name__ == "__main__":
    main()
