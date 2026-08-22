"""
D: Concat-prompt baseline (3 LLMs × 3 datasets = 9 cells)

Compares against the standard concat-prompt RAG paradigm:
  Standard concat: top-10 docs concatenated → 1 LLM call → answer
  Our voting:      top-10 docs × 1 LLM call each → weighted vote

NEW LLM inference required (separate cache, prefix CONCAT).

Usage:
    # Set env vars per LLM:
    export LLM_BACKEND=together
    export LLM_MODEL="Qwen/Qwen2.5-7B-Instruct-Turbo"
    export TOGETHER_API_KEY=...

    cd code/
    python3 run_concat_baseline.py --dataset nq

    # For all 3 datasets:
    python3 run_concat_baseline.py --dataset all

Outputs:
    data/llm_cache_concat_{LLM}.json   (separate from per-doc cache)
    results/concat_baseline_{LLM}_{dataset}.json
"""

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests as http_requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import exact_match
from config import (
    DATASETS, RESULTS_DIR, DATA_DIR,
    LLM_BACKEND, TOGETHER_API_KEY, TOGETHER_MODEL, TOGETHER_API_URL,
    OPENAI_API_KEY, OPENAI_MODEL,
)

# ============================================================
# Constants
# ============================================================

NUM_DOCS = 10
DOC_TRUNC = 800           # per-doc char truncation (matches main paper)
MAX_TOKENS = 50           # output tokens
NUM_PARALLEL = 8          # parallel API workers


# ============================================================
# Cache (separate from per-doc cache)
# ============================================================

def get_model_short():
    """Short name for cache file (matches main paper convention)."""
    if LLM_BACKEND == "openai":
        return OPENAI_MODEL.lower().replace("-", "_").replace(".", "_")
    else:
        return TOGETHER_MODEL.split("/")[-1].lower().replace("-", "_").replace(".", "_")


def get_cache_file():
    return os.path.join(DATA_DIR, f"llm_cache_concat_{get_model_short()}.json")


_cache = None
_cache_dirty = False


def load_cache():
    global _cache
    fn = get_cache_file()
    if os.path.exists(fn):
        with open(fn, "r", encoding="utf-8") as f:
            _cache = json.load(f)
        print(f"  Concat cache loaded: {len(_cache)} entries from {os.path.basename(fn)}")
    else:
        _cache = {}
        print(f"  Concat cache: starting fresh ({os.path.basename(fn)})")


def save_cache():
    """Atomic write with snapshot to avoid race condition with concurrent writers."""
    global _cache_dirty
    if not _cache_dirty:
        return
    fn = get_cache_file()
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    tmp = fn + ".tmp"
    # Snapshot: copy to a fresh dict so iteration during json.dump is safe
    # even if other threads mutate _cache.
    snapshot = dict(_cache)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False)
    os.replace(tmp, fn)
    _cache_dirty = False


def cache_key(model_name, question, docs):
    """MD5 hash of (model, CONCAT_K10, question, concatenated truncated docs).

    Distinct prefix CONCAT_K10 ensures no collision with per-doc cache.
    """
    combined = "|||".join(d["text"][:DOC_TRUNC] for d in docs)
    raw = f"{model_name}|||CONCAT_K{NUM_DOCS}|||{question}|||{combined}"
    return hashlib.md5(raw.encode()).hexdigest()


# ============================================================
# Concat prompt + LLM call
# ============================================================

def build_concat_prompt(question, docs):
    """top-10 docs concatenated into a single context."""
    doc_blocks = []
    for i, d in enumerate(docs[:NUM_DOCS]):
        doc_blocks.append(f"Document {i+1}: {d['text'][:DOC_TRUNC]}")
    context = "\n\n".join(doc_blocks)
    return (
        f"Based on the following documents, answer the question in a few words.\n\n"
        f"{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


def call_api(prompt):
    """Single API call with retries."""
    is_openai = (LLM_BACKEND == "openai")
    if is_openai:
        api_url = "https://api.openai.com/v1/chat/completions"
        api_key = OPENAI_API_KEY
        model = OPENAI_MODEL
    else:
        api_url = TOGETHER_API_URL
        api_key = TOGETHER_API_KEY
        model = TOGETHER_MODEL

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
    }

    for attempt in range(4):
        try:
            resp = http_requests.post(
                api_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=120,
            )
            # Detect rate-limit (HTTP 429) explicitly, with longer backoff
            if resp.status_code == 429:
                wait_s = 30 * (attempt + 1)  # 30s, 60s, 90s, 120s
                print(f"  HTTP 429 rate-limit (attempt {attempt+1}); sleeping {wait_s}s")
                time.sleep(wait_s)
                continue

            result = resp.json()

            if "error" in result:
                msg = result["error"].get("message", str(result["error"]))
                # Heuristic: rate-limit messages
                if "rate" in msg.lower() or "429" in msg:
                    wait_s = 30 * (attempt + 1)
                    print(f"  API rate-limit (attempt {attempt+1}); sleeping {wait_s}s: {msg[:120]}")
                    time.sleep(wait_s)
                else:
                    print(f"  API error (attempt {attempt+1}): {msg[:200]}")
                    time.sleep(2 ** attempt)
                continue

            answer = result["choices"][0]["message"]["content"].strip()
            answer = answer.split("\n")[0].strip()
            return answer

        except Exception as e:
            print(f"  API exception (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)

    return "unknown"


def generate_concat_answer(question, docs, model_name):
    """Cached concat-prompt generation."""
    global _cache, _cache_dirty
    key = cache_key(model_name, question, docs)
    if key in _cache:
        return _cache[key]
    prompt = build_concat_prompt(question, docs)
    answer = call_api(prompt)
    _cache[key] = {"answer_text": answer}
    _cache_dirty = True
    return _cache[key]


# ============================================================
# Main pipeline
# ============================================================

def run_dataset(dataset_name, save_every=200):
    """Run concat-prompt baseline on one dataset."""
    print(f"\n=== Concat baseline: {dataset_name} ===")
    print(f"  Backend: {LLM_BACKEND}")
    print(f"  Model: {TOGETHER_MODEL if LLM_BACKEND != 'openai' else OPENAI_MODEL}")

    # Load data
    test_file = DATASETS[dataset_name]
    with open(test_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    n_total = len(data)
    print(f"  Test queries: {n_total}")

    # Load cache
    load_cache()

    model_name = TOGETHER_MODEL if LLM_BACKEND != "openai" else OPENAI_MODEL

    # Filter to k=10 only (matches main paper)
    valid_indices = [i for i, s in enumerate(data) if len(s.get("retrieved_docs", [])) == 10]
    print(f"  Valid queries (k=10): {len(valid_indices)}")

    # Identify uncached queries
    uncached = []
    for q_idx in valid_indices:
        sample = data[q_idx]
        key = cache_key(model_name, sample["question"], sample["retrieved_docs"])
        if key not in _cache:
            uncached.append(q_idx)

    n_cached = len(valid_indices) - len(uncached)
    print(f"  Cached: {n_cached}, To generate: {len(uncached)}")

    # Generate uncached in parallel
    if uncached:
        t_gen_start = time.time()
        n_done = 0

        def _generate(q_idx):
            sample = data[q_idx]
            return q_idx, generate_concat_answer(
                sample["question"], sample["retrieved_docs"], model_name
            )

        with ThreadPoolExecutor(max_workers=NUM_PARALLEL) as pool:
            futures = {pool.submit(_generate, i): i for i in uncached}
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    print(f"  Worker exception: {e}")
                n_done += 1
                if n_done % 100 == 0:
                    elapsed = time.time() - t_gen_start
                    rate = n_done / max(elapsed, 0.1)
                    remaining = (len(uncached) - n_done) / max(rate, 0.01)
                    print(f"    Generated {n_done}/{len(uncached)} | rate={rate:.1f}/s | ETA={remaining/60:.1f}min")
                if n_done % save_every == 0:
                    save_cache()
                    print(f"    [cache saved at {n_done}]")

        save_cache()
        print(f"  Generation done in {(time.time()-t_gen_start)/60:.1f} min")

    # Compute EM/F1
    print(f"  Computing metrics...")
    em_count = 0
    n_evaluated = 0
    predictions = []

    for q_idx in valid_indices:
        sample = data[q_idx]
        gold = sample.get("answers", [])
        if not gold:
            continue
        entry = generate_concat_answer(sample["question"], sample["retrieved_docs"], model_name)
        pred = entry["answer_text"]
        is_correct = exact_match(pred, gold)
        em_count += is_correct
        n_evaluated += 1
        predictions.append({
            "q_idx": q_idx,
            "question": sample["question"],
            "gold": gold,
            "pred": pred,
            "em": is_correct,
        })

    em_rate = em_count / max(n_evaluated, 1)

    result = {
        "dataset": dataset_name,
        "method": "concat_top10",
        "n_evaluated": n_evaluated,
        "n_total": n_total,
        "EM": em_rate,
        "doc_truncation_chars": DOC_TRUNC,
        "k": NUM_DOCS,
        "model": model_name,
        "backend": LLM_BACKEND,
        "timestamp": datetime.now().isoformat(),
        # Per-query predictions for downstream McNemar / Bootstrap CI
        "predictions": predictions,
    }

    print(f"  EM = {em_rate:.4f} ({em_count}/{n_evaluated})")

    # Save
    out_file = os.path.join(
        RESULTS_DIR, f"concat_baseline_{get_model_short()}_{dataset_name}.json"
    )
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {os.path.basename(out_file)}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="nq", choices=["nq", "triviaqa", "popqa", "all"])
    args = parser.parse_args()

    datasets = ["nq", "triviaqa", "popqa"] if args.dataset == "all" else [args.dataset]

    print(f"=== D: Concat-prompt baseline ===")
    for ds in datasets:
        try:
            run_dataset(ds)
        except KeyboardInterrupt:
            print("\nInterrupted; saving cache...")
            save_cache()
            sys.exit(1)
        except Exception as e:
            print(f"  ERROR on {ds}: {e}")
            save_cache()
            raise


if __name__ == "__main__":
    main()
