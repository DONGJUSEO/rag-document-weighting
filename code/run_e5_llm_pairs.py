#!/usr/bin/env python3
"""
Pairs-parallel per-document LLM answer runner for the E5/BGE experiment (script 4/4).

WHY: the paper's generation.py parallelizes only the 10 calls WITHIN one query and
walks queries sequentially — measured at ~1.28 pairs/s, i.e. ~66-69 h for
E5 x Llama-70B across 3 datasets. This runner flattens ALL (query, doc) pairs into
a single global queue and drives them through a ThreadPoolExecutor (I/O-bound API
calls; default 32 workers), targeting ~2-9 h wall-clock.

SAFETY: writes to a SEPARATE cache file (llm_cache_{slug}_{retriever}.json) so the
submitted DPR caches are never mutated. Cache key = md5(model|||question|||doc[:800]),
BYTE-IDENTICAL to generation._cache_key, so compute_gold_subset_analysis.py consumes
these answers unchanged — just point its DATASETS/LLM_CONFIGS at {ds}_e5.json and the
new cache. Prompt is imported from generation (_build_prompt), not re-typed, so it
cannot drift.

RESUMABLE: entries already in the cache are skipped, so a re-run continues where it
stopped (crash / rate-limit abort safe). temperature=0 for determinism; API-level
reproducibility is best-effort (providers do not guarantee bitwise-identical decoding).

Usage (after script 3 has written data/{ds}_{retriever}.json):
  python run_e5_llm_pairs.py --retriever e5  --model llama --datasets nq triviaqa popqa --workers 32
  python run_e5_llm_pairs.py --retriever e5  --model qwen  --datasets nq triviaqa popqa --workers 48
  python run_e5_llm_pairs.py --retriever bge --model llama --datasets nq triviaqa popqa --workers 32
--model is the LLM (llama/qwen/gpt); --retriever is the retrieval set (e5/bge). Reads
{ds}_{retriever}.json, writes llm_cache_{slug}_{retriever}.json (never the DPR cache).
If 429s appear frequently, lower --workers.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests as http_requests

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
sys.path.insert(0, os.path.join(BASE, "code"))


def _load_dotenv():
    """config.py reads os.environ ONLY (no python-dotenv), so inject .env here
    BEFORE importing config/generation. Without this every API call 401s with an
    empty key (caught by the local pilot). setdefault -> never overrides a real
    shell env var, and the .env stays git-ignored / local-only."""
    env = os.path.join(BASE, ".env")
    if os.path.exists(env):
        with open(env, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#") and "=" in s:
                    k, _, v = s.partition("=")
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

from generation import _build_prompt  # noqa: E402  (single source of truth for the prompt)
from config import TOGETHER_API_KEY, TOGETHER_API_URL  # noqa: E402

# model_key -> {backend, model (API name == cache-key model), slug (cache filename)}
MODELS = {
    "llama": {"backend": "together", "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
              "slug": "llama_3_3_70b_instruct_turbo"},
    "qwen":  {"backend": "together", "model": "Qwen/Qwen2.5-7B-Instruct-Turbo",
              "slug": "qwen2_5_7b_instruct_turbo"},
    "gpt":   {"backend": "openai",   "model": "gpt-4.1-mini",
              "slug": "gpt_4_1_mini"},
}

SAVE_EVERY = 2000       # persist the cache every N completed calls
MAX_TOKENS = 50


def cache_key(model_name, question, doc_text):
    """Identical to generation._cache_key (model|||question|||doc[:800])."""
    return hashlib.md5(f"{model_name}|||{question}|||{doc_text[:800]}".encode()).hexdigest()


def call_api(question, doc_text, backend, model_name):
    """One (question, doc) answer. Pure worker: no shared state, returns the entry."""
    prompt = _build_prompt(question, doc_text)
    if backend == "openai":
        from config import OPENAI_API_KEY
        url, key = "https://api.openai.com/v1/chat/completions", OPENAI_API_KEY
    else:
        url, key = TOGETHER_API_URL, TOGETHER_API_KEY

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
    }
    if backend == "openai":
        payload["logprobs"], payload["top_logprobs"] = True, 1
    else:
        payload["logprobs"] = 1

    for attempt in range(4):
        try:
            resp = http_requests.post(
                url,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload, timeout=60,
            )
            result = resp.json()
            if "error" in result:
                time.sleep(2 ** attempt)
                continue

            answer_text = result["choices"][0]["message"]["content"].strip().split("\n")[0].strip()

            lp = result["choices"][0].get("logprobs", {}) or {}
            if "token_logprobs" in lp:
                seq = sum(x for x in lp["token_logprobs"] if x is not None)
            elif "content" in lp:
                seq = sum(t.get("logprob", 0.0) for t in lp["content"])
            else:
                seq = -10.0
            return {"answer_text": answer_text, "sequence_logprob": seq}
        except Exception:
            time.sleep(2 ** attempt)

    return {"answer_text": "unknown", "sequence_logprob": -100.0}


def build_tasks(datasets, retriever, model_name, cache, limit_queries=None):
    """Flatten all uncached (query, doc) pairs into {key: (question, doc_text)}.

    Also RE-QUEUES entries a prior run marked as a HARD API failure
    (sequence_logprob == -100.0, the call_api fallback), so a transient 429/network
    burst is not frozen into a permanent 'unknown' answer.

    limit_queries: smoke-test mode — only the first N queries per dataset. The
    answers land in the same cache, so a smoke run costs nothing extra overall.
    """
    tasks = {}
    total_pairs = 0
    for ds in datasets:
        path = os.path.join(DATA_DIR, f"{ds}_{retriever}.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if limit_queries is not None:
            data = data[:limit_queries]
        for s in data:
            q = s["question"]
            for d in s["retrieved_docs"]:
                total_pairs += 1
                k = cache_key(model_name, q, d["text"])
                if k in tasks:
                    continue
                entry = cache.get(k)
                if entry is None or entry.get("sequence_logprob") == -100.0:
                    tasks[k] = (q, d["text"])
    return tasks, total_pairs


def load_cache(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache, path):
    """Atomic write (tmp + replace); called only from the main thread."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS),
                    help="LLM to answer with: llama|qwen|gpt")
    ap.add_argument("--retriever", default="e5", choices=["e5", "bge"],
                    help="retrieval set to read ({ds}_{retriever}.json) and cache into")
    ap.add_argument("--datasets", nargs="+", default=["nq", "triviaqa", "popqa"])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test: only the first N queries per dataset")
    a = ap.parse_args()

    cfg = MODELS[a.model]
    cache_path = os.path.join(DATA_DIR, f"llm_cache_{cfg['slug']}_{a.retriever}.json")
    cache = load_cache(cache_path)
    print(f"cache: {cache_path}  ({len(cache)} existing entries)")

    tasks, total_pairs = build_tasks(a.datasets, a.retriever, cfg["model"], cache, a.limit)
    print(f"pairs total={total_pairs:,} (dup incl.)  unique to-run={len(tasks):,}  workers={a.workers}")
    if not tasks:
        print("nothing to do (all cached).")
        return

    items = list(tasks.items())  # [(key, (q, doc)), ...]
    start = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(call_api, q, doc, cfg["backend"], cfg["model"]): k
                for k, (q, doc) in items}
        for fut in as_completed(futs):
            cache[futs[fut]] = fut.result()  # main-thread-only cache write => thread-safe
            done += 1
            if done % SAVE_EVERY == 0:
                save_cache(cache, cache_path)
                rate = done / (time.time() - start)
                eta_h = (len(items) - done) / rate / 3600 if rate > 0 else float("inf")
                print(f"  {done:,}/{len(items):,}  {rate:.1f} pairs/s  ETA {eta_h:.1f}h", flush=True)

    save_cache(cache, cache_path)
    elapsed = (time.time() - start) / 3600
    print(f"done: {done:,} new answers in {elapsed:.2f}h  -> {cache_path}")


if __name__ == "__main__":
    main()
