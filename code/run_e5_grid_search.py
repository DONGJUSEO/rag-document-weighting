#!/usr/bin/env python3
"""
(beta, lambda) grid for the E5 retriever — default-configuration robustness check
(paper Appendix S).

Reuses the E5 caches only (retrieval in data/{ds}_e5.json, CE evidence in
data/evidence_cache/{ds}_e5_cross_encoder.json, per-doc LLM answers in
data/llm_cache_{slug}_e5.json) — 0 new API calls. Raw cosine similarities are
used directly (no min-max), matching the E5 evaluation posted during the ARR
discussion and compute_gold_subset_analysis.py.

Grid mirrors the BM25 grid (run_bm25_grid_search.py / Appendix Q): 4 betas x
6 lambdas + Naive + SimW(=REPLUG, lam=0) per beta + EO-CE (lam->inf), but over
ALL 3 LLMs x 3 datasets (the E5 caches cover all nine cells).

Sanity gate: the (beta=0.5, lambda=30) point and the Naive / SimW / EO-CE
columns must reproduce results/gold_subset_analysis_e5.json ("full" bucket)
exactly, cell by cell. The script FAILS (exit 1) if any cell mismatches.

Usage (from the package root):
    python code/run_e5_grid_search.py

Output:
    results/e5_grid_search.json
"""
import json
import os
import sys
from datetime import datetime, timezone

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)

from compute_gold_subset_analysis import (  # noqa: E402
    build_datasets, build_llm_configs, cache_key, evidence_only_weights, K,
)
from weighting import naive_weights, replug_weights, dirichlet_weights  # noqa: E402
from generation import weighted_majority_vote  # noqa: E402
from metrics import exact_match  # noqa: E402

RETRIEVER = "e5"
BETAS = [0.5, 1.0, 2.0, 4.0]
LAMBDAS = [0.1, 0.5, 1.0, 3.0, 10.0, 30.0]
CANONICAL = os.path.join(BASE, "results", "gold_subset_analysis_e5.json")
OUT_PATH = os.path.join(BASE, "results", "e5_grid_search.json")
TOL = 1e-6  # EM percentages are exact re-aggregations; allow float dust only


def load_cell_queries(ds_key, llm_key, datasets, llm_configs, llm_cache):
    """Per-query (doc_answers, sims, evs, gold) for one cell — mirrors
    compute_gold_subset_analysis.run_cell's loading exactly (raw cosine, no
    min-max; missing cache -> 'unknown')."""
    dc = datasets[ds_key]
    with open(dc["test"], "r", encoding="utf-8") as f:
        test = json.load(f)
    with open(dc["ev"], "r", encoding="utf-8") as f:
        ev = json.load(f)
    assert len(test) == len(ev), f"{ds_key}: test/evidence length mismatch"
    model = llm_configs[llm_key]["model"]

    queries, n_skip, n_missing = [], 0, 0
    for qi, s in enumerate(test):
        docs = s.get("retrieved_docs", [])
        if len(docs) != K:
            n_skip += 1
            continue
        q, ans = s["question"], s["answers"]
        sims = [float(d["score"]) for d in docs]
        evs = [float(x) for x in ev[qi]]
        doc_answers = []
        for d in docs:
            entry = llm_cache.get(cache_key(model, q, d["text"]))
            if entry is None:
                n_missing += 1
                doc_answers.append("unknown")
            else:
                doc_answers.append(entry["answer_text"])
        queries.append({"doc_answers": doc_answers, "sims": sims,
                        "evs": evs, "answers": ans})
    return queries, n_skip, n_missing


def em_for_weights(queries, weight_fn):
    """EM (%) over the cell with per-query weights from weight_fn(q)."""
    n = len(queries)
    correct = 0
    for q in queries:
        pred = weighted_majority_vote(q["doc_answers"], weight_fn(q))[0]
        correct += exact_match(pred, q["answers"])
    return round(100.0 * correct / n, 4)


def main():
    datasets = build_datasets(RETRIEVER)
    llm_configs = build_llm_configs(RETRIEVER)
    with open(CANONICAL, "r", encoding="utf-8") as f:
        canonical = {(c["llm"], c["dataset"]): c["buckets"]["full"]
                     for c in json.load(f)}

    out = {
        "config": {
            "retriever": RETRIEVER, "K": K, "betas": BETAS, "lambdas": LAMBDAS,
            "note": ("0 new API calls; re-aggregates E5 LLM answer cache + E5 CE "
                     "evidence cache. Raw cosine (no min-max), identical to the "
                     "posted E5 evaluation. SimW row = REPLUG (lambda=0); EO-CE = "
                     "lambda->inf. Sanity gate: (0.5, 30) point + Naive/SimW/EO-CE "
                     "must equal gold_subset_analysis_e5.json full bucket."),
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "sanity_gate": {"all_pass": True, "rows": []},
        "cells": [],
    }

    for llm_key in ("qwen", "gpt", "llama"):
        cache_path = llm_configs[llm_key]["cache"]
        print(f"── loading LLM cache: {os.path.basename(cache_path)}", flush=True)
        with open(cache_path, "r", encoding="utf-8") as f:
            llm_cache = json.load(f)
        for ds_key in ("nq", "triviaqa", "popqa"):
            queries, n_skip, n_missing = load_cell_queries(
                ds_key, llm_key, datasets, llm_configs, llm_cache)
            n = len(queries)
            print(f"=== {llm_key}/{ds_key}: n={n} (skip={n_skip}, miss={n_missing})",
                  flush=True)

            cell = {"llm": llm_key, "dataset": ds_key, "n": n,
                    "n_skip_docs_ne_10": n_skip, "n_missing_cache": n_missing}
            cell["EM_naive"] = em_for_weights(queries, lambda q: naive_weights(K))
            cell["EM_eo_ce"] = em_for_weights(
                queries, lambda q: evidence_only_weights(q["evs"]))
            cell["EM_simw"] = {}
            for b in BETAS:
                cell["EM_simw"][str(b)] = em_for_weights(
                    queries, lambda q, b=b: replug_weights(q["sims"], beta=b))
            cell["EM_grid"] = {}
            for b in BETAS:
                for lam in LAMBDAS:
                    key = f"b{b}_l{lam}"
                    cell["EM_grid"][key] = em_for_weights(
                        queries,
                        lambda q, b=b, lam=lam: dirichlet_weights(
                            q["sims"], q["evs"], beta=b, lam=lam))
                    print(f"    beta={b:<4} lam={lam:<5} EM={cell['EM_grid'][key]:.4f}",
                          flush=True)

            can = canonical[(llm_key, ds_key)]
            checks = {
                "n": (n, can["n"]),
                "EM_naive": (cell["EM_naive"], can["EM_naive"]),
                "EM_simw_b0.5": (cell["EM_simw"]["0.5"], can["EM_simw"]),
                "EM_dir_b0.5_l30": (cell["EM_grid"]["b0.5_l30.0"], can["EM_dir_ce"]),
                "EM_eo_ce": (cell["EM_eo_ce"], can["EM_eo_ce"]),
            }
            ok = all(abs(a - b) <= TOL for a, b in checks.values())
            out["sanity_gate"]["rows"].append(
                {"cell": f"{llm_key}/{ds_key}", "pass": ok,
                 "checks": {k: {"recomputed": a, "canonical": b}
                            for k, (a, b) in checks.items()}})
            if not ok:
                out["sanity_gate"]["all_pass"] = False
                print(f"  !! SANITY GATE FAIL: {checks}", flush=True)
            out["cells"].append(cell)
        del llm_cache  # free ~40MB before the next model

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\nSanity gate all_pass = {out['sanity_gate']['all_pass']}", flush=True)
    print(f"Wrote {OUT_PATH}", flush=True)
    if not out["sanity_gate"]["all_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
