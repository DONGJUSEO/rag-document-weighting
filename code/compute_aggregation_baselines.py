"""
Aggregation-baseline comparison (rebuttal, Reviewer QJ7g Weakness 1).

Reviewer QJ7g W1 asks us to compare Dir-CE against other voting / answer-
aggregation RAG methods. This script adds THREE aggregation rules on top of the
paper's soft weighted vote, computed with ZERO new API calls (it re-uses the
exact same per-document LLM answer cache and cross-encoder evidence cache as the
canonical gold-subset diagnostic):

  1. CE-Top1 Selection : take the answer of the single document with the highest
     cross-encoder (CE) evidence score. No vote. A no-extra-LLM-call proxy for
     selection-based RAG (SuRe-style "pick the best passage, then answer").
  2. Borda-CE          : rank documents by CE score (descending), give Borda
     weight w_i = (K - r_i + 1) (highest CE -> K=10, lowest -> 1), and run the
     paper's existing weighted_majority_vote with those rank weights. The
     standard rank-based voting rule.
  4. CE-Rerank (top-5) : hard-select the 5 highest-CE documents with uniform
     weights (the CE-Rerank baseline of Sec. 4; same rule as
     run_additional_baselines.ce_rerank_uniform_weights), recomputed here on
     the same strict-k cached answers so that it is comparable cell-by-cell
     with the other rules (added for the camera-ready, 2026-08-22).
  3. RRF               : Reciprocal Rank Fusion (Cormack et al. 2009), a published
     training-free rank-fusion method. Fuse the retrieval-similarity rank and the
     CE rank: w_i = 1/(60 + rank_sim_i) + 1/(60 + rank_CE_i), then run the paper's
     weighted_majority_vote. The rank-based counterpart to Dir-CE's sim+evidence
     fusion, hence the most directly comparable published aggregation rule.

Reference columns Naive (uniform) and Dir-CE (beta=0.5, lambda=30) are recomputed
here and MUST reproduce the canonical results/gold_subset_analysis.json "full"
bucket to within +/-0.01 EM (a path-integrity sanity gate: same test set, same
CE evidence, same LLM cache, same cache_key, same missing-cache handling).

This is a strict FORK of compute_gold_subset_analysis.py: it imports that script's
dataset/cache/key builders and constants so the loading path is byte-identical.
Only the aggregation rules differ. No cache file is written or modified.

Determinism: fully deterministic. argmax / rank ties are broken by document order
(lowest index first), matching the weighted-vote tie-break already used in the
paper. Re-running yields byte-identical numbers.

Run:
  python compute_aggregation_baselines.py           # DPR, all 9 cells, save JSON
  python compute_aggregation_baselines.py qwen nq   # single cell, print only
"""
import argparse
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "code"))

# Reuse the CANONICAL loaders/constants so the reference columns are guaranteed
# to be computed on an identical path (single source of truth).
from compute_gold_subset_analysis import (  # noqa: E402
    build_datasets, build_llm_configs, cache_key, BETA, LAMBDA_DIR_CE, K,
)
from weighting import naive_weights, dirichlet_weights  # noqa: E402
from generation import weighted_majority_vote  # noqa: E402
from metrics import exact_match, mcnemar_test  # noqa: E402

CANONICAL = os.path.join(BASE, "results", "gold_subset_analysis.json")

# Reciprocal Rank Fusion standard constant (Cormack et al. 2009).
RRF_C = 60


def _is_answer(a):
    """A document 'answered' iff its cached text is non-empty and not 'unknown'
    (identical predicate to weighted_majority_vote's abstention filter)."""
    a = a.strip().lower()
    return a != "" and a != "unknown"


def run_cell(llm_key, ds_key, llm_cache, datasets, llm_configs, verbose=True):
    """Compute {Naive, Dir-CE, CE-Top1, Borda-CE} EM for one (LLM, dataset) cell.

    'full' bucket only (every query with exactly K=10 retrieved docs) — matches
    the canonical 'full' bucket used for the sanity gate.
    """
    lc, dc = llm_configs[llm_key], datasets[ds_key]
    with open(dc["test"], "r", encoding="utf-8") as f:
        test = json.load(f)
    with open(dc["ev"], "r", encoding="utf-8") as f:
        ev = json.load(f)
    assert len(test) == len(ev), \
        f"{ds_key}: test/evidence length mismatch ({len(test)} vs {len(ev)})"
    model = lc["model"]

    # Per-query 0/1 correctness vectors.
    vec = {"naive": [], "dir_ce": [], "ce_top1": [], "borda_ce": [], "rrf": [],
           "ce_rerank5": []}
    vec_ce_top1_ans = []          # auxiliary: CE-Top1 restricted to answering docs
    n_skip = n_missing = 0
    n_top1_abstain = 0            # times the single top-CE doc abstained (unknown/missing)

    for qi, s in enumerate(test):
        docs = s.get("retrieved_docs", [])
        if len(docs) != K:
            n_skip += 1
            continue
        q, ans = s["question"], s["answers"]
        sims = [float(d["score"]) for d in docs]
        evs = [float(x) for x in ev[qi]]

        # Per-document cached answers (NO new API calls; missing -> 'unknown',
        # exactly as compute_gold_subset_analysis.py handles abstention).
        doc_answers = []
        for d in docs:
            entry = llm_cache.get(cache_key(model, q, d["text"]))
            if entry is None:
                n_missing += 1
                doc_answers.append("unknown")
            else:
                doc_answers.append(entry["answer_text"])

        # --- reference columns (sanity gate) ---
        w_naive = naive_weights(K)
        w_dir = dirichlet_weights(sims, evs, beta=BETA, lam=LAMBDA_DIR_CE)
        c_naive = exact_match(weighted_majority_vote(doc_answers, w_naive)[0], ans)
        c_dir = exact_match(weighted_majority_vote(doc_answers, w_dir)[0], ans)

        # --- CE-Top1 selection: single highest-CE doc, no vote ---
        # tie-break: highest CE, then lowest index (document order).
        top_idx = max(range(K), key=lambda i: (evs[i], -i))
        top_ans = doc_answers[top_idx]
        c_ce_top1 = exact_match(top_ans, ans)
        if not _is_answer(top_ans):
            n_top1_abstain += 1

        # --- Borda-CE: rank by CE desc -> weight (K - rank + 1); then vote ---
        order_ce = sorted(range(K), key=lambda i: (-evs[i], i))  # CE desc, tie by index
        rank_ce = [0] * K
        for pos, i in enumerate(order_ce):
            rank_ce[i] = pos + 1                                 # r_i in 1..K
        w_borda = [float(K - rank_ce[i] + 1) for i in range(K)]  # 10..1
        c_borda = exact_match(weighted_majority_vote(doc_answers, w_borda)[0], ans)

        # --- RRF (Reciprocal Rank Fusion, Cormack et al. 2009): fuse sim rank
        #     and CE rank -> w_i = 1/(60+rank_sim_i) + 1/(60+rank_ce_i); then vote ---
        order_sim = sorted(range(K), key=lambda i: (-sims[i], i))  # sim desc, tie by index
        rank_sim = [0] * K
        for pos, i in enumerate(order_sim):
            rank_sim[i] = pos + 1                                # 1..K
        w_rrf = [1.0 / (RRF_C + rank_sim[i]) + 1.0 / (RRF_C + rank_ce[i]) for i in range(K)]
        c_rrf = exact_match(weighted_majority_vote(doc_answers, w_rrf)[0], ans)

        # --- CE-Rerank (top-5 uniform): same rule as run_additional_baselines ---
        top5 = sorted(range(K), key=lambda i: evs[i], reverse=True)[:5]
        w_rr5 = [0.0] * K
        for i in top5:
            w_rr5[i] = 1.0 / 5
        c_rr5 = exact_match(weighted_majority_vote(doc_answers, w_rr5)[0], ans)

        # --- auxiliary: CE-Top1 among *answering* docs (skip abstentions) ---
        ans_idx = [i for i in range(K) if _is_answer(doc_answers[i])]
        if ans_idx:
            best = max(ans_idx, key=lambda i: (evs[i], -i))
            c_ce_top1_ans = exact_match(doc_answers[best], ans)
        else:
            c_ce_top1_ans = 0

        vec["naive"].append(c_naive)
        vec["dir_ce"].append(c_dir)
        vec["ce_top1"].append(c_ce_top1)
        vec["borda_ce"].append(c_borda)
        vec["rrf"].append(c_rrf)
        vec["ce_rerank5"].append(c_rr5)
        vec_ce_top1_ans.append(c_ce_top1_ans)

    n = len(vec["naive"])
    em = {m: 100.0 * sum(v) / n for m, v in vec.items()}
    em_ce_top1_ans = 100.0 * sum(vec_ce_top1_ans) / n

    # McNemar: Dir-CE vs each new aggregation rule (b = Dir-CE right only).
    chi2_t1, p_t1 = mcnemar_test(vec["dir_ce"], vec["ce_top1"])
    chi2_bd, p_bd = mcnemar_test(vec["dir_ce"], vec["borda_ce"])
    chi2_rrf, p_rrf = mcnemar_test(vec["dir_ce"], vec["rrf"])
    chi2_rr5, p_rr5 = mcnemar_test(vec["dir_ce"], vec["ce_rerank5"])
    n_dir_only_rr5 = sum(1 for d, o in zip(vec["dir_ce"], vec["ce_rerank5"]) if d == 1 and o == 0)
    n_rr5_only = sum(1 for d, o in zip(vec["dir_ce"], vec["ce_rerank5"]) if d == 0 and o == 1)
    n_dir_only_t1 = sum(1 for d, o in zip(vec["dir_ce"], vec["ce_top1"]) if d == 1 and o == 0)
    n_t1_only = sum(1 for d, o in zip(vec["dir_ce"], vec["ce_top1"]) if d == 0 and o == 1)
    n_dir_only_bd = sum(1 for d, o in zip(vec["dir_ce"], vec["borda_ce"]) if d == 1 and o == 0)
    n_bd_only = sum(1 for d, o in zip(vec["dir_ce"], vec["borda_ce"]) if d == 0 and o == 1)
    n_dir_only_rrf = sum(1 for d, o in zip(vec["dir_ce"], vec["rrf"]) if d == 1 and o == 0)
    n_rrf_only = sum(1 for d, o in zip(vec["dir_ce"], vec["rrf"]) if d == 0 and o == 1)

    out = {
        "llm": llm_key, "dataset": ds_key, "n": n,
        "n_skip_docs_ne_10": n_skip, "n_missing_cache": n_missing,
        "n_ce_top1_abstain": n_top1_abstain,
        "beta": BETA, "lambda_dir_ce": LAMBDA_DIR_CE, "K": K,
        "EM_naive": round(em["naive"], 4),
        "EM_dir_ce": round(em["dir_ce"], 4),
        "EM_ce_top1": round(em["ce_top1"], 4),
        "EM_borda_ce": round(em["borda_ce"], 4),
        "EM_rrf": round(em["rrf"], 4),
        "EM_ce_rerank5": round(em["ce_rerank5"], 4),
        "dEM_dir_vs_ce_rerank5": round(em["dir_ce"] - em["ce_rerank5"], 4),
        "mcnemar_dir_vs_ce_rerank5": {
            "chi2": round(chi2_rr5, 4), "p": p_rr5,
            "n_dir_right_only": n_dir_only_rr5, "n_ce_rerank5_right_only": n_rr5_only},
        "EM_ce_top1_answering_aux": round(em_ce_top1_ans, 4),
        "dEM_dir_vs_ce_top1": round(em["dir_ce"] - em["ce_top1"], 4),
        "dEM_dir_vs_borda_ce": round(em["dir_ce"] - em["borda_ce"], 4),
        "dEM_dir_vs_rrf": round(em["dir_ce"] - em["rrf"], 4),
        "mcnemar_dir_vs_ce_top1": {
            "chi2": round(chi2_t1, 4), "p": p_t1,
            "n_dir_right_only": n_dir_only_t1, "n_ce_top1_right_only": n_t1_only},
        "mcnemar_dir_vs_borda_ce": {
            "chi2": round(chi2_bd, 4), "p": p_bd,
            "n_dir_right_only": n_dir_only_bd, "n_borda_right_only": n_bd_only},
        "mcnemar_dir_vs_rrf": {
            "chi2": round(chi2_rrf, 4), "p": p_rrf,
            "n_dir_right_only": n_dir_only_rrf, "n_rrf_right_only": n_rrf_only},
    }

    if verbose:
        print(f"=== {llm_key}/{ds_key}  (n={n}, skip={n_skip}, missing={n_missing}, "
              f"top1_abstain={n_top1_abstain}) ===")
        print(f"  Naive={out['EM_naive']:6.2f}  Dir-CE={out['EM_dir_ce']:6.2f}  "
              f"CE-Top1={out['EM_ce_top1']:6.2f}  Borda-CE={out['EM_borda_ce']:6.2f}  "
              f"RRF={out['EM_rrf']:6.2f}  CE-Rerank5={out['EM_ce_rerank5']:6.2f}  "
              f"(CE-Top1[ans]={out['EM_ce_top1_answering_aux']:6.2f})")
        print(f"  Dir-CE vs CE-Top1:  dEM={out['dEM_dir_vs_ce_top1']:+.2f}  "
              f"p={p_t1:.2e}  (dir-only={n_dir_only_t1}, ce_top1-only={n_t1_only})")
        print(f"  Dir-CE vs Borda-CE: dEM={out['dEM_dir_vs_borda_ce']:+.2f}  "
              f"p={p_bd:.2e}  (dir-only={n_dir_only_bd}, borda-only={n_bd_only})")
        print(f"  Dir-CE vs RRF:      dEM={out['dEM_dir_vs_rrf']:+.2f}  "
              f"p={p_rrf:.2e}  (dir-only={n_dir_only_rrf}, rrf-only={n_rrf_only})")
        print()
    return out


def sanity_gate(results):
    """Recomputed Naive/Dir-CE 'full' EM must match canonical within +/-0.01."""
    if not os.path.exists(CANONICAL):
        print(f"  [sanity] canonical not found ({CANONICAL}) — gate SKIPPED")
        return None
    with open(CANONICAL, "r", encoding="utf-8") as f:
        canon = {(r["llm"], r["dataset"]): r["buckets"]["full"] for r in json.load(f)}

    rows, all_pass = [], True
    print("=" * 78)
    print("SANITY GATE  (recomputed 'full' vs canonical gold_subset_analysis.json)")
    print(f"{'cell':<16}{'n(re/can)':<16}{'Naive re/can':<22}{'Dir-CE re/can':<22}{'ok'}")
    for r in results:
        key = (r["llm"], r["dataset"])
        c = canon.get(key)
        if c is None:
            print(f"  {r['llm']}/{r['dataset']}: no canonical row"); all_pass = False; continue
        dn = abs(r["EM_naive"] - c["EM_naive"])
        dd = abs(r["EM_dir_ce"] - c["EM_dir_ce"])
        n_ok = (r["n"] == c["n"])
        ok = (dn <= 0.01) and (dd <= 0.01) and n_ok
        all_pass = all_pass and ok
        cell = f"{r['llm']}/{r['dataset']}"
        n_str = f"{r['n']}/{c['n']}"
        naive_str = f"{r['EM_naive']:.4f}/{c['EM_naive']:.4f}"
        dir_str = f"{r['EM_dir_ce']:.4f}/{c['EM_dir_ce']:.4f}"
        print(f"{cell:<16}{n_str:<16}{naive_str:<22}{dir_str:<22}{'PASS' if ok else 'FAIL'}")
        rows.append({"cell": cell, "n_re": r["n"], "n_can": c["n"],
                     "EM_naive_re": r["EM_naive"], "EM_naive_can": c["EM_naive"],
                     "EM_dir_ce_re": r["EM_dir_ce"], "EM_dir_ce_can": c["EM_dir_ce"],
                     "d_naive": round(dn, 6), "d_dir_ce": round(dd, 6),
                     "n_match": n_ok, "pass": ok})
    print(f"\n  SANITY GATE: {'ALL PASS' if all_pass else 'FAILED — DO NOT TRUST RESULTS'}")
    print("=" * 78)
    return {"all_pass": all_pass, "rows": rows}


def main():
    ap = argparse.ArgumentParser(description="Aggregation-baseline comparison (rebuttal).")
    ap.add_argument("--retriever", default="dpr", choices=["dpr", "e5", "bge"])
    ap.add_argument("llm", nargs="?", default="all", help="qwen|gpt|llama|all")
    ap.add_argument("dataset", nargs="?", default="all", help="nq|triviaqa|popqa|all")
    a = ap.parse_args()

    datasets = build_datasets(a.retriever)
    llm_configs = build_llm_configs(a.retriever)
    llms = list(llm_configs) if a.llm == "all" else [a.llm]
    dss = list(datasets) if a.dataset == "all" else [a.dataset]

    results = []
    for lk in llms:
        cache_path = llm_configs[lk]["cache"]
        if not os.path.exists(cache_path):
            print(f"  [skip] {lk}: cache not found ({os.path.basename(cache_path)})")
            continue
        with open(cache_path, "r", encoding="utf-8") as f:
            llm_cache = json.load(f)
        for dk in dss:
            results.append(run_cell(lk, dk, llm_cache, datasets, llm_configs))

    if not results:
        print("no cells computed (missing caches).")
        return

    gate = sanity_gate(results)

    # Cross-cell means (only meaningful for a full 9-cell run).
    means = {}
    if len(results) > 1:
        print("\nCROSS-CELL MEAN EM")
        for m in ("EM_naive", "EM_dir_ce", "EM_ce_top1", "EM_borda_ce", "EM_rrf",
                  "EM_ce_rerank5", "EM_ce_top1_answering_aux"):
            means[m] = round(sum(r[m] for r in results) / len(results), 4)
            print(f"  {m:<28} = {means[m]:6.3f}")
        means["dEM_dir_vs_ce_top1_mean"] = round(
            sum(r["dEM_dir_vs_ce_top1"] for r in results) / len(results), 4)
        means["dEM_dir_vs_borda_ce_mean"] = round(
            sum(r["dEM_dir_vs_borda_ce"] for r in results) / len(results), 4)
        means["dEM_dir_vs_rrf_mean"] = round(
            sum(r["dEM_dir_vs_rrf"] for r in results) / len(results), 4)
        means["dEM_dir_vs_ce_rerank5_mean"] = round(
            sum(r["dEM_dir_vs_ce_rerank5"] for r in results) / len(results), 4)
        print(f"  MEAN dEM(Dir-CE - CE-Top1)  = {means['dEM_dir_vs_ce_top1_mean']:+.3f}")
        print(f"  MEAN dEM(Dir-CE - Borda-CE) = {means['dEM_dir_vs_borda_ce_mean']:+.3f}")
        print(f"  MEAN dEM(Dir-CE - RRF)      = {means['dEM_dir_vs_rrf_mean']:+.3f}")
        w_t1 = sum(1 for r in results if r["dEM_dir_vs_ce_top1"] > 0)
        w_bd = sum(1 for r in results if r["dEM_dir_vs_borda_ce"] > 0)
        w_rrf = sum(1 for r in results if r["dEM_dir_vs_rrf"] > 0)
        print(f"  Dir-CE beats CE-Top1 in {w_t1}/{len(results)} cells; "
              f"Borda-CE in {w_bd}/{len(results)}; RRF in {w_rrf}/{len(results)}")

    # Save ONLY on a full 9-cell run and ONLY if the sanity gate passed.
    full_run = (len(llms) == len(llm_configs) and len(dss) == len(datasets)
                and len(results) == len(llm_configs) * len(datasets))
    if full_run:
        if gate is None or not gate["all_pass"]:
            print("\n  Sanity gate did not pass — results NOT saved.")
            return
        suffix = "" if a.retriever == "dpr" else f"_{a.retriever}"
        out_path = os.path.join(BASE, "results", f"aggregation_baselines{suffix}.json")
        payload = {
            "config": {"beta": BETA, "lambda_dir_ce": LAMBDA_DIR_CE, "K": K, "rrf_c": RRF_C,
                       "retriever": a.retriever,
                       "note": "0 new API calls; reuses per-doc LLM cache + CE evidence cache. "
                               "CE-Top1 = single highest-CE doc's answer (no vote). "
                               "Borda-CE = rank(CE) weight (K-r+1) into weighted_majority_vote. "
                               "RRF (Cormack 2009) = 1/(60+rank_sim)+1/(60+rank_CE) into "
                               "weighted_majority_vote."},
            "sanity_gate": gate,
            "cells": results,
            "means": means,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved: {out_path}")
    else:
        print("\n  subset run — JSON NOT written (run with no positional args for full artifact)")


if __name__ == "__main__":
    main()
