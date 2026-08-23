"""
Gold-subset diagnostic (CANONICAL — single source of truth for the has-gold tables).

Recomputes Naive / SimW(REPLUG) / Dir-CE / EO-CE weighted-voting EM restricted to
three query buckets:
  - full          : all queries with exactly K=10 retrieved docs
  - gold_subset   : queries whose top-10 contains >=1 doc with type=='gold'
  - nogold_subset : queries whose top-10 contains NO gold doc

Purpose (paper §5.2 / Appendix R): separate retrieval failure (no gold
in top-10) from document-weighting failure (gold present but not surfaced by the
weights). Uses ONLY cached per-document LLM answers — NO new API calls.

Determinism: fully deterministic. No RNG anywhere (cache lookup + weighted vote
whose tie-break follows document order via dict-insertion max). Re-running yields
byte-identical numbers.

gold definition: dataset-native `type=='gold'` (the first answer-bearing document
as tagged in the retrieval file, per convert_dpr_cosine.py / build_popqa_contriever.py).
Cross-checked against a string-match definition (answer span appears in doc text);
the only cell whose Bonferroni verdict (alpha/27 ~= 1.85e-3) flips between the two
definitions is TQA/GPT (p ~= 1.5e-3 native vs ~= 2e-3 string-match) — report it as
"marginal", never as clearly significant.

McNemar: metrics.mcnemar_test (chi-squared with Yates continuity correction), the
exact test used for the paper's significance tables. b = Naive-right-only,
c = Dir-CE-right-only; chi2 = max(0, |b - c| - 1)^2 / (b + c).

Run:
  python compute_gold_subset_analysis.py                 # DPR, all 9 cells, saves canonical JSON
  python compute_gold_subset_analysis.py --retriever e5  # E5, all 9 cells (needs script 2-4 outputs)
  python compute_gold_subset_analysis.py qwen nq         # single cell, prints only (NO save -> canonical safe)
"""
import argparse
import json
import hashlib
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "code"))
from weighting import naive_weights, replug_weights, dirichlet_weights  # noqa: E402
from generation import weighted_majority_vote  # noqa: E402
from metrics import exact_match, mcnemar_test  # noqa: E402

# Paper default configuration (Appendix K).
BETA = 0.5
LAMBDA_DIR_CE = 30.0
K = 10

# Retriever-bound paths. Passing --retriever atomically switches the retrieval set,
# its MATCHING cross-encoder evidence, AND the LLM answer cache together, so a
# test/evidence mismatch (DPR evidence silently applied to E5 docs -> corrupted
# Dir-CE only) is structurally impossible.
_MODELS = {"qwen": "Qwen/Qwen2.5-7B-Instruct-Turbo",
           "gpt": "gpt-4.1-mini",
           "llama": "meta-llama/Llama-3.3-70B-Instruct-Turbo"}
_SLUGS = {"qwen": "qwen2_5_7b_instruct_turbo",
          "gpt": "gpt_4_1_mini",
          "llama": "llama_3_3_70b_instruct_turbo"}
_DPR_TEST = {"nq": "nq_cosine.json", "triviaqa": "triviaqa_cosine.json",
             "popqa": "popqa_contriever.json"}


def build_datasets(retriever):
    """test retrieval file + its MATCHING CE-evidence file, bound by retriever."""
    if retriever == "dpr":
        return {ds: {"test": f"{BASE}/data/{_DPR_TEST[ds]}",
                     "ev": f"{BASE}/data/evidence_cache/{ds}_cross_encoder.json"}
                for ds in _DPR_TEST}
    return {ds: {"test": f"{BASE}/data/{ds}_{retriever}.json",
                 "ev": f"{BASE}/data/evidence_cache/{ds}_{retriever}_cross_encoder.json"}
            for ds in _DPR_TEST}


def build_llm_configs(retriever):
    """LLM answer cache (retriever-suffixed, so E5 answers never mix into DPR)."""
    suffix = "" if retriever == "dpr" else f"_{retriever}"
    return {k: {"cache": f"{BASE}/data/llm_cache_{_SLUGS[k]}{suffix}.json",
                "model": _MODELS[k]} for k in _MODELS}


BUCKETS = ("full", "gold_subset", "nogold_subset")


def evidence_only_weights(evidence_scores):
    """Evidence-only weighting: w_i = e_i / sum_j e_j (lambda -> infinity limit of
    dirichlet_weights). Copied verbatim from compute_c1_answer_support_mass.py so the
    EO-CE column here matches the paper pipeline's definition exactly."""
    total = sum(evidence_scores)
    if total <= 0:
        k = len(evidence_scores)
        return [1.0 / k] * k
    return [e / total for e in evidence_scores]


def cache_key(model_name, question, doc_text):
    """Reproduces generation.py:_cache_key exactly."""
    raw = f"{model_name}|||{question}|||{doc_text[:800]}"
    return hashlib.md5(raw.encode()).hexdigest()


def run_cell(llm_key, ds_key, llm_cache, datasets, llm_configs, verbose=True):
    """Compute EM buckets for one (LLM, dataset) cell using a pre-loaded LLM cache."""
    lc, dc = llm_configs[llm_key], datasets[ds_key]
    with open(dc["test"], "r", encoding="utf-8") as f:
        test = json.load(f)
    with open(dc["ev"], "r", encoding="utf-8") as f:
        ev = json.load(f)
    assert len(test) == len(ev), \
        f"{ds_key}: test/evidence length mismatch ({len(test)} vs {len(ev)})"
    model = lc["model"]

    # Per-query 0/1 correctness vectors, split by bucket.
    vec = {b: {"naive": [], "simw": [], "dir_ce": [], "eo_ce": []} for b in BUCKETS}
    n_skip = n_missing = n_unknown = 0

    for qi, s in enumerate(test):
        docs = s.get("retrieved_docs", [])
        if len(docs) != K:
            n_skip += 1
            continue
        q, ans = s["question"], s["answers"]
        sims = [float(d["score"]) for d in docs]
        evs = [float(x) for x in ev[qi]]
        has_gold = any(d.get("type") == "gold" for d in docs)

        # Look up per-document answers from the cache (no new API calls).
        doc_answers = []
        for d in docs:
            entry = llm_cache.get(cache_key(model, q, d["text"]))
            if entry is None:
                n_missing += 1
                doc_answers.append("unknown")  # matches paper's abstention handling
            else:
                ans_text = entry["answer_text"]
                doc_answers.append(ans_text)
                if ans_text.strip().lower() == "unknown":
                    n_unknown += 1

        w_naive = naive_weights(K)
        w_simw = replug_weights(sims, beta=BETA)                       # lambda = 0
        w_dir = dirichlet_weights(sims, evs, beta=BETA, lam=LAMBDA_DIR_CE)
        w_eo = evidence_only_weights(evs)                              # lambda -> inf

        c_naive = exact_match(weighted_majority_vote(doc_answers, w_naive)[0], ans)
        c_simw = exact_match(weighted_majority_vote(doc_answers, w_simw)[0], ans)
        c_dir = exact_match(weighted_majority_vote(doc_answers, w_dir)[0], ans)
        c_eo = exact_match(weighted_majority_vote(doc_answers, w_eo)[0], ans)

        for b in ("full", "gold_subset" if has_gold else "nogold_subset"):
            vec[b]["naive"].append(c_naive)
            vec[b]["simw"].append(c_simw)
            vec[b]["dir_ce"].append(c_dir)
            vec[b]["eo_ce"].append(c_eo)

    out = {"llm": llm_key, "dataset": ds_key,
           "n_skip_docs_ne_10": n_skip, "n_missing_cache": n_missing,
           "n_unknown_answer": n_unknown,
           "beta": BETA, "lambda_dir_ce": LAMBDA_DIR_CE, "buckets": {}}
    for b in BUCKETS:
        n = len(vec[b]["naive"])
        if n == 0:
            out["buckets"][b] = {"n": 0}
            continue
        em_naive = 100.0 * sum(vec[b]["naive"]) / n
        em_simw = 100.0 * sum(vec[b]["simw"]) / n
        em_dir = 100.0 * sum(vec[b]["dir_ce"]) / n
        em_eo = 100.0 * sum(vec[b]["eo_ce"]) / n
        chi2, p = mcnemar_test(vec[b]["naive"], vec[b]["dir_ce"])
        chi2_eo, p_eo = mcnemar_test(vec[b]["naive"], vec[b]["eo_ce"])
        n01 = sum(1 for a, d in zip(vec[b]["naive"], vec[b]["dir_ce"]) if a == 0 and d == 1)
        n10 = sum(1 for a, d in zip(vec[b]["naive"], vec[b]["dir_ce"]) if a == 1 and d == 0)
        out["buckets"][b] = {
            "n": n,
            "EM_naive": round(em_naive, 4),
            "EM_simw": round(em_simw, 4),
            "EM_dir_ce": round(em_dir, 4),
            "EM_eo_ce": round(em_eo, 4),
            "dEM_dir_vs_naive": round(em_dir - em_naive, 4),
            "dEM_simw_vs_naive": round(em_simw - em_naive, 4),
            "dEM_eo_vs_naive": round(em_eo - em_naive, 4),
            "mcnemar_chi2": round(chi2, 4),
            "mcnemar_p": p,
            "mcnemar_chi2_eo": round(chi2_eo, 4),
            "mcnemar_p_eo": p_eo,
            "n01_naive_wrong_dir_right": n01,
            "n10_naive_right_dir_wrong": n10,
        }

    if verbose:
        print(f"=== {llm_key}/{ds_key}  (skip_docs!=10={n_skip}, missing_cache={n_missing}) ===")
        for b in BUCKETS:
            d = out["buckets"][b]
            if d.get("n", 0) == 0:
                print(f"  {b:14s} n=0")
                continue
            print(f"  {b:14s} n={d['n']:6d}  Naive={d['EM_naive']:6.2f}  "
                  f"SimW={d['EM_simw']:6.2f}  Dir-CE={d['EM_dir_ce']:6.2f}  "
                  f"EO-CE={d['EM_eo_ce']:6.2f}  "
                  f"dEM(Dir-Naive)={d['dEM_dir_vs_naive']:+.2f}  "
                  f"dEM(EO-Naive)={d['dEM_eo_vs_naive']:+.2f}  "
                  f"n01={d['n01_naive_wrong_dir_right']:4d} "
                  f"n10={d['n10_naive_right_dir_wrong']:4d}  p={d['mcnemar_p']:.2e}  "
                  f"p_eo={d['mcnemar_p_eo']:.2e}")
        print()
    return out


def main():
    ap = argparse.ArgumentParser(description="Gold-subset diagnostic (canonical).")
    ap.add_argument("--retriever", default="dpr", choices=["dpr", "e5", "bge"],
                    help="binds test set + CE evidence + LLM cache together")
    ap.add_argument("llm", nargs="?", default="all", help="qwen|gpt|llama|all")
    ap.add_argument("dataset", nargs="?", default="all", help="nq|triviaqa|popqa|all")
    a = ap.parse_args()

    datasets = build_datasets(a.retriever)
    llm_configs = build_llm_configs(a.retriever)
    llms = list(llm_configs) if a.llm == "all" else [a.llm]
    dss = list(datasets) if a.dataset == "all" else [a.dataset]

    results = []
    # Load each LLM cache exactly once (each is 30-45 MB); iterate datasets inside.
    for lk in llms:
        cache_path = llm_configs[lk]["cache"]
        if not os.path.exists(cache_path):
            hint = (f"run run_e5_llm_pairs.py --retriever {a.retriever} --model {lk}"
                    if a.retriever != "dpr" else
                    "build it with the main pipeline (run_main.py; see README §1)")
            print(f"  [skip] {lk}: cache not found ({os.path.basename(cache_path)}) — {hint}")
            continue
        with open(cache_path, "r", encoding="utf-8") as f:
            llm_cache = json.load(f)
        for dk in dss:
            results.append(run_cell(lk, dk, llm_cache, datasets, llm_configs))

    if not results:
        print("no cells computed (missing caches).")
        return

    # Cross-cell averages (only meaningful for the full 9-cell run).
    if len(results) > 1:
        print("=" * 60)
        for b in BUCKETS:
            vals = [r["buckets"][b]["dEM_dir_vs_naive"]
                    for r in results if r["buckets"][b].get("n", 0) > 0]
            if vals:
                print(f"  MEAN dEM(Dir-CE - Naive) [{b:14s}] = "
                      f"{sum(vals) / len(vals):+.3f}   (over {len(vals)} cells)")

    total_missing = sum(r["n_missing_cache"] for r in results)
    total_unknown = sum(r.get("n_unknown_answer", 0) for r in results)
    if total_missing:
        print(f"  WARNING: {total_missing:,} cache misses — LLM answers incomplete for "
              f"retriever='{a.retriever}'. EM/Dir-CE understated until run_e5_llm_pairs.py finishes.")
    print(f"  (answers marked 'unknown': {total_unknown:,} — compare to the DPR run's rate before trusting EM)")

    # Save ONLY on a full 9-cell run so a subset re-check cannot clobber the
    # canonical artifact. Retriever-specific filename.
    full_run = (len(llms) == len(llm_configs) and len(dss) == len(datasets)
                and len(results) == len(llm_configs) * len(datasets))
    if full_run:
        suffix = "" if a.retriever == "dpr" else f"_{a.retriever}"
        out_path = os.path.join(BASE, "results", f"gold_subset_analysis{suffix}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {out_path}")
    else:
        print("\n  subset run — canonical JSON NOT overwritten "
              "(run with no positional args for the full 9-cell artifact)")


if __name__ == "__main__":
    main()
