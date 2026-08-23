"""
SimW-vs-Naive and Dir-ES-vs-Naive McNemar tests (Appendix R).

Recomputes, from the released caches (no new LLM calls):
  1. SimW (beta=0.5, lambda=0) vs Naive  -- quantifies where the similarity-only
     prior *hurts* (paper title claim; significant harm on 2/9 cells, both NQ).
  2. Dir-ES (beta=0.5, lambda=30, embedding-stability evidence) vs Naive
     -- 8/9 Bonferroni-significant (exception TQA/GPT-4.1-mini, same as CE).

Population: strict k=10 (queries with exactly 10 retrieved documents), matching
compute_gold_subset_analysis.py. McNemar with Yates continuity correction,
Bonferroni alpha/27 (Sec. 3.4).

Sanity gate: the Dir-CE-vs-Naive discordant pairs recomputed here must match
results/gold_subset_analysis.json exactly; the script aborts on mismatch.

Usage:
    python3 compute_simw_es_mcnemar.py

Outputs:
    results/simw_vs_naive_mcnemar.json
    results/dir_es_vs_naive_mcnemar.json
"""
import json
import hashlib
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "code"))
from weighting import naive_weights, replug_weights, dirichlet_weights  # noqa: E402
from generation import weighted_majority_vote  # noqa: E402
from metrics import exact_match  # noqa: E402

BETA, LAM, K = 0.5, 30.0, 10
_MODELS = {"qwen": ("Qwen/Qwen2.5-7B-Instruct-Turbo", "qwen2_5_7b_instruct_turbo"),
           "gpt": ("gpt-4.1-mini", "gpt_4_1_mini"),
           "llama": ("meta-llama/Llama-3.3-70B-Instruct-Turbo", "llama_3_3_70b_instruct_turbo")}
_TESTS = {"nq": "nq_cosine.json", "triviaqa": "triviaqa_cosine.json",
          "popqa": "popqa_contriever.json"}


def cache_key(model_name, question, doc_text):
    """Reproduces generation.py:_cache_key exactly."""
    raw = f"{model_name}|||{question}|||{doc_text[:800]}"
    return hashlib.md5(raw.encode()).hexdigest()


def mcnemar_yates(n01, n10):
    from scipy.stats import chi2
    if n01 + n10 == 0:
        return 0.0, 1.0
    stat = max(0.0, abs(n01 - n10) - 1) ** 2 / (n01 + n10)  # Yates; clamp matters only when n01 == n10
    return stat, float(chi2.sf(stat, df=1))


def run(evidence_kind):
    """evidence_kind: 'simw' (CE evidence file only used for the gate) or 'es'."""
    gold_ref = {(e["llm"], e["dataset"]): e["buckets"]["full"]
                for e in json.load(open(f"{BASE}/results/gold_subset_analysis.json"))}
    label = ("SimW (beta=0.5, lambda=0) vs Naive" if evidence_kind == "simw"
             else "Dir-ES (beta=0.5, lambda=30, ES evidence) vs Naive")
    out = {"_meta": {"comparison": f"{label}; strict k=10; "
                                   "McNemar with Yates continuity correction",
                     "alpha_27": 0.05 / 27}}
    ev_file = "cross_encoder" if evidence_kind == "simw" else "embedding_stability"
    for ds, tf in _TESTS.items():
        test = json.load(open(f"{BASE}/data/{tf}"))
        ev = json.load(open(f"{BASE}/data/evidence_cache/{ds}_{ev_file}.json"))
        assert len(test) == len(ev), f"{ds}: test/evidence length mismatch"
        for mkey, (model, slug) in _MODELS.items():
            lc = json.load(open(f"{BASE}/data/llm_cache_{slug}.json"))
            cn, cx, cd_gate = [], [], []
            for qi, s in enumerate(test):
                docs = s.get("retrieved_docs", [])
                if len(docs) != K:
                    continue
                q, ans = s["question"], s["answers"]
                sims = [float(d["score"]) for d in docs]
                evs = [float(x) for x in ev[qi]]
                doc_answers = []
                for d in docs:
                    entry = lc.get(cache_key(model, q, d["text"]))
                    doc_answers.append("unknown" if entry is None else entry["answer_text"])
                cn.append(exact_match(
                    weighted_majority_vote(doc_answers, naive_weights(K))[0], ans))
                if evidence_kind == "simw":
                    w = replug_weights(sims, beta=BETA)
                    cd_gate.append(exact_match(weighted_majority_vote(
                        doc_answers, dirichlet_weights(sims, evs, beta=BETA, lam=LAM))[0], ans))
                else:
                    w = dirichlet_weights(sims, evs, beta=BETA, lam=LAM)
                cx.append(exact_match(weighted_majority_vote(doc_answers, w)[0], ans))
            n = len(cn)
            ref = gold_ref[(mkey, ds)]
            assert n == ref["n"], f"{mkey}/{ds}: n={n} != canonical {ref['n']}"
            if evidence_kind == "simw":
                n01_d = sum(1 for a, d in zip(cn, cd_gate) if a == 0 and d == 1)
                n10_d = sum(1 for a, d in zip(cn, cd_gate) if a == 1 and d == 0)
                assert (n01_d == ref["n01_naive_wrong_dir_right"]
                        and n10_d == ref["n10_naive_right_dir_wrong"]), \
                    f"{mkey}/{ds}: Dir-CE gate mismatch vs gold_subset_analysis.json"
            n01 = sum(1 for a, x in zip(cn, cx) if a == 0 and x == 1)
            n10 = sum(1 for a, x in zip(cn, cx) if a == 1 and x == 0)
            stat, p = mcnemar_yates(n01, n10)
            dem = 100.0 * (sum(cx) - sum(cn)) / n
            out[f"{mkey}_{ds}"] = {
                "n": n, "EM_naive": round(100.0 * sum(cn) / n, 4),
                "EM_variant": round(100.0 * sum(cx) / n, 4),
                "dEM_vs_naive": round(dem, 4),
                "n01_naive_wrong_variant_right": n01,
                "n10_naive_right_variant_wrong": n10,
                "chi2_yates": round(stat, 4), "p_value": p,
                "sig_alpha27": p < 0.05 / 27}
            print(f"  {mkey:5s}/{ds:8s} dEM={dem:+.3f} n01={n01} n10={n10} "
                  f"p={p:.4g} sig@a/27={p < 0.05 / 27}")
    fname = ("simw_vs_naive_mcnemar.json" if evidence_kind == "simw"
             else "dir_es_vs_naive_mcnemar.json")
    with open(f"{BASE}/results/{fname}", "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> results/{fname}")


if __name__ == "__main__":
    print("=== SimW vs Naive ===")
    run("simw")
    print("=== Dir-ES vs Naive ===")
    run("es")
