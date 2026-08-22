"""
Main experiment runner for the unified Dirichlet-style RAG framework.

Supports a (dataset x evidence x beta x lambda) grid search plus multiple
aggregation modes.

Usage examples:
  python3 run_main.py --dataset nq --evidence cross_encoder
  python3 run_main.py --dataset nq --evidence all --max_questions 100
  python3 run_main.py --dataset all --evidence all --aggregation sequence
"""
import json
import time
import sys
import os
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATASETS, RESULTS_DIR, BETAS, LAMBDAS, SEED,
    EVIDENCE_METHODS, HF_TOKEN,
)
from evidence_scores import compute_all_evidence
from weighting import naive_weights, replug_weights, dirichlet_weights
from generation import (
    generate_all_doc_answers, apply_weights, load_cache, save_cache,
    load_hf_model, get_model_and_tokenizer,
)
from metrics import evaluate_all, mcnemar_test, bootstrap_ci

np.random.seed(SEED)

# Make HF_TOKEN visible to the HuggingFace libraries via environment variable.
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN


def run_single(dataset_name, evidence_method, aggregation="voting",
               max_questions=None, betas=None, lambdas=None, top_k=None):
    """
    Run one dataset x one evidence method with a full (beta, lambda) grid.

    Args:
        dataset_name: "nq", "triviaqa", or "popqa"
        evidence_method: evidence method name
        aggregation: "voting" or "sequence"
        max_questions: optional cap on the number of questions
        betas: list of beta values (default from config if None)
        lambdas: list of lambda values (default from config if None)
        top_k: if set, truncate each query's retrieved_docs to the first top_k
            entries. Queries with fewer than top_k docs are excluded. Used for
            the Table 16 top-k ablation (k=3, 5, 10).
    """
    if betas is None:
        betas = BETAS
    if lambdas is None:
        lambdas = LAMBDAS

    data_file = DATASETS[dataset_name]
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name} | Evidence: {evidence_method} | "
          f"Aggregation: {aggregation}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    # Load the dataset
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if max_questions and len(data) > max_questions:
        data = data[:max_questions]
        print(f"Loaded {len(data)} questions (limited)")
    else:
        print(f"Loaded {len(data)} questions")

    # Optional top-k truncation (Table 16 ablation). Queries with fewer than
    # top_k retrieved documents are excluded, matching the evaluation policy
    # reported in Appendix F.
    if top_k is not None:
        before = len(data)
        data = [s for s in data if len(s.get("retrieved_docs", [])) >= top_k]
        for s in data:
            s["retrieved_docs"] = s["retrieved_docs"][:top_k]
        excluded = before - len(data)
        print(f"Top-k={top_k}: kept {len(data)}/{before} queries ({excluded} excluded for <k docs)")

    total = len(data)

    results = {}

    # ============================================================
    # Step 1: compute evidence scores
    # ============================================================
    print(f"\n[Step 1] Computing {evidence_method} evidence...")
    all_evidence = []
    ev_start = time.time()

    # Preflight: utility_predictor requires a trained model on disk.
    if evidence_method == "utility_predictor":
        from config import UTILITY_PREDICTOR_DIR
        if not os.path.exists(UTILITY_PREDICTOR_DIR):
            print(f"  SKIP: utility_predictor model not found at {UTILITY_PREDICTOR_DIR}")
            print(f"  Run train_utility_predictor.py first.")
            return {}

    # LLM-as-Judge and response_confidence need an HF model/tokenizer.
    ev_kwargs = {}
    if evidence_method in ("llm_judge", "response_confidence"):
        model, tokenizer = get_model_and_tokenizer()
        ev_kwargs = {"model": model, "tokenizer": tokenizer}

    for sample in tqdm(data, desc="Evidence"):
        evidence = compute_all_evidence(
            sample["question"], sample["retrieved_docs"],
            method=evidence_method, **ev_kwargs,
        )
        all_evidence.append(evidence)

    ev_time = time.time() - ev_start
    print(f"  Evidence time: {ev_time:.1f}s ({ev_time/total:.2f}s/q)")

    # ============================================================
    # Step 2: generate per-document LLM answers (cache-aware)
    # ============================================================
    print(f"\n[Step 2] LLM answers (cache)...")
    load_cache()
    all_cached = []
    gen_start = time.time()

    for i, sample in enumerate(tqdm(data, desc="LLM Gen")):
        cached = generate_all_doc_answers(
            sample["question"], sample["retrieved_docs"]
        )
        all_cached.append(cached)
        if (i + 1) % max(1, total // 10) == 0:
            save_cache()

    save_cache()
    gen_time = time.time() - gen_start
    print(f"  LLM time: {gen_time:.1f}s ({gen_time/60:.1f}min)")

    # ============================================================
    # Step 3: apply weights and evaluate
    # ============================================================
    print(f"\n[Step 3] Evaluating ({len(betas)} betas × {len(lambdas)} lambdas)...")
    ground_truths = [s["answers"] for s in data]

    # Store per-question correctness for downstream statistical tests.
    per_question = {}

    # --- Naive RAG ---
    print("  Naive RAG...")
    preds, confs = [], []
    for i, sample in enumerate(data):
        k = len(sample["retrieved_docs"])
        w = naive_weights(k)
        r = apply_weights(
            all_cached[i], w, method=aggregation,
            question=sample["question"], docs=sample["retrieved_docs"],
        )
        preds.append(r["answer"])
        confs.append(r["confidence"])
    results["naive"] = evaluate_all(preds, ground_truths, confs)
    from metrics import exact_match as _em
    per_question["naive"] = [_em(p, g) for p, g in zip(preds, ground_truths)]
    print(f"    EM={results['naive']['EM']:.4f}")

    # --- REPLUG + Dirichlet (β × λ grid) ---
    for beta in betas:
        # REPLUG (λ=0)
        print(f"  REPLUG β={beta}...")
        preds, confs = [], []
        for i, sample in enumerate(data):
            sims = [d["score"] for d in sample["retrieved_docs"]]
            w = replug_weights(sims, beta=beta)
            r = apply_weights(
                all_cached[i], w, method=aggregation,
                question=sample["question"], docs=sample["retrieved_docs"],
            )
            preds.append(r["answer"])
            confs.append(r["confidence"])
        results[f"replug_b{beta}"] = evaluate_all(preds, ground_truths, confs)
        per_question[f"replug_b{beta}"] = [_em(p, g) for p, g in zip(preds, ground_truths)]

        # Dirichlet-style weights across lambda values.
        for lam in lambdas:
            if lam == 0:
                continue  # lambda=0 reduces to REPLUG (Proposition 1)

            preds, confs = [], []
            for i, sample in enumerate(data):
                sims = [d["score"] for d in sample["retrieved_docs"]]
                w = dirichlet_weights(
                    sims, all_evidence[i], beta=beta, lam=lam
                )
                r = apply_weights(
                    all_cached[i], w, method=aggregation,
                    question=sample["question"],
                    docs=sample["retrieved_docs"],
                )
                preds.append(r["answer"])
                confs.append(r["confidence"])

            key = f"dirichlet_b{beta}_l{lam}_{evidence_method}"
            results[key] = evaluate_all(preds, ground_truths, confs)
            per_question[key] = [_em(p, g) for p, g in zip(preds, ground_truths)]

        # Print the best EM at the current beta.
        rep_em = results[f"replug_b{beta}"]["EM"]
        dir_keys = [k for k in results if k.startswith(f"dirichlet_b{beta}")]
        if dir_keys:
            best_dir_em = max(results[k]["EM"] for k in dir_keys)
            print(f"    β={beta}: REPLUG EM={rep_em:.4f}, "
                  f"Best Dir EM={best_dir_em:.4f}")
        else:
            print(f"    β={beta}: REPLUG EM={rep_em:.4f}")

    # ============================================================
    # Metadata + save
    # ============================================================
    # Provenance metadata
    import subprocess
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__),
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = "unknown"

    import torch, transformers, sentence_transformers
    from config import LLM_BACKEND, TOGETHER_MODEL, HF_MODEL

    results["_meta"] = {
        "dataset": dataset_name,
        "evidence_method": evidence_method,
        "aggregation": aggregation,
        "num_questions": total,
        "betas": betas,
        "lambdas": lambdas,
        "evidence_time_s": ev_time,
        "generation_time_s": gen_time,
        "timestamp": datetime.now().isoformat(),
        "provenance": {
            "git_hash": git_hash,
            "llm_backend": LLM_BACKEND,
            "llm_model": TOGETHER_MODEL if LLM_BACKEND in ("together", "openai") else HF_MODEL,
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sentence_transformers": sentence_transformers.__version__,
        },
    }
    results["_per_question"] = per_question  # for downstream statistical tests

    os.makedirs(RESULTS_DIR, exist_ok=True)
    from config import TOGETHER_MODEL
    _model_tag = TOGETHER_MODEL.split("/")[-1].lower().replace("-", "_").replace(".", "_")
    result_file = os.path.join(
        RESULTS_DIR, f"{dataset_name}_{evidence_method}_{aggregation}_{_model_tag}.json"
    )

    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=convert, ensure_ascii=False)
    print(f"\n  Saved: {result_file}")

    # Quick summary of best scores
    naive_em = results["naive"]["EM"]
    rep_ems = [results[k]["EM"] for k in results if k.startswith("replug")]
    dir_ems = [results[k]["EM"] for k in results if k.startswith("dirichlet")]

    best_rep = max(rep_ems) if rep_ems else 0
    best_dir = max(dir_ems) if dir_ems else 0

    print(f"\n  Naive={naive_em:.4f}  REPLUG={best_rep:.4f}  "
          f"Dirichlet={best_dir:.4f}")
    if best_dir > best_rep:
        print("  → Dirichlet > REPLUG")
    elif best_dir == best_rep:
        print("  → Dirichlet ≈ REPLUG")
    else:
        print("  → Dirichlet < REPLUG")

    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Dirichlet Bayesian RAG Main Experiment"
    )
    parser.add_argument(
        "--dataset", default="nq",
        choices=list(DATASETS.keys()) + ["all"],
    )
    parser.add_argument(
        "--evidence", default="cross_encoder",
        choices=EVIDENCE_METHODS + ["all"],
    )
    parser.add_argument(
        "--aggregation", default="voting",
        choices=["voting", "sequence"],
    )
    parser.add_argument(
        "--max_questions", type=int, default=None,
        help="Limit questions for quick testing",
    )
    parser.add_argument(
        "--top_k", type=int, default=None,
        help="Truncate each query's retrieved documents to top-k before weighting (default: use all retrieved docs, typically 10). Used for the Table 16 top-k ablation (k=3, 5, 10).",
    )
    args = parser.parse_args()

    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    evidences = EVIDENCE_METHODS if args.evidence == "all" else [args.evidence]

    print(f"Dirichlet Bayesian RAG Main Experiment")
    print(f"Datasets: {datasets}")
    print(f"Evidence: {evidences}")
    print(f"Aggregation: {args.aggregation}")
    print(f"Betas: {BETAS}")
    print(f"Lambdas: {LAMBDAS}")
    if args.max_questions:
        print(f"Max questions: {args.max_questions}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for ds in datasets:
        for ev in evidences:
            run_single(
                ds, ev,
                aggregation=args.aggregation,
                max_questions=args.max_questions,
                top_k=args.top_k,
            )

    print(f"\n{'='*70}")
    print(f"All done! {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
