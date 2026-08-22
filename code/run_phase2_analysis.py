"""
Phase-2 analysis script (tasks 4-10).
=========================================
4.  Repeated split stability (seeds 42 / 123 / 456)
5.  Cross-dataset transfer
6.  Evidence AUC (gold vs non-gold)
7.  SmoothECE
8.  Reliability diagram
9.  Brier score decomposition
10. AURC / risk-coverage curve

Runs for all three LLMs, reusing the existing LLM and evidence caches
(no new API calls).
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from math import exp
from scipy.optimize import minimize_scalar
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATASETS, RESULTS_DIR, DATA_DIR, SEED, BETAS, LAMBDAS
from generation import load_cache, generate_all_doc_answers
from weighting import naive_weights, replug_weights, dirichlet_weights
from metrics import (
    exact_match, f1_score, expected_calibration_error,
    normalize_answer, compute_aurc, risk_at_coverage,
)

EVIDENCE_CACHE_DIR = os.path.join(DATA_DIR, "evidence_cache")
FIGURES_DIR = os.environ.get(
    "FIGURES_DIR",
    os.path.join(os.path.dirname(RESULTS_DIR), "figures"),
)
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_evidence_cache(dataset, method):
    cache_file = os.path.join(EVIDENCE_CACHE_DIR, f"{dataset}_{method}.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)
    return None


def aggregate_vote(cached_entries, weights):
    answers = [e["answer_text"] for e in cached_entries]
    total_weight = float(sum(weights))
    answer_scores = {}
    for answer, weight in zip(answers, weights):
        normed = normalize_answer(answer)
        if normed and normed != "unknown":
            answer_scores[normed] = answer_scores.get(normed, 0.0) + weight
    if not answer_scores:
        return "", 0.0
    best_normed = max(answer_scores, key=answer_scores.get)
    confidence = answer_scores[best_normed] / total_weight if total_weight > 0 else 0.0
    for answer in answers:
        if normalize_answer(answer) == best_normed:
            return answer, confidence
    return best_normed, confidence


def compute_all_metrics(preds, ground_truths, confs):
    from sklearn.metrics import roc_auc_score
    ems = [exact_match(p, g) for p, g in zip(preds, ground_truths)]
    f1s = [f1_score(p, g) for p, g in zip(preds, ground_truths)]
    conf_arr = np.clip(np.array(confs, dtype=float), 0.0, 1.0)
    em_arr = np.array(ems, dtype=float)

    ece = expected_calibration_error(conf_arr.tolist(), ems)
    auroc = float(roc_auc_score(ems, conf_arr.tolist())) if len(set(ems)) >= 2 else 0.5
    aurc = compute_aurc(conf_arr.tolist(), ems)
    r08 = risk_at_coverage(conf_arr.tolist(), ems, 0.8)
    r09 = risk_at_coverage(conf_arr.tolist(), ems, 0.9)
    brier = float(np.mean((conf_arr - em_arr) ** 2))

    return {
        "EM": float(np.mean(ems)), "F1": float(np.mean(f1s)),
        "ECE": ece, "AUROC": auroc, "AURC": aurc,
        "Risk@0.8": r08, "Risk@0.9": r09, "Brier": brier,
        "per_em": ems, "per_conf": confs,
    }


def load_data_and_cache(dataset_name):
    with open(DATASETS[dataset_name], "r", encoding="utf-8") as f:
        data = json.load(f)
    load_cache()
    all_cached = []
    for sample in data:
        cached = generate_all_doc_answers(sample["question"], sample["retrieved_docs"])
        all_cached.append(cached)
    ground_truths = [s["answers"] for s in data]
    return data, all_cached, ground_truths


# ============================================================
# 4. Repeated Split (3 seeds)
# ============================================================
def run_repeated_split(data, all_cached, ground_truths, all_evidence, dataset_name, evidence_method):
    print(f"\n  [4] Repeated Split — {dataset_name}×{evidence_method}")
    n = len(data)
    seeds = [42, 123, 456]
    results_per_seed = []

    for seed in seeds:
        rng = np.random.RandomState(seed)
        indices = np.arange(n)
        rng.shuffle(indices)
        cal_idx = indices[:n // 2]
        test_idx = indices[n // 2:]

        # Select best (beta, lambda) on the calibration half by EM.
        best_em = -1
        best_cfg = None
        cal_gts = [ground_truths[i] for i in cal_idx]
        for beta in BETAS:
            for lam in LAMBDAS:
                if lam == 0: continue
                preds = []
                for idx in cal_idx:
                    sims = [d["score"] for d in data[idx]["retrieved_docs"]]
                    w = dirichlet_weights(sims, all_evidence[idx], beta=beta, lam=lam)
                    ans, _ = aggregate_vote(all_cached[idx], w)
                    preds.append(ans)
                em = np.mean([exact_match(p, g) for p, g in zip(preds, cal_gts)])
                if em > best_em:
                    best_em = em
                    best_cfg = (beta, lam)

        # Evaluate on the held-out test half.
        b, l = best_cfg
        test_preds, test_confs = [], []
        for idx in test_idx:
            sims = [d["score"] for d in data[idx]["retrieved_docs"]]
            w = dirichlet_weights(sims, all_evidence[idx], beta=b, lam=l)
            ans, conf = aggregate_vote(all_cached[idx], w)
            test_preds.append(ans)
            test_confs.append(conf)

        test_gts = [ground_truths[i] for i in test_idx]
        m = compute_all_metrics(test_preds, test_gts, test_confs)
        m["seed"] = seed
        m["config"] = best_cfg
        del m["per_em"]
        del m["per_conf"]
        results_per_seed.append(m)
        print(f"    seed={seed}: β={b},λ={l} → EM={m['EM']:.4f}")

    # Mean and standard deviation across seeds.
    avg = {}
    for metric in ["EM", "F1", "ECE", "AUROC", "AURC", "Brier"]:
        vals = [r[metric] for r in results_per_seed]
        avg[f"{metric}_mean"] = float(np.mean(vals))
        avg[f"{metric}_std"] = float(np.std(vals))
    print(f"    mean EM: {avg['EM_mean']:.4f} +/- {avg['EM_std']:.4f}")
    return {"per_seed": results_per_seed, "average": avg}


# ============================================================
# 5. Transfer Experiment
# ============================================================
def run_transfer(data_dict, cached_dict, gt_dict, evidence_dict, evidence_method):
    print(f"\n  [5] Transfer Experiment — {evidence_method}")
    datasets = ["nq", "triviaqa", "popqa"]
    results = {}

    for source_ds in datasets:
        # Find the source's best (beta, lambda) over its full data.
        src_data = data_dict[source_ds]
        src_cached = cached_dict[source_ds]
        src_gt = gt_dict[source_ds]
        src_ev = evidence_dict[source_ds]

        best_em = -1
        best_cfg = None
        for beta in BETAS:
            for lam in LAMBDAS:
                if lam == 0: continue
                preds = []
                for i in range(len(src_data)):
                    sims = [d["score"] for d in src_data[i]["retrieved_docs"]]
                    w = dirichlet_weights(sims, src_ev[i], beta=beta, lam=lam)
                    ans, _ = aggregate_vote(src_cached[i], w)
                    preds.append(ans)
                em = np.mean([exact_match(p, g) for p, g in zip(preds, src_gt)])
                if em > best_em:
                    best_em = em
                    best_cfg = (beta, lam)

        b, l = best_cfg

        # Apply the source config to every other dataset.
        for target_ds in datasets:
            tgt_data = data_dict[target_ds]
            tgt_cached = cached_dict[target_ds]
            tgt_gt = gt_dict[target_ds]
            tgt_ev = evidence_dict[target_ds]

            preds, confs = [], []
            for i in range(len(tgt_data)):
                sims = [d["score"] for d in tgt_data[i]["retrieved_docs"]]
                w = dirichlet_weights(sims, tgt_ev[i], beta=b, lam=l)
                ans, conf = aggregate_vote(tgt_cached[i], w)
                preds.append(ans)
                confs.append(conf)

            em = np.mean([exact_match(p, g) for p, g in zip(preds, tgt_gt)])
            key = f"{source_ds}→{target_ds}"
            results[key] = {"EM": em, "source_config": best_cfg}
            marker = "★" if source_ds == target_ds else ""
            print(f"    {key}: β={b},λ={l} → EM={em:.4f} {marker}")

    return results


# ============================================================
# 6. Evidence AUC (gold vs non-gold discrimination)
# ============================================================
def run_evidence_auc(data, all_evidence, ground_truths, all_cached, dataset_name, evidence_method):
    print(f"\n  [6] Evidence AUC — {dataset_name}×{evidence_method}")
    from sklearn.metrics import roc_auc_score

    gold_labels = []  # 1 if doc contains answer, 0 otherwise
    evidence_scores = []

    for i, sample in enumerate(data):
        answers = sample["answers"]
        for j, doc in enumerate(sample["retrieved_docs"]):
            # Gold label: whether any gold answer appears in the (normalized) document text.
            # Apply the same normalization as normalize_answer (lower-case, strip punctuation, collapse whitespace).
            import string
            doc_text_raw = doc.get("text", "")
            doc_lower = doc_text_raw.lower()
            doc_no_punc = ''.join(ch for ch in doc_lower if ch not in set(string.punctuation))
            doc_clean = ' '.join(doc_no_punc.split())
            # Empty-string guard: normalize_answer can produce "" for pathological
            # aliases (all-article or all-punctuation). "" in doc_clean is always
            # True, which would spuriously mark every document as gold. Skip such
            # aliases so they do not contaminate the AUC computation.
            is_gold = any(
                normalize_answer(a) != "" and normalize_answer(a) in doc_clean
                for a in answers
            )
            gold_labels.append(1 if is_gold else 0)
            evidence_scores.append(all_evidence[i][j])

    if len(set(gold_labels)) < 2:
        print(f"    Cannot distinguish Gold vs Non-Gold")
        return {"auc": 0.5}

    auc = roc_auc_score(gold_labels, evidence_scores)
    n_gold = sum(gold_labels)
    n_total = len(gold_labels)
    print(f"    Gold docs: {n_gold}/{n_total} ({n_gold/n_total*100:.1f}%)")
    print(f"    Evidence AUC: {auc:.4f}")
    return {"auc": auc, "n_gold": n_gold, "n_total": n_total}


# ============================================================
# 7. SmoothECE (kernel-based)
# ============================================================
def smooth_ece(confs, accs, bandwidth=0.1):
    """Simple kernel-smoothed ECE"""
    confs = np.array(confs, dtype=float)
    accs = np.array(accs, dtype=float)
    n = len(confs)
    if n == 0:
        return 0.0

    grid = np.linspace(0, 1, 100)
    total_error = 0.0
    total_weight = 0.0

    for p in grid:
        # Gaussian kernel weights
        weights = np.exp(-0.5 * ((confs - p) / bandwidth) ** 2)
        w_sum = weights.sum()
        if w_sum < 1e-8:
            continue
        avg_conf = np.sum(weights * confs) / w_sum
        avg_acc = np.sum(weights * accs) / w_sum
        total_error += w_sum * abs(avg_acc - avg_conf)
        total_weight += w_sum

    return float(total_error / total_weight) if total_weight > 0 else 0.0


def run_smooth_ece(preds_dict, gt_dict, confs_dict, dataset_name):
    print(f"\n  [7] SmoothECE — {dataset_name}")
    results = {}
    for method, confs in confs_dict.items():
        gts = gt_dict[method]
        ems = [exact_match(p, g) for p, g in zip(preds_dict[method], gts)]
        sece = smooth_ece(confs, ems)
        results[method] = sece
        print(f"    {method}: SmoothECE={sece:.4f}")
    return results


# ============================================================
# 8. Reliability Diagram
# ============================================================
def plot_reliability_diagram(confs_dict, ems_dict, dataset_name, model_tag, ev_method="cross_encoder"):
    print(f"\n  [8] Reliability Diagram — {dataset_name} ({model_tag})")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    methods = ["Naive", "REPLUG", "Dirichlet"]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    for ax, method, color in zip(axes, methods, colors):
        confs = np.array(confs_dict[method], dtype=float)
        ems = np.array(ems_dict[method], dtype=float)

        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_accs, bin_confs, bin_counts = [], [], []

        for b in range(n_bins):
            mask = (confs >= bin_edges[b]) & (confs < bin_edges[b + 1])
            if b == n_bins - 1:
                mask = (confs >= bin_edges[b]) & (confs <= bin_edges[b + 1])
            if mask.sum() > 0:
                bin_accs.append(ems[mask].mean())
                bin_confs.append(confs[mask].mean())
                bin_counts.append(mask.sum())
            else:
                bin_accs.append(0)
                bin_confs.append((bin_edges[b] + bin_edges[b + 1]) / 2)
                bin_counts.append(0)

        bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(n_bins)]

        ax.bar(bin_centers, bin_accs, width=0.09, alpha=0.7, color=color, label="Accuracy")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{method}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)

    plt.suptitle(f"Reliability Diagram — {dataset_name} ({model_tag})", fontsize=14)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, f"reliability_{dataset_name}_{ev_method}_{model_tag}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {fig_path}")


# ============================================================
# 9. Brier Score Decomposition
# ============================================================
def brier_decomposition(confs, ems, n_bins=10):
    """Brier = Reliability - Resolution + Uncertainty"""
    confs = np.array(confs, dtype=float)
    ems = np.array(ems, dtype=float)
    n = len(confs)
    if n == 0:
        return {"reliability": 0, "resolution": 0, "uncertainty": 0, "brier": 0}

    overall_acc = ems.mean()
    uncertainty = overall_acc * (1 - overall_acc)

    bin_ids = np.minimum((confs * n_bins).astype(int), n_bins - 1)
    reliability = 0.0
    resolution = 0.0

    for b in range(n_bins):
        mask = bin_ids == b
        nb = mask.sum()
        if nb == 0:
            continue
        avg_conf = confs[mask].mean()
        avg_acc = ems[mask].mean()
        reliability += (nb / n) * (avg_conf - avg_acc) ** 2
        resolution += (nb / n) * (avg_acc - overall_acc) ** 2

    brier = reliability - resolution + uncertainty
    return {
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "brier": float(brier),
    }


# ============================================================
# 10. AURC / Risk-Coverage Curve
# ============================================================
def plot_risk_coverage(confs_dict, ems_dict, dataset_name, model_tag, ev_method="cross_encoder"):
    print(f"\n  [10] Risk-Coverage Curve — {dataset_name} ({model_tag})")
    fig, ax = plt.subplots(figsize=(8, 6))

    methods = ["Naive", "REPLUG", "Dirichlet"]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    styles = ["--", "-.", "-"]

    for method, color, style in zip(methods, colors, styles):
        confs = np.array(confs_dict[method], dtype=float)
        ems = np.array(ems_dict[method], dtype=float)
        n = len(confs)

        sorted_indices = np.argsort(-confs)
        sorted_correct = ems[sorted_indices]

        coverages = np.linspace(1 / n, 1.0, n)
        risks = []
        for i in range(1, n + 1):
            risk = 1.0 - np.mean(sorted_correct[:i])
            risks.append(risk)

        ax.plot(coverages, risks, color=color, linestyle=style, label=method, linewidth=2)

    ax.set_xlabel("Coverage", fontsize=12)
    ax.set_ylabel("Risk (1 - Accuracy)", fontsize=12)
    ax.set_title(f"Risk-Coverage Curve — {dataset_name} ({model_tag})", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, f"risk_coverage_{dataset_name}_{ev_method}_{model_tag}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {fig_path}")


# ============================================================
# Main
# ============================================================
def run_all_for_model(model_tag, env_vars=None):
    if env_vars:
        for k, v in env_vars.items():
            os.environ[k] = v
        # Reload config / generation modules so the new env vars take effect.
        import importlib
        import config
        importlib.reload(config)
        import generation
        importlib.reload(generation)

    print(f"\n{'='*70}")
    print(f"Phase 2 Analysis — {model_tag}")
    print(f"{'='*70}")

    datasets = ["nq", "triviaqa", "popqa"]
    ev_methods = ["cross_encoder", "embedding_stability"]  # NLI is excluded (no evidence cache)

    all_results = {}

    # Load datasets one by one.
    data_dict, cached_dict, gt_dict = {}, {}, {}
    for ds in datasets:
        print(f"\n  Loading {ds}...")
        data_dict[ds], cached_dict[ds], gt_dict[ds] = load_data_and_cache(ds)

    for ev in ev_methods:
        print(f"\n{'='*50}")
        print(f"Evidence: {ev}")
        print(f"{'='*50}")

        evidence_dict = {}
        for ds in datasets:
            evidence_dict[ds] = load_evidence_cache(ds, ev)
            if not evidence_dict[ds]:
                print(f"  WARNING: {ds}_{ev} evidence cache not found!")
                continue

        # --- 4. Repeated Split ---
        for ds in datasets:
            if not evidence_dict.get(ds):
                continue
            key = f"repeated_split_{ds}_{ev}"
            all_results[key] = run_repeated_split(
                data_dict[ds], cached_dict[ds], gt_dict[ds],
                evidence_dict[ds], ds, ev
            )

        # --- 5. Transfer ---
        valid_ds = [ds for ds in datasets if evidence_dict.get(ds)]
        if len(valid_ds) == 3:
            key = f"transfer_{ev}"
            all_results[key] = run_transfer(
                data_dict, cached_dict, gt_dict, evidence_dict, ev
            )

        # --- 6. Evidence AUC ---
        for ds in datasets:
            if not evidence_dict.get(ds):
                continue
            key = f"evidence_auc_{ds}_{ev}"
            all_results[key] = run_evidence_auc(
                data_dict[ds], evidence_dict[ds], gt_dict[ds],
                cached_dict[ds], ds, ev
            )

        # --- 7-10: Per-dataset analysis ---
        for ds in datasets:
            if not evidence_dict.get(ds):
                continue

            data = data_dict[ds]
            all_cached = cached_dict[ds]
            gts = gt_dict[ds]
            evidence = evidence_dict[ds]

            # Collect predictions and confidences for each method.
            preds_d, confs_d, ems_d, gt_d = {}, {}, {}, {}

            # Naive
            preds, confs = [], []
            for i in range(len(data)):
                k = len(data[i]["retrieved_docs"])
                w = naive_weights(k)
                ans, conf = aggregate_vote(all_cached[i], w)
                preds.append(ans)
                confs.append(conf)
            preds_d["Naive"] = preds
            confs_d["Naive"] = confs
            gt_d["Naive"] = gts
            ems_d["Naive"] = np.array([exact_match(p, g) for p, g in zip(preds, gts)], dtype=float)

            # REPLUG
            best_rep_beta = 0.5
            preds, confs = [], []
            for i in range(len(data)):
                sims = [d["score"] for d in data[i]["retrieved_docs"]]
                w = replug_weights(sims, beta=best_rep_beta)
                ans, conf = aggregate_vote(all_cached[i], w)
                preds.append(ans)
                confs.append(conf)
            preds_d["REPLUG"] = preds
            confs_d["REPLUG"] = confs
            gt_d["REPLUG"] = gts
            ems_d["REPLUG"] = np.array([exact_match(p, g) for p, g in zip(preds, gts)], dtype=float)

            # Dirichlet (best EM config from full data)
            best_em = -1
            best_cfg = None
            for beta in BETAS:
                for lam in LAMBDAS:
                    if lam == 0: continue
                    ps = []
                    for i in range(len(data)):
                        sims = [d["score"] for d in data[i]["retrieved_docs"]]
                        w = dirichlet_weights(sims, evidence[i], beta=beta, lam=lam)
                        ans, _ = aggregate_vote(all_cached[i], w)
                        ps.append(ans)
                    em = np.mean([exact_match(p, g) for p, g in zip(ps, gts)])
                    if em > best_em:
                        best_em = em
                        best_cfg = (beta, lam)

            b, l = best_cfg
            preds, confs = [], []
            for i in range(len(data)):
                sims = [d["score"] for d in data[i]["retrieved_docs"]]
                w = dirichlet_weights(sims, evidence[i], beta=b, lam=l)
                ans, conf = aggregate_vote(all_cached[i], w)
                preds.append(ans)
                confs.append(conf)
            preds_d["Dirichlet"] = preds
            confs_d["Dirichlet"] = confs
            gt_d["Dirichlet"] = gts
            ems_d["Dirichlet"] = np.array([exact_match(p, g) for p, g in zip(preds, gts)], dtype=float)

            # 7. SmoothECE
            sece_results = {}
            for method in ["Naive", "REPLUG", "Dirichlet"]:
                sece = smooth_ece(confs_d[method], ems_d[method].tolist())
                sece_results[method] = sece
            all_results[f"smooth_ece_{ds}_{ev}"] = sece_results
            print(f"\n  [7] SmoothECE — {ds}×{ev}: N={sece_results['Naive']:.4f} R={sece_results['REPLUG']:.4f} D={sece_results['Dirichlet']:.4f}")

            # 8. Reliability Diagram
            plot_reliability_diagram(confs_d, ems_d, ds, model_tag, ev_method=ev)

            # 9. Brier Decomposition
            brier_results = {}
            for method in ["Naive", "REPLUG", "Dirichlet"]:
                bd = brier_decomposition(confs_d[method], ems_d[method].tolist())
                brier_results[method] = bd
            all_results[f"brier_decomp_{ds}_{ev}"] = brier_results
            print(f"  [9] Brier Decomp — {ds}×{ev}:")
            for method in ["Naive", "REPLUG", "Dirichlet"]:
                bd = brier_results[method]
                print(f"    {method}: Rel={bd['reliability']:.4f} Res={bd['resolution']:.4f} Unc={bd['uncertainty']:.4f} Brier={bd['brier']:.4f}")

            # 10. Risk-Coverage Curve
            plot_risk_coverage(confs_d, ems_d, ds, model_tag, ev_method=ev)

    # Save results
    output_file = os.path.join(RESULTS_DIR, f"phase2_analysis_{model_tag}.json")

    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=convert, ensure_ascii=False)
    print(f"\nSaved: {output_file}")


def main():
    print("=" * 70)
    print(f"Phase 2 Analysis — Started {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 70)

    # Qwen
    run_all_for_model("qwen2_5_7b", env_vars={})

    # gpt-4.1-mini
    run_all_for_model("gpt_4_1_mini", env_vars={
        "LLM_BACKEND": "openai",
        "LLM_MODEL": "gpt-4.1-mini",
    })

    # Llama-3.3-70B
    run_all_for_model("llama_3_3_70b", env_vars={
        "LLM_BACKEND": "together",
        "LLM_MODEL": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    })

    print(f"\n{'='*70}")
    print(f"Phase 2 Complete! {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
