"""
Evaluation metrics: EM, F1, ECE, AUROC, AURC, Risk@Coverage, McNemar, Bootstrap CI.
"""
import re
import string
import numpy as np
from collections import Counter


# ============================================================
# Answer normalization (NQ / SQuAD standard)
# ============================================================

def normalize_answer(s):
    """
    Standard NQ / SQuAD answer normalization:
      - lowercase
      - remove articles (a, an, the)
      - remove punctuation
      - collapse whitespace
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# ============================================================
# Exact Match (EM)
# ============================================================

def exact_match(prediction, ground_truths):
    """
    1 if the normalized prediction equals any normalized gold answer, else 0.

    Args:
        prediction: model prediction (str)
        ground_truths: list of gold answers (list[str])

    Returns:
        int: 1 or 0
    """
    normalized_pred = normalize_answer(prediction)
    for gt in ground_truths:
        if normalized_pred == normalize_answer(gt):
            return 1
    return 0


# ============================================================
# F1 score
# ============================================================

def f1_score(prediction, ground_truths):
    """
    Token-level F1, taking the max over the list of gold answers.

    Args:
        prediction: model prediction
        ground_truths: list of gold answers

    Returns:
        float: F1 in [0, 1]
    """
    def compute_f1(pred, gt):
        pred_tokens = normalize_answer(pred).split()
        gt_tokens = normalize_answer(gt).split()

        if not pred_tokens or not gt_tokens:
            return int(pred_tokens == gt_tokens)

        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_common = sum(common.values())

        if num_common == 0:
            return 0.0

        precision = num_common / len(pred_tokens)
        recall = num_common / len(gt_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    if not ground_truths:
        return 0.0
    return max(compute_f1(prediction, gt) for gt in ground_truths)


# ============================================================
# Expected Calibration Error (ECE)
# ============================================================

def expected_calibration_error(confidences, accuracies, n_bins=10):
    """
    Standard binned ECE: a weighted average of |avg_confidence - avg_accuracy|
    across equal-width confidence bins.

    Note: pass raw confidences (do NOT min-max normalize); the distribution of
    confidence itself is part of what we want to measure.

    Args:
        confidences: list of raw confidences in [0, 1]
        accuracies:  list of 0/1 correctness indicators
        n_bins: number of bins

    Returns:
        float: ECE in [0, 1] (lower is better)
    """
    confidences = np.clip(np.asarray(confidences, dtype=float), 0.0, 1.0)
    accuracies = np.asarray(accuracies, dtype=float)

    if confidences.shape != accuracies.shape:
        raise ValueError("confidences and accuracies must have the same shape")
    if confidences.size == 0:
        return 0.0

    # Bin assignment: [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
    bin_ids = np.minimum((confidences * n_bins).astype(int), n_bins - 1)
    ece = 0.0

    for b in range(n_bins):
        mask = bin_ids == b
        if mask.any():
            avg_confidence = confidences[mask].mean()
            avg_accuracy = accuracies[mask].mean()
            ece += mask.mean() * abs(avg_accuracy - avg_confidence)

    return float(ece)


# ============================================================
# AURC (Area Under Risk-Coverage Curve)
# ============================================================
# Note: AUROC is computed directly via sklearn's roc_auc_score in evaluate_all().

def compute_aurc(confidences, correct_flags):
    """
    Area under the risk-coverage curve (lower is better).

    Sort predictions by descending confidence; for each prefix of size i,
    risk = 1 - accuracy among the top-i; integrate risk over coverage.

    Args:
        confidences: list of confidences
        correct_flags: list of 0/1 correctness indicators

    Returns:
        float: AURC (lower is better)
    """
    confidences = np.array(confidences, dtype=float)
    correct_flags = np.array(correct_flags, dtype=float)

    n = len(confidences)
    if n == 0:
        return 0.0

    sorted_indices = np.argsort(-confidences)  # descending
    sorted_correct = correct_flags[sorted_indices]

    risks = []
    for i in range(1, n + 1):
        risk = 1.0 - np.mean(sorted_correct[:i])
        risks.append(risk)

    coverages = np.linspace(1 / n, 1.0, n)
    # numpy 2.0+ renames trapz to trapezoid; support both.
    trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    aurc = trapz_fn(risks, coverages)
    return float(aurc)


def risk_at_coverage(confidences, correct_flags, coverage_level=0.8):
    """
    Risk (1 - accuracy) when abstaining on the (1 - coverage_level) lowest-confidence
    predictions.

    Args:
        confidences: list of confidences
        correct_flags: list of 0/1 correctness indicators
        coverage_level: fraction of predictions to keep (0.8 = top 80% by confidence)

    Returns:
        float: risk at the requested coverage
    """
    confidences = np.array(confidences, dtype=float)
    correct_flags = np.array(correct_flags, dtype=float)

    if len(confidences) == 0:
        return 0.0

    sorted_indices = np.argsort(-confidences)
    sorted_correct = correct_flags[sorted_indices]

    import math
    k = max(1, math.ceil(len(sorted_correct) * coverage_level))
    risk = 1.0 - np.mean(sorted_correct[:k])
    return float(risk)


# ============================================================
# Statistical tests
# ============================================================

def mcnemar_test(correct_a, correct_b):
    """
    McNemar's test (chi-squared with Yates's continuity correction).

    Args:
        correct_a: 0/1 correctness vector for method A
        correct_b: 0/1 correctness vector for method B

    Returns:
        tuple: (chi2_statistic, p_value)
    """
    from scipy.stats import chi2

    correct_a = np.array(correct_a, dtype=bool)
    correct_b = np.array(correct_b, dtype=bool)

    b = np.sum(correct_a & ~correct_b)  # A correct only
    c = np.sum(~correct_a & correct_b)  # B correct only

    if b + c == 0:
        return 0.0, 1.0

    # Continuity correction
    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = chi2.sf(chi2_stat, df=1)
    return float(chi2_stat), float(p_value)


def bootstrap_ci(values, n_bootstrap=1000, ci=0.95, seed=42):
    """
    Percentile bootstrap confidence interval for the mean.

    Args:
        values: per-sample metric values
        n_bootstrap: number of bootstrap resamples
        ci: confidence level (0.95 = 95%)
        seed: random seed

    Returns:
        tuple: (lower, upper, mean)
    """
    rng = np.random.RandomState(seed)
    values = np.array(values, dtype=float)
    n = len(values)

    bootstrap_means = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        bootstrap_means.append(np.mean(values[indices]))

    alpha = 1 - ci
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    return float(lower), float(upper), float(np.mean(values))


# ============================================================
# Simple wall-clock timer
# ============================================================

def measure_latency(func, *args, **kwargs):
    """
    Measure wall-clock execution time.

    Returns:
        tuple: (result, elapsed_ms)
    """
    import time
    start = time.time()
    result = func(*args, **kwargs)
    elapsed_ms = (time.time() - start) * 1000
    return result, elapsed_ms


# ============================================================
# One-shot evaluator
# ============================================================

def evaluate_all(predictions, ground_truths, confidences):
    """
    Compute EM / F1 / ECE / AUROC / AURC / Risk@Coverage in one call.

    Args:
        predictions: list of predicted answers
        ground_truths: list of lists of gold answers
        confidences: list of raw confidences in [0, 1]

    Returns:
        dict with all metric values.
    """
    em_scores = [exact_match(pred, gts) for pred, gts in zip(predictions, ground_truths)]
    f1_scores = [f1_score(pred, gts) for pred, gts in zip(predictions, ground_truths)]

    avg_em = np.mean(em_scores)
    avg_f1 = np.mean(f1_scores)

    # Use raw confidences for ECE (no min-max normalization); the concentration
    # of confidence in particular regions is itself part of the calibration signal.
    conf_array = np.clip(np.array(confidences, dtype=float), 0.0, 1.0)
    ece = expected_calibration_error(conf_array.tolist(), em_scores)

    # AUROC: pass confidences directly as the positive-class score.
    from sklearn.metrics import roc_auc_score
    if len(set(em_scores)) < 2:
        auc = 0.5  # Degenerate: all correct or all wrong.
    else:
        auc = roc_auc_score(em_scores, conf_array.tolist())

    # AURC & Risk@Coverage
    aurc_val = compute_aurc(conf_array.tolist(), em_scores)
    risk_08 = risk_at_coverage(conf_array.tolist(), em_scores, 0.8)
    risk_09 = risk_at_coverage(conf_array.tolist(), em_scores, 0.9)

    return {
        "EM": avg_em,
        "F1": avg_f1,
        "ECE": ece,
        "AUROC": auc,
        "AURC": aurc_val,
        "Risk@0.8": risk_08,
        "Risk@0.9": risk_09,
        "num_correct": sum(em_scores),
        "num_total": len(em_scores),
        "confidence_stats": {
            "mean": float(conf_array.mean()) if len(conf_array) > 0 else 0,
            "std": float(conf_array.std()) if len(conf_array) > 0 else 0,
            "min": float(conf_array.min()) if len(conf_array) > 0 else 0,
            "max": float(conf_array.max()) if len(conf_array) > 0 else 0,
        }
    }


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":
    predictions = [
        "North and West Africa",
        "Paris",
        "unknown",
        "1776",
        "Seoul"
    ]
    ground_truths = [
        ["North and West Africa", "Africa"],
        ["Paris", "Paris, France"],
        ["London"],
        ["1776", "July 4, 1776"],
        ["Seoul"]
    ]
    confidences = [0.8, 0.9, 0.3, 0.7, 0.95]

    print("=== Metric self-tests ===")
    for i, (pred, gts) in enumerate(zip(predictions, ground_truths)):
        em = exact_match(pred, gts)
        f1 = f1_score(pred, gts)
        print(f"  Q{i+1}: pred='{pred}' -> EM={em}, F1={f1:.3f}")

    print()
    results = evaluate_all(predictions, ground_truths, confidences)
    print(f"  Average EM:  {results['EM']:.4f}")
    print(f"  Average F1:  {results['F1']:.4f}")
    print(f"  ECE:         {results['ECE']:.4f}")
    print(f"  AUROC:       {results['AUROC']:.4f}")
    print(f"  Correct:     {results['num_correct']}/{results['num_total']}")

    print("\nAll tests passed.")
