"""
Document-weighting module.
Three configurations: Naive, REPLUG, Dirichlet (our parameter family).

Formulas:
  Naive:     w_i = 1/k
  REPLUG:    w_i = softmax(beta * s_i)
  Dirichlet: w_i = (exp(beta * s_i) + lambda * e_i) / sum_j(exp(beta * s_j) + lambda * e_j)

  s_i = DPR cosine similarity (in [-1, 1])
  e_i = evidence score (in [0, 1])
  beta  = prior strength
  lambda = evidence strength

  Proposition 1: when lambda = 0, Dirichlet reduces to REPLUG, i.e. softmax(beta * s).
"""
import numpy as np
from math import exp


def naive_weights(k):
    """
    Naive RAG: uniform weights.
    w_i = 1/k
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    return [1.0 / k] * k


def replug_weights(similarities, beta=1.0):
    """
    REPLUG: softmax(beta * s).
    w_i = exp(beta * s_i) / sum_j exp(beta * s_j)

    Args:
        similarities: list of cosine similarities
        beta: temperature (prior strength)

    Returns:
        list[float]: weights that sum to 1
    """
    if not similarities:
        raise ValueError("similarities must be non-empty")
    scores = [beta * s for s in similarities]
    # Subtract max for numerical stability
    max_score = max(scores)
    exp_scores = [exp(s - max_score) for s in scores]
    total = sum(exp_scores)
    return [e / total for e in exp_scores]


def dirichlet_weights(similarities, evidence_scores, beta=1.0, lam=1.0):
    """
    Dirichlet-style unified weights.

    w_i = (exp(beta * s_i) + lambda * e_i) / sum_j(exp(beta * s_j) + lambda * e_j)

    alpha_i = exp(beta * s_i)  -> prior term (retrieval similarity)
    lambda * e_i               -> scaled pseudo-count (evidence)

    Proposition 1: when lambda = 0, w_i reduces to softmax(beta * s) = REPLUG.

    Args:
        similarities: list of cosine similarities (s_i in [-1, 1])
        evidence_scores: list of evidence scores (e_i in [0, 1])
        beta: prior strength (beta > 0)
        lam: evidence strength (lam >= 0)

    Returns:
        list[float]: weights that sum to 1
    """
    if not similarities:
        raise ValueError("similarities must be non-empty")
    assert beta > 0, f"beta must be positive, got {beta}"
    assert len(similarities) == len(evidence_scores), \
        "similarities and evidence must have same length"
    assert all(e >= 0 for e in evidence_scores), \
        "evidence scores must be non-negative"
    assert lam >= 0, f"lambda must be non-negative, got {lam}"

    max_exp_arg = beta * max(similarities) if similarities else 0
    assert max_exp_arg < 700, f"Risk of overflow: beta*max(sim)={max_exp_arg}"

    # Prior: alpha_i = exp(beta * s_i)
    alphas = [exp(beta * s) for s in similarities]

    # Numerator (alpha_i + lambda * e_i) normalized by the total.
    numerators = [a + lam * e for a, e in zip(alphas, evidence_scores)]
    total = sum(numerators)

    return [n / total for n in numerators]


def get_dirichlet_concentration(similarities, evidence_scores, beta=1.0, lam=1.0):
    """
    Return the concentration parameters of the Dirichlet posterior.

    gamma_i = exp(beta * s_i) + lambda * e_i
    S = sum_i gamma_i  (larger S -> more concentrated distribution)

    Returns:
        tuple: (gammas, total_concentration)
    """
    alphas = [exp(beta * s) for s in similarities]
    gammas = [a + lam * e for a, e in zip(alphas, evidence_scores)]
    return gammas, sum(gammas)


def get_dirichlet_variance(similarities, evidence_scores, beta=1.0, lam=1.0):
    """
    Variance of each coordinate of the Dirichlet posterior.

    Var(w_i) = gamma_i (S - gamma_i) / (S^2 (S + 1))
    where gamma_i = alpha_i + lambda * e_i and S = sum_j gamma_j.

    Returns:
        list[float]: variance of each w_i
    """
    gammas, S = get_dirichlet_concentration(
        similarities, evidence_scores, beta, lam
    )
    variances = [g * (S - g) / (S * S * (S + 1)) for g in gammas]
    return variances


# ============================================================
# Verification helpers
# ============================================================

def verify_weights(weights, name=""):
    """Check that weights are non-negative and sum to 1."""
    total = sum(weights)
    all_positive = all(w > 0 for w in weights)
    assert abs(total - 1.0) < 1e-6, f"{name}: sum={total}, expected 1.0"
    assert all_positive, f"{name}: negative weights found"
    return True


def verify_replug_equivalence(similarities, beta=1.0, tol=1e-6):
    """Verify Proposition 1: with lambda = 0, Dirichlet weights equal REPLUG weights for any evidence."""
    k = len(similarities)
    replug_w = replug_weights(similarities, beta)

    # With lambda = 0, Dirichlet should agree with REPLUG regardless of evidence values.
    for evidence in [[0.0] * k, [0.5] * k, [1.0] * k, [0.3, 0.7, 0.1] + [0.5] * (k - 3)]:
        if len(evidence) != k:
            evidence = evidence[:k] if len(evidence) > k else evidence + [0.5] * (k - len(evidence))
        dirichlet_w = dirichlet_weights(similarities, evidence, beta, lam=0.0)
        for i in range(k):
            diff = abs(replug_w[i] - dirichlet_w[i])
            assert diff < tol, (
                f"lambda=0 equivalence failed at {i}: "
                f"REPLUG={replug_w[i]:.8f}, Dir={dirichlet_w[i]:.8f}, "
                f"evidence={evidence}"
            )
    return True


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":
    # Sample cosine similarities spanning the typical range.
    sims = [0.637, 0.127, 0.536, 0.003, 0.058, -0.106, 0.236, 0.103, 0.067, 0.429]
    evidence = [0.51, 0.01, 0.99, 0.01, 0.01, 0.01, 0.30, 0.01, 0.01, 0.60]

    print("=== Naive Weights ===")
    w_naive = naive_weights(10)
    verify_weights(w_naive, "Naive")
    print(f"  All equal: {w_naive[0]:.4f}")

    print("\n=== REPLUG Weights (beta=2.0) ===")
    w_replug = replug_weights(sims, beta=2.0)
    verify_weights(w_replug, "REPLUG")
    for i, (s, w) in enumerate(zip(sims, w_replug)):
        print(f"  Doc {i+1}: sim={s:.3f} -> w={w:.4f}")

    print("\n=== Dirichlet Weights (beta=2.0, lambda=1.0) ===")
    w_dir = dirichlet_weights(sims, evidence, beta=2.0, lam=1.0)
    verify_weights(w_dir, "Dirichlet")
    for i, (s, e, w) in enumerate(zip(sims, evidence, w_dir)):
        print(f"  Doc {i+1}: sim={s:.3f}, evi={e:.2f} -> w={w:.4f}")

    print("\n=== Dirichlet Weights (beta=2.0, lambda=10.0) ===")
    w_dir10 = dirichlet_weights(sims, evidence, beta=2.0, lam=10.0)
    verify_weights(w_dir10, "Dirichlet lambda=10")
    for i, (s, e, w) in enumerate(zip(sims, evidence, w_dir10)):
        print(f"  Doc {i+1}: sim={s:.3f}, evi={e:.2f} -> w={w:.4f}")

    print("\n=== Proposition 1: lambda=0 -> REPLUG (all beta, all evidence) ===")
    for beta in [0.5, 1.0, 2.0, 4.0]:
        verified = verify_replug_equivalence(sims, beta)
        print(f"  beta={beta}: {'PASS' if verified else 'FAIL'}")

    print("\n=== Weight change as lambda varies ===")
    for lam in [0, 0.1, 0.5, 1.0, 3.0, 10.0, 30.0]:
        w = dirichlet_weights(sims, evidence, beta=2.0, lam=lam)
        print(f"  lambda={lam:5.1f}: max_w={max(w):.4f}, "
              f"min_w={min(w):.4f}, top_doc={np.argmax(w)+1}")

    print("\n=== Variance (beta=2.0, lambda=1.0) ===")
    var = get_dirichlet_variance(sims, evidence, beta=2.0, lam=1.0)
    for i, (w, v) in enumerate(zip(w_dir, var)):
        print(f"  Doc {i+1}: w={w:.4f}, var={v:.6f}")

    print("\nAll tests passed!")
