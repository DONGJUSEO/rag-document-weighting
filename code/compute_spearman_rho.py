#!/usr/bin/env python3
"""Compute pooled Spearman rho between retriever score and gold-document label
for NQ / TriviaQA / PopQA.

Reproduces the values reported in §6.1 of the paper:
  NQ       rho = 0.20   (computed 0.2036)
  TriviaQA rho = 0.30   (computed 0.3002)
  PopQA    rho = 0.29   (computed 0.2874)

Method: dataset-level pooled Spearman. We concatenate all (similarity_score,
gold_label) pairs across all queries and compute a single Spearman rank
correlation, matching the pooled ROC framing used for AUC (Appendix C).
Per-query averaging yields a different (generally smaller) correlation;
we chose the pooled form because "retriever tracks answer utility" is
naturally a dataset-level claim.
"""
import os
import json
import string
import numpy as np
from scipy.stats import spearmanr

DATA_DIR = os.environ.get("DATA_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")))

DATA_FILES = {
    "nq":       os.path.join(DATA_DIR, "nq_cosine.json"),
    "triviaqa": os.path.join(DATA_DIR, "triviaqa_cosine.json"),
    "popqa":    os.path.join(DATA_DIR, "popqa_contriever.json"),
}


def normalize_answer(s):
    """Match metrics.normalize_answer: lowercase, strip articles, strip punct."""
    if not s:
        return ""
    s = s.lower()
    articles = {"a", "an", "the"}
    s = " ".join(t for t in s.split() if t not in articles)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.split())


def is_gold(doc_text, answers):
    doc_lower = doc_text.lower()
    doc_no_punc = "".join(ch for ch in doc_lower if ch not in set(string.punctuation))
    doc_clean = " ".join(doc_no_punc.split())
    for a in answers:
        if normalize_answer(a) and normalize_answer(a) in doc_clean:
            return 1
    return 0


def compute_rho(dataset):
    path = DATA_FILES[dataset]
    with open(path) as f:
        data = json.load(f)

    scores, labels = [], []
    for sample in data:
        answers = sample.get("answers", [])
        if not answers:
            continue
        for doc in sample.get("retrieved_docs", []):
            sim = float(doc.get("score", 0.0))
            gold = is_gold(doc.get("text", ""), answers)
            scores.append(sim)
            labels.append(gold)
    scores = np.array(scores)
    labels = np.array(labels)
    rho, _ = spearmanr(scores, labels)
    return float(rho), len(scores)


if __name__ == "__main__":
    print("Pooled Spearman rho (retriever score vs gold-label):")
    for ds in ["nq", "triviaqa", "popqa"]:
        rho, n = compute_rho(ds)
        print(f"  {ds:10s}  rho = {rho:.4f}  (n = {n} query-document pairs)")
