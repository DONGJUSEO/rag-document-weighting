"""Compute NLI both directions (forward and reverse) on all 3 datasets,
then compute gold-vs-non-gold AUC for each direction.

Forward: premise=query, hypothesis=document (current paper choice)
Reverse: premise=document, hypothesis=query (standard NLI direction)

Output: JSON with per-dataset AUC for both directions.
"""
import json
import os
import sys
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = "./data"
RESULTS_DIR = "./results"

DATASETS = {
    "nq": os.path.join(DATA_DIR, "nq_cosine.json"),
    "triviaqa": os.path.join(DATA_DIR, "triviaqa_cosine.json"),
    "popqa": os.path.join(DATA_DIR, "popqa_contriever.json"),
}

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
SAMPLE_SIZE = 500  # queries per dataset — balances speed vs AUC stability
BATCH_SIZE = 16


def has_gold_answer(doc_text, answers):
    """Check if doc contains any gold answer string."""
    if not doc_text:
        return 0
    doc_lower = doc_text.lower()
    for ans in answers:
        if ans and ans.lower() in doc_lower:
            return 1
    return 0


def load_model():
    print(f"Loading NLI model: {NLI_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
    model.eval()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    config = AutoConfig.from_pretrained(NLI_MODEL)
    label2id = {label.lower(): idx for idx, label in config.id2label.items()}
    label_ids = {
        "entailment": label2id["entailment"],
        "neutral": label2id["neutral"],
        "contradiction": label2id["contradiction"],
    }
    print(f"Device: {device}")
    print(f"Labels: {label_ids}")
    return model, tokenizer, label_ids, device


def compute_relevance_batch(model, tokenizer, label_ids, device, premises, hypotheses):
    """Compute P(ent) + 0.5 * P(neutral) for a batch of (premise, hypothesis) pairs."""
    inputs = tokenizer(
        premises, hypotheses,
        return_tensors="pt", truncation=True, max_length=512, padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
    ent_id = label_ids["entailment"]
    neu_id = label_ids["neutral"]
    return (probs[:, ent_id] + 0.5 * probs[:, neu_id]).cpu().numpy().tolist()


def run_one_dataset(dataset_name, dataset_path, model, tokenizer, label_ids, device):
    print(f"\n=== {dataset_name} ===")
    with open(dataset_path) as f:
        data = json.load(f)

    # Sample (fixed seed for reproducibility)
    rng = np.random.RandomState(42)
    n_total = len(data)
    indices = rng.choice(n_total, size=min(SAMPLE_SIZE, n_total), replace=False)
    indices = sorted(indices)

    forward_scores = []  # (query, doc)
    reverse_scores = []  # (doc, query)
    gold_labels = []

    for idx in tqdm(indices, desc=dataset_name):
        sample = data[idx]
        q = sample["question"]
        docs = sample["retrieved_docs"]
        answers = sample.get("answers", [])

        doc_texts = [d.get("text", "")[:1000] for d in docs]

        # Forward: premise=query, hypothesis=document
        premises_f = [q] * len(docs)
        hypotheses_f = doc_texts

        # Reverse: premise=document, hypothesis=query
        premises_r = doc_texts
        hypotheses_r = [q] * len(docs)

        # Batch both directions
        for b_start in range(0, len(docs), BATCH_SIZE):
            b_end = min(b_start + BATCH_SIZE, len(docs))
            fs = compute_relevance_batch(
                model, tokenizer, label_ids, device,
                premises_f[b_start:b_end], hypotheses_f[b_start:b_end]
            )
            rs = compute_relevance_batch(
                model, tokenizer, label_ids, device,
                premises_r[b_start:b_end], hypotheses_r[b_start:b_end]
            )
            forward_scores.extend(fs)
            reverse_scores.extend(rs)

        # Gold labels (per doc)
        for d in docs:
            gold_labels.append(has_gold_answer(d.get("text", ""), answers))

    # AUC (gold vs non-gold)
    forward_auc = float(roc_auc_score(gold_labels, forward_scores)) if len(set(gold_labels)) > 1 else None
    reverse_auc = float(roc_auc_score(gold_labels, reverse_scores)) if len(set(gold_labels)) > 1 else None

    print(f"  Forward AUC (q→d):  {forward_auc:.4f}")
    print(f"  Reverse AUC (d→q):  {reverse_auc:.4f}")
    print(f"  n_pairs: {len(gold_labels)}, gold fraction: {np.mean(gold_labels):.3f}")

    return {
        "dataset": dataset_name,
        "n_queries_sampled": len(indices),
        "n_pairs": len(gold_labels),
        "gold_fraction": float(np.mean(gold_labels)),
        "forward_auc_q_as_premise": forward_auc,
        "reverse_auc_d_as_premise": reverse_auc,
    }


def main():
    model, tokenizer, label_ids, device = load_model()
    results = {}
    for ds_name, ds_path in DATASETS.items():
        results[ds_name] = run_one_dataset(
            ds_name, ds_path, model, tokenizer, label_ids, device
        )

    # Save
    out_path = os.path.join(RESULTS_DIR, "nli_direction_auc.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"{'Dataset':<10} {'Forward (q→d)':<15} {'Reverse (d→q)':<15}")
    for ds_name, r in results.items():
        print(f"{ds_name:<10} {r['forward_auc_q_as_premise']:<15.4f} {r['reverse_auc_d_as_premise']:<15.4f}")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
