"""Appendix M/N recomputation: gold-vs-non-gold AUC on a 500-query sample per dataset
for (a) the alternative NLI model roberta-large-mnli (Table 18) and (b) embedding
stability at 512- vs 1,000-character truncation (Table 19).

Sample and gold definition are shared with compute_nli_reverse.py (seed 42,
np.random.RandomState(42).choice; gold = any lower-cased gold answer string is a
substring of the lower-cased document), so every AUC in Tables 4, 18, 19, and 20
that is computed on the 500-query sample uses one protocol. The deberta-v3-base
forward AUC is also recomputed here as a consistency check against
results/nli_direction_auc.json.

Output: results/app_m_auc.json
"""
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SEED, hf_revision  # noqa: E402
from compute_nli_reverse import (  # noqa: E402  (shared sample + gold definition)
    DATASETS, RESULTS_DIR, SAMPLE_SIZE, BATCH_SIZE, has_gold_answer,
)

NLI_MODELS = ["cross-encoder/nli-deberta-v3-base", "roberta-large-mnli"]
EMBED_MODEL = "all-MiniLM-L6-v2"
ES_SIGMA = 0.01
ES_N_NOISE = 10
ES_CONFIGS = {"trunc_512": (512, None), "trunc_1000": (1000, 512)}  # chars, max_seq_length


def sample_indices(n_total):
    rng = np.random.RandomState(42)
    return sorted(rng.choice(n_total, size=min(SAMPLE_SIZE, n_total), replace=False))


def load_sample(dataset_path):
    with open(dataset_path) as f:
        data = json.load(f)
    pairs, gold = [], []
    for idx in sample_indices(len(data)):
        s = data[idx]
        for d in s["retrieved_docs"]:
            pairs.append((s["question"], d.get("text", "")))
            gold.append(has_gold_answer(d.get("text", ""), s["answers"]))
    return pairs, gold


def nli_forward_scores(model_name, pairs, device):
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, revision=hf_revision(model_name))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, revision=hf_revision(model_name)).to(device).eval()
    cfg = AutoConfig.from_pretrained(model_name, revision=hf_revision(model_name))
    label2id = {label.lower(): idx for idx, label in cfg.id2label.items()}
    ent, neu = label2id["entailment"], label2id["neutral"]
    scores = []
    for i in tqdm(range(0, len(pairs), BATCH_SIZE), desc=model_name):
        batch = pairs[i:i + BATCH_SIZE]
        premises = [q for q, _ in batch]             # forward: query as premise
        hypotheses = [d[:1000] for _, d in batch]    # document (1,000 chars) as hypothesis
        enc = tok(premises, hypotheses, return_tensors="pt", truncation=True,
                  max_length=512, padding=True).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1)
        scores.extend((probs[:, ent] + 0.5 * probs[:, neu]).tolist())
    del model
    return scores


def es_scores(pairs, trunc_chars, max_seq_length):
    """Embedding-stability evidence (evidence_scores.compute_embedding_stability_evidence
    protocol: Gaussian noise sigma=0.01, 10 perturbations, evidence =
    1/(1+10*std) * max(0, mean cosine)), with the document truncated to
    `trunc_chars` and the encoder's max_seq_length optionally raised."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL, revision=hf_revision(EMBED_MODEL))
    if max_seq_length is not None:
        model.max_seq_length = max_seq_length
    torch.manual_seed(SEED)
    scores = []
    for q, d in tqdm(pairs, desc=f"ES@{trunc_chars}"):
        q_emb = model.encode(q, convert_to_tensor=True)
        d_emb = model.encode(d[:trunc_chars], convert_to_tensor=True)
        sims = []
        for _ in range(ES_N_NOISE):
            qp = q_emb + torch.randn_like(q_emb) * ES_SIGMA
            dp = d_emb + torch.randn_like(d_emb) * ES_SIGMA
            sims.append(torch.nn.functional.cosine_similarity(qp.unsqueeze(0), dp.unsqueeze(0)).item())
        ev = (1.0 / (1.0 + float(np.std(sims)) * 10)) * max(0.0, float(np.mean(sims)))
        scores.append(max(0.0, min(1.0, ev)))
    return scores


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    results = {"sample_size_per_dataset": SAMPLE_SIZE, "seed": 42,
               "gold_definition": "compute_nli_reverse.has_gold_answer (lower-cased substring)",
               "nli_score": "P(entailment) + 0.5*P(neutral), forward (query as premise), doc[:1000]",
               "es_protocol": f"sigma={ES_SIGMA}, n_noise={ES_N_NOISE}, torch seed {SEED} per config",
               "datasets": {}}
    for ds, path in DATASETS.items():
        t0 = time.time()
        pairs, gold = load_sample(path)
        r = {"n_pairs": len(gold), "n_gold": int(sum(gold)), "gold_fraction": float(np.mean(gold)), "nli": {}, "es": {}}
        for m in NLI_MODELS:
            r["nli"][m] = {"auc_forward": float(roc_auc_score(gold, nli_forward_scores(m, pairs, device)))}
        for name, (chars, msl) in ES_CONFIGS.items():
            r["es"][name] = {"trunc_chars": chars, "max_seq_length": msl,
                             "auc": float(roc_auc_score(gold, es_scores(pairs, chars, msl)))}
        r["elapsed_sec"] = time.time() - t0
        results["datasets"][ds] = r
        print(f"{ds}: n={r['n_pairs']} gold={r['n_gold']} ({r['gold_fraction']:.3f}) "
              f"deberta={r['nli'][NLI_MODELS[0]]['auc_forward']:.4f} roberta={r['nli'][NLI_MODELS[1]]['auc_forward']:.4f} "
              f"ES@512={r['es']['trunc_512']['auc']:.4f} ES@1000={r['es']['trunc_1000']['auc']:.4f} ({r['elapsed_sec']:.0f}s)", flush=True)
    out = os.path.join(RESULTS_DIR, "app_m_auc.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
