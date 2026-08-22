"""
Convert pre-computed DPR retrieval results to our experiment format with
cosine-similarity scores.

ReConsider JSONL -> {question, answers, retrieved_docs:[{text, type, score}]}.

Usage: python3 convert_dpr_cosine.py
"""
import json
import os
import sys
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, SEED

np.random.seed(SEED)

NUM_DOCS = 10

# DPR models (lazy-loaded)
_q_model = None
_q_tokenizer = None
_c_model = None
_c_tokenizer = None


def load_dpr_models():
    """Load the DPR question and context encoders."""
    global _q_model, _q_tokenizer, _c_model, _c_tokenizer
    if _q_model is not None:
        return

    from transformers import (
        DPRQuestionEncoder, DPRQuestionEncoderTokenizer,
        DPRContextEncoder, DPRContextEncoderTokenizer,
    )

    print("  Loading DPR question encoder...")
    _q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
        "facebook/dpr-question_encoder-multiset-base"
    )
    _q_model = DPRQuestionEncoder.from_pretrained(
        "facebook/dpr-question_encoder-multiset-base"
    )
    _q_model.eval()

    print("  Loading DPR context encoder...")
    _c_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
        "facebook/dpr-ctx_encoder-multiset-base"
    )
    _c_model = DPRContextEncoder.from_pretrained(
        "facebook/dpr-ctx_encoder-multiset-base"
    )
    _c_model.eval()
    print("  DPR models loaded.")


def encode_question(question):
    """Encode a question with the DPR question encoder and L2-normalize."""
    inputs = _q_tokenizer(question, return_tensors="pt", max_length=256, truncation=True)
    with torch.no_grad():
        emb = _q_model(**inputs).pooler_output  # [1, 768]
    return torch.nn.functional.normalize(emb, p=2, dim=1)


def encode_passages(passages, batch_size=32):
    """Encode a list of passages with the DPR context encoder and L2-normalize."""
    all_embs = []
    for i in range(0, len(passages), batch_size):
        batch = passages[i:i + batch_size]
        inputs = _c_tokenizer(
            batch, return_tensors="pt", max_length=512,
            truncation=True, padding=True
        )
        with torch.no_grad():
            emb = _c_model(**inputs).pooler_output  # [batch, 768]
        all_embs.append(emb)
    embs = torch.cat(all_embs, dim=0)
    return torch.nn.functional.normalize(embs, p=2, dim=1)


def parse_passage(passage_entry):
    """ReConsider passage [title, tokens, answer_spans] → text"""
    title = passage_entry[0]
    tokens = passage_entry[1]
    text = " ".join(tokens)
    has_answer = len(passage_entry[2]) > 0 if len(passage_entry) > 2 else False
    return title, text, has_answer


def convert_file(input_path, output_path, dataset_name):
    """Convert one ReConsider JSONL file to our cosine-similarity format."""
    print(f"\nConverting {dataset_name}...")
    load_dpr_models()

    # Read the JSONL input.
    raw_data = []
    with open(input_path, "r") as f:
        for line in f:
            raw_data.append(json.loads(line.strip()))
    print(f"  Loaded {len(raw_data)} questions")

    results = []
    gold_count = 0
    all_cosine_scores = []

    for item in tqdm(raw_data, desc=f"  {dataset_name}"):
        question = item["question"]
        answers = item["answers"]
        all_passages = item["positives"]

        if len(all_passages) == 0:
            continue

        # Parse top-N passages with simple deduplication.
        parsed = []
        seen_texts = set()
        for p in all_passages:
            if len(parsed) >= NUM_DOCS:
                break
            title, text, has_answer = parse_passage(p)
            full_text = f"{title}. {text}" if title else text
            # Dedup: skip if the leading 200 chars already appeared.
            text_key = full_text[:200].lower().strip()
            if text_key in seen_texts:
                continue
            seen_texts.add(text_key)
            parsed.append((full_text, has_answer))

        if not parsed:
            continue

        # Compute DPR embeddings.
        q_emb = encode_question(question)  # [1, 768]
        passage_texts = [p[0] for p in parsed]
        p_embs = encode_passages(passage_texts)  # [k, 768]

        # Cosine similarity = dot product after L2 normalization.
        cosine_sims = torch.mm(q_emb, p_embs.t()).squeeze(0)  # [k]
        cosine_sims = cosine_sims.tolist()

        # Gold / related / noise tagging (stored as doc["type"]).
        docs = []
        has_gold = False
        for i, (full_text, has_answer_flag) in enumerate(parsed):
            text_lower = full_text.lower()
            actually_has_answer = has_answer_flag or any(
                a.lower() in text_lower for a in answers if len(a) > 2
            )

            if actually_has_answer and not has_gold:
                doc_type = "gold"
                has_gold = True
            elif actually_has_answer:
                doc_type = "related"
            else:
                doc_type = "noise"

            docs.append({
                "text": full_text,
                "type": doc_type,
                "score": round(cosine_sims[i], 6),
            })
            all_cosine_scores.append(cosine_sims[i])

        if has_gold:
            gold_count += 1

        results.append({
            "question": question,
            "answers": answers,
            "retrieved_docs": docs,
        })

    # Save the converted file.
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Summary statistics
    scores = np.array(all_cosine_scores)
    print(f"\n  Saved: {output_path}")
    print(f"  Questions: {len(results)}")
    print(f"  Gold doc: {gold_count}/{len(results)} ({gold_count / len(results) * 100:.1f}%)")
    print(f"\n  Score distribution:")
    print(f"    Mean:   {scores.mean():.4f}")
    print(f"    Std:    {scores.std():.4f}")
    print(f"    Min:    {scores.min():.4f}")
    print(f"    Max:    {scores.max():.4f}")
    print(f"    Median: {np.median(scores):.4f}")
    print(f"\n  exp(β * max_score) check:")
    for beta in [0.5, 1.0, 2.0, 4.0]:
        val = np.exp(beta * scores.max())
        print(f"    β={beta}: exp(β*max)={val:.2f}")

    return results


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # NQ
    nq_raw = os.path.join(DATA_DIR, "nq_dpr_dev.json")
    if os.path.exists(nq_raw):
        convert_file(
            nq_raw,
            os.path.join(DATA_DIR, "nq_cosine.json"),
            "NQ",
        )
    else:
        print(f"NQ raw file not found: {nq_raw}")

    # TriviaQA
    tqa_raw = os.path.join(DATA_DIR, "tqa_dpr_dev.json")
    if os.path.exists(tqa_raw):
        convert_file(
            tqa_raw,
            os.path.join(DATA_DIR, "triviaqa_cosine.json"),
            "TriviaQA",
        )
    else:
        print(f"TriviaQA raw file not found: {tqa_raw}")

    print("\nDone!")


if __name__ == "__main__":
    main()
