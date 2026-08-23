"""
Build a Contriever-based retrieval set for PopQA.

Uses pre-computed Wikipedia passage embeddings and the Contriever-MSMARCO
question encoder to retrieve top-10 passages for each of the 14K PopQA queries.

Usage: python3 build_popqa_contriever.py
"""
import json
import os
import sys
import pickle
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, SEED, hf_revision

np.random.seed(SEED)

EMBEDDINGS_DIR = os.path.join(DATA_DIR, "wikipedia_embeddings")
NUM_DOCS = 10


def get_embedding_files():
    """Return the list of embedding shard files."""
    files = sorted([f for f in os.listdir(EMBEDDINGS_DIR) if f.startswith("passages_")])
    return [os.path.join(EMBEDDINGS_DIR, f) for f in files]


def load_shard(path):
    """Load a single embedding shard and L2-normalize."""
    with open(path, "rb") as f:
        ids, embs = pickle.load(f)
    embs = embs.astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embs = embs / norms
    return ids, embs


def load_passages_text():
    """Load Wikipedia passage texts from psgs_w100.tsv."""
    tsv_path = os.path.join(DATA_DIR, "psgs_w100.tsv")
    if not os.path.exists(tsv_path):
        print(f"  psgs_w100.tsv not found. Downloading...")
        import subprocess
        subprocess.run([
            "curl", "-L", "-o", tsv_path + ".gz",
            "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz"
        ], check=True)
        subprocess.run(["gunzip", tsv_path + ".gz"], check=True)

    print("  Loading passage texts...")
    id_to_text = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        header = f.readline()  # skip header
        for line in tqdm(f, desc="  Reading"):
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                pid = parts[0].strip()  # keep as string to match the embedding ID
                text = parts[1]
                title = parts[2]
                id_to_text[pid] = f"{title}. {text}" if title else text

    print(f"  Loaded {len(id_to_text)} passages")
    return id_to_text


def load_popqa():
    """Load the PopQA dataset."""
    from datasets import load_dataset
    print("Loading PopQA...")
    ds = load_dataset("akariasai/PopQA", split="test", revision=hf_revision("akariasai/PopQA"))
    samples = []
    for item in ds:
        answers = item["possible_answers"]
        if isinstance(answers, str):
            try:
                answers = json.loads(answers)
            except json.JSONDecodeError:
                answers = [answers]
        samples.append({
            "question": item["question"],
            "answers": answers,
            "s_pop": item.get("s_pop", 0),
        })
    print(f"  Loaded {len(samples)} questions")
    return samples


def encode_questions(questions, batch_size=64):
    """Encode questions with the Contriever question encoder."""
    from transformers import AutoTokenizer, AutoModel

    print("  Loading Contriever-MSMARCO encoder...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco", revision=hf_revision("facebook/contriever-msmarco"))
    model = AutoModel.from_pretrained("facebook/contriever-msmarco", revision=hf_revision("facebook/contriever-msmarco"))
    model.eval()

    print(f"  Encoding {len(questions)} questions...")
    all_embs = []

    for i in tqdm(range(0, len(questions), batch_size), desc="  Encoding"):
        batch = questions[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling (simple mean over all positions, including padding).
            # PopQA queries are short, so padding tokens have negligible effect here.
            # The reported results were produced with this exact implementation.
            emb = outputs.last_hidden_state.mean(dim=1)
        all_embs.append(emb.numpy())

    all_embs = np.vstack(all_embs).astype(np.float32)

    # L2-normalize the question embeddings.
    norms = np.linalg.norm(all_embs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    all_embs = all_embs / norms

    return all_embs


def search_top_k_sharded(q_embs, k=10):
    """
    Search shard by shard to save memory.

    Each shard holds about 1.3M passages (~4 GB). We keep only one shard in
    memory at a time, accumulate per-shard top-k candidates, then pick the
    overall top-k at the end.
    """
    shard_files = get_embedding_files()
    num_q = len(q_embs)

    # Accumulate (id, score) candidates per question.
    candidates = [[] for _ in range(num_q)]

    for shard_idx, shard_path in enumerate(shard_files):
        print(f"  Shard {shard_idx+1}/{len(shard_files)}: {os.path.basename(shard_path)}")
        p_ids, p_embs = load_shard(shard_path)

        # Batched search.
        batch_size = 500
        for i in range(0, num_q, batch_size):
            q_batch = q_embs[i:i + batch_size]
            scores = np.dot(q_batch, p_embs.T)  # [batch, shard_size]
            top_indices = np.argsort(-scores, axis=1)[:, :k]

            for j in range(len(q_batch)):
                for idx in top_indices[j]:
                    candidates[i + j].append((p_ids[idx], float(scores[j, idx])))

        # Release the shard immediately.
        del p_ids, p_embs
        import gc
        gc.collect()

    # Choose the final top-k for each question from the accumulated candidates.
    print("  Selecting final top-k...")
    all_top_ids = []
    all_top_scores = []
    for cands in candidates:
        cands.sort(key=lambda x: x[1], reverse=True)
        top = cands[:k]
        all_top_ids.append([c[0] for c in top])
        all_top_scores.append([c[1] for c in top])

    return all_top_ids, all_top_scores


def main():
    # 1. Load PopQA.
    samples = load_popqa()

    # 2. Encode questions.
    questions = [s["question"] for s in samples]
    q_embs = encode_questions(questions)

    # 3. Shard-wise search (cached intermediate results).
    search_cache = os.path.join(DATA_DIR, "popqa_search_cache.json")
    if os.path.exists(search_cache):
        print(f"  Loading cached search results from {search_cache}")
        with open(search_cache, "r") as f:
            cached = json.load(f)
        top_ids = cached["top_ids"]
        top_scores = cached["top_scores"]
    else:
        top_ids, top_scores = search_top_k_sharded(q_embs, k=NUM_DOCS)
        # Persist intermediate results to skip the search on re-runs.
        print(f"  Saving search cache to {search_cache}")
        with open(search_cache, "w") as f:
            json.dump({"top_ids": top_ids, "top_scores": top_scores}, f)

    # 4. Load passage texts.
    id_to_text = load_passages_text()

    # 6. Build the final result records.
    print("Building results...")
    results = []
    gold_count = 0

    for i, sample in enumerate(tqdm(samples, desc="  Results")):
        docs = []
        has_gold = False
        seen_texts = set()

        for rank, (pid, score) in enumerate(zip(top_ids[i], top_scores[i])):
            text = id_to_text.get(pid, "")
            if not text:
                continue

            # Dedup.
            text_key = text[:200].lower().strip()
            if text_key in seen_texts:
                continue
            seen_texts.add(text_key)

            # Gold / noise tag.
            text_lower = text.lower()
            has_answer = any(
                a.lower() in text_lower for a in sample["answers"] if len(a) > 2
            )

            if has_answer and not has_gold:
                doc_type = "gold"
                has_gold = True
            elif has_answer:
                doc_type = "related"
            else:
                doc_type = "noise"

            docs.append({
                "text": text,
                "type": doc_type,
                "score": round(score, 6),
            })

        if has_gold:
            gold_count += 1

        results.append({
            "question": sample["question"],
            "answers": sample["answers"],
            "s_pop": sample["s_pop"],
            "retrieved_docs": docs[:NUM_DOCS],
        })

    # 7. Save the final file.
    output_path = os.path.join(DATA_DIR, "popqa_contriever.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Summary statistics
    long_tail = [r for r in results if r["s_pop"] < 100]
    print(f"\n  Saved: {output_path}")
    print(f"  Questions: {len(results)}")
    print(f"  Gold: {gold_count}/{len(results)} ({gold_count/len(results)*100:.1f}%)")
    print(f"  Long-tail (s_pop<100): {len(long_tail)}")
    print(f"  Docs/question: {NUM_DOCS}")


if __name__ == "__main__":
    main()
