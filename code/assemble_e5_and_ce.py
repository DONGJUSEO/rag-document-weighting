#!/usr/bin/env python3
"""
Local assembly of E5/BGE retrieval + Cross-Encoder evidence (script 3 of 4).

Joins the pod's compact {dataset}_{model}_search.json (per-query [[pid, score], ...])
against the local psgs_w100.tsv to recover passage text, applies the SAME
gold-tagging + dedup rules as build_popqa_contriever.py / convert_dpr_cosine.py,
writes {dataset}_{model}.json (paper retrieval-set format), then computes
Cross-Encoder evidence into evidence_cache/{dataset}_{model}_cross_encoder.json.

The output files slot directly into compute_gold_subset_analysis.py and the
per-doc LLM runner (script 4) by pointing their DATASETS entries at
{dataset}_{model}.json + {dataset}_{model}_cross_encoder.json.

Everything runs locally: the 13.7 GB corpus is streamed exactly ONCE (only the
~few-hundred-thousand needed pids are kept in RAM), and CE is
cross-encoder/ms-marco-MiniLM-L-6-v2 on MPS/CPU. Deterministic (CE is forward-only,
torch seeded in evidence_scores at import).

Gold tagging (identical to the two builders): a doc is gold if any answer string
of length > 2 appears (case-insensitive substring) in its text; only the FIRST
such doc per query is "gold", later matches are "related", the rest "noise".
Dedup: skip a doc whose leading 200 chars (lowercased) already appeared.

Usage (after downloading *_search.json from the pod into data/):
  python assemble_e5_and_ce.py --model e5  --datasets nq triviaqa popqa
  python assemble_e5_and_ce.py --model bge --datasets nq triviaqa popqa
"""
import argparse
import json
import os
import sys

from tqdm import tqdm

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
sys.path.insert(0, os.path.join(BASE, "code"))
from evidence_scores import compute_cross_encoder_evidence_batch  # noqa: E402

CORPUS_TSV = os.path.join(DATA_DIR, "psgs_w100.tsv")
NUM_DOCS = 10


def collect_pids(model_key, datasets):
    """Union of every pid referenced by the target search files."""
    pids = set()
    for ds in datasets:
        path = os.path.join(DATA_DIR, f"{ds}_{model_key}_search.json")
        with open(path, "r", encoding="utf-8") as f:
            search = json.load(f)
        for cand in search:
            for pid, _score in cand:
                pids.add(pid)
    return pids


def load_texts(needed_pids):
    """One streaming pass over psgs_w100.tsv; keep only the needed pids.

    TSV columns: id \t text \t title (parts[0], [1], [2]) — same as the builders.
    """
    id_to_text = {}
    with open(CORPUS_TSV, "r", encoding="utf-8") as f:
        f.readline()  # header
        for line in tqdm(f, desc="  scanning psgs_w100"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                pid = parts[0].strip()
                if pid in needed_pids:
                    text, title = parts[1], parts[2]
                    id_to_text[pid] = f"{title}. {text}" if title else text
    return id_to_text


def assemble(ds, model_key, id_to_text):
    """Build the paper-format retrieval set for one (dataset, retriever)."""
    with open(os.path.join(DATA_DIR, f"{ds}_{model_key}_search.json"), "r", encoding="utf-8") as f:
        search = json.load(f)
    with open(os.path.join(DATA_DIR, f"e5_queries_{ds}.json"), "r", encoding="utf-8") as f:
        queries = json.load(f)
    assert len(search) == len(queries), \
        f"{ds}: search/queries length mismatch ({len(search)} vs {len(queries)})"

    results = []
    gold_count = 0
    short = 0  # queries left with < 10 docs after dedup/missing-text

    for q, cand in zip(queries, search):
        question, answers = q["question"], q["answers"]
        docs, has_gold, seen = [], False, set()

        for pid, score in cand:
            text = id_to_text.get(pid, "")
            if not text:
                continue
            key = text[:200].lower().strip()
            if key in seen:
                continue
            seen.add(key)

            text_lower = text.lower()
            has_answer = any(a.lower() in text_lower for a in answers if len(a) > 2)
            if has_answer and not has_gold:
                doc_type, has_gold = "gold", True
            elif has_answer:
                doc_type = "related"
            else:
                doc_type = "noise"

            docs.append({"text": text, "type": doc_type, "score": round(float(score), 6)})

        docs = docs[:NUM_DOCS]
        if has_gold:
            gold_count += 1
        if len(docs) < NUM_DOCS:
            short += 1
        results.append({"question": question, "answers": answers, "retrieved_docs": docs})

    out = os.path.join(DATA_DIR, f"{ds}_{model_key}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  {ds}: {len(results)} q  gold={gold_count} "
          f"({100 * gold_count / len(results):.1f}%)  <10docs={short}  -> {out}")
    return results


def compute_ce(ds, model_key, results):
    """Cross-Encoder evidence per query; same list[query][doc] shape as the paper."""
    os.makedirs(os.path.join(DATA_DIR, "evidence_cache"), exist_ok=True)
    ev = []
    for r in tqdm(results, desc=f"  CE {ds}"):
        docs = r["retrieved_docs"]
        ev.append(compute_cross_encoder_evidence_batch(r["question"], docs) if docs else [])
    out = os.path.join(DATA_DIR, "evidence_cache", f"{ds}_{model_key}_cross_encoder.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(ev, f)
    print(f"  {ds}: CE evidence -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="e5", choices=["e5", "bge"])
    ap.add_argument("--datasets", nargs="+", default=["nq", "triviaqa", "popqa"])
    a = ap.parse_args()

    print("Collecting unique pids from search files...")
    pids = collect_pids(a.model, a.datasets)
    print(f"  {len(pids):,} unique pids")

    print("Streaming psgs_w100.tsv (one pass, ~13.7 GB)...")
    id_to_text = load_texts(pids)
    print(f"  recovered {len(id_to_text):,} / {len(pids):,} passage texts")

    for ds in a.datasets:
        results = assemble(ds, a.model, id_to_text)
        compute_ce(ds, a.model, results)
    print("Assembly + CE done. Point script 4 / gold-subset at {ds}_%s.json." % a.model)


if __name__ == "__main__":
    main()
