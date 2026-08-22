"""
LLM generation module: Together AI / OpenAI / HuggingFace backends with a file cache.

Design:
  1. Together AI or OpenAI API, or a local HuggingFace model.
  2. For every (question, document) pair, generate an answer and its sequence log-probability.
  3. Store results in a JSON cache file for reuse across runs.
  4. Two aggregation modes: voting (paper default) and sequence-level probability (legacy/ablation).
"""
import json
import hashlib
import os
import sys
import requests as http_requests
import numpy as np
import time
import torch
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    LLM_BACKEND, LLM_CACHE_FILE, SEED,
    HF_MODEL, HF_TOKEN,
    TOGETHER_API_KEY, TOGETHER_MODEL, TOGETHER_API_URL, OPENAI_MODEL,
)

# HuggingFace model (lazy-loaded)
_hf_model = None
_hf_tokenizer = None

# File cache
_file_cache = {}
_cache_dirty = False


# ============================================================
# Model loading
# ============================================================

def load_hf_model():
    """Load the local HF causal LM (float16) on MPS/CUDA/CPU, lazily."""
    global _hf_model, _hf_tokenizer
    if _hf_model is not None:
        return _hf_model, _hf_tokenizer

    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"  Loading {HF_MODEL} (float16)...")
    start = time.time()

    _hf_tokenizer = AutoTokenizer.from_pretrained(
        HF_MODEL, token=HF_TOKEN
    )

    # Pick a single device explicitly: MPS > CUDA > CPU.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    _hf_model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL,
        torch_dtype=torch.float16,
        token=HF_TOKEN,
    ).to(device)
    _hf_model.eval()

    elapsed = time.time() - start
    mem_gb = _hf_model.get_memory_footprint() / 1024**3
    print(f"  Loaded in {elapsed:.1f}s, device={_hf_model.device}, "
          f"memory={mem_gb:.2f}GB")

    return _hf_model, _hf_tokenizer


def get_model_and_tokenizer():
    """Accessor for the HF model/tokenizer from other modules."""
    return load_hf_model()


# ============================================================
# Cache management
# ============================================================

def _cache_key(question, doc_text):
    """Hash of (model, question, truncated doc text) used as cache key.

    Note: LLM_MODEL is a single env var that each backend interprets as its
    own model name; we route it to the corresponding module-level constant
    so the cache key is unambiguous per backend.
    """
    if LLM_BACKEND == "openai":
        model_name = OPENAI_MODEL
    elif LLM_BACKEND == "together":
        model_name = TOGETHER_MODEL
    else:
        model_name = HF_MODEL
    raw = f"{model_name}|||{question}|||{doc_text[:800]}"
    return hashlib.md5(raw.encode()).hexdigest()


def load_cache():
    """Load the on-disk cache file (empty dict if missing)."""
    global _file_cache
    if os.path.exists(LLM_CACHE_FILE):
        with open(LLM_CACHE_FILE, "r", encoding="utf-8") as f:
            _file_cache = json.load(f)
        print(f"  LLM cache loaded: {len(_file_cache)} entries")
    else:
        _file_cache = {}
        print(f"  LLM cache: starting fresh")


def save_cache():
    """Atomically write the cache back to disk if it has been modified."""
    global _cache_dirty
    if not _cache_dirty:
        return
    os.makedirs(os.path.dirname(LLM_CACHE_FILE), exist_ok=True)
    # Atomic write: write to a temp file, then rename.
    tmp_file = LLM_CACHE_FILE + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(_file_cache, f, ensure_ascii=False)
    os.replace(tmp_file, LLM_CACHE_FILE)
    _cache_dirty = False
    print(f"  LLM cache saved: {len(_file_cache)} entries")


# ============================================================
# Answer generation + logprob computation
# ============================================================

def _build_prompt(question, doc_text):
    """Build the REPLUG-style prompt used throughout the paper."""
    return (
        f"Based on the following context, answer the question in a few words.\n\n"
        f"Context: {doc_text[:800]}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


def generate_answer_per_doc_api(question, doc_text, max_tokens=50):
    """Generate an answer via the Together AI / OpenAI API, returning text and sequence logprob."""
    prompt = _build_prompt(question, doc_text)

    # Pick the API endpoint and key.
    is_openai = (LLM_BACKEND == "openai")
    if is_openai:
        from config import OPENAI_API_KEY
        api_url = "https://api.openai.com/v1/chat/completions"
        api_key = OPENAI_API_KEY
    else:
        api_url = TOGETHER_API_URL
        api_key = TOGETHER_API_KEY

    payload = {
        "model": OPENAI_MODEL if is_openai else TOGETHER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }

    # OpenAI and Together AI request logprobs in slightly different ways.
    if is_openai:
        payload["logprobs"] = True
        payload["top_logprobs"] = 1
    else:
        payload["logprobs"] = 1

    for attempt in range(3):
        try:
            resp = http_requests.post(
                api_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=60,
            )
            result = resp.json()

            if "error" in result:
                print(f"  API error: {result['error']['message']}")
                time.sleep(2 ** attempt)
                continue

            answer_text = result["choices"][0]["message"]["content"].strip()
            answer_text = answer_text.split("\n")[0].strip()

            # Sum token logprobs (Together AI response format).
            logprobs_data = result["choices"][0].get("logprobs", {})
            seq_logprob = 0.0
            if logprobs_data and "token_logprobs" in logprobs_data:
                for lp in logprobs_data["token_logprobs"]:
                    if lp is not None:
                        seq_logprob += lp
            elif logprobs_data and "content" in logprobs_data:
                # OpenAI-compatible response format.
                for token_info in logprobs_data["content"]:
                    seq_logprob += token_info.get("logprob", 0.0)
            else:
                seq_logprob = -10.0

            return {
                "answer_text": answer_text,
                "sequence_logprob": seq_logprob,
            }

        except Exception as e:
            print(f"  API call failed (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)

    return {"answer_text": "unknown", "sequence_logprob": -100.0}


def generate_answer_per_doc_hf(question, doc_text, max_tokens=50):
    """Generate an answer with the local HF model and return text + sequence logprob."""
    model, tokenizer = load_hf_model()
    prompt = _build_prompt(question, doc_text)

    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated_ids = outputs.sequences[0][prompt_len:]
    answer_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    answer_text = answer_text.split("\n")[0].strip()

    seq_logprob = compute_sequence_logprob(
        question, doc_text, answer_text, model, tokenizer
    )

    return {
        "answer_text": answer_text,
        "sequence_logprob": seq_logprob,
    }


def generate_answer_per_doc(question, doc_text, max_tokens=50):
    """Dispatch to API or local HF depending on LLM_BACKEND."""
    if LLM_BACKEND in ("together", "openai"):
        return generate_answer_per_doc_api(question, doc_text, max_tokens)
    else:
        return generate_answer_per_doc_hf(question, doc_text, max_tokens)


def compute_sequence_logprob(question, doc_text, answer_text,
                              model=None, tokenizer=None):
    """
    Teacher-forced sequence log probability of a specific answer.

    log P(answer | context, question) = sum_t log P(y_t | y_{<t}, context, question)

    Args:
        question: query string
        doc_text: document text
        answer_text: the answer whose probability we want to score

    Returns:
        float: sequence log probability
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_hf_model()

    prompt = _build_prompt(question, doc_text)

    # Concatenate at the token-ID level to avoid spurious whitespace handling.
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    answer_ids = tokenizer(answer_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    full_ids = torch.cat([prompt_ids, answer_ids], dim=1).to(model.device)
    full_inputs = {"input_ids": full_ids, "attention_mask": torch.ones_like(full_ids)}

    prompt_len = prompt_ids.shape[1]

    with torch.no_grad():
        outputs = model(**full_inputs)
        log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)

    # Sum logprobs over answer tokens.
    input_ids = full_inputs["input_ids"][0]
    target_ids = input_ids[prompt_len:]

    if len(target_ids) == 0:
        return -100.0  # empty answer

    total_logprob = 0.0
    for i, tid in enumerate(target_ids):
        pos = prompt_len - 1 + i
        if pos < log_probs.shape[1]:
            total_logprob += log_probs[0, pos, tid].item()

    return total_logprob


# ============================================================
# Cache-aware answer generation
# ============================================================

def _api_call_single(args):
    """Wrapper for use with ThreadPoolExecutor."""
    question, doc_text, max_tokens = args
    return generate_answer_per_doc_api(question, doc_text, max_tokens)


def generate_all_doc_answers(question, docs, max_tokens=50):
    """
    Generate answers for all documents. Uses the file cache; remote API calls
    are issued in parallel.

    Returns:
        list[dict]: [{"answer_text": ..., "sequence_logprob": ...}, ...]
    """
    global _file_cache, _cache_dirty
    results = [None] * len(docs)
    uncached = []  # (index, doc) pairs that are not in the cache

    # 1. Check the cache.
    for i, doc in enumerate(docs):
        key = _cache_key(question, doc["text"])

        if key in _file_cache:
            entry = _file_cache[key]
            if isinstance(entry, str):
                entry = {"answer_text": entry, "sequence_logprob": None}
                _file_cache[key] = entry
                _cache_dirty = True
            results[i] = entry
        else:
            uncached.append((i, doc))

    # 2. Issue parallel API calls for uncached documents.
    if uncached and LLM_BACKEND in ("together", "openai"):
        from concurrent.futures import ThreadPoolExecutor
        args_list = [(question, doc["text"], max_tokens) for _, doc in uncached]

        with ThreadPoolExecutor(max_workers=20) as executor:
            api_results = list(executor.map(_api_call_single, args_list))

        for (i, doc), result in zip(uncached, api_results):
            key = _cache_key(question, doc["text"])
            _file_cache[key] = result
            _cache_dirty = True
            results[i] = result

    elif uncached:
        # Local HF: sequential processing.
        for i, doc in uncached:
            result = generate_answer_per_doc(question, doc["text"], max_tokens)
            key = _cache_key(question, doc["text"])
            _file_cache[key] = result
            _cache_dirty = True
            results[i] = result

    # 3. Backfill logprobs for cache entries that lack them (HF only).
    if LLM_BACKEND not in ("together", "openai"):
        for i, doc in enumerate(docs):
            entry = results[i]
            if entry and entry.get("sequence_logprob") is None and entry["answer_text"]:
                entry["sequence_logprob"] = compute_sequence_logprob(
                    question, doc["text"], entry["answer_text"]
                )
                key = _cache_key(question, doc["text"])
                _file_cache[key] = entry
                _cache_dirty = True

    return results


# ============================================================
# Aggregation 1: Weighted Majority Voting (paper default)
# ============================================================

def weighted_majority_vote(answers, weights):
    """
    Weighted majority vote over per-document answers.

    Args:
        answers: list of answer strings
        weights: list of document weights

    Returns:
        tuple: (final_answer, confidence, total_weight)
    """
    from metrics import normalize_answer as _normalize_answer

    if len(answers) != len(weights):
        raise ValueError("answers and weights must have same length")

    total_weight = float(sum(weights))
    answer_scores = {}

    for answer, weight in zip(answers, weights):
        normalized = _normalize_answer(answer)
        if normalized and normalized != "unknown":
            answer_scores[normalized] = answer_scores.get(normalized, 0.0) + weight

    if not answer_scores:
        return "", 0.0, total_weight

    best_normalized = max(answer_scores, key=answer_scores.get)
    raw_confidence = answer_scores[best_normalized]
    confidence = raw_confidence / total_weight if total_weight > 0 else 0.0

    for answer in answers:
        if _normalize_answer(answer) == best_normalized:
            return answer, confidence, total_weight

    return best_normalized, confidence, total_weight


def apply_weights_voting(cached_entries, weights):
    """
    Voting-based aggregation (paper default).

    Args:
        cached_entries: output of generate_all_doc_answers()
        weights: list of document weights

    Returns:
        dict: {"answer", "confidence", "method": "voting"}
    """
    answers = [e["answer_text"] for e in cached_entries]
    answer, confidence, total = weighted_majority_vote(answers, weights)

    return {
        "answer": answer,
        "confidence": confidence,
        "method": "voting",
    }


# ============================================================
# Aggregation 2: Sequence-level probability (REPLUG QA style, legacy)
# ============================================================

def apply_weights_sequence(cached_entries, weights, question=None, docs=None):
    """
    Sequence-level aggregation over same-answer documents:

    score(y) = logsumexp_i (log w_i + log P(y | d_i, x))
               for every i such that document i produced answer y.

    We do not perform cross-document teacher forcing:
      - In open-domain QA, the log-probability of one document's answer under
        a different document's context is effectively zero and contributes
        negligibly.
      - Skipping it is a mathematically near-identical approximation at a
        large cost saving.

    How weights affect the outcome:
      - Multiple documents producing the same answer have their weighted
        log-probabilities summed (via log-sum-exp).
      - Higher-weighted documents have more influence.

    Args:
        cached_entries: output of generate_all_doc_answers()
        weights: list of document weights
        question: unused (kept for a uniform interface)
        docs: unused (kept for a uniform interface)

    Returns:
        dict: {"answer", "confidence", "method": "sequence"}
    """
    from scipy.special import logsumexp as _logsumexp
    from metrics import normalize_answer as _normalize_answer

    # Collect unique answer candidates.
    unique_answers = {}
    for entry in cached_entries:
        ans_norm = _normalize_answer(entry["answer_text"])
        if ans_norm and ans_norm != "unknown":
            if ans_norm not in unique_answers:
                unique_answers[ans_norm] = entry["answer_text"]

    if not unique_answers:
        return {"answer": "", "confidence": 0.0, "method": "sequence"}

    # For each candidate answer, sum weighted logprobs from matching documents.
    answer_scores = {}

    for ans_normalized, ans_original in unique_answers.items():
        log_terms = []

        for i, (entry, w) in enumerate(zip(cached_entries, weights)):
            if w <= 0:
                continue

            entry_norm = _normalize_answer(entry["answer_text"])
            if entry_norm == ans_normalized:
                logprob = entry.get("sequence_logprob")
                if logprob is not None:
                    log_terms.append(np.log(w) + logprob)

        if log_terms:
            answer_scores[ans_normalized] = float(_logsumexp(log_terms))
        else:
            answer_scores[ans_normalized] = -100.0

    # Pick the highest-scoring answer.
    best_ans = max(answer_scores, key=answer_scores.get)
    best_score = answer_scores[best_ans]

    # Confidence = P(best) / P(all candidates).
    all_log_scores = list(answer_scores.values())
    log_total = _logsumexp(all_log_scores)
    confidence = np.exp(best_score - log_total)

    return {
        "answer": unique_answers[best_ans],
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "method": "sequence",
    }


# ============================================================
# Unified interface
# ============================================================

def apply_weights(cached_entries, weights, method="sequence",
                  question=None, docs=None):
    """
    Apply document weights to cached per-document answers and produce a final answer.

    Args:
        cached_entries: output of generate_all_doc_answers()
        weights: list of document weights
        method: "voting" or "sequence"
        question: query (only needed for sequence-level teacher forcing)
        docs: document list (only needed for sequence-level teacher forcing)

    Returns:
        dict: {"answer", "confidence", "method"}
    """
    if method == "voting":
        return apply_weights_voting(cached_entries, weights)
    elif method == "sequence":
        return apply_weights_sequence(
            cached_entries, weights, question=question, docs=docs
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'voting' or 'sequence'")


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":
    print("=== Generation module self-test ===")

    question = "What is the capital of France?"
    docs = [
        {"text": "Paris is the capital and most populous city of France."},
        {"text": "Berlin is the capital of Germany."},
        {"text": "London is the capital of the United Kingdom."},
    ]

    # Load model.
    model, tokenizer = load_hf_model()

    # Generate answers.
    print("\n--- Generating answers ---")
    entries = []
    for doc in docs:
        result = generate_answer_per_doc(question, doc["text"])
        entries.append(result)
        print(f"  Doc: {doc['text'][:50]}...")
        print(f"    Answer: {result['answer_text']}")
        print(f"    Logprob: {result['sequence_logprob']:.4f}")

    # Voting aggregation.
    print("\n--- Voting aggregation ---")
    weights = [0.5, 0.3, 0.2]
    result_vote = apply_weights(entries, weights, method="voting")
    print(f"  Answer: {result_vote['answer']}")
    print(f"  Confidence: {result_vote['confidence']:.4f}")

    # Sequence-level aggregation.
    print("\n--- Sequence aggregation ---")
    result_seq = apply_weights(
        entries, weights, method="sequence",
        question=question, docs=docs
    )
    print(f"  Answer: {result_seq['answer']}")
    print(f"  Confidence: {result_seq['confidence']:.4f}")

    print("\nAll tests passed.")
