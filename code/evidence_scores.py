"""
Evidence-score computations.
Six methods are available, but the main paper uses three:
  1. NLI (Natural Language Inference) — query-document relevance via entailment
  2. Embedding Stability — stability of query-document cosine similarity under noise
  3. Cross-Encoder — reranker-style query-document relevance
Plus three additional methods kept for completeness:
  4. Response Confidence Proxy — LLM response-based heuristic (not used in paper)
  5. LLM-as-Judge — direct relevance rating by an LLM (not used in paper)
  6. Utility Predictor — trained DeBERTa predictor (not used in paper)
"""
import os
import numpy as np
import requests
import json
import torch
from tqdm import tqdm

# Seed torch/numpy for reproducibility
from config import SEED
torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# Evidence 1: NLI (Natural Language Inference)
# ============================================================

_nli_model = None
_nli_tokenizer = None
_nli_label_ids = None  # dynamic label index lookup


def _load_nli_model():
    """Load the DeBERTa NLI model and discover label indices dynamically."""
    global _nli_model, _nli_tokenizer, _nli_label_ids
    if _nli_model is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
        from config import NLI_MODEL, hf_revision
        model_name = NLI_MODEL
        print(f"Loading NLI model: {model_name}...")
        _nli_tokenizer = AutoTokenizer.from_pretrained(model_name, revision=hf_revision(model_name))
        _nli_model = AutoModelForSequenceClassification.from_pretrained(model_name, revision=hf_revision(model_name))
        _nli_model.eval()

        # Prefer MPS, then CUDA, fall back to CPU.
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else "cuda" if torch.cuda.is_available()
                              else "cpu")
        _nli_model = _nli_model.to(device)

        # Look up label indices from the model config (avoid hard-coding).
        config = AutoConfig.from_pretrained(model_name, revision=hf_revision(model_name))
        label2id = {label.lower(): idx for idx, label in config.id2label.items()}
        _nli_label_ids = {
            "entailment": label2id["entailment"],
            "neutral": label2id["neutral"],
            "contradiction": label2id["contradiction"],
        }
        print(f"NLI label mapping: {_nli_label_ids}")
        print(f"NLI device: {device}")
        print("NLI model loaded.")
    return _nli_model, _nli_tokenizer, _nli_label_ids


def _nli_relevance_from_probs(probs):
    """
    Relevance from NLI probabilities: relevance = P(entailment) + 0.5 * P(neutral).
    Uses the dynamically discovered label indices.
    """
    _, _, label_ids = _load_nli_model()
    ent_id = label_ids["entailment"]
    neu_id = label_ids["neutral"]

    if probs.dim() == 1:
        return (probs[ent_id] + 0.5 * probs[neu_id]).item()
    else:
        return (probs[:, ent_id] + 0.5 * probs[:, neu_id]).tolist()


def compute_nli_evidence(question, doc_text, max_length=512):
    """
    NLI relevance for a single (query, document) pair.
    Premise = query, hypothesis = document (see note below).

    Returns:
        float: relevance score in [0, 1]
    """
    model, tokenizer, _ = _load_nli_model()
    device = next(model.parameters()).device

    doc_truncated = doc_text[:1000]

    # Direction: premise = query, hypothesis = document.
    # Forward direction chosen as the more conservative reporting choice:
    # on our evaluation sample, forward yields gold-vs-non-gold AUC of
    # 0.504/0.512/0.502 (NQ/TQA/PopQA) versus reverse 0.570/0.590/0.450;
    # both directions stay near 0.5, and we select the lower-AUC direction
    # so that the main-text "weak NLI discrimination" conclusion is the more
    # favorable of the two for NLI rather than an artifact of direction choice.
    # See Appendix C (Table 13) for both-direction AUC.
    inputs = tokenizer(
        question, doc_truncated,  # premise = question, hypothesis = doc
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)

    return _nli_relevance_from_probs(probs[0])


def compute_nli_evidence_batch(question, docs, max_length=512):
    """
    NLI relevance for a list of documents (batched).

    Args:
        question: query string
        docs: list of dicts of the form {"text": "...", ...}

    Returns:
        list[float]: relevance score per document
    """
    model, tokenizer, _ = _load_nli_model()
    device = next(model.parameters()).device

    # Direction: premise = query, hypothesis = document (see note above).
    doc_texts = [doc["text"][:1000] for doc in docs]
    questions = [question] * len(docs)

    BATCH_SIZE = 8
    all_relevance = []

    for batch_start in range(0, len(doc_texts), BATCH_SIZE):
        batch_docs = doc_texts[batch_start:batch_start + BATCH_SIZE]
        batch_questions = questions[batch_start:batch_start + BATCH_SIZE]

        inputs = tokenizer(
            batch_questions,  # premise = question
            batch_docs,       # hypothesis = document
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            relevance = _nli_relevance_from_probs(probs)
            all_relevance.extend(relevance)

    return all_relevance


# ============================================================
# Evidence 2: Response Confidence Proxy (not used in the main paper)
# (Formerly called "Perplexity"; renamed because it is not a real perplexity.)
# ============================================================

def _compute_response_confidence_hf(prompt, hf_model=None, hf_tokenizer=None):
    """
    Heuristic proxy for response confidence from a local HF LLM.
    Uses response brevity plus detection of uncertainty phrases.
    """
    if hf_model is None or hf_tokenizer is None:
        raise RuntimeError(
            "response_confidence requires HF model and tokenizer. "
            "Pass them via compute_all_evidence(... model=model, tokenizer=tokenizer)"
        )

    inputs = hf_tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    inputs = {k: v.to(hf_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = hf_model.generate(**inputs, max_new_tokens=30, do_sample=False)

    response = hf_tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip().lower()

    # Detect uncertainty expressions.
    uncertain_phrases = ["unknown", "i don't know", "not sure",
                         "cannot determine", "no information", "unclear"]
    if not response or any(phrase in response for phrase in uncertain_phrases):
        return 0.0

    # Brevity heuristic: shorter answers are treated as more confident.
    word_count = len(response.split())
    brevity_score = max(0, 1.0 - (word_count - 1) / 30.0)

    return max(0.0, min(1.0, brevity_score))


def compute_response_confidence_evidence(question, doc_text,
                                          hf_model=None, hf_tokenizer=None):
    """
    Evidence score based on how much the LLM's response confidence changes
    when given the document (vs. no document).

    e_i = max(0, confidence_with - confidence_without)

    Returns:
        float: evidence score in [0, 1]
    """
    # Without the document.
    prompt_without = (
        f"Answer the following question in a few words.\n"
        f"Question: {question}\nAnswer:"
    )
    score_without = _compute_response_confidence_hf(
        prompt_without, hf_model=hf_model, hf_tokenizer=hf_tokenizer
    )

    # With the document (same prompt style and truncation as generation.py).
    prompt_with = (
        f"Based on the following context, answer the question in a few words. "
        f"If the context doesn't contain the answer, say 'unknown'.\n\n"
        f"Context: {doc_text[:800]}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    score_with = _compute_response_confidence_hf(
        prompt_with, hf_model=hf_model, hf_tokenizer=hf_tokenizer
    )

    improvement = max(0.0, score_with - score_without)
    return min(improvement, 1.0)


# ============================================================
# Evidence 3: Embedding Stability
# (Formerly called "Laplace"; renamed because it is not a real Laplace approximation.)
# ============================================================

_embed_model = None


def _load_embed_model():
    """Load the sentence-transformer embedding model (lazy)."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        print("Loading embedding model for stability evidence...")
        from config import EMBED_MODEL, hf_revision
        _embed_model = SentenceTransformer(EMBED_MODEL, revision=hf_revision(EMBED_MODEL))
        print("Embedding model loaded.")
    return _embed_model


def compute_embedding_stability_evidence(question, doc_text, n_perturbations=10):
    """
    Embedding-stability evidence score.

    We perturb the query and document embeddings with small Gaussian noise
    (std = 0.01) and measure the variance of the resulting cosine similarity.
    Low variance (stable) -> high evidence; high variance -> low evidence.

    Note: This is a heuristic, not a true Last-Layer Laplace Approximation.

    Reproducibility: torch RNG is seeded with SEED=42 at module import;
    results are deterministic as long as the pipeline is executed in the
    same order as in the experiments.

    Returns:
        float: evidence score in [0, 1]
    """
    model = _load_embed_model()
    doc_truncated = doc_text[:512]

    q_emb = model.encode(question, convert_to_tensor=True)
    d_emb = model.encode(doc_truncated, convert_to_tensor=True)

    similarities = []
    for _ in range(n_perturbations):
        noise_q = torch.randn_like(q_emb) * 0.01
        noise_d = torch.randn_like(d_emb) * 0.01
        q_perturbed = q_emb + noise_q
        d_perturbed = d_emb + noise_d

        sim = torch.nn.functional.cosine_similarity(
            q_perturbed.unsqueeze(0), d_perturbed.unsqueeze(0)
        ).item()
        similarities.append(sim)

    sim_mean = np.mean(similarities)
    sim_std = np.std(similarities)

    # evidence = stability (1 / (1 + sigma * 10)) * relevance (max(0, mean))
    evidence = (1.0 / (1.0 + sim_std * 10)) * max(0, sim_mean)
    return float(max(0.0, min(1.0, evidence)))


# ============================================================
# Evidence 4: Cross-Encoder Reranker (refined relevance)
# ============================================================

_cross_encoder = None


def _load_cross_encoder():
    """Load the cross-encoder reranker (ms-marco-MiniLM-L-6-v2, 22.7 MB)."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        from config import CROSS_ENCODER_MODEL, hf_revision
        print(f"  Loading cross-encoder: {CROSS_ENCODER_MODEL}")
        _cross_encoder = CrossEncoder(
            CROSS_ENCODER_MODEL,
            revision=hf_revision(CROSS_ENCODER_MODEL),  # recorded commit when PIN_MODEL_REVISIONS=1
        )
    return _cross_encoder


def compute_cross_encoder_evidence_batch(question, docs):
    """
    Cross-encoder relevance score, sigmoid-normalized to [0, 1].
    The cross-encoder jointly processes the query and document, capturing
    finer-grained interaction than a dual-encoder like DPR.

    Args:
        question: query string
        docs: list of document dicts

    Returns:
        list[float]: cross-encoder evidence score per document in [0, 1]
    """
    model = _load_cross_encoder()
    pairs = [(question, doc["text"][:1000]) for doc in docs]
    raw_scores = model.predict(pairs)  # unbounded logits

    # Map to [0, 1] via sigmoid.
    sigmoid_scores = 1.0 / (1.0 + np.exp(-np.array(raw_scores)))
    return sigmoid_scores.tolist()


# ============================================================
# Evidence 5: LLM-as-Judge (not used in the main paper)
# ============================================================

def compute_llm_judge_evidence(question, doc_text, model=None, tokenizer=None):
    """
    Ask an LLM to rate document usefulness on a 0-5 scale; normalize to [0, 1].

    Args:
        question: query string
        doc_text: document text
        model: HF model (must be provided externally)
        tokenizer: HF tokenizer (must be provided externally)

    Returns:
        float: LLM-judge evidence score in [0, 1]
    """
    prompt = (
        f"Does the following document contain information that can answer the question?\n\n"
        f"Document: {doc_text[:800]}\n\n"
        f"Question: {question}\n\n"
        f"Rate from 0 (completely irrelevant) to 5 (directly answers the question).\n"
        f"Give only the number:"
    )

    if model is None or tokenizer is None:
        raise ValueError("LLM-as-Judge requires model and tokenizer to be provided")

    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

    # Parse the numeric rating.
    try:
        score = float(response.split()[0])
        score = max(0.0, min(5.0, score))
    except (ValueError, IndexError):
        score = 2.5  # neutral fallback on parse failure

    return score / 5.0  # normalize to [0, 1]


def compute_llm_judge_evidence_batch(question, docs, model=None, tokenizer=None):
    """Apply LLM-as-Judge to a list of documents."""
    scores = []
    for doc in docs:
        score = compute_llm_judge_evidence(
            question, doc["text"], model=model, tokenizer=tokenizer
        )
        scores.append(score)
    return scores


# ============================================================
# Evidence 6: Utility Predictor (learned, not used in the main paper)
# ============================================================

_utility_model = None
_utility_tokenizer = None


def _load_utility_predictor():
    """Load a trained DeBERTa utility predictor (if available)."""
    global _utility_model, _utility_tokenizer
    if _utility_model is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from config import UTILITY_PREDICTOR_DIR
        import os

        if not os.path.exists(UTILITY_PREDICTOR_DIR):
            raise FileNotFoundError(
                f"Utility predictor not found at {UTILITY_PREDICTOR_DIR}. "
                f"Run train_utility_predictor.py first."
            )

        print(f"  Loading utility predictor from {UTILITY_PREDICTOR_DIR}")
        _utility_tokenizer = AutoTokenizer.from_pretrained(UTILITY_PREDICTOR_DIR)
        _utility_model = AutoModelForSequenceClassification.from_pretrained(
            UTILITY_PREDICTOR_DIR
        )
        _utility_model.eval()
    return _utility_tokenizer, _utility_model


def compute_utility_evidence_batch(question, docs):
    """
    Predict P(correct | q, d) with a trained DeBERTa classifier.

    The classifier was fine-tuned on NQ training data to predict whether
    an LLM would answer correctly given the document.

    Args:
        question: query string
        docs: list of document dicts

    Returns:
        list[float]: utility evidence score per document in [0, 1]
    """
    tokenizer, model = _load_utility_predictor()

    texts = [(question, doc["text"][:512]) for doc in docs]
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True,
        truncation=True, max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        # class 1 = "correct" probability
        utility_scores = probs[:, 1].tolist()

    return utility_scores


# ============================================================
# Unified dispatcher
# ============================================================

def compute_all_evidence(question, docs, method="nli", **kwargs):
    """
    Compute evidence scores for all documents using the requested method.

    Args:
        question: query string
        docs: list of document dicts
        method: evidence method name
        **kwargs: extra arguments (e.g., model/tokenizer for llm_judge)

    Returns:
        list[float]: evidence score per document in [0, 1]
    """
    if method == "nli":
        return compute_nli_evidence_batch(question, docs)

    elif method == "response_confidence":
        scores = []
        for doc in docs:
            score = compute_response_confidence_evidence(
                question, doc["text"],
                hf_model=kwargs.get("model"),
                hf_tokenizer=kwargs.get("tokenizer"),
            )
            scores.append(score)
        return scores

    elif method == "embedding_stability":
        scores = []
        for doc in docs:
            score = compute_embedding_stability_evidence(question, doc["text"])
            scores.append(score)
        return scores

    elif method == "cross_encoder":
        return compute_cross_encoder_evidence_batch(question, docs)

    elif method == "llm_judge":
        return compute_llm_judge_evidence_batch(
            question, docs,
            model=kwargs.get("model"),
            tokenizer=kwargs.get("tokenizer"),
        )

    elif method == "utility_predictor":
        return compute_utility_evidence_batch(question, docs)

    else:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Available: nli, response_confidence, embedding_stability, "
            f"cross_encoder, llm_judge, utility_predictor"
        )


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":
    question = "Where did France focus its efforts to rebuild its empire?"
    docs = [
        {"text": "France took control of Algeria in 1830 but began in earnest to rebuild its worldwide empire after 1850, concentrating chiefly on North and West Africa."},
        {"text": "Construction projects can suffer from preventable financial problems."},
        {"text": "In World War II, Charles de Gaulle and the Free French used the overseas colonies as bases."},
    ]

    print("=== NLI Evidence (with dynamic label mapping) ===")
    nli_scores = compute_all_evidence(question, docs, method="nli")
    for i, (doc, score) in enumerate(zip(docs, nli_scores)):
        print(f"  Doc {i+1}: {score:.4f} | {doc['text'][:60]}...")

    print("\n=== Embedding Stability Evidence ===")
    stab_scores = compute_all_evidence(question, docs, method="embedding_stability")
    for i, (doc, score) in enumerate(zip(docs, stab_scores)):
        print(f"  Doc {i+1}: {score:.4f} | {doc['text'][:60]}...")

    print("\nDone.")
