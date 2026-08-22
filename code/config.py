"""
Main-experiment configuration for the unified Dirichlet-style RAG framework.
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(BASE_DIR, "results"))

# Datasets
DATASETS = {
    "nq": os.path.join(DATA_DIR, "nq_cosine.json"),
    "triviaqa": os.path.join(DATA_DIR, "triviaqa_cosine.json"),
    "popqa": os.path.join(DATA_DIR, "popqa_contriever.json"),
}

# LLM settings
LLM_BACKEND = os.environ.get("LLM_BACKEND", "together")  # "together", "openai", or "hf"

# Together AI API
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "")
TOGETHER_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct-Turbo")
TOGETHER_API_URL = os.environ.get("LLM_API_URL", "https://api.together.xyz/v1/chat/completions")

# OpenAI API (e.g., GPT-4.1-mini)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
# Default OpenAI model falls back to the LLM_MODEL env var (preserves existing behavior)
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", os.environ.get("LLM_MODEL", "gpt-4.1-mini"))

# HuggingFace LLM (fallback)
HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# DPR models (for cosine-similarity computation)
DPR_QUESTION_ENCODER = "facebook/dpr-question_encoder-multiset-base"
DPR_CTX_ENCODER = "facebook/dpr-ctx_encoder-multiset-base"

# Evidence-score models
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
EMBED_MODEL = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Experiment hyperparameters
NUM_DOCS = 10
BETAS = [0.5, 1.0, 2.0, 4.0]
LAMBDAS = [0, 0.1, 0.5, 1.0, 3.0, 10.0, 30.0]
NUM_DOCS_OPTIONS = [3, 5, 10, 20]  # used by the k-sweep
SEED = 42

# Evidence methods (the main paper uses cross_encoder, embedding_stability, nli)
EVIDENCE_METHODS = [
    "nli",
    "embedding_stability",
    "response_confidence",
    "cross_encoder",
    "llm_judge",
    "utility_predictor",
]

# LLM cache file (one per model, keyed by model short name)
_model_short = TOGETHER_MODEL.split("/")[-1].lower().replace("-", "_").replace(".", "_")
LLM_CACHE_FILE = os.path.join(DATA_DIR, os.environ.get(
    "LLM_CACHE_FILE",
    f"llm_cache_{_model_short}.json"
))

# Utility predictor (optional, not used in the main paper)
UTILITY_PREDICTOR_DIR = os.path.join(DATA_DIR, "utility_predictor")
