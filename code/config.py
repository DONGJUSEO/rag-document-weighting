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

# Hugging Face revisions (commit hashes) resolved when the paper's runs were made
# (README, "Pinned inputs and model revisions"). They are passed to the
# from_pretrained() / load_dataset() calls only when PIN_MODEL_REVISIONS=1;
# by default the Hub head is loaded.
MODEL_REVISIONS = {
    "facebook/dpr-question_encoder-multiset-base": "5325e4ee906435291d63046f535476cb3fc60d43",
    "facebook/dpr-ctx_encoder-multiset-base": "fdb3d46584386d2f20aa00724ae31cebc348d16b",
    "facebook/contriever-msmarco": "abe8c1493371369031bcb1e02acb754cf4e162fa",
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "c5ee24cb16019beea0893ab7796b1df96625c6b8",
    "all-MiniLM-L6-v2": "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
    "sentence-transformers/all-MiniLM-L6-v2": "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
    "cross-encoder/nli-deberta-v3-base": "6c749ce3425cd33b46d187e45b92bbf96ee12ec7",
    "intfloat/e5-base-v2": "f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
    "roberta-large-mnli": "2a8f12d27941090092df78e4ba6f0928eb5eac98",
    "akariasai/PopQA": "098765c79ea10a2cb19c828324e33281b8336ec0",
}


def hf_revision(name):
    """Revision for Hugging Face loaders: the recorded commit if PIN_MODEL_REVISIONS=1, else None (Hub head)."""
    if os.environ.get("PIN_MODEL_REVISIONS") == "1":
        return MODEL_REVISIONS.get(name)
    return None

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
