"""
Centralised factory for LLM and Embedding objects.

Supported LLM providers
-----------------------
    ollama            → ChatOllama  (local / docker Ollama server)
    groq              → ChatGroq    (Groq Cloud API)
    anthropic         → ChatAnthropic (Claude API)
    huggingface_api   → ChatHuggingFace via HuggingFaceEndpoint (remote HF Inference API)
    huggingface_local → ChatHuggingFace via HuggingFacePipeline (runs locally on CPU/GPU)
    google            → ChatGoogleGenerativeAI (Gemini API)

Supported Embedding providers
------------------------------
    ollama            → OllamaEmbeddings
    huggingface       → HuggingFaceEmbeddings  (sentence-transformers, local)
    google            → GoogleGenerativeAIEmbeddings

HuggingFace API would be costly and unnecessary for Embeddings purpose

Auto-fallback
-------------
    groq + anthropic have no embedding API.
    When either is used as the embedding provider it falls back automatically
    to HuggingFace local (sentence-transformers/all-MiniLM-L6-v2).
"""

import os
from typing import Callable, Optional, Dict
import time

from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from config import (
    OLLAMA_BASE_URL,
    MAX_LLM_RETRIES_ON_API_LIMITS,
    LIMIT_HIT_RETRY_BASE_DELAY,
)
from llm import (
    RetryConfig,
    MultiProviderChatLLM,
)


_model_cache: dict[str, Embeddings] = {}
_RETRY_ERRORS = ("429", "resource_exhausted", "quota", "rate limit", "too many requests")

def _cache_model(provider : str) :
    def decorator(func) :
        def wrapper(model: str, api_keys: dict) -> Embeddings :
            cache_key = f"{provider}:{model}"
            if cache_key not in _model_cache:
                _model_cache[cache_key] = func(model, api_keys)
            return _model_cache[cache_key]
        return wrapper
    return decorator

def retry_embeddings(Embedding_Class, params: dict, max_attempts: int = 6, base_delay: float = 2.0):
    def _run(fn, *args, **kwargs):
        for attempt in range(max_attempts):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if any(x in str(e).lower() for x in _RETRY_ERRORS) and attempt < max_attempts - 1:
                    time.sleep(base_delay * (2 ** attempt))
                    continue
                raise

    class _WithRetry(Embedding_Class):
        def embed_documents(self, texts, **kwargs):
            return _run(super().embed_documents, texts, **kwargs)

        def embed_query(self, text, **kwargs):
            return _run(super().embed_query, text, **kwargs)

    return _WithRetry(**params)

# ===========================================================================
# LLM factory
# ===========================================================================

def build_llm(
    config:       Dict[str, Dict],
    retry_config: Optional[RetryConfig] = None,
) -> MultiProviderChatLLM:
    """
    Build a MultiProviderChatLLM from a provider config dict.

    Config shape
    ────────────
    {
        "<provider_name>": {
            "keys":     ["key1", "key2"],           # required — API keys
            "models":   ["model-a", "model-b"],     # required — model IDs
            "base_url": "https://...",              # optional — custom base URL
        },
        ...
    }

    Providers are tried in insertion order (Python 3.7+ dict ordering).
    Within each provider every (key, model) pair is a separate endpoint and
    they are round-robined.

    Example
    ───────
    llm = get_llm(
        config={
            "openai": {
                "keys":   ["sk-aaa", "sk-bbb"],
                "models": ["gpt-4o", "gpt-4o-mini"],
            },
            "anthropic": {
                "keys":   ["sk-ant-xxx"],
                "models": ["claude-3-5-sonnet-20241022"],
            },
            "groq": {
                "keys":   ["gsk_yyy"],
                "models": ["llama-3.1-70b-versatile"],
                "base_url": "https://api.groq.com/openai/v1",
            },
        },
        retry_config=RetryConfig(max_retries=2, cooldown_seconds=30),
    )
    """
    return MultiProviderChatLLM(
        providers_config=config,
        retry_config=retry_config or RetryConfig(),
    )

# ===========================================================================
# Embeddings factory
# ===========================================================================


def _emb_ollama(model: str, api_keys: dict) -> Embeddings:
    from langchain_ollama.embeddings import OllamaEmbeddings
    from ollama_service import get_ollama_service

    return retry_embeddings(OllamaEmbeddings,
    dict(
        model = get_ollama_service().validate_and_ensure_embedding(model),
        base_url = OLLAMA_BASE_URL
    ),
    max_attempts = MAX_LLM_RETRIES_ON_API_LIMITS,
    base_delay = LIMIT_HIT_RETRY_BASE_DELAY,
    )


def _emb_google(model: str, api_keys: dict) -> Embeddings:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    api_key = api_keys.get("google") or os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError(
            "Google API key is required for Google embeddings. "
            "Provide it via api_keys['google'] or GOOGLE_API_KEY."
        )
    return retry_embeddings(GoogleGenerativeAIEmbeddings,
    dict(
        model = model,
        google_api_key = api_key,
    ),
    max_attempts = MAX_LLM_RETRIES_ON_API_LIMITS,
    base_delay = LIMIT_HIT_RETRY_BASE_DELAY,
    )


# ---------------------------------------------------------------------------
# Embedding registry  —  provider string → builder function
# ---------------------------------------------------------------------------

_EMBEDDING_REGISTRY: dict[str, Callable[[str, dict], Embeddings]] = {
    "ollama":      _emb_ollama,
    "google":      _emb_google,
}


# ---------------------------------------------------------------------------
# Embeddings factory
# ---------------------------------------------------------------------------

def build_embeddings(
    provider: str,
    model: str,
    api_keys: Optional[dict] = None,
) -> Embeddings:
    """Build and return an Embeddings object for the selected provider."""
    api_keys = api_keys or {}
    provider = provider.lower()

    builder = _EMBEDDING_REGISTRY.get(provider)
    if builder is None:
        raise ValueError(
            f"Unknown embedding provider '{provider}'. "
            f"Choose one of: {', '.join(sorted(_EMBEDDING_REGISTRY))}"
        )
    return builder(model, api_keys)