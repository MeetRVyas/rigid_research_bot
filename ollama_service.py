from typing import Optional
import ollama as ollama_client
from fastapi import HTTPException
from config import (
    LLM_MODEL,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_ALLOWED_EMBEDDING_MODELS,
)


class OllamaModelService:
    def __init__(self, base_url: str = None):
        self.base_url = base_url or OLLAMA_BASE_URL
        self.client = ollama_client.Client(host=self.base_url)

    # ---------------------------
    # Introspection
    # ---------------------------

    def list_installed(self) -> set[str]:
        try:
            response = self.client.list()
            # ollama client returns ModelResponse with .models list
            return {m.model for m in response.models}
        except Exception as e:
            print(e)
            return set()

    def is_installed(self, model: str) -> bool:
        installed = self.list_installed()
        # Normalize: ollama may suffix ":latest"
        return model in installed or f"{model}:latest" in installed

    # ---------------------------
    # Pull
    # ---------------------------

    def pull(self, model: str) -> None:
        print(f"Pulling Ollama model: {model}")
        try:
            self.client.pull(model)
            print(f"Model '{model}' pulled successfully.")
        except Exception as e:
            print(f"Failed to pull model '{model}': {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Failed to pull model '{model}': {str(e)}"
            )

    # ---------------------------
    # Validation + auto-pull
    # ---------------------------

    def validate_and_ensure_embedding(self, model: str) -> str:
        if model not in OLLAMA_ALLOWED_EMBEDDING_MODELS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Embedding model '{model}' is not allowed. "
                    f"Allowed models: {sorted(OLLAMA_ALLOWED_EMBEDDING_MODELS)}"
                )
            )
        if not self.is_installed(model):
            print(f"Embedding model '{model}' not installed — pulling.")
            self.pull(model)
        return model

    # ---------------------------
    # Startup preload
    # ---------------------------

    def preload_defaults(self) -> None:
        """Pull default LLM + embedding model if missing. Called on app startup."""
        defaults = [LLM_MODEL, EMBEDDING_MODEL]
        for model in defaults:
            if not self.is_installed(model):
                print(f"Preloading default model: {model}")
                try:
                    self.pull(model)
                except HTTPException as e:
                    # Don't crash startup — warn and continue
                    print(f"Could not preload '{model}': {e.detail}")
            else:
                print(f"Default model '{model}' already installed.")


# Module-level singleton
_ollama_service: Optional[OllamaModelService] = None


def get_ollama_service() -> OllamaModelService:
    global _ollama_service
    if _ollama_service is None:
        _ollama_service = OllamaModelService()
    return _ollama_service