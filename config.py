EMBEDDING_PROVIDER = "google"
PROVIDER = "google"
EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-3.5-flash-lite"

GOOGLE_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"

OLLAMA_BASE_URL="http://ollama:11434"
OLLAMA_ALLOWED_LLM_MODELS=["llama3.1:8b","phi3:mini","mistral:v0.3","qwen2.5:1.5b"]
OLLAMA_ALLOWED_EMBEDDING_MODELS=["nomic-embed-text","embeddinggemma:300m"]

MAX_LLM_RETRIES_ON_API_LIMITS = 6
LIMIT_HIT_RETRY_BASE_DELAY = 2

LLM_CONFIG_PATH = "llm_config.yaml"