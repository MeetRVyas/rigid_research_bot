import os


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

keys = {}
for name in ["GEMINI_API_KEY"] :
    i = 1
    while True :
        key = os.getenv(name + str(i))
        if key is None :
            break
        i += 1
        keys.setdefault(name, []).append(key)

LLM_CONFIG = {
    "google" : {
        "models" : [
            # "gemini-3.5-flash",
            "gemini-3.1-flash-lite",
        ],
        "keys" : keys["GEMINI_API_KEY"],
        "base_url" : GOOGLE_BASE_URL
    },
    # "ollama" : {
    #     "models" : [
    #         "mistral:v0.3",
    #     ],
    #     "keys" : [],
    #     "base_url" : OLLAMA_BASE_URL
    # },
}