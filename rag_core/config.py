import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # ── Storage ──────────────────────────────────────────────────────────────
    data_dir: Path = Path(os.getenv("DATA_DIR", "./data"))
    upload_dir: Path = Path(os.getenv("UPLOAD_DIR", "./data/uploads"))
    tinydb_path: Path = Path(os.getenv("TINYDB_PATH", "./data/metadata.json"))

    # ── Vector DB ────────────────────────────────────────────────────────────
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "rag_chunks")

    # ── Embeddings ───────────────────────────────────────────────────────────
    fastembed_model: str = os.getenv("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")

    # ── Chunking / retrieval ─────────────────────────────────────────────────
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))

    # ── Ollama (local, offline) ──────────────────────────────────────────────
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "mistral")
    ollama_timeout_seconds: float = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))

    # ── OpenAI ───────────────────────────────────────────────────────────────
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # ── Groq ─────────────────────────────────────────────────────────────────
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama3-70b-8192")

    # ── Google Gemini (OpenAI-compat endpoint) ───────────────────────────────
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    # ── Active provider ───────────────────────────────────────────────────────
    # Set LLM_PROVIDER in .env to one of: ollama | openai | groq | gemini
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.upload_dir.mkdir(parents=True, exist_ok=True)
