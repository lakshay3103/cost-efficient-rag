"""
Centralized configuration. Everything tunable lives here and is sourced
from environment variables (via .env in local dev). No secrets or magic
numbers are hardcoded elsewhere in the app.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Secrets ---
    openai_api_key: str

    # --- Embedding ---
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # --- Generation ---
    llm_model: str = "gpt-4o-mini"

    # --- Chunking defaults ---
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # --- Retrieval ---
    default_top_k: int = 5
    min_relevance_score: float = 0.3

    # --- Storage ---
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection_name: str = "rag_corpus"

    # --- Dev/test only: use deterministic fake embeddings + canned LLM
    # responses instead of calling OpenAI. Lets the pipeline be exercised
    # end-to-end in environments without network access to the OpenAI API.
    # MUST be false for real use -- set via MOCK_MODE=true in .env.
    mock_mode: bool = False


settings = Settings()
