"""Configuration management."""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment."""
    
    # Database
    database_url: str = "postgresql://postgres:postgres@localhost:5432/mnemosyne"
    redis_url: str = "redis://localhost:6379/0"
    
    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384  # MiniLM-L6-v2 dimension
    
    # Memory settings
    importance_threshold: float = 0.32  # Minimum importance to store
    default_top_k: int = 10  # Default number of memories to retrieve
    
    # OpenAI (for LLM-based importance scoring)
    openai_api_key: str = ""
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
