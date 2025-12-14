"""
Application configuration using pydantic-settings.

Loads configuration from environment variables and .env files.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Database
    database_url: PostgresDsn = Field(
        default="postgresql+asyncpg://user:password@localhost:5432/recruiting_agent",
        description="PostgreSQL connection string",
    )

    # LLM Configuration (Ollama)
    llm_model_name: str = Field(
        default="gpt-oss:20b",
        description="Ollama model name to use (e.g., gpt-oss:20b)",
    )
    llm_timeout: int = Field(
        default=120,
        description="Timeout in seconds for LLM requests",
    )
    llm_max_retries: int = Field(
        default=1,
        description="Number of retries on LLM failure",
    )

    # OSCAR Reasoner
    oscar_endpoint: str = Field(
        default="http://localhost:9000",
        description="Endpoint for the OSCAR defeasible reasoner",
    )
    oscar_timeout: int = Field(
        default=30,
        description="Timeout in seconds for OSCAR requests",
    )

    # Vector Store
    vector_store_path: str = Field(
        default="./data/vector_store",
        description="Path to the vector store data directory",
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings",
    )

    # Application
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Get cached application settings.

    Returns:
        Settings: Application settings instance.
    """
    return Settings()
