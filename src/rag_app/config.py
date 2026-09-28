import os
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for RAG application."""

    def __init__(self, env_path: Optional[str] = None):
        if env_path is None:
            env_path = Path(__file__).parent.parent.parent / ".env"

        logger.info(f"Loading .env from: {env_path}")
        logger.info(f".env exists: {env_path.exists()}")

        load_dotenv(env_path)

        # Validate ANTHROPIC_API_KEY
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        logger.debug(f"ANTHROPIC_API_KEY read: {'***' if self.anthropic_api_key else 'NOT FOUND'}")
        if not self.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found in .env file. "
                "Please set it before running the application."
            )

        # Environment settings
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.log_level = os.getenv("LOG_LEVEL", "INFO")

        # Document processing settings
        self.chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 200))
        self.k_docs = int(os.getenv("K_DOCS", 3))

        # Model settings - now using Claude instead of OpenAI
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")  # HuggingFace model
        self.llm_model = os.getenv("LLM_MODEL", "claude-sonnet-5")  # Claude model

        # Paths
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / "data"
        self.sample_docs_dir = self.data_dir / "sample_documents"
        self.vectorstore_dir = self.data_dir / "vectorstore"
        self.vectorstore_path = self.vectorstore_dir / "faiss_index"

        # Create directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        self.sample_docs_dir.mkdir(exist_ok=True)
        self.vectorstore_dir.mkdir(exist_ok=True)

        logger.info(f"Config initialized: environment={self.environment}")
        logger.info(f"Using Claude ({self.llm_model}) with {self.embedding_model} embeddings")
        logger.debug(f"Vectorstore path: {self.vectorstore_path}")

    def __repr__(self) -> str:
        return (
            f"Config(environment={self.environment}, "
            f"chunk_size={self.chunk_size}, "
            f"embedding_model={self.embedding_model}, "
            f"llm_model={self.llm_model})"
        )
