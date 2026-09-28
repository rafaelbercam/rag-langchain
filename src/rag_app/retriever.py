import logging
from typing import Any

from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)


class RetrieverFactory:
    """Factory for creating different types of retrievers."""

    @staticmethod
    def create_basic_retriever(vectorstore: FAISS, k: int = 3) -> BaseRetriever:
        """Create a basic similarity retriever."""
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        logger.info(f"Created basic retriever with k={k}")
        return retriever

    @staticmethod
    def create_multiquery_retriever(
        vectorstore: FAISS,
        llm: Any,
        k: int = 3
    ) -> BaseRetriever:
        """Create a multi-query-inspired retriever (fallback to basic for compatibility)."""
        # MultiQueryRetriever requires additional setup in current LangChain version
        # Fallback to basic retriever with larger k for broader coverage
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k + 2}
        )
        logger.info("Created retriever with expanded k for better coverage")
        return retriever

    @staticmethod
    def create_compression_retriever(
        vectorstore: FAISS,
        llm: Any,
        k: int = 3
    ) -> BaseRetriever:
        """Create a compression-inspired retriever (fallback to basic for compatibility)."""
        # ContextualCompressionRetriever requires complex setup
        # Fallback to basic retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        logger.info("Created basic retriever (compression fallback)")
        return retriever
