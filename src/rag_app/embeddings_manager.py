import logging
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class EmbeddingsManager:
    """Manages embeddings and vector store operations."""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model = embedding_model
        logger.info(f"Loading embeddings model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        """Create a FAISS vector store from documents."""
        if not documents:
            raise ValueError("No documents provided to create vectorstore")

        logger.info(f"Creating vectorstore from {len(documents)} documents...")
        vectorstore = FAISS.from_documents(documents, self.embeddings)
        logger.info(f"Vectorstore created successfully")

        return vectorstore

    def save_vectorstore(self, vectorstore: FAISS, path: str) -> None:
        """Save vector store to disk."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        vectorstore.save_local(path)
        logger.info(f"Vectorstore saved to {path}")

    def load_vectorstore(self, path: str) -> FAISS:
        """Load vector store from disk."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Vectorstore not found at {path}")

        vectorstore = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
        logger.info(f"Vectorstore loaded from {path}")

        return vectorstore

    def search_similar(
        self,
        vectorstore: FAISS,
        query: str,
        k: int = 3
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents with scores."""
        results = vectorstore.similarity_search_with_score(query, k=k)
        logger.debug(f"Search query: {query}, found {len(results)} results")

        return results

    def vectorstore_exists(self, path: str) -> bool:
        """Check if vectorstore exists."""
        return Path(path).exists()
