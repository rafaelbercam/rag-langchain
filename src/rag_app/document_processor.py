import logging
from pathlib import Path
from typing import List
from datetime import datetime

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document loading and splitting."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="\n",
        )

    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """Load documents from file paths."""
        documents = []

        for file_path in file_paths:
            path = Path(file_path)

            if not path.exists():
                logger.warning(f"File not found: {file_path}")
                continue

            if path.is_dir():
                loader = DirectoryLoader(
                    file_path,
                    glob="**/*.txt",
                    loader_cls=TextLoader,
                    show_progress=True,
                )
                docs = loader.load()
            else:
                if path.suffix in [".txt", ".md"]:
                    loader = TextLoader(file_path)
                    docs = loader.load()
                else:
                    logger.warning(f"Unsupported file format: {file_path}")
                    continue

            documents.extend(docs)
            logger.info(f"Loaded {len(docs)} documents from {file_path}")

        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        split_docs = self.text_splitter.split_documents(documents)
        logger.info(
            f"Split {len(documents)} documents into {len(split_docs)} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return split_docs

    def process(self, file_paths: List[str]) -> List[Document]:
        """Load and process documents."""
        documents = self.load_documents(file_paths)
        if not documents:
            logger.error("No documents loaded")
            return []

        split_docs = self.split_documents(documents)

        # Add timestamp to metadata
        for doc in split_docs:
            doc.metadata["timestamp"] = datetime.now().isoformat()

        logger.info(f"Processing complete: {len(split_docs)} chunks ready")
        return split_docs
