import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from langchain.schema import Document

from rag_app.embeddings_manager import EmbeddingsManager


@pytest.fixture
def mock_embeddings():
    """Create a mock embeddings object."""
    with patch('rag_app.embeddings_manager.OpenAIEmbeddings') as mock:
        yield mock


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(page_content="This is a test document about RAG.", metadata={"source": "test1.txt"}),
        Document(page_content="RAG stands for Retrieval-Augmented Generation.", metadata={"source": "test2.txt"}),
        Document(page_content="It combines retrieval and generation models.", metadata={"source": "test3.txt"}),
    ]


def test_embeddings_manager_initialization(mock_embeddings):
    """Test EmbeddingsManager initialization."""
    manager = EmbeddingsManager(embedding_model="text-embedding-3-large")
    assert manager.embedding_model == "text-embedding-3-large"


@patch('rag_app.embeddings_manager.FAISS')
def test_create_vectorstore(mock_faiss, mock_embeddings, sample_documents):
    """Test creating a vectorstore."""
    manager = EmbeddingsManager()

    mock_vectorstore = MagicMock()
    mock_faiss.from_documents.return_value = mock_vectorstore

    vectorstore = manager.create_vectorstore(sample_documents)

    assert vectorstore is not None
    mock_faiss.from_documents.assert_called_once()


@patch('rag_app.embeddings_manager.FAISS')
def test_create_vectorstore_empty_documents(mock_faiss, mock_embeddings):
    """Test creating a vectorstore with empty documents."""
    manager = EmbeddingsManager()

    with pytest.raises(ValueError):
        manager.create_vectorstore([])


@patch('rag_app.embeddings_manager.FAISS')
def test_save_vectorstore(mock_faiss, mock_embeddings, sample_documents):
    """Test saving a vectorstore."""
    manager = EmbeddingsManager()

    mock_vectorstore = MagicMock()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "test_vectorstore")
        manager.save_vectorstore(mock_vectorstore, path)

        # Check that save_local was called
        mock_vectorstore.save_local.assert_called_once()


@patch('rag_app.embeddings_manager.FAISS')
def test_load_vectorstore(mock_faiss, mock_embeddings):
    """Test loading a vectorstore."""
    manager = EmbeddingsManager()

    mock_vectorstore = MagicMock()
    mock_faiss.load_local.return_value = mock_vectorstore

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir))
        Path(path).mkdir(exist_ok=True)

        # Create dummy files to simulate vectorstore
        Path(path, "index.faiss").touch()
        Path(path, "index.pkl").touch()

        vectorstore = manager.load_vectorstore(path)
        assert vectorstore is not None


def test_load_nonexistent_vectorstore(mock_embeddings):
    """Test loading a non-existent vectorstore."""
    manager = EmbeddingsManager()

    with pytest.raises(FileNotFoundError):
        manager.load_vectorstore("/nonexistent/path")


@patch('rag_app.embeddings_manager.FAISS')
def test_search_similar(mock_faiss, mock_embeddings, sample_documents):
    """Test searching for similar documents."""
    manager = EmbeddingsManager()

    mock_vectorstore = MagicMock()
    mock_result = [(sample_documents[0], 0.95), (sample_documents[1], 0.87)]
    mock_vectorstore.similarity_search_with_score.return_value = mock_result

    results = manager.search_similar(mock_vectorstore, "RAG", k=2)

    assert len(results) == 2
    assert results[0][1] == 0.95
    assert results[1][1] == 0.87


def test_vectorstore_exists(mock_embeddings):
    """Test checking if vectorstore exists."""
    manager = EmbeddingsManager()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = tmpdir
        assert manager.vectorstore_exists(path)

        assert not manager.vectorstore_exists("/nonexistent/path")
