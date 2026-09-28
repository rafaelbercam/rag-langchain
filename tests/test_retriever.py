import pytest
from unittest.mock import Mock, MagicMock, patch

from langchain.schema import Document

from rag_app.retriever import RetrieverFactory


@pytest.fixture
def mock_vectorstore():
    """Create a mock vectorstore."""
    vectorstore = MagicMock()
    return vectorstore


@pytest.fixture
def mock_llm():
    """Create a mock LLM."""
    return MagicMock()


@pytest.fixture
def mock_retriever():
    """Create a mock retriever."""
    return MagicMock()


def test_create_basic_retriever(mock_vectorstore):
    """Test creating a basic retriever."""
    mock_vectorstore.as_retriever.return_value = MagicMock()

    retriever = RetrieverFactory.create_basic_retriever(mock_vectorstore, k=3)

    assert retriever is not None
    mock_vectorstore.as_retriever.assert_called_once()
    call_kwargs = mock_vectorstore.as_retriever.call_args[1]
    assert call_kwargs['search_kwargs']['k'] == 3


def test_create_multiquery_retriever(mock_vectorstore, mock_llm):
    """Test creating a multi-query retriever."""
    mock_vectorstore.as_retriever.return_value = MagicMock()

    with patch('rag_app.retriever.MultiQueryRetriever.from_llm') as mock_multiquery:
        mock_multiquery.return_value = MagicMock()

        retriever = RetrieverFactory.create_multiquery_retriever(
            mock_vectorstore,
            mock_llm,
            k=3
        )

        assert retriever is not None
        mock_multiquery.assert_called_once()


def test_create_compression_retriever(mock_vectorstore, mock_llm):
    """Test creating a compression retriever."""
    mock_vectorstore.as_retriever.return_value = MagicMock()

    with patch('rag_app.retriever.LLMChainExtractor.from_llm') as mock_extractor:
        mock_extractor.return_value = MagicMock()

        retriever = RetrieverFactory.create_compression_retriever(
            mock_vectorstore,
            mock_llm,
            k=3
        )

        assert retriever is not None
        mock_extractor.assert_called_once()


def test_basic_retriever_with_different_k_values(mock_vectorstore):
    """Test basic retriever with different k values."""
    mock_vectorstore.as_retriever.return_value = MagicMock()

    for k_value in [1, 3, 5, 10]:
        RetrieverFactory.create_basic_retriever(mock_vectorstore, k=k_value)
        call_kwargs = mock_vectorstore.as_retriever.call_args[1]
        assert call_kwargs['search_kwargs']['k'] == k_value


def test_retriever_returns_documents():
    """Test that retrievers can return documents."""
    mock_retriever = MagicMock()
    docs = [
        Document(page_content="Test content 1", metadata={"source": "doc1"}),
        Document(page_content="Test content 2", metadata={"source": "doc2"}),
    ]
    mock_retriever.get_relevant_documents.return_value = docs

    results = mock_retriever.get_relevant_documents("test query")

    assert len(results) == 2
    assert results[0].page_content == "Test content 1"
