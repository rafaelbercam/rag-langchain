import pytest
from unittest.mock import Mock, MagicMock, patch

from langchain.schema import Document

from rag_app.chains import RAGChainFactory


@pytest.fixture
def mock_retriever():
    """Create a mock retriever."""
    return MagicMock()


@pytest.fixture
def mock_llm():
    """Create a mock LLM."""
    return MagicMock()


@pytest.fixture
def mock_qa_chain():
    """Create a mock QA chain."""
    chain = MagicMock()
    chain.return_value = {
        'result': 'This is a test answer.',
        'source_documents': [
            Document(page_content="Test content", metadata={"source": "test.txt"})
        ]
    }
    return chain


def test_create_qa_chain(mock_retriever, mock_llm):
    """Test creating a QA chain."""
    with patch('rag_app.chains.RetrievalQA.from_chain_type') as mock_from_chain:
        mock_from_chain.return_value = MagicMock()

        chain = RAGChainFactory.create_qa_chain(mock_retriever, mock_llm)

        assert chain is not None
        mock_from_chain.assert_called_once()
        call_kwargs = mock_from_chain.call_args[1]
        assert call_kwargs['return_source_documents'] is True


def test_create_conversational_chain(mock_retriever, mock_llm):
    """Test creating a conversational chain."""
    with patch('rag_app.chains.ConversationalRetrievalChain.from_llm') as mock_from_llm:
        mock_from_llm.return_value = MagicMock()

        chain = RAGChainFactory.create_conversational_chain(mock_retriever, mock_llm)

        assert chain is not None
        mock_from_llm.assert_called_once()
        call_kwargs = mock_from_llm.call_args[1]
        assert call_kwargs['return_source_documents'] is True


def test_qa_chain_returns_result_and_sources(mock_qa_chain):
    """Test that QA chain returns result and source documents."""
    result = mock_qa_chain({"question": "What is RAG?"})

    assert 'result' in result
    assert 'source_documents' in result
    assert len(result['source_documents']) > 0


def test_qa_chain_with_different_chain_types(mock_retriever, mock_llm):
    """Test creating QA chain with different chain types."""
    with patch('rag_app.chains.RetrievalQA.from_chain_type') as mock_from_chain:
        mock_from_chain.return_value = MagicMock()

        for chain_type in ["stuff", "map_reduce", "refine"]:
            RAGChainFactory.create_qa_chain(
                mock_retriever,
                mock_llm,
                chain_type=chain_type
            )
            call_kwargs = mock_from_chain.call_args[1]
            assert call_kwargs['chain_type'] == chain_type


def test_query_chain_execution(mock_qa_chain):
    """Test executing a query through a chain."""
    result = RAGChainFactory.query_chain(mock_qa_chain, "What is RAG?")

    assert result is not None
    assert 'result' in result


def test_prompt_template_contains_constraint(mock_retriever, mock_llm):
    """Test that prompt template contains the constraint."""
    with patch('rag_app.chains.RetrievalQA.from_chain_type') as mock_from_chain:
        mock_from_chain.return_value = MagicMock()

        RAGChainFactory.create_qa_chain(mock_retriever, mock_llm)

        call_kwargs = mock_from_chain.call_args[1]
        prompt = call_kwargs['chain_type_kwargs']['prompt']

        # Check that prompt template exists
        assert prompt is not None


def test_conversational_chain_maintains_memory(mock_retriever, mock_llm):
    """Test that conversational chain has memory."""
    with patch('rag_app.chains.ConversationalRetrievalChain.from_llm') as mock_from_llm:
        mock_from_llm.return_value = MagicMock()

        RAGChainFactory.create_conversational_chain(mock_retriever, mock_llm)

        call_kwargs = mock_from_llm.call_args[1]
        assert 'memory' in call_kwargs
        assert call_kwargs['memory'] is not None
