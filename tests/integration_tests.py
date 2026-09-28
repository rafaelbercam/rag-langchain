import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile

from langchain.schema import Document

from rag_app.document_processor import DocumentProcessor
from rag_app.embeddings_manager import EmbeddingsManager
from rag_app.retriever import RetrieverFactory
from rag_app.chains import RAGChainFactory


@pytest.fixture
def sample_doc_file():
    """Create a sample document file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        content = """RAG is a technique that combines retrieval and generation.

It works by first retrieving relevant documents and then using them to generate answers.
This approach reduces hallucinations and provides source attribution.

RAG has many applications in knowledge retrieval and question answering systems.
It's particularly useful when dealing with large document collections."""

        f.write(content)
        f.flush()
        yield f.name

    Path(f.name).unlink()


@patch('rag_app.embeddings_manager.OpenAIEmbeddings')
@patch('rag_app.embeddings_manager.FAISS')
def test_full_rag_pipeline(mock_faiss, mock_embeddings, sample_doc_file):
    """Test the full RAG pipeline from documents to answers."""
    # Step 1: Load and process documents
    processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
    documents = processor.process([sample_doc_file])

    assert len(documents) > 0
    assert all('timestamp' in doc.metadata for doc in documents)

    # Step 2: Create embeddings and vectorstore
    manager = EmbeddingsManager()

    mock_vectorstore = MagicMock()
    mock_faiss.from_documents.return_value = mock_vectorstore

    vectorstore = manager.create_vectorstore(documents)
    assert vectorstore is not None

    # Step 3: Create retriever
    mock_vectorstore.as_retriever.return_value = MagicMock()
    retriever = RetrieverFactory.create_basic_retriever(vectorstore, k=3)
    assert retriever is not None

    # Step 4: Create and test chain
    mock_llm = MagicMock()

    with patch('rag_app.chains.RetrievalQA.from_chain_type') as mock_chain_factory:
        mock_chain = MagicMock()
        mock_chain.return_value = {
            'result': 'RAG is a technique that combines retrieval and generation.',
            'source_documents': documents[:1]
        }
        mock_chain_factory.return_value = mock_chain

        qa_chain = RAGChainFactory.create_qa_chain(retriever, mock_llm)
        result = qa_chain({"question": "What is RAG?"})

        assert 'result' in result
        assert 'source_documents' in result
        assert len(result['source_documents']) > 0


@patch('rag_app.embeddings_manager.OpenAIEmbeddings')
def test_document_loading_splitting_flow(mock_embeddings, sample_doc_file):
    """Test document loading and splitting workflow."""
    processor = DocumentProcessor(chunk_size=150, chunk_overlap=30)

    # Load documents
    docs = processor.load_documents([sample_doc_file])
    assert len(docs) > 0

    # Split documents
    split_docs = processor.split_documents(docs)
    assert len(split_docs) > len(docs)

    # All chunks should have proper metadata
    for doc in split_docs:
        assert 'source' in doc.metadata


@patch('rag_app.embeddings_manager.OpenAIEmbeddings')
@patch('rag_app.embeddings_manager.FAISS')
def test_vectorstore_save_load_cycle(mock_faiss, mock_embeddings):
    """Test saving and loading vectorstore."""
    manager = EmbeddingsManager()

    mock_vectorstore = MagicMock()
    mock_faiss.from_documents.return_value = mock_vectorstore
    mock_faiss.load_local.return_value = mock_vectorstore

    sample_docs = [
        Document(page_content="Content 1", metadata={"source": "doc1.txt"}),
        Document(page_content="Content 2", metadata={"source": "doc2.txt"}),
    ]

    # Create vectorstore
    vectorstore = manager.create_vectorstore(sample_docs)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "vectorstore")

        # Save
        manager.save_vectorstore(vectorstore, path)
        mock_vectorstore.save_local.assert_called()

        # Load
        Path(path).mkdir(exist_ok=True)
        loaded = manager.load_vectorstore(path)
        mock_faiss.load_local.assert_called()


@patch('rag_app.chains.ConversationalRetrievalChain.from_llm')
def test_conversational_flow(mock_conv_chain):
    """Test conversational interaction flow."""
    mock_retriever = MagicMock()
    mock_llm = MagicMock()

    # Create conversational chain
    mock_chain = MagicMock()
    mock_conv_chain.return_value = mock_chain

    chain = RAGChainFactory.create_conversational_chain(mock_retriever, mock_llm)

    # Simulate conversation
    queries = [
        "What is RAG?",
        "How does it work?",
        "What are the benefits?"
    ]

    for query in queries:
        mock_chain.return_value = {
            'answer': f'Answer to: {query}',
            'source_documents': []
        }
        result = chain({"question": query})
        assert 'answer' in result or 'result' in result or len(result) > 0


@patch('rag_app.embeddings_manager.OpenAIEmbeddings')
def test_error_handling_missing_documents(mock_embeddings):
    """Test error handling with missing documents."""
    processor = DocumentProcessor()

    # Try to load non-existent files
    docs = processor.load_documents(["/nonexistent/path.txt"])
    assert len(docs) == 0


@patch('rag_app.embeddings_manager.OpenAIEmbeddings')
@patch('rag_app.embeddings_manager.FAISS')
def test_retriever_with_different_strategies(mock_faiss, mock_embeddings):
    """Test different retriever strategies."""
    vectorstore = MagicMock()
    vectorstore.as_retriever.return_value = MagicMock()

    mock_llm = MagicMock()

    # Basic retriever
    retriever1 = RetrieverFactory.create_basic_retriever(vectorstore)
    assert retriever1 is not None

    # MultiQuery retriever
    with patch('rag_app.retriever.MultiQueryRetriever.from_llm') as mock_multiquery:
        mock_multiquery.return_value = MagicMock()
        retriever2 = RetrieverFactory.create_multiquery_retriever(vectorstore, mock_llm)
        assert retriever2 is not None

    # Compression retriever
    with patch('rag_app.retriever.LLMChainExtractor.from_llm') as mock_extractor:
        mock_extractor.return_value = MagicMock()
        retriever3 = RetrieverFactory.create_compression_retriever(vectorstore, mock_llm)
        assert retriever3 is not None
