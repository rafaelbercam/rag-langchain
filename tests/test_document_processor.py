import pytest
import tempfile
from pathlib import Path

from rag_app.document_processor import DocumentProcessor


@pytest.fixture
def temp_text_file():
    """Create a temporary text file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        content = """This is a test document.

It contains multiple paragraphs that should be split into chunks.
Each chunk will have a size of 100 characters with 20 character overlap.

The DocumentProcessor should handle this correctly.
It should create chunks with proper metadata.

This is the final paragraph of the test document."""

        f.write(content)
        f.flush()
        yield f.name

    Path(f.name).unlink()


@pytest.fixture
def processor():
    """Create a DocumentProcessor instance."""
    return DocumentProcessor(chunk_size=100, chunk_overlap=20)


def test_load_documents(processor, temp_text_file):
    """Test loading documents from file."""
    documents = processor.load_documents([temp_text_file])

    assert len(documents) > 0
    assert "test document" in documents[0].page_content.lower()


def test_split_documents(processor, temp_text_file):
    """Test splitting documents into chunks."""
    documents = processor.load_documents([temp_text_file])
    split_docs = processor.split_documents(documents)

    assert len(split_docs) > 0
    # Chunks should not exceed chunk_size
    for doc in split_docs:
        assert len(doc.page_content) <= processor.chunk_size + 50


def test_process_full_pipeline(processor, temp_text_file):
    """Test the full process pipeline."""
    docs = processor.process([temp_text_file])

    assert len(docs) > 0
    # Check that metadata is added
    for doc in docs:
        assert "timestamp" in doc.metadata
        assert "source" in doc.metadata


def test_chunk_overlap(processor, temp_text_file):
    """Test that chunks have proper overlap."""
    documents = processor.load_documents([temp_text_file])
    split_docs = processor.split_documents(documents)

    if len(split_docs) > 1:
        # Check that consecutive chunks have overlap
        first_chunk = split_docs[0].page_content
        second_chunk = split_docs[1].page_content

        # The end of the first chunk should appear in the second chunk
        last_chars = first_chunk[-50:].strip().split()[-3:]
        assert any(word in second_chunk for word in last_chars if word)


def test_nonexistent_file(processor):
    """Test handling of non-existent files."""
    documents = processor.load_documents(["/nonexistent/file.txt"])
    assert len(documents) == 0


def test_unsupported_file_format(processor):
    """Test handling of unsupported file formats."""
    with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
        f.write(b"test content")
        f.flush()

        documents = processor.load_documents([f.name])
        assert len(documents) == 0

        Path(f.name).unlink()
