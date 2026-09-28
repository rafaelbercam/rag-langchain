# RAG Application with LangChain

A production-ready Python application implementing Retrieval-Augmented Generation (RAG) with LangChain, OpenAI embeddings, and FAISS vector store.

## Overview

This application combines document retrieval with generative AI to answer questions based on a custom knowledge base. It reduces hallucinations by grounding responses in actual documents and provides source attribution.

## Features

- 📄 **Multiple Document Formats**: Load .txt and .md files
- 🔍 **Advanced Retrieval**: Basic, multi-query, and compression-based retrievers
- 💬 **Conversational AI**: Memory-enabled chat with context awareness
- 📊 **Vector Search**: FAISS-based similarity search
- 🧩 **Modular Architecture**: Clean separation of concerns
- 🛡️ **Type Hints**: Full type annotations for better IDE support
- ✅ **Comprehensive Tests**: 80%+ coverage with pytest

## Installation

### Prerequisites
- Python 3.9+
- OpenAI API key

### Setup

1. Clone the repository and navigate to the project directory:
```bash
cd rag-langchain
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Quick Start

### Initialize the Vector Store

Place your documents in `data/sample_documents/` and initialize:

```bash
python -m rag_app.cli init --docs-path ./data/sample_documents
```

This will:
- Load all documents from the specified directory
- Split them into chunks (1000 chars, 200 char overlap)
- Generate embeddings using OpenAI's text-embedding-3-large
- Store vectors in FAISS for fast retrieval

### Query the System

**Single Query Mode:**
```bash
python -m rag_app.cli query "What is RAG and how does it work?"
```

**Interactive Chat Mode:**
```bash
python -m rag_app.cli chat
```

In chat mode, type your questions and the system will maintain conversation history. Type `exit` to quit.

**Document Search Only:**
```bash
python -m rag_app.cli search "embedding" --k 5
```

## Architecture

### Core Components

```
src/rag_app/
├── config.py              # Configuration management
├── document_processor.py   # Document loading and chunking
├── embeddings_manager.py   # Embeddings and vector store
├── retriever.py            # Retriever strategies
├── chains.py               # RAG chains
├── utils.py                # Utility functions
└── cli.py                  # Command-line interface
```

### Data Flow

```
Documents → Processing → Embeddings → Vector Store → Retrieval → Chain → Response
    ↓          ↓              ↓           ↓            ↓        ↓       ↓
  Load     Chunking      OpenAI API    FAISS      k-Nearest  LLM    Answer +
 Files    + Metadata    text-embedding  Index      Neighbors  gpt    Sources
```

## Configuration

Edit `.env` to customize:

```env
OPENAI_API_KEY=sk-...              # Your OpenAI API key
CHUNK_SIZE=1000                    # Document chunk size
CHUNK_OVERLAP=200                  # Overlap between chunks
K_DOCS=3                           # Number of documents to retrieve
EMBEDDING_MODEL=text-embedding-3-large
LLM_MODEL=gpt-3.5-turbo
```

## Advanced Usage

### Using Different Retrievers

```bash
# Multi-query retriever (reformulates queries)
python -m rag_app.cli query "What is RAG?" --retriever-type multiquery

# Compression retriever (reranks documents)
python -m rag_app.cli query "What is RAG?" --retriever-type compression
```

### Retriever Options

- **basic**: Simple similarity search (default, fast)
- **multiquery**: Reformulates query in multiple ways for better coverage
- **compression**: Uses LLM to rerank and extract relevant information

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

With coverage report:

```bash
pytest tests/ --cov=src/rag_app --cov-report=html
```

Test files:
- `test_document_processor.py` - Document loading and chunking
- `test_embeddings.py` - Vector store operations
- `test_retriever.py` - Retriever implementations
- `test_chains.py` - Chain configurations
- `integration_tests.py` - End-to-end workflows

## Project Structure

```
rag-langchain/
├── src/rag_app/            # Main application code
├── tests/                  # Test suite
├── data/
│   ├── sample_documents/   # Your documents here
│   └── vectorstore/        # FAISS index (auto-generated)
├── notebooks/
│   └── demo.ipynb         # Usage examples
├── requirements.txt        # Python dependencies
├── .env.example           # Configuration template
├── .env                   # Your configuration (not in git)
├── README.md              # This file
└── blog.md                # Implementation process documentation
```

## Performance

### Typical Query Execution Time
- Document loading: 1-2 seconds
- Vectorstore creation: 5-10 seconds (one-time)
- Query execution: 2-4 seconds
- Total response time: < 5 seconds

### Cost Estimation
- Embeddings: $0.02 per 1M tokens
- LLM calls: $0.001-0.01 per 1K tokens (varies by model)
- Example: 100 queries × 500 tokens ≈ $0.05

## Troubleshooting

### "OPENAI_API_KEY not found"
- Verify `.env` file exists in project root
- Check that `OPENAI_API_KEY` is set correctly

### "Vector store not found"
- Run `python -m rag_app.cli init` before executing queries

### Slow responses
- Reduce `K_DOCS` in `.env` (fewer documents to process)
- Use "basic" retriever instead of "multiquery"
- Check OpenAI API status

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Reference](https://platform.openai.com/docs)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [RAG Paper: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)

## Roadmap

Future enhancements:
- [ ] Support for more document formats (PDF, docx)
- [ ] Multiple vector store backends (Pinecone, Weaviate)
- [ ] Query caching and semantic deduplication
- [ ] Document auto-refresh and versioning
- [ ] Eval framework for answer quality assessment
- [ ] Web UI and REST API

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing documentation in `blog.md`
