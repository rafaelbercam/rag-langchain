import logging
import click
import sys
from pathlib import Path
from typing import Optional

from langchain_anthropic import ChatAnthropic

from .config import Config
from .document_processor import DocumentProcessor
from .embeddings_manager import EmbeddingsManager
from .retriever import RetrieverFactory
from .chains import RAGChainFactory
from .utils import setup_logging, format_search_results, format_qa_result

logger = logging.getLogger(__name__)


@click.group()
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
@click.pass_context
def cli(ctx, verbose):
    """RAG Application CLI - Retrieval-Augmented Generation with LangChain."""
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level)

    try:
        ctx.ensure_object(dict)
        ctx.obj["config"] = Config()
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--docs-path",
    type=click.Path(exists=True),
    default="./data/sample_documents",
    help="Path to documents directory or file"
)
@click.pass_context
def init(ctx, docs_path):
    """Initialize the RAG application and create vector store."""
    config = ctx.obj["config"]

    try:
        click.echo(f"Loading documents from {docs_path}...")
        processor = DocumentProcessor(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        documents = processor.process([docs_path])

        if not documents:
            click.echo("No documents found.", err=True)
            sys.exit(1)

        click.echo(f"Processing {len(documents)} document chunks...")

        embeddings_manager = EmbeddingsManager(
            embedding_model=config.embedding_model
        )
        vectorstore = embeddings_manager.create_vectorstore(documents)
        embeddings_manager.save_vectorstore(
            vectorstore,
            str(config.vectorstore_path)
        )

        click.echo(f"✓ Vector store created and saved to {config.vectorstore_path}")

    except Exception as e:
        click.echo(f"Error during initialization: {e}", err=True)
        logger.exception("Initialization failed")
        sys.exit(1)


@cli.command()
@click.argument("query")
@click.option(
    "--retriever-type",
    type=click.Choice(["basic", "multiquery", "compression"]),
    default="basic",
    help="Type of retriever to use"
)
@click.option(
    "--k",
    type=int,
    default=3,
    help="Number of documents to retrieve"
)
@click.pass_context
def query(ctx, query: str, retriever_type: str, k: int):
    """Execute a single query against the RAG system."""
    config = ctx.obj["config"]

    try:
        if not Path(config.vectorstore_path).exists():
            click.echo(
                f"Vector store not found. Run 'rag init' first.",
                err=True
            )
            sys.exit(1)

        # Initialize components
        embeddings_manager = EmbeddingsManager(
            embedding_model=config.embedding_model
        )
        vectorstore = embeddings_manager.load_vectorstore(str(config.vectorstore_path))

        llm = ChatAnthropic(
            api_key=config.anthropic_api_key,
            model=config.llm_model
        )

        # Select retriever type
        if retriever_type == "basic":
            retriever = RetrieverFactory.create_basic_retriever(vectorstore, k=k)
        elif retriever_type == "multiquery":
            retriever = RetrieverFactory.create_multiquery_retriever(vectorstore, llm, k=k)
        else:  # compression
            retriever = RetrieverFactory.create_compression_retriever(vectorstore, llm, k=k)

        # Create and execute chain
        qa_chain = RAGChainFactory.create_qa_chain(retriever, llm)
        result = qa_chain({"question": query})

        click.echo(format_qa_result(result))

    except Exception as e:
        click.echo(f"Error executing query: {e}", err=True)
        logger.exception("Query execution failed")
        sys.exit(1)


@cli.command()
@click.option(
    "--k",
    type=int,
    default=3,
    help="Number of documents to retrieve"
)
@click.pass_context
def chat(ctx, k: int):
    """Start an interactive conversational session."""
    config = ctx.obj["config"]

    try:
        if not Path(config.vectorstore_path).exists():
            click.echo(
                f"Vector store not found. Run 'rag init' first.",
                err=True
            )
            sys.exit(1)

        # Initialize components
        embeddings_manager = EmbeddingsManager(
            embedding_model=config.embedding_model
        )
        vectorstore = embeddings_manager.load_vectorstore(str(config.vectorstore_path))

        llm = ChatAnthropic(
            api_key=config.anthropic_api_key,
            model=config.llm_model
        )

        retriever = RetrieverFactory.create_basic_retriever(vectorstore, k=k)
        qa_chain = RAGChainFactory.create_conversational_chain(retriever, llm)

        click.echo("Starting conversational session (type 'exit' to quit)...\n")

        while True:
            user_input = click.prompt("You")

            if user_input.lower() == "exit":
                click.echo("Goodbye!")
                break

            result = qa_chain({"question": user_input})
            click.echo(f"Assistant: {result['answer']}\n")

    except Exception as e:
        click.echo(f"Error in chat mode: {e}", err=True)
        logger.exception("Chat mode failed")
        sys.exit(1)


@cli.command()
@click.argument("query")
@click.option(
    "--k",
    type=int,
    default=3,
    help="Number of documents to retrieve"
)
@click.pass_context
def search(ctx, query: str, k: int):
    """Search for documents without generating an answer."""
    config = ctx.obj["config"]

    try:
        if not Path(config.vectorstore_path).exists():
            click.echo(
                f"Vector store not found. Run 'rag init' first.",
                err=True
            )
            sys.exit(1)

        # Initialize components
        embeddings_manager = EmbeddingsManager(
            embedding_model=config.embedding_model
        )
        vectorstore = embeddings_manager.load_vectorstore(str(config.vectorstore_path))

        # Search
        results = embeddings_manager.search_similar(vectorstore, query, k=k)

        if not results:
            click.echo("No similar documents found.")
        else:
            click.echo(f"Found {len(results)} relevant documents:\n")
            click.echo(format_search_results(results))

    except Exception as e:
        click.echo(f"Error during search: {e}", err=True)
        logger.exception("Search failed")
        sys.exit(1)


if __name__ == "__main__":
    cli()
