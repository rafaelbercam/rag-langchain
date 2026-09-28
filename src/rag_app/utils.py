import logging
import time
from typing import Callable, Any

logger = logging.getLogger(__name__)


def timer(func: Callable) -> Callable:
    """Decorator to measure execution time."""
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.info(f"{func.__name__} executed in {elapsed_time:.2f} seconds")
        return result
    return wrapper


def format_search_results(results: list) -> str:
    """Format search results for display."""
    formatted = []
    for i, (doc, score) in enumerate(results, 1):
        formatted.append(f"\n[Result {i}] (Score: {score:.4f})")
        formatted.append(f"Source: {doc.metadata.get('source', 'Unknown')}")
        formatted.append(f"Content: {doc.page_content[:200]}...")
    return "".join(formatted)


def format_qa_result(result: dict) -> str:
    """Format QA result for display."""
    output = f"\nAnswer:\n{result.get('result', 'No answer')}\n"

    source_docs = result.get('source_documents', [])
    if source_docs:
        output += f"\nSources ({len(source_docs)} documents):\n"
        for i, doc in enumerate(source_docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            output += f"  {i}. {source}\n"

    return output


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger.info(f"Logging configured at level {log_level}")
