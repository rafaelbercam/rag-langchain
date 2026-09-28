import logging
from typing import Any, Dict

from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """Use the following pieces of context to answer the question.
If you don't know the answer based on the context, say "I don't have enough information in the provided context to answer this question."
Do not make up information. Always cite the source document in your answer.

Context:
{context}

Question: {question}
Answer:"""


class RAGChainFactory:
    """Factory for creating RAG chains."""

    @staticmethod
    def create_qa_chain(
        retriever: BaseRetriever,
        llm: Any,
        chain_type: str = "stuff"
    ) -> Any:
        """Create a RetrievalQA-like chain using LangChain LCEL."""
        try:
            from langchain.chains import RetrievalQA

            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=SYSTEM_PROMPT_TEMPLATE
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type=chain_type,
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt}
            )

            logger.info(f"Created RetrievalQA chain (chain_type={chain_type})")
            return qa_chain
        except ImportError:
            logger.warning("RetrievalQA not available, creating basic chain")
            return _create_basic_qa_chain(retriever, llm)

    @staticmethod
    def create_conversational_chain(
        retriever: BaseRetriever,
        llm: Any,
        chain_type: str = "stuff"
    ) -> Any:
        """Create a conversational chain with simple memory."""
        logger.info("Created conversational chain")
        return _create_conversational_wrapper(retriever, llm)

    @staticmethod
    def query_chain(chain: Any, query: str) -> Dict[str, Any]:
        """Execute a query against a chain."""
        logger.info(f"Executing query: {query[:50]}...")
        result = chain({"question": query})
        logger.debug(f"Query result: {len(result.get('source_documents', []))} source documents")
        return result


def _create_basic_qa_chain(retriever: BaseRetriever, llm: Any) -> Any:
    """Create a simple QA chain using LCEL."""

    def chain_func(inputs: Dict[str, Any]) -> Dict[str, Any]:
        docs = retriever.invoke(inputs["question"])
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt_text = f"""Use the following pieces of context to answer the question.
If you don't know the answer based on the context, say "I don't have enough information."
Do not make up information.

Context:
{context}

Question: {inputs["question"]}
Answer:"""

        response = llm.invoke(prompt_text)

        return {
            "result": str(response),
            "source_documents": docs
        }

    return chain_func


def _create_conversational_wrapper(
    retriever: BaseRetriever,
    llm: Any
) -> Any:
    """Create a conversational chain wrapper."""

    # Simple conversation history stored as list of dicts
    conversation_history = []

    def chain_func(inputs: Dict[str, Any]) -> Dict[str, Any]:
        docs = retriever.invoke(inputs["question"])
        context = "\n\n".join([doc.page_content for doc in docs])

        # Build history string
        history_str = ""
        for msg in conversation_history[-10:]:  # Last 10 messages
            history_str += f"\n{msg['role']}: {msg['content']}"

        prompt_text = f"""Use the following pieces of context to answer the question.
If you don't know the answer based on the context, say "I don't have enough information."
Do not make up information.

Context:
{context}
{f"Chat History:{history_str}" if history_str else ""}

Question: {inputs['question']}
Answer:"""

        response = llm.invoke(prompt_text)
        response_str = str(response)

        # Add to history
        conversation_history.append({"role": "user", "content": inputs["question"]})
        conversation_history.append({"role": "assistant", "content": response_str})

        return {
            "answer": response_str,
            "source_documents": docs
        }

    return chain_func
