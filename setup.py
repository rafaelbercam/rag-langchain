from setuptools import setup, find_packages

setup(
    name="rag-app",
    version="1.0.0",
    description="RAG Application with LangChain",
    author="Rafael Bercam",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "langchain>=0.2.0",
        "langchain-openai>=0.2.0",
        "langchain-community>=0.2.0",
        "langchain-text-splitters>=0.2.0",
        "openai>=1.50.0",
        "faiss-cpu>=1.7.4",
        "python-dotenv>=1.0.0",
        "click>=8.1.0",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "rag=rag_app.cli:cli",
        ],
    },
)
