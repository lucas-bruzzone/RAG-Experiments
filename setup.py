"""Setup for RAG package"""

from setuptools import setup, find_packages

setup(
    name="rag",
    version="0.1.0",
    description="RAG (Retrieval Augmented Generation) system",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "langchain==0.1.0",
        "langchain-community==0.0.10",
        "chromadb==0.4.22",
        "pysqlite3-binary==0.5.2.post1",
        "sentence-transformers==2.3.1",
        "pypdf==4.0.1",
        "python-dotenv==1.0.1",
        "pyyaml==6.0.1",
    ],
    extras_require={
        "dev": [
            "jupyter==1.0.0",
            "black==24.1.1",
            "pytest==8.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "rag=cli:main",
        ]
    }
)
