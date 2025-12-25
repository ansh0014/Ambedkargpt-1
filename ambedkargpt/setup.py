from setuptools import setup, find_packages

setup(
    name="ambedkargpt",
    version="0.1.0",
    description="SemRAG-based QA system over Dr. B.R. Ambedkar's writings",
    author="Anshul Jagota",
    packages=find_packages(),
    install_requires=[
        "pypdf",
        "nltk",
        "spacy",
        "networkx",
        "python-louvain",
        "sentence-transformers",
        "scikit-learn",
        "numpy",
        "ollama",
        "pyyaml",
    ],
    python_requires=">=3.9",
)
