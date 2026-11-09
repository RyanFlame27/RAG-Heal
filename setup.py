"""Install source dependencies; launch from the unpacked project root."""

from setuptools import setup, find_packages
from pathlib import Path

setup(
    name="self-healing-rag",
    version="4.2.0",
    description="BERT-guided failure detection and local evidence recovery for RAG",
    author="Prathamesh Y Biradar, Ryan Thomas, Ankit Saha",
    python_requires=">=3.10",
    packages=find_packages(),
    py_modules=["config"],
    install_requires=[line for raw in Path(__file__).with_name("requirements.txt").read_text().splitlines()
                      if (line := raw.strip()) and not line.startswith("#")],
)
