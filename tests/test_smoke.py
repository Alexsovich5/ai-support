import chromadb
import fastapi
import langchain
import numpy


def test_pinned_versions():
    assert fastapi.__version__ == "0.111.0"
    assert chromadb.__version__ == "0.5.3"
    assert langchain.__version__ == "0.1.20"
    assert numpy.__version__ == "1.26.4"
