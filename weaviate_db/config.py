"""Configuration settings for the Weaviate module."""

import os

WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", 8080))
WEAVIATE_URL = f"http://{WEAVIATE_HOST}:{WEAVIATE_PORT}"

CLASS_NAME = "Document"
CLASS_PROPERTIES = {
    "content": "text",
    "source": "text",
    "page": "int",
    "chunk_index": "int",
}
