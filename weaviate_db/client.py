"""Weaviate database client for storing and retrieving embeddings."""

import weaviate
from weaviate.connect import ConnectionParams
from typing import List, Dict, Any, Optional
import uuid
from .config import WEAVIATE_URL, CLASS_NAME


class WeaviateClient:
    """Client for interacting with Weaviate database."""

    def __init__(self):
        """Initialize Weaviate client."""
        try:
            self.client = weaviate.connect_to_local(
                host="localhost",
                port=8080,
                skip_init_checks=False,
            )
            self.client.is_ready()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Weaviate: {str(e)}")

    def create_class(self):
        """Create the Document class in Weaviate if it doesn't exist."""
        try:
            if self.client.collections.exists(CLASS_NAME):
                print(f"Class {CLASS_NAME} already exists")
                return

            self.client.collections.create(
                name=CLASS_NAME,
                properties=[
                    weaviate.classes.Property(
                        name="content", data_type=weaviate.classes.DataType.TEXT
                    ),
                    weaviate.classes.Property(
                        name="source", data_type=weaviate.classes.DataType.TEXT
                    ),
                    weaviate.classes.Property(
                        name="page", data_type=weaviate.classes.DataType.INT
                    ),
                    weaviate.classes.Property(
                        name="chunk_index", data_type=weaviate.classes.DataType.INT
                    ),
                ],
                vectorizer_config=weaviate.classes.config.Configure.Vectorizer.none(),
            )
            print(f"Created class {CLASS_NAME}")
        except Exception as e:
            print(f"Error creating class: {str(e)}")

    def store_document(
        self,
        content: str,
        embedding: List[float],
        source: str,
        page: int = 1,
        chunk_index: int = 0,
    ) -> Optional[str]:
        """
        Store a document chunk with embedding in Weaviate.

        Args:
            content: Text content
            embedding: Vector embedding
            source: Source document path
            page: Page number
            chunk_index: Chunk index in the document

        Returns:
            UUID of stored object or None if failed
        """
        try:
            collection = self.client.collections.get(CLASS_NAME)

            doc_uuid = str(uuid.uuid4())
            collection.data.insert(
                uuid=doc_uuid,
                properties={
                    "content": content,
                    "source": source,
                    "page": page,
                    "chunk_index": chunk_index,
                },
                vector=embedding,
            )
            return doc_uuid
        except Exception as e:
            print(f"Error storing document: {str(e)}")
            return None

    def store_documents_batch(
        self, documents: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Store multiple documents in batch.

        Args:
            documents: List of document dicts with keys:
                - content: Text content
                - embedding: Vector embedding
                - source: Source document path
                - page: Page number
                - chunk_index: Chunk index

        Returns:
            List of UUIDs of stored objects
        """
        uuids = []
        try:
            collection = self.client.collections.get(CLASS_NAME)

            with collection.batch.dynamic() as batch:
                for doc in documents:
                    doc_uuid = str(uuid.uuid4())
                    batch.add_object(
                        uuid=doc_uuid,
                        properties={
                            "content": doc["content"],
                            "source": doc["source"],
                            "page": doc.get("page", 1),
                            "chunk_index": doc.get("chunk_index", 0),
                        },
                        vector=doc["embedding"],
                    )
                    uuids.append(doc_uuid)
        except Exception as e:
            print(f"Error storing documents in batch: {str(e)}")

        return uuids

    def search(
        self, query_embedding: List[float], limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query vector embedding
            limit: Number of results to return

        Returns:
            List of search results
        """
        try:
            collection = self.client.collections.get(CLASS_NAME)
            results = collection.query.near_vector(
                near_vector=query_embedding, limit=limit
            )

            return [
                {
                    "id": obj.uuid,
                    "content": obj.properties.get("content"),
                    "source": obj.properties.get("source"),
                    "page": obj.properties.get("page"),
                    "chunk_index": obj.properties.get("chunk_index"),
                    "distance": obj.metadata.distance,
                }
                for obj in results.objects
            ]
        except Exception as e:
            print(f"Error searching documents: {str(e)}")
            return []

    def delete_all(self):
        """Delete all documents from the class."""
        try:
            self.client.collections.delete(CLASS_NAME)
            print(f"Deleted all documents from {CLASS_NAME}")
        except Exception as e:
            print(f"Error deleting documents: {str(e)}")

    def close(self):
        """Close the Weaviate connection."""
        try:
            self.client.close()
        except Exception as e:
            print(f"Error closing connection: {str(e)}")
