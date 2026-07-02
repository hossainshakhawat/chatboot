"""Main worker orchestrator for PDF processing and embedding."""

import os
from typing import List, Optional
from .pdf_reader import PDFReader
from embedding import Embedder
from weaviate_db import WeaviateClient
from .config import CHUNK_SIZE, CHUNK_OVERLAP


class PDFWorker:
    """Orchestrates PDF processing, embedding, and storage in Weaviate."""

    def __init__(self):
        """Initialize the PDF worker with all required components."""
        self.pdf_reader = PDFReader()
        self.embedder = Embedder()
        self.weaviate = WeaviateClient()
        self.weaviate.create_class()

    def process_pdf(self, file_path: str) -> List[str]:
        """
        Process a PDF file: extract text, create embeddings, store in Weaviate.

        Args:
            file_path: Path to the PDF file

        Returns:
            List of UUIDs of stored documents
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        print(f"Processing PDF: {file_path}")

        # Extract text from PDF
        text_pages = self.pdf_reader.read_pdf(file_path)
        print(f"Extracted {len(text_pages)} pages from PDF")

        # Process each page
        all_documents = []
        chunk_index = 0

        for text, page_num in text_pages:
            # Split into chunks
            chunks = self.pdf_reader.chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
            print(f"Page {page_num}: Split into {len(chunks)} chunks")

            for chunk in chunks:
                if not chunk.strip():
                    continue

                # Generate embedding
                embedding = self.embedder.embed_text(chunk)

                if embedding:
                    all_documents.append(
                        {
                            "content": chunk,
                            "embedding": embedding,
                            "source": file_path,
                            "page": page_num,
                            "chunk_index": chunk_index,
                        }
                    )
                    chunk_index += 1

        # Store all documents in Weaviate
        print(f"Storing {len(all_documents)} document chunks in Weaviate")
        uuids = self.weaviate.store_documents_batch(all_documents)

        print(f"Successfully processed PDF. Stored {len(uuids)} chunks")
        return uuids

    def process_directory(self, directory_path: str) -> dict:
        """
        Process all PDF files in a directory.

        Args:
            directory_path: Path to directory containing PDF files

        Returns:
            Dictionary with processing statistics
        """
        if not os.path.isdir(directory_path):
            raise NotADirectoryError(f"Directory not found: {directory_path}")

        pdf_files = [f for f in os.listdir(directory_path) if f.endswith(".pdf")]
        print(f"Found {len(pdf_files)} PDF files in {directory_path}")

        stats = {"total_files": len(pdf_files), "successful": 0, "failed": 0}

        for pdf_file in pdf_files:
            file_path = os.path.join(directory_path, pdf_file)
            try:
                uuids = self.process_pdf(file_path)
                stats["successful"] += 1
            except Exception as e:
                print(f"Error processing {pdf_file}: {str(e)}")
                stats["failed"] += 1

        return stats

    def search(self, query: str, limit: int = 10) -> List[dict]:
        """
        Search for documents similar to a query.

        Args:
            query: Text query
            limit: Number of results to return

        Returns:
            List of search results
        """
        print(f"Searching for: {query}")

        # Generate embedding for query
        query_embedding = self.embedder.embed_text(query)

        if not query_embedding:
            print("Failed to generate query embedding")
            return []

        # Search in Weaviate
        results = self.weaviate.search(query_embedding, limit)
        print(f"Found {len(results)} results")

        return results

    def clear_database(self):
        """Clear all stored documents from Weaviate."""
        self.weaviate.delete_all()

    def close(self):
        """Close all connections."""
        self.weaviate.close()
