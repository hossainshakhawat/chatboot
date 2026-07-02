"""Unit tests for the worker module."""

import unittest
from unittest.mock import patch, MagicMock
from .pdf_reader import PDFReader
from embedding import Embedder


class TestPDFReader(unittest.TestCase):
    """Tests for PDFReader class."""

    def test_chunk_text(self):
        """Test text chunking functionality."""
        reader = PDFReader()
        text = "a" * 1000
        chunks = reader.chunk_text(text, chunk_size=100, overlap=20)

        # Verify chunks are created
        self.assertGreater(len(chunks), 0)

        # Verify chunk sizes
        for chunk in chunks[:-1]:
            self.assertLessEqual(len(chunk), 100)

    def test_chunk_text_with_small_text(self):
        """Test chunking with text smaller than chunk size."""
        reader = PDFReader()
        text = "short text"
        chunks = reader.chunk_text(text, chunk_size=100, overlap=20)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)


class TestEmbedder(unittest.TestCase):
    """Tests for Embedder class."""

    @patch("embedding.embedder.SentenceTransformer")
    def test_embed_text(self, mock_model):
        """Test text embedding."""
        import numpy as np

        mock_instance = MagicMock()
        mock_instance.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_model.return_value = mock_instance

        embedder = Embedder()
        embedding = embedder.embed_text("test text")

        self.assertEqual(len(embedding), 3)
        self.assertEqual(embedding, [0.1, 0.2, 0.3])

    def test_embed_empty_text(self):
        """Test embedding with empty text."""
        embedder = Embedder()
        embedding = embedder.embed_text("")

        self.assertEqual(embedding, [])

    def test_embed_whitespace_text(self):
        """Test embedding with whitespace-only text."""
        embedder = Embedder()
        embedding = embedder.embed_text("   ")

        self.assertEqual(embedding, [])


if __name__ == "__main__":
    unittest.main()
