"""Text embedding module using BERT and Spacy."""

from sentence_transformers import SentenceTransformer
import spacy
from typing import List
import numpy as np
from .config import EMBEDDING_MODEL, SPACY_MODEL


class Embedder:
    """Generates embeddings using BERT model and Spacy NLP."""

    def __init__(self):
        """Initialize the embedder with BERT and Spacy models."""
        try:
            self.model = SentenceTransformer(EMBEDDING_MODEL)
        except Exception as e:
            raise RuntimeError(f"Error loading BERT model: {str(e)}")

        try:
            self.nlp = spacy.load(SPACY_MODEL)
        except OSError:
            print(f"Spacy model {SPACY_MODEL} not found. Please install it using:")
            print(f"python -m spacy download {SPACY_MODEL}")
            self.nlp = None

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a text using BERT.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        if not text or not text.strip():
            return []

        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            raise ValueError(f"Error generating embedding: {str(e)}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings
        """
        if not texts:
            return []

        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            raise ValueError(f"Error generating embeddings: {str(e)}")

    def process_with_spacy(self, text: str) -> dict:
        """
        Process text with Spacy NLP pipeline.

        Args:
            text: Text to process

        Returns:
            Dictionary with NLP processing results
        """
        if not self.nlp:
            return {"entities": [], "tokens": [], "processed": False}

        try:
            doc = self.nlp(text)
            return {
                "entities": [
                    {"text": ent.text, "label": ent.label_} for ent in doc.ents
                ],
                "tokens": [token.text for token in doc],
                "processed": True,
            }
        except Exception as e:
            print(f"Error processing with Spacy: {str(e)}")
            return {"entities": [], "tokens": [], "processed": False}
