"""Intent classification module using BERT embeddings.

Each intent is defined by a set of example utterances. The examples are
embedded once with BERT, and incoming text is classified by cosine
similarity against those example vectors.
"""

from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional
import numpy as np
from .config import EMBEDDING_MODEL, INTENT_SIMILARITY_THRESHOLD


DEFAULT_INTENTS: Dict[str, List[str]] = {
    "greeting": [
        "hello",
        "hi there",
        "good morning",
        "hey, how are you?",
    ],
    "goodbye": [
        "bye",
        "goodbye",
        "see you later",
        "thanks, that's all for now",
    ],
    "leave_request": [
        "how do I apply for annual leave?",
        "I want to take a day off next week",
        "how many vacation days do I have left?",
        "what is the sick leave policy?",
        "can I take unpaid leave?",
    ],
    "payroll": [
        "when will I get my salary?",
        "I have a question about my payslip",
        "how do I view my salary statement?",
        "my overtime pay is missing this month",
        "how is my tax deducted from salary?",
    ],
    "benefits": [
        "what health insurance do we have?",
        "how do I enroll in the benefits program?",
        "does the company offer a retirement plan?",
        "what allowances am I entitled to?",
    ],
    "it_support": [
        "my laptop is not working",
        "I forgot my password and can't log in",
        "how do I connect to the office VPN?",
        "I need access to the shared drive",
        "my email is not syncing",
    ],
    "hr_policy": [
        "what is the remote work policy?",
        "what are the official working hours?",
        "what is the dress code at the office?",
        "where can I find the employee handbook?",
        "what is the policy on overtime?",
    ],
    "attendance": [
        "how do I mark my attendance?",
        "I forgot to clock in this morning",
        "how do I correct my timesheet?",
        "what happens if I arrive late?",
    ],
    "onboarding": [
        "I am a new employee, where do I start?",
        "what documents do I need to submit for joining?",
        "who is my assigned buddy or mentor?",
        "how do I complete my new hire training?",
    ],
    "facilities": [
        "how do I book a meeting room?",
        "where can I get a parking pass?",
        "the air conditioning in my area is broken",
        "how do I request office supplies?",
    ],
    "training": [
        "what training courses are available?",
        "how do I enroll in a certification program?",
        "does the company reimburse courses?",
        "when is the next skills workshop?",
    ],
    "document_question": [
        "what does the document say about this?",
        "summarize the pdf",
        "find information in the file",
        "search the document for details",
    ],
    "help": [
        "what can you do?",
        "how does this work?",
        "I need help",
        "show me your features",
    ],
}


class IntentEmbedder:
    """Classifies text into intents using BERT embedding similarity."""

    def __init__(self, intents: Optional[Dict[str, List[str]]] = None):
        """
        Initialize the intent embedder with BERT and precompute intent vectors.

        Args:
            intents: Mapping of intent name to example utterances.
                     Defaults to DEFAULT_INTENTS.
        """
        try:
            self.model = SentenceTransformer(EMBEDDING_MODEL)
        except Exception as e:
            raise RuntimeError(f"Error loading BERT model: {str(e)}")

        self.intents: Dict[str, List[str]] = {}
        self.intent_vectors: Dict[str, np.ndarray] = {}

        for name, examples in (intents or DEFAULT_INTENTS).items():
            self.add_intent(name, examples)

    def add_intent(self, name: str, examples: List[str]) -> None:
        """
        Add or replace an intent and embed its example utterances.

        Args:
            name: Intent name
            examples: Example utterances that express the intent
        """
        if not name or not examples:
            raise ValueError("Intent name and examples are required")

        try:
            vectors = self.model.encode(examples, convert_to_tensor=False)
        except Exception as e:
            raise ValueError(f"Error embedding intent examples: {str(e)}")

        self.intents[name] = list(examples)
        self.intent_vectors[name] = np.asarray(vectors)

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

    def classify(self, text: str) -> dict:
        """
        Classify text into an intent and return the embedding vector.

        Args:
            text: Text to classify

        Returns:
            Dictionary with the detected intent, confidence score,
            per-intent scores, and the embedding vector. The intent is
            None when no score reaches INTENT_SIMILARITY_THRESHOLD.
        """
        embedding = self.embed_text(text)
        if not embedding:
            return {
                "intent": None,
                "confidence": 0.0,
                "scores": {},
                "embedding": [],
            }

        vector = np.asarray(embedding)
        scores = {
            name: self._best_similarity(vector, vectors)
            for name, vectors in self.intent_vectors.items()
        }

        best_intent = max(scores, key=scores.get)
        confidence = scores[best_intent]

        return {
            "intent": best_intent if confidence >= INTENT_SIMILARITY_THRESHOLD else None,
            "confidence": round(confidence, 4),
            "scores": {name: round(score, 4) for name, score in scores.items()},
            "embedding": embedding,
        }

    @staticmethod
    def _best_similarity(vector: np.ndarray, example_vectors: np.ndarray) -> float:
        """
        Compute the highest cosine similarity between a vector and examples.

        Args:
            vector: Embedding of the input text
            example_vectors: Matrix of example embeddings for one intent

        Returns:
            Best cosine similarity as a float in [-1, 1]
        """
        norms = np.linalg.norm(example_vectors, axis=1) * np.linalg.norm(vector)
        norms[norms == 0] = 1e-12
        similarities = example_vectors @ vector / norms
        return float(np.max(similarities))
