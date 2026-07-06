"""Intent classification module using BERT embeddings.

Each intent is defined by a set of sub-intents, and each sub-intent by a
set of example utterances. The examples are embedded once with BERT, and
incoming text is classified by cosine similarity against those example
vectors. The best-matching sub-intent determines both the sub-intent and
its parent intent.
"""

from pathlib import Path

from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional, Union
import numpy as np
import yaml
from .config import EMBEDDING_MODEL, INTENT_SIMILARITY_THRESHOLD


# Sub-intent name used when an intent is defined with a flat example list.
GENERAL_SUB_INTENT = "general"

INTENTS_FILE = Path(__file__).parent / "intents.yml"


def load_intents(path: Union[str, Path] = INTENTS_FILE) -> Dict[str, Dict[str, List[str]]]:
    """
    Load intent definitions from a YAML file.

    Args:
        path: Path to a YAML file mapping intent name to sub-intent name to
              example utterances. Defaults to intents.yml in this package.

    Returns:
        Mapping of intent name to sub-intent name to example utterances
    """
    with open(path, encoding="utf-8") as f:
        intents = yaml.safe_load(f)
    if not isinstance(intents, dict) or not intents:
        raise ValueError(f"No intents found in {path}")
    return intents


DEFAULT_INTENTS: Dict[str, Dict[str, List[str]]] = load_intents()


class IntentEmbedder:
    """Classifies text into intents and sub-intents using BERT embedding similarity."""

    def __init__(
        self,
        intents: Optional[Dict[str, Union[Dict[str, List[str]], List[str]]]] = None,
    ):
        """
        Initialize the intent embedder with BERT and precompute intent vectors.

        Args:
            intents: Mapping of intent name to either a mapping of sub-intent
                     name to example utterances, or a flat list of examples
                     (stored under the "general" sub-intent).
                     Defaults to DEFAULT_INTENTS.
        """
        try:
            self.model = SentenceTransformer(EMBEDDING_MODEL)
        except Exception as e:
            raise RuntimeError(f"Error loading BERT model: {str(e)}")

        self.intents: Dict[str, Dict[str, List[str]]] = {}
        self.intent_vectors: Dict[str, Dict[str, np.ndarray]] = {}

        for name, examples in (intents or DEFAULT_INTENTS).items():
            self.add_intent(name, examples)

    def add_intent(
        self,
        name: str,
        examples: Union[Dict[str, List[str]], List[str]],
    ) -> None:
        """
        Add or replace an intent and embed its example utterances.

        Args:
            name: Intent name
            examples: Mapping of sub-intent name to example utterances, or a
                      flat list of examples (stored under the "general"
                      sub-intent)
        """
        if not name or not examples:
            raise ValueError("Intent name and examples are required")

        if isinstance(examples, dict):
            sub_intents = examples
        else:
            sub_intents = {GENERAL_SUB_INTENT: examples}

        embedded: Dict[str, np.ndarray] = {}
        for sub_name, sub_examples in sub_intents.items():
            if not sub_name or not sub_examples:
                raise ValueError(
                    f"Sub-intent name and examples are required (intent '{name}')"
                )
            try:
                vectors = self.model.encode(sub_examples, convert_to_tensor=False)
            except Exception as e:
                raise ValueError(f"Error embedding intent examples: {str(e)}")
            embedded[sub_name] = np.asarray(vectors)

        self.intents[name] = {
            sub_name: list(sub_examples)
            for sub_name, sub_examples in sub_intents.items()
        }
        self.intent_vectors[name] = embedded

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
        Classify text into an intent and sub-intent and return the embedding.

        Args:
            text: Text to classify

        Returns:
            Dictionary with the detected intent, sub-intent, confidence
            score, per-intent scores, per-sub-intent scores for the best
            intent, and the embedding vector. The intent and sub-intent are
            None when no score reaches INTENT_SIMILARITY_THRESHOLD.
        """
        embedding = self.embed_text(text)
        if not embedding:
            return {
                "intent": None,
                "sub_intent": None,
                "confidence": 0.0,
                "scores": {},
                "sub_intent_scores": {},
                "embedding": [],
            }

        vector = np.asarray(embedding)

        # Best similarity per sub-intent, grouped by intent.
        sub_scores: Dict[str, Dict[str, float]] = {
            name: {
                sub_name: self._best_similarity(vector, vectors)
                for sub_name, vectors in sub_vectors.items()
            }
            for name, sub_vectors in self.intent_vectors.items()
        }

        # An intent's score is the best score among its sub-intents.
        scores = {name: max(subs.values()) for name, subs in sub_scores.items()}

        best_intent = max(scores, key=scores.get)
        best_subs = sub_scores[best_intent]
        best_sub_intent = max(best_subs, key=best_subs.get)
        confidence = scores[best_intent]
        accepted = confidence >= INTENT_SIMILARITY_THRESHOLD

        return {
            "intent": best_intent if accepted else None,
            "sub_intent": best_sub_intent if accepted else None,
            "confidence": round(confidence, 4),
            "scores": {name: round(score, 4) for name, score in scores.items()},
            "sub_intent_scores": {
                sub_name: round(score, 4) for sub_name, score in best_subs.items()
            },
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
