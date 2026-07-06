"""Intent classification module using BERT embeddings.

Each intent is defined by a set of sub-intents, and each sub-intent by a
set of example utterances. The examples are embedded once with BERT, and
incoming text is classified by cosine similarity against those example
vectors. The best-matching sub-intent determines both the sub-intent and
its parent intent.
"""

from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional, Union
import numpy as np
from .config import EMBEDDING_MODEL, INTENT_SIMILARITY_THRESHOLD


# Sub-intent name used when an intent is defined with a flat example list.
GENERAL_SUB_INTENT = "general"

DEFAULT_INTENTS: Dict[str, Dict[str, List[str]]] = {
    "greeting": {
        "general": [
            "hello",
            "hi there",
            "good morning",
            "hey, how are you?",
        ],
    },
    "goodbye": {
        "general": [
            "bye",
            "goodbye",
            "see you later",
            "thanks, that's all for now",
        ],
    },
    "leave_request": {
        "apply_leave": [
            "how do I apply for annual leave?",
            "I want to take a day off next week",
            "can I take unpaid leave?",
            "I need to request time off",
        ],
        "leave_balance": [
            "how many vacation days do I have left?",
            "what is my remaining leave balance?",
            "check my available leave days",
        ],
        "leave_policy": [
            "what is the sick leave policy?",
            "how does maternity leave work here?",
            "what types of leave does the company offer?",
        ],
    },
    "payroll": {
        "salary_payment": [
            "when will I get my salary?",
            "what date is payday this month?",
            "my salary has not been credited yet",
        ],
        "payslip": [
            "I have a question about my payslip",
            "how do I view my salary statement?",
            "where can I download my payslip?",
        ],
        "overtime_pay": [
            "my overtime pay is missing this month",
            "how is overtime calculated in my salary?",
        ],
        "tax_deduction": [
            "how is my tax deducted from salary?",
            "why is my tax deduction higher this month?",
            "how do I get my tax certificate?",
        ],
    },
    "benefits": {
        "health_insurance": [
            "what health insurance do we have?",
            "how do I add my family to the health plan?",
            "does the insurance cover dental care?",
        ],
        "retirement": [
            "does the company offer a retirement plan?",
            "how does the provident fund work?",
        ],
        "enrollment": [
            "how do I enroll in the benefits program?",
            "when is the benefits enrollment period?",
        ],
        "allowances": [
            "what allowances am I entitled to?",
            "is there a transport or meal allowance?",
        ],
    },
    "it_support": {
        "hardware": [
            "my laptop is not working",
            "my monitor screen is flickering",
            "I need a replacement keyboard",
        ],
        "account_access": [
            "I forgot my password and can't log in",
            "my account is locked",
            "I need access to the shared drive",
        ],
        "network": [
            "how do I connect to the office VPN?",
            "the office wifi is not working",
        ],
        "email": [
            "my email is not syncing",
            "I can't send emails from outlook",
        ],
    },
    "hr_policy": {
        "remote_work": [
            "what is the remote work policy?",
            "can I work from home this week?",
        ],
        "working_hours": [
            "what are the official working hours?",
            "what is the policy on overtime?",
        ],
        "dress_code": [
            "what is the dress code at the office?",
            "can I wear casual clothes on friday?",
        ],
        "handbook": [
            "where can I find the employee handbook?",
            "where are the company policies documented?",
        ],
    },
    "attendance": {
        "clock_in": [
            "how do I mark my attendance?",
            "I forgot to clock in this morning",
        ],
        "timesheet": [
            "how do I correct my timesheet?",
            "my working hours are recorded wrong",
        ],
        "late_arrival": [
            "what happens if I arrive late?",
            "I will be late to the office today",
        ],
    },
    "onboarding": {
        "getting_started": [
            "I am a new employee, where do I start?",
            "what should I do on my first day?",
        ],
        "documents": [
            "what documents do I need to submit for joining?",
            "where do I upload my joining paperwork?",
        ],
        "mentorship": [
            "who is my assigned buddy or mentor?",
            "who should I contact for onboarding questions?",
        ],
        "new_hire_training": [
            "how do I complete my new hire training?",
            "where do I find the orientation schedule?",
        ],
    },
    "facilities": {
        "room_booking": [
            "how do I book a meeting room?",
            "is the conference room free this afternoon?",
        ],
        "parking": [
            "where can I get a parking pass?",
            "is there visitor parking at the office?",
        ],
        "maintenance": [
            "the air conditioning in my area is broken",
            "the light in the hallway is not working",
        ],
        "supplies": [
            "how do I request office supplies?",
            "I need a new chair for my desk",
        ],
    },
    "training": {
        "courses": [
            "what training courses are available?",
            "where can I browse the learning catalog?",
        ],
        "certification": [
            "how do I enroll in a certification program?",
            "does the company support professional certifications?",
        ],
        "reimbursement": [
            "does the company reimburse courses?",
            "how do I claim my course fees?",
        ],
        "workshops": [
            "when is the next skills workshop?",
            "are there any upcoming training sessions?",
        ],
    },
    "document_question": {
        "general": [
            "what does the document say about this?",
            "summarize the pdf",
            "find information in the file",
            "search the document for details",
        ],
    },
    "help": {
        "general": [
            "what can you do?",
            "how does this work?",
            "I need help",
            "show me your features",
        ],
    },
}


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
