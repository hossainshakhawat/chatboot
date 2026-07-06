import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from embedding.intent_embedder import IntentEmbedder

_intent_embedder = None


def get_intent_embedder():
    """Return a shared IntentEmbedder, loading the BERT model only once."""
    global _intent_embedder
    if _intent_embedder is None:
        _intent_embedder = IntentEmbedder()
    return _intent_embedder


@csrf_exempt
@require_POST
def message_view(request):
    try:
        data = json.loads(request.body)
    except (TypeError, ValueError):
        return JsonResponse({"error": "Request body must be valid JSON"}, status=400)

    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        return JsonResponse({"error": "Field 'text' is required"}, status=400)

    try:
        result = get_intent_embedder().classify(text)
    except (RuntimeError, ValueError) as e:
        return JsonResponse({"error": f"Intent classification failed: {e}"}, status=500)

    return JsonResponse(
        {
            "text": text,
            "intent": result["intent"],
            "confidence": result["confidence"],
            "scores": result["scores"],
        }
    )
