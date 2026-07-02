import json
from django.http import JsonResponse
from django.views.decorators.http import require_POST


@require_POST
def message_view(request):
    try:
        data = json.loads(request.body)
    except (TypeError, ValueError):
        return JsonResponse({"error": "Request body must be valid JSON"}, status=400)

    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        return JsonResponse({"error": "Field 'text' is required"}, status=400)

    return JsonResponse({"text": text})
