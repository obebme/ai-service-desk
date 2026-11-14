import math

import httpx

from app.config import settings

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiError(RuntimeError):
    """Base class for expected Gemini integration failures."""


class GeminiConfigurationError(GeminiError):
    """Gemini credentials or model configuration is unavailable."""


class GeminiResponseError(GeminiError, ValueError):
    """Gemini returned a successful HTTP response with an unusable payload."""


def _headers() -> dict[str, str]:
    if not settings.gemini_api_key:
        raise GeminiConfigurationError("Gemini API key is not configured")
    return {"x-goog-api-key": settings.gemini_api_key}


def _json_body(response: httpx.Response) -> dict:
    try:
        body = response.json()
    except ValueError as exc:
        raise GeminiResponseError("Gemini returned invalid JSON") from exc
    if not isinstance(body, dict):
        raise GeminiResponseError("Gemini returned an invalid response payload")
    return body


async def generate(prompt: str, *, json_output: bool = False) -> str:
    if not settings.gemini_model:
        raise GeminiConfigurationError("Gemini generation model is not configured")
    generation_config = {"temperature": 0.1}
    if json_output:
        generation_config["responseMimeType"] = "application/json"
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{API_ROOT}/{settings.gemini_model}:generateContent",
            headers=_headers(),
            json={"contents": [{"parts": [{"text": prompt}]}],
                  "generationConfig": generation_config},
        )
        response.raise_for_status()
    body = _json_body(response)
    candidates = body.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise GeminiResponseError("Gemini returned no candidate")
    candidate = candidates[0]
    content = candidate.get("content") if isinstance(candidate, dict) else None
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list) or not parts or not isinstance(parts[0], dict):
        raise GeminiResponseError("Gemini returned no generated text")
    generated_text = parts[0].get("text")
    if not isinstance(generated_text, str) or not generated_text.strip():
        raise GeminiResponseError("Gemini returned no generated text")
    return generated_text.strip()


async def embed(text: str, *, task_type: str) -> list[float]:
    if not settings.gemini_embedding_model:
        raise GeminiConfigurationError("Gemini embedding model is not configured")
    if settings.embedding_dimensions <= 0:
        raise GeminiConfigurationError("Gemini embedding dimensions are invalid")
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{API_ROOT}/{settings.gemini_embedding_model}:embedContent",
            headers=_headers(),
            json={"model": f"models/{settings.gemini_embedding_model}",
                  "content": {"parts": [{"text": text[:8000]}]},
                  "taskType": task_type,
                  "outputDimensionality": settings.embedding_dimensions},
        )
        response.raise_for_status()
    body = _json_body(response)
    embedding = body.get("embedding")
    vector = embedding.get("values") if isinstance(embedding, dict) else None
    if not isinstance(vector, list):
        raise GeminiResponseError("Gemini returned an invalid embedding")
    if len(vector) != settings.embedding_dimensions:
        raise GeminiResponseError("Gemini returned an unexpected embedding dimension")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in vector):
        raise GeminiResponseError("Gemini returned an invalid embedding value")
    return [float(value) for value in vector]
