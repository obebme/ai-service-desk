import asyncio

import pytest

from app import gemini
from app.ai import analyze


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self.body


def test_generate_uses_configured_model_and_json_mode(monkeypatch):
    request = {}

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, **kwargs):
            request.update(url=url, **kwargs)
            return FakeResponse({"candidates": [{"content": {"parts": [{"text": '{"ok":true}'}]}}]})

    monkeypatch.setattr(gemini, "settings", gemini.settings.model_copy(update={"gemini_api_key": "test-key"}))
    monkeypatch.setattr(gemini.httpx, "AsyncClient", lambda **kwargs: FakeClient())
    result = asyncio.run(gemini.generate("prompt", json_output=True))

    assert result == '{"ok":true}'
    assert request["url"].endswith(f"/{gemini.settings.gemini_model}:generateContent")
    assert request["headers"]["x-goog-api-key"] == "test-key"
    assert request["json"]["generationConfig"]["responseMimeType"] == "application/json"


def test_embed_checks_dimension_and_sets_retrieval_task(monkeypatch):
    request = {}
    settings = gemini.settings.model_copy(update={"gemini_api_key": "test-key", "embedding_dimensions": 3})

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, **kwargs):
            request.update(url=url, **kwargs)
            return FakeResponse({"embedding": {"values": [0.1, 0.2, 0.3]}})

    monkeypatch.setattr(gemini, "settings", settings)
    monkeypatch.setattr(gemini.httpx, "AsyncClient", lambda **kwargs: FakeClient())
    vector = asyncio.run(gemini.embed("vpn help", task_type="RETRIEVAL_QUERY"))

    assert vector == [0.1, 0.2, 0.3]
    assert request["json"]["taskType"] == "RETRIEVAL_QUERY"
    assert request["json"]["outputDimensionality"] == 3


def test_analysis_validates_gemini_json(monkeypatch):
    async def fake_generate(prompt, *, json_output=False):
        assert json_output is True
        return ('{"category":"network","subcategory":"vpn","priority":"high",'
                '"summary":"VPN unavailable","sentiment":"neutral","needs_human":true,'
                '"confidence":0.9,"suggested_action":"escalate"}')

    monkeypatch.setattr("app.ai.generate", fake_generate)
    result = asyncio.run(analyze("VPN down"))
    assert result.category == "network"
    assert result.needs_human is True


def test_analysis_rejects_malformed_gemini_json(monkeypatch):
    async def fake_generate(*args, **kwargs):
        return "not-json"

    monkeypatch.setattr("app.ai.generate", fake_generate)
    with pytest.raises(ValueError, match="invalid analysis"):
        asyncio.run(analyze("VPN down"))

@pytest.mark.parametrize(("payload", "expected"), [
    ('{"answer":"Restart the VPN client","source_ids":[2,2]}', ("Restart the VPN client", ["VPN.md"])),
    ('null', None),
    ('{"answer":"Unsupported","source_ids":[3]}', "invalid"),
    ('{"answer":"Unsupported","source_ids":[]}', "invalid"),
])
def test_grounded_answer_uses_only_valid_selected_sources(monkeypatch, payload, expected):
    from app.ai import AIResponseError, grounded_answer

    async def generate(prompt, *, json_output):
        assert json_output is True
        assert '"id": 2' in prompt
        return payload

    monkeypatch.setattr("app.ai.generate", generate)
    passages = [{"title": "Other.md", "content": "Different topic"},
                {"title": "VPN.md", "content": "Restart the VPN client"}]
    if expected == "invalid":
        with pytest.raises(AIResponseError):
            asyncio.run(grounded_answer("VPN help", passages))
    else:
        assert asyncio.run(grounded_answer("VPN help", passages)) == expected
