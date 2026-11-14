import asyncio

from app import rag


def test_search_limits_results_to_current_embedding_model(monkeypatch):
    calls = {}

    async def fake_embed(query, *, task_type):
        assert task_type == "RETRIEVAL_QUERY"
        return [0.1, 0.2]

    class Result:
        def mappings(self):
            return [{"content": "Restart VPN", "title": "VPN guide", "source_name": "vpn.md", "score": 0.8}]

    class FakeDB:
        def execute(self, statement, params):
            calls["sql"] = str(statement)
            calls["params"] = params
            return Result()

    monkeypatch.setattr(rag, "embed", fake_embed)
    rows = asyncio.run(rag.search(FakeDB(), "VPN unavailable", limit=2))

    assert rows[0]["title"] == "VPN guide"
    assert "embedding_model = :model" in calls["sql"]
    assert calls["params"]["limit"] == 2
    assert calls["params"]["model"] == rag.settings.gemini_embedding_model
