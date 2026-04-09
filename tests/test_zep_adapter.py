"""Tests for the Zep baseline adapters."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from zep_python import NotFoundError

from benchmarks.zep_baseline import create_zep_adapter
from benchmarks.zep_real import RealZepMemoryAdapter


class FakeMemoryClient:
    def __init__(self):
        self.sessions: dict[str, list] = {}

    async def get_session(self, session_id: str):
        if session_id not in self.sessions:
            raise NotFoundError(body={"message": "not found"})
        return SimpleNamespace(session_id=session_id)

    async def add_session(self, session_id: str, user_id=None, metadata=None):
        self.sessions.setdefault(session_id, [])
        return SimpleNamespace(session_id=session_id, user_id=user_id, metadata=metadata)

    async def add(self, session_id: str, messages):
        self.sessions.setdefault(session_id, []).extend(messages)
        return SimpleNamespace(message="ok")

    async def search_sessions(self, **kwargs):
        text = kwargs.get("text", "")
        tokens = set(text.lower().replace("\n", " ").split())
        session_ids = kwargs.get("session_ids")
        results = []

        for session_id, messages in self.sessions.items():
            if session_ids and session_id not in session_ids:
                continue

            for message in messages:
                if tokens & set((message.content or "").lower().split()):
                    results.append(
                        SimpleNamespace(
                            message=message,
                            summary=None,
                            fact=None,
                            score=0.91,
                            session_id=session_id,
                        )
                    )
                    break

        return SimpleNamespace(results=results)

    async def get_session_messages(self, session_id: str, limit: int = 1000):
        messages = self.sessions.get(session_id, [])[:limit]
        return SimpleNamespace(messages=messages, total_count=len(messages))

    async def delete(self, session_id: str):
        self.sessions.pop(session_id, None)
        return SimpleNamespace(message="deleted")


class FakeUserClient:
    def __init__(self):
        self.users: set[str] = set()

    async def get(self, user_id: str):
        if user_id not in self.users:
            raise NotFoundError(body={"message": "not found"})
        return SimpleNamespace(user_id=user_id)

    async def add(self, user_id: str | None = None, **kwargs):
        self.users.add(user_id)
        return SimpleNamespace(user_id=user_id)


class FakeZepClient:
    def __init__(self):
        self.memory = FakeMemoryClient()
        self.user = FakeUserClient()
        self._client_wrapper = SimpleNamespace(
            httpx_client=SimpleNamespace(aclose=lambda: asyncio.sleep(0))
        )


def test_factory_defaults_to_simulated_adapter():
    adapter = create_zep_adapter(use_real=False)
    assert adapter.backend_name == "Zep Memory (simulated)"


def test_real_zep_adapter_requires_explicit_endpoint(monkeypatch):
    monkeypatch.delenv("ZEP_API_URL", raising=False)
    with pytest.raises(ValueError, match="ZEP_API_URL"):
        RealZepMemoryAdapter()


def test_real_zep_adapter_uses_env_base_url(monkeypatch):
    captured: dict[str, str | None] = {}

    class CapturingAsyncZep:
        def __init__(self, base_url: str, api_key: str | None):
            captured["base_url"] = base_url
            captured["api_key"] = api_key
            self.memory = FakeMemoryClient()
            self.user = FakeUserClient()
            self._client_wrapper = SimpleNamespace(
                httpx_client=SimpleNamespace(aclose=lambda: asyncio.sleep(0))
            )

    monkeypatch.setenv("ZEP_API_URL", "https://zep.example.com")
    monkeypatch.setenv("ZEP_API_KEY", "secret-key")
    monkeypatch.setattr("benchmarks.zep_real.AsyncZep", CapturingAsyncZep)

    adapter = RealZepMemoryAdapter()

    assert adapter.backend_name == "Zep Memory (real)"
    assert captured["base_url"] == "https://zep.example.com"
    assert captured["api_key"] == "secret-key"


@pytest.mark.asyncio
async def test_real_zep_adapter_with_fake_client():
    client = FakeZepClient()
    adapter = RealZepMemoryAdapter(client=client)

    stored = await adapter.write(
        "Customer wants a refund for order #12345",
        "agent-1",
        "user-1",
        {"category": "refund"},
    )
    assert stored is True
    assert await adapter.count("agent-1", "user-1") == 1

    results = await adapter.retrieve(
        "refund order",
        "agent-1",
        "user-1",
        "billing refund",
        top_k=5,
    )
    assert results
    assert results[0]["content"].startswith("Customer wants a refund")
    assert results[0]["metadata"]["category"] == "refund"

    await adapter.clear("agent-1")
    assert await adapter.count("agent-1", "user-1") == 0
