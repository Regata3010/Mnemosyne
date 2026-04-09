"""Real Zep adapter used when the Zep service is configured."""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any

from zep_python import Message, NotFoundError
from zep_python.client import AsyncZep


def _truthy(value: str | None) -> bool:
    return value is not None and value.lower() in {"1", "true", "yes", "on"}


def should_use_real_zep() -> bool:
    """Return True when the environment asks for the real Zep backend."""
    return _truthy(os.getenv("MNEMOSYNE_USE_REAL_ZEP")) or bool(os.getenv("ZEP_API_URL"))


class RealZepMemoryAdapter:
    """Adapter that talks to a live Zep deployment."""

    backend_name = "Zep Memory (real)"

    def __init__(
        self,
        use_real_embeddings: bool = True,  # Kept for API compatibility with the simulator
        base_url: str | None = None,
        api_key: str | None = None,
        client: AsyncZep | None = None,
    ):
        self._use_real_embeddings = use_real_embeddings
        resolved_base_url = base_url or os.getenv("ZEP_API_URL")
        if client is None and not resolved_base_url:
            raise ValueError("Real Zep requires base_url or ZEP_API_URL to be set.")
        self._client = client or AsyncZep(
            base_url=resolved_base_url,
            api_key=api_key or os.getenv("ZEP_API_KEY") or "local",
        )
        self._base_url = resolved_base_url
        self._agent_sessions: dict[str, set[str]] = defaultdict(set)
        self._message_counts: dict[str, int] = defaultdict(int)

    def _session_id(self, agent_id: str, user_id: str | None) -> str:
        return f"{agent_id}:{user_id or 'anonymous'}"

    async def _ensure_session(self, session_id: str, user_id: str | None, metadata: dict[str, Any]) -> None:
        try:
            await self._client.memory.get_session(session_id)
        except NotFoundError:
            await self._client.memory.add_session(
                session_id=session_id,
                user_id=user_id,
                metadata=metadata,
            )

    async def write(self, content: str, agent_id: str, user_id: str, metadata: dict[str, Any]) -> bool:
        """Write a customer message into a Zep session."""
        session_id = self._session_id(agent_id, user_id)
        session_metadata = {"agent_id": agent_id, "user_id": user_id}
        session_metadata.update(metadata)
        await self._ensure_session(session_id, user_id, session_metadata)

        message = Message(
            content=content,
            role="user",
            role_type="user",
            metadata=session_metadata,
        )
        await self._client.memory.add(
            session_id=session_id,
            messages=[message],
        )

        self._agent_sessions[agent_id].add(session_id)
        self._message_counts[session_id] += 1
        return True

    async def retrieve(
        self,
        query: str,
        agent_id: str,
        user_id: str,
        task_context: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Retrieve matching memories from Zep."""
        search_text = f"{task_context}\n{query}".strip() if task_context else query

        search_kwargs: dict[str, Any] = {
            "text": search_text,
            "limit": top_k,
            "search_scope": "messages",
            "search_type": "similarity",
        }
        if user_id:
            search_kwargs["user_id"] = user_id
        else:
            session_ids = sorted(self._agent_sessions.get(agent_id, set()))
            if session_ids:
                search_kwargs["session_ids"] = session_ids

        response = await self._client.memory.search_sessions(**search_kwargs)
        results = response.results or []

        flattened: list[dict[str, Any]] = []
        for result in results[:top_k]:
            message = result.message
            summary = result.summary
            fact = result.fact

            if message is not None and message.content is not None:
                content = message.content
                result_user_id = user_id or (message.metadata or {}).get("user_id")
                metadata = dict(message.metadata or {})
            elif summary is not None and summary.content is not None:
                content = summary.content
                result_user_id = user_id
                metadata = dict(summary.metadata or {})
            elif fact is not None and fact.fact is not None:
                content = fact.fact
                result_user_id = user_id
                metadata = {}
            else:
                continue

            flattened.append(
                {
                    "content": content,
                    "user_id": result_user_id,
                    "metadata": metadata,
                    "relevance_score": float(result.score or 0.0),
                    "session_id": result.session_id,
                }
            )

        return flattened

    async def count(self, agent_id: str, user_id: str | None = None) -> int:
        """Count messages in Zep for the given scope."""
        if user_id:
            session_id = self._session_id(agent_id, user_id)
            try:
                response = await self._client.memory.get_session_messages(session_id, limit=1000)
            except NotFoundError:
                return 0
            return int(response.total_count or len(response.messages or []))

        total = 0
        for session_id in self._agent_sessions.get(agent_id, set()):
            try:
                response = await self._client.memory.get_session_messages(session_id, limit=1000)
            except NotFoundError:
                continue
            total += int(response.total_count or len(response.messages or []))
        return total

    async def clear(self, agent_id: str) -> None:
        """Delete every session we created for the given agent."""
        for session_id in list(self._agent_sessions.get(agent_id, set())):
            try:
                await self._client.memory.delete(session_id)
            except NotFoundError:
                pass
            self._message_counts.pop(session_id, None)
        self._agent_sessions.pop(agent_id, None)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client._client_wrapper.httpx_client.aclose()


def create_real_zep_adapter(
    use_real_embeddings: bool = True,
    base_url: str | None = None,
    api_key: str | None = None,
    client: AsyncZep | None = None,
) -> RealZepMemoryAdapter:
    """Create a real Zep adapter."""
    return RealZepMemoryAdapter(
        use_real_embeddings=use_real_embeddings,
        base_url=base_url,
        api_key=api_key,
        client=client,
    )
