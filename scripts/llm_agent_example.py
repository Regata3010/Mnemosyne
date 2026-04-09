#!/usr/bin/env python3
"""
LLM agent example using Mnemosyne memory.

Default mode uses the in-memory Mnemosyne simulator so the demo works out of the box.
Pass --live-api to route memory calls through the real Mnemosyne SDK/server.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.simulated_mnemosyne import create_simulated_mnemosyne
from src.core.models import MemoryType
from src.sdk.client import MnemosyneClient


@dataclass
class TurnResult:
    """Result of a single agent turn."""

    stored: bool
    memories: list[dict[str, Any]]
    reply: str
    task_context: str


class MemoryBackend(Protocol):
    """Memory backend contract used by the demo agent."""

    async def remember(
        self,
        content: str,
        user_id: str,
        metadata: dict[str, Any],
        importance_hint: float | None = None,
    ) -> bool:
        ...

    async def recall(
        self,
        query: str,
        user_id: str,
        task_context: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        ...

    async def count(self, user_id: str | None = None) -> int:
        ...

    async def health(self) -> bool:
        ...

    async def close(self) -> None:
        ...


class SimulatedMemoryBackend:
    """In-memory backend for the demo."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.adapter = create_simulated_mnemosyne(
            use_real_embeddings=True,
            importance_threshold=0.32,
        )

    async def remember(
        self,
        content: str,
        user_id: str,
        metadata: dict[str, Any],
        importance_hint: float | None = None,
    ) -> bool:
        _ = importance_hint
        return await self.adapter.write(
            content=content,
            agent_id=self.agent_id,
            user_id=user_id,
            metadata=metadata,
        )

    async def recall(
        self,
        query: str,
        user_id: str,
        task_context: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        return await self.adapter.retrieve(
            query=query,
            agent_id=self.agent_id,
            user_id=user_id,
            task_context=task_context,
            top_k=top_k,
        )

    async def count(self, user_id: str | None = None) -> int:
        return await self.adapter.count(self.agent_id, user_id)

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class LiveMemoryBackend:
    """Backend that talks to the real Mnemosyne API."""

    def __init__(self, agent_id: str, base_url: str):
        self.client = MnemosyneClient(agent_id=agent_id, base_url=base_url)

    async def remember(
        self,
        content: str,
        user_id: str,
        metadata: dict[str, Any],
        importance_hint: float | None = None,
    ) -> bool:
        result = await self.client.remember(
            content,
            user_id=user_id,
            memory_type=MemoryType.EPISODIC,
            metadata=metadata,
            importance_hint=importance_hint,
        )
        return result.stored

    async def recall(
        self,
        query: str,
        user_id: str,
        task_context: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        results = await self.client.recall(
            query,
            user_id=user_id,
            task_context=task_context,
            top_k=top_k,
        )
        return [
            {
                "content": item.memory.content,
                "user_id": item.memory.user_id,
                "metadata": item.memory.metadata,
                "relevance_score": item.relevance_score,
            }
            for item in results
        ]

    async def count(self, user_id: str | None = None) -> int:
        return await self.client.count(user_id=user_id)

    async def health(self) -> bool:
        return await self.client.health()

    async def close(self) -> None:
        await self.client.close()


class MockLLMProvider:
    """Deterministic fallback so the demo runs without API keys."""

    name = "mock"

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt
        memory_lines = [
            line.strip()[2:].strip()
            for line in user_prompt.splitlines()
            if line.startswith("- ")
        ]
        memory_hint = memory_lines[0] if memory_lines else "no prior memory"
        issue = "the customer's request"
        marker = "Customer message: "
        if marker in user_prompt:
            issue = user_prompt.split(marker, 1)[1].split("\n\n", 1)[0].strip()
        return (
            f"I found prior context: {memory_hint}. "
            f"For {issue}, I would acknowledge the issue, confirm the order details, "
            "and prioritize a refund or replacement."
        )


class OpenAIProvider:
    """OpenAI-backed LLM provider."""

    def __init__(self, model: str):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI()
        self.model = model
        self.name = f"openai:{model}"

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("OpenAI returned an empty response")
        return content.strip()


def infer_task_context(message: str) -> str:
    """Map a customer message to a rough task context."""
    text = message.lower()
    if any(word in text for word in ("refund", "chargeback", "billing", "charged")):
        return "billing refund"
    if any(word in text for word in ("shipping", "delivery", "package", "tracking", "arrived")):
        return "shipping issue"
    if any(word in text for word in ("cancel", "subscription", "membership")):
        return "cancellation"
    if any(word in text for word in ("password", "login", "account", "access")):
        return "account access"
    return "general support"


def importance_hint(message: str) -> float:
    """Rough hint to help the demo prioritize important turns."""
    text = message.lower()
    if any(word in text for word in ("refund", "cancel", "damaged", "broken", "missing", "urgent")):
        return 0.9
    if any(word in text for word in ("hi", "hello", "thanks", "thank you")):
        return 0.1
    return 0.5


class MemoryAwareAgent:
    """Minimal agent that reads/writes Mnemosyne memory around LLM calls."""

    def __init__(self, memory: MemoryBackend, llm: Any):
        self.memory = memory
        self.llm = llm

    @staticmethod
    def _format_memories(memories: list[dict[str, Any]]) -> str:
        customer_memories = [
            item
            for item in memories
            if item.get("metadata", {}).get("role") == "customer"
        ]

        if not customer_memories:
            return "No prior memory."

        lines = []
        for item in customer_memories[:4]:
            lines.append(
                f"- ({item.get('relevance_score', 0.0):.2f}) {item.get('content', '')}"
            )
        return "\n".join(lines)

    async def respond(self, user_id: str, message: str) -> TurnResult:
        task_context = infer_task_context(message)
        stored = await self.memory.remember(
            content=message,
            user_id=user_id,
            metadata={
                "role": "customer",
                "task_context": task_context,
            },
            importance_hint=importance_hint(message),
        )

        memories = await self.memory.recall(
            query=message,
            user_id=user_id,
            task_context=task_context,
            top_k=4,
        )
        prompt_memories = [
            item
            for item in memories
            if item.get("metadata", {}).get("role") == "customer"
        ]
        if not prompt_memories:
            prompt_memories = memories

        system_prompt = (
            "You are a helpful customer support agent.\n"
            "Use prior memory only when it is relevant and accurate.\n"
            "Never invent details that are not present in the memory or the user message.\n"
            "Keep the response concise, specific, and empathetic."
        )
        user_prompt = (
            f"Relevant memory:\n{self._format_memories(prompt_memories)}\n\n"
            f"Customer message: {message}\n\n"
            "Write a helpful response."
        )

        reply = await self.llm.complete(system_prompt, user_prompt)

        return TurnResult(
            stored=stored,
            memories=memories,
            reply=reply,
            task_context=task_context,
        )


def build_llm_provider(mock_llm: bool, model: str) -> Any:
    """Choose OpenAI when available, otherwise fall back to a mock response."""
    if mock_llm or not os.getenv("OPENAI_API_KEY"):
        return MockLLMProvider()
    return OpenAIProvider(model=model)


async def run_demo(live_api: bool, base_url: str, model: str, mock_llm: bool) -> None:
    """Run a short returning-customer demo."""
    agent_id = "llm-agent-demo"
    user_id = "customer-88210"

    memory: MemoryBackend
    if live_api:
        memory = LiveMemoryBackend(agent_id=agent_id, base_url=base_url)
    else:
        memory = SimulatedMemoryBackend(agent_id=agent_id)

    llm = build_llm_provider(mock_llm=mock_llm, model=model)
    agent = MemoryAwareAgent(memory=memory, llm=llm)

    try:
        if live_api and not await memory.health():
            raise RuntimeError(f"Mnemosyne API is not healthy at {base_url}")

        print("=" * 72)
        print("LLM AGENT EXAMPLE")
        print("=" * 72)
        print(f"Memory backend: {'live API' if live_api else 'simulated'}")
        print(f"LLM backend:    {llm.name}")
        print(f"Agent ID:       {agent_id}")
        print(f"User ID:        {user_id}")
        print()

        turns = [
            "Hi",
            "My order #88210 arrived damaged and I need a refund.",
            "I'm following up on order #88210. I was charged $199.99, it still hasn't arrived, and I need the refund processed today.",
        ]

        for index, message in enumerate(turns, start=1):
            print(f"Turn {index} | customer: {message}")
            result = await agent.respond(user_id=user_id, message=message)
            print(f"Task context: {result.task_context}")
            print(f"Stored: {result.stored}")
            print("Retrieved memories:")
            print(MemoryAwareAgent._format_memories(result.memories))
            print("Agent reply:")
            print(result.reply)
            print("-" * 72)

        total = await memory.count(user_id=user_id)
        print(f"Final stored memories for {user_id}: {total}")
    finally:
        await memory.close()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run the Mnemosyne LLM agent example.")
    parser.add_argument(
        "--live-api",
        action="store_true",
        help="Use the real Mnemosyne HTTP API instead of the in-memory demo backend.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("MNEMOSYNE_BASE_URL", "http://localhost:8000"),
        help="Base URL for the Mnemosyne API when --live-api is enabled.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        help="OpenAI model name for the live LLM path.",
    )
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="Use deterministic mock responses even when OPENAI_API_KEY is set.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    asyncio.run(
        run_demo(
            live_api=args.live_api,
            base_url=args.base_url,
            model=args.model,
            mock_llm=args.mock_llm,
        )
    )


if __name__ == "__main__":
    main()
