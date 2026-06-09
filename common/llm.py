"""Shared LLM factory for all agents.

Uses OpenRouter as an OpenAI-compatible API, so any provider's model
can be selected via the OPENROUTER_MODEL env var.
"""

import os

from langchain_openai import ChatOpenAI


def get_llm() -> ChatOpenAI:
    """Return a ChatOpenAI client pointed at OpenRouter."""
    model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5").strip()
    api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()

    return ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        max_tokens=500,
        temperature=0.3,
    )
