"""OpenRouter-backed chat models for the agent."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_URL = "https://openrouter.ai/api/v1"
TOOL_MODEL = "meta-llama/llama-3.3-70b-instruct"  # reliable at tool calls
FAST_MODEL = "meta-llama/llama-3.1-8b-instruct"  # cheap, routing / plain chat

load_dotenv(PROJECT_ROOT / ".env")


def get_llm(model: str, temperature: float = 0, max_tokens: int | None = None) -> ChatOpenAI:
    """max_tokens caps the reply length; without it only the provider's (very large) limit applies."""
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set. Add it to the project .env file.")
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        base_url=BASE_URL,
        api_key=api_key,
        max_tokens=max_tokens,
    )


def get_tool_llm(max_tokens: int | None = None) -> ChatOpenAI:
    return get_llm(TOOL_MODEL, max_tokens=max_tokens)


def get_fast_llm() -> ChatOpenAI:
    return get_llm(FAST_MODEL)
