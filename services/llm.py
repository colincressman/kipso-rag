"""LLM service facade.

All answer generation, streaming, and summarization go through here.
server/ and main.py import from this module only — never from llm/ directly.
"""

from __future__ import annotations

from typing import Any, Dict, Generator, List, Optional

from api import llm_answer as _llm_answer


def answer(
    query: str,
    retrieval_result: Dict[str, Any],
    *,
    history: Optional[List[Dict[str, str]]] = None,
    llm_model: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    llm_timeout_seconds: Optional[float] = None,
    llm_temperature: Optional[float] = None,
    config_path: str = "configs/llm.yaml",
) -> Dict[str, Any]:
    """Generate a grounded answer from a retrieval result.

    Returns dict with keys: answer, citations, grounded, mode.
    """
    return _llm_answer(
        query,
        retrieval_result,
        history=history,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_timeout_seconds=llm_timeout_seconds,
        llm_temperature=llm_temperature,
        config_path=config_path,
    )


def chat(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Single-turn chat with an Ollama model. Returns the full response dict."""
    from llm.generation import ollama_chat
    return ollama_chat(model=model, system_prompt=system_prompt, user_prompt=user_prompt, **kwargs)


def stream(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    **kwargs: Any,
) -> Generator[str, None, None]:
    """Streaming chat with an Ollama model. Yields token strings."""
    from llm.generation import ollama_stream
    return ollama_stream(model=model, system_prompt=system_prompt, user_prompt=user_prompt, **kwargs)


def summarize(
    text: str,
    *,
    model: Optional[str] = None,
    max_words: int = 150,
) -> str:
    """Summarize a block of text. Returns the summary string."""
    from llm.summarize import summarize_text
    return summarize_text(text, model=model, max_words=max_words)
