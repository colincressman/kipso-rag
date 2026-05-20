"""Conversation CRUD endpoints."""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from db.client import (
    archive_conversation,
    archive_stale_conversations,
    create_conversation,
    get_conversation,
    list_conversations,
)
from server.shared import DEFAULT_DB_DSN, _conv_retention_days, _gpu_state, _gpu_busy_response, _gpu_is_tight

router = APIRouter()


@router.get("/api/conversations")
def get_conversations() -> List[Dict[str, Any]]:
    """Return non-archived conversations, most recent first."""
    try:
        archive_stale_conversations(DEFAULT_DB_DSN, days=_conv_retention_days)
    except Exception:
        pass
    try:
        return list_conversations(DEFAULT_DB_DSN)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/api/conversations")
def new_conversation() -> Any:
    """Create a new conversation and return its ID."""
    lock = _gpu_state["lock"]
    if not lock._value and _gpu_is_tight():  # type: ignore[attr-defined]
        return _gpu_busy_response()
    try:
        cid = create_conversation(DEFAULT_DB_DSN)
        return {"conversation_id": cid}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/api/conversations/{conversation_id}")
def load_conversation(conversation_id: str) -> Dict[str, Any]:
    """Return a conversation with its full message history."""
    try:
        conv = get_conversation(DEFAULT_DB_DSN, conversation_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@router.delete("/api/conversations/{conversation_id}", status_code=204)
def remove_conversation(conversation_id: str) -> None:
    """Soft-delete (archive) a conversation."""
    try:
        archive_conversation(DEFAULT_DB_DSN, conversation_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
