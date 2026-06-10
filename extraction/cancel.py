"""Cooperative cancellation helpers for extraction runs."""

from __future__ import annotations

from typing import Callable, Optional


class ExtractionCancelled(RuntimeError):
    """Raised when a user requests cancellation of an active extraction run."""


CancelCheck = Optional[Callable[[], bool]]


def raise_if_cancelled(cancel_check: CancelCheck) -> None:
    """Raise ExtractionCancelled when the provided checker reports cancellation."""
    if cancel_check and cancel_check():
        raise ExtractionCancelled("Run aborted by user.")
