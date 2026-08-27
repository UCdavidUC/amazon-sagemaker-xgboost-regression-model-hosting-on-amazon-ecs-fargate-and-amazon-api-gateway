"""Request/record schemas and helpers.

Defines the wire contract for a submitted inference request and the shape of
the DynamoDB record that tracks it through its lifecycle.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .config import (
    STATUS_QUEUED,
    VALID_BACKENDS,
    VALID_MODELS,
)
from .errors import InvalidRequestError, UnknownRouteError


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_request_id() -> str:
    return uuid.uuid4().hex


def validate_route(backend: str, model: str) -> None:
    """Ensure the backend and model in the path are supported."""
    if backend not in VALID_BACKENDS:
        raise UnknownRouteError(
            f"Unknown backend '{backend}'. Valid backends: {', '.join(VALID_BACKENDS)}"
        )
    if model not in VALID_MODELS:
        raise UnknownRouteError(
            f"Unknown model '{model}'. Valid models: {', '.join(VALID_MODELS)}"
        )


def validate_body(body: Any) -> Dict[str, Any]:
    """Ensure the request body is a JSON object."""
    if body is None:
        raise InvalidRequestError("Request body is required")
    if not isinstance(body, dict):
        raise InvalidRequestError("Request body must be a JSON object")
    return body


@dataclass
class InferenceRequest:
    """A submitted request as it is enqueued and persisted."""

    request_id: str
    backend: str
    model: str
    status: str
    input: Dict[str, Any]
    created_at: str
    updated_at: str
    ttl: int
    environment: str
    attempts: int = 0
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    client_request_id: Optional[str] = None

    @classmethod
    def create(
        cls,
        backend: str,
        model: str,
        input_body: Dict[str, Any],
        environment: str,
        ttl_days: int,
        client_request_id: Optional[str] = None,
    ) -> "InferenceRequest":
        now = utc_now_iso()
        ttl = int(time.time()) + ttl_days * 24 * 3600
        return cls(
            request_id=new_request_id(),
            backend=backend,
            model=model,
            status=STATUS_QUEUED,
            input=input_body,
            created_at=now,
            updated_at=now,
            ttl=ttl,
            environment=environment,
            client_request_id=client_request_id,
        )

    def to_item(self) -> Dict[str, Any]:
        """Serialize to a DynamoDB-friendly dict (drops None values)."""
        item = asdict(self)
        return {k: v for k, v in item.items() if v is not None}

    def public_view(self) -> Dict[str, Any]:
        """The representation returned to API callers."""
        view = {
            "request_id": self.request_id,
            "backend": self.backend,
            "model": self.model,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self.output is not None:
            view["output"] = self.output
        if self.error is not None:
            view["error"] = self.error
        return view


@dataclass
class QueueMessage:
    """The payload placed on SQS for a worker to process."""

    request_id: str
    backend: str
    model: str
    input: Dict[str, Any]

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "backend": self.backend,
            "model": self.model,
            "input": self.input,
        }

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "QueueMessage":
        missing = [k for k in ("request_id", "backend", "model", "input") if k not in data]
        if missing:
            raise InvalidRequestError(f"Queue message missing fields: {missing}")
        return cls(
            request_id=data["request_id"],
            backend=data["backend"],
            model=data["model"],
            input=data["input"],
        )
