"""In-memory fakes for unit tests (no AWS calls)."""
from __future__ import annotations

from typing import Any, Dict, List

from backend.common.errors import RequestNotFoundError
from backend.common.schemas import InferenceRequest, QueueMessage


class FakeRepository:
    """Records lifecycle calls and stores items in a dict."""

    def __init__(self) -> None:
        self.items: Dict[str, Dict[str, Any]] = {}
        self.calls: List[str] = []

    def save(self, request: InferenceRequest) -> None:
        self.calls.append("save")
        self.items[request.request_id] = request.to_item()

    def mark_processing(self, request_id: str) -> None:
        self.calls.append("mark_processing")
        self.items.setdefault(request_id, {})["status"] = "PROCESSING"

    def mark_completed(self, request_id: str, output: Dict[str, Any]) -> None:
        self.calls.append("mark_completed")
        item = self.items.setdefault(request_id, {})
        item["status"] = "COMPLETED"
        item["output"] = output

    def mark_failed(self, request_id: str, error: str) -> None:
        self.calls.append("mark_failed")
        item = self.items.setdefault(request_id, {})
        item["status"] = "FAILED"
        item["error"] = error

    def get(self, request_id: str) -> Dict[str, Any]:
        if request_id not in self.items:
            raise RequestNotFoundError(request_id)
        return self.items[request_id]


class FakePublisher:
    def __init__(self) -> None:
        self.published: List[QueueMessage] = []

    def publish(self, message: QueueMessage) -> str:
        self.published.append(message)
        return "fake-message-id"


class FakeTable:
    """Minimal DynamoDB Table stand-in capturing calls."""

    def __init__(self) -> None:
        self.store: Dict[str, Dict[str, Any]] = {}
        self.update_calls: List[Dict[str, Any]] = []

    def put_item(self, Item: Dict[str, Any]) -> None:  # noqa: N803 - boto3 kw
        self.store[Item["request_id"]] = dict(Item)

    def get_item(self, Key: Dict[str, Any]) -> Dict[str, Any]:  # noqa: N803
        item = self.store.get(Key["request_id"])
        return {"Item": item} if item is not None else {}

    def update_item(self, **kwargs: Any) -> None:
        self.update_calls.append(kwargs)
