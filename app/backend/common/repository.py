"""DynamoDB persistence for request tracking.

Stores one item per inference request keyed by ``request_id`` and moves it
through the QUEUED -> PROCESSING -> COMPLETED/FAILED lifecycle. Floats are
converted to ``Decimal`` on write (DynamoDB requirement) and back to native
numbers on read so callers never deal with ``Decimal``.

The table object is injectable, which keeps the class fully unit-testable with
an in-memory fake and no AWS calls.
"""
from __future__ import annotations

import decimal
from typing import Any, Dict, Optional

from .config import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PROCESSING,
    Config,
    load_config,
)
from .errors import RequestNotFoundError
from .schemas import InferenceRequest, utc_now_iso


def _to_dynamo(value: Any) -> Any:
    """Recursively convert floats to Decimal for DynamoDB."""
    if isinstance(value, float):
        # str() avoids binary float artifacts that Decimal(float) would keep.
        return decimal.Decimal(str(value))
    if isinstance(value, dict):
        return {k: _to_dynamo(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_dynamo(v) for v in value]
    return value


def _from_dynamo(value: Any) -> Any:
    """Recursively convert Decimal back to int/float for callers."""
    if isinstance(value, decimal.Decimal):
        as_int = int(value)
        return as_int if value == as_int else float(value)
    if isinstance(value, dict):
        return {k: _from_dynamo(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_from_dynamo(v) for v in value]
    return value


class DynamoRequestRepository:
    """Read/write access to the request-tracking DynamoDB table."""

    def __init__(
        self,
        table: Any = None,
        config: Optional[Config] = None,
    ) -> None:
        self.config = config or load_config()
        self._table = table

    @property
    def table(self) -> Any:
        if self._table is None:
            import boto3  # imported lazily so tests need no AWS SDK client

            resource = boto3.resource("dynamodb", region_name=self.config.aws_region)
            if not self.config.table_name:
                raise ValueError("REQUESTS_TABLE_NAME is not configured")
            self._table = resource.Table(self.config.table_name)
        return self._table

    # -- writes -----------------------------------------------------------
    def save(self, request: InferenceRequest) -> None:
        """Persist a brand-new request item (idempotent create)."""
        self.table.put_item(Item=_to_dynamo(request.to_item()))

    def mark_processing(self, request_id: str) -> None:
        self._update(
            request_id,
            "SET #s = :s, updated_at = :u ADD attempts :one",
            {":s": STATUS_PROCESSING, ":u": utc_now_iso(), ":one": 1},
        )

    def mark_completed(self, request_id: str, output: Dict[str, Any]) -> None:
        self._update(
            request_id,
            "SET #s = :s, updated_at = :u, output = :o REMOVE #e",
            {":s": STATUS_COMPLETED, ":u": utc_now_iso(), ":o": _to_dynamo(output)},
            extra_names={"#e": "error"},
        )

    def mark_failed(self, request_id: str, error: str) -> None:
        self._update(
            request_id,
            "SET #s = :s, updated_at = :u, #e = :err",
            {":s": STATUS_FAILED, ":u": utc_now_iso(), ":err": str(error)[:1024]},
            extra_names={"#e": "error"},
        )

    def _update(
        self,
        request_id: str,
        expression: str,
        values: Dict[str, Any],
        extra_names: Optional[Dict[str, str]] = None,
    ) -> None:
        names = {"#s": "status"}
        if extra_names:
            names.update(extra_names)
        self.table.update_item(
            Key={"request_id": request_id},
            UpdateExpression=expression,
            ExpressionAttributeNames=names,
            ExpressionAttributeValues=values,
        )

    # -- reads ------------------------------------------------------------
    def get(self, request_id: str) -> Dict[str, Any]:
        response = self.table.get_item(Key={"request_id": request_id})
        item = response.get("Item")
        if item is None:
            raise RequestNotFoundError(f"No request found with id '{request_id}'")
        return _from_dynamo(item)
