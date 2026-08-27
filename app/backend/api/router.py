"""Framework-agnostic request router for the API Gateway Lambda.

Kept separate from the Lambda entrypoint so it can be unit-tested with plain
dicts and a stubbed :class:`InferenceService` - no AWS, no API Gateway.

Supported routes (all under the stage base path ``/api``):

    GET  /api                                  -> operation catalog
    GET  /api/health                           -> liveness probe
    POST /api/backend/{backend}/model/{model}  -> submit an inference request
    GET  /api/requests/{request_id}            -> fetch request status/result

Both API Gateway REST (payload v1) and HTTP API (payload v2) event shapes are
accepted so the same Lambda works with either integration.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from ..common.config import VALID_BACKENDS, VALID_MODELS
from ..common.errors import (
    InvalidRequestError,
    RequestNotFoundError,
    UnknownRouteError,
)
from ..common.inference_service import InferenceService

logger = logging.getLogger(__name__)


def _response(status: int, body: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            # Defence in depth; API Gateway also enforces TLS in transit.
            "Strict-Transport-Security": "max-age=63072000; includeSubDomains",
            "X-Content-Type-Options": "nosniff",
        },
        "body": json.dumps(body),
    }


def _method(event: Dict[str, Any]) -> str:
    if "httpMethod" in event:  # REST API (v1)
        return event["httpMethod"]
    # HTTP API (v2)
    return event.get("requestContext", {}).get("http", {}).get("method", "GET")


def _path(event: Dict[str, Any]) -> str:
    # v1 uses "path"; v2 uses "rawPath". Strip any stage prefix already removed
    # by API Gateway; keep the leading "/api" that our resources define.
    return event.get("path") or event.get("rawPath") or "/"


def _parse_body(event: Dict[str, Any]) -> Any:
    raw = event.get("body")
    if raw is None or raw == "":
        return None
    if event.get("isBase64Encoded"):
        import base64

        raw = base64.b64decode(raw).decode("utf-8")
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        raise InvalidRequestError("Request body must be valid JSON")


def _segments(path: str) -> List[str]:
    return [s for s in path.split("/") if s]


def build_catalog() -> Dict[str, Any]:
    """The JSON catalog returned by ``GET /api``."""
    operations: List[Dict[str, str]] = []
    for backend in VALID_BACKENDS:
        for model in VALID_MODELS:
            operations.append(
                {
                    "method": "POST",
                    "path": f"/api/backend/{backend}/model/{model}",
                    "backend": backend,
                    "model": model,
                    "description": f"Submit an inference request to the {model} "
                    f"model on the {backend} backend.",
                }
            )
    operations.append(
        {
            "method": "GET",
            "path": "/api/requests/{request_id}",
            "description": "Fetch the status, input, and output of a request.",
        }
    )
    return {
        "service": "multi-model-inference-api",
        "backends": list(VALID_BACKENDS),
        "models": list(VALID_MODELS),
        "operations": operations,
    }


def route(event: Dict[str, Any], service: InferenceService) -> Dict[str, Any]:
    """Dispatch an API Gateway event to the right handler."""
    method = _method(event)
    segments = _segments(_path(event))

    try:
        return _dispatch(method, segments, event, service)
    except (UnknownRouteError, RequestNotFoundError) as exc:
        return _response(404, {"error": str(exc)})
    except InvalidRequestError as exc:
        return _response(400, {"error": str(exc)})
    except Exception as exc:  # pragma: no cover - defensive catch-all
        logger.exception("Unhandled error")
        return _response(500, {"error": "Internal server error", "detail": str(exc)})


def _dispatch(
    method: str,
    segments: List[str],
    event: Dict[str, Any],
    service: InferenceService,
) -> Dict[str, Any]:
    # segments always start with "api" for our resources; tolerate its absence.
    if segments and segments[0] == "api":
        segments = segments[1:]

    # GET /api
    if method == "GET" and not segments:
        return _response(200, build_catalog())

    # GET /api/health
    if method == "GET" and segments == ["health"]:
        return _response(200, {"status": "ok"})

    # GET /api/requests/{request_id}
    if method == "GET" and len(segments) == 2 and segments[0] == "requests":
        record = service.get_status(segments[1])
        return _response(200, _public_record(record))

    # POST /api/backend/{backend}/model/{model}
    if (
        method == "POST"
        and len(segments) == 4
        and segments[0] == "backend"
        and segments[2] == "model"
    ):
        backend, model = segments[1], segments[3]
        body = _parse_body(event)
        client_request_id = _client_request_id(event)
        request = service.submit(backend, model, body, client_request_id)
        return _response(
            202,
            {
                "request_id": request.request_id,
                "status": request.status,
                "backend": backend,
                "model": model,
                "status_url": f"/api/requests/{request.request_id}",
            },
        )

    raise UnknownRouteError(f"No route for {method} /{'/'.join(segments)}")


def _client_request_id(event: Dict[str, Any]) -> Optional[str]:
    headers = event.get("headers") or {}
    # Header names are case-insensitive; normalize.
    for key, value in headers.items():
        if key.lower() == "x-client-request-id":
            return value
    return None


def _public_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Trim the raw DynamoDB item to the fields callers should see."""
    fields = (
        "request_id",
        "backend",
        "model",
        "status",
        "created_at",
        "updated_at",
        "input",
        "output",
        "error",
    )
    return {k: record[k] for k in fields if k in record}
