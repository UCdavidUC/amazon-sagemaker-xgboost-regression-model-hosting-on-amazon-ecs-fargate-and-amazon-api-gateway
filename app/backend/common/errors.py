"""Typed exceptions shared across the inference backend.

Using explicit exception types lets the API layer map failures to the correct
HTTP status codes and lets the workers decide whether a message should be
retried (transient) or sent straight to the dead-letter queue (permanent).
"""
from __future__ import annotations


class InferenceError(Exception):
    """Base class for all backend errors."""


class InvalidRequestError(InferenceError):
    """The caller supplied an invalid request (maps to HTTP 400).

    These errors are *permanent* - retrying the same payload will fail again,
    so workers must not requeue the message.
    """


class UnknownRouteError(InvalidRequestError):
    """The requested backend or model does not exist (maps to HTTP 404)."""


class ModelLoadError(InferenceError):
    """A model artifact could not be loaded (maps to HTTP 503).

    Treated as *transient*: the artifact may become available (for example the
    S3 object is being replaced during a deployment), so the worker should let
    the message return to the queue and retry.
    """


class RequestNotFoundError(InferenceError):
    """No DynamoDB record exists for the supplied request id (HTTP 404)."""
