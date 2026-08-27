"""Environment-driven configuration for the inference backend.

Every value comes from an environment variable so the exact same code can run
unchanged on Lambda (variables set on the function) and on ECS Fargate
(variables set on the task definition). The CloudFormation templates in
``app/infrastructure`` are the source of truth for these variables.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

# Logical names used throughout the codebase and in the API routes.
BACKEND_LAMBDA = "lambda"
BACKEND_ECS = "ecs-fargate"
VALID_BACKENDS = (BACKEND_LAMBDA, BACKEND_ECS)

MODEL_WEIGHTED = "weighted"
MODEL_ARIMA = "arima"
MODEL_SARIMA = "sarima"
MODEL_XGBOOST = "xgboost"
VALID_MODELS = (MODEL_WEIGHTED, MODEL_ARIMA, MODEL_SARIMA, MODEL_XGBOOST)

# Request lifecycle states persisted in DynamoDB.
STATUS_QUEUED = "QUEUED"
STATUS_PROCESSING = "PROCESSING"
STATUS_COMPLETED = "COMPLETED"
STATUS_FAILED = "FAILED"


def _get(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def _get_int(name: str, default: int) -> int:
    raw = _get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Config:
    """Resolved runtime configuration."""

    environment: str = field(default_factory=lambda: _get("ENVIRONMENT", "dev"))
    aws_region: Optional[str] = field(default_factory=lambda: _get("AWS_REGION"))

    # DynamoDB request-tracking table.
    table_name: Optional[str] = field(default_factory=lambda: _get("REQUESTS_TABLE_NAME"))
    # Number of days a request record is retained before the TTL removes it.
    record_ttl_days: int = field(default_factory=lambda: _get_int("RECORD_TTL_DAYS", 30))

    # Per-backend SQS queue URLs (only the queue for the active backend is required).
    lambda_queue_url: Optional[str] = field(default_factory=lambda: _get("LAMBDA_QUEUE_URL"))
    ecs_queue_url: Optional[str] = field(default_factory=lambda: _get("ECS_QUEUE_URL"))

    # Optional S3 location for model artifacts (bucket + key prefix).
    model_bucket: Optional[str] = field(default_factory=lambda: _get("MODEL_BUCKET"))
    model_prefix: str = field(default_factory=lambda: _get("MODEL_PREFIX", "models"))

    # Local directory that model artifacts may already be baked into (container image).
    model_dir: str = field(default_factory=lambda: _get("MODEL_DIR", "/opt/models"))

    log_level: str = field(default_factory=lambda: _get("LOG_LEVEL", "INFO"))

    def queue_url_for(self, backend: str) -> Optional[str]:
        """Return the SQS queue URL for the given backend name."""
        if backend == BACKEND_LAMBDA:
            return self.lambda_queue_url
        if backend == BACKEND_ECS:
            return self.ecs_queue_url
        return None


def load_config() -> Config:
    """Build a :class:`Config` from the current environment.

    A fresh instance is returned on each call so tests can monkeypatch the
    environment and reload without leaking state between cases.
    """
    return Config()
