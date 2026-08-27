"""AWS Lambda entrypoint for the API Gateway integration.

Referenced by CloudFormation as ``backend.api.handler.lambda_handler``.

The :class:`InferenceService` is created once per execution environment and
reused across invocations so the DynamoDB/SQS clients and model instances stay
warm.
"""
from __future__ import annotations

import logging
import os

from ..common.inference_service import InferenceService
from .router import route

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

# Module-level singleton: initialized on cold start, reused while warm.
_SERVICE = InferenceService()


def lambda_handler(event, context):  # noqa: ANN001 - AWS signature
    return route(event, _SERVICE)
