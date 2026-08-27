"""Shared inference backend package.

This package holds all the code that is common to both compute backends
(AWS Lambda and Amazon ECS Fargate):

* ``config``            - environment-driven configuration.
* ``errors``            - typed exceptions used across the stack.
* ``schemas``           - request validation and DynamoDB record shaping.
* ``models``            - pluggable inference model implementations.
* ``repository``        - DynamoDB persistence for request tracking.
* ``queue``             - SQS producer helper.
* ``inference_service`` - orchestration used by the API and the workers.

Keeping this logic in one place guarantees that a request processed on the
Lambda backend and the same request processed on the ECS Fargate backend
produce identical results and identical DynamoDB records.
"""

__all__ = [
    "config",
    "errors",
    "schemas",
    "models",
    "repository",
    "queue",
    "inference_service",
]
