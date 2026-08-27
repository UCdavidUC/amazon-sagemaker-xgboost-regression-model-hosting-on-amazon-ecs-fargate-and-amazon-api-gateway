"""Shared paths and constants for the CDK stacks."""
import os

# stacks/ -> cdk/ -> app/   (this is the `backend` package parent + Docker context)
APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# When packaging the Lambda code asset we ship only the `backend/` package so
# the handlers resolve as backend.api.handler.lambda_handler etc. Everything
# else in app/ is excluded.
LAMBDA_ASSET_EXCLUDES = [
    "cdk",
    "tests",
    "cicd",
    "infrastructure",
    "data",
    "frontend",
    "notebooks",
    # Baked ECS model artifacts are large and only used by the ECS image
    # (the Lambda worker reads /opt/models, not this path) - keep them out.
    "backend/ecs_worker/models",
    "*.md",
    "__pycache__",
    "**/__pycache__",
    "*.pyc",
    ".DS_Store",
    ".pytest_cache",
]

# ECS worker Dockerfile, relative to APP_DIR (the build context).
ECS_DOCKERFILE = os.path.join("backend", "ecs_worker", "Dockerfile")
