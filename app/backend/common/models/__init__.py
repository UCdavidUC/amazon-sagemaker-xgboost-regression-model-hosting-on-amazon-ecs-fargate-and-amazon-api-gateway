"""Pluggable inference models exposed through the API.

Adding a new model is a two-step change:

1. Implement a subclass of :class:`~app.backend.common.models.base.InferenceModel`.
2. Register it in :mod:`registry` and add its name to
   :data:`app.backend.common.config.VALID_MODELS`.
"""
from .base import InferenceModel
from .registry import get_model, reset_cache

__all__ = ["InferenceModel", "get_model", "reset_cache"]
