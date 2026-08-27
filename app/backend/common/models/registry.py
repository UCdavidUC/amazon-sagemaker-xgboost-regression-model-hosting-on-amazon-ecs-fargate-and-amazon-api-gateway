"""Model registry: maps a model name to a cached model instance.

Instances are cached per process so that an artifact is loaded at most once per
Lambda execution environment or ECS worker, then reused across many requests.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

from ..config import (
    MODEL_ARIMA,
    MODEL_SARIMA,
    MODEL_WEIGHTED,
    MODEL_XGBOOST,
    VALID_MODELS,
    Config,
)
from ..errors import UnknownRouteError
from .arima import ArimaModel
from .base import InferenceModel
from .sarima import SarimaModel
from .weighted import WeightedModel
from .xgboost_model import XGBoostModel

_FACTORIES: Dict[str, Callable[[Optional[Config]], InferenceModel]] = {
    MODEL_WEIGHTED: WeightedModel,
    MODEL_ARIMA: ArimaModel,
    MODEL_SARIMA: SarimaModel,
    MODEL_XGBOOST: XGBoostModel,
}

_CACHE: Dict[str, InferenceModel] = {}


def get_model(name: str, config: Optional[Config] = None) -> InferenceModel:
    """Return a cached model instance for ``name``.

    Raises :class:`UnknownRouteError` if the name is not a supported model.
    """
    if name not in _FACTORIES:
        raise UnknownRouteError(
            f"Unknown model '{name}'. Valid models: {', '.join(VALID_MODELS)}"
        )
    if name not in _CACHE:
        _CACHE[name] = _FACTORIES[name](config)
    return _CACHE[name]


def reset_cache() -> None:
    """Clear the instance cache (used by tests)."""
    _CACHE.clear()
