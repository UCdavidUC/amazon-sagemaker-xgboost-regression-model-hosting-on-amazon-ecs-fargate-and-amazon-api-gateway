"""Base contract shared by every inference model.

A model is responsible for two things:

1. ``validate_input`` - turn the raw JSON body into a normalized, typed payload
   and reject anything malformed with :class:`InvalidRequestError`.
2. ``predict`` - run inference on the normalized payload and return a plain
   ``dict`` that is safe to store in DynamoDB and return as JSON.

Model artifacts (pickled estimators, fitted statsmodels results, ...) are
loaded lazily by ``ensure_loaded`` so that importing the package is cheap and
unit tests can exercise validation without any heavy dependency present.
"""
from __future__ import annotations

import abc
import os
from typing import Any, Dict, Optional

from ..config import Config
from ..errors import ModelLoadError


class InferenceModel(abc.ABC):
    """Abstract base class for all models exposed through the API."""

    #: Logical model name, e.g. ``"xgboost"``. Set by subclasses.
    name: str = "base"

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config
        self._loaded = False
        self._artifact: Any = None

    # -- artifact loading -------------------------------------------------
    def artifact_path(self) -> Optional[str]:
        """Return the local filesystem path of the model artifact, if any.

        Looks in the configured ``model_dir`` for ``<name>-model.pkl``. Returns
        ``None`` when no artifact is present, in which case the model falls
        back to its deterministic built-in behaviour.
        """
        if not self.config:
            return None
        candidate = os.path.join(self.config.model_dir, f"{self.name}-model.pkl")
        return candidate if os.path.exists(candidate) else None

    def ensure_loaded(self) -> None:
        """Load the artifact exactly once (idempotent)."""
        if self._loaded:
            return
        try:
            self._artifact = self._load_artifact()
        except ModelLoadError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise ModelLoadError(
                f"Failed to load artifact for model '{self.name}': {exc}"
            ) from exc
        self._loaded = True

    def _load_artifact(self) -> Any:
        """Subclass hook to load a model artifact. Default: no artifact."""
        return None

    # -- inference contract ----------------------------------------------
    @abc.abstractmethod
    def validate_input(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize the request body.

        Must raise :class:`InvalidRequestError` for bad input and return a
        JSON-serializable dict on success.
        """

    @abc.abstractmethod
    def predict(self, normalized_input: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference on a previously validated payload."""

    # -- convenience ------------------------------------------------------
    def run(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Validate, ensure the artifact is loaded, then predict."""
        normalized = self.validate_input(body)
        self.ensure_loaded()
        return self.predict(normalized)
