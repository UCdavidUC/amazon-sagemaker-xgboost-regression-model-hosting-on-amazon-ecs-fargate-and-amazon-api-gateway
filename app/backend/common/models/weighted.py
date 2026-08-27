"""Weighted linear baseline model for California Housing.

The ``weighted`` model is a transparent, dependency-light regressor: it takes a
dot product of the standardized feature vector with a weight vector and adds an
intercept. It serves two purposes in this solution:

* It is a fast, always-available baseline that needs no heavy ML runtime, which
  makes it ideal for smoke-testing both the Lambda and the ECS Fargate paths.
* It doubles as the ensemble-weighting primitive: supply ``weights`` in the
  request body to score a custom linear combination of the features.

If a pickled artifact (``weighted-model.pkl`` with ``weights`` and
``intercept``) is present it is used; otherwise sensible built-in coefficients
derived from the California Housing feature ordering are applied.
"""
from __future__ import annotations

import pickle
from typing import Any, Dict, List, Optional

from ..errors import InvalidRequestError
from .base import InferenceModel
from .features import CA_HOUSING_FEATURES, NUM_FEATURES, parse_features

# Default weights: emphasise median income and rooms, penalise nothing heavily.
# These are illustrative coefficients for a standardized feature space, not a
# fitted model. Replace by shipping a weighted-model.pkl artifact.
_DEFAULT_WEIGHTS: List[float] = [0.82, 0.12, 0.11, -0.09, -0.04, -0.03, -0.35, -0.31]
_DEFAULT_INTERCEPT: float = 2.07  # median house value in $100k units, roughly


class WeightedModel(InferenceModel):
    name = "weighted"

    def _load_artifact(self) -> Optional[Dict[str, Any]]:
        path = self.artifact_path()
        if not path:
            return None
        with open(path, "rb") as handle:
            return pickle.load(handle)

    def validate_input(self, body: Dict[str, Any]) -> Dict[str, Any]:
        features = parse_features(body)
        weights: Optional[List[float]] = None
        if body.get("weights") is not None:
            raw = body["weights"]
            if not isinstance(raw, (list, tuple)) or len(raw) != NUM_FEATURES:
                raise InvalidRequestError(
                    f"'weights' must be an array of {NUM_FEATURES} numbers"
                )
            try:
                weights = [float(w) for w in raw]
            except (TypeError, ValueError):
                raise InvalidRequestError("All 'weights' values must be numeric")
        return {"features": features, "weights": weights}

    def predict(self, normalized_input: Dict[str, Any]) -> Dict[str, Any]:
        # Pure-Python scoring: no numpy required, so the weighted model runs on
        # the Lambda backend (which ships no scientific stack) as well as on ECS.
        features = normalized_input["features"]

        if normalized_input.get("weights") is not None:
            weights = normalized_input["weights"]
            intercept = 0.0
            source = "request-weights"
        elif isinstance(self._artifact, dict):
            weights = [float(w) for w in self._artifact["weights"]]
            intercept = float(self._artifact.get("intercept", 0.0))
            source = "artifact"
        else:
            weights = _DEFAULT_WEIGHTS
            intercept = _DEFAULT_INTERCEPT
            source = "default-weights"

        prediction = float(sum(f * w for f, w in zip(features, weights)) + intercept)
        return {
            "model": self.name,
            "prediction": prediction,
            "weight_source": source,
            "feature_order": CA_HOUSING_FEATURES,
        }
