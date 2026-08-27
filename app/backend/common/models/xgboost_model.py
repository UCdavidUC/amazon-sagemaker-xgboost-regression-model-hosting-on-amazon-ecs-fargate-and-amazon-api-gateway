"""XGBoost regression model for California Housing.

Wraps the trained XGBoost booster produced by the SageMaker training job in the
notebook. The heavy ``xgboost`` dependency is imported lazily inside
``_load_artifact`` / ``predict`` so that importing this module (for validation,
routing, or unit tests) does not require the library to be installed.

When no artifact is available the model raises :class:`ModelLoadError` at
prediction time - unlike the weighted baseline, an XGBoost prediction is not
meaningful without a trained booster.
"""
from __future__ import annotations

import pickle
from typing import Any, Dict, Optional

from ..errors import ModelLoadError
from .base import InferenceModel
from .features import CA_HOUSING_FEATURES, parse_features


class XGBoostModel(InferenceModel):
    name = "xgboost"

    def _load_artifact(self) -> Optional[Any]:
        path = self.artifact_path()
        if not path:
            return None
        with open(path, "rb") as handle:
            return pickle.load(handle)

    def validate_input(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return {"features": parse_features(body)}

    def predict(self, normalized_input: Dict[str, Any]) -> Dict[str, Any]:
        if self._artifact is None:
            raise ModelLoadError(
                "No XGBoost artifact loaded. Provide xgboost-model.pkl via the "
                "model directory or S3 model bucket."
            )
        try:
            import xgboost as xgb  # lazy import - heavy native dependency
        except ImportError as exc:  # pragma: no cover - depends on runtime image
            raise ModelLoadError(f"xgboost is not installed: {exc}") from exc

        dmatrix = xgb.DMatrix([normalized_input["features"]])
        prediction = float(self._artifact.predict(dmatrix)[0])
        return {
            "model": self.name,
            "prediction": prediction,
            "feature_order": CA_HOUSING_FEATURES,
        }
