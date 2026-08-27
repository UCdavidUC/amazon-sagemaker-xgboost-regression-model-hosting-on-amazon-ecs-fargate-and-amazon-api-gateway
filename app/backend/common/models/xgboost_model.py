"""XGBoost regression model for California Housing.

Wraps the trained XGBoost booster produced by the SageMaker training job in the
notebook. The heavy ``xgboost`` dependency is imported lazily so importing this
module (for validation, routing, or unit tests) does not require the library.

Artifact format
----------------
The preferred artifact is XGBoost's **portable model format**
(``xgboost-model.ubj`` or ``xgboost-model.json``), loaded with
``Booster.load_model``. That format is forward-compatible across XGBoost
versions, so a model produced by the SageMaker built-in XGBoost (e.g. 1.7) loads
cleanly in the serving runtime (2.x) - unlike a pickled ``Booster``, which is
not guaranteed compatible across major versions.

A legacy pickled booster (``xgboost-model.pkl``) is still accepted as a fallback
for backward compatibility.

When no artifact is available the model raises :class:`ModelLoadError` at
prediction time - an XGBoost prediction is not meaningful without a booster.
"""
from __future__ import annotations

import os
import pickle
from typing import Any, Dict, Optional, Tuple

from ..errors import ModelLoadError
from .base import InferenceModel
from .features import CA_HOUSING_FEATURES, parse_features

# Preferred portable formats, in priority order, then the legacy pickle.
_NATIVE_EXTENSIONS = ("ubj", "json")
_LEGACY_PICKLE = "xgboost-model.pkl"


class XGBoostModel(InferenceModel):
    name = "xgboost"

    def _discover_artifact(self) -> Tuple[Optional[str], Optional[str]]:
        """Return (kind, path) for the artifact, or (None, None) if absent.

        ``kind`` is ``"native"`` for a save_model file or ``"pickle"`` for a
        legacy pickled booster.
        """
        if not self.config:
            return None, None
        model_dir = self.config.model_dir
        for ext in _NATIVE_EXTENSIONS:
            candidate = os.path.join(model_dir, f"xgboost-model.{ext}")
            if os.path.exists(candidate):
                return "native", candidate
        legacy = os.path.join(model_dir, _LEGACY_PICKLE)
        if os.path.exists(legacy):
            return "pickle", legacy
        return None, None

    def _load_artifact(self) -> Optional[Any]:
        kind, path = self._discover_artifact()
        if not path:
            return None
        if kind == "native":
            try:
                import xgboost as xgb  # lazy import - heavy native dependency
            except ImportError as exc:  # pragma: no cover - depends on image
                raise ModelLoadError(f"xgboost is not installed: {exc}") from exc
            booster = xgb.Booster()
            booster.load_model(path)
            return booster
        # Legacy pickled booster.
        with open(path, "rb") as handle:
            return pickle.load(handle)

    def validate_input(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return {"features": parse_features(body)}

    def predict(self, normalized_input: Dict[str, Any]) -> Dict[str, Any]:
        if self._artifact is None:
            raise ModelLoadError(
                "No XGBoost artifact loaded. Provide xgboost-model.ubj "
                "(preferred) or xgboost-model.pkl via the model directory or "
                "S3 model bucket."
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
