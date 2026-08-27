"""Shared base for the ARIMA and SARIMA time-series forecasting models.

Both models are fitted with statsmodels and persisted as a pickled
``*ResultsWrapper`` that exposes ``get_forecast(steps, exog=...)``. The request
contract mirrors the existing SARIMA Flask server: a required ``steps`` count
and optional ``exog`` matrix.

The statsmodels / numpy imports are done lazily so validation and routing work
without the heavy dependency. When no artifact is present the base falls back
to a deterministic naive forecast (last-value carry-forward with a small drift)
so both compute backends remain smoke-testable end to end.
"""
from __future__ import annotations

import pickle
from typing import Any, Dict, List, Optional

from ..errors import InvalidRequestError
from .base import InferenceModel

MIN_STEPS = 1
MAX_STEPS = 365


class TimeSeriesForecastModel(InferenceModel):
    """Common validation and forecasting logic for ARIMA-family models."""

    #: Whether exogenous regressors are meaningful for this model.
    supports_exog: bool = False

    def _load_artifact(self) -> Optional[Any]:
        path = self.artifact_path()
        if not path:
            return None
        with open(path, "rb") as handle:
            return pickle.load(handle)

    def validate_input(self, body: Dict[str, Any]) -> Dict[str, Any]:
        if body.get("steps") is None:
            raise InvalidRequestError("'steps' is required")
        try:
            steps = int(body["steps"])
        except (TypeError, ValueError):
            raise InvalidRequestError("'steps' must be an integer")
        if steps < MIN_STEPS or steps > MAX_STEPS:
            raise InvalidRequestError(
                f"'steps' must be between {MIN_STEPS} and {MAX_STEPS}"
            )

        exog = body.get("exog")
        if exog is not None:
            if not self.supports_exog:
                raise InvalidRequestError(
                    f"The '{self.name}' model does not accept exogenous variables"
                )
            if not isinstance(exog, (list, tuple)) or len(exog) != steps:
                raise InvalidRequestError(
                    f"'exog' must be an array with one entry per step ({steps})"
                )
        return {"steps": steps, "exog": list(exog) if exog is not None else None}

    def predict(self, normalized_input: Dict[str, Any]) -> Dict[str, Any]:
        steps = normalized_input["steps"]
        exog = normalized_input.get("exog")

        if self._artifact is None:
            return self._naive_forecast(steps)

        import numpy as np  # lazy import

        exog_array = None
        if exog is not None:
            exog_array = np.asarray(exog, dtype=float)
            if exog_array.ndim == 1:
                exog_array = exog_array.reshape(-1, 1)

        forecast_result = self._artifact.get_forecast(steps=steps, exog=exog_array)
        mean = [float(v) for v in forecast_result.predicted_mean]
        conf = forecast_result.conf_int()
        lower = [float(v) for v in conf.iloc[:, 0]]
        upper = [float(v) for v in conf.iloc[:, 1]]
        return {
            "model": self.name,
            "steps": steps,
            "forecast": mean,
            "confidence_interval_lower": lower,
            "confidence_interval_upper": upper,
            "source": "artifact",
        }

    def _naive_forecast(self, steps: int) -> Dict[str, Any]:
        """Deterministic fallback when no fitted model is available."""
        base = 1.0
        drift = 0.01
        forecast: List[float] = [round(base + drift * i, 6) for i in range(1, steps + 1)]
        return {
            "model": self.name,
            "steps": steps,
            "forecast": forecast,
            "source": "naive-fallback",
            "note": "No fitted artifact present; returned a naive carry-forward forecast.",
        }
