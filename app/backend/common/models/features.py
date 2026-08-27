"""Feature parsing helpers shared by the tabular regression models.

The California Housing models (XGBoost and the weighted baseline) expect eight
standardized features in a fixed order. Callers may supply them either as a
JSON array (``features``) or as the legacy comma-separated string
(``pred_x_csv``) used by the existing Flask inference servers, so both paths
stay compatible.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..errors import InvalidRequestError

# Canonical feature order for the California Housing dataset.
CA_HOUSING_FEATURES: List[str] = [
    "median_income",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "latitude",
    "longitude",
]
NUM_FEATURES = len(CA_HOUSING_FEATURES)


def parse_features(body: Dict[str, Any]) -> List[float]:
    """Extract and validate the eight standardized features from a request.

    Accepts either ``features`` (list of numbers) or ``pred_x_csv`` (string of
    comma-separated numbers). Raises :class:`InvalidRequestError` on any
    malformed input.
    """
    raw: Any
    if "features" in body and body["features"] is not None:
        raw = body["features"]
        if isinstance(raw, str):
            raw = _split_csv(raw)
        elif not isinstance(raw, (list, tuple)):
            raise InvalidRequestError(
                "'features' must be an array of numbers or a comma-separated string"
            )
    elif "pred_x_csv" in body and body["pred_x_csv"] is not None:
        if not isinstance(body["pred_x_csv"], str):
            raise InvalidRequestError("'pred_x_csv' must be a comma-separated string")
        raw = _split_csv(body["pred_x_csv"])
    else:
        raise InvalidRequestError(
            "Request must include 'features' (array) or 'pred_x_csv' (string)"
        )

    if len(raw) != NUM_FEATURES:
        raise InvalidRequestError(
            f"Expected {NUM_FEATURES} features in the order "
            f"{CA_HOUSING_FEATURES}, got {len(raw)}"
        )

    try:
        return [float(v) for v in raw]
    except (TypeError, ValueError):
        raise InvalidRequestError("All feature values must be numeric")


def _split_csv(text: str) -> List[str]:
    parts = [p.strip() for p in text.split(",")]
    return [p for p in parts if p != ""]
