"""ARIMA time-series model (non-seasonal, no exogenous regressors)."""
from __future__ import annotations

from .timeseries import TimeSeriesForecastModel


class ArimaModel(TimeSeriesForecastModel):
    name = "arima"
    supports_exog = False
