"""SARIMA / SARIMAX time-series model (seasonal, optional exogenous regressors)."""
from __future__ import annotations

from .timeseries import TimeSeriesForecastModel


class SarimaModel(TimeSeriesForecastModel):
    name = "sarima"
    supports_exog = True
