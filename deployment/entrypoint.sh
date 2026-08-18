#!/bin/bash
# Entrypoint script — selects the correct inference server based on MODEL_TYPE
set -e

MODEL_TYPE="${MODEL_TYPE:-xgboost}"

case "$MODEL_TYPE" in
    xgboost)
        echo "Starting XGBoost inference server..."
        exec /opt/appenv/bin/python /app/server_xgboost.py
        ;;
    sarima)
        echo "Starting SARIMA inference server..."
        exec /opt/appenv/bin/python /app/server_sarima.py
        ;;
    *)
        echo "Error: Unknown MODEL_TYPE='${MODEL_TYPE}'. Must be 'xgboost' or 'sarima'."
        exit 1
        ;;
esac
