# Multi-model Dockerfile for ML inference on ECS Fargate
# Supports both XGBoost and SARIMA models via MODEL_TYPE environment variable
#
# Build context should contain:
#   - this Dockerfile
#   - server_xgboost.py (XGBoost Flask inference script)
#   - server_sarima.py (SARIMA Flask inference script)
#   - entrypoint.sh (selects the correct server based on MODEL_TYPE)
#   - requirements.txt (combined Python dependencies)
#   - model.pkl (the trained model pickle file)
#
# Build:
#   docker build -t ml-inference .
#
# Run locally:
#   docker run -p 8080:80 -e MODEL_TYPE=xgboost ml-inference
#   docker run -p 8080:80 -e MODEL_TYPE=sarima ml-inference

FROM public.ecr.aws/amazonlinux/amazonlinux:2023

WORKDIR /app

# Install Python 3, pip, and curl (needed for health checks)
RUN dnf -y install python3 python3-pip curl && dnf clean all

# Create a virtual environment for isolation
RUN python3 -m venv /opt/appenv

# Install Python dependencies (combined for both model types)
COPY requirements.txt .
RUN /opt/appenv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/appenv/bin/pip install --no-cache-dir -r requirements.txt

# Copy both inference servers
COPY server_xgboost.py ./server_xgboost.py
COPY server_sarima.py ./server_sarima.py

# Copy the entrypoint script
COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

# Copy the trained model (named generically for either model type)
COPY model.pkl ./model.pkl

# Environment variables
# MODEL_TYPE: "xgboost" or "sarima" — determines which server script to run
ENV MODEL_TYPE=xgboost
ENV MODEL_PICKLE_FILE_PATH=/app/model.pkl
ENV FLASK_SERVER_LOG_LEVEL=INFO
ENV FLASK_SERVER_HOSTNAME=0.0.0.0
ENV FLASK_SERVER_PORT=80
ENV FLASK_SERVER_DEBUG=False

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:80/healthcheck || exit 1

ENTRYPOINT ["./entrypoint.sh"]
