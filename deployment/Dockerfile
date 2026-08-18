# Dockerfile for XGBoost California Housing inference container
# Deployed on Amazon ECS Fargate
#
# Build context should contain:
#   - this Dockerfile
#   - server.py (the Flask inference script)
#   - requirements.txt (Python dependencies)
#   - xgboost-model (the trained model pickle file)
#
# Build:
#   docker build -t sm-xgboost-ca-housing-inference .
#
# Run locally for testing:
#   docker run -p 8080:80 sm-xgboost-ca-housing-inference

FROM public.ecr.aws/amazonlinux/amazonlinux:2023

WORKDIR /app

# Install Python 3 and pip
RUN dnf -y install python3 python3-pip && dnf clean all

# Create a virtual environment for isolation
RUN python3 -m venv /opt/appenv

# Install Python dependencies
COPY requirements.txt .
RUN /opt/appenv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/appenv/bin/pip install --no-cache-dir -r requirements.txt

# Copy the trained model and inference server
COPY xgboost-model ./xgboost-model
COPY server.py ./server.py

# Environment variables for the Flask inference server
ENV MODEL_PICKLE_FILE_PATH=/app/xgboost-model
ENV FLASK_SERVER_LOG_LEVEL=INFO
ENV FLASK_SERVER_HOSTNAME=0.0.0.0
ENV FLASK_SERVER_PORT=80
ENV FLASK_SERVER_DEBUG=False

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:80/healthcheck || exit 1

ENTRYPOINT ["/opt/appenv/bin/python", "server.py"]
