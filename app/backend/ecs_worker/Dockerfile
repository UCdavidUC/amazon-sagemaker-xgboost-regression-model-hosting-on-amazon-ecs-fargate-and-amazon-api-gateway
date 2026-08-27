# ECS Fargate worker image for the multi-model inference backend.
#
# Built for Graviton (arm64). The build context must be the `app/` directory so
# that the whole `backend/` package is copied in:
#
#   docker buildx build --platform linux/arm64 \
#     -f backend/ecs_worker/Dockerfile -t inference-ecs-worker app/
#
# Model artifacts (e.g. xgboost-model.ubj, sarima-model.pkl) can be baked into
# /opt/models at build time or mounted/synced at runtime from the model S3
# bucket (MODEL_BUCKET / MODEL_PREFIX).
FROM public.ecr.aws/docker/library/python:3.12-slim

# Graviton/arm64 target. Fargate runtimePlatform is set to ARM64 in the task def.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_DIR=/opt/models \
    HEARTBEAT_FILE=/tmp/worker-heartbeat

WORKDIR /app

# System deps needed by numpy/scipy/statsmodels wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better layer caching.
COPY backend/ecs_worker/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy the shared backend package.
COPY backend /app/backend

# Baked-in model artifacts (optional). Populate backend/ecs_worker/models/ with
# model artifacts (e.g. xgboost-model.ubj, sarima-model.pkl) before building;
# the folder may be empty for the fallback (weighted/arima/sarima) models.
RUN mkdir -p /opt/models
COPY backend/ecs_worker/models/ /opt/models/

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app /opt/models
USER appuser

# Health check uses the heartbeat file (no inbound port on a worker).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-m", "backend.ecs_worker.healthcheck"]

ENTRYPOINT ["python", "-m", "backend.ecs_worker.worker"]
