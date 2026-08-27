# Optional container image for the Lambda backend worker.
#
# Use this when you need the heavy model runtimes (xgboost, statsmodels) on the
# Lambda backend, since they exceed the practical size of a zip package. Build
# with the `app/` directory as the context:
#
#   docker buildx build --platform linux/arm64 \
#     -f backend/lambda_worker/Dockerfile -t inference-lambda-worker app/
#
# CloudFormation then references this image and sets the handler via CMD.
FROM public.ecr.aws/lambda/python:3.12-arm64

ENV MODEL_DIR=/opt/models

# Heavy model runtimes (arm64 wheels).
COPY backend/ecs_worker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Shared backend package into the Lambda task root.
COPY backend ${LAMBDA_TASK_ROOT}/backend
RUN mkdir -p /opt/models

# handler module.function
CMD ["backend.lambda_worker.handler.lambda_handler"]
