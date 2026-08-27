#!/usr/bin/env bash
# =============================================================================
# deploy-app.sh - Package and deploy the multi-model inference application.
#
# Order of operations:
#   1. Package the shared `backend` Python package into backend.zip and upload
#      it to the code S3 bucket (used by the API and Lambda-worker functions).
#   2. Deploy the core stack (KMS, VPC, DynamoDB, SQS, log groups).
#   3. Build and push the ECS worker container image (arm64/Graviton) to ECR.
#   4. Deploy the API + Lambda backend stack and the ECS Fargate backend stack.
#
# The CI/CD stack (30-cicd.yaml) is deployed separately once, see README.md.
#
# Usage:
#   ./deploy-app.sh --environment dev --region us-east-1 \
#       --code-bucket my-code-bucket --ecr-repo inference-worker
#
# Requires: AWS CLI v2, Docker with buildx (for the arm64 image), zip.
# =============================================================================
set -euo pipefail

ENVIRONMENT="dev"
REGION=""
ENV_NAME="inference"
CODE_BUCKET=""
ECR_REPO="inference-worker"
IMAGE_TAG="latest"
SKIP_IMAGE=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"          # .../app
BUILD_DIR="${SCRIPT_DIR}/build"

while [[ $# -gt 0 ]]; do
  case $1 in
    --environment) ENVIRONMENT="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    --env-name) ENV_NAME="$2"; shift 2 ;;
    --code-bucket) CODE_BUCKET="$2"; shift 2 ;;
    --ecr-repo) ECR_REPO="$2"; shift 2 ;;
    --image-tag) IMAGE_TAG="$2"; shift 2 ;;
    --skip-image) SKIP_IMAGE=true; shift ;;
    -h|--help) grep -E '^#' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ -z "$REGION" ]]; then
  REGION=$(aws configure get region 2>/dev/null || echo "")
  [[ -z "$REGION" ]] && { echo "Error: --region required"; exit 1; }
fi
[[ -z "$CODE_BUCKET" ]] && { echo "Error: --code-bucket required"; exit 1; }

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region "$REGION")
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE_URI="${ECR_URI}/${ECR_REPO}:${IMAGE_TAG}"
CODE_KEY="backend/backend-${ENVIRONMENT}.zip"

echo "== Environment=${ENVIRONMENT} Region=${REGION} Account=${ACCOUNT_ID} =="

# -----------------------------------------------------------------------------
# 1. Package and upload the backend zip
# -----------------------------------------------------------------------------
echo ">>> Packaging backend.zip"
rm -rf "$BUILD_DIR"; mkdir -p "$BUILD_DIR"
# Zip so that `backend/` is at the archive root (handler = backend.api.handler...).
# Baked ECS model artifacts are excluded: the Lambda worker reads /opt/models,
# not backend/ecs_worker/models, so shipping them in the zip is unnecessary.
( cd "$APP_DIR" && zip -qr "${BUILD_DIR}/backend.zip" backend \
    -x '*/__pycache__/*' '*.pyc' 'backend/ecs_worker/models/*' )
aws s3 cp "${BUILD_DIR}/backend.zip" "s3://${CODE_BUCKET}/${CODE_KEY}" --region "$REGION"

# -----------------------------------------------------------------------------
# 2. Core stack
# -----------------------------------------------------------------------------
echo ">>> Deploying core stack"
aws cloudformation deploy \
  --template-file "${SCRIPT_DIR}/00-core.yaml" \
  --stack-name "${ENV_NAME}-${ENVIRONMENT}-core" \
  --parameter-overrides EnvironmentName="$ENV_NAME" Environment="$ENVIRONMENT" \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
  --region "$REGION" --no-fail-on-empty-changeset

# -----------------------------------------------------------------------------
# 3. Build + push the ECS worker image (arm64)
# -----------------------------------------------------------------------------
if [[ "$SKIP_IMAGE" == false ]]; then
  echo ">>> Building and pushing ECS worker image (arm64)"
  aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$REGION" >/dev/null 2>&1 || \
    aws ecr create-repository --repository-name "$ECR_REPO" \
      --image-scanning-configuration scanOnPush=true --region "$REGION" >/dev/null
  aws ecr get-login-password --region "$REGION" | \
    docker login --username AWS --password-stdin "$ECR_URI"
  docker buildx build --platform linux/arm64 \
    -f "${APP_DIR}/backend/ecs_worker/Dockerfile" \
    -t "$IMAGE_URI" "${APP_DIR}" --push
fi

# -----------------------------------------------------------------------------
# 4. API + Lambda backend, then ECS Fargate backend
# -----------------------------------------------------------------------------
echo ">>> Deploying API + Lambda backend stack"
aws cloudformation deploy \
  --template-file "${SCRIPT_DIR}/10-api-lambda.yaml" \
  --stack-name "${ENV_NAME}-${ENVIRONMENT}-api-lambda" \
  --parameter-overrides EnvironmentName="$ENV_NAME" Environment="$ENVIRONMENT" \
      CodeS3Bucket="$CODE_BUCKET" CodeS3Key="$CODE_KEY" \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
  --region "$REGION" --no-fail-on-empty-changeset

echo ">>> Deploying ECS Fargate backend stack"
aws cloudformation deploy \
  --template-file "${SCRIPT_DIR}/20-ecs-fargate.yaml" \
  --stack-name "${ENV_NAME}-${ENVIRONMENT}-ecs" \
  --parameter-overrides EnvironmentName="$ENV_NAME" Environment="$ENVIRONMENT" \
      ContainerImageUri="$IMAGE_URI" \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
  --region "$REGION" --no-fail-on-empty-changeset

echo ""
echo "== Done. Private API base URL: =="
aws cloudformation describe-stacks \
  --stack-name "${ENV_NAME}-${ENVIRONMENT}-api-lambda" \
  --region "$REGION" \
  --query "Stacks[0].Outputs[?OutputKey=='PrivateInvokeUrl'].OutputValue" \
  --output text
