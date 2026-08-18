#!/usr/bin/env bash
# =============================================================================
# deploy.sh — Build, push, and deploy ML inference container to ECS Fargate
#
# Supports both XGBoost and SARIMA model types.
#
# Usage:
#   ./deploy.sh --model-type <xgboost|sarima> --s3-bucket <bucket> --training-job-name <job-name> [options]
#   ./deploy.sh --model-type sarima --model-file ./my_sarima_model.pkl [options]
#
# Required:
#   --model-type          Model type: "xgboost" or "sarima"
#   --s3-bucket           S3 bucket where model artifacts are stored
#   --training-job-name   SageMaker training job name (used to locate model.tar.gz)
#
# Optional:
#   --region              AWS region (default: from AWS CLI config)
#   --stack-name          CloudFormation stack name (default: ml-inference)
#   --ecr-repo-name       ECR repository name (default: ml-inference)
#   --image-tag           Docker image tag (default: latest)
#   --task-cpu            Fargate task CPU (default: 512)
#   --task-memory         Fargate task memory MB (default: 1024)
#   --desired-count       Number of ECS tasks (default: 2)
#   --model-file          Path to a local model pickle file (skips S3 download)
#   --skip-build          Skip Docker build/push (use existing image in ECR)
#   --skip-infra          Skip CloudFormation deployment (only build and push image)
# =============================================================================
set -euo pipefail

# ---------------------------
# Default values
# ---------------------------
REGION=""
STACK_NAME="ml-inference"
ECR_REPO_NAME="ml-inference"
IMAGE_TAG="latest"
TASK_CPU="512"
TASK_MEMORY="1024"
DESIRED_COUNT="2"
S3_BUCKET=""
TRAINING_JOB_NAME=""
MODEL_FILE=""
MODEL_TYPE=""
SKIP_BUILD=false
SKIP_INFRA=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# Notebook name prefix used in S3 paths by the notebook
NB_NAME="sm-xgboost-ca-housing-ecs-container-model-hosting"

# ---------------------------
# Parse arguments
# ---------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-type) MODEL_TYPE="$2"; shift 2 ;;
        --s3-bucket) S3_BUCKET="$2"; shift 2 ;;
        --training-job-name) TRAINING_JOB_NAME="$2"; shift 2 ;;
        --region) REGION="$2"; shift 2 ;;
        --stack-name) STACK_NAME="$2"; shift 2 ;;
        --ecr-repo-name) ECR_REPO_NAME="$2"; shift 2 ;;
        --image-tag) IMAGE_TAG="$2"; shift 2 ;;
        --task-cpu) TASK_CPU="$2"; shift 2 ;;
        --task-memory) TASK_MEMORY="$2"; shift 2 ;;
        --desired-count) DESIRED_COUNT="$2"; shift 2 ;;
        --model-file) MODEL_FILE="$2"; shift 2 ;;
        --skip-build) SKIP_BUILD=true; shift ;;
        --skip-infra) SKIP_INFRA=true; shift ;;
        -h|--help)
            head -30 "$0" | grep -E '^\s*#' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---------------------------
# Validate inputs
# ---------------------------
if [[ -z "$MODEL_TYPE" ]]; then
    echo "Error: --model-type is required (xgboost or sarima)."
    exit 1
fi

if [[ "$MODEL_TYPE" != "xgboost" && "$MODEL_TYPE" != "sarima" ]]; then
    echo "Error: --model-type must be 'xgboost' or 'sarima'."
    exit 1
fi

if [[ "$SKIP_BUILD" == false ]]; then
    if [[ -z "$MODEL_FILE" && ( -z "$S3_BUCKET" || -z "$TRAINING_JOB_NAME" ) ]]; then
        echo "Error: Either --model-file OR both --s3-bucket and --training-job-name are required."
        echo "Run with --help for usage."
        exit 1
    fi
fi

# Resolve region
if [[ -z "$REGION" ]]; then
    REGION=$(aws configure get region 2>/dev/null || echo "")
    if [[ -z "$REGION" ]]; then
        echo "Error: AWS region not set. Use --region or configure AWS CLI."
        exit 1
    fi
fi

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region "$REGION")
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
FULL_IMAGE_URI="${ECR_URI}/${ECR_REPO_NAME}:${IMAGE_TAG}"

echo "============================================"
echo "  ML Inference — ECS Fargate Deployment"
echo "============================================"
echo "  Model Type:    ${MODEL_TYPE}"
echo "  Region:        ${REGION}"
echo "  Account:       ${ACCOUNT_ID}"
echo "  Stack:         ${STACK_NAME}"
echo "  ECR Repo:      ${ECR_REPO_NAME}"
echo "  Image:         ${FULL_IMAGE_URI}"
echo "  Task CPU:      ${TASK_CPU}"
echo "  Task Memory:   ${TASK_MEMORY} MB"
echo "  Desired Count: ${DESIRED_COUNT}"
echo "============================================"
echo ""

# =============================================================================
# Step 1: Prepare build context
# =============================================================================
if [[ "$SKIP_BUILD" == false ]]; then
    echo ">>> Step 1: Preparing build context..."
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"

    # Copy Dockerfile and entrypoint
    cp "${SCRIPT_DIR}/Dockerfile" "${BUILD_DIR}/"
    cp "${SCRIPT_DIR}/entrypoint.sh" "${BUILD_DIR}/"

    # Copy both inference scripts (both are baked into the image)
    cp "${PROJECT_ROOT}/notebooks/scripts/container_sm_xgboost_ca_housing_inference.py" "${BUILD_DIR}/server_xgboost.py"
    cp "${PROJECT_ROOT}/notebooks/scripts/container_sarima_inference.py" "${BUILD_DIR}/server_sarima.py"

    # Copy combined requirements
    cp "${SCRIPT_DIR}/requirements.txt" "${BUILD_DIR}/requirements.txt"

    # Get the model file
    if [[ -n "$MODEL_FILE" ]]; then
        echo "    Using local model file: ${MODEL_FILE}"
        cp "$MODEL_FILE" "${BUILD_DIR}/model.pkl"
    else
        echo "    Downloading model from S3..."
        MODEL_S3_KEY="${NB_NAME}/output/${TRAINING_JOB_NAME}/output/model.tar.gz"
        aws s3 cp "s3://${S3_BUCKET}/${MODEL_S3_KEY}" "${BUILD_DIR}/model.tar.gz" --region "$REGION"
        echo "    Extracting model pickle file..."
        tar -xzf "${BUILD_DIR}/model.tar.gz" -C "${BUILD_DIR}/"
        rm -f "${BUILD_DIR}/model.tar.gz"

        # SageMaker XGBoost outputs 'xgboost-model'; rename to generic 'model.pkl'
        if [[ -f "${BUILD_DIR}/xgboost-model" ]]; then
            mv "${BUILD_DIR}/xgboost-model" "${BUILD_DIR}/model.pkl"
        elif [[ -f "${BUILD_DIR}/model.pkl" ]]; then
            : # already correct name
        else
            echo "Error: Could not find model file in extracted archive."
            echo "Contents:"
            ls -la "${BUILD_DIR}/"
            exit 1
        fi
    fi

    echo "    Build context ready at: ${BUILD_DIR}/"
    echo ""

# =============================================================================
# Step 2: Create ECR repository (if it doesn't exist)
# =============================================================================
    echo ">>> Step 2: Ensuring ECR repository exists..."
    aws ecr describe-repositories \
        --repository-names "$ECR_REPO_NAME" \
        --region "$REGION" > /dev/null 2>&1 || \
    aws ecr create-repository \
        --repository-name "$ECR_REPO_NAME" \
        --image-scanning-configuration scanOnPush=true \
        --region "$REGION" > /dev/null

    echo "    ECR repository: ${ECR_REPO_NAME}"
    echo ""

# =============================================================================
# Step 3: Build and push Docker image
# =============================================================================
    echo ">>> Step 3: Building Docker image..."
    docker build -t "${ECR_REPO_NAME}:${IMAGE_TAG}" "${BUILD_DIR}/"
    echo ""

    echo ">>> Step 4: Pushing image to ECR..."
    aws ecr get-login-password --region "$REGION" | \
        docker login --username AWS --password-stdin "$ECR_URI"
    docker tag "${ECR_REPO_NAME}:${IMAGE_TAG}" "$FULL_IMAGE_URI"
    docker push "$FULL_IMAGE_URI"
    echo "    Pushed: ${FULL_IMAGE_URI}"
    echo ""

    # Cleanup build directory
    rm -rf "$BUILD_DIR"
fi

# =============================================================================
# Step 5: Deploy CloudFormation stack
# =============================================================================
if [[ "$SKIP_INFRA" == false ]]; then
    echo ">>> Step 5: Deploying CloudFormation stack..."
    aws cloudformation deploy \
        --template-file "${SCRIPT_DIR}/cloudformation.yaml" \
        --stack-name "$STACK_NAME" \
        --parameter-overrides \
            ContainerImageUri="$FULL_IMAGE_URI" \
            ModelType="$MODEL_TYPE" \
            TaskCpu="$TASK_CPU" \
            TaskMemory="$TASK_MEMORY" \
            DesiredCount="$DESIRED_COUNT" \
            EnvironmentName="$STACK_NAME" \
        --capabilities CAPABILITY_NAMED_IAM \
        --region "$REGION" \
        --no-fail-on-empty-changeset

    echo ""
    echo ">>> Step 6: Waiting for stack to complete..."
    aws cloudformation wait stack-create-complete \
        --stack-name "$STACK_NAME" \
        --region "$REGION" 2>/dev/null || \
    aws cloudformation wait stack-update-complete \
        --stack-name "$STACK_NAME" \
        --region "$REGION" 2>/dev/null || true

    # Get the ALB URL
    ALB_URL=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query "Stacks[0].Outputs[?OutputKey=='LoadBalancerURL'].OutputValue" \
        --output text)

    echo ""
    echo "============================================"
    echo "  Deployment Complete!"
    echo "============================================"
    echo ""
    echo "  Model Type:         ${MODEL_TYPE}"
    echo "  Inference endpoint: ${ALB_URL}"
    echo ""

    if [[ "$MODEL_TYPE" == "xgboost" ]]; then
        echo "  Test (XGBoost):"
        echo "    curl -X POST -H 'Content-Type: application/json' \\"
        echo "      --data '{\"response_content_type\":\"application/json\",\"pred_x_csv\":\"0.12,-0.45,0.78,-0.23,0.56,-0.89,1.23,-0.67\"}' \\"
        echo "      ${ALB_URL}"
    else
        echo "  Test (SARIMA):"
        echo "    curl -X POST -H 'Content-Type: application/json' \\"
        echo "      --data '{\"response_content_type\":\"application/json\",\"steps\":5}' \\"
        echo "      ${ALB_URL}"
    fi

    echo ""
    echo "  View logs:"
    echo "    aws logs tail /ecs/${STACK_NAME} --follow --region ${REGION}"
    echo ""
else
    echo ""
    echo "  Infrastructure deployment skipped (--skip-infra)."
    echo "  Image URI: ${FULL_IMAGE_URI}"
    echo ""
fi
