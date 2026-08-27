#!/usr/bin/env bash
# =============================================================================
# setup-prerequisites.sh — Provision the resources the notebooks require
#
# Deploys the notebook-prerequisites.yaml CloudFormation stack (S3 bucket,
# ECS task execution role, ECS task role, VPC, public subnet, and security
# group) and prints the exact copy-paste values for the notebook config cell.
#
# This is the FIRST step for replicating the solution in a new AWS account.
# After running it, paste the printed values into the notebook and run the
# notebook cells. Once training is done, use deploy.sh for hosting.
#
# Usage:
#   ./setup-prerequisites.sh [options]
#
# Options:
#   --stack-name       CloudFormation stack name (default: ml-notebook-prereqs)
#   --resource-prefix  Name prefix for created resources (default: ml-nb-prereq)
#   --region           AWS region (default: from AWS CLI config)
#   --inbound-cidr     CIDR allowed to reach the ECS task on port 80
#                      (default: 0.0.0.0/0). Use YOUR_IP/32 to restrict.
#   --container-port   Container listen port (default: 80)
#   --notebook-role    Name of the SageMaker execution role to grant notebook
#                      permissions (including cloudformation:DescribeStacks for
#                      section E). Get it from print(get_execution_role()) in
#                      the notebook (the part after 'role/'). Optional.
#   --teardown         Delete the prerequisites stack (empties the S3 bucket first)
#   -h, --help         Show this help
# =============================================================================
set -euo pipefail

# ---------------------------
# Default values
# ---------------------------
STACK_NAME="ml-notebook-prereqs"
RESOURCE_PREFIX="ml-nb-prereq"
REGION=""
INBOUND_CIDR="0.0.0.0/0"
CONTAINER_PORT="80"
NOTEBOOK_ROLE=""
TEARDOWN=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_FILE="${SCRIPT_DIR}/notebook-prerequisites.yaml"

# ---------------------------
# Parse arguments
# ---------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --stack-name) STACK_NAME="$2"; shift 2 ;;
        --resource-prefix) RESOURCE_PREFIX="$2"; shift 2 ;;
        --region) REGION="$2"; shift 2 ;;
        --inbound-cidr) INBOUND_CIDR="$2"; shift 2 ;;
        --container-port) CONTAINER_PORT="$2"; shift 2 ;;
        --notebook-role) NOTEBOOK_ROLE="$2"; shift 2 ;;
        --teardown) TEARDOWN=true; shift ;;
        -h|--help)
            grep -E '^#' "$0" | grep -v '^#!' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---------------------------
# Normalize the notebook role
# ---------------------------
# IAM managed-policy Roles expects a role NAME, not an ARN. Accept either form:
# strip any ARN prefix (everything up to and including the last '/') and any
# surrounding whitespace so 'arn:aws:iam::123:role/path/MyRole' -> 'MyRole'.
if [[ -n "$NOTEBOOK_ROLE" ]]; then
    NOTEBOOK_ROLE="${NOTEBOOK_ROLE##*/}"
    NOTEBOOK_ROLE="${NOTEBOOK_ROLE//[[:space:]]/}"
    if [[ ! "$NOTEBOOK_ROLE" =~ ^[a-zA-Z0-9+=,.@_-]+$ ]]; then
        echo "Error: --notebook-role must be a role NAME (letters, numbers, +=,.@_-),"
        echo "       not an ARN or a value with other characters. Got: '${NOTEBOOK_ROLE}'"
        exit 1
    fi
fi

# ---------------------------
# Preconditions
# ---------------------------
if ! command -v aws > /dev/null 2>&1; then
    echo "Error: AWS CLI is not installed or not on PATH."
    exit 1
fi

if [[ ! -f "$TEMPLATE_FILE" ]]; then
    echo "Error: Template not found at ${TEMPLATE_FILE}"
    exit 1
fi

# Resolve region
if [[ -z "$REGION" ]]; then
    REGION=$(aws configure get region 2>/dev/null || echo "")
    if [[ -z "$REGION" ]]; then
        echo "Error: AWS region not set. Use --region or configure the AWS CLI."
        exit 1
    fi
fi

# Verify credentials early with a clear message
if ! ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region "$REGION" 2>/dev/null); then
    echo "Error: Unable to authenticate with AWS. Check your credentials."
    exit 1
fi

# =============================================================================
# Teardown path
# =============================================================================
if [[ "$TEARDOWN" == true ]]; then
    echo "============================================"
    echo "  Tearing down prerequisites stack"
    echo "============================================"
    echo "  Stack:   ${STACK_NAME}"
    echo "  Region:  ${REGION}"
    echo "  Account: ${ACCOUNT_ID}"
    echo ""

    # The S3 bucket must be emptied before CloudFormation can delete it.
    BUCKET_NAME=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query "Stacks[0].Outputs[?OutputKey=='S3BucketName'].OutputValue" \
        --output text 2>/dev/null || echo "")

    if [[ -n "$BUCKET_NAME" && "$BUCKET_NAME" != "None" ]]; then
        echo ">>> Emptying S3 bucket: ${BUCKET_NAME}"
        # Remove all object versions and delete markers (bucket is versioned),
        # then remove any remaining current objects.
        aws s3 rm "s3://${BUCKET_NAME}" --recursive --region "$REGION" 2>/dev/null || true

        # Delete versioned objects and delete markers if versioning is on.
        VERSIONS=$(aws s3api list-object-versions \
            --bucket "$BUCKET_NAME" \
            --region "$REGION" \
            --output json \
            --query '{Objects: Versions[].{Key:Key,VersionId:VersionId}, DeleteMarkers: DeleteMarkers[].{Key:Key,VersionId:VersionId}}' 2>/dev/null || echo '{}')

        if command -v python3 > /dev/null 2>&1; then
            echo "$VERSIONS" | python3 - "$BUCKET_NAME" "$REGION" <<'PY' || true
import json, subprocess, sys
data = sys.stdin.read().strip()
bucket, region = sys.argv[1], sys.argv[2]
if not data:
    sys.exit(0)
try:
    parsed = json.loads(data)
except json.JSONDecodeError:
    sys.exit(0)
items = []
for group in ("Objects", "DeleteMarkers"):
    for entry in (parsed.get(group) or []):
        if entry and entry.get("Key"):
            items.append(entry)
# Delete in batches of up to 1000 objects.
for i in range(0, len(items), 1000):
    batch = items[i:i + 1000]
    payload = json.dumps({"Objects": [{"Key": e["Key"], "VersionId": e["VersionId"]} for e in batch]})
    subprocess.run(
        ["aws", "s3api", "delete-objects", "--bucket", bucket,
         "--region", region, "--delete", payload],
        check=False, stdout=subprocess.DEVNULL,
    )
PY
        fi
    fi

    echo ">>> Deleting CloudFormation stack: ${STACK_NAME}"
    aws cloudformation delete-stack --stack-name "$STACK_NAME" --region "$REGION"
    echo ">>> Waiting for stack deletion to complete..."
    aws cloudformation wait stack-delete-complete --stack-name "$STACK_NAME" --region "$REGION"

    echo ""
    echo "  Prerequisites stack deleted."
    echo ""
    exit 0
fi

# =============================================================================
# Deploy path
# =============================================================================
echo "============================================"
echo "  Notebook Prerequisites Setup"
echo "============================================"
echo "  Stack:           ${STACK_NAME}"
echo "  Resource prefix: ${RESOURCE_PREFIX}"
echo "  Region:          ${REGION}"
echo "  Account:         ${ACCOUNT_ID}"
echo "  Inbound CIDR:    ${INBOUND_CIDR}"
echo "  Container port:  ${CONTAINER_PORT}"
echo "  Notebook role:   ${NOTEBOOK_ROLE:-(none - skipping permissions policy)}"
echo "============================================"
echo ""

if [[ "$INBOUND_CIDR" == "0.0.0.0/0" ]]; then
    echo "  NOTE: Inbound CIDR is 0.0.0.0/0 — the task will be reachable from"
    echo "        the public internet on port ${CONTAINER_PORT}. For a tighter"
    echo "        setup, re-run with --inbound-cidr YOUR_PUBLIC_IP/32."
    echo ""
fi

echo ">>> Validating template..."
aws cloudformation validate-template \
    --template-body "file://${TEMPLATE_FILE}" \
    --region "$REGION" > /dev/null
echo "    Template is valid."
echo ""

echo ">>> Deploying stack (this creates IAM roles, so CAPABILITY_NAMED_IAM is used)..."
aws cloudformation deploy \
    --template-file "$TEMPLATE_FILE" \
    --stack-name "$STACK_NAME" \
    --parameter-overrides \
        ResourcePrefix="$RESOURCE_PREFIX" \
        InboundTestCIDR="$INBOUND_CIDR" \
        ContainerPort="$CONTAINER_PORT" \
        NotebookExecutionRoleName="$NOTEBOOK_ROLE" \
    --capabilities CAPABILITY_NAMED_IAM \
    --region "$REGION" \
    --no-fail-on-empty-changeset
echo ""

# ---------------------------
# Fetch outputs
# ---------------------------
get_output() {
    aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" \
        --output text
}

S3_BUCKET=$(get_output S3BucketName)
EXEC_ROLE_ARN=$(get_output ECSTaskExecutionRoleArn)
TASK_ROLE_ARN=$(get_output ECSTaskRoleArn)
SUBNET_ID=$(get_output PublicSubnetId)
SG_ID=$(get_output SecurityGroupId)
ECR_URL_PREFIX="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

echo "============================================"
echo "  Setup Complete — copy these into the notebook"
echo "============================================"
echo ""
echo "In the notebook's \"Create common objects\" cell, replace the matching"
echo "placeholder assignments with the following:"
echo ""
echo "# ----- copy from here -----"
echo "s3_bucket = '${S3_BUCKET}'"
echo "container_registry_url_prefix = '${ECR_URL_PREFIX}'"
echo "ecs_fargate_task_execution_role = '${EXEC_ROLE_ARN}'"
echo "ecs_fargate_task_role = '${TASK_ROLE_ARN}'"
echo "ecs_fargate_task_subnet_list = ['${SUBNET_ID}']"
echo "ecs_fargate_task_security_group_list = ['${SG_ID}']"
echo "# ----- copy to here -----"
echo ""
echo "Also written to: ${SCRIPT_DIR}/notebook-config.generated.txt"

# Persist the snippet to a file for convenience.
cat > "${SCRIPT_DIR}/notebook-config.generated.txt" <<EOF
# Generated by setup-prerequisites.sh on $(date -u +"%Y-%m-%dT%H:%M:%SZ")
# Stack: ${STACK_NAME} | Region: ${REGION} | Account: ${ACCOUNT_ID}
#
# Paste these into the notebook "Create common objects" cell, replacing the
# matching placeholder assignments.

s3_bucket = '${S3_BUCKET}'
container_registry_url_prefix = '${ECR_URL_PREFIX}'
ecs_fargate_task_execution_role = '${EXEC_ROLE_ARN}'
ecs_fargate_task_role = '${TASK_ROLE_ARN}'
ecs_fargate_task_subnet_list = ['${SUBNET_ID}']
ecs_fargate_task_security_group_list = ['${SG_ID}']
EOF

echo ""
echo "To remove everything later:  ./setup-prerequisites.sh --teardown --stack-name ${STACK_NAME} --region ${REGION}"
echo ""
