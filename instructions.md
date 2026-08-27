# Deployment instructions

End-to-end process to train a model and deploy the multi-model inference API.
There are three phases:

1. **Provision SageMaker prerequisites** with CloudFormation (S3 bucket, IAM
   roles, VPC values the notebook needs).
2. **Run the Jupyter notebook** manually to train the model and write the
   artifact to S3.
3. **Deploy the application with the AWS CDK**, baking the model from step 2
   into the ECS worker image.

```text
Step 1: CloudFormation prerequisites ─► Step 2: train in notebook ─► Step 3: CDK deploy
   (S3 bucket, roles, VPC/SG)              (model.tar.gz in S3)        (image + stacks)
```

## Prerequisites

- AWS CLI v2, authenticated (`aws sts get-caller-identity` works).
- A region exported for convenience: `export AWS_REGION=us-east-1`.
- Docker with `buildx` (to build the arm64 ECS worker image).
- Node.js + AWS CDK CLI (`npm install -g aws-cdk`).
- Python 3.12+ and permission to deploy CloudFormation, IAM, VPC, ECR, ECS,
  Lambda, API Gateway, DynamoDB, SQS, KMS, and SageMaker training.

---

## Step 1 — Provision SageMaker prerequisites (CloudFormation)

This deploys `deployment/notebook-prerequisites.yaml` and prints the exact
values ("SageMaker variables") to paste into the notebook: the S3 bucket, ECR
registry prefix, the two ECS IAM role ARNs, the subnet, and the security group.

First find your SageMaker execution role **name** (run in a notebook cell in the
target SageMaker environment):

```python
from sagemaker.core.helper.session_helper import get_execution_role
print(get_execution_role())   # take the part after 'role/'
```

```bash
export AWS_REGION=us-east-1
export NOTEBOOK_ROLE=AmazonSageMaker-ExecutionRole-Example   # from above

./deployment/setup-prerequisites.sh \
  --region "$AWS_REGION" \
  --notebook-role "$NOTEBOOK_ROLE" \
  --inbound-cidr "$(curl -s ifconfig.me)/32"
```

The script deploys the stack (default name `ml-notebook-prereqs`) and writes the
paste-ready values to `deployment/notebook-config.generated.txt`, for example:

```python
s3_bucket = 'ml-nb-prereq-123456789012-us-east-1'
container_registry_url_prefix = '123456789012.dkr.ecr.us-east-1.amazonaws.com'
ecs_fargate_task_execution_role = 'arn:aws:iam::123456789012:role/ml-nb-prereq-ecs-task-execution-role'
ecs_fargate_task_role = 'arn:aws:iam::123456789012:role/ml-nb-prereq-ecs-task-role'
ecs_fargate_task_subnet_list = ['subnet-0abc123def456']
ecs_fargate_task_security_group_list = ['sg-0abc123def456']
```

Capture the bucket name for later:

```bash
export S3_BUCKET=$(aws cloudformation describe-stacks \
  --stack-name ml-notebook-prereqs --region "$AWS_REGION" \
  --query "Stacks[0].Outputs[?OutputKey=='S3BucketName'].OutputValue" --output text)
echo "$S3_BUCKET"
```

> Security note: `--inbound-cidr` restricts who can reach the notebook's
> standalone test task. Use your own IP (`/32`), not `0.0.0.0/0`.

---

## Step 2 — Train the model in the notebook (manual)

1. Open `notebooks/sm_xgboost_ca_housing_ecs_container_model_hosting.ipynb` in
   SageMaker Studio / JupyterLab (Amazon SageMaker Distribution 4.3.3, Python
   3.12).
2. In the **Create common objects** cell, paste the values from
   `deployment/notebook-config.generated.txt` (or keep
   `PREREQ_STACK_NAME='ml-notebook-prereqs'` and `USE_CFN_OUTPUTS=True` to have
   the notebook read them from the stack automatically).
3. Run the data-prep and training sections. Record the printed training job
   name:

   ```bash
   export TRAINING_JOB_NAME='train-sm-xgboost-ca-housing-ecs-container-model-hosting-...'
   ```

The trained artifact is written to:

```text
s3://$S3_BUCKET/sm-xgboost-ca-housing-ecs-container-model-hosting/output/$TRAINING_JOB_NAME/output/model.tar.gz
```

The notebook then **stages the trained model automatically** into
`app/backend/ecs_worker/models/xgboost-model.pkl` (section "Stage the trained
model for the inference app"). Run that cell and the artifact is ready for the
worker image build — no manual download/rename needed (step 3.1 below is only a
fallback). By default the cell writes to `app/backend/ecs_worker/models/`
relative to the notebook; set `ECS_WORKER_MODELS_DIR` to override the location.

> Optional: to also serve a real SARIMA model, run
> `notebooks/sarima_arima_ca_housing_ecs_container_model_hosting.ipynb` and note
> its artifact location as well. If you skip this, the `arima`/`sarima` routes
> still respond using the built-in naive-forecast fallback.

---

## Step 3 — Deploy the app with CDK using the trained model

The ECS Fargate worker serves the heavy `xgboost` model from a pickle baked into
its image. We fetch the artifact from S3, stage it, build/push the arm64 image,
then deploy the CDK stacks pointing at that image.

### 3.1 Fetch and stage the model artifact

> **Already done in the notebook.** The training notebook stages the pickle into
> `app/backend/ecs_worker/models/` for you (xgboost -> `xgboost-model.pkl`; the
> SARIMA/ARIMA notebook -> `sarima-model.pkl` / `arima-model.pkl`). If you ran
> that staging cell, skip to 3.2. The manual commands below are a fallback for
> when you train outside the notebook or need to re-stage an existing artifact.

```bash
cd app/backend/ecs_worker/models

# Download and extract the trained model from step 2.
aws s3 cp \
  "s3://${S3_BUCKET}/sm-xgboost-ca-housing-ecs-container-model-hosting/output/${TRAINING_JOB_NAME}/output/model.tar.gz" \
  ./model.tar.gz --region "$AWS_REGION"
tar -xzf model.tar.gz
rm -f model.tar.gz

# SageMaker XGBoost outputs a file named 'xgboost-model'. The worker loads
# '<model-name>-model.pkl', so rename it to 'xgboost-model.pkl'.
[ -f xgboost-model ] && mv xgboost-model xgboost-model.pkl
ls -l   # expect: xgboost-model.pkl   (and optionally sarima-model.pkl)
cd -
```

> If you trained SARIMA in step 2, place its pickle here as `sarima-model.pkl`
> (and `arima-model.pkl` for ARIMA). Files in this folder are git-ignored.

### 3.2 Build and push the ECS worker image (arm64)

```bash
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ECR_REPO=inference-worker
export ECR_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
export IMAGE_URI="${ECR_URI}/${ECR_REPO}:latest"

# Create the repository (idempotent) and authenticate Docker.
aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$AWS_REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$ECR_REPO" \
       --image-scanning-configuration scanOnPush=true --region "$AWS_REGION" >/dev/null
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$ECR_URI"

# Build for Graviton/arm64 with the build context = app/ and push.
docker buildx build --platform linux/arm64 \
  -f app/backend/ecs_worker/Dockerfile \
  -t "$IMAGE_URI" app/ --push
```

### 3.3 Deploy the CDK stacks

CDK uploads the Lambda code as an asset automatically, so no manual zip is
needed. Pass the image built above via `-c ecsImageUri` so CDK uses it directly
(no local Docker build during deploy).

```bash
cd app/cdk
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# One-time per account/region.
cdk bootstrap aws://${ACCOUNT_ID}/${AWS_REGION}

# Deploy core + API/Lambda + ECS for the dev environment.
cdk deploy -c environment=dev --all --require-approval never \
  -c ecsImageUri="$IMAGE_URI"
```

To deploy another environment, change `-c environment=qa` (or `prod`) and rebuild
the image tag if desired.

Optional — deploy the CI/CD pipeline so future SageMaker image pushes redeploy
automatically:

```bash
export QA_API_URL=$(aws cloudformation describe-stacks \
  --stack-name inference-dev-api-lambda --region "$AWS_REGION" \
  --query "Stacks[0].Outputs[?OutputKey=='PrivateInvokeUrl'].OutputValue" --output text)

cdk deploy inference-cicd \
  -c deployCicd=true \
  -c ecrRepo="$ECR_REPO" \
  -c approvalEmail=you@example.com \
  -c qaApiBaseUrl="$QA_API_URL"
```

### 3.4 Test the deployed API

The API is a **private** API Gateway (reachable only from inside the VPC) and
uses **IAM authorization**, so requests must be SigV4-signed and originate from
within the VPC (for example from an EC2 instance or ECS task in a private
subnet, or CloudShell attached to the VPC).

Get the base URL:

```bash
aws cloudformation describe-stacks --stack-name inference-dev-api-lambda \
  --region "$AWS_REGION" \
  --query "Stacks[0].Outputs[?OutputKey=='PrivateInvokeUrl'].OutputValue" --output text
```

From an in-VPC host with `awscurl` (SigV4 signing):

```bash
pip install awscurl
BASE=<PrivateInvokeUrl>

# Submit an xgboost inference (served by the ECS backend).
awscurl --service execute-api --region "$AWS_REGION" -X POST \
  "$BASE/backend/ecs-fargate/model/xgboost" \
  -H 'Content-Type: application/json' \
  -d '{"features":[0.12,-0.45,0.78,-0.23,0.56,-0.89,1.23,-0.67]}'
# -> {"request_id":"...","status":"QUEUED","status_url":"/api/requests/..."}

# Poll for the result.
awscurl --service execute-api --region "$AWS_REGION" \
  "$BASE/requests/<request_id>"
# -> {"status":"COMPLETED","output":{"prediction": ...}}
```

Which models need the trained artifact:

| Route | Backend | Needs artifact from step 2? |
|---|---|---|
| `.../model/xgboost` | ECS (baked image) | Yes — `xgboost-model.pkl` |
| `.../model/weighted` | Lambda or ECS | No (built-in coefficients) |
| `.../model/arima`, `.../model/sarima` | Lambda or ECS | Optional (naive fallback if absent) |

> The Lambda backend `xgboost` route needs the heavy runtime + artifact via a
> Lambda layer or container image; by default xgboost is served by the ECS
> backend. See `app/cdk/README.md`.

---

## Updating the model later

Retrain in the notebook (step 2), re-stage the new pickle (3.1), rebuild/push
the image (3.2), then either `cdk deploy ... -c ecsImageUri=$IMAGE_URI` again, or
— if the CI/CD stack is deployed — simply push the new image to ECR and the
pipeline runs SAST → QA → tests → approval → prod automatically.

## Cleanup

```bash
# App stacks (from app/cdk with the venv active).
cdk destroy -c environment=dev --all -c ecsImageUri="$IMAGE_URI"

# ECR repository.
aws ecr delete-repository --repository-name "$ECR_REPO" --force --region "$AWS_REGION"

# SageMaker prerequisites (empties the versioned S3 bucket first).
./deployment/setup-prerequisites.sh --teardown \
  --stack-name ml-notebook-prereqs --region "$AWS_REGION"
```

> Teardown permanently deletes the prerequisite S3 bucket contents, including
> training data and model artifacts. Copy anything you need first.
