# SageMaker XGBoost training with ECS Fargate hosting

Train an XGBoost regression model with Amazon SageMaker AI, package the trained artifact in a Docker image, and host it on Amazon ECS with AWS Fargate. The recommended deployment path creates an internet-facing Application Load Balancer (ALB), an ECS service, CloudWatch logs, and service auto scaling through CloudFormation.

This guide is the complete, repeatable path for the XGBoost solution:

```text
Provision prerequisites → Train in SageMaker notebook → Build/push image → Deploy ALB + ECS service → Test → Clean up
```

> **Scope:** The repository does not provision Amazon API Gateway. The final section explains the manual follow-on work required if you need an HTTPS API Gateway endpoint.

## Architecture

```text
SageMaker Studio / JupyterLab
        │
        ├── SageMaker training job ──► S3 model artifact
        │
        └── deployment/deploy.sh ──► ECR image
                                         │
Client ──► ALB (HTTP) ──► ECS Fargate service ──► Flask + XGBoost inference
                                         │
                                  CloudWatch Logs
```

## What this repository provides

| Component | Purpose |
|---|---|
| `notebooks/sm_xgboost_ca_housing_ecs_container_model_hosting.ipynb` | Prepares California Housing data, launches the SageMaker XGBoost training job, and includes an optional standalone ECS-task tutorial. |
| `deployment/notebook-prerequisites.yaml` | Creates notebook prerequisites: encrypted S3 bucket, ECS task roles, VPC, public subnet, security group, and optional notebook-role policy. |
| `deployment/setup-prerequisites.sh` | One-command deployment and teardown of the notebook-prerequisites stack. |
| `deployment/deploy.sh` | Downloads the model artifact, builds/pushes the image, and deploys the production-oriented ALB/ECS-service stack. |
| `deployment/cloudformation.yaml` | Creates the ALB, ECS cluster/service, roles, logs, and auto scaling used by `deploy.sh`. |
| `notebooks/scripts/container_sm_xgboost_ca_housing_inference.py` | Flask inference server used by the notebook’s direct-task path. |
| `docs/README.md` | Longer workshop material and architecture background. |

## Before you begin

### 1. Local and AWS prerequisites

You need:

- AWS CLI authenticated through either an AWS profile or the SageMaker execution role; supply a region with `--region` or `AWS_REGION`.
- Permission to deploy CloudFormation stacks, create IAM policies/roles, use EC2/VPC, ECR, ECS, ELB, CloudWatch Logs, and S3.
- Docker installed and a running Docker daemon where you run `deployment/deploy.sh`.
- A SageMaker Studio JupyterLab Space that can run Docker for the notebook-only container build path.
- A SageMaker execution role with the usual SageMaker training permissions, including the ability to create a training job and pass its execution role as required by your account policy.

Clone the repository and confirm the AWS identity before creating resources:

```bash
git clone https://github.com/UCdavidUC/amazon-sagemaker-xgboost-regression-model-hosting-on-amazon-ecs-fargate-and-amazon-api-gateway.git
cd amazon-sagemaker-xgboost-regression-model-hosting-on-amazon-ecs-fargate-and-amazon-api-gateway

export AWS_REGION=us-east-1
aws sts get-caller-identity --region "$AWS_REGION"
```

### 2. Choose the SageMaker environment

Use the CPU image of **Amazon SageMaker Distribution 4.3.3**. The notebook is validated against its Python 3.12.13 runtime, SageMaker Python SDK 3.12.0, pandas 2.3.3, scikit-learn 1.7.2, boto3 1.43.46, and CPU XGBoost 2.1.4. See the [4.3.3 release manifest](https://github.com/aws/sagemaker-distribution/blob/main/build_artifacts/v4/v4.3/v4.3.3/RELEASE.md).

The notebook’s Docker cell confirms that Docker and Buildx are usable. If it reports that the daemon is unavailable, use a SageMaker environment configured with Docker support before attempting the notebook’s direct container-build sections.

### 3. Identify the SageMaker execution-role name

The prerequisite stack can attach the notebook-specific permissions needed to read its CloudFormation outputs, access the created S3 bucket, work with ECR/ECS, and pass the ECS task roles. It needs the **role name**, not the ARN.

In a temporary notebook cell in the target SageMaker environment, run:

```python
from sagemaker.core.helper.session_helper import get_execution_role
print(get_execution_role())
```

For an ARN such as `arn:aws:iam::123456789012:role/AmazonSageMaker-ExecutionRole-Example`, set:

```bash
export NOTEBOOK_ROLE=AmazonSageMaker-ExecutionRole-Example
```

> The generated policy does not replace the normal SageMaker training permissions already required by the execution role. Keep your organization’s approved SageMaker training policy attached.

## Step 1: Provision notebook prerequisites

Deploy the prerequisite stack once per environment:

```bash
./deployment/setup-prerequisites.sh \
  --region "$AWS_REGION" \
  --notebook-role "$NOTEBOOK_ROLE" \
  --inbound-cidr "$(curl -s ifconfig.me)/32"
```

The stack creates:

- A versioned, encrypted S3 bucket for training data, checkpoints, and model artifacts.
- An ECS task execution role and ECS task role for the notebook’s optional standalone-task deployment.
- A VPC, internet gateway, public subnet, route, and task security group.
- A managed policy attached to `NOTEBOOK_ROLE`, when supplied.

The script defaults to stack name `ml-notebook-prereqs`. It writes account-specific output values to `deployment/notebook-config.generated.txt`; the file is ignored by Git.

> **Security:** `--inbound-cidr` controls access to the notebook-created standalone Fargate task on port 80. Use your public IP with `/32` for testing. Do not use the default `0.0.0.0/0` unless public access is intentional.

Record the S3 bucket value for the deployment step:

```bash
export S3_BUCKET=$(aws cloudformation describe-stacks \
  --stack-name ml-notebook-prereqs \
  --region "$AWS_REGION" \
  --query "Stacks[0].Outputs[?OutputKey=='S3BucketName'].OutputValue" \
  --output text)

echo "$S3_BUCKET"
```

## Step 2: Run the notebook through training

1. Open `notebooks/sm_xgboost_ca_housing_ecs_container_model_hosting.ipynb` in the SageMaker environment.
2. Run the prerequisite/version cells in section 1.
3. In section **1.E Create common objects**, keep these defaults unless you used a custom stack name:

   ```python
   PREREQ_STACK_NAME = 'ml-notebook-prereqs'
   USE_CFN_OUTPUTS = True
   ```

   The notebook reads `S3BucketName`, both ECS role ARNs, the subnet, and security group directly from CloudFormation. No manual copy-paste is required.
4. Run sections **2. Prepare the data** and **3. Perform training**.
5. Record the printed training-job name:

   ```text
   Training job name: train-sm-xgboost-ca-housing-ecs-container-model-hosting-...
   ```

   Set it in your terminal:

   ```bash
   export TRAINING_JOB_NAME='paste-the-printed-training-job-name-here'
   ```

The notebook writes the trained artifact to:

```text
s3://$S3_BUCKET/sm-xgboost-ca-housing-ecs-container-model-hosting/output/$TRAINING_JOB_NAME/output/model.tar.gz
```

> **Cost:** Running section 3 creates a separate, temporary SageMaker training instance. It is billed independently of the SageMaker Space instance running the notebook.

## Step 3: Deploy the recommended ALB + ECS service path

The recommended repeatable hosting path is `deployment/deploy.sh`. It downloads the model from S3, extracts it, builds the image, pushes it to ECR, and deploys an ALB plus an ECS Fargate service.

```bash
./deployment/deploy.sh \
  --model-type xgboost \
  --s3-bucket "$S3_BUCKET" \
  --training-job-name "$TRAINING_JOB_NAME" \
  --region "$AWS_REGION" \
  --stack-name ml-inference \
  --ecr-repo-name ml-inference
```

`--model-type xgboost` is required. With the names above, the script defaults to two 0.5-vCPU / 1-GB tasks and creates the `ml-inference` CloudFormation stack and ECR repository.

Useful options:

```bash
# Reduce the service to one task for a lower-cost non-HA test deployment.
--desired-count 1

# Use an already-downloaded/created model instead of a SageMaker model artifact.
--model-file /absolute/path/to/xgboost-model

# Build and push only; do not create the ALB/ECS stack.
--skip-infra
```

The command prints the ALB endpoint after CloudFormation completes. You can also retrieve it later:

```bash
export ALB_URL=$(aws cloudformation describe-stacks \
  --stack-name ml-inference \
  --region "$AWS_REGION" \
  --query "Stacks[0].Outputs[?OutputKey=='LoadBalancerURL'].OutputValue" \
  --output text)

echo "$ALB_URL"
```

## Step 4: Test the service

Check health first:

```bash
curl --fail "$ALB_URL/healthcheck"
```

Send an XGBoost prediction request:

```bash
curl --fail-with-body -X POST "$ALB_URL/" \
  -H 'Content-Type: application/json' \
  --data '{
    "response_content_type": "application/json",
    "pred_x_csv": "0.12,-0.45,0.78,-0.23,0.56,-0.89,1.23,-0.67"
  }'
```

`pred_x_csv` must contain eight **standardized** values in this order:

```text
median_income, housing_median_age, total_rooms, total_bedrooms,
population, households, latitude, longitude
```

The sample values above are only a smoke test. For meaningful predictions, use values transformed with the same `StandardScaler` fitted in the notebook.

View service logs:

```bash
aws logs tail /ecs/ml-inference --follow --region "$AWS_REGION"
```

## Alternative: notebook-only standalone ECS task

The notebook itself also contains sections 4 and 5 that build an image, push it to ECR, and run one standalone ECS Fargate task with a public IP.

Use that path only when you want the guided, direct-task demonstration. It is separate from the recommended `deploy.sh` path:

| Notebook direct task | `deploy.sh` service path |
|---|---|
| One standalone task with a public IP | ECS service behind an ALB |
| Uses the prerequisite stack’s VPC/subnet/security group | Creates its own hosting VPC, roles, ALB, and ECS service |
| Test URL: `http://<task-public-ip>:80/` | Test URL: the stack’s `LoadBalancerURL` |
| Good for tutorial exploration | Recommended repeatable deployment |

Do not run both paths unless you intentionally want both sets of billable resources.

## Optional API Gateway integration

This repository does **not** create API Gateway resources. The recommended deployment already provides the required ECS service and ALB, but you must manually create an HTTP API or REST API, configure the ALB integration, add authorization/TLS/domain controls as needed, and test the resulting API Gateway URL.

The notebook’s API Gateway section is guidance only; it does not automate API Gateway creation or expose a ready-made API Gateway endpoint.

## Cleanup

Delete resources in the reverse order in which you created them. The commands below remove the recommended ALB/ECS-service path and then the prerequisite stack.

```bash
# 1. Delete the production ALB/ECS CloudFormation stack.
aws cloudformation delete-stack \
  --stack-name ml-inference \
  --region "$AWS_REGION"
aws cloudformation wait stack-delete-complete \
  --stack-name ml-inference \
  --region "$AWS_REGION"

# 2. Delete the production ECR repository and images.
aws ecr delete-repository \
  --repository-name ml-inference \
  --force \
  --region "$AWS_REGION"

# 3. Delete notebook prerequisites. This empties all current S3 objects,
#    object versions, and delete markers before deleting the stack.
./deployment/setup-prerequisites.sh \
  --teardown \
  --stack-name ml-notebook-prereqs \
  --region "$AWS_REGION"
```

If you used the notebook-only standalone-task path, run its section 7 cleanup cells first to stop the task, deregister its task definition, remove its ECR repository, and delete notebook-created S3 objects. Also remove local Docker images if desired:

```bash
docker system prune
```

> `setup-prerequisites.sh --teardown` permanently deletes the prerequisite S3 bucket contents, including training data, checkpoints, model artifacts, and all object versions. Keep or copy any artifacts you need before running it.

## Costs and security

- SageMaker training, Fargate tasks, ALB capacity, ECR storage, S3, CloudWatch Logs, and data transfer can incur charges. Delete resources when the solution is no longer needed.
- The production template exposes an HTTP ALB. Configure HTTPS, certificates, authentication, WAF, and restrictive security policies before using it with sensitive or internet-facing workloads.
- The model artifact uses Python pickle. Only unpickle model files you created or otherwise trust.

## Additional documentation

- [Deployment runbook](deployment/README.md) — script and infrastructure details.
- [Workshop guide](docs/README.md) — background concepts and the notebook-oriented walkthrough.
- [Contributing and security reporting](CONTRIBUTING.md#security-issue-notifications).

## License

This project is licensed under the MIT-0 License. See [LICENSE](LICENSE).
