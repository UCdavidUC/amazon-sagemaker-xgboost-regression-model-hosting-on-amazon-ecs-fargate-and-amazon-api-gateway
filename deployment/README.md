# ECS Fargate Deployment

Deploy the trained XGBoost California Housing model as a scalable inference API on Amazon ECS Fargate behind an Application Load Balancer.

## Replicating in a New AWS Account (End-to-End)

The full flow has two stages. Stage 1 provisions what the **notebook** needs to run; stage 2 hosts the **trained model** as a production API.

```
Stage 1 (once)        Stage 2 (after training)
setup-prerequisites.sh  ─►  run notebook  ─►  deploy.sh
   │                          │                  │
   ▼                          ▼                  ▼
S3 bucket, IAM roles,     trains model      builds image, pushes to ECR,
VPC/subnet/SG             to S3             deploys ALB + ECS service
```

### Stage 1 — Provision notebook prerequisites (one command)

Instead of manually creating an S3 bucket, IAM roles, a VPC/subnet, and a security group, run:

```bash
cd deployment
./setup-prerequisites.sh --region us-east-1
```

This deploys the `notebook-prerequisites.yaml` CloudFormation stack and prints a ready-to-paste block, for example:

```python
s3_bucket = 'ml-nb-prereq-123456789012-us-east-1'
container_registry_url_prefix = '123456789012.dkr.ecr.us-east-1.amazonaws.com'
ecs_fargate_task_execution_role = 'arn:aws:iam::123456789012:role/ml-nb-prereq-ecs-task-execution-role'
ecs_fargate_task_role = 'arn:aws:iam::123456789012:role/ml-nb-prereq-ecs-task-role'
ecs_fargate_task_subnet_list = ['subnet-0abc123def456']
ecs_fargate_task_security_group_list = ['sg-0abc123def456']
```

The same values are also written to `deployment/notebook-config.generated.txt`. Paste them into the notebook's "Create common objects" cell, replacing the `<Specify ...>` placeholders.

**Options:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--region` | AWS CLI default | AWS region |
| `--stack-name` | `ml-notebook-prereqs` | CloudFormation stack name |
| `--resource-prefix` | `ml-nb-prereq` | Name prefix for created resources (also the S3 bucket prefix) |
| `--inbound-cidr` | `0.0.0.0/0` | CIDR allowed to reach the task on port 80. Use `YOUR_IP/32` to restrict |
| `--container-port` | `80` | Container listen port |
| `--teardown` | — | Delete the prerequisites stack (empties the versioned S3 bucket first) |

> **Security note:** the default `--inbound-cidr 0.0.0.0/0` opens the task's port to the public internet for easy testing. For a tighter setup, pass your public IP: `--inbound-cidr $(curl -s ifconfig.me)/32`.

### Stage 2 — Train, then deploy

1. Run the notebook end-to-end (it trains the model and uploads the artifact to the S3 bucket from stage 1).
2. Deploy the hosting stack with `deploy.sh` (see [Quick Start](#quick-start) below).

### Tearing everything down

```bash
# Stage 2 hosting stack + ECR repo (see Cleanup section)
aws cloudformation delete-stack --stack-name ml-inference --region us-east-1

# Stage 1 prerequisites (empties the S3 bucket, then deletes the stack)
./setup-prerequisites.sh --teardown --region us-east-1
```


## Architecture

```
Client → ALB (port 80) → ECS Fargate Service → Flask container (model inference)
                                                        ↑
                                               ECR (container image)
```

The CloudFormation template creates:
- VPC with 2 public subnets across AZs
- Internet-facing Application Load Balancer
- ECS Cluster with Fargate launch type
- ECS Service (auto-scaling 1–4 tasks based on CPU utilization)
- IAM roles for task execution and task runtime
- CloudWatch Log Group for container logs
- Security groups (ALB accepts HTTP/80; tasks only accept traffic from ALB)

## Prerequisites

- AWS CLI configured with appropriate credentials
- Docker installed and running
- The SageMaker training job has completed (model artifact exists in S3)
- IAM permissions: ECR, ECS, CloudFormation, EC2, ELB, IAM PassRole, CloudWatch Logs, S3 read

## Quick Start

```bash
# After running the notebook training (step 3), deploy to ECS:
./deploy.sh \
  --s3-bucket my-sagemaker-bucket \
  --training-job-name xgboost-ca-housing-2024-01-15-12-30-00-000 \
  --region us-east-1
```

The script will:
1. Download and extract the model from S3
2. Build the Docker container with the model and inference server
3. Create an ECR repository and push the image
4. Deploy the CloudFormation stack (VPC, ALB, ECS Service)
5. Print the ALB endpoint URL for inference

## Usage

```bash
./deploy.sh --s3-bucket <bucket> --training-job-name <job-name> [options]
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--s3-bucket` | S3 bucket where SageMaker stored the model artifact |
| `--training-job-name` | The SageMaker training job name |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--region` | AWS CLI default | AWS region for deployment |
| `--stack-name` | `xgboost-inference` | CloudFormation stack name |
| `--ecr-repo-name` | `sm-xgboost-ca-housing-inference` | ECR repository name |
| `--image-tag` | `latest` | Docker image tag |
| `--task-cpu` | `256` | Fargate CPU units (256 = 0.25 vCPU) |
| `--task-memory` | `512` | Fargate memory in MB |
| `--desired-count` | `1` | Initial number of tasks |
| `--model-file` | — | Path to a local model pickle file (skips S3 download) |
| `--skip-build` | — | Skip Docker build/push, use existing ECR image |
| `--skip-infra` | — | Only build/push image, skip CloudFormation |

### Using a Local Model File

If you have the model pickle file locally (e.g., from the notebook's `container-artifacts/` directory):

```bash
./deploy.sh \
  --model-file ../notebooks/container-artifacts/xgboost-model \
  --region us-east-1
```

## Testing the Endpoint

Once deployed, the script prints the ALB URL. Test with:

```bash
# JSON response
curl -X POST -H 'Content-Type: application/json' \
  --data '{"response_content_type":"application/json","pred_x_csv":"0.12,-0.45,0.78,-0.23,0.56,-0.89,1.23,-0.67"}' \
  http://<alb-dns-name>

# Plain text response
curl -X POST -H 'Content-Type: application/json' \
  --data '{"response_content_type":"text/plain","pred_x_csv":"0.12,-0.45,0.78,-0.23,0.56,-0.89,1.23,-0.67"}' \
  http://<alb-dns-name>
```

The `pred_x_csv` field expects 8 comma-separated standardized feature values matching the training input order:
`median_income, housing_median_age, total_rooms, total_bedrooms, population, households, latitude, longitude`

## Health Check

The container exposes `GET /healthcheck` returning `200 OK`. The ALB target group and ECS both use this for health monitoring.

## Viewing Logs

```bash
aws logs tail /ecs/xgboost-inference --follow --region us-east-1
```

## Updating the Model

To deploy a new model version:

```bash
# Rebuild and push with the new model, then force ECS to pull the new image
./deploy.sh \
  --model-file /path/to/new/xgboost-model \
  --region us-east-1
```

The CloudFormation update will trigger a rolling deployment of the new image.

## Cleanup

Delete all resources created by this deployment:

```bash
# Delete the CloudFormation stack (VPC, ALB, ECS, IAM roles, logs)
aws cloudformation delete-stack --stack-name xgboost-inference --region us-east-1
aws cloudformation wait stack-delete-complete --stack-name xgboost-inference --region us-east-1

# Delete the ECR repository and images
aws ecr delete-repository \
  --repository-name sm-xgboost-ca-housing-inference \
  --force \
  --region us-east-1
```

## Cost Estimate

With default settings (1 task, 0.25 vCPU, 0.5 GB):
- Fargate: ~$9/month (running 24/7)
- ALB: ~$16/month + data transfer
- CloudWatch Logs: minimal (pay per ingestion)

Scale to zero by setting `--desired-count 0` when not in use, or delete the stack entirely.

## Files

| File | Purpose |
|------|---------|
| `notebook-prerequisites.yaml` | Stage 1 template: S3 bucket, ECS IAM roles, VPC/subnet/security group the notebook needs |
| `setup-prerequisites.sh` | One-command provisioning of the notebook prerequisites; prints copy-paste notebook values |
| `Dockerfile` | Container image definition (Amazon Linux 2023 + Flask server + model) |
| `cloudformation.yaml` | Stage 2 template: full hosting infrastructure (VPC, ALB, ECS, IAM, Auto Scaling) |
| `deploy.sh` | One-command build/push/deploy script for hosting the trained model |
| `README.md` | This file |
