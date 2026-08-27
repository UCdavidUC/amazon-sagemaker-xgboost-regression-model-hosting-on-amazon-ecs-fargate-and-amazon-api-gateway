# Infrastructure (CloudFormation)

Four stacks compose the multi-model inference platform. They deploy per
environment (`dev`, `qa`, `prod`) and communicate through CloudFormation
exports named `inference-<env>-*`.

| Stack | File | Creates |
|---|---|---|
| Core | `00-core.yaml` | KMS CMK, private VPC (2 private + 2 public subnets, NAT), Gateway + Interface VPC endpoints (S3, DynamoDB, SQS, ECR, Logs, KMS), DynamoDB requests table, SQS queues + DLQs, custom CloudWatch log groups |
| API + Lambda | `10-api-lambda.yaml` | Private REST API Gateway (execute-api VPC endpoint, IAM auth), submit/status API Lambda, Lambda SQS-consumer worker + event source mapping, per-env stage with access logging |
| ECS Fargate | `20-ecs-fargate.yaml` | ECS cluster, Graviton (arm64) task def (2 vCPU / 8 GB), queue-driven worker service across 2 AZs, SQS backlog-per-task autoscaling |
| CI/CD | `30-cicd.yaml` | CodePipeline (ECR-push triggered), CodeBuild SAST/API-test/load-test/deploy projects, SNS approval topic, EventBridge trigger |

## Data protection

- **At rest:** the customer-managed KMS key encrypts DynamoDB, both SQS queues
  and their DLQs, the CloudWatch log groups, CI/CD artifacts, and the SNS topic.
- **In transit:** API Gateway is HTTPS-only; the SQS queue policies deny any
  request where `aws:SecureTransport` is false; all AWS service calls stay on
  the private network through VPC endpoints.
- **Network:** compute runs in private subnets with no public IPs. The API is a
  PRIVATE API Gateway reachable only through the `execute-api` interface
  endpoint, intended to be fronted by the public-subnet frontend/proxy tier.

## Deploy

```bash
# 1. Core + API/Lambda + ECS backend for an environment (builds the arm64 image).
cd app/infrastructure
./deploy-app.sh \
  --environment dev \
  --region us-east-1 \
  --code-bucket <existing-s3-bucket-for-code> \
  --ecr-repo inference-worker

# 2. CI/CD pipeline (deploy once; reads the core CMK + private subnets).
aws cloudformation deploy \
  --template-file 30-cicd.yaml \
  --stack-name inference-cicd \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
      EnvironmentName=inference \
      EcrRepoName=inference-worker \
      SourceBucket=<source-bundle-bucket> \
      KmsKeyArn=<core-stack KmsKeyArn output> \
      QaApiBaseUrl=<qa PrivateInvokeUrl output> \
      ApprovalEmail=you@example.com \
      VpcId=<core VpcId> \
      PrivateSubnetIds=<subnet-a,subnet-b> \
  --region us-east-1
```

## Scaling model (200 requests/minute target)

The ECS worker scales on **SQS backlog per task** using target tracking with
metric math (`ApproximateNumberOfMessagesVisible / RunningTaskCount`), the
AWS-recommended pattern for queue-driven workers. With a target of 30 queued
messages per task and each 2 vCPU task processing several inferences
concurrently, the service comfortably absorbs 200 requests/min and bursts well
beyond it up to `MaxTasks`. A secondary CPU target-tracking policy guards
against compute-bound tasks. The SQS visibility timeout (300 s default) is set
above the longest expected processing time to prevent duplicate delivery.

## Validation

Templates are linted with [cfn-lint](https://github.com/aws-cloudformation/cfn-lint):

```bash
cfn-lint infrastructure/*.yaml
```
