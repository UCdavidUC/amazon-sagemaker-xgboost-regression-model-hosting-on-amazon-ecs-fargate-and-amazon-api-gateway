# CDK deployment (Python)

Deploys the multi-model inference platform with the AWS CDK. This is an
alternative to the raw CloudFormation templates in
[`../infrastructure`](../infrastructure) and provisions the same architecture,
with CDK handling Lambda code and ECS image asset packaging.

## Stacks

| Stack (id) | Construct | Provisions |
|---|---|---|
| `<prefix>-<env>-core` | `CoreStack` | KMS CMK, private VPC + gateway/interface endpoints, DynamoDB table, SQS queues + DLQs, custom log groups |
| `<prefix>-<env>-api-lambda` | `ApiLambdaStack` | private REST API (execute-api endpoint, IAM auth, access logging), submit Lambda, SQS-consumer worker Lambda |
| `<prefix>-<env>-ecs` | `EcsFargateStack` | Graviton Fargate worker service (2 vCPU / 8 GB), SQS backlog-per-task autoscaling |
| `<prefix>-cicd` | `CicdStack` | CodePipeline (ECR-push triggered), SAST/API-test/load-test/deploy projects, SNS approval |

Defaults: `prefix=inference`, `env=dev`.

## Prerequisites

- Node.js and the AWS CDK CLI (`npm install -g aws-cdk`).
- Python 3 and a virtual environment for the CDK library.
- Docker **only if** you let CDK build the ECS image locally (see below).
- AWS credentials and a bootstrapped environment (`cdk bootstrap`).

## Setup

```bash
cd app/cdk
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Synthesize and deploy

```bash
# Synthesize (no Docker needed if you pass a prebuilt image URI).
cdk synth -c environment=dev \
  -c ecsImageUri=<acct>.dkr.ecr.<region>.amazonaws.com/inference-worker:latest

# Deploy the runtime stacks for an environment.
cdk deploy -c environment=dev --all \
  -c ecsImageUri=<acct>.dkr.ecr.<region>.amazonaws.com/inference-worker:latest

# Deploy the CI/CD pipeline as well.
cdk deploy inference-cicd \
  -c deployCicd=true \
  -c ecrRepo=inference-worker \
  -c approvalEmail=you@example.com \
  -c qaApiBaseUrl=https://<api-id>.execute-api.<region>.amazonaws.com/qa/api
```

## Context flags

| Flag | Default | Purpose |
|---|---|---|
| `environment` | `dev` | Target environment (`dev`/`qa`/`prod`); drives naming and removal policies |
| `envName` | `inference` | Resource name prefix |
| `ecsImageUri` | *(empty)* | Prebuilt ECS worker image. If empty, CDK builds `backend/ecs_worker/Dockerfile` locally (requires Docker) |
| `deployCicd` | `false` | Also deploy the `CicdStack` |
| `ecrRepo` | `inference-worker` | ECR repository the pipeline watches |
| `approvalEmail` | — | Email subscribed to the production-approval SNS topic |
| `qaApiBaseUrl` | — | QA API base URL used by the endpoint and load tests |

## Image packaging

The ECS worker image can be supplied two ways:

- **Prebuilt (recommended for CI/CD):** pass `-c ecsImageUri=...`. The image the
  pipeline builds and pushes to ECR is used directly, and `cdk synth`/`deploy`
  need no local Docker.
- **Local asset build:** omit `ecsImageUri` and CDK builds
  `backend/ecs_worker/Dockerfile` for `linux/arm64` (requires Docker/buildx).

Lambda code is packaged from the `backend/` package as a zip asset. Heavy model
runtimes (xgboost, statsmodels) are best served by the ECS backend; to run them
on the Lambda backend, attach a layer or switch the worker function to a
container image.

## Notes

- Cross-stack wiring uses direct construct references (CDK generates the
  necessary exports), so there is no manual `Fn::ImportValue` as in the raw
  templates.
- The `CoreStack` is env-agnostic and needs no `cdk` context lookups, so
  `cdk synth` works without AWS credentials when a prebuilt `ecsImageUri` is
  supplied.
