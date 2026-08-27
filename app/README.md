# Multi-Model Inference API

An asynchronous API component that exposes several ML models through a single
endpoint and runs inference on two interchangeable compute backends: **AWS
Lambda** and **Amazon ECS Fargate**. Request status, input, and output are
tracked in **Amazon DynamoDB**, and processing is decoupled from the API with
**Amazon SQS**.

This document is both the specification and the description of what has been
built. See [Improvements to the original specification](#improvements-to-the-original-specification)
for the researched refinements.

## Objective

Expose a set of ML models that can be invoked from a single, versioned endpoint
through Amazon API Gateway, with the same routes fronting two separate compute
backends (Lambda and ECS Fargate on Graviton).

## Architecture

The API is **asynchronous**. A submission is accepted, queued, and answered with
a `request_id`; the result is retrieved by polling a status endpoint. This keeps
the API fast and lets each backend absorb bursts through its queue.

```text
Client ─► API Gateway (PRIVATE) ─► API Lambda
                                       │  1. write QUEUED record  ─► DynamoDB
                                       │  2. enqueue              ─► SQS (per backend)
                                       ▼
                        ┌──────────────┴───────────────┐
                        ▼                               ▼
              Lambda worker (SQS ESM)        ECS Fargate worker (SQS poll)
                        │                               │
                        └──── 3. run model, update ─────┘ ─► DynamoDB (COMPLETED/FAILED)

Client ─► GET /api/requests/{id} ─► API Lambda ─► DynamoDB
```

Diagrams (Draw.io):
[`docs/diagrams/inference-api-architecture.drawio`](../docs/diagrams/inference-api-architecture.drawio)
and [`docs/diagrams/cicd-pipeline.drawio`](../docs/diagrams/cicd-pipeline.drawio).

## API routes

`GET /api` returns the list of available operations as JSON.

```text
/api
  /health                                     GET  liveness probe
  /backend
    /lambda
      /model
        /weighted    POST   submit to weighted model on the Lambda backend
        /arima       POST   submit to arima    model on the Lambda backend
        /sarima      POST   submit to sarima   model on the Lambda backend
        /xgboost     POST   submit to xgboost  model on the Lambda backend
    /ecs-fargate
      /model
        /weighted    POST   submit to weighted model on the ECS Fargate backend
        /arima       POST   ...
        /sarima      POST   ...
        /xgboost     POST   ...
  /requests/{request_id}                      GET  status, input, and output
```

A `POST` returns `202 Accepted` with `{ request_id, status, status_url }`.
Full contract: [`backend/api/openapi.yaml`](backend/api/openapi.yaml).

### Request bodies

Tabular models (`weighted`, `xgboost`) — eight standardized California Housing
features, as an array or the legacy CSV string:

```json
{ "features": [0.12, -0.45, 0.78, -0.23, 0.56, -0.89, 1.23, -0.67] }
```

Time-series models (`arima`, `sarima`) — forecast horizon (and optional
exogenous regressors for `sarima`):

```json
{ "steps": 5 }
```

## Models

| Model | Purpose | Notes |
|---|---|---|
| `weighted` | Transparent weighted-linear baseline / ensemble primitive | numpy only; always available; accepts per-request `weights` |
| `xgboost` | XGBoost regression (California Housing) | uses the trained SageMaker booster artifact |
| `arima` | Non-seasonal ARIMA forecast | statsmodels results artifact |
| `sarima` | Seasonal SARIMA/SARIMAX forecast | supports exogenous variables |

Models are pluggable: implement `InferenceModel`, register it in
[`backend/common/models/registry.py`](backend/common/models/registry.py), and
add its name to `VALID_MODELS`. Heavy dependencies (xgboost, statsmodels) are
imported lazily so the API and validation paths stay lightweight, and the
`weighted`/time-series models fall back to a deterministic result when no
artifact is present, so the whole pipeline is smoke-testable end to end.

## Code layout

```text
app/
  backend/
    common/          shared: config, errors, schemas, DynamoDB repo, SQS publisher,
                     inference service, and the pluggable model implementations
    api/             API Gateway Lambda (router + handler) and OpenAPI spec
    lambda_worker/   Lambda SQS-consumer backend (+ optional container Dockerfile)
    ecs_worker/      ECS Fargate SQS-poller backend (worker, healthcheck, Dockerfile)
  cicd/              buildspecs, smoke test, load test (Locust), approval notifier
  infrastructure/    CloudFormation stacks + deploy-app.sh
  cdk/               AWS CDK (Python) app that deploys the same architecture
  tests/             unit tests (stdlib unittest)
```

## Requirements (as built)

### API
- Private API Gateway REST API, reachable only through an `execute-api`
  interface VPC endpoint, intended to be fronted by the public-subnet
  frontend/proxy tier. Compute runs in private subnets.
- Data secured in transit (HTTPS, TLS-only SQS policies, VPC endpoints) and at
  rest (customer-managed KMS CMK across DynamoDB, SQS/DLQ, logs, artifacts).
- Requests logged to a **custom** CloudWatch access log group via CloudFormation.
- Per-environment stages: `dev`, `qa`, `prod`.

### Lambda
- Runs model inference as an SQS consumer (submit path + worker).
- IAM scoped to DynamoDB, SQS, KMS, and optional S3 model reads.
- Memory ≥ 1024 MB; **Python 3.12** runtime; arm64 (Graviton) architecture.

### ECS / Fargate
- Graviton (arm64) tasks, **2 vCPU / 8 GB** minimum, across two AZs.
- Fargate launch type on ECS; queue-driven worker (no inbound port).
- Scales on **SQS backlog-per-task** (target tracking with metric math) to
  absorb 200+ inference requests/minute.

### CI/CD
- Triggered by an ECR image push (EventBridge) — e.g. SageMaker publishing a new
  model image.
- Multi-step CodePipeline: SAST + unit tests → deploy QA → API endpoint tests +
  load test → metrics-rich SNS approval notification → manual approval → deploy
  prod.
- CodeBuild uses ARM containers that mirror the production runtime; the load-test
  stage follows AWS Well-Architected guidance (prod-like target, percentile and
  error-budget thresholds).

## Deploy

Two equivalent paths provision the same architecture — pick one.

**AWS CDK (Python)** — see [`cdk/README.md`](cdk/README.md):

```bash
cd app/cdk
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
cdk deploy -c environment=dev --all \
  -c ecsImageUri=<acct>.dkr.ecr.<region>.amazonaws.com/inference-worker:latest
```

**Raw CloudFormation** — see [`infrastructure/README.md`](infrastructure/README.md):

```bash
cd app/infrastructure
./deploy-app.sh --environment dev --region us-east-1 \
  --code-bucket <code-bucket> --ecr-repo inference-worker
```

## Test

```bash
cd app
python3 -m unittest discover -s tests -p "test_*.py" -v   # 36 unit tests
cfn-lint infrastructure/*.yaml                            # template linting
```

## Improvements to the original specification

Researched and validated against current AWS guidance; changes made and why:

1. **Asynchronous, queue-decoupled design.** The original routes implied
   synchronous per-model endpoints. To satisfy "DynamoDB stores request status,
   input and output as inference is made" and "SQS manages those states and
   processed messages", the API now returns `202 + request_id` and workers
   update DynamoDB. This isolates the API from slow models and lets each backend
   scale on its own queue.
2. **Backlog-per-task autoscaling for ECS.** Replaced CPU/ALB-request scaling
   with SQS backlog-per-task target tracking (`messages / running tasks`) using
   metric math — the AWS-recommended pattern for queue workers — sized for the
   200 req/min target. A CPU policy remains as a safety net.
3. **Graviton everywhere.** Both the ECS tasks and the Lambda functions run on
   arm64 for better price/performance.
4. **Private API + VPC endpoints.** The API is a PRIVATE API Gateway behind an
   `execute-api` endpoint; S3, DynamoDB, SQS, ECR, Logs, and KMS are reached over
   VPC endpoints, keeping traffic off the public internet.
5. **Customer-managed KMS key** for encryption at rest across all stateful
   services, and TLS-only enforcement on the queues for encryption in transit.
6. **Partial-batch failure handling and DLQs.** The Lambda consumer reports
   per-message failures (`ReportBatchItemFailures`) and both queues have DLQs
   with a bounded `maxReceiveCount`, so poison messages do not block the queue.
7. **Runtime note.** The spec requires Python 3.12; AWS Lambda now also offers
   3.13 and 3.14 if a future bump is desired. We stay on 3.12 as specified.
8. **Load testing as a gated stage** with explicit p95-latency and error-budget
   thresholds, plus an approval notification carrying pipeline runtime, SAST and
   dependency-vulnerability counts, and load-test metrics.

Content in this section was informed by AWS documentation on
[SQS-based ECS/Application Auto Scaling with metric math](https://docs.aws.amazon.com/autoscaling/application/userguide/application-auto-scaling-target-tracking-metric-math.html)
and the [AWS Lambda Python runtimes](https://docs.aws.amazon.com/lambda/latest/dg/lambda-runtimes.html);
it was rephrased for compliance with licensing restrictions.
