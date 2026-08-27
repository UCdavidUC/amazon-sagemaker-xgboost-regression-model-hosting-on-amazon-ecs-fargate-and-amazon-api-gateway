#!/usr/bin/env python3
"""CDK entrypoint for the multi-model inference platform.

Deploys the same architecture as the CloudFormation templates in
``app/infrastructure`` but as native CDK constructs, with CDK handling the
Lambda code and ECS image asset packaging.

Stacks (per environment):
  <prefix>-<env>-core        KMS, VPC + endpoints, DynamoDB, SQS + DLQs, log groups
  <prefix>-<env>-api-lambda  private REST API + submit Lambda + SQS-consumer worker
  <prefix>-<env>-ecs         Graviton Fargate worker service with backlog scaling
  <prefix>-cicd              CodePipeline (optional; deployCicd=true)

Usage:
  cdk synth  -c environment=dev
  cdk deploy -c environment=dev --all
  cdk deploy -c environment=qa -c deployCicd=true -c approvalEmail=you@example.com
"""
import os

import aws_cdk as cdk

from stacks.api_lambda_stack import ApiLambdaStack
from stacks.cicd_stack import CicdStack
from stacks.core_stack import CoreStack
from stacks.ecs_fargate_stack import EcsFargateStack

app = cdk.App()


def ctx(key: str, default: str = "") -> str:
    value = app.node.try_get_context(key)
    return default if value is None else str(value)


env_name = ctx("environment", "dev")
prefix = ctx("envName", "inference")
deploy_cicd = ctx("deployCicd", "false").lower() == "true"

aws_env = cdk.Environment(
    account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
    region=os.environ.get("CDK_DEFAULT_REGION"),
)

common = dict(env_name=env_name, prefix=prefix, env=aws_env)

core = CoreStack(app, f"{prefix}-{env_name}-core", **common)

api = ApiLambdaStack(app, f"{prefix}-{env_name}-api-lambda", core=core, **common)
api.add_dependency(core)

ecs = EcsFargateStack(
    app,
    f"{prefix}-{env_name}-ecs",
    core=core,
    image_uri=ctx("ecsImageUri", ""),
    **common,
)
ecs.add_dependency(core)

if deploy_cicd:
    CicdStack(
        app,
        f"{prefix}-cicd",
        core=core,
        prefix=prefix,
        ecr_repo_name=ctx("ecrRepo", "inference-worker"),
        approval_email=ctx("approvalEmail", "changeme@example.com"),
        qa_api_base_url=ctx("qaApiBaseUrl", ""),
        env=aws_env,
    )

for stack in (core, api, ecs):
    cdk.Tags.of(stack).add("Project", "multi-model-inference-api")
    cdk.Tags.of(stack).add("Environment", env_name)

app.synth()
