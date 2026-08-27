"""API layer + Lambda compute backend.

Private REST API Gateway (reachable only through an execute-api interface
endpoint) fronting the submit/status Lambda, plus the SQS-consumer worker Lambda
with partial-batch failure reporting.
"""
from aws_cdk import (
    CfnOutput,
    Duration,
    Stack,
)
from aws_cdk import aws_apigateway as apigateway
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_lambda as _lambda
from aws_cdk.aws_lambda_event_sources import SqsEventSource
from constructs import Construct

from ._common import APP_DIR, LAMBDA_ASSET_EXCLUDES


class ApiLambdaStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        core,
        env_name: str,
        prefix: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        code = _lambda.Code.from_asset(APP_DIR, exclude=LAMBDA_ASSET_EXCLUDES)
        common_env = {
            "ENVIRONMENT": env_name,
            "REQUESTS_TABLE_NAME": core.requests_table.table_name,
            "LOG_LEVEL": "INFO",
        }

        # -- API Lambda (submit / status / catalog) --------------------------
        api_fn = _lambda.Function(
            self,
            "ApiFunction",
            function_name=f"{prefix}-{env_name}-api",
            runtime=_lambda.Runtime.PYTHON_3_12,
            architecture=_lambda.Architecture.ARM_64,
            handler="backend.api.handler.lambda_handler",
            code=code,
            memory_size=1024,
            timeout=Duration.seconds(120),
            vpc=core.vpc,
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
            ),
            environment={
                **common_env,
                "LAMBDA_QUEUE_URL": core.lambda_queue.queue_url,
                "ECS_QUEUE_URL": core.ecs_queue.queue_url,
            },
        )
        core.requests_table.grant_read_write_data(api_fn)
        core.lambda_queue.grant_send_messages(api_fn)
        core.ecs_queue.grant_send_messages(api_fn)

        # -- Lambda worker (SQS consumer) ------------------------------------
        worker_fn = _lambda.Function(
            self,
            "WorkerFunction",
            function_name=f"{prefix}-{env_name}-lambda-worker",
            runtime=_lambda.Runtime.PYTHON_3_12,
            architecture=_lambda.Architecture.ARM_64,
            handler="backend.lambda_worker.handler.lambda_handler",
            code=code,
            memory_size=1024,
            timeout=Duration.seconds(120),
            vpc=core.vpc,
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
            ),
            environment={**common_env, "MODEL_DIR": "/opt/models"},
        )
        core.requests_table.grant_read_write_data(worker_fn)
        worker_fn.add_event_source(
            SqsEventSource(
                core.lambda_queue,
                batch_size=10,
                max_batching_window=Duration.seconds(5),
                report_batch_item_failures=True,
                max_concurrency=20,
            )
        )

        # -- Private REST API ------------------------------------------------
        execute_api_endpoint = ec2.InterfaceVpcEndpoint(
            self,
            "ExecuteApiEndpoint",
            vpc=core.vpc,
            service=ec2.InterfaceVpcEndpointAwsService.APIGATEWAY,
            private_dns_enabled=True,
            subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
            ),
        )

        from aws_cdk import aws_iam as iam

        api_policy = iam.PolicyDocument(
            statements=[
                iam.PolicyStatement(
                    effect=iam.Effect.DENY,
                    principals=[iam.AnyPrincipal()],
                    actions=["execute-api:Invoke"],
                    resources=["execute-api:/*"],
                    conditions={
                        "StringNotEquals": {
                            "aws:SourceVpce": execute_api_endpoint.vpc_endpoint_id
                        }
                    },
                ),
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    principals=[iam.AnyPrincipal()],
                    actions=["execute-api:Invoke"],
                    resources=["execute-api:/*"],
                ),
            ]
        )

        self.api = apigateway.LambdaRestApi(
            self,
            "RestApi",
            rest_api_name=f"{prefix}-{env_name}-api",
            handler=api_fn,
            proxy=True,
            cloud_watch_role=True,
            endpoint_configuration=apigateway.EndpointConfiguration(
                types=[apigateway.EndpointType.PRIVATE],
                vpc_endpoints=[execute_api_endpoint],
            ),
            policy=api_policy,
            default_method_options=apigateway.MethodOptions(
                authorization_type=apigateway.AuthorizationType.IAM
            ),
            deploy_options=apigateway.StageOptions(
                stage_name=env_name,
                access_log_destination=apigateway.LogGroupLogDestination(
                    core.api_access_log_group
                ),
                access_log_format=apigateway.AccessLogFormat.json_with_standard_fields(
                    caller=False,
                    http_method=True,
                    ip=True,
                    protocol=True,
                    request_time=True,
                    resource_path=True,
                    response_length=True,
                    status=True,
                    user=False,
                ),
                logging_level=apigateway.MethodLoggingLevel.INFO,
                metrics_enabled=True,
                tracing_enabled=True,
                throttling_rate_limit=100,
                throttling_burst_limit=200,
            ),
        )

        CfnOutput(
            self,
            "PrivateInvokeUrl",
            value=f"{self.api.url}api",
            description="Private API base URL (reachable from within the VPC).",
        )
        CfnOutput(self, "ApiId", value=self.api.rest_api_id)
