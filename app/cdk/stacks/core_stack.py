"""Core shared infrastructure: KMS, VPC + endpoints, DynamoDB, SQS, log groups."""
from aws_cdk import (
    Duration,
    RemovalPolicy,
    Stack,
)
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_kms as kms
from aws_cdk import aws_logs as logs
from aws_cdk import aws_sqs as sqs
from constructs import Construct


class CoreStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        env_name: str,
        prefix: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        self.env_name = env_name
        self.prefix = prefix
        is_prod = env_name == "prod"
        removal = RemovalPolicy.RETAIN if is_prod else RemovalPolicy.DESTROY

        # -- KMS customer managed key (encryption at rest) --------------------
        self.key = kms.Key(
            self,
            "EncryptionKey",
            description=f"CMK for {prefix} inference platform ({env_name})",
            enable_key_rotation=True,
            alias=f"alias/{prefix}-{env_name}",
            removal_policy=removal,
        )

        # -- VPC (private by default, 2 AZs, single NAT) ----------------------
        self.vpc = ec2.Vpc(
            self,
            "Vpc",
            ip_addresses=ec2.IpAddresses.cidr("10.20.0.0/16"),
            max_azs=2,
            nat_gateways=1,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="public", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name="private",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=24,
                ),
            ],
        )

        # Gateway endpoints (free) keep S3/DynamoDB traffic on the private net.
        self.vpc.add_gateway_endpoint(
            "S3Endpoint", service=ec2.GatewayVpcEndpointAwsService.S3
        )
        self.vpc.add_gateway_endpoint(
            "DynamoDbEndpoint", service=ec2.GatewayVpcEndpointAwsService.DYNAMODB
        )
        # Interface endpoints for the control/data plane services used privately.
        for name, service in (
            ("SqsEndpoint", ec2.InterfaceVpcEndpointAwsService.SQS),
            ("EcrApiEndpoint", ec2.InterfaceVpcEndpointAwsService.ECR),
            ("EcrDkrEndpoint", ec2.InterfaceVpcEndpointAwsService.ECR_DOCKER),
            ("LogsEndpoint", ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH_LOGS),
            ("KmsEndpoint", ec2.InterfaceVpcEndpointAwsService.KMS),
        ):
            self.vpc.add_interface_endpoint(name, service=service)

        # -- DynamoDB request-tracking table ---------------------------------
        self.requests_table = dynamodb.Table(
            self,
            "RequestsTable",
            table_name=f"{prefix}-{env_name}-requests",
            partition_key=dynamodb.Attribute(
                name="request_id", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            time_to_live_attribute="ttl",
            point_in_time_recovery=True,
            encryption=dynamodb.TableEncryption.CUSTOMER_MANAGED,
            encryption_key=self.key,
            removal_policy=removal,
        )

        # -- SQS queues + DLQs (one pair per backend) ------------------------
        self.lambda_queue = self._make_queue("Lambda", f"{prefix}-{env_name}-lambda")
        self.ecs_queue = self._make_queue("Ecs", f"{prefix}-{env_name}-ecs")

        # -- Custom CloudWatch log groups ------------------------------------
        self.api_access_log_group = logs.LogGroup(
            self,
            "ApiAccessLogGroup",
            log_group_name=f"/inference/{env_name}/api-access",
            retention=logs.RetentionDays.ONE_MONTH,
            encryption_key=self.key,
            removal_policy=removal,
        )
        self.application_log_group = logs.LogGroup(
            self,
            "ApplicationLogGroup",
            log_group_name=f"/inference/{env_name}/application",
            retention=logs.RetentionDays.ONE_MONTH,
            encryption_key=self.key,
            removal_policy=removal,
        )

    def _make_queue(self, id_prefix: str, name_prefix: str) -> sqs.Queue:
        dlq = sqs.Queue(
            self,
            f"{id_prefix}Dlq",
            queue_name=f"{name_prefix}-dlq",
            encryption=sqs.QueueEncryption.KMS,
            encryption_master_key=self.key,
            enforce_ssl=True,
            retention_period=Duration.days(14),
        )
        return sqs.Queue(
            self,
            f"{id_prefix}Queue",
            queue_name=f"{name_prefix}-queue",
            visibility_timeout=Duration.seconds(300),
            retention_period=Duration.days(4),
            encryption=sqs.QueueEncryption.KMS,
            encryption_master_key=self.key,
            enforce_ssl=True,
            dead_letter_queue=sqs.DeadLetterQueue(max_receive_count=5, queue=dlq),
        )
