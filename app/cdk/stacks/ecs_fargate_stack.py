"""ECS Fargate compute backend.

A queue-driven worker on Graviton (arm64) Fargate tasks (2 vCPU / 8 GB) across
two AZs, scaling on SQS backlog-per-task with a CloudWatch metric-math target
tracking policy. The metric-math target tracking is expressed with the L1
Application Auto Scaling constructs because the L2 helper only supports direct
(single) metrics.
"""
from aws_cdk import (
    Duration,
    Stack,
)
from aws_cdk import aws_applicationautoscaling as appscaling
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecs as ecs
from constructs import Construct

from ._common import APP_DIR, ECS_DOCKERFILE

_P = appscaling.CfnScalingPolicy


class EcsFargateStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        core,
        env_name: str,
        prefix: str,
        image_uri: str = "",
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        cluster = ecs.Cluster(
            self,
            "Cluster",
            cluster_name=f"{prefix}-{env_name}-cluster",
            vpc=core.vpc,
            container_insights=True,
        )

        task_def = ecs.FargateTaskDefinition(
            self,
            "TaskDef",
            family=f"{prefix}-{env_name}-worker",
            cpu=2048,
            memory_limit_mib=8192,
            runtime_platform=ecs.RuntimePlatform(
                cpu_architecture=ecs.CpuArchitecture.ARM64,
                operating_system_family=ecs.OperatingSystemFamily.LINUX,
            ),
        )

        # Use a prebuilt image (e.g. the one the CI/CD pipeline pushes to ECR)
        # when an image URI is supplied; otherwise build the arm64 asset locally.
        # Providing a URI also lets `cdk synth` run without a local Docker daemon.
        if image_uri:
            image = ecs.ContainerImage.from_registry(image_uri)
        else:
            from aws_cdk.aws_ecr_assets import Platform

            image = ecs.ContainerImage.from_asset(
                directory=APP_DIR, file=ECS_DOCKERFILE, platform=Platform.LINUX_ARM64
            )

        task_def.add_container(
            "worker",
            image=image,
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="ecs-worker",
                log_group=core.application_log_group,
            ),
            health_check=ecs.HealthCheck(
                command=[
                    "CMD-SHELL",
                    "python -m backend.ecs_worker.healthcheck || exit 1",
                ],
                interval=Duration.seconds(30),
                timeout=Duration.seconds(5),
                retries=3,
                start_period=Duration.seconds(20),
            ),
            environment={
                "ENVIRONMENT": env_name,
                "REQUESTS_TABLE_NAME": core.requests_table.table_name,
                "ECS_QUEUE_URL": core.ecs_queue.queue_url,
                "MODEL_DIR": "/opt/models",
                "LOG_LEVEL": "INFO",
                "MAX_WORKERS": "8",
            },
        )

        core.requests_table.grant_read_write_data(task_def.task_role)
        core.ecs_queue.grant_consume_messages(task_def.task_role)

        service = ecs.FargateService(
            self,
            "Service",
            service_name=f"{prefix}-{env_name}-worker",
            cluster=cluster,
            task_definition=task_def,
            desired_count=3,
            assign_public_ip=False,
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
            ),
            min_healthy_percent=100,
            max_healthy_percent=200,
        )

        self._add_scaling(cluster, service, core, prefix, env_name)

    def _add_scaling(self, cluster, service, core, prefix, env_name) -> None:
        # App Auto Scaling target. RoleARN omitted so AWS uses the ECS service-
        # linked role automatically.
        target = appscaling.CfnScalableTarget(
            self,
            "ScalableTarget",
            max_capacity=25,
            min_capacity=2,
            resource_id=f"service/{cluster.cluster_name}/{service.service_name}",
            scalable_dimension="ecs:service:DesiredCount",
            service_namespace="ecs",
        )
        target.node.add_dependency(service)

        # Backlog per task = visible messages / running tasks (metric math).
        appscaling.CfnScalingPolicy(
            self,
            "BacklogPolicy",
            policy_name=f"{prefix}-{env_name}-backlog-per-task",
            policy_type="TargetTrackingScaling",
            scaling_target_id=target.ref,
            target_tracking_scaling_policy_configuration=_P.TargetTrackingScalingPolicyConfigurationProperty(
                target_value=30,
                scale_in_cooldown=120,
                scale_out_cooldown=30,
                customized_metric_specification=_P.CustomizedMetricSpecificationProperty(
                    metrics=[
                        _P.TargetTrackingMetricDataQueryProperty(
                            id="backlog",
                            label="BacklogPerTask",
                            expression="messages / tasks",
                            return_data=True,
                        ),
                        _P.TargetTrackingMetricDataQueryProperty(
                            id="messages",
                            return_data=False,
                            metric_stat=_P.TargetTrackingMetricStatProperty(
                                stat="Average",
                                metric=_P.TargetTrackingMetricProperty(
                                    namespace="AWS/SQS",
                                    metric_name="ApproximateNumberOfMessagesVisible",
                                    dimensions=[
                                        _P.TargetTrackingMetricDimensionProperty(
                                            name="QueueName",
                                            value=core.ecs_queue.queue_name,
                                        )
                                    ],
                                ),
                            ),
                        ),
                        _P.TargetTrackingMetricDataQueryProperty(
                            id="tasks",
                            return_data=False,
                            metric_stat=_P.TargetTrackingMetricStatProperty(
                                stat="Average",
                                metric=_P.TargetTrackingMetricProperty(
                                    namespace="ECS/ContainerInsights",
                                    metric_name="RunningTaskCount",
                                    dimensions=[
                                        _P.TargetTrackingMetricDimensionProperty(
                                            name="ClusterName", value=cluster.cluster_name
                                        ),
                                        _P.TargetTrackingMetricDimensionProperty(
                                            name="ServiceName", value=service.service_name
                                        ),
                                    ],
                                ),
                            ),
                        ),
                    ],
                ),
            ),
        )

        # CPU safety-net target tracking.
        appscaling.CfnScalingPolicy(
            self,
            "CpuPolicy",
            policy_name=f"{prefix}-{env_name}-cpu",
            policy_type="TargetTrackingScaling",
            scaling_target_id=target.ref,
            target_tracking_scaling_policy_configuration=_P.TargetTrackingScalingPolicyConfigurationProperty(
                target_value=65,
                scale_in_cooldown=120,
                scale_out_cooldown=30,
                predefined_metric_specification=_P.PredefinedMetricSpecificationProperty(
                    predefined_metric_type="ECSServiceAverageCPUUtilization",
                ),
            ),
        )
