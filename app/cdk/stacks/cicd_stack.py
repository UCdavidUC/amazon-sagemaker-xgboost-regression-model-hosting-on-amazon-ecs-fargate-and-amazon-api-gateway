"""CI/CD pipeline.

Triggered by an ECR image push (e.g. SageMaker publishing a model image). Runs
SAST + unit tests, deploys to QA, runs API endpoint tests and a load test,
publishes a metrics-rich approval notification, waits for a manual approval,
then deploys to production. CodeBuild uses ARM containers that mirror the
production runtime; the verification projects run inside the VPC to reach the
private API.
"""
from aws_cdk import (
    Duration,
    RemovalPolicy,
    Stack,
)
from aws_cdk import aws_codebuild as codebuild
from aws_cdk import aws_codepipeline as codepipeline
from aws_cdk import aws_codepipeline_actions as cpactions
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_iam as iam
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_sns as sns
from aws_cdk import aws_sns_subscriptions as subs
from constructs import Construct


class CicdStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        core,
        prefix: str,
        ecr_repo_name: str,
        approval_email: str,
        qa_api_base_url: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        arm_env = codebuild.BuildEnvironment(
            build_image=codebuild.LinuxArmBuildImage.AMAZON_LINUX_2_STANDARD_3_0,
            compute_type=codebuild.ComputeType.SMALL,
        )
        vpc_subnets = ec2.SubnetSelection(
            subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
        )

        source_bucket = s3.Bucket(
            self,
            "SourceBucket",
            versioned=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.KMS,
            encryption_key=core.key,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        approval_topic = sns.Topic(
            self, "ApprovalTopic", topic_name=f"{prefix}-prod-approval", master_key=core.key
        )
        approval_topic.add_subscription(subs.EmailSubscription(approval_email))

        ecr_repo = ecr.Repository.from_repository_name(
            self, "EcrRepo", ecr_repo_name
        )

        # -- CodeBuild projects ----------------------------------------------
        sast = codebuild.PipelineProject(
            self,
            "Sast",
            project_name=f"{prefix}-sast",
            environment=arm_env,
            build_spec=codebuild.BuildSpec.from_source_filename(
                "app/cicd/buildspec-sast.yml"
            ),
            encryption_key=core.key,
        )

        api_test = codebuild.PipelineProject(
            self,
            "ApiTest",
            project_name=f"{prefix}-apitest",
            environment=arm_env,
            build_spec=codebuild.BuildSpec.from_source_filename(
                "app/cicd/buildspec-apitest.yml"
            ),
            environment_variables={
                "API_BASE_URL": codebuild.BuildEnvironmentVariable(value=qa_api_base_url)
            },
            vpc=core.vpc,
            subnet_selection=vpc_subnets,
            encryption_key=core.key,
        )

        load_test = codebuild.PipelineProject(
            self,
            "LoadTest",
            project_name=f"{prefix}-loadtest",
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxArmBuildImage.AMAZON_LINUX_2_STANDARD_3_0,
                compute_type=codebuild.ComputeType.MEDIUM,
            ),
            timeout=Duration.minutes(30),
            build_spec=codebuild.BuildSpec.from_source_filename(
                "app/cicd/buildspec-loadtest.yml"
            ),
            environment_variables={
                "API_BASE_URL": codebuild.BuildEnvironmentVariable(value=qa_api_base_url)
            },
            vpc=core.vpc,
            subnet_selection=vpc_subnets,
            encryption_key=core.key,
        )

        deploy_qa = self._deploy_project("DeployQa", prefix, "qa", arm_env, core.key)
        deploy_prod = self._deploy_project("DeployProd", prefix, "prod", arm_env, core.key)

        notify = codebuild.PipelineProject(
            self,
            "NotifyApproval",
            project_name=f"{prefix}-notify-approval",
            environment=arm_env,
            environment_variables={
                "APPROVAL_TOPIC_ARN": codebuild.BuildEnvironmentVariable(
                    value=approval_topic.topic_arn
                )
            },
            build_spec=codebuild.BuildSpec.from_object(
                {
                    "version": "0.2",
                    "phases": {
                        "install": {"runtime-versions": {"python": "3.12"}},
                        "build": {"commands": ["python app/cicd/notify_approval.py"]},
                    },
                }
            ),
            encryption_key=core.key,
        )
        approval_topic.grant_publish(notify.role)

        # Deploy projects need to roll ECS services and update Lambda code.
        for proj in (deploy_qa, deploy_prod):
            proj.add_to_role_policy(
                iam.PolicyStatement(
                    actions=[
                        "ecs:UpdateService",
                        "ecs:DescribeServices",
                        "ecs:DescribeTaskDefinition",
                        "ecs:RegisterTaskDefinition",
                        "lambda:UpdateFunctionCode",
                        "lambda:GetFunction",
                        "iam:PassRole",
                    ],
                    resources=["*"],
                )
            )

        # -- Artifacts + actions ---------------------------------------------
        source_artifact = codepipeline.Artifact("SourceOutput")
        image_artifact = codepipeline.Artifact("ImageOutput")
        sast_artifact = codepipeline.Artifact("SastOutput")

        pipeline = codepipeline.Pipeline(
            self,
            "Pipeline",
            pipeline_name=f"{prefix}-inference-pipeline",
            cross_account_keys=False,
            restart_execution_on_update=True,
        )

        pipeline.add_stage(
            stage_name="Source",
            actions=[
                cpactions.S3SourceAction(
                    action_name="SourceBundle",
                    bucket=source_bucket,
                    bucket_key="cicd/source.zip",
                    output=source_artifact,
                    trigger=cpactions.S3Trigger.NONE,
                ),
                cpactions.EcrSourceAction(
                    action_name="ContainerImage",
                    repository=ecr_repo,
                    image_tag="latest",
                    output=image_artifact,
                ),
            ],
        )
        pipeline.add_stage(
            stage_name="Build_And_Test",
            actions=[
                cpactions.CodeBuildAction(
                    action_name="SAST_UnitTests",
                    project=sast,
                    input=source_artifact,
                    outputs=[sast_artifact],
                )
            ],
        )
        pipeline.add_stage(
            stage_name="Deploy_QA",
            actions=[
                cpactions.CodeBuildAction(
                    action_name="DeployToQA", project=deploy_qa, input=source_artifact
                )
            ],
        )
        pipeline.add_stage(
            stage_name="Verify_QA",
            actions=[
                cpactions.CodeBuildAction(
                    action_name="ApiEndpointTests",
                    project=api_test,
                    input=source_artifact,
                    run_order=1,
                ),
                cpactions.CodeBuildAction(
                    action_name="LoadTest",
                    project=load_test,
                    input=source_artifact,
                    run_order=1,
                ),
            ],
        )
        pipeline.add_stage(
            stage_name="Approve_Prod",
            actions=[
                cpactions.CodeBuildAction(
                    action_name="PublishMetrics",
                    project=notify,
                    input=sast_artifact,
                    run_order=1,
                ),
                cpactions.ManualApprovalAction(
                    action_name="ManualApproval",
                    notification_topic=approval_topic,
                    additional_information=(
                        "Review the SAST, API test, and load-test results emailed "
                        "by the PublishMetrics step before approving production."
                    ),
                    run_order=2,
                ),
            ],
        )
        pipeline.add_stage(
            stage_name="Deploy_Prod",
            actions=[
                cpactions.CodeBuildAction(
                    action_name="DeployToProd",
                    project=deploy_prod,
                    input=source_artifact,
                )
            ],
        )

    def _deploy_project(self, cid, prefix, target_env, arm_env, key):
        return codebuild.PipelineProject(
            self,
            cid,
            project_name=f"{prefix}-deploy-{target_env}",
            environment=arm_env,
            environment_variables={
                "TARGET_ENV": codebuild.BuildEnvironmentVariable(value=target_env),
                "ECS_CLUSTER": codebuild.BuildEnvironmentVariable(
                    value=f"{prefix}-{target_env}-cluster"
                ),
                "ECS_SERVICE": codebuild.BuildEnvironmentVariable(
                    value=f"{prefix}-{target_env}-worker"
                ),
            },
            build_spec=codebuild.BuildSpec.from_object(
                {
                    "version": "0.2",
                    "phases": {
                        "build": {
                            "commands": [
                                'echo "Deploying image to $TARGET_ENV ECS service $ECS_SERVICE"',
                                'aws ecs update-service --cluster "$ECS_CLUSTER" '
                                '--service "$ECS_SERVICE" --force-new-deployment',
                                'aws ecs wait services-stable --cluster "$ECS_CLUSTER" '
                                '--services "$ECS_SERVICE"',
                            ]
                        }
                    },
                }
            ),
            encryption_key=key,
        )
