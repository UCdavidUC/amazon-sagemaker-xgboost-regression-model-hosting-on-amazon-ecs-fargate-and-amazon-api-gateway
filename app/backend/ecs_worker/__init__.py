"""Amazon ECS Fargate compute backend.

A long-running container that polls the ECS SQS queue, runs inference through
the shared :class:`InferenceService`, and updates DynamoDB. Runs on Graviton
(arm64) Fargate tasks across multiple Availability Zones and scales on SQS
backlog-per-task.
"""
