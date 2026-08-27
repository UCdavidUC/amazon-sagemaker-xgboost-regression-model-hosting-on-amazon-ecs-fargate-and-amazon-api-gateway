"""Lambda SQS consumer for the ``lambda`` backend.

Referenced by CloudFormation as ``backend.lambda_worker.handler.lambda_handler``
and wired to the Lambda queue through an SQS event-source mapping with
``FunctionResponseTypes: [ReportBatchItemFailures]`` enabled.

For each record it runs :meth:`InferenceService.process`. A record that fails
transiently is reported back to Lambda so only that message is retried; the
rest of the batch is acknowledged. Messages that keep failing are moved to the
dead-letter queue by SQS after ``maxReceiveCount`` attempts.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

from ..common.inference_service import InferenceService
from ..common.schemas import QueueMessage

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

_SERVICE = InferenceService()


def lambda_handler(event, context):  # noqa: ANN001 - AWS signature
    """Process an SQS batch and report per-message failures."""
    failures: List[Dict[str, str]] = []
    for record in event.get("Records", []):
        message_id = record.get("messageId", "")
        try:
            body = json.loads(record["body"])
            message = QueueMessage.from_json_dict(body)
            _SERVICE.process(message)
        except Exception:
            # Transient failures (permanent ones are swallowed inside process()
            # after being marked FAILED). Report so SQS redelivers this message.
            logger.exception("Failed to process message %s", message_id)
            failures.append({"itemIdentifier": message_id})
    return {"batchItemFailures": failures}
