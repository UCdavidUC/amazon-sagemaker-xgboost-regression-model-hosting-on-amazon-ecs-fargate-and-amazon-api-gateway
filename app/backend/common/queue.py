"""SQS producer helper.

The submit handler uses this to place a :class:`QueueMessage` on the correct
per-backend queue. The SQS client is injectable for unit tests.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from .config import Config, load_config
from .errors import UnknownRouteError
from .schemas import QueueMessage


class QueuePublisher:
    def __init__(self, client: Any = None, config: Optional[Config] = None) -> None:
        self.config = config or load_config()
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            import boto3  # lazy import

            self._client = boto3.client("sqs", region_name=self.config.aws_region)
        return self._client

    def publish(self, message: QueueMessage) -> str:
        """Send the message to the queue for its backend; returns the SQS MessageId."""
        queue_url = self.config.queue_url_for(message.backend)
        if not queue_url:
            raise UnknownRouteError(
                f"No queue configured for backend '{message.backend}'"
            )
        response = self.client.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps(message.to_json_dict()),
        )
        return response.get("MessageId", "")
