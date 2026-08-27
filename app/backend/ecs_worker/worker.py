"""SQS polling worker for the ECS Fargate backend.

Design notes
------------
* Long polling (``WaitTimeSeconds=20``) minimises empty receives and cost.
* Up to ``BATCH_SIZE`` messages are pulled per receive and processed
  concurrently across a thread pool so a 2 vCPU / 8 GB Graviton task can keep
  several inferences in flight.
* A message is deleted only after :meth:`InferenceService.process` succeeds or
  records a *permanent* failure. Transient failures leave the message on the
  queue; SQS redelivers it and, after ``maxReceiveCount``, routes it to the DLQ.
* ``SIGTERM`` (sent by ECS on scale-in / deployment) triggers a graceful drain:
  the receive loop stops and in-flight work is allowed to finish.
* A heartbeat file is touched each loop so the container health check can tell
  the worker is alive without exposing an inbound port.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from ..common.config import load_config
from ..common.errors import InvalidRequestError
from ..common.inference_service import InferenceService
from ..common.schemas import QueueMessage

logger = logging.getLogger("ecs_worker")

HEARTBEAT_FILE = os.environ.get("HEARTBEAT_FILE", "/tmp/worker-heartbeat")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "10"))
WAIT_TIME_SECONDS = int(os.environ.get("WAIT_TIME_SECONDS", "20"))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "8"))


class Worker:
    def __init__(
        self,
        service: Optional[InferenceService] = None,
        sqs_client: Any = None,
    ) -> None:
        self.config = load_config()
        self.service = service or InferenceService(config=self.config)
        self.queue_url = self.config.ecs_queue_url
        self._sqs = sqs_client
        self._running = True

    @property
    def sqs(self) -> Any:
        if self._sqs is None:
            import boto3

            self._sqs = boto3.client("sqs", region_name=self.config.aws_region)
        return self._sqs

    def request_stop(self, *_: Any) -> None:
        logger.info("Shutdown signal received; draining and stopping.")
        self._running = False

    def _touch_heartbeat(self) -> None:
        try:
            with open(HEARTBEAT_FILE, "w") as handle:
                handle.write(str(int(time.time())))
        except OSError:  # pragma: no cover - non-fatal
            logger.warning("Could not write heartbeat file %s", HEARTBEAT_FILE)

    def handle_message(self, record: Dict[str, Any]) -> Optional[str]:
        """Process one SQS message. Returns the ReceiptHandle to delete, or None.

        Returning the receipt handle means "safe to delete" (success or a
        permanent failure). Returning None leaves the message on the queue for
        redelivery.
        """
        try:
            body = json.loads(record["Body"])
            message = QueueMessage.from_json_dict(body)
        except (json.JSONDecodeError, KeyError, InvalidRequestError) as exc:
            # Unparseable message: nothing to retry. Delete so it does not loop.
            logger.error("Dropping malformed message: %s", exc)
            return record.get("ReceiptHandle")

        try:
            self.service.process(message)
            return record.get("ReceiptHandle")
        except Exception:
            # Transient failure: leave on queue for SQS redelivery / DLQ.
            logger.exception("Transient failure for %s; will retry", message.request_id)
            return None

    def poll_once(self) -> int:
        """Receive and process one batch. Returns the number of messages handled."""
        response = self.sqs.receive_message(
            QueueUrl=self.queue_url,
            MaxNumberOfMessages=min(BATCH_SIZE, 10),
            WaitTimeSeconds=WAIT_TIME_SECONDS,
            AttributeNames=["ApproximateReceiveCount"],
        )
        messages: List[Dict[str, Any]] = response.get("Messages", [])
        if not messages:
            return 0

        to_delete: List[Dict[str, str]] = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(self.handle_message, m): m for m in messages}
            for future in as_completed(futures):
                receipt = future.result()
                if receipt:
                    to_delete.append(
                        {"Id": futures[future]["MessageId"], "ReceiptHandle": receipt}
                    )

        if to_delete:
            # DeleteMessageBatch accepts up to 10 entries, matching BATCH_SIZE.
            self.sqs.delete_message_batch(QueueUrl=self.queue_url, Entries=to_delete)
        return len(messages)

    def run(self) -> None:
        if not self.queue_url:
            raise RuntimeError("ECS_QUEUE_URL is not configured")
        logger.info("Worker started. Polling %s", self.queue_url)
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)
        while self._running:
            self._touch_heartbeat()
            try:
                self.poll_once()
            except Exception:  # pragma: no cover - keep the loop alive
                logger.exception("Poll cycle failed; backing off 5s")
                time.sleep(5)
        logger.info("Worker stopped cleanly.")


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    Worker().run()


if __name__ == "__main__":
    main()
