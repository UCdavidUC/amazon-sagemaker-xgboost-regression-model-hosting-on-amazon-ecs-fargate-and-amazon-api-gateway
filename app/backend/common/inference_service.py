"""Orchestration used by both the API layer and the workers.

Two responsibilities:

* :meth:`submit` - called by the API. Validates the route + body, writes the
  QUEUED record to DynamoDB, and enqueues the work on SQS. Returns the created
  :class:`InferenceRequest` so the API can respond ``202 Accepted``.
* :meth:`process` - called by a worker (Lambda consumer or ECS poller). Marks
  the record PROCESSING, runs the model, and records COMPLETED or FAILED.

Splitting submit/process keeps the two compute backends identical: they share
this exact orchestration and only differ in how messages are delivered.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .config import Config, load_config
from .errors import InvalidRequestError
from .models import get_model
from .queue import QueuePublisher
from .repository import DynamoRequestRepository
from .schemas import (
    InferenceRequest,
    QueueMessage,
    validate_body,
    validate_route,
)

logger = logging.getLogger(__name__)


class InferenceService:
    def __init__(
        self,
        config: Optional[Config] = None,
        repository: Optional[DynamoRequestRepository] = None,
        publisher: Optional[QueuePublisher] = None,
    ) -> None:
        self.config = config or load_config()
        self.repository = repository or DynamoRequestRepository(config=self.config)
        self.publisher = publisher or QueuePublisher(config=self.config)

    # -- API side ---------------------------------------------------------
    def submit(
        self,
        backend: str,
        model: str,
        body: Any,
        client_request_id: Optional[str] = None,
    ) -> InferenceRequest:
        """Validate, persist as QUEUED, and enqueue an inference request."""
        validate_route(backend, model)
        parsed_body = validate_body(body)

        # Validate the model-specific payload up front so the caller gets an
        # immediate 400 instead of an async failure they would have to poll for.
        model_impl = get_model(model, self.config)
        normalized_input = model_impl.validate_input(parsed_body)

        request = InferenceRequest.create(
            backend=backend,
            model=model,
            input_body=normalized_input,
            environment=self.config.environment,
            ttl_days=self.config.record_ttl_days,
            client_request_id=client_request_id,
        )
        self.repository.save(request)
        logger.info(
            "Queued request %s (backend=%s model=%s)",
            request.request_id, backend, model,
        )

        message = QueueMessage(
            request_id=request.request_id,
            backend=backend,
            model=model,
            input=normalized_input,
        )
        self.publisher.publish(message)
        return request

    def get_status(self, request_id: str) -> Dict[str, Any]:
        """Return the current DynamoDB view of a request."""
        return self.repository.get(request_id)

    # -- worker side ------------------------------------------------------
    def process(self, message: QueueMessage) -> Dict[str, Any]:
        """Run inference for a dequeued message and update DynamoDB.

        Raises on transient failures so the worker can let the message return to
        the queue. Permanent failures (bad input) are recorded as FAILED and
        swallowed so the message is not retried forever.
        """
        request_id = message.request_id
        self.repository.mark_processing(request_id)
        model_impl = get_model(message.model, self.config)
        try:
            normalized = model_impl.validate_input(message.input)
            model_impl.ensure_loaded()
            output = model_impl.predict(normalized)
        except InvalidRequestError as exc:
            # Permanent: the payload will never become valid. Record and stop.
            logger.warning("Permanent failure for %s: %s", request_id, exc)
            self.repository.mark_failed(request_id, str(exc))
            return {"request_id": request_id, "status": "FAILED", "permanent": True}
        except Exception as exc:
            # Transient (e.g. artifact not yet available). Record and re-raise so
            # SQS redelivers; the DLQ catches it after maxReceiveCount.
            logger.exception("Transient failure for %s", request_id)
            self.repository.mark_failed(request_id, str(exc))
            raise

        self.repository.mark_completed(request_id, output)
        logger.info("Completed request %s", request_id)
        return {"request_id": request_id, "status": "COMPLETED", "output": output}
