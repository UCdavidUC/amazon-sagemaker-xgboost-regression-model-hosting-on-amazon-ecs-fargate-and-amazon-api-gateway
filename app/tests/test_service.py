import unittest

from backend.common.config import Config
from backend.common.errors import InvalidRequestError, UnknownRouteError
from backend.common.inference_service import InferenceService
from backend.common.models import reset_cache
from backend.common.schemas import QueueMessage

from tests.fakes import FakePublisher, FakeRepository

VALID_FEATURES = [0.12, -0.45, 0.78, -0.23, 0.56, -0.89, 1.23, -0.67]


def make_service():
    repo = FakeRepository()
    pub = FakePublisher()
    cfg = Config()  # environment defaults; no AWS needed
    return InferenceService(config=cfg, repository=repo, publisher=pub), repo, pub


class SubmitTests(unittest.TestCase):
    def setUp(self):
        reset_cache()

    def test_submit_queues_and_publishes(self):
        service, repo, pub = make_service()
        req = service.submit("lambda", "weighted", {"features": VALID_FEATURES})
        self.assertEqual(req.status, "QUEUED")
        self.assertIn(req.request_id, repo.items)
        self.assertEqual(len(pub.published), 1)
        self.assertEqual(pub.published[0].model, "weighted")

    def test_submit_unknown_backend(self):
        service, _, _ = make_service()
        with self.assertRaises(UnknownRouteError):
            service.submit("nope", "weighted", {"features": VALID_FEATURES})

    def test_submit_invalid_body(self):
        service, _, _ = make_service()
        with self.assertRaises(InvalidRequestError):
            service.submit("lambda", "weighted", {"features": [1, 2]})

    def test_submit_body_must_be_object(self):
        service, _, _ = make_service()
        with self.assertRaises(InvalidRequestError):
            service.submit("lambda", "weighted", "not-a-dict")


class ProcessTests(unittest.TestCase):
    def setUp(self):
        reset_cache()

    def test_process_completes(self):
        service, repo, _ = make_service()
        msg = QueueMessage("r1", "ecs-fargate", "weighted", {"features": VALID_FEATURES})
        result = service.process(msg)
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(repo.items["r1"]["status"], "COMPLETED")
        self.assertIn("mark_processing", repo.calls)
        self.assertIn("mark_completed", repo.calls)

    def test_process_permanent_failure_swallowed(self):
        service, repo, _ = make_service()
        # Missing features -> InvalidRequestError -> permanent
        msg = QueueMessage("r2", "lambda", "weighted", {"features": [1, 2]})
        result = service.process(msg)
        self.assertEqual(result["status"], "FAILED")
        self.assertTrue(result["permanent"])
        self.assertEqual(repo.items["r2"]["status"], "FAILED")

    def test_process_transient_failure_reraises(self):
        service, repo, _ = make_service()
        # xgboost with no artifact -> ModelLoadError -> transient -> re-raise
        msg = QueueMessage("r3", "ecs-fargate", "xgboost", {"features": VALID_FEATURES})
        with self.assertRaises(Exception):
            service.process(msg)
        self.assertEqual(repo.items["r3"]["status"], "FAILED")


if __name__ == "__main__":
    unittest.main()
