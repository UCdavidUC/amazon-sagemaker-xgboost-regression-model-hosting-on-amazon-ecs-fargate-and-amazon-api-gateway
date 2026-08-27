import unittest

from backend.common.config import Config
from backend.common.errors import RequestNotFoundError
from backend.common.repository import DynamoRequestRepository, _from_dynamo, _to_dynamo
from backend.common.schemas import InferenceRequest

from tests.fakes import FakeTable

VALID_FEATURES = [0.12, -0.45, 0.78, -0.23, 0.56, -0.89, 1.23, -0.67]


class DecimalConversionTests(unittest.TestCase):
    def test_roundtrip_preserves_values(self):
        import decimal

        data = {"a": 1.5, "b": [1, 2.25], "c": {"d": 3.0}}
        dynamo = _to_dynamo(data)
        self.assertIsInstance(dynamo["a"], decimal.Decimal)
        back = _from_dynamo(dynamo)
        self.assertEqual(back["a"], 1.5)
        self.assertEqual(back["b"], [1, 2.25])
        self.assertEqual(back["c"]["d"], 3)  # 3.0 collapses to int 3


class RepositoryTests(unittest.TestCase):
    def setUp(self):
        self.table = FakeTable()
        self.repo = DynamoRequestRepository(table=self.table, config=Config())

    def _make_request(self):
        return InferenceRequest.create(
            backend="lambda",
            model="weighted",
            input_body={"features": VALID_FEATURES},
            environment="dev",
            ttl_days=30,
        )

    def test_save_and_get(self):
        req = self._make_request()
        self.repo.save(req)
        fetched = self.repo.get(req.request_id)
        self.assertEqual(fetched["request_id"], req.request_id)
        self.assertEqual(fetched["status"], "QUEUED")

    def test_get_missing_raises(self):
        with self.assertRaises(RequestNotFoundError):
            self.repo.get("missing")

    def test_mark_completed_builds_update(self):
        self.repo.mark_completed("r1", {"prediction": 2.5})
        call = self.table.update_calls[-1]
        self.assertEqual(call["Key"], {"request_id": "r1"})
        self.assertIn("output", call["UpdateExpression"])
        self.assertEqual(call["ExpressionAttributeValues"][":s"], "COMPLETED")

    def test_mark_failed_builds_update(self):
        self.repo.mark_failed("r2", "boom")
        call = self.table.update_calls[-1]
        self.assertEqual(call["ExpressionAttributeValues"][":s"], "FAILED")
        self.assertEqual(call["ExpressionAttributeValues"][":err"], "boom")


if __name__ == "__main__":
    unittest.main()
