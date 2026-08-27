import json
import unittest

from backend.api.router import route
from backend.common.errors import RequestNotFoundError

VALID_FEATURES = [0.12, -0.45, 0.78, -0.23, 0.56, -0.89, 1.23, -0.67]


class StubService:
    """Stands in for InferenceService in router tests."""

    def __init__(self):
        self.submitted = []

    def submit(self, backend, model, body, client_request_id=None):
        self.submitted.append((backend, model, body, client_request_id))

        class _Req:
            request_id = "abc123"
            status = "QUEUED"

        return _Req()

    def get_status(self, request_id):
        if request_id == "known":
            return {
                "request_id": "known",
                "backend": "lambda",
                "model": "weighted",
                "status": "COMPLETED",
                "created_at": "t0",
                "updated_at": "t1",
                "output": {"prediction": 2.5},
            }
        raise RequestNotFoundError(request_id)


def event(method, path, body=None, headers=None):
    return {
        "httpMethod": method,
        "path": path,
        "body": json.dumps(body) if body is not None else None,
        "headers": headers or {},
    }


class RouterTests(unittest.TestCase):
    def setUp(self):
        self.service = StubService()

    def test_catalog(self):
        resp = route(event("GET", "/api"), self.service)
        self.assertEqual(resp["statusCode"], 200)
        body = json.loads(resp["body"])
        self.assertEqual(body["service"], "multi-model-inference-api")
        # 2 backends x 4 models + 1 status op = 9 operations
        self.assertEqual(len(body["operations"]), 9)

    def test_health(self):
        resp = route(event("GET", "/api/health"), self.service)
        self.assertEqual(resp["statusCode"], 200)
        self.assertEqual(json.loads(resp["body"])["status"], "ok")

    def test_submit_returns_202(self):
        resp = route(
            event(
                "POST",
                "/api/backend/ecs-fargate/model/weighted",
                {"features": VALID_FEATURES},
                headers={"X-Client-Request-Id": "corr-1"},
            ),
            self.service,
        )
        self.assertEqual(resp["statusCode"], 202)
        body = json.loads(resp["body"])
        self.assertEqual(body["request_id"], "abc123")
        self.assertEqual(body["status_url"], "/api/requests/abc123")
        # header propagated
        self.assertEqual(self.service.submitted[0][3], "corr-1")

    def test_status_found(self):
        resp = route(event("GET", "/api/requests/known"), self.service)
        self.assertEqual(resp["statusCode"], 200)
        self.assertEqual(json.loads(resp["body"])["status"], "COMPLETED")

    def test_status_not_found(self):
        resp = route(event("GET", "/api/requests/nope"), self.service)
        self.assertEqual(resp["statusCode"], 404)

    def test_bad_json_body(self):
        evt = event("POST", "/api/backend/lambda/model/weighted")
        evt["body"] = "{not json"
        resp = route(evt, self.service)
        self.assertEqual(resp["statusCode"], 400)

    def test_unknown_route(self):
        resp = route(event("GET", "/api/unknown/thing"), self.service)
        self.assertEqual(resp["statusCode"], 404)

    def test_http_api_v2_shape(self):
        evt = {
            "rawPath": "/api",
            "requestContext": {"http": {"method": "GET"}},
        }
        resp = route(evt, self.service)
        self.assertEqual(resp["statusCode"], 200)


if __name__ == "__main__":
    unittest.main()
