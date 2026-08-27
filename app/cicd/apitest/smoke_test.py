"""Post-deploy smoke test for the QA environment.

Exercises the full asynchronous contract for both backends and a couple of
models: submit -> poll -> assert terminal state. Writes a JSON summary that the
pipeline captures as an artifact and exits non-zero on failure so the stage
gates the release.

Environment:
  API_BASE_URL  base URL including the /api base path and stage, e.g.
                https://abc123.execute-api.us-east-1.amazonaws.com/qa/api
  POLL_TIMEOUT  seconds to wait for a request to reach a terminal state (default 60)
"""
from __future__ import annotations

import json
import os
import sys
import time

import requests  # provided by the buildspec install phase

BASE = os.environ["API_BASE_URL"].rstrip("/")
POLL_TIMEOUT = int(os.environ.get("POLL_TIMEOUT", "60"))

FEATURES = [0.12, -0.45, 0.78, -0.23, 0.56, -0.89, 1.23, -0.67]

CASES = [
    ("lambda", "weighted", {"features": FEATURES}),
    ("lambda", "arima", {"steps": 5}),
    ("ecs-fargate", "weighted", {"features": FEATURES}),
    ("ecs-fargate", "sarima", {"steps": 3}),
]


def submit(backend: str, model: str, body: dict) -> str:
    resp = requests.post(f"{BASE}/backend/{backend}/model/{model}", json=body, timeout=15)
    resp.raise_for_status()
    assert resp.status_code == 202, f"expected 202, got {resp.status_code}"
    return resp.json()["request_id"]


def poll(request_id: str) -> dict:
    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        resp = requests.get(f"{BASE}/requests/{request_id}", timeout=15)
        resp.raise_for_status()
        record = resp.json()
        if record["status"] in ("COMPLETED", "FAILED"):
            return record
        time.sleep(2)
    raise TimeoutError(f"{request_id} did not finish within {POLL_TIMEOUT}s")


def main() -> int:
    # Catalog must be reachable.
    catalog = requests.get(BASE, timeout=15)
    catalog.raise_for_status()

    results = []
    failures = 0
    for backend, model, body in CASES:
        entry = {"backend": backend, "model": model}
        try:
            rid = submit(backend, model, body)
            record = poll(rid)
            entry["status"] = record["status"]
            entry["ok"] = record["status"] == "COMPLETED"
            if not entry["ok"]:
                entry["error"] = record.get("error")
                failures += 1
        except Exception as exc:  # noqa: BLE001 - report and continue
            entry["ok"] = False
            entry["error"] = str(exc)
            failures += 1
        results.append(entry)
        print(json.dumps(entry))

    with open("api-smoke-results.json", "w") as handle:
        json.dump({"results": results, "failures": failures}, handle, indent=2)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
