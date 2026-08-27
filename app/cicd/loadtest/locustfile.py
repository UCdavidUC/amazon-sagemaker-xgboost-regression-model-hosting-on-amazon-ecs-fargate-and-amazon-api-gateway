"""Locust load profile for the inference API.

Submits inference requests across both backends and models at a representative
mix. Because the API is asynchronous, the submit call (202) is the latency-
critical hot path; a fraction of virtual users also poll status to exercise the
read path. The host is the API base URL including the /api base path.
"""
from __future__ import annotations

import random

from locust import HttpUser, between, task

FEATURES = [0.12, -0.45, 0.78, -0.23, 0.56, -0.89, 1.23, -0.67]

SUBMIT_MIX = [
    ("lambda", "weighted", {"features": FEATURES}),
    ("lambda", "arima", {"steps": 5}),
    ("ecs-fargate", "weighted", {"features": FEATURES}),
    ("ecs-fargate", "sarima", {"steps": 7}),
]


class InferenceUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(9)
    def submit(self):
        backend, model, body = random.choice(SUBMIT_MIX)
        with self.client.post(
            f"/backend/{backend}/model/{model}",
            json=body,
            name=f"POST /backend/{backend}/model/{model}",
            catch_response=True,
        ) as resp:
            if resp.status_code == 202:
                resp.success()
            else:
                resp.failure(f"unexpected status {resp.status_code}")

    @task(1)
    def catalog(self):
        self.client.get("", name="GET /api")
