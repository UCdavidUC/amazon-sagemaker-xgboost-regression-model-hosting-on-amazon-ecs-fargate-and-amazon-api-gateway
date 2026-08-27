"""Evaluate Locust CSV output against Well-Architected latency/error thresholds.

Reads ``loadtest_stats.csv`` produced by ``locust --csv loadtest`` and fails the
build if the aggregated p95 latency or failure ratio exceed the configured
budgets. Writes ``loadtest-verdict.json`` for the approval notification.
"""
from __future__ import annotations

import csv
import json
import os
import sys

MAX_P95_MS = float(os.environ.get("MAX_P95_MS", "1500"))
MAX_FAIL_RATIO = float(os.environ.get("MAX_FAIL_RATIO", "0.01"))


def read_aggregated(path: str = "loadtest_stats.csv") -> dict:
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        # Locust writes a row named "Aggregated" summarising all endpoints.
        if row.get("Name") == "Aggregated":
            return row
    raise SystemExit("No Aggregated row found in loadtest_stats.csv")


def main() -> int:
    row = read_aggregated()
    requests = float(row.get("Request Count", 0) or 0)
    failures = float(row.get("Failure Count", 0) or 0)
    # Column name is "95%" in recent Locust versions.
    p95 = float(row.get("95%") or row.get("95%ile") or 0)
    fail_ratio = (failures / requests) if requests else 1.0

    passed = p95 <= MAX_P95_MS and fail_ratio <= MAX_FAIL_RATIO
    verdict = {
        "requests": requests,
        "failures": failures,
        "failure_ratio": round(fail_ratio, 5),
        "p95_ms": p95,
        "max_p95_ms": MAX_P95_MS,
        "max_failure_ratio": MAX_FAIL_RATIO,
        "passed": passed,
    }
    with open("loadtest-verdict.json", "w") as handle:
        json.dump(verdict, handle, indent=2)
    print(json.dumps(verdict, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
