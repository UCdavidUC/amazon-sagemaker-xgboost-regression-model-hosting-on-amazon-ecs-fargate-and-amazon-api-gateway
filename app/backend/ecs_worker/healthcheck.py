"""Container health check for the ECS worker.

Exits 0 if the worker heartbeat file was updated recently, non-zero otherwise.
Referenced by the task definition's ``HealthCheck.Command``. This avoids
exposing an inbound port on a queue-driven worker.
"""
from __future__ import annotations

import os
import sys
import time

HEARTBEAT_FILE = os.environ.get("HEARTBEAT_FILE", "/tmp/worker-heartbeat")
# Allow one long-poll cycle plus margin before considering the worker unhealthy.
MAX_AGE_SECONDS = int(os.environ.get("HEARTBEAT_MAX_AGE", "60"))


def main() -> int:
    try:
        age = time.time() - os.path.getmtime(HEARTBEAT_FILE)
    except OSError:
        print("heartbeat file missing", file=sys.stderr)
        return 1
    if age > MAX_AGE_SECONDS:
        print(f"heartbeat stale ({age:.0f}s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
