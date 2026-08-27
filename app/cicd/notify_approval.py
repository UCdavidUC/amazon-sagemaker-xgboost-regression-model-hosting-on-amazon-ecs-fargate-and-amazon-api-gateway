"""Compose and publish the production-approval notification.

Runs in the ``PublishMetrics`` CodeBuild action right before the manual approval
gate. It aggregates whatever verification artifacts are present in the build
workspace (SAST report, dependency audit, API smoke results, load-test verdict)
plus pipeline execution context, and publishes a single SNS message so the
human approver sees running time, error/vulnerability counts, and model/perf
metrics in one place.

Environment:
  APPROVAL_TOPIC_ARN  SNS topic to publish to.
  CODEBUILD_*         Provided automatically by CodeBuild.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


def _load_json(path: str) -> Optional[Any]:
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _count_bandit_findings(report: Optional[Dict[str, Any]]) -> Dict[str, int]:
    if not report or "results" not in report:
        return {"total": 0, "high": 0, "medium": 0}
    high = sum(1 for r in report["results"] if r.get("issue_severity") == "HIGH")
    medium = sum(1 for r in report["results"] if r.get("issue_severity") == "MEDIUM")
    return {"total": len(report["results"]), "high": high, "medium": medium}


def _count_audit_vulns(audit: Optional[Any]) -> int:
    # pip-audit JSON is a list of dependencies each with a "vulns" list.
    if not audit:
        return 0
    deps = audit.get("dependencies", audit) if isinstance(audit, dict) else audit
    if not isinstance(deps, list):
        return 0
    return sum(len(d.get("vulns", [])) for d in deps if isinstance(d, dict))


def build_message() -> Dict[str, Any]:
    bandit = _load_json("bandit-report.json")
    audit = _load_json("pip-audit.json")
    smoke = _load_json("api-smoke-results.json")
    load = _load_json("loadtest-verdict.json")

    findings = _count_bandit_findings(bandit)
    return {
        "pipeline": os.environ.get("CODEBUILD_INITIATOR", "unknown"),
        "build_id": os.environ.get("CODEBUILD_BUILD_ID", "unknown"),
        "region": os.environ.get("AWS_REGION", "unknown"),
        "security": {
            "sast_findings_total": findings["total"],
            "sast_findings_high": findings["high"],
            "sast_findings_medium": findings["medium"],
            "dependency_vulnerabilities": _count_audit_vulns(audit),
        },
        "api_tests": {
            "failures": (smoke or {}).get("failures", "n/a"),
        },
        "load_test": {
            "passed": (load or {}).get("passed", "n/a"),
            "p95_ms": (load or {}).get("p95_ms", "n/a"),
            "failure_ratio": (load or {}).get("failure_ratio", "n/a"),
            "requests": (load or {}).get("requests", "n/a"),
        },
    }


def format_text(message: Dict[str, Any]) -> str:
    sec = message["security"]
    load = message["load_test"]
    return (
        "Production deployment approval requested\n"
        f"Build: {message['build_id']} ({message['region']})\n\n"
        "Security\n"
        f"  SAST findings (total/high/medium): {sec['sast_findings_total']}/"
        f"{sec['sast_findings_high']}/{sec['sast_findings_medium']}\n"
        f"  Dependency vulnerabilities: {sec['dependency_vulnerabilities']}\n\n"
        "API tests\n"
        f"  Failures: {message['api_tests']['failures']}\n\n"
        "Load test (QA, prod-like config)\n"
        f"  Passed: {load['passed']}\n"
        f"  p95 latency (ms): {load['p95_ms']}\n"
        f"  Failure ratio: {load['failure_ratio']}\n"
        f"  Requests: {load['requests']}\n"
    )


def main() -> int:
    message = build_message()
    text = format_text(message)
    print(text)

    topic = os.environ.get("APPROVAL_TOPIC_ARN")
    if topic:
        import boto3

        boto3.client("sns").publish(
            TopicArn=topic,
            Subject="[Inference API] Production approval - review metrics",
            Message=text,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
