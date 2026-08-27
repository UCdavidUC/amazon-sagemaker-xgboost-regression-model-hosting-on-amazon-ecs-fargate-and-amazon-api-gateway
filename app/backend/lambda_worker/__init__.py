"""AWS Lambda compute backend.

An SQS event-source mapping invokes this function with batches of queued
inference requests. Named ``lambda_worker`` because ``lambda`` is a reserved
Python keyword and cannot be an importable package name.
"""
