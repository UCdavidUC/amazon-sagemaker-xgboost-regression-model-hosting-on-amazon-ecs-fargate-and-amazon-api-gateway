"""Top-level package for the inference backend.

Both compute backends and the API layer import from here. Deployment packages
ship the ``backend`` directory as the Python package root, so Lambda handlers
are referenced as ``backend.api.handler.lambda_handler`` and
``backend.lambda_worker.handler.lambda_handler``.
"""
