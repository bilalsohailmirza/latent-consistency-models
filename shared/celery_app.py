"""
Celery application instance.

Imported by:
  - The API server (to call .delay() / .apply_async() on tasks)
  - The worker process (to discover and execute tasks)
"""

from celery import Celery
from shared.config import CELERY_BROKER_URL, CELERY_RESULT_BACKEND

celery_app = Celery(
    "lcm_worker",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Worker settings optimized for heavy inference tasks:
    # Each worker process handles ONE task at a time (inference is CPU-bound).
    worker_concurrency=1,
    worker_prefetch_multiplier=1,

    # Results expire after 1 hour
    result_expires=3600,

    # Task routing
    task_routes={
        "worker.tasks.generate_image": {"queue": "inference"},
    },
)

# Auto-discover tasks in the worker package
celery_app.autodiscover_tasks(["worker"])
