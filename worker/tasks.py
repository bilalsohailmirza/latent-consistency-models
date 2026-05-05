# ~/lcm-inference-server/worker/tasks.py
"""
Thin Celery worker — sends inference requests to the model server.
No PyTorch, no model loading. Instant startup.
"""

import os
import time
import base64
import logging

import requests as http_requests
from shared.celery_app import celery_app
from shared.config import (
    DEFAULT_STEPS,
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    DEFAULT_GUIDANCE_SCALE,
    OUTPUT_DIR,
)

logger = logging.getLogger(__name__)

# The model server's internal Kubernetes service URL
MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://model-server:8001")


@celery_app.task(
    name="worker.tasks.generate_image",
    bind=True,
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=2,
    time_limit=300,
    soft_time_limit=240,
)
def generate_image(
    self,
    prompt: str,
    negative_prompt: str = "",
    num_inference_steps: int = DEFAULT_STEPS,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    seed: int | None = None,
):
    """
    Pull a task from Redis, send it to the model server,
    save the returned image, report back.
    """
    task_id = self.request.id
    logger.info("Task %s: Sending to model server — '%s'", task_id, prompt[:60])

    start = time.time()

    # Call the model server
    try:
        response = http_requests.post(
            f"{MODEL_SERVER_URL}/infer",
            json={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
                "width": width,
                "height": height,
                "guidance_scale": guidance_scale,
                "seed": seed,
            },
            timeout=240,
        )
        response.raise_for_status()
    except http_requests.ConnectionError:
        logger.error("Task %s: Model server unreachable, retrying...", task_id)
        raise self.retry(countdown=10)
    except http_requests.HTTPError as e:
        logger.error("Task %s: Model server error: %s", task_id, e)
        raise

    result = response.json()
    elapsed = time.time() - start

    # Decode base64 image and save to shared volume
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"{task_id}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)

    image_bytes = base64.b64decode(result["image_base64"])
    with open(filepath, "wb") as f:
        f.write(image_bytes)

    logger.info("Task %s: Done in %.1fs → %s", task_id, elapsed, filename)

    return {
        "filename": filename,
        "inference_time_seconds": result["inference_time_seconds"],
        "total_time_seconds": round(elapsed, 2),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
        "width": width,
        "height": height,
        "guidance_scale": guidance_scale,
        "seed": seed,
    }