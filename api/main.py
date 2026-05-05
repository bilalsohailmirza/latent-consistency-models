"""
FastAPI API Gateway for the LCM Inference Server.

Endpoints:
  POST /generate          — Submit a new image generation task
  GET  /status/{task_id}  — Check the status of a generation task
  GET  /result/{task_id}  — Retrieve the generated image file
  GET  /health            — Health check for the API + Redis
  GET  /queue/stats       — Current queue depth and worker info
"""

import os
import time
from contextlib import asynccontextmanager

import redis
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from shared.config import (
    REDIS_HOST,
    REDIS_PORT,
    DEFAULT_STEPS,
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    DEFAULT_GUIDANCE_SCALE,
    OUTPUT_DIR,
)
from shared.celery_app import celery_app


# ── Pydantic Models ──────────────────────────────────────────────

class GenerateRequest(BaseModel):
    """Request body for the /generate endpoint."""
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Text prompt describing the image to generate",
        examples=["a beautiful cyborg with golden hair, portrait, 8k"],
    )
    negative_prompt: str = Field(
        default="",
        max_length=500,
        description="What to avoid in the generated image",
        examples=["blurry, low quality, deformed"],
    )
    num_inference_steps: int = Field(
        default=DEFAULT_STEPS,
        ge=1,
        le=8,
        description="Number of denoising steps (LCM works best with 2-8)",
    )
    width: int = Field(
        default=DEFAULT_WIDTH,
        ge=256,
        le=768,
        description="Image width in pixels (must be divisible by 8)",
    )
    height: int = Field(
        default=DEFAULT_HEIGHT,
        ge=256,
        le=768,
        description="Image height in pixels (must be divisible by 8)",
    )
    guidance_scale: float = Field(
        default=DEFAULT_GUIDANCE_SCALE,
        ge=0.0,
        le=2.0,
        description="Classifier-free guidance scale (LCM works best at 1.0)",
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility",
    )


class GenerateResponse(BaseModel):
    """Response body for the /generate endpoint."""
    task_id: str
    status: str
    message: str


class StatusResponse(BaseModel):
    """Response body for the /status endpoint."""
    task_id: str
    status: str
    result: dict | None = None
    error: str | None = None


# ── App Lifecycle ────────────────────────────────────────────────

redis_client: redis.Redis | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle for the FastAPI app."""
    global redis_client
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=True,
    )
    # Verify Redis is reachable
    redis_client.ping()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    yield
    redis_client.close()


# ── FastAPI App ──────────────────────────────────────────────────

app = FastAPI(
    title="LCM Inference Server",
    description=(
        "Async image generation API powered by Latent Consistency Models. "
        "Submit a prompt, get a task ID, poll for status, download the image."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ── Endpoints ────────────────────────────────────────────────────

@app.post("/generate", response_model=GenerateResponse, status_code=202)
async def generate(req: GenerateRequest):
    """
    Submit an image generation task.

    The task is queued in Redis and processed asynchronously by a
    Celery worker. Returns a task_id that can be used to poll status.
    """
    # Validate dimensions are divisible by 8 (required by SD)
    if req.width % 8 != 0 or req.height % 8 != 0:
        raise HTTPException(
            status_code=400,
            detail="width and height must be divisible by 8",
        )

    task = celery_app.send_task(
        "worker.tasks.generate_image",
        kwargs={
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "num_inference_steps": req.num_inference_steps,
            "width": req.width,
            "height": req.height,
            "guidance_scale": req.guidance_scale,
            "seed": req.seed,
        },
        queue="inference",
    )

    return GenerateResponse(
        task_id=task.id,
        status="queued",
        message="Task submitted. Poll /status/{task_id} for progress.",
    )


@app.get("/status/{task_id}", response_model=StatusResponse)
async def get_status(task_id: str):
    """
    Check the status of a generation task.

    Possible statuses: PENDING, STARTED, SUCCESS, FAILURE, REVOKED.
    When SUCCESS, the result dict contains the image filename and metadata.
    """
    result = celery_app.AsyncResult(task_id)

    response = StatusResponse(task_id=task_id, status=result.status)

    if result.status == "SUCCESS":
        response.result = result.result
    elif result.status == "FAILURE":
        response.error = str(result.result)

    return response


@app.get("/result/{task_id}")
async def get_result(task_id: str):
    """
    Download the generated image.

    Only available after the task reaches SUCCESS status.
    Returns the PNG file directly.
    """
    result = celery_app.AsyncResult(task_id)

    if result.status != "SUCCESS":
        raise HTTPException(
            status_code=404,
            detail=f"Task not ready. Current status: {result.status}",
        )

    filename = result.result.get("filename")
    filepath = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Image file not found")

    return FileResponse(
        filepath,
        media_type="image/png",
        filename=filename,
    )


@app.get("/health")
async def health():
    """Health check — verifies the API and Redis are operational."""
    try:
        redis_client.ping()
        redis_ok = True
    except Exception:
        redis_ok = False

    status = "healthy" if redis_ok else "degraded"
    return {
        "status": status,
        "redis": "connected" if redis_ok else "disconnected",
        "timestamp": time.time(),
    }


@app.get("/queue/stats")
async def queue_stats():
    """
    Returns current queue depth and active worker info.
    Useful for monitoring and will feed into KEDA scaling later.
    """
    # Get queue length from Redis directly
    queue_length = redis_client.llen("inference")

    # Get registered workers from Celery
    inspector = celery_app.control.inspect()
    active = inspector.active() or {}
    registered = inspector.registered() or {}

    return {
        "queue_name": "inference",
        "queue_length": queue_length,
        "active_tasks": {
            worker: len(tasks) for worker, tasks in active.items()
        },
        "registered_workers": list(registered.keys()),
    }
