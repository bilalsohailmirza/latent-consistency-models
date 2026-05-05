# ~/lcm-inference-server/model-server/server.py
"""
Dedicated model server — loads the LCM pipeline once and serves
inference requests over HTTP. Runs as a single always-on pod.
"""

import os
import io
import time
import logging
import base64

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from diffusers import (
    AutoPipelineForText2Image,
    LCMScheduler,
    AutoencoderTiny,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ID = os.getenv("MODEL_ID", "Lykon/dreamshaper-8-lcm")
TORCH_DTYPE = os.getenv("TORCH_DTYPE", "float32")
USE_TINY_VAE = os.getenv("USE_TINY_VAE", "true").lower() == "true"
TORCH_NUM_THREADS = int(os.getenv("TORCH_NUM_THREADS", 4))

pipeline = None


class InferRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    num_inference_steps: int = Field(default=4, ge=1, le=8)
    width: int = Field(default=512, ge=256, le=768)
    height: int = Field(default=512, ge=256, le=768)
    guidance_scale: float = Field(default=1.0, ge=0.0, le=2.0)
    seed: int | None = None


class InferResponse(BaseModel):
    image_base64: str
    inference_time_seconds: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup, keep it in memory forever."""
    global pipeline

    logger.info("=" * 60)
    logger.info("LOADING MODEL: %s", MODEL_ID)
    logger.info("=" * 60)

    start = time.time()
    torch.set_num_threads(TORCH_NUM_THREADS)
    dtype = getattr(torch, TORCH_DTYPE, torch.float32)

    pipeline = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    pipeline.scheduler = LCMScheduler.from_config(
        pipeline.scheduler.config
    )

    if USE_TINY_VAE:
        logger.info("Loading Tiny AutoEncoder (TAESD)")
        pipeline.vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesd",
            torch_dtype=dtype,
        )

    pipeline = pipeline.to("cpu")
    torch.set_grad_enabled(False)

    elapsed = time.time() - start
    logger.info("Model loaded in %.1fs", elapsed)
    logger.info("Torch threads: %d | dtype: %s | Tiny VAE: %s",
                torch.get_num_threads(), dtype, USE_TINY_VAE)

    yield

    del pipeline


app = FastAPI(title="LCM Model Server", version="0.1.0", lifespan=lifespan)


@app.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest):
    """
    Run inference and return the image as base64-encoded PNG.
    Workers call this endpoint — they never touch the model directly.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    generator = None
    if req.seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(req.seed)

    start = time.time()

    with torch.inference_mode():
        result = pipeline(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt or None,
            num_inference_steps=req.num_inference_steps,
            width=req.width,
            height=req.height,
            guidance_scale=req.guidance_scale,
            generator=generator,
        )

    elapsed = time.time() - start
    image = result.images[0]

    # Encode image as base64 PNG to send over HTTP
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    logger.info("Inference: %.1fs | %dx%d | %d steps | '%s'",
                elapsed, req.width, req.height,
                req.num_inference_steps, req.prompt[:50])

    return InferResponse(
        image_base64=image_b64,
        inference_time_seconds=round(elapsed, 2),
    )


@app.get("/health")
async def health():
    return {
        "status": "ready" if pipeline is not None else "loading",
        "model": MODEL_ID,
    }
