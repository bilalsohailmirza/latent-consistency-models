"""
Shared configuration for the LCM inference server.
All settings are read from environment variables with sensible defaults
for a CPU-only laptop (24 GB RAM, 8th gen i5, no GPU).
"""

import os

# ── Redis / Celery ────────────────────────────────────────────────
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"
CELERY_BROKER_URL = REDIS_URL
CELERY_RESULT_BACKEND = REDIS_URL

# ── Model Settings ────────────────────────────────────────────────
# Using the same Dreamshaper 8 LCM we served via LocalAI, but now
# loaded directly with HuggingFace diffusers for full control.
MODEL_ID = os.getenv("MODEL_ID", "Lykon/dreamshaper-8-lcm")

# IMPORTANT: On CPU, float32 is FASTER than float16.
# Intel 8th-gen i5 lacks native FP16 compute units, so float16
# on CPU actually emulates via float32 and adds overhead.
# We use float32 for CPU inference — this is the key optimization
# adjustment from the original FP16 spec.
TORCH_DTYPE = os.getenv("TORCH_DTYPE", "float32")

# Use Tiny AutoEncoder (TAESD) for ~10x faster VAE decoding.
# Saves ~1 GB RAM and significantly speeds up the decode step.
USE_TINY_VAE = os.getenv("USE_TINY_VAE", "true").lower() == "true"

# ── Generation Defaults ───────────────────────────────────────────
DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", 4))
DEFAULT_WIDTH = int(os.getenv("DEFAULT_WIDTH", 512))
DEFAULT_HEIGHT = int(os.getenv("DEFAULT_HEIGHT", 512))
DEFAULT_GUIDANCE_SCALE = float(os.getenv("DEFAULT_GUIDANCE_SCALE", 1.0))

# Number of CPU threads for PyTorch (match physical cores)
TORCH_NUM_THREADS = int(os.getenv("TORCH_NUM_THREADS", 4))

# ── Storage ───────────────────────────────────────────────────────
# Shared volume where generated images are saved by workers
# and served by the API.
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/app/outputs")
