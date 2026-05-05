# LCM Inference Server

An async image generation server powered by Latent Consistency Models (LCM),
built with FastAPI, Celery, and Redis.

## Architecture

```
                    ┌─────────────────────────────────────────────────┐
                    │              Docker Compose Stack               │
                    │                                                 │
  HTTP Request      │  ┌──────────┐    ┌───────┐    ┌────────────┐    │
  POST /generate ──►│  │  FastAPI │───►│ Redis │───►│   Celery   │    │
                    │  │   :8000  │    │ :6379 │    │   Worker   │    │
  GET /status/id ──►│  │          │◄───│       │◄───│            │    │
                    │  │ (gateway)│    │(broker│    │  (PyTorch  │    │
  GET /result/id ──►│  │          │    │  +    │    │    LCM     │    │
                    │  │          │    │result)│    │  pipeline) │    │
                    │  └──────────┘    └───────┘    └─────┬──────┘    │
                    │       │                             │           │
                    │       └──────── /app/outputs ───────┘           │
                    │              (shared volume)                    │
                    └─────────────────────────────────────────────────┘
```

**Request flow:**

1. Client sends `POST /generate` with a prompt
2. FastAPI validates the request and pushes a task onto the Redis queue
3. FastAPI returns `202 Accepted` with a `task_id` immediately
4. Celery worker picks the task from the queue and runs inference
5. Generated image is saved to the shared volume
6. Client polls `GET /status/{task_id}` until status is `SUCCESS`
7. Client downloads the image via `GET /result/{task_id}`


## CPU Optimizations

This stack is tuned for a **24 GB RAM, 8th-gen Intel i5, no GPU** laptop:

| Optimization | What it does |
|---|---|
| `float32` instead of `float16` | Intel 8th-gen CPUs lack native FP16 compute — float16 on CPU emulates via float32 and is *slower*. We use native float32. |
| Tiny AutoEncoder (TAESD) | Replaces the full VAE decoder (~320M params) with a tiny one (~2M params). ~10x faster decoding, ~1 GB less RAM, minimal quality loss. |
| LCM at 4 steps | Latent Consistency Models produce good images in just 4 denoising steps vs 20-30 for normal SD. |
| Safety checker disabled | Saves ~1 GB RAM and speeds up the pipeline (the safety classifier is a full CLIP model). |
| `solo` Celery pool | No prefork overhead — the worker runs inference in a single process with no multiprocessing complexity. |
| `worker_concurrency=1` | One task at a time since inference is CPU-bound and would thrash with parallelism. |
| Memory limit 8 GB | Docker memory cap prevents the worker from consuming all 24 GB. |
| CPU-only PyTorch | Uses the `torch+cpu` wheel (~700 MB vs ~2.5 GB with CUDA). |


## Setup & Run

### Prerequisites

- Docker and Docker Compose installed
- ~12 GB free disk (for Docker images + model weights)
- First build takes ~15 minutes (downloading PyTorch + model weights)

### Start the stack

```bash
cd lcm-inference-server

# Build and start all services
docker compose up --build

# Or run in the background
docker compose up --build -d
```

Watch the worker logs to see when the model is loaded:

```bash
docker compose logs -f worker
```

You'll see:
```
worker  | [INFO] LOADING MODEL: Lykon/dreamshaper-8-lcm
worker  | [INFO] Loading Tiny AutoEncoder (TAESD) for fast decoding
worker  | [INFO] Model loaded in 28.3s
worker  | [INFO] Torch threads: 4
```

### Stop the stack

```bash
docker compose down          # Stop containers
docker compose down -v       # Stop + remove volumes (deletes cached models)
```


## API Usage

### Generate an image

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a cyberpunk city at night, neon lights, rain, 8k",
    "negative_prompt": "blurry, low quality",
    "num_inference_steps": 4,
    "width": 512,
    "height": 512,
    "seed": 42
  }'
```

Response (`202 Accepted`):
```json
{
  "task_id": "a1b2c3d4-...",
  "status": "queued",
  "message": "Task submitted. Poll /status/{task_id} for progress."
}
```

### Check task status

```bash
curl http://localhost:8000/status/a1b2c3d4-...
```

Response (while processing):
```json
{
  "task_id": "a1b2c3d4-...",
  "status": "STARTED",
  "result": null,
  "error": null
}
```

Response (when done):
```json
{
  "task_id": "a1b2c3d4-...",
  "status": "SUCCESS",
  "result": {
    "filename": "a1b2c3d4-....png",
    "inference_time_seconds": 32.5,
    "prompt": "a cyberpunk city at night, neon lights, rain, 8k",
    "width": 512,
    "height": 512,
    "num_inference_steps": 4,
    "guidance_scale": 1.0,
    "seed": 42
  }
}
```

### Download the image

```bash
curl -o generated.png http://localhost:8000/result/a1b2c3d4-...
```

Or open in a browser: `http://localhost:8000/result/a1b2c3d4-...`

### Health check

```bash
curl http://localhost:8000/health
```

### Queue stats (useful for KEDA later)

```bash
curl http://localhost:8000/queue/stats
```


## API Documentation

FastAPI auto-generates interactive docs:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc


## Project Structure

```
lcm-inference-server/
├── docker-compose.yml          # Orchestrates Redis + API + Worker
├── Dockerfile.api              # Slim image for the FastAPI gateway
├── Dockerfile.worker           # Heavy image with PyTorch + model
├── api/
│   ├── main.py                 # FastAPI app with all endpoints
│   └── requirements.txt        # API-only dependencies (no PyTorch)
├── model-server/
│   ├── server.py               # FastAPI app that holds the model in memory
│   └── requirements.txt        # PyTorch + diffusers 
├── worker/
│   ├── __init__.py
│   ├── tasks.py                # Celery tasks + model referencign from model server
│   └── requirements.txt        # Celery + Redis
├── shared/
│   ├── __init__.py
│   ├── config.py               # Environment-driven configuration
│   └── celery_app.py           # Celery app instance (shared)
└── README.md
```
