import asyncio
from fastapi import FastAPI, HTTPException, Security, Request
from fastapi.exceptions import RequestValidationError
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, field_validator
from inference import SecureInference
import time
import base64
from contextlib import asynccontextmanager
import numpy as np
import tensorflow as tf
import os


# -----------------------------
# Load model ONCE (global)
# -----------------------------
print("Loading model...")
model = SecureInference(model_path="./model")
print("Model loaded")


class InferenceJob:
    def __init__(self, req, audio_bytes, future):
        self.req = req
        self.future = future

        self.audio_chunks, self.mask_chunks = model.prepare_chunks(audio_bytes)
        self.total = self.audio_chunks.shape[0]

        self.offset = 0
        self.scores = np.zeros(self.total, dtype=np.float32)

    def remaining(self):
        return self.total - self.offset


# -----------------------------
# FastAPI app
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting inference worker...")
    asyncio.create_task(inference_worker())
    yield

    print("Shutting down inference worker...")


app = FastAPI(title="TF Inference API with Queue", lifespan=lifespan)

# -----------------------------
# Middleware: Limit Request Size
# -----------------------------
MAX_BODY_SIZE = 10 * 1024 * 1024  # 10 MB


@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length:
        if int(content_length) > MAX_BODY_SIZE:
            return JSONResponse(
                status_code=413,
                content={
                    "status": "error",
                    "message": "Request entity too large. Limit is 10MB.",
                },
            )
    return await call_next(request)


# -----------------------------
# Queue configuration
# -----------------------------
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", 64))
BATCH_WAIT_MS = int(os.getenv("BATCH_WAIT_MS", 100))
MAX_CHUNKS_PER_BATCH = int(os.getenv("MAX_CHUNKS_PER_BATCH", 32))
MAX_CONCURRENT_BATCHES = int(os.getenv("MAX_CONCURRENT_BATCHES", 1))
INFERENCE_TIMEOUT = int(os.getenv("INFERENCE_TIMEOUT", 30))

inference_queue = asyncio.Queue(MAX_QUEUE_SIZE)
semaphore = asyncio.Semaphore(MAX_CONCURRENT_BATCHES)

# ================= BATCH BUILDER =================


def build_chunk_batch(jobs):
    audio_list, mask_list = [], []
    slice_map = []
    used = 0

    for job in jobs:
        if used >= MAX_CHUNKS_PER_BATCH:
            break

        take = min(job.remaining(), MAX_CHUNKS_PER_BATCH - used)
        s, e = job.offset, job.offset + take

        audio_list.append(job.audio_chunks[s:e])
        mask_list.append(job.mask_chunks[s:e])

        slice_map.append((job, s, e))

        job.offset += take
        used += take

    return (
        tf.concat(audio_list, axis=0),
        tf.concat(mask_list, axis=0),
        slice_map,
    )


# -----------------------------
# Request / Response schema
# -----------------------------
class PredictRequest(BaseModel):
    audioBase64: str

    def convert_to_bytes(self):
        return base64.b64decode(self.audioBase64)


class PredictResponse(BaseModel):
    status: str = "success"
    classification: str
    confidenceScore: float
    explanation: str


# -----------------------------
# Inference worker
# -----------------------------
async def inference_worker():
    pending = []

    while True:
        job = await inference_queue.get()
        pending.append(job)

        start = time.time()
        while time.time() - start < BATCH_WAIT_MS / 1000:
            try:
                pending.append(inference_queue.get_nowait())
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.001)

        try:
            async with semaphore:
                audio, mask, slices = build_chunk_batch(pending)
                outputs = model.model.serve([audio, mask])
                scores = outputs.numpy().ravel()

            cursor = 0
            completed = []

            for job, s, e in slices:
                length = e - s
                job.scores[s:e] = scores[cursor : cursor + length]
                cursor += length

                if job.remaining() == 0:
                    completed.append(job)

            for job in completed:
                result = model.post_process(job.scores)
                if not job.future.done():
                    job.future.set_result(
                        PredictResponse(
                            classification=result["label"],
                            confidenceScore=result["confidence"],
                            explanation=result["explanation"],
                        )
                    )
                pending.remove(job)

        except Exception as e:
            for job in pending:
                if not job.future.done():
                    job.future.set_exception(e)
            pending.clear()

        finally:
            inference_queue.task_done()


# -----------------------------
# Predict endpoint (polite waiting)
# -----------------------------
@app.post("/api/voice-detection", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if inference_queue.full():
        raise HTTPException(
            status_code=503, detail="Server busy. Queue full, retry later."
        )

    future = asyncio.get_running_loop().create_future()
    job = InferenceJob(req, req.convert_to_bytes(), future)

    await inference_queue.put(job)

    try:
        return await asyncio.wait_for(future, timeout=INFERENCE_TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Inference timed out")


# -----------------------------
# Error Handler
# -----------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": str(exc)},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Construct a single string from all errors
    errors = []
    for error in exc.errors():
        loc = " -> ".join(str(x) for x in error["loc"])
        msg = error["msg"]
        errors.append(f"{loc}: {msg}")

    return JSONResponse(
        status_code=422,
        content={"status": "error", "message": "; ".join(errors)},
    )


# -----------------------------
# Health check
# -----------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "queue_size": inference_queue.qsize(),
        "queue_capacity": MAX_QUEUE_SIZE,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
