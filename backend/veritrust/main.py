"""FastAPI application. Routes only, all analysis logic lives in engine.py."""

from __future__ import annotations

import asyncio
import base64
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import settings
from .engine import Engine
from .preprocessing import ImageRejected
from .audio import AudioRejected
from .schemas import AnalyzeResponse, Base64Request, HealthResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("veritrust")

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"

engine = Engine(settings)

inference_slots = asyncio.Semaphore(settings.max_concurrency)
"""Inference is synchronous and GPU bound, so it runs in a worker thread to keep the event loop
free. The semaphore caps how many run at once, since concurrent forward passes contend for the
same VRAM and a burst of uploads would otherwise trip an out of memory error."""


async def run_analysis(data: bytes, want_heatmap: bool) -> dict:
    async with inference_slots:
        try:
            analysis = await run_in_threadpool(engine.analyze, data, want_heatmap)
        except (ImageRejected, AudioRejected) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return analysis.as_dict()


UPLOAD_CEILING = max(settings.max_upload_bytes, settings.max_audio_bytes)
"""The larger of the two per modality limits, applied at the HTTP boundary.

This route accepts either modality and cannot tell which it has before reading the body, so it can
only reject what neither could accept. decode_image and decode_audio then apply their own limit to
what they actually received, which is where a 30 MB image gets refused while a 30 MB recording does
not. Rejecting everything over the image limit here would make the audio limit unreachable."""


def _too_large(size: int | None) -> bool:
    return size is not None and size > UPLOAD_CEILING


@asynccontextmanager
async def lifespan(_app: FastAPI):
    log.info("Loading detectors on device=%s", engine.registry.device)
    engine.load()
    status = engine.status()
    log.info(
        "Ensemble ready: %s model(s), %s failure(s), face backend=%s",
        status["ensemble_size"],
        len(status["failures"]),
        status["face_backend"],
    )
    for failure in status["failures"]:
        log.warning("Detector %s unavailable: %s", failure["key"], failure["error"])
    if not status["calibrated"]:
        log.warning("Scores are uncalibrated. Run eval/calibrate.py on a labelled set.")
    yield


app = FastAPI(
    title="VeriTrust",
    version=__version__,
    description=(
        "Deepfake and AI generated media detection. Images fuse a synthetic ensemble, a face "
        "forgery pathway and provenance metadata into a three band verdict. Audio fuses a "
        "spoofed speech ensemble scored over overlapping windows into the same three bands. "
        "Both go to /api/v1/analyze, which decides the modality from the file header."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allow_origins),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(ImageRejected)
@app.exception_handler(AudioRejected)
async def media_rejected_handler(_request, exc: Exception):
    return JSONResponse(status_code=400, content={"error": str(exc)})


@app.get("/api/v1/health", response_model=HealthResponse)
async def health():
    status = engine.status()
    return {
        "status": "ok" if status["ensemble_size"] else "degraded",
        "version": __version__,
        "models_loaded": status["ensemble_size"],
        "device": status["device"],
    }


@app.get("/api/v1/models")
async def models():
    """Exact load state of every checkpoint, including resolved label mappings."""
    return engine.status()


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...), heatmap: bool = True):
    """Accepts an image or an audio file. The modality is decided from the magic bytes in engine.py,
    so there is one upload field and the client does not have to declare what it has."""
    if _too_large(getattr(file, "size", None)):
        raise HTTPException(status_code=413, detail="File too large.")
    data = await file.read()
    if _too_large(len(data)):
        raise HTTPException(status_code=413, detail="File too large.")
    return await run_analysis(data, heatmap)


@app.post("/api/v1/analyze-base64", response_model=AnalyzeResponse)
async def analyze_base64(payload: Base64Request):
    raw = payload.image
    if "," in raw and raw.strip().startswith("data:"):
        raw = raw.split(",", 1)[1]
    try:
        data = base64.b64decode(raw, validate=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Malformed base64 payload.") from exc
    return await run_analysis(data, payload.heatmap)


if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
