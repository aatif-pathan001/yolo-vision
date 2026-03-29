"""
FastAPI REST backend for YOLO Vision.
Provides endpoints for image detection, video processing, and webcam streaming.
"""

import os
import time
import shutil
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.detector import YOLODetector
from app.utils.helpers import (
    build_output_path, ensure_output_dirs, frame_to_jpeg_bytes, summarize_detections
)

# ─── Global Detector Instance ────────────────────────────────────────────────

detector = YOLODetector(
    model_path=os.getenv("YOLO_MODEL_PATH", "yolov8n.pt"),
    confidence_threshold=float(os.getenv("CONF_THRESHOLD", "0.25")),
    iou_threshold=float(os.getenv("IOU_THRESHOLD", "0.45")),
)

OUTPUT_DIRS = ensure_output_dirs("outputs")


# ─── App Lifespan ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup."""
    detector.load_model()
    print(f"✅ Model loaded: {detector.model_path}")
    yield


# ─── App Instance ────────────────────────────────────────────────────────────

app = FastAPI(
    title="YOLO Vision API",
    description="Real-time object detection powered by Ultralytics YOLO",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve saved outputs statically
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


# ─── Health ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check():
    """Return API health and model status."""
    return {
        "status": "ok",
        "model_loaded": detector.is_loaded,
        "model_path": str(detector.model_path),
        "classes": detector.class_names if detector.is_loaded else {},
    }


# ─── Image Detection ─────────────────────────────────────────────────────────

@app.post("/detect/image", tags=["Detection"])
async def detect_image(file: UploadFile = File(...)):
    """
    Upload an image and receive detection results + annotated image URL.

    Returns:
        JSON with detections metadata and paths to annotated image and crops.
    """
    if not detector.is_loaded:
        raise HTTPException(503, "Model not loaded")

    # Save upload to a temp file
    suffix = Path(file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        t0 = time.perf_counter()
        annotated, detections = detector.detect_image(tmp_path)
        elapsed = round((time.perf_counter() - t0) * 1000, 1)

        # Save annotated image
        out_img_path = build_output_path(file.filename, str(OUTPUT_DIRS["images"]))
        import cv2
        cv2.imwrite(out_img_path, annotated)

        # Save crops
        crop_paths = detector.save_crops(detections, str(OUTPUT_DIRS["crops"]))
        crop_urls = [f"/outputs/crops/{Path(p).name}" for p in crop_paths]

        return {
            "inference_time_ms": elapsed,
            "summary": summarize_detections(detections),
            "detections": [d.to_dict() for d in detections],
            "annotated_image_url": f"/outputs/images/{Path(out_img_path).name}",
            "crop_urls": crop_urls,
        }
    finally:
        os.unlink(tmp_path)


# ─── Video Detection ─────────────────────────────────────────────────────────

@app.post("/detect/video", tags=["Detection"])
async def detect_video(file: UploadFile = File(...)):
    """
    Upload a video file; returns a download URL for the annotated video.

    Processing is synchronous — suitable for short clips.
    """
    if not detector.is_loaded:
        raise HTTPException(503, "Model not loaded")

    suffix = Path(file.filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        out_path = build_output_path(file.filename, str(OUTPUT_DIRS["videos"]), "_detected")
        t0 = time.perf_counter()
        stats = detector.detect_video(tmp_path, out_path)
        elapsed = round(time.perf_counter() - t0, 2)

        return {
            "processing_time_s": elapsed,
            "stats": stats,
            "download_url": f"/outputs/videos/{Path(out_path).name}",
        }
    finally:
        os.unlink(tmp_path)


# ─── Webcam Stream ───────────────────────────────────────────────────────────

def _mjpeg_generator(camera_index: int):
    """
    Yield MJPEG frames for webcam streaming.

    Args:
        camera_index: Camera device index.
    """
    for annotated_frame, _ in detector.stream_webcam(camera_index):
        jpg = frame_to_jpeg_bytes(annotated_frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        )


@app.get("/stream/webcam", tags=["Stream"])
def stream_webcam(camera: int = Query(0, description="Camera device index")):
    """
    MJPEG stream from the webcam with real-time YOLO annotations.

    Consume via an <img src="/stream/webcam"> tag.
    """
    if not detector.is_loaded:
        raise HTTPException(503, "Model not loaded")
    return StreamingResponse(
        _mjpeg_generator(camera),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ─── Model Config ─────────────────────────────────────────────────────────────

@app.patch("/config", tags=["System"])
def update_config(
    confidence: float = Query(None, ge=0.01, le=1.0),
    iou: float = Query(None, ge=0.01, le=1.0),
):
    """Dynamically update confidence and IOU thresholds."""
    if confidence is not None:
        detector.confidence_threshold = confidence
    if iou is not None:
        detector.iou_threshold = iou
    return {
        "confidence_threshold": detector.confidence_threshold,
        "iou_threshold": detector.iou_threshold,
    }
