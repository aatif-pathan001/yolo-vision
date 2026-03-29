"""
Utility functions for image encoding, file validation, and output management.
"""

import io
import cv2
import numpy as np
from pathlib import Path
from PIL import Image


# ─── Supported file types ────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


def is_image(path: str) -> bool:
    """Return True if the path points to a supported image format."""
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def is_video(path: str) -> bool:
    """Return True if the path points to a supported video format."""
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def validate_model_path(path: str) -> Path:
    """
    Validate that a model file exists and has a .pt extension.

    Args:
        path: File path string.

    Returns:
        Resolved Path object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the extension is not .pt.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")
    if p.suffix.lower() != ".pt":
        raise ValueError(f"Expected a .pt model file, got: {p.suffix}")
    return p


# ─── Image Encoding ──────────────────────────────────────────────────────────

def bgr_to_pil(frame: np.ndarray) -> Image.Image:
    """Convert a BGR OpenCV frame to a PIL Image (RGB)."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def encode_frame_to_bytes(frame: np.ndarray, fmt: str = "JPEG", quality: int = 90) -> bytes:
    """
    Encode an OpenCV BGR frame to raw image bytes.

    Args:
        frame: BGR numpy array.
        fmt: PIL format string ('JPEG', 'PNG', etc.).
        quality: JPEG quality (1–95).

    Returns:
        Encoded image bytes.
    """
    pil_img = bgr_to_pil(frame)
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt, quality=quality)
    buf.seek(0)
    return buf.read()


def frame_to_jpeg_bytes(frame: np.ndarray) -> bytes:
    """Encode a BGR frame to JPEG bytes (convenience wrapper)."""
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buffer.tobytes()


# ─── Output Path Management ──────────────────────────────────────────────────

def ensure_output_dirs(base: str = "outputs") -> dict[str, Path]:
    """
    Create standard output directory structure.

    Args:
        base: Root output directory.

    Returns:
        Dict mapping name → Path for 'images', 'videos', 'crops'.
    """
    root = Path(base)
    dirs = {
        "images": root / "images",
        "videos": root / "videos",
        "crops": root / "crops",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def build_output_path(input_path: str, output_dir: str, suffix: str = "_detected") -> str:
    """
    Build an output file path derived from the input path.

    Args:
        input_path: Original file path.
        output_dir: Target directory.
        suffix: Appended to the stem before the extension.

    Returns:
        Full output path string.
    """
    src = Path(input_path)
    dest = Path(output_dir) / f"{src.stem}{suffix}{src.suffix}"
    return str(dest)


# ─── Detection Stats ─────────────────────────────────────────────────────────

def summarize_detections(detections: list) -> dict:
    """
    Aggregate detection results into a summary dict.

    Args:
        detections: List of Detection objects.

    Returns:
        Dict with counts per class and average confidence.
    """
    class_counts: dict[str, int] = {}
    total_conf = 0.0

    for det in detections:
        class_counts[det.class_label] = class_counts.get(det.class_label, 0) + 1
        total_conf += det.confidence

    return {
        "total_objects": len(detections),
        "class_counts": class_counts,
        "avg_confidence": round(total_conf / len(detections), 4) if detections else 0.0,
    }
