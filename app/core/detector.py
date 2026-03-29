"""
YOLO Object Detector Core Module
Handles model loading, inference, and result processing.
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from ultralytics import YOLO


@dataclass
class Detection:
    """Represents a single object detection result."""
    class_id: int
    class_label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    cropped_image: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def bbox_width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def bbox_height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    def to_dict(self) -> dict:
        """Serialize detection metadata (excluding image array)."""
        return {
            "class_id": self.class_id,
            "class_label": self.class_label,
            "confidence": round(self.confidence, 4),
            "bbox": {
                "x1": self.bbox[0],
                "y1": self.bbox[1],
                "x2": self.bbox[2],
                "y2": self.bbox[3],
                "width": self.bbox_width,
                "height": self.bbox_height,
            },
        }


class YOLODetector:
    """
    Core YOLO object detection engine.

    Wraps Ultralytics YOLO model with convenience methods for
    image detection, video processing, and webcam streaming.
    """

    # Color palette for bounding boxes (BGR)
    COLORS = [
        (0, 212, 255), (255, 61, 0), (0, 255, 127), (255, 0, 212),
        (255, 200, 0), (0, 100, 255), (200, 255, 0), (255, 0, 100),
        (0, 255, 200), (100, 0, 255), (255, 150, 50), (50, 255, 150),
    ]

    def __init__(self, model_path: str, confidence_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Initialize the YOLO detector.

        Args:
            model_path: Path to the YOLO model weights (.pt file).
            confidence_threshold: Minimum confidence to accept a detection.
            iou_threshold: IOU threshold for Non-Maximum Suppression.
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model: Optional[YOLO] = None
        self.class_names: dict[int, str] = {}

    def load_model(self) -> bool:
        """
        Load the YOLO model from disk.

        Returns:
            True if successful, False otherwise.
        """
        try:
            self.model = YOLO(str(self.model_path))
            self.class_names = self.model.names  # {0: 'person', 1: 'bicycle', ...}
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from '{self.model_path}': {e}") from e

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self.model is not None

    def _get_color(self, class_id: int) -> tuple[int, int, int]:
        """Return a consistent color for a given class ID."""
        return self.COLORS[class_id % len(self.COLORS)]

    def _parse_results(self, results, frame: np.ndarray) -> list[Detection]:
        """
        Parse raw YOLO results into Detection objects.

        Args:
            results: Raw Ultralytics result object.
            frame: Original image frame (for cropping).

        Returns:
            List of Detection objects.
        """
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_label = self.class_names.get(class_id, f"class_{class_id}")

                # Crop the detected region
                h, w = frame.shape[:2]
                cx1, cy1 = max(0, x1), max(0, y1)
                cx2, cy2 = min(w, x2), min(h, y2)
                crop = frame[cy1:cy2, cx1:cx2].copy() if cy2 > cy1 and cx2 > cx1 else None

                detections.append(Detection(
                    class_id=class_id,
                    class_label=class_label,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    cropped_image=crop,
                ))
        return detections

    def draw_detections(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """
        Draw bounding boxes and labels on a frame.

        Args:
            frame: Image array to annotate.
            detections: List of Detection objects.

        Returns:
            Annotated frame.
        """
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = self._get_color(det.class_id)
            label = f"{det.class_label}  {det.confidence:.0%}"

            # Draw bounding box with rounded corners effect (thick + thin)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw corner accents
            corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
            thickness = 3
            for (px, py), (dx, dy) in [
                ((x1, y1), (1, 1)),
                ((x2, y1), (-1, 1)),
                ((x1, y2), (1, -1)),
                ((x2, y2), (-1, -1)),
            ]:
                cv2.line(annotated, (px, py), (px + dx * corner_len, py), color, thickness)
                cv2.line(annotated, (px, py), (px, py + dy * corner_len), color, thickness)

            # Draw label background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            font_thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            label_y = max(y1 - 6, th + 6)
            cv2.rectangle(annotated, (x1, label_y - th - 6), (x1 + tw + 8, label_y + baseline), color, -1)
            cv2.putText(annotated, label, (x1 + 4, label_y - 2), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

        return annotated

    # ─── Image Detection ────────────────────────────────────────────────────────

    def detect_image(self, image_path: str) -> tuple[np.ndarray, list[Detection]]:
        """
        Run object detection on a single image.

        Args:
            image_path: Path to the input image file.

        Returns:
            Tuple of (annotated_frame, list of Detection objects).
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        detections = self._parse_results(results, frame)
        annotated = self.draw_detections(frame, detections)
        return annotated, detections

    def save_crops(self, detections: list[Detection], output_dir: str) -> list[str]:
        """
        Save cropped detection regions to disk.

        Args:
            detections: List of Detection objects with crop images.
            output_dir: Directory to save crops.

        Returns:
            List of saved file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []

        for i, det in enumerate(detections):
            if det.cropped_image is None or det.cropped_image.size == 0:
                continue
            filename = f"crop_{i:03d}_{det.class_label}_{det.confidence:.2f}.jpg"
            path = output_dir / filename
            cv2.imwrite(str(path), det.cropped_image)
            saved_paths.append(str(path))

        return saved_paths

    # ─── Video Detection ─────────────────────────────────────────────────────────

    def detect_video(
        self,
        video_path: str,
        output_path: str,
        progress_callback=None,
    ) -> dict:
        """
        Run object detection on a video file and save the annotated output.

        Args:
            video_path: Path to the input video.
            output_path: Path to save the annotated output video.
            progress_callback: Optional callable(frame_num, total_frames).

        Returns:
            Dict with processing statistics.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        total_detections = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = self.model.predict(
                    frame,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    verbose=False,
                )
                detections = self._parse_results(results, frame)
                annotated = self.draw_detections(frame, detections)

                # Overlay frame counter
                cv2.putText(
                    annotated,
                    f"Frame {frame_count + 1}/{total_frames}  |  Objects: {len(detections)}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 212, 255), 2, cv2.LINE_AA,
                )

                writer.write(annotated)
                frame_count += 1
                total_detections += len(detections)

                if progress_callback:
                    progress_callback(frame_count, total_frames)
        finally:
            cap.release()
            writer.release()

        return {
            "frames_processed": frame_count,
            "total_detections": total_detections,
            "avg_detections_per_frame": round(total_detections / max(frame_count, 1), 2),
            "output_path": output_path,
        }

    # ─── Webcam Streaming ────────────────────────────────────────────────────────

    def stream_webcam(self, camera_index: int = 0):
        """
        Generator that yields annotated frames from the webcam.

        Args:
            camera_index: Camera device index (default 0).

        Yields:
            Tuple of (annotated_frame, list[Detection]).
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {camera_index}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = self.model.predict(
                    frame,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    verbose=False,
                )
                detections = self._parse_results(results, frame)
                annotated = self.draw_detections(frame, detections)
                yield annotated, detections
        finally:
            cap.release()
