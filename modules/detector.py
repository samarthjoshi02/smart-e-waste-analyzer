"""Core Detection Engine for Intelligent E-Waste Circuit Component Classifier.

Separates model loading, pre-processing, YOLO inference, and post-processing
from the UI layer.
"""

from collections import Counter
import io
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageOps
import yaml

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# 15 MB Maximum allowed upload size
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024

# Supported image file extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Standard 10 PCB Component Classes
EXPECTED_CLASSES = [
    "Resistor", "Capacitor", "Diode", "LED", "IC",
    "Connector", "Switch/Button", "Header", "Crystal", "Debug/TP"
]

# Color palette (RGB) for distinct component classes
CLASS_COLORS = {
    "Resistor": (255, 87, 34),       # Deep Orange
    "Capacitor": (33, 150, 243),     # Blue
    "Diode": (156, 39, 176),         # Purple
    "LED": (255, 235, 59),           # Yellow
    "IC": (233, 30, 99),             # Pink / Magenta
    "Connector": (0, 188, 212),      # Cyan
    "Switch/Button": (76, 175, 80),  # Green
    "Header": (255, 152, 0),         # Orange
    "Crystal": (63, 81, 181),        # Indigo
    "Debug/TP": (158, 158, 158)      # Grey
}


class PCBDetector:
    """Encapsulates YOLOv8s PCB component detection and analysis logic."""

    def __init__(self, model_path: Union[str, Path], data_yaml_path: Optional[Union[str, Path]] = None):
        self.model_path = Path(model_path)
        self.data_yaml_path = Path(data_yaml_path) if data_yaml_path else None
        self.model = None
        self.class_names: List[str] = EXPECTED_CLASSES
        self._load_model()

    def _load_model(self) -> None:
        """Loads and verifies the YOLOv8 model checkpoint."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model weights file not found: {self.model_path}")

        file_size = self.model_path.stat().st_size
        if file_size < 1000:
            raise ValueError(
                f"Model file at {self.model_path} appears to be an invalid pointer file ({file_size} bytes)."
            )

        if YOLO is None:
            raise ImportError("Ultralytics library is not installed in the active environment.")

        # Load YOLO model checkpoint
        self.model = YOLO(str(self.model_path))

        # Verify class mapping against data.yaml if provided
        if self.data_yaml_path and self.data_yaml_path.exists():
            try:
                with open(self.data_yaml_path, "r", encoding="utf-8") as f:
                    yaml_data = yaml.safe_load(f)
                    yaml_classes = yaml_data.get("names", [])
                    if yaml_classes:
                        self.class_names = yaml_classes
            except Exception as e:
                print(f"[Warning] Could not parse data.yaml ({e}), falling back to checkpoint classes.")

        # Synchronize with model's internal classes if available
        if hasattr(self.model, "names") and self.model.names:
            if isinstance(self.model.names, dict):
                self.class_names = [self.model.names[i] for i in sorted(self.model.names.keys())]
            elif isinstance(self.model.names, list):
                self.class_names = self.model.names

    @staticmethod
    def validate_and_sanitize_image(image_bytes: bytes, filename: str) -> Image.Image:
        """Validates file constraints and sanitizes EXIF orientation safely in-memory.

        Args:
            image_bytes: Raw bytes from the uploaded file.
            filename: Original name of the uploaded file.

        Returns:
            Sanitized PIL Image in RGB format.

        Raises:
            ValueError: If file size exceeds 15 MB, extension is invalid, or image is corrupt.
        """
        # 1. Enforce 15 MB upload limit
        if len(image_bytes) > MAX_FILE_SIZE_BYTES:
            size_mb = len(image_bytes) / (1024 * 1024)
            raise ValueError(f"Uploaded file size ({size_mb:.2f} MB) exceeds the 15 MB maximum limit.")

        # 2. Prevent directory traversal and validate extension
        safe_name = Path(filename).name.lower()
        suffix = Path(safe_name).suffix
        if suffix not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format '{suffix}'. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )

        # 3. Read image binary header and sanitize EXIF metadata in-memory
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                # Transpose image according to EXIF orientation (strips orientation drift)
                img = ImageOps.exif_transpose(img)
                # Ensure RGB mode (handles RGBA, Palette, or Grayscale)
                sanitized = img.convert("RGB")
                return sanitized
        except Exception as e:
            raise ValueError(f"Corrupt or invalid image file. Could not decode image: {e}")

    def predict(
        self,
        image: Image.Image,
        conf_threshold: float = 0.30,
        iou_threshold: float = 0.45,
        imgsz: int = 960,
    ) -> Dict[str, Any]:
        """Performs component detection on a sanitized PCB image.

        Args:
            image: Sanitized PIL Image.
            conf_threshold: Confidence threshold for predictions (0.0 to 1.0).
            iou_threshold: NMS IoU threshold.
            imgsz: Inference image size.

        Returns:
            Dictionary containing:
                - annotated_image: PIL Image with rendered bounding boxes and labels
                - total_detections: Total component count
                - counts_by_class: Dict mapping each of the 10 classes to count
                - detections_table: List of detection details (class, confidence, bbox)
                - inference_time_ms: Latency in milliseconds
        """
        t0 = time.time()

        # Run inference on CPU
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            device="cpu",
            verbose=False,
        )

        inference_time_ms = round((time.time() - t0) * 1000, 1)

        result = results[0]
        boxes = result.boxes

        # 1. Initialize complete zero-count dictionary for all 10 classes
        counts_by_class: Dict[str, int] = {cls_name: 0 for cls_name in self.class_names}

        # 2. Extract structured detection records
        detections_table: List[Dict[str, Any]] = []

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"Class_{cls_id}"
                conf = float(box.conf[0].item())
                xyxy = [int(round(coord)) for coord in box.xyxy[0].tolist()]

                counts_by_class[cls_name] = counts_by_class.get(cls_name, 0) + 1

                detections_table.append({
                    "Component": cls_name,
                    "Confidence (%)": round(conf * 100, 1),
                    "Confidence_Raw": conf,
                    "Xmin": xyxy[0],
                    "Ymin": xyxy[1],
                    "Xmax": xyxy[2],
                    "Ymax": xyxy[3],
                    "Width": xyxy[2] - xyxy[0],
                    "Height": xyxy[3] - xyxy[1],
                })

            # Sort detection table by confidence descending
            detections_table.sort(key=lambda item: item["Confidence_Raw"], reverse=True)

            # Render annotated image using Ultralytics plot() (returns BGR numpy array)
            annotated_bgr = result.plot(line_width=2, font_size=1)
            # Convert BGR to RGB PIL Image
            annotated_rgb = Image.fromarray(annotated_bgr[..., ::-1])
        else:
            # No detections above threshold: return original image with no annotations
            annotated_rgb = image.copy()

        return {
            "annotated_image": annotated_rgb,
            "total_detections": len(detections_table),
            "counts_by_class": counts_by_class,
            "detections_table": detections_table,
            "inference_time_ms": inference_time_ms,
        }


def get_available_samples(val_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    """Discovers available validation sample PCB images from the dataset.

    Returns a list of dictionaries with 'name', 'path', and 'size_kb'.
    If the directory does not exist or is empty, returns an empty list.
    """
    path = Path(val_dir)
    if not path.exists() or not path.is_dir():
        return []

    samples = []
    for f in sorted(path.glob("*")):
        if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS:
            samples.append({
                "name": f.name,
                "path": f,
                "size_kb": round(f.stat().st_size / 1024, 1),
            })
    return samples

