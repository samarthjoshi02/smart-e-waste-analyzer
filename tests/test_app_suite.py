"""Automated Test Suite for Phase 3 & 4 Verification.

Tests:
1. Valid PCB image detection (bounding boxes, class labels, counts, confidences).
2. Invalid file handling (corrupt bytes, oversized file >15MB, bad extensions).
3. Zero-detection handling (blank white image, no crash, 0 counts).
4. Code compilation of app.py and detector.py.
"""

import io
from pathlib import Path
import sys
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.detector import PCBDetector, EXPECTED_CLASSES, MAX_FILE_SIZE_BYTES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolov8s_pcb_best.pt"
DATA_YAML = PROJECT_ROOT / "dataset" / "data.yaml"
VALID_IMAGE_PATH = PROJECT_ROOT / "dataset" / "images" / "val" / "IMG_20260529_163319.jpg"


def test_1_detector_initialization():
    print("\n--- Test 1: Detector Initialization ---")
    detector = PCBDetector(model_path=MODEL_PATH, data_yaml_path=DATA_YAML)
    assert detector.model is not None, "YOLO model failed to initialize"
    assert detector.class_names == EXPECTED_CLASSES, f"Class names mismatch: {detector.class_names}"
    print(f"[PASS] Detector initialized successfully with {len(detector.class_names)} classes.")
    return detector


def test_2_valid_pcb_detection(detector: PCBDetector):
    print("\n--- Test 2: Valid PCB Image Inference ---")
    assert VALID_IMAGE_PATH.exists(), f"Missing valid test image: {VALID_IMAGE_PATH}"
    img_bytes = VALID_IMAGE_PATH.read_bytes()

    # Validate and sanitize
    sanitized = PCBDetector.validate_and_sanitize_image(img_bytes, VALID_IMAGE_PATH.name)
    assert isinstance(sanitized, Image.Image), "Sanitized output is not a PIL Image"
    print(f"[PASS] Image validation & EXIF sanitization passed. Size: {sanitized.size}")

    # Predict at conf=0.30
    pred = detector.predict(sanitized, conf_threshold=0.30, iou_threshold=0.45, imgsz=960)

    assert pred["total_detections"] > 0, "Expected at least 1 detection on validation image"
    assert len(pred["detections_table"]) == pred["total_detections"], "Detections table size mismatch"
    assert sum(pred["counts_by_class"].values()) == pred["total_detections"], "Class counts sum mismatch"
    assert isinstance(pred["annotated_image"], Image.Image), "Annotated image is not a PIL Image"

    print(f"[PASS] Total detections: {pred['total_detections']}")
    print(f"[PASS] Latency: {pred['inference_time_ms']} ms")
    print(f"[PASS] Class counts: {dict(pred['counts_by_class'])}")

    # Verify first detection item schema
    top_det = pred["detections_table"][0]
    required_keys = ["Component", "Confidence (%)", "Xmin", "Ymin", "Xmax", "Ymax", "Width", "Height"]
    for k in required_keys:
        assert k in top_det, f"Missing key {k} in detection item"
    print(f"[PASS] Top detection: {top_det['Component']} ({top_det['Confidence (%)']}%) BBox: [{top_det['Xmin']}, {top_det['Ymin']}, {top_det['Xmax']}, {top_det['Ymax']}]")


def test_3_invalid_file_handling():
    print("\n--- Test 3: Invalid File Handling ---")

    # 3a. Disallowed extension
    try:
        PCBDetector.validate_and_sanitize_image(b"fake content", "malicious.exe")
        assert False, "Failed to reject .exe extension"
    except ValueError as e:
        assert "Unsupported file format" in str(e)
        print(f"[PASS] Successfully rejected unsupported file extension: {e}")

    # 3b. Oversized file (>15 MB)
    oversized_bytes = b"0" * (MAX_FILE_SIZE_BYTES + 1024)
    try:
        PCBDetector.validate_and_sanitize_image(oversized_bytes, "huge_board.jpg")
        assert False, "Failed to reject oversized file"
    except ValueError as e:
        assert "exceeds the 15 MB maximum limit" in str(e)
        print(f"[PASS] Successfully rejected file exceeding 15 MB: {e}")

    # 3c. Corrupt binary data with valid extension
    try:
        PCBDetector.validate_and_sanitize_image(b"not an actual image content", "corrupt.jpg")
        assert False, "Failed to reject corrupt image"
    except ValueError as e:
        assert "Could not decode image" in str(e)
        print(f"[PASS] Successfully caught corrupt image binary: {e}")


def test_4_zero_detections_handling(detector: PCBDetector):
    print("\n--- Test 4: Zero-Detection Image Handling ---")

    # Create a blank white 800x800 image (no components)
    blank_img = Image.new("RGB", (800, 800), color=(255, 255, 255))
    buf = io.BytesIO()
    blank_img.save(buf, format="JPEG")
    blank_bytes = buf.getvalue()

    sanitized = PCBDetector.validate_and_sanitize_image(blank_bytes, "blank_white.jpg")
    pred = detector.predict(sanitized, conf_threshold=0.50, iou_threshold=0.45)

    assert pred["total_detections"] == 0, f"Expected 0 detections on blank image, got {pred['total_detections']}"
    assert len(pred["detections_table"]) == 0, "Detections table should be empty"
    assert sum(pred["counts_by_class"].values()) == 0, "Class counts should all be 0"
    assert pred["annotated_image"].size == sanitized.size, "Annotated image size mismatch"
    print(f"[PASS] Handled blank zero-detection image gracefully without error. Total detections: {pred['total_detections']}")


def test_5_syntax_compilation():
    print("\n--- Test 5: Syntax Compilation of app.py ---")
    import py_compile
    py_compile.compile(str(PROJECT_ROOT / "app.py"), doraise=True)
    py_compile.compile(str(PROJECT_ROOT / "modules" / "detector.py"), doraise=True)
    print("[PASS] app.py and modules/detector.py compiled successfully with zero syntax errors.")


if __name__ == "__main__":
    detector = test_1_detector_initialization()
    test_2_valid_pcb_detection(detector)
    test_3_invalid_file_handling()
    test_4_zero_detections_handling(detector)
    test_5_syntax_compilation()
    print("\n==========================================")
    print("ALL 5 AUTOMATED UNIT TESTS PASSED SUCCESSFULLY!")
    print("==========================================")
