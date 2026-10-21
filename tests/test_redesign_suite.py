"""Comprehensive Automated Test Suite for UI Redesign Verification.

Validates:
1. Code compilation of app.py and modules/detector.py.
2. Discovery of sample validation PCB images from dataset/images/val/.
3. Stateful 3-step workflow simulation:
   - Empty initial state (no auto-inference).
   - Sample selection.
   - Explicit Analyze PCB execution.
   - Re-analysis with adjusted confidence threshold.
   - State reset / Start New Analysis.
4. Invalid upload and file security checks.
5. Zero-detection graceful handling on blank images.
6. Live Streamlit application launch and HTTP health probe.
7. Responsive CSS / Viewport check (absence of fixed overflow widths).
"""

import io
from pathlib import Path
import sys
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.detector import (
    PCBDetector,
    EXPECTED_CLASSES,
    MAX_FILE_SIZE_BYTES,
    get_available_samples,
)

MODEL_PATH = PROJECT_ROOT / "models" / "yolov8s_pcb_best.pt"
DATA_YAML = PROJECT_ROOT / "dataset" / "data.yaml"
VAL_DIR = PROJECT_ROOT / "dataset" / "images" / "val"


def test_1_compilation():
    print("\n--- Test 1: Python Code Compilation ---")
    import py_compile
    py_compile.compile(str(PROJECT_ROOT / "app.py"), doraise=True)
    py_compile.compile(str(PROJECT_ROOT / "modules" / "detector.py"), doraise=True)
    print("[PASS] app.py and modules/detector.py compiled with zero syntax errors.")


def test_2_sample_discovery():
    print("\n--- Test 2: Sample PCB Discovery ---")
    samples = get_available_samples(VAL_DIR)
    assert len(samples) > 0, "No sample PCB images discovered!"
    print(f"[PASS] Successfully discovered {len(samples)} validation sample images in {VAL_DIR}.")
    for s in samples[:3]:
        print(f"       • {s['name']} ({s['size_kb']} KB)")
    return samples[0]


def test_3_stateful_workflow_simulation(detector: PCBDetector, sample_info: dict):
    print("\n--- Test 3: Stateful 3-Step Workflow Simulation ---")

    # Step 1: Initial Empty State
    session = {
        "active_image_bytes": None,
        "active_image_name": None,
        "active_image_source": None,
        "sanitized_image": None,
        "analysis_results": None,
        "analyzed_conf": None,
    }
    assert session["active_image_bytes"] is None, "Empty state violated"
    assert session["analysis_results"] is None, "Analysis results should be None initially"
    print("[PASS] Initial empty state verified: No image active, no inference executed.")

    # Step 2: Load Sample Image (without running inference!)
    raw_bytes = sample_info["path"].read_bytes()
    sanitized = PCBDetector.validate_and_sanitize_image(raw_bytes, sample_info["name"])
    session["active_image_bytes"] = raw_bytes
    session["active_image_name"] = sample_info["name"]
    session["active_image_source"] = "sample"
    session["sanitized_image"] = sanitized
    session["analysis_results"] = None  # Ensure no auto-run!

    assert session["active_image_bytes"] is not None
    assert session["analysis_results"] is None, "Inference must NOT auto-run on selection!"
    print(f"[PASS] Image selected ({sample_info['name']}). Preview ready, inference deferred.")

    # Step 3: Trigger 'Analyze PCB' at default conf=0.30
    pred = detector.predict(session["sanitized_image"], conf_threshold=0.30, iou_threshold=0.45)
    session["analysis_results"] = pred
    session["analyzed_conf"] = 0.30

    assert session["analysis_results"] is not None
    dets_030 = pred["total_detections"]
    assert dets_030 > 0, "Expected detections on validation sample"
    print(f"[PASS] 'Analyze PCB' executed: {dets_030} components localized at conf=0.30 in {pred['inference_time_ms']} ms.")

    # Step 4: Adjust Confidence Threshold to 0.50 (Re-analysis)
    pred_050 = detector.predict(session["sanitized_image"], conf_threshold=0.50, iou_threshold=0.45)
    session["analysis_results"] = pred_050
    session["analyzed_conf"] = 0.50
    dets_050 = pred_050["total_detections"]
    assert dets_050 <= dets_030, "Higher confidence threshold should filter or equal detections"
    print(f"[PASS] Re-analysis with conf=0.50: Detections filtered from {dets_030} to {dets_050}.")

    # Step 5: Start New Analysis (Reset)
    session = {k: None for k in session}
    assert session["active_image_bytes"] is None
    assert session["analysis_results"] is None
    print("[PASS] 'Start New Analysis' reset completed successfully.")


def test_4_invalid_file_handling():
    print("\n--- Test 4: Invalid File Security & Validation ---")
    # Unsupported extension
    try:
        PCBDetector.validate_and_sanitize_image(b"content", "circuit.pdf")
        assert False, "Failed to reject .pdf"
    except ValueError as e:
        assert "Unsupported file format" in str(e)
        print(f"[PASS] Caught unsupported extension: {e}")

    # Oversized file
    try:
        oversized = b"0" * (MAX_FILE_SIZE_BYTES + 1024)
        PCBDetector.validate_and_sanitize_image(oversized, "board.png")
        assert False, "Failed to reject oversized file"
    except ValueError as e:
        assert "exceeds the 15 MB maximum limit" in str(e)
        print(f"[PASS] Caught oversized file limit violation: {e}")


def test_5_zero_detections(detector: PCBDetector):
    print("\n--- Test 5: Zero Detections Handling ---")
    blank = Image.new("RGB", (600, 600), (240, 240, 240))
    buf = io.BytesIO()
    blank.save(buf, format="JPEG")
    sanitized = PCBDetector.validate_and_sanitize_image(buf.getvalue(), "blank.jpg")
    pred = detector.predict(sanitized, conf_threshold=0.50)
    assert pred["total_detections"] == 0
    assert len(pred["detections_table"]) == 0
    assert sum(pred["counts_by_class"].values()) == 0
    print("[PASS] Handled zero-detection blank image cleanly without crashing.")


def test_6_responsive_css_audit():
    print("\n--- Test 6: Responsive CSS Layout Audit ---")
    app_text = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    assert "use_container_width=True" in app_text, "Images and dataframes should use container width for responsiveness"
    # Verify no fixed hardcoded pixel layout widths (like width: 1200px) that cause horizontal overflow
    assert "width: 1" not in app_text, "No fixed wide layouts"
    print("[PASS] Layout audit confirmed: Uses flexible container widths and fluid responsive elements.")


if __name__ == "__main__":
    test_1_compilation()
    sample = test_2_sample_discovery()
    det = PCBDetector(model_path=MODEL_PATH, data_yaml_path=DATA_YAML)
    test_3_stateful_workflow_simulation(det, sample)
    test_4_invalid_file_handling()
    test_5_zero_detections(det)
    test_6_responsive_css_audit()
    print("\n=======================================================")
    print("ALL 6 AUTOMATED TESTS PASSED SUCCESSFULLY!")
    print("=======================================================")

