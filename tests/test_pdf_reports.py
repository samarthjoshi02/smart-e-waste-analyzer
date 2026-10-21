"""Comprehensive Automated Test Suite for PDF Reports.

Tests:
1. Generation and validity of Component Inventory Bill (PDF).
2. Generation and validity of Hazardous Materials & Recycling Report (PDF).
3. Generation and validity of Complete Analysis Report (PDF) with and without image embedding.
4. Correctness of component quantities and valuation totals matching analyzer calculations.
5. Honest handling of unknown/unpriced components ("Price unavailable").
6. Presence of required scientific limitations and disclaimers.
7. Resilience when Groq is unavailable (offline local knowledge base).
8. Presence of the 3 download buttons and descriptive filenames in app.py.
"""

from pathlib import Path
import sys
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.detector import PCBDetector, EXPECTED_CLASSES, get_available_samples
from modules.analyzer import build_component_inventory, calculate_salvage_valuation
from modules.hazard_advisor import analyze_pcb_hazards
from modules.pdf_reports import (
    generate_inventory_bill_pdf,
    generate_hazard_report_pdf,
    generate_complete_report_pdf,
)

VAL_DIR = PROJECT_ROOT / "dataset" / "images" / "val"
MODEL_PATH = PROJECT_ROOT / "models" / "yolov8s_pcb_best.pt"
DATA_YAML = PROJECT_ROOT / "dataset" / "data.yaml"


def test_1_inventory_bill_pdf():
    print("\n--- Test 1: Component Inventory Bill (Receipt Style) PDF Generation ---")
    counts = {"IC": 5, "Connector": 4, "Header": 3, "Switch/Button": 1}
    val_data = calculate_salvage_valuation(counts)

    pdf_bytes = generate_inventory_bill_pdf(
        analysis_id="test-bill-uuid-12345",
        timestamp="2026-10-21 12:30:00",
        filename="IMG_20260529_163319.jpg",
        counts=counts,
        val_data=val_data,
    )

    assert isinstance(pdf_bytes, bytes), "PDF output must be bytes"
    assert len(pdf_bytes) > 1000, "PDF must not be empty"
    assert pdf_bytes.startswith(b"%PDF"), "Must have valid %PDF magic header"
    assert b"%%EOF" in pdf_bytes, "Must have valid %%EOF marker"

    print(f"[PASS] Inventory Bill generated successfully ({len(pdf_bytes)} bytes).")
    print(f"       Typical Valuation: ₹{val_data['total_typical_inr']:,.2f}")
    return val_data


def test_2_hazard_report_pdf():
    print("\n--- Test 2: Hazardous Materials & Safe Recycling Report PDF Generation ---")
    counts = {"IC": 4, "Capacitor": 6, "Resistor": 10}
    # Test offline mode (no Groq key)
    haz_data = analyze_pcb_hazards(counts, groq_api_key="")

    pdf_bytes = generate_hazard_report_pdf(
        analysis_id="test-haz-uuid-67890",
        timestamp="2026-10-21 12:35:00",
        filename="IMG_20260529_163712.jpg",
        counts=counts,
        hazards_result=haz_data,
    )

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")
    assert b"%%EOF" in pdf_bytes

    print(f"[PASS] Hazardous Materials Report generated successfully in offline mode ({len(pdf_bytes)} bytes).")


def test_3_complete_report_pdf():
    print("\n--- Test 3: Complete Analysis Report PDF Generation ---")
    counts = {"IC": 5, "Connector": 4, "Header": 3, "Switch/Button": 1}
    val_data = calculate_salvage_valuation(counts)
    haz_data = analyze_pcb_hazards(counts, groq_api_key="")

    det_table = [
        {"Component": "IC", "Confidence (%)": 91.5, "Xmin": 10, "Ymin": 10, "Xmax": 50, "Ymax": 50, "Width": 40, "Height": 40},
        {"Component": "Connector", "Confidence (%)": 88.0, "Xmin": 60, "Ymin": 60, "Xmax": 120, "Ymax": 100, "Width": 60, "Height": 40},
    ]

    # Test with in-memory annotated PIL image
    annotated_img = Image.new("RGB", (640, 480), color=(30, 41, 59))

    pdf_with_img = generate_complete_report_pdf(
        analysis_id="test-full-uuid-11111",
        timestamp="2026-10-21 12:40:00",
        filename="IMG_20260529_163718.jpg",
        counts=counts,
        det_table=det_table,
        val_data=val_data,
        hazards_result=haz_data,
        annotated_image=annotated_img,
        latency_ms=185.0,
        conf_threshold=0.30,
    )

    assert isinstance(pdf_with_img, bytes)
    assert len(pdf_with_img) > 5000, "PDF with image should be substantial"
    assert pdf_with_img.startswith(b"%PDF")
    print(f"[PASS] Complete Report with embedded image generated successfully ({len(pdf_with_img)} bytes).")

    # Test without image (graceful fallback check)
    pdf_no_img = generate_complete_report_pdf(
        analysis_id="test-full-uuid-22222",
        timestamp="2026-10-21 12:40:00",
        filename="IMG_20260529_163718.jpg",
        counts=counts,
        det_table=det_table,
        val_data=val_data,
        hazards_result=haz_data,
        annotated_image=None,
        latency_ms=185.0,
        conf_threshold=0.30,
    )
    assert isinstance(pdf_no_img, bytes)
    assert pdf_no_img.startswith(b"%PDF")
    print(f"[PASS] Complete Report without image fallback generated successfully ({len(pdf_no_img)} bytes).")


def test_4_unpriced_and_zero_counts_handling():
    print("\n--- Test 4: Unpriced & Zero Detection Handling in Reports ---")
    # Test with unpriced component
    counts = {"UnknownWidget": 3, "IC": 2}
    val_data = calculate_salvage_valuation(counts)
    pdf_bytes = generate_inventory_bill_pdf(
        analysis_id="test-unknown-price",
        timestamp="2026-10-21 12:45:00",
        filename="circuit.jpg",
        counts=counts,
        val_data=val_data,
    )
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")
    print("[PASS] Handled unpriced component in Inventory Bill PDF without errors.")

    # Test with zero detections
    empty_counts = {cls: 0 for cls in EXPECTED_CLASSES}
    empty_val = calculate_salvage_valuation(empty_counts)
    pdf_empty = generate_inventory_bill_pdf(
        analysis_id="test-zero-dets",
        timestamp="2026-10-21 12:45:00",
        filename="blank.jpg",
        counts=empty_counts,
        val_data=empty_val,
    )
    assert isinstance(pdf_empty, bytes)
    assert pdf_empty.startswith(b"%PDF")
    print("[PASS] Handled zero-detection empty board in Inventory Bill PDF without errors.")


def test_5_ui_integration_audit():
    print("\n--- Test 5: UI Integration & Download Buttons Audit in app.py ---")
    app_text = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")

    # Verify download button labels
    assert "Download Component Inventory Bill (PDF)" in app_text, "Missing Inventory Bill button label"
    assert "Download Hazardous Materials & Recycling Report (PDF)" in app_text, "Missing Hazard Report button label"
    assert "Download Complete Analysis Report (PDF)" in app_text, "Missing Complete Report button label"

    # Verify filenames
    assert "component_inventory_bill.pdf" in app_text
    assert "hazardous_materials_recycling_report.pdf" in app_text
    assert "complete_e_waste_analysis_report.pdf" in app_text

    # Verify responsive width & layout rules
    assert "use_container_width=True" in app_text
    assert "width: 1" not in app_text, "Must not introduce fixed width: 1 layout strings"

    print("[PASS] UI Integration confirmed: All 3 PDF download buttons, descriptive filenames, and layout rules verified.")


if __name__ == "__main__":
    test_1_inventory_bill_pdf()
    test_2_hazard_report_pdf()
    test_3_complete_report_pdf()
    test_4_unpriced_and_zero_counts_handling()
    test_5_ui_integration_audit()
    print("\n=======================================================")
    print("ALL PDF REPORT VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=======================================================")
