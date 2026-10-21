"""Verification script for Phase 2: Model Loading and Validation Inference.

Tests YOLOv8s PCB model checkpoint loading, internal class names,
inference execution on an existing validation image, detection counts,
confidence scores, and CPU latency.
"""

from collections import Counter
from pathlib import Path
import time
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolov8s_pcb_best.pt"
VAL_IMAGE = PROJECT_ROOT / "dataset" / "images" / "val" / "IMG_20260529_163319.jpg"
OUTPUT_DIR = PROJECT_ROOT / "runs"


def run_test():
    print(f"[*] Testing model path: {MODEL_PATH}")
    assert MODEL_PATH.exists(), f"Model checkpoint not found at {MODEL_PATH}"
    print(f"[*] Model file size: {MODEL_PATH.stat().st_size:,} bytes")

    # 1. Load model
    print("[*] Loading YOLO model...")
    t0 = time.time()
    model = YOLO(str(MODEL_PATH))
    load_time = time.time() - t0
    print(f"[+] Model loaded successfully in {load_time:.2f} seconds.")

    # 2. Inspect classes stored in checkpoint
    print(f"[*] Checkpoint class names: {model.names}")
    expected_classes = [
        "Resistor", "Capacitor", "Diode", "LED", "IC",
        "Connector", "Switch/Button", "Header", "Crystal", "Debug/TP"
    ]
    model_class_list = [model.names[i] for i in range(len(model.names))]
    assert model_class_list == expected_classes, (
        f"Class mismatch! Model: {model_class_list} vs Expected: {expected_classes}"
    )
    print("[+] Class names in model checkpoint match data.yaml exactly.")

    # 3. Run inference on sample validation image
    print(f"[*] Running inference on: {VAL_IMAGE}")
    assert VAL_IMAGE.exists(), f"Validation image not found: {VAL_IMAGE}"

    t1 = time.time()
    results = model.predict(
        source=str(VAL_IMAGE),
        conf=0.25,
        iou=0.45,
        imgsz=960,
        device="cpu",
        verbose=False,
    )
    inference_time = (time.time() - t1) * 1000  # ms
    print(f"[+] Inference completed in {inference_time:.1f} ms on CPU.")

    result = results[0]
    boxes = result.boxes
    total_detections = len(boxes)
    print(f"[+] Total detected components: {total_detections}")

    counts = Counter()
    detected_items = []
    for box in boxes:
        cls_id = int(box.cls[0].item())
        cls_name = model.names[cls_id]
        conf = float(box.conf[0].item())
        xyxy = [round(v, 1) for v in box.xyxy[0].tolist()]
        counts[cls_name] += 1
        detected_items.append({
            "class": cls_name,
            "confidence": round(conf, 3),
            "bbox_xyxy": xyxy,
        })

    print("[*] Detections by class:")
    for cls_name, count in sorted(counts.items()):
        print(f"    - {cls_name}: {count}")

    print("\n[*] Sample high-confidence detections:")
    for item in sorted(detected_items, key=lambda x: x["confidence"], reverse=True)[:6]:
        print(f"    - {item['class']} ({item['confidence']*100:.1f}%): BBox {item['bbox_xyxy']}")

    # 4. Save visualization to runs/
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "val_inference_result.jpg"
    result.save(filename=str(out_path))
    print(f"[+] Annotated detection image saved to: {out_path} ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    run_test()

