# 🌿 Smart E-Waste Analyzer (SEWA)
### Intelligent E-Waste Circuit Component Classifier & Material Analyzer

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8s-brightgreen.svg)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Dataset License](https://img.shields.io/badge/Dataset_License-CC_BY_4.0-lightgrey.svg)](LICENSE-DATASET)

---

## 📌 Overview

**Smart E-Waste Analyzer** is a computer-vision powered application designed for circular electronics manufacturing, automated recycling sorting, and educational optical inspection.

Powered by a custom-trained **YOLOv8s** architecture fine-tuned on real-world printed circuit boards (PCBs), the platform detects and classifies **10 electronic component categories**, computes second-hand component salvage valuations in **Indian Rupees (₹ / INR)**, generates safety and toxicological recycling advisories, and exports multi-page audit-grade PDF reports.

---

## ✨ Key Capabilities

1. **High-Accuracy Component Detection (YOLOv8s)**:
   - Trained specifically at 960px resolution on real PCB surfaces.
   - Accurately detects and localizes surface-mount and through-hole electronic parts with bounding boxes, confidence percentages, and dimensions.
2. **10 Electronic Component Classes**:
   - `Resistor`, `Capacitor`, `Diode`, `LED`, `IC` (Integrated Circuit), `Connector`, `Switch/Button`, `Header`, `Crystal`, and `Debug/TP` (Test Points).
3. **Second-Hand Salvage Valuation**:
   - Computes low, typical, and high secondary-market recovery values in **INR (₹)** based on curated benchmark datasets.
4. **Hazardous Materials & Safe Recycling Advisory**:
   - Identifies flame retardants (TBBP-A), lead/tin solder considerations (RoHS compliance), heavy metals, and electrolyte contents.
   - Built-in local offline safety knowledge base + optional cloud AI enrichment using Groq API (`llama-3.3-70b-versatile`).
5. **PDF Documentation Generation**:
   - **🧾 Component Inventory Bill**: Supermarket receipt-style itemized bill with line totals.
   - **⚠️ Hazardous Materials Report**: Toxicological profiles and safe disposal guides.
   - **📑 Complete Analysis Report**: Full technical audit report with embedded annotated PCB imagery.
6. **Local SQLite History Persistence**:
   - Tracks past runs, metrics, and valuations in an offline SQLite database (`app_data/history.db`).

---

## 📂 Project Architecture

```
PCB-Component-Detection-main/
│
├── .env.example                 # Environment configuration template
├── .gitignore                   # Ignored files, cache directories, and local DBs
├── app.py                       # Main Streamlit web application & UI
├── requirements.txt             # Project Python dependencies
├── LICENSE                      # Apache 2.0 software license
├── LICENSE-DATASET              # CC BY 4.0 dataset license
│
├── dataset/                     # YOLO-formatted dataset
│   ├── data.yaml                # Class names & YOLO dataset mappings
│   ├── images/                  # Train and validation PCB images
│   └── labels/                  # YOLO bounding box annotations (.txt)
│
├── models/                      # Deep learning model checkpoints
│   └── yolov8s_pcb_best.pt      # Fine-tuned YOLOv8s PCB weights (960px)
│
├── modules/                     # Core application engine modules
│   ├── analyzer.py              # Second-hand salvage pricing & valuation logic
│   ├── detector.py              # YOLO inference, image sanitization & predictions
│   ├── hazard_advisor.py        # Environmental safety, RoHS & Groq LLM advisor
│   ├── history.py               # SQLite analysis history persistence
│   └── pdf_reports.py           # ReportLab PDF report generation
│
├── tests/                       # Automated test & verification suite
│   ├── test_app_suite.py        # Core unit tests (inference, sanitization, zero-count)
│   ├── test_complete_ps2_suite.py # End-to-end integration tests
│   ├── test_inference.py        # Model loading & standalone inference test
│   ├── test_pdf_reports.py      # PDF generation test suite
│   ├── test_redesign_suite.py   # Stateful UI workflow simulation
│   ├── test_ui_redesign_inr.py  # Currency formatting & tab verification
│   └── test_ui_results_visibility.py # Visualization & result audit tests
│
├── tools/                       # Utilities and helper scripts
│   └── sanitize_metadata.py     # EXIF metadata stripper for dataset images
│
└── training/                    # Model training documentation & logs
    └── results.csv              # Metric history (mAP50, loss, precision, recall)
```

---

## 🚀 Quick Start

### 1. Prerequisites
- **Python**: Version 3.10, 3.11, or 3.12.
- **Hardware**: CPU supported; CUDA-enabled GPU optional for faster inference.

### 2. Installation

Clone or extract the repository, then install dependencies:

```bash
# Clone the repository
git clone https://github.com/your-username/PCB-Component-Detection.git
cd PCB-Component-Detection

# (Recommended) Create and activate a virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Environment Setup (Optional)

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

To enable AI-enriched environmental advisories, add your free [Groq API Key](https://console.groq.com/) in `.env`:
```ini
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```
*(If no API key is provided, the application automatically uses its comprehensive built-in offline knowledge base).*

### 4. Running the Web Application

Launch the Streamlit interactive dashboard:

```bash
streamlit run app.py
```

Open your browser at **`http://localhost:8501`**.

---

## 🔍 CLI Usage & Model Training

### Running Standalone Inference via YOLO CLI
```bash
yolo detect predict model=models/yolov8s_pcb_best.pt source=dataset/images/val/IMG_20260529_163319.jpg imgsz=960 conf=0.30
```

### Running Validation on Dataset
```bash
yolo detect val model=models/yolov8s_pcb_best.pt data=dataset/data.yaml imgsz=960
```

### Continuing Training
```bash
yolo detect train model=models/yolov8s_pcb_best.pt data=dataset/data.yaml epochs=50 imgsz=960
```

---

## 🧪 Automated Testing

Execute the automated test suites using Python:

```bash
# Test YOLO model loading and CPU inference
python tests/test_inference.py

# Test core detection pipeline & input validation
python tests/test_app_suite.py

# Run complete integration test suite
python tests/test_complete_ps2_suite.py

# Verify PDF report generation
python tests/test_pdf_reports.py
```

---

## 🏷️ Classes & Detection Palette

| ID | Class Name | Description | Bounding Box Color |
|:---|:---|:---|:---|
| 0 | **Resistor** | Surface mount (SMD) & through-hole resistors | Deep Orange |
| 1 | **Capacitor** | Ceramic, electrolytic, and tantalum capacitors | Blue |
| 2 | **Diode** | Rectifiers, Zener, and Schottky diodes | Purple |
| 3 | **LED** | Light emitting indicators and SMD LEDs | Yellow |
| 4 | **IC** | Integrated Circuits (QFP, SOP, DIP, BGA packages) | Magenta / Pink |
| 5 | **Connector** | USB, DC power jacks, terminal blocks | Cyan |
| 6 | **Switch/Button** | Tactile buttons, DIP switches, toggles | Green |
| 7 | **Header** | Pin headers, male/female Dupont headers | Orange |
| 8 | **Crystal** | Quartz crystal oscillators & resonators | Indigo |
| 9 | **Debug/TP** | Test pads, gold contact pads, debug pins | Grey |

---

## 📄 License & Attribution

- **Dataset & Annotations**: Licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE-DATASET).
- **Code & Model Weights**: Licensed under the [Apache License 2.0](LICENSE).
