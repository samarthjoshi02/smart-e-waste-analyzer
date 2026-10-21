
"""Smart E-Waste Analyzer: Modern Streamlit Application.

Problem Statement 2 (PS 2): Intelligent E-Waste Circuit Component Classifier & Material Analyzer.
Provides:
  - Clean white/light theme with sustainability-inspired green and teal accents.
  - Guided 3-step workflow for beginners and hackathon judges.
  - Immediate display of YOLOv8s bounding boxes, component counts, charts, and detection tables.
  - Second-hand salvage valuations displayed exclusively in Indian Rupees (INR, ₹).
  - Safe recycling guidance organized into 5 plain-English sections.
  - SQLite analysis history with INR valuations.
"""

import time
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

# Internal protected modules
from modules.analyzer import (
    build_component_inventory,
    calculate_salvage_valuation,
)
from modules.detector import (
    EXPECTED_CLASSES,
    PCBDetector,
    get_available_samples,
)
from modules.hazard_advisor import (
    analyze_pcb_hazards,
)
from modules.history import (
    clear_all_history,
    get_history_records,
    init_history_db,
    save_analysis_record,
)
from modules.pdf_reports import (
    generate_complete_report_pdf,
    generate_hazard_report_pdf,
    generate_inventory_bill_pdf,
)

# -----------------------------------------------------------------------------
# Configuration and Path Setup
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolov8s_pcb_best.pt"
DATA_YAML_PATH = PROJECT_ROOT / "dataset" / "data.yaml"
VAL_DIR = PROJECT_ROOT / "dataset" / "images" / "val"

# Initialize local SQLite history database on startup
init_history_db()

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart E-Waste Analyzer",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Clean White & Sustainability Theme Styling (CSS)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Main container background and readable dark text */
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #0F172A !important;
        font-weight: 700;
        letter-spacing: -0.015em;
    }

    /* Hero header banner */
    .hero-banner {
        background: linear-gradient(135deg, #F0FDFA 0%, #FFFFFF 100%);
        border: 1px solid #CCFBF1;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0F172A;
        margin: 0 0 6px 0;
    }
    .hero-subtitle {
        font-size: 1.02rem;
        color: #475569;
        margin: 0 0 14px 0;
        line-height: 1.5;
    }

    /* Workflow step pills */
    .step-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 8px;
    }
    .step-pill {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        color: #475569;
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .step-pill.active {
        background: #0D9488;
        border-color: #0D9488;
        color: #FFFFFF;
    }

    /* White content card containers */
    .content-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .content-card.teal-border {
        border-left: 4px solid #0D9488;
    }
    .content-card.green-border {
        border-left: 4px solid #059669;
    }
    .content-card.amber-border {
        border-left: 4px solid #D97706;
    }

    /* Summary metric cards */
    .metric-box {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #0D9488;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }
    .metric-box.green {
        border-left-color: #059669;
        background-color: #F0FDF4;
    }
    .metric-box.amber {
        border-left-color: #D97706;
        background-color: #FFFBEB;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.55rem;
        font-weight: 800;
        color: #0F172A;
        margin-top: 4px;
    }

    /* Class tally badge */
    .badge-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 10px 12px;
        text-align: center;
        margin-bottom: 8px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
    }
    .badge-card.has-count {
        border-color: #0D9488;
        background: #F0FDFA;
    }
    .badge-num {
        font-size: 1.35rem;
        font-weight: 800;
        color: #0D9488;
    }
    .badge-name {
        font-size: 0.8rem;
        color: #334155;
        font-weight: 600;
    }

    /* Section guidance cards */
    .guide-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 14px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }
    .guide-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Clean Streamlit Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid #E2E8F0;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #FFFFFF;
        border-radius: 8px 8px 0 0;
        border: 1px solid #E2E8F0;
        border-bottom: none;
        padding: 8px 18px;
        color: #64748B;
        font-weight: 600;
        font-size: 0.92rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #F0FDFA !important;
        border-color: #0D9488 !important;
        color: #0D9488 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Cached Model Loader (Loads YOLOv8s weights once)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing AI Component Detection Engine...")
def get_detector() -> PCBDetector:
    """Loads and caches the model once across session reruns."""
    return PCBDetector(model_path=MODEL_PATH, data_yaml_path=DATA_YAML_PATH)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
def init_session_state():
    """Ensures necessary session state keys exist."""
    defaults = {
        "active_image_bytes": None,
        "active_image_name": None,
        "active_image_source": None,  # 'upload' or 'sample'
        "sanitized_image": None,
        "analysis_results": None,
        "analyzed_conf": None,
        "current_analysis_id": None,
        "history_reopened_record": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def clear_analysis_state():
    """Resets analysis results when a new image is selected."""
    st.session_state["analysis_results"] = None
    st.session_state["analyzed_conf"] = None
    st.session_state["current_analysis_id"] = None


def reset_all_state():
    """Fully resets session to empty landing state."""
    st.session_state["active_image_bytes"] = None
    st.session_state["active_image_name"] = None
    st.session_state["active_image_source"] = None
    st.session_state["sanitized_image"] = None
    st.session_state["analysis_results"] = None
    st.session_state["analyzed_conf"] = None
    st.session_state["current_analysis_id"] = None
    st.session_state["history_reopened_record"] = None


# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
def main():
    init_session_state()
    detector = get_detector()

    # -------------------------------------------------------------------------
    # Sidebar: Simple, Clean Status & Reassuring Guidance
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.subheader("🌿 Smart E-Waste Analyzer")
        st.caption("Intelligent Circuit Component Classifier & Material Analyzer")

        # Friendly Reassuring Status Indicator
        st.markdown(
            """
            <div style="background:#F0FDF4; border:1px solid #BBF7D0; border-radius:8px; padding:10px 14px; margin-bottom:14px;">
                <span style="color:#059669; font-weight:700;">● Active</span>
                <span style="color:#15803D; font-size:0.85rem; margin-left:8px; font-weight:600;">AI component detection is ready</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Advanced Settings in a clean, optional collapsed expander
        with st.expander("⚙️ Advanced Settings (Optional)", expanded=False):
            conf_slider = st.slider(
                "Minimum detection confidence",
                min_value=0.10,
                max_value=0.90,
                value=0.30,
                step=0.05,
                help="Higher values show fewer, more confident detections; lower values include subtle components.",
            )
            iou_slider = st.slider(
                "Overlap suppression (IoU)",
                min_value=0.20,
                max_value=0.70,
                value=0.45,
                step=0.05,
                help="Controls merging of overlapping bounding boxes.",
            )

        st.markdown("### 🏷️ 10 Detectable Categories")
        st.caption(
            "Capacitor, IC (Microchip), Resistor, Diode, LED, Switch/Button, Connector, Header, Crystal, Debug/TP"
        )

        st.divider()
        st.caption("Hackathon PS 2: Intelligent E-Waste Circuit Component Classifier & Material Analyzer")

    # -------------------------------------------------------------------------
    # Welcoming Hero Banner (🏠 Home)
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">🏠 Home — Smart E-Waste Circuit Component Classifier</div>
            <div class="hero-subtitle">
                Upload an electronic circuit board photograph to detect components, 
                estimate second-hand reuse value in Indian Rupees (₹), and review responsible e-waste recycling guidance.
            </div>
            <div class="step-container">
                <div class="step-pill active">Step 1: Upload a PCB photo or choose a sample</div>
                <div class="step-pill">Step 2: Click Analyze PCB</div>
                <div class="step-pill">Step 3: Review components, value in ₹, and safe recycling advice</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # STEP 1: Choose or Upload a Circuit Board
    # -------------------------------------------------------------------------
    st.markdown("### 📥 STEP 1: Choose or Upload a Circuit Board")
    tab_upload, tab_sample = st.tabs(["📤 Upload PCB Image", "🖼️ Try Sample PCB Gallery"])

    # UPLOAD TAB
    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload a circuit board photo (JPG, JPEG, PNG, WEBP — up to 15 MB):",
            type=["jpg", "jpeg", "png", "webp"],
            help="Ensure the circuit board is clearly lit and focused. Max file size: 15 MB.",
            key="user_file_uploader",
        )
        if uploaded_file is not None:
            raw_bytes = uploaded_file.getvalue()
            if st.session_state["active_image_name"] != uploaded_file.name or st.session_state["active_image_source"] != "upload":
                try:
                    sanitized = PCBDetector.validate_and_sanitize_image(raw_bytes, uploaded_file.name)
                    st.session_state["active_image_bytes"] = raw_bytes
                    st.session_state["active_image_name"] = uploaded_file.name
                    st.session_state["active_image_source"] = "upload"
                    st.session_state["sanitized_image"] = sanitized
                    clear_analysis_state()
                    st.rerun()
                except ValueError as ve:
                    st.error(f"❌ Upload Error: {ve}")

    # SAMPLE GALLERY TAB
    with tab_sample:
        sample_list = get_available_samples(VAL_DIR)
        if not sample_list:
            st.warning("⚠️ No validation sample images found in dataset/images/val/. Please use the manual uploader.")
        else:
            st.caption(f"Select one of {len(sample_list)} verified validation PCB images to test:")
            cols = st.columns(min(len(sample_list), 4))
            for i, sample in enumerate(sample_list[:4]):
                with cols[i]:
                    try:
                        thumb = Image.open(sample["path"])
                        st.image(thumb, use_container_width=True)
                        st.caption(f"**{sample['name']}** ({sample['size_kb']} KB)")
                        if st.button(f"Use Sample {i+1}", key=f"btn_sample_{i}", use_container_width=True):
                            raw_bytes = sample["path"].read_bytes()
                            sanitized = PCBDetector.validate_and_sanitize_image(raw_bytes, sample["name"])
                            st.session_state["active_image_bytes"] = raw_bytes
                            st.session_state["active_image_name"] = sample["name"]
                            st.session_state["active_image_source"] = "sample"
                            st.session_state["sanitized_image"] = sanitized
                            clear_analysis_state()
                            st.rerun()
                    except Exception as e:
                        st.caption(f"Error loading {sample['name']}: {e}")

            sample_options = [s["name"] for s in sample_list]
            chosen_sample_name = st.selectbox(
                "Or choose from all available sample images:",
                options=sample_options,
                index=0,
                key="sample_dropdown_picker",
            )
            if st.button("Load Selected Sample", key="btn_load_dropdown"):
                sample_path = VAL_DIR / chosen_sample_name
                raw_bytes = sample_path.read_bytes()
                sanitized = PCBDetector.validate_and_sanitize_image(raw_bytes, chosen_sample_name)
                st.session_state["active_image_bytes"] = raw_bytes
                st.session_state["active_image_name"] = chosen_sample_name
                st.session_state["active_image_source"] = "sample"
                st.session_state["sanitized_image"] = sanitized
                clear_analysis_state()
                st.rerun()

    active_img = st.session_state.get("sanitized_image")
    active_name = st.session_state.get("active_image_name")
    active_bytes = st.session_state.get("active_image_bytes")

    # -------------------------------------------------------------------------
    # STEP 2: Preview & Click Analyze PCB
    # -------------------------------------------------------------------------
    if active_img is not None:
        st.divider()
        st.markdown("### 🔍 STEP 2: Image Preview & Analysis")

        prev_col1, prev_col2 = st.columns([1, 2])
        with prev_col1:
            st.image(active_img, caption="Loaded Image Preview", use_container_width=True)

        with prev_col2:
            st.markdown(
                f"""
                <div class="content-card">
                    <h4 style="margin:0 0 10px 0; color:#0F172A;">Image Properties</h4>
                    <p style="margin:4px 0; color:#334155;"><strong>File Name:</strong> {active_name}</p>
                    <p style="margin:4px 0; color:#334155;"><strong>Dimensions:</strong> {active_img.width} × {active_img.height} pixels</p>
                    <p style="margin:4px 0; color:#334155;"><strong>File Size:</strong> {len(active_bytes)/(1024*1024):.2f} MB</p>
                    <p style="margin:4px 0; color:#334155;"><strong>Active Confidence Threshold:</strong> {conf_slider:.2f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            btn_col1, btn_col2 = st.columns([1, 1])
            with btn_col1:
                analyze_clicked = st.button("🔍 Analyze PCB", type="primary", use_container_width=True)
            with btn_col2:
                if st.button("🔄 Clear / Select Another Image", use_container_width=True):
                    reset_all_state()
                    st.rerun()

            if analyze_clicked:
                with st.spinner("Analyzing circuit board components with YOLOv8s..."):
                    pred = detector.predict(
                        image=active_img,
                        conf_threshold=conf_slider,
                        iou_threshold=iou_slider,
                        imgsz=960,
                    )
                    analysis_id = str(uuid.uuid4())
                    st.session_state["analysis_results"] = pred
                    st.session_state["analyzed_conf"] = conf_slider
                    st.session_state["current_analysis_id"] = analysis_id

                    val_summary = calculate_salvage_valuation(pred["counts_by_class"])
                    top_det_name = pred["detections_table"][0]["Component"] if pred["detections_table"] else "None"
                    top_det_conf = pred["detections_table"][0]["Confidence (%)"] if pred["detections_table"] else 0.0

                    history_record = {
                        "analysis_uuid": analysis_id,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "filename": active_name,
                        "total_components": pred["total_detections"],
                        "categories_count": sum(1 for c, n in pred["counts_by_class"].items() if n > 0),
                        "top_confidence": top_det_conf,
                        "top_component": top_det_name,
                        "inference_time_ms": pred["inference_time_ms"],
                        "conf_threshold": conf_slider,
                        "counts_by_class": pred["counts_by_class"],
                        "valuation_typical_usd": val_summary["total_typical_usd"],
                        "valuation_low_usd": val_summary["total_low_usd"],
                        "valuation_high_usd": val_summary["total_high_usd"],
                    }
                    save_analysis_record(history_record)
                    st.rerun()

    else:
        # Informative empty state card when no image is loaded
        st.markdown(
            """
            <div class="content-card" style="text-align: center; padding: 36px 20px; margin-top: 20px;">
                <h3 style="color:#0F172A; margin-bottom: 8px;">No Circuit Board Selected Yet</h3>
                <p style="color:#64748B; max-width: 580px; margin: 0 auto 16px auto; line-height: 1.6;">
                    Upload a photograph of an electronic circuit board or choose one of our sample boards above.
                    Once an image is loaded, click <strong>Analyze PCB</strong> to detect components, calculate reuse value in Indian Rupees, 
                    and review safe recycling practices.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Quick check for past history when in empty state
        recent_history = get_history_records(limit=5)
        if recent_history:
            with st.expander("📜 Past Analysis History (Quick Check)", expanded=False):
                st.caption(f"Displaying {len(recent_history)} past analyses stored in local SQLite (`app_data/history.db`):")
                hist_summary_data = []
                for h in recent_history:
                    inr_val = h.get("valuation_typical_inr") or round(float(h.get("valuation_typical_usd", 0.0)) * 85.0, 2)
                    hist_summary_data.append({
                        "Date / Time": h["timestamp"],
                        "File Name": h["filename"],
                        "Total Parts": h["total_components"],
                        "Categories": h["categories_count"],
                        "Est. Value (INR)": f"₹{inr_val:,.2f}",
                    })
                st.dataframe(pd.DataFrame(hist_summary_data), use_container_width=True, hide_index=True)
        return

    # -------------------------------------------------------------------------
    # STEP 3: Complete Results Section (Immediately Visible After Analysis)
    # -------------------------------------------------------------------------
    results = st.session_state.get("analysis_results")
    if results is None:
        return

    st.divider()
    st.markdown("### 📊 STEP 3: Component Analysis, Inventory & Valuation")

    total_dets = results["total_detections"]
    counts = results["counts_by_class"]
    det_table = results["detections_table"]
    latency = results["inference_time_ms"]
    annotated_img = results["annotated_image"]
    analyzed_conf = st.session_state.get("analyzed_conf", conf_slider)
    active_categories = sum(1 for c, n in counts.items() if n > 0)
    top_detection = f"{det_table[0]['Confidence (%)']}% ({det_table[0]['Component']})" if det_table else "None"
    val_data = calculate_salvage_valuation(counts)
    hazards_result = analyze_pcb_hazards(counts_by_class=counts)

    # 1. Success Notification Above the Full Results
    st.markdown(
        f"""
        <div class="content-card green-border" style="background:#F0FDF4; margin-bottom:16px;">
            <h4 style="color:#15803D; margin-top:0;">✅ Analysis Completed Successfully</h4>
            <p style="color:#166534; margin:0; line-height:1.5;">
                Identified <strong>{total_dets} components</strong> across 
                <strong>{active_categories} categories</strong> in <strong>{latency:.0f} ms</strong>.
                Typical estimated reuse value: <strong>₹{val_data['total_typical_inr']:,.2f}</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Four Summary Metrics Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Total Components</div>
                <div class="metric-value">{total_dets}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Categories Found</div>
                <div class="metric-value">{active_categories} <span style="font-size:0.9rem; color:#64748B;">of {len(EXPECTED_CLASSES)}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Highest Confidence</div>
                <div class="metric-value" style="font-size:1.25rem;">{top_detection}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Detection Speed</div>
                <div class="metric-value">{latency:.0f} <span style="font-size:0.9rem; color:#64748B;">ms</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Slider change indicator
    if conf_slider != analyzed_conf:
        st.info(
            f"ℹ️ Minimum confidence slider adjusted to **{conf_slider:.2f}** (Displaying results from threshold **{analyzed_conf:.2f}**). "
            "Click **Analyze PCB** above to re-evaluate with this new threshold."
        )

    # -------------------------------------------------------------------------
    # 5 Dedicated Tabs for Complete PS 2 Requirements
    # -------------------------------------------------------------------------
    tab_dets, tab_inv, tab_val, tab_haz, tab_hist = st.tabs([
        "🎯 Detection Results",
        "📋 Component Inventory",
        "💰 Reuse Value",
        "♻️ Safe Recycling Guide",
        "📜 Analysis History",
    ])

    # =========================================================================
    # TAB 1: DETECTION RESULTS (Active by default, immediately visible)
    # =========================================================================
    with tab_dets:
        st.markdown("#### 🖼️ Visual Comparison: Original vs YOLOv8s Annotated")
        img_c1, img_c2 = st.columns(2)
        with img_c1:
            st.markdown(f"**Original PCB Photo:** `{active_name}`")
            st.image(active_img, use_container_width=True)
        with img_c2:
            st.markdown(f"**YOLOv8s Detected Bounding Boxes:** ({total_dets} components at confidence ≥ {analyzed_conf:.2f})")
            st.image(annotated_img, use_container_width=True)

        if total_dets == 0:
            st.warning(
                f"⚠️ No components were localized above the {analyzed_conf:.2f} confidence threshold. "
                "Try lowering the confidence threshold under 'Advanced Settings' in the sidebar and click Analyze PCB again."
            )
        else:
            # 10-Class Component Breakdown Badges
            st.markdown("#### 🧩 Component Counts by Category")
            tally_cols = st.columns(5)
            for idx, cls_name in enumerate(detector.class_names):
                col_target = tally_cols[idx % 5]
                cnt = counts.get(cls_name, 0)
                card_class = "badge-card has-count" if cnt > 0 else "badge-card"
                with col_target:
                    st.markdown(
                        f"""
                        <div class="{card_class}">
                            <div class="badge-num">{cnt}</div>
                            <div class="badge-name">{cls_name}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Readable Visual Bar Chart Breakdown
            active_counts = {cls: counts[cls] for cls in detector.class_names if counts.get(cls, 0) > 0}
            if active_counts:
                st.markdown("##### 📊 Detected Components Chart Breakdown")
                chart_df = pd.DataFrame(
                    list(active_counts.items()),
                    columns=["Component", "Count"],
                ).set_index("Component")
                st.bar_chart(chart_df, color="#0D9488", use_container_width=True)

            # Detailed Bounding-Box Detection List Table
            st.markdown("#### 📋 Detailed Detection List (Coordinates & Confidence)")
            df_dets = pd.DataFrame(det_table)
            df_display = df_dets.drop(columns=["Confidence_Raw"], errors="ignore")
            st.dataframe(
                df_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Component": st.column_config.TextColumn("Component Name", width="medium"),
                    "Confidence (%)": st.column_config.ProgressColumn(
                        "Confidence",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "Xmin": st.column_config.NumberColumn("Xmin"),
                    "Ymin": st.column_config.NumberColumn("Ymin"),
                    "Xmax": st.column_config.NumberColumn("Xmax"),
                    "Ymax": st.column_config.NumberColumn("Ymax"),
                    "Width": st.column_config.NumberColumn("Width (px)"),
                    "Height": st.column_config.NumberColumn("Height (px)"),
                },
            )

    # =========================================================================
    # TAB 2: COMPONENT INVENTORY
    # =========================================================================
    with tab_inv:
        st.markdown("#### 📋 Component Inventory & Reuse Feasibility")
        st.markdown(
            """
            This inventory aggregates the detected components, estimates their salvage value in Indian Rupees (₹), 
            and outlines practical next steps for testing and recovery.
            """
        )

        st.info(
            "⚠️ **Visual Inspection Boundary:** "
            "The computer vision model recognizes component packaging from an image, but **cannot verify electrical functionality**. "
            "All items are considered **'Potentially Recoverable — Inspection Required'** until desoldered and tested with appropriate bench equipment."
        )

        inventory_data = build_component_inventory(det_table, counts)
        if not inventory_data:
            st.warning("No components detected to inventory.")
        else:
            inv_df = pd.DataFrame(inventory_data)
            st.dataframe(
                inv_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Component Name": st.column_config.TextColumn("Component", width="small"),
                    "Quantity Detected": st.column_config.NumberColumn("Quantity", width="small"),
                    "Average Confidence (%)": st.column_config.ProgressColumn(
                        "Avg Confidence", format="%.1f%%", min_value=0, max_value=100
                    ),
                    "Potential Reuse": st.column_config.TextColumn("Potential Reuse", width="medium"),
                    "Indicative Value (INR)": st.column_config.TextColumn("Indicative Value (INR)", width="medium"),
                    "Recommended Next Step": st.column_config.TextColumn("Recommended Inspection & Testing Protocol", width="large"),
                },
            )

            st.caption(
                f"Generated from {total_dets} detected components across {active_categories} categories."
            )

    # =========================================================================
    # TAB 3: ESTIMATED REUSE VALUE (INR)
    # =========================================================================
    with tab_val:
        st.markdown("#### 💰 Second-Hand Component Resale Value Estimation (INR)")
        st.markdown(
            """
            Indicative second-hand reuse values calculated using surplus component benchmarks and repair community averages, 
            converted to Indian Rupees (₹).
            """
        )

        # 3 Valuation Metric Cards in INR
        v1, v2, v3 = st.columns(3)
        with v1:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Conservative (Low) Estimate</div>
                    <div class="metric-value">₹{val_data['total_low_inr']:,.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with v2:
            st.markdown(
                f"""
                <div class="metric-box green">
                    <div class="metric-label">Typical Indicative Value</div>
                    <div class="metric-value" style="color:#059669;">₹{val_data['total_typical_inr']:,.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with v3:
            st.markdown(
                f"""
                <div class="metric-box amber">
                    <div class="metric-label">Optimistic (High) Estimate</div>
                    <div class="metric-value" style="color:#D97706;">₹{val_data['total_high_inr']:,.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Itemized Pricing Table in INR exclusively
        st.markdown("##### 💵 Itemized Component Valuation Breakdown (INR)")
        val_items_df = pd.DataFrame(val_data["itemized_breakdown"])

        inr_display_cols = [
            "Component",
            "Quantity",
            "Unit Price Range (INR)",
            "Typical Est. Value (INR)",
            "Low Est. (INR)",
            "High Est. (INR)",
            "Pricing Status",
            "Pricing Basis",
        ]
        clean_val_df = val_items_df[[c for c in inr_display_cols if c in val_items_df.columns]]

        st.dataframe(
            clean_val_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Component": st.column_config.TextColumn("Component", width="small"),
                "Quantity": st.column_config.NumberColumn("Quantity", width="small"),
                "Unit Price Range (INR)": st.column_config.TextColumn("Unit Price Range (INR)", width="medium"),
                "Typical Est. Value (INR)": st.column_config.TextColumn("Typical Est. Value (INR)", width="medium"),
                "Low Est. (INR)": st.column_config.TextColumn("Low Est. (INR)", width="small"),
                "High Est. (INR)": st.column_config.TextColumn("High Est. (INR)", width="small"),
                "Pricing Status": st.column_config.TextColumn("Pricing Status", width="small"),
                "Pricing Basis": st.column_config.TextColumn("Pricing Basis", width="large"),
            },
        )

        st.markdown(
            """
            <div class="content-card" style="background:#F8FAFC; border:1px solid #E2E8F0; margin-top:14px;">
                <p style="margin:0 0 6px 0; color:#334155; font-size:0.92rem;">
                    <strong>📌 Benchmark Pricing Basis:</strong> Surplus electronics pricing benchmark (October 2026), 
                    converted to Indian Rupees at the fixed exchange rate of <strong>1 USD = ₹85.00</strong>.
                </p>
                <p style="margin:0; color:#64748B; font-size:0.88rem;">
                    <strong>⚠️ Valuation Disclaimer:</strong> Estimates represent theoretical replacement value of functional components. 
                    Realized salvage revenue depends heavily on clean desoldering without heat damage, pin straightness, 
                    electrical verification, and local electronics market demand.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =========================================================================
    # TAB 4: SAFE RECYCLING GUIDE (5 PLAIN-ENGLISH SECTIONS)
    # =========================================================================
    with tab_haz:
        st.markdown("#### ♻️ Safe Recycling & Material Guide")
        st.markdown(
            """
            Electronic circuit boards contain valuable recoverable materials alongside potentially hazardous chemicals. 
            Below is practical guidance for safe handling and responsible e-waste recycling.
            """
        )

        hazards_result = analyze_pcb_hazards(counts_by_class=counts)
        friendly_status = hazards_result.get("user_friendly_status", "Using built-in recycling guidance.")
        st.markdown(
            f"""
            <div style="background:#F0FDF4; border:1px solid #BBF7D0; border-radius:8px; padding:10px 16px; margin-bottom:18px;">
                <span style="color:#059669; font-weight:700;">Recycling Advisory Status:</span>
                <span style="color:#15803D; margin-left:8px; font-weight:600;">{friendly_status}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # AI-Assisted commentary if available
        if hazards_result.get("ai_explanation"):
            st.markdown(
                f"""
                <div class="content-card teal-border">
                    <h5 style="color:#0D9488; margin-top:0;">🤖 AI-Assisted Safety Commentary</h5>
                    <div style="color:#334155; line-height:1.6; font-size:0.94rem;">
                        {hazards_result['ai_explanation']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 5 Plain-English Sections
        st.markdown(
            """
            <div class="guide-card">
                <div class="guide-header">
                    <span>🔬 A. What might be present?</span>
                </div>
                <p style="color:#334155; margin:0 0 8px 0; line-height:1.6;">
                    Circuit boards are made from a mix of metals, plastics, and semiconductors:
                </p>
                <ul style="color:#475569; margin:0 0 8px 0; padding-left:22px; line-height:1.6;">
                    <li><strong>Solder:</strong> Older circuit boards (pre-2006) often contain Tin-Lead solder. Modern consumer boards typically use RoHS-compliant lead-free solders (Tin, Silver, Copper).</li>
                    <li><strong>Circuit Board Base (FR-4):</strong> Woven fiberglass bonded with epoxy resins containing brominated flame retardants (like TBBPA) to prevent electrical fires.</li>
                    <li><strong>Components:</strong> Integrated circuits contain silicon, microscopic gold or copper wire bonds, and trace metals. Capacitors may contain organic chemical electrolytes.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="guide-card">
                <div class="guide-header">
                    <span>⚠️ B. Why should I care?</span>
                </div>
                <p style="color:#334155; margin:0 0 8px 0; line-height:1.6;">
                    Improper disposal of e-waste poses real risks to personal health and the environment:
                </p>
                <ul style="color:#475569; margin:0 0 8px 0; padding-left:22px; line-height:1.6;">
                    <li><strong>Groundwater Contamination:</strong> If circuit boards end up in open landfills, lead and heavy metals can leach into soil and nearby water sources over time.</li>
                    <li><strong>Toxic Fumes:</strong> Heating plastics or burning circuit boards in open air releases toxic brominated dioxin gases and hazardous fumes.</li>
                    <li><strong>Resource Depletion:</strong> Throwing away circuit boards wastes precious metals (gold, copper, silver) that can be recovered through authorized industrial recycling.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="guide-card">
                <div class="guide-header">
                    <span>🛡️ C. What should I do?</span>
                </div>
                <p style="color:#334155; margin:0 0 8px 0; line-height:1.6;">
                    Follow these simple safety practices when handling old electronic boards:
                </p>
                <ul style="color:#475569; margin:0 0 8px 0; padding-left:22px; line-height:1.6;">
                    <li><strong>Wear nitrile gloves</strong> when handling dusty, broken, or weathered circuit boards.</li>
                    <li><strong>Do NOT burn, crush, or grind</strong> circuit boards into dust at home.</li>
                    <li><strong>Never use home acid baths</strong> (such as nitric acid) to extract gold. Acid leaching creates lethal nitrogen dioxide fumes and hazardous liquid waste.</li>
                    <li><strong>Desolder safely:</strong> Always work in a well-ventilated space with a dedicated fume extractor when desoldering parts.</li>
                    <li><strong>Store cleanly:</strong> Keep discarded electronics in a dry cardboard box out of reach of children and pets until recycling.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="guide-card">
                <div class="guide-header">
                    <span>🤖 D. What can the AI actually know?</span>
                </div>
                <p style="color:#334155; margin:0 0 8px 0; line-height:1.6;">
                    It is important to understand the honest boundaries of computer vision technology:
                </p>
                <ul style="color:#475569; margin:0 0 8px 0; padding-left:22px; line-height:1.6;">
                    <li><strong>What the AI sees:</strong> Visual shapes, sizes, outlines, and packaging types of components from a 2D photograph.</li>
                    <li><strong>What the AI CANNOT know:</strong> Whether solder contains lead or is lead-free (requires X-ray XRF spectrometry), whether a chip is functional or burnt out, exact metal weights, or RoHS certification status without laboratory testing.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="guide-card">
                <div class="guide-header">
                    <span>🏢 E. Where can I recycle?</span>
                </div>
                <p style="color:#334155; margin:0 0 8px 0; line-height:1.6;">
                    Ensure your e-waste reaches responsible circular recycling channels:
                </p>
                <ul style="color:#475569; margin:0 0 8px 0; padding-left:22px; line-height:1.6;">
                    <li><strong>Authorized E-Waste Centers:</strong> Deliver non-functional boards to government-authorized e-waste collection centers or certified facilities (e-Stewards or R2v3).</li>
                    <li><strong>Manufacturer Return Programs:</strong> Major electronics brands provide free take-back kiosks and drop-off points for old devices.</li>
                    <li><strong>Community E-Waste Drives:</strong> Watch for municipal electronic waste collection events rather than throwing electronics into household trash.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Component-Specific Hazards Table
        if hazards_result["identified_hazards"]:
            st.markdown("##### 🔬 Chemical Details for Detected Components")
            haz_rows = []
            for h in hazards_result["identified_hazards"]:
                haz_rows.append({
                    "Component": f"{h['component']} ({h['count']}x)",
                    "Potential Substances": ", ".join(h["substances"]),
                    "Health & Environmental Risk": h["environmental_risk"],
                    "Safe Handling Tip": h["safe_handling"],
                })
            st.dataframe(
                pd.DataFrame(haz_rows),
                use_container_width=True,
                hide_index=True,
            )

        st.caption(f"📌 {hazards_result['disclaimer']}")

    # =========================================================================
    # TAB 5: ANALYSIS HISTORY (SQLITE)
    # =========================================================================
    with tab_hist:
        st.markdown("#### 📜 Saved Analysis History")
        st.markdown(
            """
            Review previously analyzed circuit boards stored locally on this machine in SQLite (`app_data/history.db`).
            """
        )

        history_records = get_history_records(limit=50)
        if not history_records:
            st.info("No saved analysis history yet. Analyze a circuit board above to record your first run.")
        else:
            st.caption(f"Displaying {len(history_records)} saved analyses (values in INR, ₹):")
            hist_table_data = []
            for h in history_records:
                inr_val = h.get("valuation_typical_inr") or round(float(h.get("valuation_typical_usd", 0.0)) * 85.0, 2)
                hist_table_data.append({
                    "ID": h["id"],
                    "Date & Time": h["timestamp"],
                    "Image Name": h["filename"],
                    "Total Components": h["total_components"],
                    "Categories": h["categories_count"],
                    "Top Component": f"{h['top_component']} ({h['top_confidence']:.1f}%)",
                    "Detection Time": f"{h['inference_time_ms']:.0f} ms",
                    "Typical Est. Value (INR)": f"₹{inr_val:,.2f}",
                })

            st.dataframe(pd.DataFrame(hist_table_data), use_container_width=True, hide_index=True)

            st.divider()
            c_col1, c_col2 = st.columns([1, 2])
            with c_col1:
                confirm_clear = st.checkbox("I confirm I want to clear all history records.", key="chk_confirm_clear")
                if st.button("🗑️ Clear History", type="secondary", disabled=not confirm_clear, use_container_width=True):
                    if clear_all_history():
                        st.success("Analysis history successfully cleared.")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Could not clear history.")
            with c_col2:
                st.caption(
                    "**Privacy & Local Storage:** Analysis history is stored locally in SQLite (`app_data/history.db`). "
                    "Original uploaded images are processed in-memory and are never stored permanently."
                )

    # -------------------------------------------------------------------------
    # Download Reports Section
    # -------------------------------------------------------------------------
    st.divider()
    st.markdown("### 📥 Download Reports")
    st.caption("Export professional PDF documentation generated from current analysis results.")

    analysis_uuid = st.session_state.get("current_analysis_id", "SEWA-RUN")
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Generate in-memory PDF reports
    try:
        pdf_bill_bytes = generate_inventory_bill_pdf(
            analysis_id=analysis_uuid,
            timestamp=timestamp_str,
            filename=active_name,
            counts=counts,
            val_data=val_data,
        )
    except Exception as e:
        pdf_bill_bytes = b""
        st.warning(f"Note: Inventory Bill generation encountered an issue: {e}")

    try:
        pdf_haz_bytes = generate_hazard_report_pdf(
            analysis_id=analysis_uuid,
            timestamp=timestamp_str,
            filename=active_name,
            counts=counts,
            hazards_result=hazards_result,
        )
    except Exception as e:
        pdf_haz_bytes = b""
        st.warning(f"Note: Hazardous Materials Report generation encountered an issue: {e}")

    try:
        pdf_comp_bytes = generate_complete_report_pdf(
            analysis_id=analysis_uuid,
            timestamp=timestamp_str,
            filename=active_name,
            counts=counts,
            det_table=det_table,
            val_data=val_data,
            hazards_result=hazards_result,
            annotated_image=annotated_img,
            latency_ms=latency,
            conf_threshold=analyzed_conf,
        )
    except Exception as e:
        pdf_comp_bytes = b""
        st.warning(f"Note: Complete Analysis Report generation encountered an issue: {e}")

    dl_col1, dl_col2, dl_col3 = st.columns(3)
    with dl_col1:
        st.markdown(
            """
            <div class="content-card" style="margin-bottom: 8px;">
                <h5 style="margin-top:0; color:#0F172A;">🧾 Component Inventory Bill</h5>
                <p style="color:#64748B; font-size:0.85rem; margin-bottom:4px;">
                    Itemized supermarket receipt style bill with unit benchmarks, quantities, and line totals in INR (₹).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            label="Download Component Inventory Bill (PDF)",
            data=pdf_bill_bytes,
            file_name="component_inventory_bill.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="btn_dl_inventory_bill",
        )

    with dl_col2:
        st.markdown(
            """
            <div class="content-card" style="margin-bottom: 8px;">
                <h5 style="margin-top:0; color:#0F172A;">⚠️ Hazardous Materials Report</h5>
                <p style="color:#64748B; font-size:0.85rem; margin-bottom:4px;">
                    Evidence-based toxicological profiles, safe handling practices, and scientific limitations.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            label="Download Hazardous Materials & Recycling Report (PDF)",
            data=pdf_haz_bytes,
            file_name="hazardous_materials_recycling_report.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="btn_dl_hazard_report",
        )

    with dl_col3:
        st.markdown(
            """
            <div class="content-card" style="margin-bottom: 8px;">
                <h5 style="margin-top:0; color:#0F172A;">📑 Complete Analysis Report</h5>
                <p style="color:#64748B; font-size:0.85rem; margin-bottom:4px;">
                    Full technical audit including detection metrics, annotated PCB image, inventory bill, and recycling guide.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            label="Download Complete Analysis Report (PDF)",
            data=pdf_comp_bytes,
            file_name="complete_e_waste_analysis_report.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="btn_dl_complete_report",
        )

    # -------------------------------------------------------------------------
    # Bottom Action Bar
    # -------------------------------------------------------------------------
    st.markdown("---")
    foot_col1, foot_col2, _foot_col3 = st.columns([1, 1, 2])
    with foot_col1:
        if st.button("🔄 Start New Analysis", key="btn_foot_reset", use_container_width=True):
            reset_all_state()
            st.rerun()
    with foot_col2:
        if st.button("⚡ Re-Analyze (Current Threshold)", key="btn_foot_reanalyze", use_container_width=True):
            with st.spinner("Re-analyzing with current settings..."):
                pred = detector.predict(
                    image=active_img,
                    conf_threshold=conf_slider,
                    iou_threshold=iou_slider,
                    imgsz=960,
                )
                analysis_id = str(uuid.uuid4())
                st.session_state["analysis_results"] = pred
                st.session_state["analyzed_conf"] = conf_slider
                st.session_state["current_analysis_id"] = analysis_id

                val_summary = calculate_salvage_valuation(pred["counts_by_class"])
                top_det_name = pred["detections_table"][0]["Component"] if pred["detections_table"] else "None"
                top_det_conf = pred["detections_table"][0]["Confidence (%)"] if pred["detections_table"] else 0.0

                history_record = {
                    "analysis_uuid": analysis_id,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "filename": active_name,
                    "total_components": pred["total_detections"],
                    "categories_count": sum(1 for c, n in pred["counts_by_class"].items() if n > 0),
                    "top_confidence": top_det_conf,
                    "top_component": top_det_name,
                    "inference_time_ms": pred["inference_time_ms"],
                    "conf_threshold": conf_slider,
                    "counts_by_class": pred["counts_by_class"],
                    "valuation_typical_usd": val_summary["total_typical_usd"],
                    "valuation_low_usd": val_summary["total_low_usd"],
                    "valuation_high_usd": val_summary["total_high_usd"],
                }
                save_analysis_record(history_record)
                st.rerun()


if __name__ == "__main__":
    main()
