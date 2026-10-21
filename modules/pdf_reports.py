"""Professional PDF Report Generation Module for Smart E-Waste Analyzer.

Problem Statement 2 (PS 2): Intelligent E-Waste Circuit Component Classifier & Material Analyzer.
Provides three downloadable PDF reports:
  1. Component Inventory Bill (Supermarket receipt style with itemized amounts & totals in INR).
  2. Hazardous Materials & Safe Recycling Report (Evidence-based safety guidance & scientific limitations).
  3. Complete Analysis Report (Comprehensive report combining metrics, annotated image, inventory, and recycling).

Generated entirely in-memory using ReportLab without temporary files.
"""

from datetime import datetime
import io
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# -----------------------------------------------------------------------------
# Font Registration & Typography Setup
# -----------------------------------------------------------------------------
FONT_REGULAR = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
FONT_MONO = "Courier"
FONT_MONO_BOLD = "Courier-Bold"
HAS_RUPEE_GLYPH = False

def _init_fonts() -> None:
    """Initializes TrueType fonts that natively support the Indian Rupee symbol (₹)."""
    global FONT_REGULAR, FONT_BOLD, FONT_MONO, FONT_MONO_BOLD, HAS_RUPEE_GLYPH

    font_candidates = [
        ("Arial", "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
        ("SegoeUI", "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf"),
        ("Calibri", "C:/Windows/Fonts/calibri.ttf", "C:/Windows/Fonts/calibrib.ttf"),
        ("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]

    for name, reg_path, bold_path in font_candidates:
        if os.path.exists(reg_path):
            try:
                pdfmetrics.registerFont(TTFont(name, reg_path))
                if os.path.exists(bold_path):
                    pdfmetrics.registerFont(TTFont(f"{name}-Bold", bold_path))
                    FONT_BOLD = f"{name}-Bold"
                else:
                    FONT_BOLD = name
                FONT_REGULAR = name
                HAS_RUPEE_GLYPH = True
                break
            except Exception:
                continue

    # Also check for Consolas / Courier New for receipt styling
    mono_candidates = [
        ("Consolas", "C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/consolab.ttf"),
        ("CourierNew", "C:/Windows/Fonts/cour.ttf", "C:/Windows/Fonts/courbd.ttf"),
    ]
    for name, reg_path, bold_path in mono_candidates:
        if os.path.exists(reg_path):
            try:
                pdfmetrics.registerFont(TTFont(name, reg_path))
                if os.path.exists(bold_path):
                    pdfmetrics.registerFont(TTFont(f"{name}-Bold", bold_path))
                    FONT_MONO_BOLD = f"{name}-Bold"
                else:
                    FONT_MONO_BOLD = name
                FONT_MONO = name
                break
            except Exception:
                continue

_init_fonts()


def _format_inr(val: Any) -> str:
    """Formats a float or string into an INR currency string.
    
    If TrueType font with Rupee glyph is available, uses ₹.
    Otherwise gracefully uses Rs. to prevent encoding errors on standard Helvetica.
    """
    symbol = "₹" if HAS_RUPEE_GLYPH else "Rs. "
    if isinstance(val, (int, float)):
        return f"{symbol}{val:,.2f}"
    val_str = str(val).strip()
    if not val_str or val_str.lower() in ("price unavailable", "none", "nan"):
        return "Price unavailable"
    # If string already starts with ₹ and font doesn't have glyph, replace it
    if val_str.startswith("₹") and not HAS_RUPEE_GLYPH:
        return "Rs. " + val_str[1:].strip()
    return val_str


# -----------------------------------------------------------------------------
# Report Palette (Matching Clean White & Teal Application UI)
# -----------------------------------------------------------------------------
PRIMARY_TEAL = colors.HexColor("#0D9488")
DARK_TEAL = colors.HexColor("#0F766E")
LIGHT_TEAL = colors.HexColor("#F0FDFA")
DEEP_NAVY = colors.HexColor("#0F172A")
SLATE_BODY = colors.HexColor("#334155")
MUTED_SLATE = colors.HexColor("#64748B")
BORDER_GRAY = colors.HexColor("#CBD5E1")
LIGHT_GRAY_BG = colors.HexColor("#F8FAFC")
CARD_BG = colors.HexColor("#FFFFFF")
SUCCESS_GREEN = colors.HexColor("#059669")
SUCCESS_BG = colors.HexColor("#F0FDF4")
AMBER_WARNING = colors.HexColor("#D97706")


# -----------------------------------------------------------------------------
# Numbered Canvas for Running Headers and Footers
# -----------------------------------------------------------------------------
class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and display total page count."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states: List[Dict[str, Any]] = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_page_elements(num_pages)
            super().showPage()
        super().save()

    def _draw_page_elements(self, page_count: int):
        self.saveState()
        # Footer line and text
        self.setStrokeColor(BORDER_GRAY)
        self.setLineWidth(0.5)
        self.line(36, 32, 612 - 36, 32)

        self.setFont(FONT_REGULAR, 7.5)
        self.setFillColor(MUTED_SLATE)
        self.drawString(36, 20, "Smart E-Waste Analyzer • Autonomous Circuit Classifier & Material Analyzer")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 36, 20, page_str)
        self.restoreState()


# -----------------------------------------------------------------------------
# Styles Factory
# -----------------------------------------------------------------------------
def _get_report_styles() -> Dict[str, ParagraphStyle]:
    """Returns a dictionary of cohesive ParagraphStyles for report generation."""
    styles = getSampleStyleSheet()

    custom_styles = {
        "DocTitle": ParagraphStyle(
            "DocTitle",
            parent=styles["Normal"],
            fontName=FONT_BOLD,
            fontSize=18,
            leading=22,
            textColor=DEEP_NAVY,
            alignment=0,
            spaceAfter=4,
        ),
        "DocSubtitle": ParagraphStyle(
            "DocSubtitle",
            parent=styles["Normal"],
            fontName=FONT_REGULAR,
            fontSize=10,
            leading=14,
            textColor=MUTED_SLATE,
            alignment=0,
            spaceAfter=12,
        ),
        "SectionHeader": ParagraphStyle(
            "SectionHeader",
            parent=styles["Normal"],
            fontName=FONT_BOLD,
            fontSize=12,
            leading=16,
            textColor=DEEP_NAVY,
            spaceBefore=10,
            spaceAfter=6,
        ),
        "Body": ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontName=FONT_REGULAR,
            fontSize=8.5,
            leading=12,
            textColor=SLATE_BODY,
        ),
        "BodyBold": ParagraphStyle(
            "BodyBold",
            parent=styles["Normal"],
            fontName=FONT_BOLD,
            fontSize=8.5,
            leading=12,
            textColor=DEEP_NAVY,
        ),
        "MutedNote": ParagraphStyle(
            "MutedNote",
            parent=styles["Normal"],
            fontName=FONT_REGULAR,
            fontSize=7.5,
            leading=10.5,
            textColor=MUTED_SLATE,
        ),
        "TableHead": ParagraphStyle(
            "TableHead",
            parent=styles["Normal"],
            fontName=FONT_BOLD,
            fontSize=8,
            leading=11,
            textColor=DEEP_NAVY,
            alignment=0,
        ),
        "TableCell": ParagraphStyle(
            "TableCell",
            parent=styles["Normal"],
            fontName=FONT_REGULAR,
            fontSize=8,
            leading=11,
            textColor=SLATE_BODY,
        ),
        "TableCellBold": ParagraphStyle(
            "TableCellBold",
            parent=styles["Normal"],
            fontName=FONT_BOLD,
            fontSize=8,
            leading=11,
            textColor=DEEP_NAVY,
        ),
        "TableCellRight": ParagraphStyle(
            "TableCellRight",
            parent=styles["Normal"],
            fontName=FONT_REGULAR,
            fontSize=8,
            leading=11,
            textColor=SLATE_BODY,
            alignment=2,
        ),
        "TableCellRightBold": ParagraphStyle(
            "TableCellRightBold",
            parent=styles["Normal"],
            fontName=FONT_BOLD,
            fontSize=8,
            leading=11,
            textColor=DEEP_NAVY,
            alignment=2,
        ),
        # Supermarket Receipt Styles
        "ReceiptTitle": ParagraphStyle(
            "ReceiptTitle",
            parent=styles["Normal"],
            fontName=FONT_BOLD,
            fontSize=15,
            leading=18,
            textColor=DEEP_NAVY,
            alignment=1,
            spaceAfter=2,
        ),
        "ReceiptSubtitle": ParagraphStyle(
            "ReceiptSubtitle",
            parent=styles["Normal"],
            fontName=FONT_REGULAR,
            fontSize=9,
            leading=12,
            textColor=MUTED_SLATE,
            alignment=1,
            spaceAfter=8,
        ),
        "ReceiptMono": ParagraphStyle(
            "ReceiptMono",
            parent=styles["Normal"],
            fontName=FONT_MONO,
            fontSize=8,
            leading=11,
            textColor=SLATE_BODY,
        ),
        "ReceiptMonoBold": ParagraphStyle(
            "ReceiptMonoBold",
            parent=styles["Normal"],
            fontName=FONT_MONO_BOLD,
            fontSize=8.5,
            leading=11.5,
            textColor=DEEP_NAVY,
        ),
        "ReceiptMonoRight": ParagraphStyle(
            "ReceiptMonoRight",
            parent=styles["Normal"],
            fontName=FONT_MONO,
            fontSize=8,
            leading=11,
            textColor=SLATE_BODY,
            alignment=2,
        ),
        "ReceiptMonoRightBold": ParagraphStyle(
            "ReceiptMonoRightBold",
            parent=styles["Normal"],
            fontName=FONT_MONO_BOLD,
            fontSize=9,
            leading=12,
            textColor=DEEP_NAVY,
            alignment=2,
        ),
        "CalloutText": ParagraphStyle(
            "CalloutText",
            parent=styles["Normal"],
            fontName=FONT_REGULAR,
            fontSize=8,
            leading=11.5,
            textColor=SLATE_BODY,
        ),
    }

    return custom_styles


# =============================================================================
# 1. COMPONENT INVENTORY BILL (SUPERMARKET RECEIPT STYLE)
# =============================================================================
def generate_inventory_bill_pdf(
    analysis_id: str,
    timestamp: str,
    filename: str,
    counts: Dict[str, int],
    val_data: Dict[str, Any],
) -> bytes:
    """Generates a professional, supermarket-receipt styled Component Inventory Bill.
    
    Includes itemized items, quantities, benchmark prices in INR, conservative/typical/high
    totals, benchmark source details, and non-guaranteed valuation disclaimers.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=42,
    )

    styles = _get_report_styles()
    story: List[Any] = []
    page_width = letter[0] - 72  # 540 pt printable width

    # Receipt Header Box
    story.append(Paragraph("SMART E-WASTE ANALYZER", styles["ReceiptTitle"]))
    story.append(Paragraph("COMPONENT INVENTORY BILL & REUSE VALUATION", styles["ReceiptSubtitle"]))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY_TEAL, spaceBefore=4, spaceAfter=8))

    # Receipt Metadata Key-Value Grid
    ref_code = f"SEWA-{analysis_id[:8].upper()}" if analysis_id else "SEWA-RUN-001"
    total_components = sum(counts.values())
    categories_found = sum(1 for c, n in counts.items() if n > 0)

    meta_data = [
        [
            Paragraph(f"<b>RECEIPT REF:</b> {ref_code}", styles["ReceiptMonoBold"]),
            Paragraph(f"<b>DATE / TIME:</b> {timestamp}", styles["ReceiptMono"]),
        ],
        [
            Paragraph(f"<b>PCB IMAGE:</b> {filename}", styles["ReceiptMono"]),
            Paragraph(f"<b>TOTAL PARTS:</b> {total_components} ({categories_found} categories)", styles["ReceiptMono"]),
        ],
        [
            Paragraph("<b>CURRENCY:</b> Indian Rupees (INR, ₹)", styles["ReceiptMono"]),
            Paragraph("<b>BENCHMARK BASIS:</b> Fixed 1 USD = ₹85.00", styles["ReceiptMono"]),
        ],
    ]
    meta_table = Table(meta_data, colWidths=[270, 270])
    meta_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(meta_table)
    story.append(HRFlowable(width="100%", thickness=0.8, color=BORDER_GRAY, dash=[3, 3], spaceBefore=8, spaceAfter=8))

    # Itemized Receipt Table
    # Columns: # (24), Item Name (156), Qty (40), Benchmark Unit Price (160), Line Total (160) -> Total: 540 pt
    receipt_rows = [
        [
            Paragraph("<b>#</b>", styles["TableHead"]),
            Paragraph("<b>ITEM DESCRIPTION</b>", styles["TableHead"]),
            Paragraph("<b>QTY</b>", styles["TableCellRightBold"]),
            Paragraph("<b>BENCHMARK UNIT PRICE</b>", styles["TableCellRightBold"]),
            Paragraph("<b>TYPICAL EST. VALUE (INR)</b>", styles["TableCellRightBold"]),
        ]
    ]

    itemized = val_data.get("itemized_breakdown", [])
    row_idx = 1
    for item in itemized:
        if item.get("Quantity", 0) <= 0:
            continue
        comp_name = item.get("Component", "Unknown")
        qty = item.get("Quantity", 0)
        unit_price = item.get("Unit Price Range (INR)", "Price unavailable")
        line_total = item.get("Typical Est. Value (INR)", "Price unavailable")

        receipt_rows.append([
            Paragraph(str(row_idx), styles["ReceiptMono"]),
            Paragraph(comp_name, styles["ReceiptMonoBold"]),
            Paragraph(str(qty), styles["ReceiptMonoRight"]),
            Paragraph(unit_price, styles["ReceiptMonoRight"]),
            Paragraph(line_total, styles["ReceiptMonoRightBold"]),
        ])
        row_idx += 1

    if len(receipt_rows) == 1:
        receipt_rows.append([
            Paragraph("-", styles["ReceiptMono"]),
            Paragraph("No components localized above threshold", styles["ReceiptMono"]),
            Paragraph("0", styles["ReceiptMonoRight"]),
            Paragraph("—", styles["ReceiptMonoRight"]),
            Paragraph("₹0.00", styles["ReceiptMonoRight"]),
        ])

    items_table = Table(receipt_rows, colWidths=[24, 156, 40, 160, 160])
    items_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), LIGHT_GRAY_BG),
        ("LINEBELOW", (0, 0), (-1, 0), 1, DEEP_NAVY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 1), (-1, -1), 0.5, colors.HexColor("#F1F5F9")),
    ]))
    story.append(items_table)
    story.append(HRFlowable(width="100%", thickness=1, color=DEEP_NAVY, spaceBefore=8, spaceAfter=8))

    # Totals Section (Supermarket Style with Low, Typical, High bounds)
    typ_inr = val_data.get("total_typical_inr", 0.0)
    low_inr = val_data.get("total_low_inr", 0.0)
    high_inr = val_data.get("total_high_inr", 0.0)

    totals_data = [
        [
            Paragraph("<b>TOTAL RECOVERABLE ITEMS:</b>", styles["ReceiptMono"]),
            Paragraph(f"<b>{total_components} units</b>", styles["ReceiptMonoRight"]),
        ],
        [
            Paragraph("Conservative (Low) Estimate:", styles["ReceiptMono"]),
            Paragraph(_format_inr(low_inr), styles["ReceiptMonoRight"]),
        ],
        [
            Paragraph("<b>TYPICAL INDICATIVE RESALE VALUE:</b>", styles["ReceiptMonoBold"]),
            Paragraph(f"<b>{_format_inr(typ_inr)}</b>", styles["ReceiptMonoRightBold"]),
        ],
        [
            Paragraph("Optimistic (High) Estimate:", styles["ReceiptMono"]),
            Paragraph(_format_inr(high_inr), styles["ReceiptMonoRight"]),
        ],
    ]

    totals_table = Table(totals_data, colWidths=[340, 200])
    totals_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 2), (-1, 2), SUCCESS_BG),
        ("LINEABOVE", (0, 2), (-1, 2), 1, SUCCESS_GREEN),
        ("LINEBELOW", (0, 2), (-1, 2), 1, SUCCESS_GREEN),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(totals_table)
    story.append(HRFlowable(width="100%", thickness=0.8, color=BORDER_GRAY, dash=[3, 3], spaceBefore=10, spaceAfter=8))

    # Transparent Benchmark Source and Limitations Notice
    story.append(Paragraph("<b>PRICING METHODOLOGY & BENCHMARK NOTES:</b>", styles["BodyBold"]))
    story.append(Paragraph(
        "Indicative salvage estimates are derived from surplus component broker benchmarks (October 2026), "
        "converted from USD to INR at the fixed baseline rate of <b>1 USD = ₹85.00</b>. "
        "Unit prices reflect working second-hand parts in cosmetic grade B or higher.",
        styles["Body"],
    ))
    story.append(Spacer(1, 6))

    # Prominent Disclaimer (Required by prompt)
    disclaimer_box = [
        [
            Paragraph(
                "<b>VALUATION DISCLAIMER:</b> These are indicative component reuse estimates, not guaranteed market prices. "
                "Actual salvage value depends on exact part number, package condition, electrical verification, and local buyer demand.",
                styles["CalloutText"],
            )
        ]
    ]
    disc_table = Table(disclaimer_box, colWidths=[540])
    disc_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GRAY_BG),
        ("BOX", (0, 0), (-1, -1), 1, BORDER_GRAY),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(disc_table)

    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>*** END OF INVENTORY BILL — RECYCLE RESPONSIBLY ***</b>", styles["ReceiptSubtitle"]))

    doc.build(story, canvasmaker=NumberedCanvas)
    return buf.getvalue()


# =============================================================================
# 2. HAZARDOUS MATERIALS & SAFE RECYCLING REPORT
# =============================================================================
def generate_hazard_report_pdf(
    analysis_id: str,
    timestamp: str,
    filename: str,
    counts: Dict[str, int],
    hazards_result: Dict[str, Any],
) -> bytes:
    """Generates a readable Hazardous Materials Assessment & Safe Recycling Report.
    
    Includes report overview, material risks, universal board-level hazards, component-specific
    toxicology profiles, certified recycling advice, and honest scientific boundaries.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=42,
    )

    styles = _get_report_styles()
    story: List[Any] = []
    ref_code = f"SEWA-HAZ-{analysis_id[:8].upper()}" if analysis_id else "SEWA-HAZ-001"
    total_components = sum(counts.values())
    categories_found = sum(1 for c, n in counts.items() if n > 0)

    # Document Header
    story.append(Paragraph("Smart E-Waste Analyzer", styles["DocSubtitle"]))
    story.append(Paragraph("Hazardous Materials & Responsible E-Waste Recycling Report", styles["DocTitle"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY_TEAL, spaceBefore=4, spaceAfter=10))

    # Section A: Report Overview
    story.append(Paragraph("A. Report Overview", styles["SectionHeader"]))
    overview_data = [
        [
            Paragraph(f"<b>Report Reference:</b> {ref_code}", styles["TableCell"]),
            Paragraph(f"<b>Assessment Date:</b> {timestamp}", styles["TableCell"]),
        ],
        [
            Paragraph(f"<b>PCB Image:</b> {filename}", styles["TableCell"]),
            Paragraph(f"<b>Localized Components:</b> {total_components} units across {categories_found} categories", styles["TableCell"]),
        ],
        [
            Paragraph(f"<b>Advisory Status:</b> {hazards_result.get('user_friendly_status', 'Using built-in recycling guidance.')}", styles["TableCellBold"]),
            Paragraph("<b>Standards Baseline:</b> WEEE / RoHS / Basel Convention", styles["TableCell"]),
        ],
    ]
    overview_table = Table(overview_data, colWidths=[270, 270])
    overview_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GRAY_BG),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER_GRAY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(overview_table)
    story.append(Spacer(1, 10))

    # Optional AI-Assisted Narrative Commentary (if available via Groq)
    ai_explanation = hazards_result.get("ai_explanation")
    if ai_explanation:
        story.append(Paragraph("AI Environmental Safety Officer Commentary", styles["SectionHeader"]))
        ai_box = [[Paragraph(ai_explanation.replace("\n", "<br/>"), styles["Body"])]]
        ai_table = Table(ai_box, colWidths=[540])
        ai_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), LIGHT_TEAL),
            ("BOX", (0, 0), (-1, -1), 1, PRIMARY_TEAL),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(ai_table)
        story.append(Spacer(1, 10))

    # Section B: Possible Material Risks & Hazards
    story.append(Paragraph("B. Possible Material Risks & Hazards", styles["SectionHeader"]))

    # Universal Board-Level Risks
    board_hazards = hazards_result.get("board_level_hazards", {})
    solder_info = board_hazards.get("solder_joints", {})
    laminate_info = board_hazards.get("pcb_laminate", {})

    board_level_box = [
        [
            Paragraph(f"<b>1. {solder_info.get('title', 'Solder Joint Metallurgy')}</b>", styles["BodyBold"]),
        ],
        [
            Paragraph(solder_info.get("details", "Solder joints may contain Lead (Sn-Pb) in legacy electronics."), styles["Body"]),
        ],
        [
            Paragraph(f"<b>2. {laminate_info.get('title', 'Circuit Board Laminate (FR-4)')}</b>", styles["BodyBold"]),
        ],
        [
            Paragraph(laminate_info.get("details", "FR-4 epoxy contains brominated flame retardants (TBBPA)."), styles["Body"]),
        ],
    ]
    bl_table = Table(board_level_box, colWidths=[540])
    bl_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GRAY_BG),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER_GRAY),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(bl_table)
    story.append(Spacer(1, 8))

    # Component-Specific Hazards Table
    identified_hazards = hazards_result.get("identified_hazards", [])
    if identified_hazards:
        story.append(Paragraph("<b>Component-Specific Chemical Profiles</b>", styles["BodyBold"]))
        haz_table_data = [
            [
                Paragraph("<b>Component</b>", styles["TableHead"]),
                Paragraph("<b>Potential Substances</b>", styles["TableHead"]),
                Paragraph("<b>Health & Environmental Concerns</b>", styles["TableHead"]),
                Paragraph("<b>Safe Handling Guidance</b>", styles["TableHead"]),
            ]
        ]
        for h in identified_hazards:
            comp_label = f"{h['component']} ({h['count']}x)"
            substances_str = ", ".join(h.get("substances", []))
            risk_str = h.get("environmental_risk", "Low acute risk.")
            handling_str = h.get("safe_handling", "Standard recycling.")

            haz_table_data.append([
                Paragraph(comp_label, styles["TableCellBold"]),
                Paragraph(substances_str, styles["TableCell"]),
                Paragraph(risk_str, styles["TableCell"]),
                Paragraph(handling_str, styles["TableCell"]),
            ])

        # Widths: 90 + 130 + 160 + 160 = 540 pt
        h_table = Table(haz_table_data, colWidths=[90, 130, 160, 160])
        h_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), LIGHT_GRAY_BG),
            ("LINEBELOW", (0, 0), (-1, 0), 1, DEEP_NAVY),
            ("GRID", (0, 0), (-1, -1), 0.5, BORDER_GRAY),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(h_table)
    else:
        story.append(Paragraph("<i>No specific hazardous component classes localized on this board.</i>", styles["Body"]))

    story.append(Spacer(1, 10))

    # Section C: Safe Handling & Recycling Recommendations
    story.append(Paragraph("C. Safe Handling & Recycling Recommendations", styles["SectionHeader"]))
    rec_points = [
        "<b>Keep Away from Children & Pets:</b> Store dismantled electronic circuit boards in dry, enclosed containers.",
        "<b>No Open-Air Burning:</b> Burning circuit boards releases toxic polybrominated dibenzodioxins and acid fumes.",
        "<b>No Backyard Acid Leaching:</b> Never use nitric acid or home aqua regia baths to extract gold — produces deadly nitrogen dioxide gas.",
        "<b>Do Not Crush or Pulverize:</b> Avoid grinding boards without HEPA industrial negative-pressure ventilation to prevent inhaling hazardous dust.",
        "<b>Wear Personal Protective Equipment:</b> Use nitrile gloves and eye protection when handling broken or weathered circuit boards.",
        "<b>Certified Recycling Channels:</b> Route non-functional circuit boards exclusively to certified R2v3 (Responsible Recycling) or e-Stewards facilities equipped with industrial baghouse filtration and automated hydrometallurgical refining.",
    ]
    for pt in rec_points:
        story.append(Paragraph(f"• {pt}", styles["Body"]))
        story.append(Spacer(1, 2))

    story.append(Spacer(1, 8))

    # Section D: Important Scientific Limitation
    story.append(Paragraph("D. Important Scientific Limitation", styles["SectionHeader"]))
    limitation_text = (
        "<b>CRITICAL SCIENTIFIC BOUNDARY:</b> Computer vision and 2D photographic inspection recognize visual "
        "component shapes and packaging types, but <b>CANNOT confirm exact chemical composition, actual hazardous-substance "
        "percentages, or RoHS exemption compliance</b>. Exact metallurgy (such as verifying Tin-Lead vs Lead-Free solders) "
        "requires laboratory X-ray fluorescence (XRF) spectrometry or destructive chemical digestion. "
        "No chemical percentages or lab certifications are fabricated by this system."
    )
    lim_table = Table([[Paragraph(limitation_text, styles["CalloutText"])]], colWidths=[540])
    lim_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GRAY_BG),
        ("BOX", (0, 0), (-1, -1), 1, AMBER_WARNING),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(lim_table)

    doc.build(story, canvasmaker=NumberedCanvas)
    return buf.getvalue()


# =============================================================================
# 3. COMPLETE ANALYSIS REPORT (ALL-IN-ONE COMPREHENSIVE PDF)
# =============================================================================
def generate_complete_report_pdf(
    analysis_id: str,
    timestamp: str,
    filename: str,
    counts: Dict[str, int],
    det_table: List[Dict[str, Any]],
    val_data: Dict[str, Any],
    hazards_result: Dict[str, Any],
    annotated_image: Optional[Image.Image] = None,
    latency_ms: Optional[float] = None,
    conf_threshold: Optional[float] = None,
) -> bytes:
    """Generates a complete, multi-page comprehensive audit report.
    
    Combines:
      - Executive summary & metrics
      - Embedded annotated PCB image with bounding boxes
      - Supermarket-style component inventory bill
      - Hazardous materials profile & safety instructions
      - Scientific boundaries and legal disclaimers
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=42,
    )

    styles = _get_report_styles()
    story: List[Any] = []
    ref_code = f"SEWA-FULL-{analysis_id[:8].upper()}" if analysis_id else "SEWA-FULL-001"
    total_components = sum(counts.values())
    categories_found = sum(1 for c, n in counts.items() if n > 0)

    # 1. Title Banner
    story.append(Paragraph("Smart E-Waste Analyzer • Autonomous Technical Audit", styles["DocSubtitle"]))
    story.append(Paragraph("Comprehensive Printed Circuit Board Analysis Report", styles["DocTitle"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY_TEAL, spaceBefore=4, spaceAfter=8))

    # 2. Executive Metadata Grid
    top_comp = det_table[0]["Component"] if det_table else "None"
    top_conf = f"{det_table[0]['Confidence (%)']:.1f}%" if det_table else "0.0%"
    speed_str = f"{latency_ms:.0f} ms" if latency_ms else "N/A"
    conf_str = f"{conf_threshold:.2f}" if conf_threshold else "0.30"

    exec_meta = [
        [
            Paragraph(f"<b>Audit Reference:</b> {ref_code}", styles["TableCell"]),
            Paragraph(f"<b>Date & Time:</b> {timestamp}", styles["TableCell"]),
            Paragraph(f"<b>Source Image:</b> {filename}", styles["TableCell"]),
        ],
        [
            Paragraph(f"<b>Total Detected:</b> {total_components} units", styles["TableCellBold"]),
            Paragraph(f"<b>Categories:</b> {categories_found} types", styles["TableCellBold"]),
            Paragraph(f"<b>Top Confidence:</b> {top_conf} ({top_comp})", styles["TableCell"]),
        ],
        [
            Paragraph(f"<b>Detection Speed:</b> {speed_str}", styles["TableCell"]),
            Paragraph(f"<b>Min Confidence:</b> {conf_str}", styles["TableCell"]),
            Paragraph(f"<b>Typical Reuse Value:</b> {_format_inr(val_data.get('total_typical_inr', 0.0))}", styles["TableCellBold"]),
        ],
    ]
    meta_tbl = Table(exec_meta, colWidths=[180, 180, 180])
    meta_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GRAY_BG),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER_GRAY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_tbl)
    story.append(Spacer(1, 10))

    # 3. Embedded Annotated PCB Image (Visual Inspection Proof)
    story.append(Paragraph("1. Computer Vision Detection Visualization (YOLOv8s)", styles["SectionHeader"]))
    if annotated_image is not None:
        try:
            # Resize image maintaining aspect ratio to fit page nicely (max width 500, max height 260)
            img_w, img_h = annotated_image.size
            max_w, max_h = 500.0, 260.0
            scale = min(max_w / img_w, max_h / img_h, 1.0)
            render_w = img_w * scale
            render_h = img_h * scale

            # Convert to in-memory JPEG bytes
            img_bytes_buf = io.BytesIO()
            # Convert RGBA to RGB if needed
            if annotated_image.mode in ("RGBA", "P"):
                annotated_image = annotated_image.convert("RGB")
            annotated_image.save(img_bytes_buf, format="JPEG", quality=85)
            img_bytes_buf.seek(0)

            rl_img = RLImage(img_bytes_buf, width=render_w, height=render_h)
            story.append(rl_img)
            story.append(Paragraph(f"<i>Figure 1: Automated YOLOv8s bounding-box component annotations for {filename}.</i>", styles["MutedNote"]))
        except Exception as e:
            story.append(Paragraph(f"<i>[Visual image embedding skipped: {e}]</i>", styles["MutedNote"]))
    else:
        story.append(Paragraph("<i>[Annotated image preview unavailable for in-memory embedding]</i>", styles["MutedNote"]))

    story.append(Spacer(1, 10))

    # 4. Component Inventory Bill Table
    story.append(Paragraph("2. Component Inventory & Second-Hand Valuation Bill", styles["SectionHeader"]))
    bill_rows = [
        [
            Paragraph("<b>#</b>", styles["TableHead"]),
            Paragraph("<b>Component Description</b>", styles["TableHead"]),
            Paragraph("<b>Qty</b>", styles["TableCellRightBold"]),
            Paragraph("<b>Unit Price (INR)</b>", styles["TableCellRightBold"]),
            Paragraph("<b>Typical Est. (INR)</b>", styles["TableCellRightBold"]),
        ]
    ]
    idx = 1
    for item in val_data.get("itemized_breakdown", []):
        if item.get("Quantity", 0) <= 0:
            continue
        bill_rows.append([
            Paragraph(str(idx), styles["TableCell"]),
            Paragraph(item.get("Component", ""), styles["TableCellBold"]),
            Paragraph(str(item.get("Quantity", 0)), styles["TableCellRight"]),
            Paragraph(item.get("Unit Price Range (INR)", "Price unavailable"), styles["TableCellRight"]),
            Paragraph(item.get("Typical Est. Value (INR)", "Price unavailable"), styles["TableCellRightBold"]),
        ])
        idx += 1

    bill_table = Table(bill_rows, colWidths=[24, 166, 40, 155, 155])
    bill_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), LIGHT_GRAY_BG),
        ("LINEBELOW", (0, 0), (-1, 0), 1, DEEP_NAVY),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_GRAY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(bill_table)
    story.append(Spacer(1, 6))

    # Valuation Summary Box
    low_inr = val_data.get("total_low_inr", 0.0)
    typ_inr = val_data.get("total_typical_inr", 0.0)
    high_inr = val_data.get("total_high_inr", 0.0)

    val_summary_box = [
        [
            Paragraph(f"<b>Conservative (Low) Total:</b> {_format_inr(low_inr)}", styles["TableCell"]),
            Paragraph(f"<b>TYPICAL INDICATIVE TOTAL:</b> {_format_inr(typ_inr)}", styles["TableCellBold"]),
            Paragraph(f"<b>Optimistic (High) Total:</b> {_format_inr(high_inr)}", styles["TableCell"]),
        ]
    ]
    val_tbl = Table(val_summary_box, colWidths=[180, 180, 180])
    val_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), SUCCESS_BG),
        ("BOX", (0, 0), (-1, -1), 1, SUCCESS_GREEN),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(val_tbl)
    story.append(Spacer(1, 10))

    # 5. Hazardous Materials & Environmental Safety
    story.append(Paragraph("3. Hazardous Materials & Safe Recycling Guidance", styles["SectionHeader"]))
    identified_hazards = hazards_result.get("identified_hazards", [])
    if identified_hazards:
        haz_rows = [
            [
                Paragraph("<b>Component</b>", styles["TableHead"]),
                Paragraph("<b>Potential Chemical Substances</b>", styles["TableHead"]),
                Paragraph("<b>Environmental & Health Risk</b>", styles["TableHead"]),
            ]
        ]
        for h in identified_hazards:
            haz_rows.append([
                Paragraph(f"{h['component']} ({h['count']}x)", styles["TableCellBold"]),
                Paragraph(", ".join(h.get("substances", [])), styles["TableCell"]),
                Paragraph(h.get("environmental_risk", ""), styles["TableCell"]),
            ])
        h_table = Table(haz_rows, colWidths=[100, 200, 240])
        h_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), LIGHT_GRAY_BG),
            ("GRID", (0, 0), (-1, -1), 0.5, BORDER_GRAY),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(h_table)
    else:
        story.append(Paragraph("<i>No specific chemical hazards identified for detected components.</i>", styles["Body"]))

    story.append(Spacer(1, 6))

    # AI Safety Officer Commentary (if available via Groq)
    ai_exp = hazards_result.get("ai_explanation")
    if ai_exp:
        story.append(Paragraph("<b>AI Safety Officer Commentary:</b>", styles["BodyBold"]))
        story.append(Paragraph(ai_exp.replace("\n", "<br/>"), styles["CalloutText"]))
        story.append(Spacer(1, 6))

    # Circular Economy & Recycling Instructions
    story.append(Paragraph("<b>Responsible Recycling Protocols:</b>", styles["BodyBold"]))
    story.append(Paragraph(
        "• Route non-functional boards exclusively to certified R2v3 or e-Stewards recycling facilities.<br/>"
        "• Do not incinerate circuit boards or execute home chemical acid baths.<br/>"
        "• Keep electronic scrap in enclosed containers away from children.",
        styles["Body"],
    ))
    story.append(Spacer(1, 8))

    # 6. Scientific Limitations & Legal Disclaimers
    story.append(Paragraph("4. Scientific Limitations & Disclaimers", styles["SectionHeader"]))
    full_disclaimer = (
        "<b>NOTICE:</b> This report is generated through computer vision detection and electronics surplus benchmarks. "
        "A photograph cannot confirm electrical functionality, internal silicon defects, or exact chemical metallurgy "
        "without laboratory bench testing and XRF spectrometry. Second-hand valuations are indicative estimates based on "
        "1 USD = ₹85.00, not guaranteed quotes. Always handle electronic scrap with appropriate protective equipment."
    )
    disc_tbl = Table([[Paragraph(full_disclaimer, styles["CalloutText"])]], colWidths=[540])
    disc_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GRAY_BG),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER_GRAY),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(disc_tbl)

    doc.build(story, canvasmaker=NumberedCanvas)
    return buf.getvalue()

