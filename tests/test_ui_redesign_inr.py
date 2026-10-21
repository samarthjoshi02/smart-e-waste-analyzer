"""Test Suite for White Theme, INR Currency, and Clean UX Verification.

Tests:
1. Streamlit Light Theme config (.streamlit/config.toml).
2. INR currency valuation bounds and fixed 85.0 conversion rate.
3. Itemized valuation table format with INR columns.
4. Component inventory plain-language text and INR indicative values.
5. User-friendly hazard advisory status strings.
6. Verification that developer clutter is absent from app.py.
7. Verification that the 6 tabs and 5 plain-English hazard sections are present in app.py.
8. SQLite history INR valuation retrieval.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.analyzer import (
    build_component_inventory,
    calculate_salvage_valuation,
    PRICING_METADATA,
)
from modules.hazard_advisor import analyze_pcb_hazards
from modules.history import (
    init_history_db,
    save_analysis_record,
    get_history_records,
    clear_all_history,
)


def test_theme_config():
    print("\n--- Test 1: Light Theme Configuration Audit ---")
    config_file = PROJECT_ROOT / ".streamlit" / "config.toml"
    assert config_file.exists(), ".streamlit/config.toml does not exist"
    content = config_file.read_text(encoding="utf-8")
    assert 'base = "light"' in content or 'base="light"' in content
    assert "#FFFFFF" in content
    assert "#0D9488" in content
    print("[PASS] .streamlit/config.toml configured with clean light theme and teal primary color.")


def test_inr_valuation_logic():
    print("\n--- Test 2: INR Currency Valuation Logic ---")
    sample_counts = {"IC": 4, "Connector": 2, "Capacitor": 10, "Resistor": 20}
    val = calculate_salvage_valuation(sample_counts, exchange_rate_usd_to_inr=85.0)

    # Check that INR values exist and match 85.0 conversion
    assert "total_low_inr" in val
    assert "total_typical_inr" in val
    assert "total_high_inr" in val

    assert abs(val["total_typical_inr"] - (val["total_typical_usd"] * 85.0)) < 0.05
    assert abs(val["total_low_inr"] - (val["total_low_usd"] * 85.0)) < 0.05
    assert abs(val["total_high_inr"] - (val["total_high_usd"] * 85.0)) < 0.05

    print(f"[PASS] USD Valuation: Typical=${val['total_typical_usd']:.2f}")
    print(f"[PASS] INR Valuation: Typical=₹{val['total_typical_inr']:,.2f} (Low=₹{val['total_low_inr']:,.2f}, High=₹{val['total_high_inr']:,.2f})")

    # Check itemized breakdown INR columns
    item = val["itemized_breakdown"][0]
    assert "Unit Price Range (INR)" in item
    assert "Typical Est. Value (INR)" in item
    assert "Low Est. (INR)" in item
    assert "High Est. (INR)" in item
    assert "₹" in item["Typical Est. Value (INR)"]
    print(f"[PASS] Itemized breakdown correctly formats prices in ₹ (e.g. {item['Typical Est. Value (INR)']}).")


def test_inventory_inr_and_plain_language():
    print("\n--- Test 3: Inventory Table Plain Language & INR Values ---")
    fake_dets = [
        {"Component": "IC", "Confidence (%)": 92.0},
        {"Component": "Connector", "Confidence (%)": 88.0},
    ]
    fake_counts = {"IC": 1, "Connector": 1}
    inv = build_component_inventory(fake_dets, fake_counts)

    assert len(inv) == 2
    for item in inv:
        assert "Indicative Value (INR)" in item
        assert "₹" in item["Indicative Value (INR)"]
        assert "Potential Reuse" in item
        assert "Recommended Next Step" in item
        assert "Potentially reusable" in item["Potential Reuse"]

    print(f"[PASS] Inventory items include INR values: {inv[0]['Component Name']} -> {inv[0]['Indicative Value (INR)']}")
    print(f"[PASS] Inventory items include user-friendly guidance: {inv[0]['Potential Reuse']}")


def test_hazard_friendly_status():
    print("\n--- Test 4: Hazard Advisory Friendly Status ---")
    h = analyze_pcb_hazards({"IC": 2}, groq_api_key="")
    assert "user_friendly_status" in h
    assert h["user_friendly_status"] == "Using built-in recycling guidance."
    print(f"[PASS] Fallback/Local mode status string: '{h['user_friendly_status']}'")


def test_app_ui_cleanliness_and_sections():
    print("\n--- Test 5: UI Cleanliness & Section Presence Audit ---")
    app_text = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")

    # Developer clutter checks
    assert "Groq Model" not in app_text, "Developer model selectbox should be removed from main UI"
    assert "llama-3.3-70b-versatile" not in app_text, "Hardcoded model dropdowns should be removed from UI"
    assert "Add GROQ_API_KEY in .env" not in app_text, "Technical .env instructions should be removed from main UI"
    assert "AI component detection is ready" in app_text, "Friendly sidebar status must be present"

    # 6 Tabs check
    assert "🏠 Home" in app_text
    assert "🎯 Detection Results" in app_text
    assert "📋 Component Inventory" in app_text
    assert "💰 Reuse Value" in app_text
    assert "♻️ Safe Recycling Guide" in app_text
    assert "📜 Analysis History" in app_text
    print("[PASS] All 6 dedicated tabs confirmed present in app.py.")

    # 5 Plain-English Hazard Sections check
    assert "A. What might be present?" in app_text
    assert "B. Why should I care?" in app_text
    assert "C. What should I do?" in app_text
    assert "D. What can the AI actually know?" in app_text
    assert "E. Where can I recycle?" in app_text
    print("[PASS] All 5 plain-English hazardous materials & recycling sections confirmed present in app.py.")

    # Currency format check
    assert "₹" in app_text
    assert "1 USD = ₹85.00" in app_text
    print("[PASS] INR currency symbol (₹) and fixed 1 USD = ₹85.00 benchmark confirmed.")


def test_sqlite_inr_history():
    print("\n--- Test 6: SQLite History INR Retrieval ---")
    records = get_history_records(limit=5)
    if records:
        r = records[0]
        assert "valuation_typical_inr" in r
        assert r["valuation_typical_inr"] >= 0.0
        print(f"[PASS] History record contains INR valuation: ₹{r['valuation_typical_inr']:,.2f}")
    else:
        print("[INFO] No history records in DB yet (clean state).")


if __name__ == "__main__":
    test_theme_config()
    test_inr_valuation_logic()
    test_inventory_inr_and_plain_language()
    test_hazard_friendly_status()
    test_app_ui_cleanliness_and_sections()
    test_sqlite_inr_history()
    print("\n=======================================================")
    print("ALL UI REDESIGN & INR VERIFICATION TESTS PASSED!")
    print("=======================================================")

