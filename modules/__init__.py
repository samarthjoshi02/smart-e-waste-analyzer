"""Intelligent E-Waste PCB Analyzer Modules Package."""

from .detector import PCBDetector, EXPECTED_CLASSES, MAX_FILE_SIZE_BYTES, get_available_samples
from .analyzer import build_component_inventory, calculate_salvage_valuation, COMPONENT_PRICING, PRICING_METADATA
from .hazard_advisor import analyze_pcb_hazards, COMPONENT_HAZARDS, BOARD_LEVEL_HAZARDS
from .history import init_history_db, save_analysis_record, get_history_records, clear_all_history

__all__ = [
    "PCBDetector",
    "EXPECTED_CLASSES",
    "MAX_FILE_SIZE_BYTES",
    "get_available_samples",
    "build_component_inventory",
    "calculate_salvage_valuation",
    "COMPONENT_PRICING",
    "PRICING_METADATA",
    "analyze_pcb_hazards",
    "COMPONENT_HAZARDS",
    "BOARD_LEVEL_HAZARDS",
    "init_history_db",
    "save_analysis_record",
    "get_history_records",
    "clear_all_history",
]
