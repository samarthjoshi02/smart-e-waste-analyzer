"""Hazardous Materials & Responsible E-Waste Recycling Advisory Module.

Problem Statement 2 (PS 2): Intelligent E-Waste Circuit Component Classifier & Material Analyzer.
Provides:
  1. A curated, evidence-based local knowledge base of electronic components and PCB chemical hazards.
  2. Optional Groq API integration (reading GROQ_API_KEY) to generate plain-language recycling advice.
  3. Resilient fallback to local knowledge base when the API key is missing or network fails.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load environment variables if python-dotenv is available
try:
    from dotenv import load_dotenv
    # Look for .env in project root or parent
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()
except ImportError:
    pass

# Try importing Groq client
try:
    from groq import Groq
except ImportError:
    Groq = None


# -----------------------------------------------------------------------------
# Curated Local Hazardous Material Knowledge Base
# -----------------------------------------------------------------------------
COMPONENT_HAZARDS: Dict[str, Dict[str, Any]] = {
    "Capacitor": {
        "substances": ["Liquid/Gel Organic Electrolytes", "Ammonium Borates", "Butyrolactone Solvents"],
        "prevalence": "Present in wet aluminum electrolytic capacitors; minimal in solid ceramic/film types.",
        "environmental_risk": "Corrosive liquid leakage if ruptured; eye and skin irritant; toxic vapors if incinerated.",
        "safe_handling": "Do not puncture, crush, or expose to high heat. Inspect for casing bulging or venting. Wear nitrile gloves if handling leaking fluid.",
    },
    "IC": {
        "substances": ["Brominated Epoxy Molding Compound (BFR)", "Antimony Trioxide Synergist", "Trace Heavy Metals (Lead in legacy packages, Nickel)"],
        "prevalence": "Universal in epoxy-encapsulated integrated circuits; modern ICs conform to RoHS/halogen-free standards.",
        "environmental_risk": "Persistent organic pollutants if incinerated at low temperatures; inhalation hazard if pulverized into fine dust.",
        "safe_handling": "Do not grind, shred without industrial HEPA extraction, or burn. Desolder under dedicated fume extraction.",
    },
    "Connector": {
        "substances": ["Nickel Barrier Plating", "Tin/Gold Flash", "Brominated Polymer Housings (PBT/Nylon)"],
        "prevalence": "Common across I/O ports, headers, and terminals.",
        "environmental_risk": "Nickel is a known contact allergen (dermatitis); brominated plastics emit hazardous halides upon combustion.",
        "safe_handling": "Clean mechanically; avoid open flame melting or burning of plastic housings.",
    },
    "Header": {
        "substances": ["Nickel Plating", "Tin/Copper Alloys", "Thermoplastic Carrier"],
        "prevalence": "Standard pin headers.",
        "environmental_risk": "Low acute toxicity in solid form; hazardous emissions if plastics are incinerated.",
        "safe_handling": "Desolder cleanly or clip for recycling.",
    },
    "Switch/Button": {
        "substances": ["Silver-Cadmium Oxide (legacy/specialty contacts)", "Beryllium-Copper Springs (trace)", "Brominated Polymers"],
        "prevalence": "Cadmium found primarily in older industrial switches; modern consumer switches are cadmium-free.",
        "environmental_risk": "Cadmium is a highly toxic heavy metal causing kidney damage and environmental persistence.",
        "safe_handling": "Avoid filing contacts or applying open flames; route through certified e-waste recyclers.",
    },
    "Crystal": {
        "substances": ["Quartz (Silicon Dioxide)", "Iron-Nickel-Cobalt (Kovar Metal Can)", "Lead Glass/Solder Seal (trace)"],
        "prevalence": "Hermetically sealed oscillators.",
        "environmental_risk": "Very low in solid state; potential lead in vintage hermetic seal glasses.",
        "safe_handling": "Keep intact; do not crush cans.",
    },
    "Diode": {
        "substances": ["Silicon / Gallium Arsenide (GaAs)", "Epoxy Resin / Glass Encapsulation", "Copper/Tin Leads"],
        "prevalence": "Standard discrete diodes.",
        "environmental_risk": "Trace arsenic bound securely within crystalline matrix; hazardous only if chemically dissolved or ground to respirable dust.",
        "safe_handling": "Safe in solid ambient form. Do not dissolve in home chemical baths.",
    },
    "LED": {
        "substances": ["Gallium Nitride (GaN) / Gallium Arsenide (GaAs)", "Indium / Phosphide Traces", "Optical Epoxy"],
        "prevalence": "Semiconductor indicator diodes.",
        "environmental_risk": "Low risk under standard handling; bound micro-quantities of heavy elements.",
        "safe_handling": "Standard electronics recycling.",
    },
    "Resistor": {
        "substances": ["Ruthenium Oxide Glaze", "Lead Borosilicate Glass Frit (RoHS-exempt in thick film)", "Alumina Ceramic Core"],
        "prevalence": "Thick-film chip resistors.",
        "environmental_risk": "Lead is bound inside vitrified glass glaze; minimal leaching under neutral pH, but leaches in acidic landfill leachate.",
        "safe_handling": "Recycle as whole board scrap.",
    },
    "Debug/TP": {
        "substances": ["Copper Trace", "Gold Flash / ENIG (Electroless Nickel Immersion Gold)", "Solder Mask"],
        "prevalence": "Board-level surface feature.",
        "environmental_risk": "Low risk; high economic value for precious metal smelting.",
        "safe_handling": "Board-level refining via pyrometallurgical or hydrometallurgical certified facilities.",
    },
}

BOARD_LEVEL_HAZARDS = {
    "solder_joints": {
        "title": "Solder Joint Metallurgy (Lead vs Lead-Free)",
        "details": (
            "Legacy and vintage electronics manufactured before July 2006 (or specialized military/medical boards) "
            "frequently use Tin-Lead solder (Sn63/Pb37). Lead (Pb) is a potent neurotoxin that causes environmental "
            "groundwater contamination if landfilled. Modern consumer electronics use RoHS-compliant lead-free solders "
            "(typically SAC305: Tin, Silver, Copper). A visual 2D image cannot confirm solder chemical composition without XRF spectrometry."
        ),
    },
    "pcb_laminate": {
        "title": "Circuit Board Laminate (FR-4 & Flame Retardants)",
        "details": (
            "Standard fiberglass circuit boards (FR-4) are bonded with epoxy resins containing Brominated Flame Retardants "
            "(typically TBBPA - Tetrabromobisphenol A). If incinerated in open air or substandard furnaces, brominated resins "
            "produce polybrominated dibenzodioxins and furans (PBDD/F). Never burn circuit boards."
        ),
    },
}

RECYCLING_FACILITIES_GUIDANCE = (
    "1. Authorized E-Waste Recyclers: Deliver discarded boards to certified R2v3 (Responsible Recycling) or "
    "e-Stewards certified recyclers equipped with industrial baghouse filtration and automated hydrometallurgical refining.\n"
    "2. No Backyard Chemical Leaching: Never attempt to extract gold or copper using nitric acid or aqua regia at home. "
    "These processes generate toxic nitrogen dioxide fumes and hazardous acid wastewater.\n"
    "3. No Open-Air Burning: Thermal desoldering must use closed ventilation. Never burn boards in open fires."
)


# -----------------------------------------------------------------------------
# Analysis Engine
# -----------------------------------------------------------------------------
def analyze_pcb_hazards(
    counts_by_class: Dict[str, int],
    groq_api_key: Optional[str] = None,
    groq_model: str = "llama-3.3-70b-versatile",
) -> Dict[str, Any]:
    """Generates a comprehensive hazardous materials and recycling analysis.

    Uses a curated local knowledge base as the ground truth. If a Groq API key is present,
    generates a dynamic, conversational explanation strictly adhering to the local facts.
    If the key is missing or the call fails, cleanly falls back to the curated knowledge base.

    Args:
        counts_by_class: Dictionary of detected component counts.
        groq_api_key: Optional Groq API key (defaults to GROQ_API_KEY env var).
        groq_model: Configurable Groq model name.

    Returns:
        Dictionary containing structured hazard assessments, warnings, and recycling guidelines.
    """
    detected_classes = [cls_name for cls_name, count in counts_by_class.items() if count > 0]

    # 1. Compile verified local component hazards
    identified_hazards: List[Dict[str, Any]] = []
    all_substances_set = set()

    for cls_name in detected_classes:
        h_info = COMPONENT_HAZARDS.get(cls_name)
        if h_info:
            identified_hazards.append({
                "component": cls_name,
                "count": counts_by_class[cls_name],
                "substances": h_info["substances"],
                "prevalence": h_info["prevalence"],
                "environmental_risk": h_info["environmental_risk"],
                "safe_handling": h_info["safe_handling"],
            })
            for sub in h_info["substances"]:
                all_substances_set.add(sub)

    # 2. Determine Groq API status
    api_key = groq_api_key or os.environ.get("GROQ_API_KEY", "").strip()
    ai_explanation: Optional[str] = None
    ai_status = "Local Knowledge Base (Offline Mode)"

    if api_key:
        if Groq is None:
            ai_status = "Local Knowledge Base (groq package not installed)"
        else:
            try:
                client = Groq(api_key=api_key)

                # Format inventory summary for prompt (only send minimum metadata)
                inv_summary_str = ", ".join([f"{cls}: {counts_by_class[cls]}" for cls in detected_classes]) or "No components localized"
                hazards_summary_str = "\n".join([
                    f"- {h['component']} ({h['count']} units): Contains {', '.join(h['substances'])}. Risk: {h['environmental_risk']}"
                    for h in identified_hazards
                ])

                prompt_content = f"""You are a certified Environmental E-Waste Safety Officer and Circular Electronics Advisor.
Explain the environmental and health hazards for a discarded printed circuit board with the following localized visual components:

DETECTED INVENTORY:
{inv_summary_str}

CURATED LOCAL KNOWLEDGE BASE FACTS:
{hazards_summary_str}

BOARD-LEVEL SUBSTANCES:
- Solder: Possible lead (Sn-Pb) in legacy/non-RoHS boards, or lead-free (SAC305) in modern boards. Cannot be determined from visual image.
- Substrate: FR-4 fiberglass containing Brominated Flame Retardants (TBBPA).
- Exact material percentages: Cannot be determined from an image without laboratory XRF assay.

STRICT INSTRUCTIONS:
1. Explain hazards in clear, professional, accessible English for an ordinary user or technician.
2. Distinguish established possibilities from certainties (e.g. 'may contain lead in older boards' rather than 'definitely contains lead').
3. Strictly forbid home chemical acid leaching or open-air burning.
4. Recommend formal certified recycling facilities (R2v3 / e-Stewards).
5. Never invent laboratory percentages, fake part numbers, or fake chemical assays. If percentages are mentioned, state 'Exact material percentage cannot be determined from the image'.
6. Keep the response concise, structured into: (A) Key Material & Health Risks, (B) Component-Specific Cautions, (C) Safe Handling & Certified Recycling Recommendations."""

                completion = client.chat.completions.create(
                    model=groq_model,
                    messages=[
                        {"role": "system", "content": "You are a professional circular electronics and hazardous e-waste safety advisor. Adhere strictly to verified facts and forbid dangerous DIY chemical processing."},
                        {"role": "user", "content": prompt_content},
                    ],
                    temperature=0.2,
                    max_tokens=800,
                )

                if completion.choices and completion.choices[0].message.content:
                    ai_explanation = completion.choices[0].message.content
                    ai_status = f"Active (Groq AI via {groq_model})"

            except Exception as e:
                # Fallback to local mode cleanly without breaking application
                ai_status = f"Fallback Mode (Groq API call unavailable: {type(e).__name__})"
                ai_explanation = None
    else:
        ai_status = "Local Knowledge Base (Set GROQ_API_KEY in .env for dynamic AI explanations)"

    user_friendly_status = (
        "AI-assisted explanation available."
        if ai_explanation
        else "Using built-in recycling guidance."
    )

    return {
        "status": ai_status,
        "user_friendly_status": user_friendly_status,
        "ai_explanation": ai_explanation,
        "detected_classes_count": len(detected_classes),
        "identified_hazards": identified_hazards,
        "board_level_hazards": BOARD_LEVEL_HAZARDS,
        "recycling_guidance": RECYCLING_FACILITIES_GUIDANCE,
        "material_percentage_note": "Exact material percentage cannot be determined from the image without laboratory XRF spectrometry or destructive chemical digestion.",
        "disclaimer": (
            "NOTICE: This assessment is based on visual component classification and standard electronics manufacturing baselines. "
            "A photograph cannot prove exact metallurgical composition, presence of RoHS exemptions, or specific chemical formulations. "
            "Always consult certified environmental recyclers and wear appropriate PPE when handling dismantled circuit boards."
        ),
    }

