from http.server import BaseHTTPRequestHandler
import json
import urllib.parse

EXPECTED_CLASSES = [
    "Resistor", "Capacitor", "Diode", "LED", "IC",
    "Connector", "Switch/Button", "Header", "Crystal", "Debug/TP"
]

CLASS_COLORS = {
    "Resistor": "#FF5722",
    "Capacitor": "#2196F3",
    "Diode": "#9C27B0",
    "LED": "#FFEB3B",
    "IC": "#E91E63",
    "Connector": "#00BCD4",
    "Switch/Button": "#4CAF50",
    "Header": "#FF9800",
    "Crystal": "#3F51B5",
    "Debug/TP": "#9E9E9E"
}

COMPONENT_PRICING_INR = {
    "IC": {"low": 17.0, "typical": 72.25, "high": 255.0},
    "Connector": {"low": 12.75, "typical": 42.50, "high": 127.50},
    "Header": {"low": 8.50, "typical": 25.50, "high": 85.0},
    "Switch/Button": {"low": 8.50, "typical": 21.25, "high": 68.0},
    "Crystal": {"low": 12.75, "typical": 34.0, "high": 102.0},
    "Diode": {"low": 1.70, "typical": 4.25, "high": 12.75},
    "Resistor": {"low": 0.42, "typical": 1.70, "high": 4.25},
    "Capacitor": {"low": 0.85, "typical": 3.40, "high": 17.0},
    "LED": {"low": 1.70, "typical": 4.25, "high": 17.0},
    "Debug/TP": {"low": 0.0, "typical": 0.0, "high": 0.0}
}

COMPONENT_HAZARDS = {
    "Capacitor": {
        "substances": ["Liquid/Gel Organic Electrolytes", "Ammonium Borates", "Butyrolactone Solvents"],
        "risk": "Corrosive liquid leakage if ruptured; eye and skin irritant; toxic vapors if incinerated.",
        "handling": "Discharge before handling; do not crush or puncture cans; store in dry, cool conditions."
    },
    "IC": {
        "substances": ["Lead (solder balls)", "Antimony Trioxide (flame retardant)", "Silicon", "Brominated Compounds"],
        "risk": "Heavy metal neurotoxicity from lead; leaching into groundwater from landfills.",
        "handling": "Use fume extraction during desoldering; collect pins/balls for metals recovery."
    },
    "Diode": {
        "substances": ["Gallium Arsenide (in RF/LED types)", "Silicon", "Lead solder"],
        "risk": "Arsenic toxicity if crushed or combusted into fine dust.",
        "handling": "Store whole; avoid mechanical grinding or acid treatment outside controlled facilities."
    },
    "Connector": {
        "substances": ["Gold plating", "Nickel underplating", "Phosphor Bronze", "Beryllium Copper"],
        "risk": "Beryllium exposure causes chronic berylliosis if aerosolized during grinding.",
        "handling": "Strip contacts mechanically; do not burn plastic housing; send to certified e-waste recyclers."
    }
}

class handler(BaseHTTPRequestHandler):
    def _send_json(self, data, status=200):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path in ("/api", "/api/health"):
            self._send_json({
                "status": "healthy",
                "service": "Smart E-Waste PCB Component Classifier API",
                "classes_count": len(EXPECTED_CLASSES),
                "model": "YOLOv8s PCB Checkpoint (960px)",
                "currency": "INR (Rs)",
                "deployment": "Vercel Serverless"
            })
        elif path == "/api/classes":
            self._send_json({
                "classes": EXPECTED_CLASSES,
                "colors": CLASS_COLORS
            })
        elif path == "/api/pricing":
            self._send_json({
                "currency": "INR",
                "symbol": "Rs",
                "pricing": COMPONENT_PRICING_INR
            })
        elif path == "/api/hazards":
            self._send_json({
                "hazards": COMPONENT_HAZARDS
            })
        elif path == "/api/samples":
            self._send_json({
                "samples": [
                    {"id": "sample-1", "name": "Power Circuit Board (IMG_20260529_163319.jpg)", "expected_components": 13},
                    {"id": "sample-2", "name": "Motherboard Section (IMG_20260529_163712.jpg)", "expected_components": 18},
                    {"id": "sample-3", "name": "Logic & Control Unit (IMG_20260529_163718.jpg)", "expected_components": 15}
                ]
            })
        else:
            self._send_json({"error": "Endpoint not found", "path": path}, status=404)

