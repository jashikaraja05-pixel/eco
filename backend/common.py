"""Shared constants for the EcoLens AI classifier service.

Single source of truth for categories, bin mapping and the uncertainty rule.
backend/main.py, waste_model.py and tests import from here; the TypeScript
side mirrors these values in src/lib/scan-schema.ts (normalizePrediction).
"""
from __future__ import annotations

MAX_BYTES = 6 * 1024 * 1024
ALLOWED = {"image/jpeg", "image/png", "image/webp"}

CATEGORIES = ["organic", "recyclable", "hazardous", "ewaste", "residual", "uncertain"]

# Below this confidence a prediction is reported as "uncertain" instead of a bin.
# Must match UNCERTAIN_THRESHOLD in src/lib/scan-schema.ts.
UNCERTAIN_THRESHOLD = 55

BINS = {
    "organic": "Green Bin (Organic / Wet Waste)",
    "recyclable": "Blue Bin (Dry Recyclable Waste)",
    "hazardous": "Red Bin (Hazardous Waste Depot)",
    "ewaste": "Red Bin (Special E-Waste Bin)",
    "residual": "Black Bin (Residual Non-Recyclable Waste)",
    "uncertain": "No bin — rescan or check local rules",
}

# Static disposal guidance per app category. The local CNN predicts a coarse
# category only — it does not generate text — so these are curated municipal
# rules, not model output.
DISPOSAL_STEPS = {
    "organic": [
        "Keep food and garden waste separate from dry and hazardous waste.",
        "Place it in the green wet-waste bin without plastic wrapping.",
        "Compost at home if possible, or hand it to the municipal wet-waste collector.",
    ],
    "recyclable": [
        "Empty and rinse the item so no food residue remains.",
        "Keep it dry and separate from wet waste.",
        "Place it in the blue dry-recyclable bin; flatten cardboard if possible.",
    ],
    "hazardous": [
        "Do not crush, burn or mix with regular household waste.",
        "Keep the item in its original container if possible.",
        "Hand it over at the municipal hazardous-waste depot or a collection drive.",
    ],
    "ewaste": [
        "Do not bin it with household waste.",
        "Erase personal data if the item stores any.",
        "Drop it at an authorised e-waste collection point or manufacturer take-back.",
    ],
    "residual": [
        "Ensure the item cannot be recycled or composted.",
        "Bag it to prevent littering.",
        "Place it in the black residual (non-recyclable) bin.",
    ],
    "uncertain": [
        "Retake the photo in good light with only one item in frame.",
        "If still unsure, keep it aside and check your municipality's segregation guide.",
    ],
}


# Shown with every result of that category. Never advise opening devices/batteries.
SAFETY_WARNINGS = {
    "hazardous": [
        "Do not open, puncture, crush or burn batteries, aerosol cans or chemical containers.",
        "Keep away from children and food; tape battery terminals before drop-off.",
    ],
    "ewaste": [
        "Do not dismantle the device or remove its battery yourself.",
        "Swollen or damaged batteries can catch fire — store them away from heat.",
    ],
    "organic": ["Remove stickers, rubber bands and plastic before composting."],
    "recyclable": ["Broken glass: wrap it in paper and label it before binning."],
    "residual": [],
    "uncertain": ["Bin colours differ between cities — confirm with your municipal corporation's guide."],
}


def sniff(data: bytes) -> str | None:
    """Magic-byte content check: returns the real image MIME type or None."""
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None
