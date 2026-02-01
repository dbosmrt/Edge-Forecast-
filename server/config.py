"""
Server configuration for fire risk prediction pipeline.
All thresholds, paths, and settings are centralized here.
"""
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.pkl"
FEATURE_ORDER_PATH = PROJECT_ROOT / "artifacts" / "feature_order.json"
DATA_PATH = PROJECT_ROOT / "dataset" / "fire_data" / "uttarakhand_fire_data.csv"
OUTPUT_PATH = PROJECT_ROOT / "server" / "output"

# These thresholds map probabilities to human-readable risk categories
RISK_THRESHOLDS = {
    "LOW": (0.0, 0.3),      # probability < 0.3
    "MEDIUM": (0.3, 0.6),   # 0.3 <= probability < 0.6
    "HIGH": (0.6, 1.0)      # probability >= 0.6
}

# Risk colors for visualization
RISK_COLORS = {
    "LOW": "#4CAF50",       # Green
    "MEDIUM": "#FF9800",    # Orange
    "HIGH": "#F44336"       # Red
}

MODEL_VERSION = "1.0.0"
PROBABILITY_THRESHOLD = 0.5  # Default classification threshold

CATEGORICAL_FEATURES = [
    "burn_count_5yr",
    "burned_last_5yr",
    "land_cover",
    "years_since_last_burn"
]

EXCLUDE_FEATURES = ["soil_moisture", "fire_occurred"]

# Columns to keep for location tracking (not used in model)
LOCATION_COLUMNS = ["latitude", "longitude"]

UPDATE_FREQUENCY_HOURS = 12  # How often to run the pipeline
