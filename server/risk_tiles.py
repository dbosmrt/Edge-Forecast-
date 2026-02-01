"""
Risk Tiles - Frontend contract output.

This module generates the data format that the client/frontend consumes.
The frontend NEVER touches ML - it only reads these risk tiles.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

from config import MODEL_VERSION, OUTPUT_PATH, RISK_THRESHOLDS
from heuristics import get_risk_summary, get_risk_color


def generate_risk_tiles(
    locations: pd.DataFrame,
    probabilities: list,
    risk_labels: list,
    timestamp: datetime = None
) -> dict:
    """
    Generate risk tiles data structure for frontend.
    
    Args:
        locations: DataFrame with 'latitude', 'longitude' columns
        probabilities: List of fire probabilities
        risk_labels: List of risk labels
        timestamp: When this prediction was made (defaults to now)
        
    Returns:
        Dict in the format expected by frontend
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    
    # Build grid cells
    grid_cells = []
    for i in range(len(probabilities)):
        cell = {
            "lat": round(float(locations.iloc[i]["latitude"]), 6),
            "lon": round(float(locations.iloc[i]["longitude"]), 6),
            "probability": round(float(probabilities[i]), 4),
            "risk": risk_labels[i],
            "color": get_risk_color(risk_labels[i])
        }
        grid_cells.append(cell)
    
    # Build metadata
    risk_summary = get_risk_summary(risk_labels)
    
    metadata = {
        "model_version": MODEL_VERSION,
        "timestamp": timestamp.isoformat(),
        "total_cells": len(grid_cells),
        "risk_distribution": risk_summary,
        "thresholds": {k: {"min": v[0], "max": v[1]} for k, v in RISK_THRESHOLDS.items()}
    }
    
    return {
        "metadata": metadata,
        "grid_cells": grid_cells
    }


def save_risk_tiles(tiles: dict, filename: str = "risk_tiles.json") -> Path:
    """
    Save risk tiles to JSON file.
    
    Args:
        tiles: Risk tiles data structure
        filename: Output filename
        
    Returns:
        Path to saved file
    """
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_PATH / filename
    
    with open(output_file, 'w') as f:
        json.dump(tiles, f, indent=2)
    
    return output_file


def load_risk_tiles(filename: str = "risk_tiles.json") -> dict:
    """Load risk tiles from JSON file."""
    input_file = OUTPUT_PATH / filename
    
    if not input_file.exists():
        raise FileNotFoundError(f"Risk tiles not found at {input_file}")
    
    with open(input_file, 'r') as f:
        return json.load(f)
