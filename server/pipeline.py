"""
Pipeline Orchestrator - Runs the full inference pipeline.

[Data Ingestion] → [Feature Builder] → [ML Inference] → [Heuristic Policy] → [Risk Tiles]
"""
import logging
from datetime import datetime, timezone
from pathlib import Path

from data_ingestion import fetch_latest_data
from feature_builder import build_features, validate_features
from inference import predict_probabilities
from heuristics import assign_risk_labels, get_risk_summary
from risk_tiles import generate_risk_tiles, save_risk_tiles
from config import OUTPUT_PATH

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pipeline(data_path: Path = None, output_filename: str = "risk_tiles.json") -> dict:
    """
    Execute the full inference pipeline.
    
    Pipeline stages:
    1. Data Ingestion - Fetch latest environmental data
    2. Feature Builder - Apply preprocessing (identical to training)
    3. ML Inference - Get fire probabilities
    4. Heuristic Policy - Assign risk labels
    5. Risk Tiles - Generate frontend output
    
    Args:
        data_path: Optional path to input data (defaults to config)
        output_filename: Name of output JSON file
        
    Returns:
        Dict with pipeline results and metadata
    """
    timestamp = datetime.now(timezone.utc)
    logger.info("=" * 60)
    logger.info("FIRE RISK PREDICTION PIPELINE")
    logger.info("=" * 60)
    

    logger.info("\n[1/5] Data Ingestion...")
    raw_data = fetch_latest_data(data_path=data_path)
    logger.info(f"      Loaded {len(raw_data)} grid cells")
    

    logger.info("\n[2/5] Feature Building...")
    
    # Validate features first
    validation = validate_features(raw_data)
    if not validation["valid"]:
        logger.warning(f"      Missing columns: {validation['missing_columns']}")
    
    X, locations = build_features(raw_data)
    logger.info(f"      Features: {X.shape[1]}")
    logger.info(f"      Samples: {X.shape[0]}")
    

    logger.info("\n[3/5] ML Inference...")
    probabilities = predict_probabilities(X)
    logger.info(f"      Mean probability: {probabilities.mean():.4f}")
    logger.info(f"      Max probability: {probabilities.max():.4f}")
    

    logger.info("\n[4/5] Heuristic Policy...")
    risk_labels = assign_risk_labels(probabilities.tolist())
    risk_summary = get_risk_summary(risk_labels)
    
    for label, stats in risk_summary.items():
        logger.info(f"      {label}: {stats['count']} cells ({stats['percentage']}%)")
    

    logger.info("\n[5/5] Generating Risk Tiles...")
    tiles = generate_risk_tiles(locations, probabilities.tolist(), risk_labels, timestamp)
    output_path = save_risk_tiles(tiles, output_filename)
    logger.info(f"      Saved to: {output_path}")
    

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Timestamp: {timestamp.isoformat()}")
    logger.info(f"Total cells: {len(risk_labels)}")
    logger.info(f"High risk zones: {risk_summary['HIGH']['count']}")
    logger.info(f"Output: {output_path}")
    
    return {
        "success": True,
        "timestamp": timestamp.isoformat(),
        "total_cells": len(risk_labels),
        "risk_summary": risk_summary,
        "output_path": str(output_path)
    }


if __name__ == "__main__":
    result = run_pipeline()
    print(f"\nPipeline result: {result}")
