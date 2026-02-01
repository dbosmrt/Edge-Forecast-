"""
FastAPI server for Edge-Forecast fire risk prediction.

Endpoints:
- GET /api/risk-tiles - Latest risk predictions
- GET /api/health - Health check
- GET /api/metadata - Model metadata
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import OUTPUT_PATH, MODEL_VERSION

app = FastAPI(
    title="Edge-Forecast API",
    description="Fire Risk Prediction API for Uttarakhand",
    version=MODEL_VERSION
)

# CORS - allow frontend to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RISK_TILES_PATH = OUTPUT_PATH / "risk_tiles.json"


def load_risk_tiles() -> dict:
    """Load latest risk tiles from JSON file."""
    if not RISK_TILES_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Risk tiles not found. Run the pipeline first."
        )
    
    with open(RISK_TILES_PATH, 'r') as f:
        return json.load(f)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Edge-Forecast API",
        "version": MODEL_VERSION,
        "docs": "/docs"
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/api/metadata")
async def get_metadata():
    """Get model and data metadata."""
    try:
        tiles = load_risk_tiles()
        return tiles.get("metadata", {})
    except HTTPException:
        return {
            "model_version": MODEL_VERSION,
            "status": "no_data",
            "message": "Run pipeline to generate predictions"
        }


@app.get("/api/risk-tiles")
async def get_risk_tiles(
    limit: Optional[int] = None,
    risk_level: Optional[str] = None
):
    """
    Get latest risk tiles.
    
    Args:
        limit: Max number of cells to return (for performance)
        risk_level: Filter by risk level (LOW, MEDIUM, HIGH)
    """
    tiles = load_risk_tiles()
    
    grid_cells = tiles.get("grid_cells", [])
    
    # Filter by risk level if specified
    if risk_level:
        risk_level = risk_level.upper()
        grid_cells = [c for c in grid_cells if c.get("risk") == risk_level]
    
    # Smart limit: prioritize HIGH and MEDIUM risk, then fill with LOW
    if limit and limit > 0 and not risk_level:
        high = [c for c in grid_cells if c.get("risk") == "HIGH"]
        medium = [c for c in grid_cells if c.get("risk") == "MEDIUM"]
        low = [c for c in grid_cells if c.get("risk") == "LOW"]
        
        # Always include all HIGH and MEDIUM, fill rest with LOW
        priority_cells = high + medium
        remaining = limit - len(priority_cells)
        
        if remaining > 0:
            grid_cells = priority_cells + low[:remaining]
        else:
            grid_cells = priority_cells[:limit]
    elif limit and limit > 0:
        grid_cells = grid_cells[:limit]
    
    return {
        "metadata": tiles.get("metadata", {}),
        "grid_cells": grid_cells,
        "count": len(grid_cells)
    }


@app.get("/api/risk-tiles/summary")
async def get_risk_summary():
    """Get summary of risk distribution without full grid data."""
    tiles = load_risk_tiles()
    metadata = tiles.get("metadata", {})
    
    return {
        "timestamp": metadata.get("timestamp"),
        "model_version": metadata.get("model_version"),
        "total_cells": metadata.get("total_cells"),
        "risk_distribution": metadata.get("risk_distribution", {})
    }


@app.post("/api/upload-dataset")
async def upload_dataset(file: bytes = None):
    """
    Upload custom dataset and run prediction pipeline.
    
    Accepts CSV file with environmental data.
    """
    from fastapi import UploadFile, File
    # This is a placeholder - full implementation would:
    # 1. Save uploaded CSV to temp location
    # 2. Validate columns match expected format
    # 3. Run feature_builder + inference + heuristics
    # 4. Save new risk_tiles.json
    # 5. Return success message
    
    return {
        "status": "success",
        "message": "Dataset feature coming soon! Currently using GEE data."
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

