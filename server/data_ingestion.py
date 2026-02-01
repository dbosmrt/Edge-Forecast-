"""
Data Ingestion - Fetch latest environmental data.

For PoC: Reads from existing CSV
For Production: Would fetch from GEE or other data source
"""
import pandas as pd
from pathlib import Path

from config import DATA_PATH


def fetch_latest_data(source: str = "csv", data_path: Path = None) -> pd.DataFrame:
    """
    Fetch latest environmental data for all grid cells.
    
    Args:
        source: Data source type ("csv" for PoC, "gee" for production)
        data_path: Optional path to CSV file
        
    Returns:
        DataFrame with lat, lon, NDVI, LST, precip, wind, etc.
    """
    if source == "csv":
        path = data_path or DATA_PATH
        if not path.exists():
            raise FileNotFoundError(f"Data file not found at {path}")
        return pd.read_csv(path)
    
    elif source == "gee":
        # TODO: Implement GEE data fetching
        # This would use the fire_extractor.py logic
        raise NotImplementedError("GEE data source not yet implemented")
    
    else:
        raise ValueError(f"Unknown data source: {source}")


def get_sample_data(n_samples: int = 100) -> pd.DataFrame:
    """
    Get a sample of data for testing.
    
    Args:
        n_samples: Number of samples to return
        
    Returns:
        DataFrame with sampled data
    """
    df = fetch_latest_data()
    return df.sample(n=min(n_samples, len(df)), random_state=42)
