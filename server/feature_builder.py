"""
Feature Builder - Deterministic preprocessing for inference.

This module applies the EXACT same transformations used during training.
No heuristics here - purely deterministic data engineering.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path

from config import (
    CATEGORICAL_FEATURES,
    EXCLUDE_FEATURES,
    LOCATION_COLUMNS,
    FEATURE_ORDER_PATH
)


def load_feature_order() -> list:
    """Load the feature order from training."""
    with open(FEATURE_ORDER_PATH, 'r') as f:
        return json.load(f)


def build_features(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Transform raw data to model-ready features.
    
    MUST be identical to preprocessing used during training.
    
    Args:
        df: Raw DataFrame with environmental data
        
    Returns:
        X: Feature matrix ready for model (np.ndarray)
        locations: DataFrame with lat/lon for each row
    """
    df = df.copy()
    
    # Save location data for output
    locations = df[LOCATION_COLUMNS].copy() if all(col in df.columns for col in LOCATION_COLUMNS) else None
    
    # 1. Drop excluded features
    cols_to_drop = [col for col in EXCLUDE_FEATURES if col in df.columns]
    X = df.drop(columns=cols_to_drop, errors='ignore')
    
    # 2. Identify categorical features present
    features_to_encode = [col for col in CATEGORICAL_FEATURES if col in X.columns]
    
    # 3. One-hot encode (drop_first=True to match training)
    X_encoded = pd.get_dummies(X, columns=features_to_encode, drop_first=True)
    
    # 4. Align columns with training feature order
    feature_order = load_feature_order()
    
    # Add any missing columns with 0s
    for col in feature_order:
        if col not in X_encoded.columns:
            X_encoded[col] = 0
    
    # Select only the columns used in training, in the correct order
    X_aligned = X_encoded[feature_order]
    
    return X_aligned.values, locations


def validate_features(df: pd.DataFrame) -> dict:
    """
    Validate that input data has required features.
    
    Returns:
        Dict with validation status and any missing columns
    """
    feature_order = load_feature_order()
    
    # Get columns that will be created after encoding
    df_copy = df.copy()
    cols_to_drop = [col for col in EXCLUDE_FEATURES if col in df_copy.columns]
    df_copy = df_copy.drop(columns=cols_to_drop, errors='ignore')
    
    features_to_encode = [col for col in CATEGORICAL_FEATURES if col in df_copy.columns]
    df_encoded = pd.get_dummies(df_copy, columns=features_to_encode, drop_first=True)
    
    available_cols = set(df_encoded.columns)
    required_cols = set(feature_order)
    
    missing = required_cols - available_cols
    extra = available_cols - required_cols
    
    return {
        "valid": len(missing) == 0,
        "missing_columns": list(missing),
        "extra_columns": list(extra),
        "required_count": len(required_cols),
        "available_count": len(available_cols)
    }
