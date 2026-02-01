"""
Preprocess fire data for model training.
Matches exactly the preprocessing from hack_earth_poc.py for best results.
"""
import pandas as pd

# Categorical features (will be one-hot encoded)
# From notebook: categorical_features = ['fire_occurred', 'burn_count_5yr', 'burned_last_5yr', 'land_cover', 'years_since_last_burn']
CATEGORICAL_FEATURES = [
    "burn_count_5yr",
    "burned_last_5yr",
    "land_cover",
    "years_since_last_burn"
]

# Features to exclude (matching notebook)
EXCLUDE_FEATURES = ["soil_moisture"]

TARGET = "fire_occurred"


def preprocess(df: pd.DataFrame, training: bool = True):
    """
    Preprocess the fire data for model training/inference.
    
    Matches exactly the notebook's preprocessing for GBC + RandomOverSampler:
    1. Drop 'fire_occurred' and 'soil_moisture'
    2. One-hot encode categorical features with drop_first=True
    
    Args:
        df: Raw DataFrame with fire data
        training: If True, returns X, y, feature_cols. If False, returns X, feature_cols.
    
    Returns:
        X: Feature matrix (encoded)
        y: Target variable (only if training=True)
        feature_cols: List of feature column names
    """
    df = df.copy()
    
    # 1. Separate target and features (matching notebook line 361-362)
    y = df[TARGET] if training else None
    X = df.drop(columns=[TARGET] + EXCLUDE_FEATURES, errors='ignore')
    
    # 2. Identify categorical features present in the data
    features_to_encode = [col for col in CATEGORICAL_FEATURES if col in X.columns]
    
    # 3. Apply one-hot encoding (matching notebook line 368)
    X_encoded = pd.get_dummies(X, columns=features_to_encode, drop_first=True)
    
    # Get feature column names
    feature_cols = list(X_encoded.columns)
    
    if training:
        return X_encoded, y, feature_cols
    
    return X_encoded, feature_cols
