"""
Train Gradient Boosting model for fire prediction with proper train/test split
and RandomOverSampler to handle class imbalance.

Based on experiments from hack_earth_poc.py - Best model: GBC + RandomOverSampler
Expected metrics: ROC-AUC ~0.98, Precision ~0.43, Recall ~0.78, F1 ~0.55
"""
import json
import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from imblearn.over_sampling import RandomOverSampler

from preprocess import preprocess

# Paths (relative to model/ directory)
DATA_PATH = Path("../dataset/fire_data/uttarakhand_fire_data.csv")
ARTIFACT_PATH = Path("../artifacts/")


def main():
    # Ensure artifacts directory exists
    ARTIFACT_PATH.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data
    print("Loading data")
    df = pd.read_csv(DATA_PATH)
    print(f"  Total samples: {len(df)}")
    print(f"  Fire events: {df['fire_occurred'].sum()} ({df['fire_occurred'].mean()*100:.2f}%)")
    
    # 2. Preprocess
    print("\n Preprocessing")
    X, y, feature_cols = preprocess(df, training=True)
    print(f"  Features: {len(feature_cols)}")
    
    # 3. Train/test split (stratified to maintain class balance)
    print("\n Splitting data")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        stratify=y, 
        random_state=42
    )
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    print(f"  Train fire rate: {y_train.mean()*100:.2f}%")
    print(f"  Test fire rate: {y_test.mean()*100:.2f}%")
    
    # 4. Apply RandomOverSampler to training data only
    print("\nApplying RandomOverSampler to training data...")
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
    print(f"  Resampled train: {len(X_train_resampled)} samples")
    print(f"  Class distribution: {y_train_resampled.value_counts().to_dict()}")
    
    # 5. Train model with hyperparameters from notebook experiments
    print("\nTraining GradientBoostingClassifier...")
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train_resampled, y_train_resampled)
    print("  Model trained successfully!")
    
    # 6. Evaluate on TEST set only (no data leakage)
    print("\nEvaluating on test set...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    

    print("TEST SET RESULTS")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 7. Save artifacts
    print("\nSaving artifacts")
    
    # Model
    joblib.dump(model, ARTIFACT_PATH / "model.pkl")
    print(f"  Model saved to {ARTIFACT_PATH / 'model.pkl'}")
    
    # Feature order
    with open(ARTIFACT_PATH / "feature_order.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"  Feature order saved to {ARTIFACT_PATH / 'feature_order.json'}")
    
    # Test data for evaluation script
    test_data = {
        'X_test': X_test.values.tolist(),
        'y_test': y_test.values.tolist(),
        'y_prob': y_prob.tolist(),
        'y_pred': y_pred.tolist(),
        'feature_cols': feature_cols
    }
    joblib.dump(test_data, ARTIFACT_PATH / "test_data.pkl")
    print(f"  Test data saved to {ARTIFACT_PATH / 'test_data.pkl'}")
    
    # Metrics
    metrics = {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "resampled_train_samples": len(X_train_resampled),
        "n_features": len(feature_cols)
    }
    with open(ARTIFACT_PATH / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {ARTIFACT_PATH / 'metrics.json'}")
    
    print("\nTraining complete!")
    return model, metrics


if __name__ == "__main__":
    main()
