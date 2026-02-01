"""
ML Inference - Model prediction returning probabilities only.

This module ONLY outputs probabilities.
It NEVER outputs risk labels - that's the heuristics layer's job.
"""
import joblib
import numpy as np
from pathlib import Path

from config import MODEL_PATH, MODEL_VERSION


class FireProbabilityModel:
    """Wrapper for the trained fire prediction model."""
    
    def __init__(self):
        self.model = None
        self.version = MODEL_VERSION
        
    def load(self):
        """Load the trained model from disk."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        self.model = joblib.load(MODEL_PATH)
        return self
    
    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        """
        Run model inference, return probabilities only.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Array of probabilities ∈ [0, 1] for fire occurrence
        """
        if self.model is None:
            self.load()
        
        # Return probability of class 1 (fire)
        return self.model.predict_proba(X)[:, 1]


# Singleton instance for efficiency
_model_instance = None


def get_model() -> FireProbabilityModel:
    """Get or create the model instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = FireProbabilityModel().load()
    return _model_instance


def predict_probabilities(X: np.ndarray) -> np.ndarray:
    """
    Convenience function for prediction.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        
    Returns:
        Array of fire probabilities ∈ [0, 1]
    """
    model = get_model()
    return model.predict_probabilities(X)
