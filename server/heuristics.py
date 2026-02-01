"""
Heuristic Policy - Risk labeling (NOT machine learning).

This module converts probabilities to human-interpretable risk categories.

IMPORTANT:
- This is NOT learned - it's a policy decision
- This is configurable WITHOUT retraining the model
- Thresholds can be adjusted based on domain expertise
"""
from config import RISK_THRESHOLDS, RISK_COLORS


def assign_risk_label(probability: float) -> str:
    """
    Map a single probability to a risk category.
    
    Args:
        probability: Fire probability ∈ [0, 1]
        
    Returns:
        Risk label: "LOW", "MEDIUM", or "HIGH"
    """
    for label, (low, high) in RISK_THRESHOLDS.items():
        if low <= probability < high:
            return label
    return "HIGH"  # Default to HIGH if >= 1.0


def assign_risk_labels(probabilities: list) -> list:
    """
    Map multiple probabilities to risk categories.
    
    Args:
        probabilities: List of fire probabilities ∈ [0, 1]
        
    Returns:
        List of risk labels
    """
    return [assign_risk_label(p) for p in probabilities]


def get_risk_color(risk_label: str) -> str:
    """Get the color associated with a risk level."""
    return RISK_COLORS.get(risk_label, "#9E9E9E")  # Gray default


def get_risk_summary(risk_labels: list) -> dict:
    """
    Generate summary statistics for risk distribution.
    
    Args:
        risk_labels: List of risk labels
        
    Returns:
        Dict with counts and percentages per risk level
    """
    total = len(risk_labels)
    summary = {}
    
    for label in RISK_THRESHOLDS.keys():
        count = risk_labels.count(label)
        summary[label] = {
            "count": count,
            "percentage": round(count / total * 100, 2) if total > 0 else 0
        }
    
    return summary
