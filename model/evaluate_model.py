"""
Evaluate trained fire prediction model with comprehensive metrics and visualizations.

Generates:
- Classification report (precision, recall, F1)
- Confusion matrix plot
- ROC curve plot
- Precision-Recall curve plot
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay
)

ARTIFACT_PATH = Path("../artifacts/")


def plot_confusion_matrix(y_test, y_pred, save_path):
    """Generate and save confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Fire', 'Fire'])
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title('Confusion Matrix\n(GradientBoosting + RandomOverSampler)', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix saved to {save_path}")
    return cm


def plot_roc_curve(y_test, y_prob, save_path):
    """Generate and save ROC curve plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    ax.fill_between(fpr, tpr, alpha=0.3, color='darkorange')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve\n(GradientBoosting + RandomOverSampler)', fontsize=12)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ROC curve saved to {save_path}")
    return roc_auc


def plot_precision_recall_curve(y_test, y_prob, save_path):
    """Generate and save Precision-Recall curve plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    
    ax.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {pr_auc:.4f})')
    ax.fill_between(recall, precision, alpha=0.3, color='green')
    
    # Add baseline (proportion of positive class)
    baseline = y_test.mean()
    ax.axhline(y=baseline, color='red', linestyle='--', lw=2, label=f'Baseline (prevalence = {baseline:.4f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Precision-Recall Curve\n(GradientBoosting + RandomOverSampler)', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Precision-Recall curve saved to {save_path}")
    return pr_auc


def main():
    print("Loading test data and model...")
    
    # Load test data
    test_data = joblib.load(ARTIFACT_PATH / "test_data.pkl")
    y_test = np.array(test_data['y_test'])
    y_prob = np.array(test_data['y_prob'])
    y_pred = np.array(test_data['y_pred'])
    
    print(f"  Test samples: {len(y_test)}")
    print(f"  Actual fires: {y_test.sum()}")
    print(f"  Predicted fires: {y_pred.sum()}")
    
    # Calculate metrics
    print("EVALUATION RESULTS")
    
    # Classification report
    print("\nClassification Report:")

    report = classification_report(y_test, y_pred)

    
    # Detailed metrics
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    
    print("\n Key Metrics for Fire Detection (Class 1):")

    
    # Extract metrics for class 1 (fire)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"  Precision: {precision:.4f} (When we predict fire, {precision*100:.1f}% are correct)")
    print(f"  Recall:    {recall:.4f} (We catch {recall*100:.1f}% of all fires)")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  PR-AUC:    {pr_auc:.4f}")
    
    print(f"\n Confusion Matrix:")
    print(f"  True Negatives:  {tn:5d} (Correctly predicted no fire)")
    print(f"  False Positives: {fp:5d} (False alarms)")
    print(f"  False Negatives: {fn:5d} (Missed fires)")
    print(f"  True Positives:  {tp:5d} (Correctly predicted fire)")
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_confusion_matrix(y_test, y_pred, ARTIFACT_PATH / "confusion_matrix.png")
    plot_roc_curve(y_test, y_prob, ARTIFACT_PATH / "roc_curve.png")
    plot_precision_recall_curve(y_test, y_prob, ARTIFACT_PATH / "pr_curve.png")
    
    # Save detailed metrics
    detailed_metrics = {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "precision_fire": float(precision),
        "recall_fire": float(recall),
        "f1_fire": float(f1),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "total_test_samples": int(len(y_test)),
        "actual_fires": int(y_test.sum()),
        "predicted_fires": int(y_pred.sum())
    }
    
    with open(ARTIFACT_PATH / "evaluation_metrics.json", "w") as f:
        json.dump(detailed_metrics, f, indent=2)
    print(f"  Detailed metrics saved to {ARTIFACT_PATH / 'evaluation_metrics.json'}")
    

    print("Evaluation complete!")

    
    return detailed_metrics


if __name__ == "__main__":
    main()
