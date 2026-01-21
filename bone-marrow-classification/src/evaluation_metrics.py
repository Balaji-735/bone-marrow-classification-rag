"""
Evaluation metrics module for classification performance analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    roc_auc_score, precision_recall_curve, average_precision_score
)
from pathlib import Path
import json

from src.config import CLASSES, NUM_CLASSES, VISUALIZATIONS_DIR, METRICS_DIR


def compute_classification_metrics(y_true, y_pred, y_probs=None):
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Predicted probabilities (optional)
        
    Returns:
        Dictionary with metrics
    """
    # Classification report
    report = classification_report(
        y_true, y_pred, 
        target_names=CLASSES, 
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Overall accuracy
    accuracy = np.mean(y_true == y_pred)
    
    metrics = {
        'accuracy': float(accuracy),
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    
    # ROC AUC if probabilities provided
    if y_probs is not None:
        try:
            # One-vs-rest ROC AUC
            roc_auc_scores = {}
            for i, class_name in enumerate(CLASSES):
                y_true_binary = (y_true == i).astype(int)
                y_prob_class = y_probs[:, i]
                
                if len(np.unique(y_true_binary)) > 1:  # Check if both classes present
                    roc_auc = roc_auc_score(y_true_binary, y_prob_class)
                    roc_auc_scores[class_name] = float(roc_auc)
            
            metrics['roc_auc_scores'] = roc_auc_scores
            metrics['mean_roc_auc'] = float(np.mean(list(roc_auc_scores.values())))
        except Exception as e:
            print(f"Warning: Could not compute ROC AUC: {e}")
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, save_path=None, normalize=False):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save plot
        normalize: Whether to normalize the matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap='Blues',
        xticklabels=CLASSES, yticklabels=CLASSES,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    plt.close()


def plot_roc_curves(y_true, y_probs, save_path=None):
    """
    Plot ROC curves for all classes.
    
    Args:
        y_true: True labels
        y_probs: Predicted probabilities (n_samples, n_classes)
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve for each class
    for i, class_name in enumerate(CLASSES):
        y_true_binary = (y_true == i).astype(int)
        y_prob_class = y_probs[:, i]
        
        if len(np.unique(y_true_binary)) > 1:
            fpr, tpr, _ = roc_curve(y_true_binary, y_prob_class)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})', linewidth=2)
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves for Multi-Class Classification', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved ROC curves to {save_path}")
    
    plt.close()


def plot_class_distribution(labels, save_path=None):
    """
    Plot class distribution.
    
    Args:
        labels: List or array of labels
        save_path: Path to save plot
    """
    unique, counts = np.unique(labels, return_counts=True)
    class_counts = {CLASSES[i]: count for i, count in zip(unique, counts)}
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_counts.keys(), class_counts.values(), color='steelblue', alpha=0.7)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Class Distribution', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved class distribution to {save_path}")
    
    plt.close()


def evaluate_and_visualize(y_true, y_pred, y_probs=None, save_dir=None):
    """
    Comprehensive evaluation and visualization.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Predicted probabilities (optional)
        save_dir: Directory to save results
        
    Returns:
        Dictionary with all metrics
    """
    if save_dir is None:
        save_dir = METRICS_DIR
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute metrics
    print("Computing classification metrics...")
    metrics = compute_classification_metrics(y_true, y_pred, y_probs)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=CLASSES))
    
    if 'mean_roc_auc' in metrics:
        print(f"\nMean ROC AUC: {metrics['mean_roc_auc']:.4f}")
    
    # Save metrics to JSON
    metrics_file = save_dir / "classification_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {metrics_file}")
    
    # Generate visualizations
    vis_dir = VISUALIZATIONS_DIR
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating visualizations...")
    
    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, vis_dir / "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, vis_dir / "confusion_matrix_normalized.png", normalize=True)
    
    # ROC curves
    if y_probs is not None:
        plot_roc_curves(y_true, y_probs, vis_dir / "roc_curves.png")
    
    # Class distribution
    plot_class_distribution(y_true, vis_dir / "class_distribution.png")
    
    print("\nEvaluation completed!")
    
    return metrics


if __name__ == "__main__":
    # Example usage
    from src.test_inference import load_trained_model, evaluate_test_set
    from src.data_preprocessing import get_dataloaders
    
    print("Loading model and test data...")
    model = load_trained_model()
    _, _, test_loader = get_dataloaders()
    
    print("Evaluating...")
    results = evaluate_test_set(model, test_loader)
    
    y_true = np.array(results['true_labels'])
    y_pred = np.array(results['predictions'])
    y_probs = np.array(results['probabilities'])
    
    evaluate_and_visualize(y_true, y_pred, y_probs)







