"""
Main orchestration script for bone marrow classification project.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.config import RAW_DATA_DIR
from src.utils import set_seed, setup_logger
from src.data_preprocessing import get_dataloaders, create_data_splits, load_dataset_from_folders
from src.model_training import train_vit
from src.test_inference import load_trained_model, evaluate_test_set
from src.evaluation_metrics import evaluate_and_visualize
from src.classical_ml_baseline import train_classical_ml_baselines
from src.explainability import save_sample_explanations
from src.rag_integration import generate_explanation
from src.uncertainty_estimation import estimate_uncertainty_for_single
import numpy as np


def train(args):
    """Train ViT model."""
    print("=" * 60)
    print("Training Vision Transformer Model")
    print("=" * 60)
    
    # Set seed
    set_seed()
    
    # Check if data exists
    if not RAW_DATA_DIR.exists() or len(list(RAW_DATA_DIR.glob("*"))) == 0:
        print(f"Error: Data directory {RAW_DATA_DIR} is empty or does not exist.")
        print("Please download the dataset from:")
        print("https://www.kaggle.com/datasets/donajui/bone-marrow-cell-classification")
        print("\nExtract the dataset so that images are organized in folders:")
        print("  data/raw/BLA/")
        print("  data/raw/EOS/")
        print("  data/raw/LYT/")
        print("  ... etc")
        return
    
    # Get dataloaders
    print("\nLoading and preprocessing data...")
    train_loader, val_loader, test_loader = get_dataloaders()
    
    # Train model
    print("\nStarting training...")
    model, history = train_vit(
        train_loader,
        val_loader,
        num_epochs=args.epochs if args.epochs else None,
        learning_rate=args.lr if args.lr else None,
        resume_from=args.resume_from if getattr(args, "resume_from", None) else None,
    )
    
    print("\nTraining completed!")
    print(f"Best model saved to: models/vit_model_best.pth")


def eval(args):
    """Evaluate model on test set."""
    print("=" * 60)
    print("Evaluating Model on Test Set")
    print("=" * 60)
    
    # Load model
    print("\nLoading trained model...")
    try:
        model = load_trained_model()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first using: python main.py train")
        return
    
    # Get test dataloader
    print("\nLoading test data...")
    _, _, test_loader = get_dataloaders()
    
    # Evaluate
    print("\nRunning evaluation...")
    results = evaluate_test_set(model, test_loader)
    
    # Compute metrics and visualizations
    y_true = np.array(results['true_labels'])
    y_pred = np.array(results['predictions'])
    y_probs = np.array(results['probabilities'])
    
    print("\nGenerating metrics and visualizations...")
    evaluate_and_visualize(y_true, y_pred, y_probs)
    
    print("\nEvaluation completed!")


def baselines(args):
    """Train classical ML baselines."""
    print("=" * 60)
    print("Training Classical ML Baselines")
    print("=" * 60)
    
    train_classical_ml_baselines()


def explain(args):
    """Generate sample explanations."""
    print("=" * 60)
    print("Generating Sample Explanations")
    print("=" * 60)
    
    # Load model
    print("\nLoading trained model...")
    try:
        model = load_trained_model()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first using: python main.py train")
        return
    
    # Get test dataloader
    print("\nLoading test data...")
    _, _, test_loader = get_dataloaders()
    
    # Generate explanations
    num_samples = args.num_samples if args.num_samples else 5
    print(f"\nGenerating {num_samples} sample explanations...")
    save_sample_explanations(model, test_loader, num_samples=num_samples)
    
    print("\nSample explanations generated!")


def rag_demo(args):
    """Generate sample RAG explanations for each class."""
    print("=" * 60)
    print("RAG Explanation Demo")
    print("=" * 60)
    
    from src.config import CLASSES
    
    print("\nGenerating RAG explanations for each cell type...")
    print("-" * 60)
    
    for class_name in CLASSES:
        print(f"\n{class_name}:")
        explanation = generate_explanation(
            predicted_class_name=class_name,
            confidence=0.95,  # Example high confidence
            uncertainty=0.05  # Example low uncertainty
        )
        
        print(explanation['explanation'])
        print("\n" + "-" * 60)
    
    print("\nRAG demo completed!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Bone Marrow Cell Classification - Main Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train                    # Train ViT model
  python main.py train --epochs 100      # Train with custom epochs
  python main.py eval                     # Evaluate on test set
  python main.py baselines                # Train classical ML baselines
  python main.py explain                  # Generate sample explanations
  python main.py explain --num_samples 10 # Generate 10 sample explanations
  python main.py rag_demo                 # Demo RAG explanations
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train ViT model')
    train_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    train_parser.add_argument('--lr', type=float, help='Learning rate')
    train_parser.add_argument(
        '--resume_from',
        type=str,
        help='Path to checkpoint to resume training from (e.g., models/vit_model_light.pth)',
    )
    
    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate model on test set')
    
    # Baselines command
    baselines_parser = subparsers.add_parser('baselines', help='Train classical ML baselines')
    
    # Explain command
    explain_parser = subparsers.add_parser('explain', help='Generate sample explanations')
    explain_parser.add_argument('--num_samples', type=int, help='Number of samples to explain')
    
    # RAG demo command
    rag_parser = subparsers.add_parser('rag_demo', help='Demo RAG explanations')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train(args)
    elif args.command == 'eval':
        eval(args)
    elif args.command == 'baselines':
        baselines(args)
    elif args.command == 'explain':
        explain(args)
    elif args.command == 'rag_demo':
        rag_demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


