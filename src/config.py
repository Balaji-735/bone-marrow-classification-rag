"""
Configuration file for bone marrow cell classification project.
Contains paths, hyperparameters, and constants.
"""

import os
from pathlib import Path
import torch

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
VIT_MODEL_PATH = MODELS_DIR / "vit_model_best.pth"
SVM_MODEL_PATH = MODELS_DIR / "svm_model.pkl"
RF_MODEL_PATH = MODELS_DIR / "rf_model.pkl"
XGB_MODEL_PATH = MODELS_DIR / "xgb_model.pkl"

# Results paths
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
VISUALIZATIONS_DIR = RESULTS_DIR / "visualizations"

# RAG paths
RAG_DIR = PROJECT_ROOT / "rag_framework"
KNOWLEDGE_BASE_PATH = RAG_DIR / "knowledge_base" / "hematology_knowledge.csv"
EMBEDDINGS_DIR = RAG_DIR / "embeddings"

# Dataset configuration
CLASSES = ["BLA", "EOS", "LYT", "MON", "NGS", "NIF", "PMO"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(CLASSES)}

# Image configuration
IMAGE_SIZE = 384

# Training hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-5  # Lower learning rate for higher resolution (384×384) training
NUM_EPOCHS = 50  # More epochs for better convergence
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 10  # Patience for early stopping
EARLY_STOPPING_MIN_DELTA = 0.0005  # Finer improvement detection

# Data split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Model architecture
# Upgraded to vit_base for better capacity to handle 384×384 resolution
# vit_base has ~86M parameters vs vit_small's ~22M, providing more capacity for fine-grained features
VIT_MODEL_NAME = "vit_base_patch16_224"
PRETRAINED = True

# Uncertainty estimation
MC_DROPOUT_SAMPLES = 30
DROPOUT_RATE = 0.1

# Explainability
GRADCAM_LAYER = "blocks.11.norm1"  # Last transformer block normalization (vit_base has 12 blocks)

# RAG configuration
RAG_TOP_K = 5  # Increased for better retrieval
RAG_EMBEDDING_DIM = 384
RAG_USE_CHROMA = False  # Temporarily disabled due to ChromaDB corruption - will auto-fallback to CSV
RAG_CHROMA_PATH = None  # None = auto-detect from rag-tutorial-v2-main
RAG_USE_LLM = True  # Enable LLM for better summarization
RAG_LLM_MODEL = "mistral"  # LLM model name
CHATBOT_LLM_MODEL = "gemma:2b"  # Chatbot model (smaller, uses less memory)

# Random seed
RANDOM_SEED = 42

# Device: use GPU if available, otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SPLITS_DIR, 
                 MODELS_DIR, RESULTS_DIR, METRICS_DIR, VISUALIZATIONS_DIR,
                 RAG_DIR, KNOWLEDGE_BASE_PATH.parent, EMBEDDINGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

