# Bone Marrow Cell Classification with RAG Framework

A deep learning project for automated classification of bone marrow cells using Vision Transformer (ViT) with Retrieval-Augmented Generation (RAG) for explainable AI.

## 🎯 Overview

This project implements a state-of-the-art bone marrow cell classification system that:
- Classifies cells into 7 categories: **BLA**, **EOS**, **LYT**, **MON**, **NGS**, **NIF**, **PMO**
- Uses Vision Transformer (ViT) for high-accuracy classification
- Provides explainable predictions using RAG (Retrieval-Augmented Generation)
- Includes an interactive Streamlit dashboard for real-time predictions
- Supports uncertainty estimation and classical ML baselines

## ✨ Features

- **Deep Learning Model**: Vision Transformer (ViT) for image classification
- **RAG Integration**: ChromaDB-based retrieval system for generating explanations from research papers
- **Explainability**: Grad-CAM visualizations and uncertainty estimation
- **Interactive Dashboard**: Streamlit-based web interface
- **Classical ML Baselines**: SVM, Random Forest, and XGBoost for comparison
- **Comprehensive Evaluation**: Metrics, confusion matrices, and ROC curves

## 📋 Requirements

- Python 3.10+
- CUDA-capable GPU (recommended, but CPU will work)
- 8GB+ RAM
- 10GB+ free disk space

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Balaji-735/bone-marrow-classification-rag.git
cd bone-marrow-classification-rag

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup

Download the bone marrow cell classification dataset from [Kaggle](https://www.kaggle.com/datasets/donajui/bone-marrow-cell-classification) and organize it as follows:

```
data/raw/
  ├── BLA/
  ├── EOS/
  ├── LYT/
  ├── MON/
  ├── NGS/
  ├── NIF/
  └── PMO/
```

### 3. Train the Model

```bash
python main.py train
```

This will:
- Preprocess and split the data (70/15/15 train/val/test)
- Train ViT for up to 50 epochs with early stopping
- Save the best model to `models/vit_model_best.pth`

### 4. Evaluate

```bash
python main.py eval
```

Generates classification metrics, confusion matrices, and ROC curves in the `results/` directory.

### 5. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Open your browser to `http://localhost:8501` for interactive predictions.

## 📁 Project Structure

```
bone-marrow-classification/
├── data/
│   ├── raw/              # Raw dataset images
│   ├── processed/        # Processed images
│   └── splits/           # Train/val/test splits
├── models/               # Trained model checkpoints
├── results/              # Metrics and visualizations
│   ├── metrics/
│   └── visualizations/
├── src/                  # Source code
│   ├── config.py         # Configuration
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── test_inference.py
│   ├── evaluation_metrics.py
│   ├── explainability.py
│   ├── rag_integration.py
│   ├── uncertainty_estimation.py
│   └── classical_ml_baseline.py
├── rag_framework/        # RAG system
│   ├── rag_model.py
│   ├── rag_model_chroma.py
│   └── knowledge_base/
├── dashboard/            # Streamlit dashboard
│   └── app.py
├── main.py               # Main orchestration script
├── requirements.txt      # Python dependencies
└── README.md
```

## 🔧 Usage

### Training

```bash
# Train ViT model
python main.py train

# Train classical ML baselines
python main.py baselines
```

### Evaluation

```bash
# Evaluate on test set
python main.py eval

# Generate explanations
python main.py explain

# RAG demo
python main.py rag_demo
```

### RAG System

The RAG system uses ChromaDB to retrieve relevant information from research papers:

```python
from src.rag_integration import generate_explanation

result = generate_explanation(
    predicted_class_name='BLA',
    confidence=0.95,
    uncertainty=0.05
)
```

To populate the ChromaDB database:

```bash
python populate_database.py
```

## 📊 Model Performance

- **Test Accuracy**: Typically 85-95% (depends on dataset quality)
- **Training Time**: ~2-4 hours on GPU
- **Model Size**: ~330MB (ViT-Base)

## 🛠️ Configuration

Key settings in `src/config.py`:

```python
# Training
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20

# RAG
RAG_USE_CHROMA = True
RAG_TOP_K = 3
RAG_USE_LLM = False  # Set True for LLM-based generation
```

## 📚 Dependencies

Key libraries:
- `torch` - PyTorch for deep learning
- `transformers` - Hugging Face transformers
- `langchain` - RAG framework
- `chromadb` - Vector database
- `streamlit` - Dashboard
- `scikit-learn` - Classical ML models
- `pandas`, `numpy` - Data processing

See `requirements.txt` for the complete list.

## 🧪 Testing

```bash
# Test RAG integration
python test_rag.py

# Test ChromaDB RAG
python test_chroma_rag.py

# Test dashboard RAG
python test_dashboard_rag.py
```

## 📖 Documentation

- [Quick Start Guide](QUICKSTART.md)
- [RAG Integration Summary](RAG_INTEGRATION_SUMMARY.md)
- [Integration Complete](INTEGRATION_COMPLETE.md)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Dataset: [Kaggle Bone Marrow Cell Classification](https://www.kaggle.com/datasets/donajui/bone-marrow-cell-classification)
- Vision Transformer: [Hugging Face Transformers](https://huggingface.co/transformers/)
- RAG Framework: [LangChain](https://www.langchain.com/)

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This project is for research and educational purposes. Always consult medical professionals for clinical decisions.
