# Explainable Deep Learning for Multi-Class Bone Marrow Cell Type Classification

A comprehensive end-to-end Python project for classifying bone marrow cell images using Vision Transformers (ViT) with explainable AI (XAI) and RAG-based evidence explanations.

## 🎯 Project Overview

This system implements an explainable deep learning approach for multi-class bone marrow cell type classification, supporting the **UN Sustainable Development Goal 3: Good Health and Well-being** by providing AI-assisted diagnostic support for hematologic analysis.

### Key Features

- **Vision Transformer (ViT) Classifier**: State-of-the-art deep learning model for cell classification
- **7 Cell Types**: BLA (Blast), EOS (Eosinophil), LYT (Lymphocyte), MON (Monocyte), NGS (Neutrophil), NIF (Immature Neutrophil), PMO (Promyelocyte)
- **Explainable AI (XAI)**:
  - Grad-CAM heatmaps showing which image regions influence predictions
  - ViT attention maps visualizing model focus areas
- **Uncertainty Estimation**: Monte Carlo Dropout for quantifying prediction confidence
- **RAG Integration**: Retrieval-Augmented Generation for evidence-backed clinical explanations
- **Classical ML Baselines**: SVM, Random Forest, and XGBoost for comparison
- **Interactive Dashboard**: Streamlit web interface for pathologists

## 📚 Dataset

The model is trained on the [Bone Marrow Cell Classification Dataset](https://www.kaggle.com/datasets/donajui/bone-marrow-cell-classification) from Kaggle.

### Dataset Structure

After downloading and extracting the dataset, organize it as follows:

```
data/
└── raw/
    ├── BLA/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── EOS/
    ├── LYT/
    ├── MON/
    ├── NGS/
    ├── NIF/
    └── PMO/
```

## 🚀 Installation

### 1. Clone or Download the Project

```bash
cd bone-marrow-classification
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Dataset

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/donajui/bone-marrow-cell-classification)
2. Extract and organize images into class folders as shown above
3. Place the organized dataset in `data/raw/`

## 📖 Usage

### Training the Model

Train the Vision Transformer model:

```bash
python main.py train
```

With custom parameters:

```bash
python main.py train --epochs 100 --lr 0.0001
```

### Evaluation

Evaluate the trained model on the test set:

```bash
python main.py eval
```

This will:
- Compute classification metrics (accuracy, precision, recall, F1)
- Generate confusion matrix
- Create ROC curves
- Save all results to `results/`

### Classical ML Baselines

Train classical ML baselines (SVM, Random Forest, XGBoost):

```bash
python main.py baselines
```

### Generate Explanations

Generate sample Grad-CAM and attention visualizations:

```bash
python main.py explain
```

With custom number of samples:

```bash
python main.py explain --num_samples 10
```

### RAG Demo

Generate sample RAG explanations for each cell type:

```bash
python main.py rag_demo
```

### Launch Dashboard

Start the Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

The dashboard provides:
- **Predict Page**: Upload images, get predictions with confidence/uncertainty, view visual explanations (Grad-CAM, attention maps), and read RAG-generated clinical explanations
- **Model Performance Page**: View classification metrics, confusion matrix, and ROC curves
- **About Page**: Project information and documentation

## 📁 Project Structure

```
bone-marrow-classification/
├── data/
│   ├── raw/                          # Downloaded Kaggle dataset (user places here)
│   ├── processed/                    # (optional) processed images
│   └── splits/                       # train/val/test index info
│
├── models/
│   ├── vit_model_best.pth           # saved trained ViT model
│   ├── svm_model.pkl
│   ├── rf_model.pkl
│   └── xgb_model.pkl                # optional
│
├── rag_framework/
│   ├── rag_model.py                 # RAG pipeline wrapper
│   ├── knowledge_base/
│   │   └── hematology_knowledge.csv # knowledge base
│   └── embeddings/
│
├── src/
│   ├── config.py                    # paths, hyperparameters, constants
│   ├── data_preprocessing.py        # dataset + transforms + dataloaders
│   ├── model_training.py            # ViT training & validation loop
│   ├── test_inference.py            # test evaluation + probability outputs
│   ├── uncertainty_estimation.py    # Monte Carlo Dropout-based uncertainty
│   ├── explainability.py            # Grad-CAM + ViT attention extraction
│   ├── classical_ml_baseline.py     # SVM / RF / XGBoost on handcrafted features
│   ├── evaluation_metrics.py        # confusion matrix, ROC, classification report
│   ├── rag_integration.py           # glue between classifier prediction and RAG
│   └── utils.py                     # shared helpers (seeding, logging, etc.)
│
├── notebooks/
│   ├── 01_eda.ipynb                 # dataset exploration (optional)
│   ├── 02_train_vit.ipynb           # optional notebook wrapper for training
│   └── 03_explainability_demo.ipynb # optional visualization demo
│
├── dashboard/
│   ├── app.py                       # Streamlit app
│   └── assets/                      # optional images/css
│
├── results/
│   ├── metrics/
│   │   └── classification_metrics.json
│   └── visualizations/
│       ├── class_distribution.png
│       ├── confusion_matrix.png
│       ├── roc_curves.png
│       ├── sample_gradcam.png
│       └── sample_attention.png
│
├── requirements.txt
├── README.md
└── main.py                          # orchestration entrypoint
```

## 🔬 Technical Details

### Model Architecture

- **Backbone**: Vision Transformer (ViT-Base, patch size 16, 224×224 input)
- **Pretrained**: ImageNet pretrained weights
- **Classifier Head**: Custom head with LayerNorm, Dropout, and Linear layer
- **Output**: 7-class softmax probabilities

### Training Configuration

- **Batch Size**: 32
- **Learning Rate**: 1e-4 (AdamW optimizer)
- **Epochs**: 50 (with early stopping)
- **Data Augmentation**: Random horizontal flip, rotation, color jitter
- **Loss Function**: CrossEntropyLoss with label smoothing (0.1)

### Explainability Methods

1. **Grad-CAM**: Gradient-weighted Class Activation Mapping
   - Highlights image regions that influence the prediction
   - Uses gradients from the last transformer block

2. **ViT Attention Maps**: 
   - Visualizes attention weights from transformer blocks
   - Shows which image patches the model focuses on

### Uncertainty Estimation

- **Method**: Monte Carlo Dropout
- **Samples**: 30 stochastic forward passes
- **Metrics**: 
  - Epistemic uncertainty (model uncertainty)
  - Aleatoric uncertainty (data ambiguity)
  - Total uncertainty

### RAG Framework

- **Knowledge Base**: CSV-based hematology knowledge base
- **Retrieval**: Keyword-based similarity search
- **Generation**: Templated explanations with retrieved evidence
- **Sources**: Clinical references for each explanation

## 📊 Evaluation Metrics

The system computes:
- Overall accuracy
- Per-class precision, recall, F1-score
- Confusion matrix
- ROC curves and AUC scores
- Class distribution

## ⚠️ Medical Disclaimer

**This system is designed for research and educational purposes only.** It should not be used as the sole basis for clinical diagnosis. All predictions should be reviewed by qualified medical professionals.

## 📖 References

- **Vision Transformer**: Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", NeurIPS 2020
- **Grad-CAM**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017
- **RAG**: Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", NeurIPS 2020
- **Monte Carlo Dropout**: Gal & Ghahramani, "Dropout as a Bayesian Approximation", ICML 2016

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project is provided for educational and research purposes.

## 👥 Authors

Developed for medical imaging research and explainable AI applications in hematology.

---

**For questions or issues, please open an issue on the project repository.**







