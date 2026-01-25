# Quick Start Guide

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended, but CPU will work)
- Dataset from [Kaggle](https://www.kaggle.com/datasets/donajui/bone-marrow-cell-classification)

## Setup (5 minutes)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download and organize dataset:**
   - Download from Kaggle
   - Extract to `data/raw/` with class folders:
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

## Usage

### 1. Train the Model
```bash
python main.py train
```
This will:
- Preprocess and split the data (70/15/15)
- Train ViT for up to 50 epochs with early stopping
- Save best model to `models/vit_model_best.pth`

### 2. Evaluate
```bash
python main.py eval
```
Generates:
- Classification metrics
- Confusion matrix
- ROC curves
- All saved to `results/`

### 3. Train Baselines (Optional)
```bash
python main.py baselines
```
Trains SVM, Random Forest, and XGBoost for comparison.

### 4. Launch Dashboard
```bash
streamlit run dashboard/app.py
```
Open browser to `http://localhost:8501`

## Expected Results

- **Training time**: ~2-4 hours on GPU (depending on dataset size)
- **Test accuracy**: Typically 85-95% (depends on dataset quality)
- **Model size**: ~330MB (ViT-Base)

## Troubleshooting

**"Model not found" error:**
- Train the model first: `python main.py train`

**"Data directory empty" error:**
- Ensure dataset is in `data/raw/` with class folders

**CUDA out of memory:**
- Reduce batch size in `src/config.py` (BATCH_SIZE)

**Import errors:**
- Ensure you're in the project root directory
- Check that all dependencies are installed: `pip install -r requirements.txt`

## Next Steps

- Explore the dashboard for interactive predictions
- Generate explanations: `python main.py explain`
- Try RAG demo: `python main.py rag_demo`
- Check `results/` for metrics and visualizations







