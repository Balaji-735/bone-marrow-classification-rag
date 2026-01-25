"""
Classical ML baselines using handcrafted features.
Implements SVM, Random Forest, and optionally XGBoost.
"""

import numpy as np
from pathlib import Path
import pickle
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import cv2
from tqdm import tqdm
from PIL import Image

from src.config import (
    RAW_DATA_DIR, MODELS_DIR, SVM_MODEL_PATH, RF_MODEL_PATH, XGB_MODEL_PATH,
    CLASSES, NUM_CLASSES
)
from src.data_preprocessing import load_data_splits
from src.utils import set_seed


def extract_handcrafted_features(image_path):
    """
    Extract handcrafted features from an image.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Feature vector as numpy array
    """
    try:
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            img = np.array(Image.open(image_path).convert('RGB'))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Resize to standard size
        img = cv2.resize(img, (224, 224))
        
        features = []
        
        # 1. Color histogram features (RGB)
        for i in range(3):
            hist = cv2.calcHist([img], [i], None, [32], [0, 256])
            features.extend(hist.flatten())
        
        # 2. Texture features (LBP-like)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Local Binary Pattern (simplified)
        lbp = np.zeros_like(gray)
        for i in range(1, gray.shape[0] - 1):
            for j in range(1, gray.shape[1] - 1):
                center = gray[i, j]
                code = 0
                code |= (gray[i-1, j-1] > center) << 7
                code |= (gray[i-1, j] > center) << 6
                code |= (gray[i-1, j+1] > center) << 5
                code |= (gray[i, j+1] > center) << 4
                code |= (gray[i+1, j+1] > center) << 3
                code |= (gray[i+1, j] > center) << 2
                code |= (gray[i+1, j-1] > center) << 1
                code |= (gray[i, j-1] > center) << 0
                lbp[i, j] = code
        
        lbp_hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
        features.extend(lbp_hist.flatten())
        
        # 3. Statistical features
        features.append(np.mean(gray))
        features.append(np.std(gray))
        features.append(np.median(gray))
        features.append(np.min(gray))
        features.append(np.max(gray))
        
        # 4. Gradient features (Sobel)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        features.append(np.mean(gradient_magnitude))
        features.append(np.std(gradient_magnitude))
        
        # 5. Shape features (contours)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            features.append(area)
            features.append(perimeter)
            if perimeter > 0:
                features.append(4 * np.pi * area / (perimeter ** 2))  # Circularity
            else:
                features.append(0)
        else:
            features.extend([0, 0, 0])
        
        return np.array(features, dtype=np.float32)
    
    except Exception as e:
        print(f"Error extracting features from {image_path}: {e}")
        # Return zero vector as fallback
        return np.zeros(32 * 3 + 256 + 5 + 2 + 3, dtype=np.float32)


def extract_features_from_dataset(image_paths, labels=None):
    """
    Extract features from a list of images.
    
    Args:
        image_paths: List of image paths
        labels: Optional list of labels
        
    Returns:
        Tuple of (feature_matrix, labels) or just feature_matrix
    """
    features_list = []
    valid_labels = []
    
    print("Extracting handcrafted features...")
    for i, img_path in enumerate(tqdm(image_paths)):
        features = extract_handcrafted_features(img_path)
        features_list.append(features)
        if labels is not None:
            valid_labels.append(labels[i])
    
    feature_matrix = np.array(features_list)
    
    if labels is not None:
        return feature_matrix, np.array(valid_labels)
    return feature_matrix


def train_svm(X_train, y_train, X_val, y_val):
    """
    Train SVM classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        
    Returns:
        Trained SVM model and scaler
    """
    print("\nTraining SVM...")
    set_seed(42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train SVM
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_acc = accuracy_score(y_train, svm.predict(X_train_scaled))
    val_acc = accuracy_score(y_val, svm.predict(X_val_scaled))
    
    print(f"SVM Train Accuracy: {train_acc:.4f}")
    print(f"SVM Val Accuracy: {val_acc:.4f}")
    
    return svm, scaler


def train_random_forest(X_train, y_train, X_val, y_val):
    """
    Train Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        
    Returns:
        Trained Random Forest model
    """
    print("\nTraining Random Forest...")
    set_seed(42)
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    # Evaluate
    train_acc = accuracy_score(y_train, rf.predict(X_train))
    val_acc = accuracy_score(y_val, rf.predict(X_val))
    
    print(f"RF Train Accuracy: {train_acc:.4f}")
    print(f"RF Val Accuracy: {val_acc:.4f}")
    
    return rf


def train_xgboost(X_train, y_train, X_val, y_val):
    """
    Train XGBoost classifier (optional).
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        
    Returns:
        Trained XGBoost model and scaler
    """
    try:
        import xgboost as xgb
    except ImportError:
        print("XGBoost not installed. Skipping XGBoost training.")
        return None, None
    
    print("\nTraining XGBoost...")
    set_seed(42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_acc = accuracy_score(y_train, xgb_model.predict(X_train_scaled))
    val_acc = accuracy_score(y_val, xgb_model.predict(X_val_scaled))
    
    print(f"XGBoost Train Accuracy: {train_acc:.4f}")
    print(f"XGBoost Val Accuracy: {val_acc:.4f}")
    
    return xgb_model, scaler


def train_classical_ml_baselines():
    """
    Train all classical ML baselines.
    """
    print("=" * 60)
    print("Training Classical ML Baselines")
    print("=" * 60)
    
    # Load data splits
    train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = load_data_splits()
    
    # Extract features
    print("\nExtracting features from training set...")
    X_train, y_train = extract_features_from_dataset(train_paths, train_labels)
    
    print("\nExtracting features from validation set...")
    X_val, y_val = extract_features_from_dataset(val_paths, val_labels)
    
    print("\nExtracting features from test set...")
    X_test, y_test = extract_features_from_dataset(test_paths, test_labels)
    
    print(f"\nFeature dimensions: {X_train.shape[1]}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    # Train models
    models = {}
    scalers = {}
    
    # SVM
    svm, svm_scaler = train_svm(X_train, y_train, X_val, y_val)
    models['svm'] = svm
    scalers['svm'] = svm_scaler
    
    # Random Forest
    rf = train_random_forest(X_train, y_train, X_val, y_val)
    models['rf'] = rf
    
    # XGBoost (optional)
    xgb_model, xgb_scaler = train_xgboost(X_train, y_train, X_val, y_val)
    if xgb_model is not None:
        models['xgb'] = xgb_model
        scalers['xgb'] = xgb_scaler
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Test Set Evaluation")
    print("=" * 60)
    
    for name, model in models.items():
        if name in scalers:
            X_test_scaled = scalers[name].transform(X_test)
            y_pred = model.predict(X_test_scaled)
        else:
            y_pred = model.predict(X_test)
        
        test_acc = accuracy_score(y_test, y_pred)
        print(f"\n{name.upper()} Test Accuracy: {test_acc:.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=CLASSES)}")
    
    # Save models
    print("\nSaving models...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save SVM
    with open(SVM_MODEL_PATH, 'wb') as f:
        pickle.dump({'model': svm, 'scaler': svm_scaler}, f)
    print(f"Saved SVM to {SVM_MODEL_PATH}")
    
    # Save RF
    with open(RF_MODEL_PATH, 'wb') as f:
        pickle.dump(rf, f)
    print(f"Saved RF to {RF_MODEL_PATH}")
    
    # Save XGBoost if available
    if xgb_model is not None:
        with open(XGB_MODEL_PATH, 'wb') as f:
            pickle.dump({'model': xgb_model, 'scaler': xgb_scaler}, f)
        print(f"Saved XGBoost to {XGB_MODEL_PATH}")
    
    print("\nClassical ML baseline training completed!")


if __name__ == "__main__":
    train_classical_ml_baselines()







