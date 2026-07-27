"""
Resume Classifier Training Script
====================================
Trains the TF-IDF + SVM resume classifier on sample data.

Since the Kaggle dataset requires manual download, this script provides:
1. A built-in sample dataset for immediate training (150+ samples, 25 categories)
2. Support for loading external CSV datasets
3. Model evaluation with accuracy, precision, recall, F1 metrics
4. Saved model files in ml_models/ directory
5. Comprehensive training visualizations (15 charts) saved to ml_models/visualizations/

Usage:
    python -m app.ml.training.train_classifier
"""

import os
import sys
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from app.ml.text_preprocessor import text_preprocessor
from app.ml.resume_classifier import ResumeClassifier


import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from app.ml.text_preprocessor import text_preprocessor
from app.ml.resume_classifier import ResumeClassifier


def load_real_dataset(csv_path):
    """
    Load real-world resume dataset from CSV (Directive 3 Task 3 Fix).
    Supports Kaggle Resume Dataset format ('Category', 'Resume' columns).
    """
    if not os.path.exists(csv_path):
        print(f"\n[ERROR] Real dataset CSV not found at: {csv_path}")
        print("\nTo train the classifier on real data:")
        print("1. Download the Kaggle Resume Dataset (UpdatedResumeDataSet.csv)")
        print("2. Place the CSV file in 'ml_models/UpdatedResumeDataSet.csv' or pass --csv_path")
        print("3. Re-run: python -m app.ml.training.train_classifier --csv_path <path-to-csv>\n")
        sys.exit(1)

    print(f"\nLoading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)

    if 'Resume' not in df.columns or 'Category' not in df.columns:
        raise ValueError("CSV dataset must contain 'Category' and 'Resume' columns.")

    print(f"Loaded {len(df)} records across {df['Category'].nunique()} categories.")
    
    texts = [text_preprocessor.preprocess(str(t)) for t in df['Resume']]
    labels = df['Category'].astype(str).tolist()

    return texts, labels


def train_model(csv_path=None, model_dir=None):
    """
    Train the resume classifier on real dataset, evaluate, and save model artifacts.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    if model_dir is None:
        model_dir = os.path.join(project_root, 'ml_models')

    if csv_path is None:
        csv_path = os.path.join(model_dir, 'UpdatedResumeDataSet.csv')

    print('=' * 60)
    print('Resume Classifier — Real Data Training Pipeline')
    print('=' * 60)

    # Step 1: Load dataset
    print('\n[1/6] Loading real-world resume dataset...')
    texts, labels = load_real_dataset(csv_path)

    # Step 2: Split into train/test
    print('\n[2/6] Splitting data (80/20)...')
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f'  Training samples: {len(X_train)}')
    print(f'  Testing samples:  {len(X_test)}')

    # Step 3: Train model
    print('\n[3/6] Training TF-IDF + LinearSVC classifier...')
    metrics = ResumeClassifier.train_and_save(X_train, y_train, model_dir)
    print(f'  Training accuracy: {metrics["accuracy"]:.4f}')

    # Step 4: Evaluate
    print('\n[4/6] Evaluating on test set...')
    classifier = ResumeClassifier()
    classifier.load_model(model_dir)

    y_pred = []
    confidences = []
    for text in X_test:
        category, confidence = classifier.predict(text)
        y_pred.append(category)
        confidences.append(confidence)

    test_accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    print(f'  Test accuracy: {test_accuracy:.4f}')

    # Step 5: Save metrics
    print('\n[5/6] Saving evaluation metrics...')
    eval_metrics = {
        'train_accuracy': float(metrics['accuracy']),
        'test_accuracy': float(test_accuracy),
        'n_samples': len(texts),
        'n_features': metrics['n_features'],
        'n_categories': len(set(labels)),
        'mean_confidence': float(np.mean(confidences)),
    }

    metrics_path = os.path.join(model_dir, 'training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(eval_metrics, f, indent=2)
    print(f'  Metrics saved to {metrics_path}')

    print(f'\n[6/6] Training complete!')
    return eval_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Resume Classifier on Real CSV Data")
    parser.add_argument('--csv_path', type=str, help="Path to real resume CSV dataset")
    parser.add_argument('--model_dir', type=str, help="Output directory for trained models")
    args = parser.parse_args()

    train_model(csv_path=args.csv_path, model_dir=args.model_dir)
