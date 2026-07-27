"""
Training Visualizations Module
================================
Generates comprehensive ML training visualizations for demonstration.

All plots are saved as high-resolution PNG images in the ml_models/visualizations/ directory.

Visualizations generated:
    1.  Confusion Matrix Heatmap
    2.  Normalized Confusion Matrix (percentage-based)
    3.  Classification Report Heatmap (Precision / Recall / F1)
    4.  Per-Class Accuracy Bar Chart
    5.  Category Distribution (Training Data)
    6.  Train vs Test Split Distribution
    7.  ROC Curves (One-vs-Rest, multi-class)
    8.  Precision-Recall Curves (One-vs-Rest)
    9.  Top TF-IDF Features per Category
    10. Model Confidence Distribution
    11. Per-Category Confidence Box Plot
    12. Learning Curve
    13. Cross-Validation Scores
    14. PCA — 2D Feature Space Visualization
    15. Training Summary Dashboard
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
)
from sklearn.model_selection import learning_curve, cross_val_score
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA


# ── Style Configuration ──────────────────────────────────────────────────────
# Professional dark theme for all visualizations
STYLE_CONFIG = {
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'axes.edgecolor': '#e94560',
    'axes.labelcolor': '#eee',
    'text.color': '#eee',
    'xtick.color': '#ccc',
    'ytick.color': '#ccc',
    'grid.color': '#333',
    'grid.alpha': 0.3,
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
}

CMAP_PRIMARY = 'magma'
CMAP_DIVERGING = 'RdYlGn'
COLOR_ACCENT = '#e94560'
COLOR_SECONDARY = '#0f3460'
COLOR_HIGHLIGHT = '#00d2ff'
COLOR_SUCCESS = '#00e676'
COLOR_WARNING = '#ffc107'


def _setup_style():
    """Apply the custom dark style to all plots."""
    plt.rcParams.update(STYLE_CONFIG)


def _save_figure(fig, output_dir, filename, dpi=200):
    """Save figure and close it."""
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'    [OK] Saved: {filename}')
    return filepath


# ═════════════════════════════════════════════════════════════════════════════
#  1. CONFUSION MATRIX (Raw Counts)
# ═════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(y_true, y_pred, labels, output_dir):
    """Plot confusion matrix with raw count values."""
    _setup_style()
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(18, 15))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='YlOrRd',
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, linecolor='#333',
        cbar_kws={'label': 'Count', 'shrink': 0.8},
        ax=ax,
    )
    ax.set_xlabel('Predicted Category', fontweight='bold', fontsize=13)
    ax.set_ylabel('True Category', fontweight='bold', fontsize=13)
    ax.set_title('Confusion Matrix (Raw Counts)', fontweight='bold', fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    return _save_figure(fig, output_dir, '01_confusion_matrix.png')


# ═════════════════════════════════════════════════════════════════════════════
#  2. NORMALIZED CONFUSION MATRIX (Percentages)
# ═════════════════════════════════════════════════════════════════════════════
def plot_normalized_confusion_matrix(y_true, y_pred, labels, output_dir):
    """Plot confusion matrix normalized by true label (shows recall per cell)."""
    _setup_style()
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Normalize by row (true labels) — avoid division by zero
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_sum[cm_sum == 0] = 1
    cm_norm = cm.astype('float') / cm_sum

    fig, ax = plt.subplots(figsize=(18, 15))
    sns.heatmap(
        cm_norm, annot=True, fmt='.1%', cmap='YlGnBu',
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, linecolor='#333',
        vmin=0, vmax=1,
        cbar_kws={'label': 'Proportion', 'shrink': 0.8},
        ax=ax,
    )
    ax.set_xlabel('Predicted Category', fontweight='bold', fontsize=13)
    ax.set_ylabel('True Category', fontweight='bold', fontsize=13)
    ax.set_title('Normalized Confusion Matrix (Row-wise %)', fontweight='bold', fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    return _save_figure(fig, output_dir, '02_confusion_matrix_normalized.png')


# ═════════════════════════════════════════════════════════════════════════════
#  3. CLASSIFICATION REPORT HEATMAP
# ═════════════════════════════════════════════════════════════════════════════
def plot_classification_report(y_true, y_pred, labels, output_dir):
    """Visualize precision, recall, F1-score, and support as a heatmap."""
    _setup_style()
    report = classification_report(y_true, y_pred, labels=labels,
                                   output_dict=True, zero_division=0)

    # Build DataFrame from per-class metrics only
    rows = []
    for label in labels:
        if label in report:
            rows.append({
                'Category': label,
                'Precision': report[label]['precision'],
                'Recall': report[label]['recall'],
                'F1-Score': report[label]['f1-score'],
                'Support': report[label]['support'],
            })

    df = pd.DataFrame(rows).set_index('Category')
    metrics_df = df[['Precision', 'Recall', 'F1-Score']]

    fig, ax = plt.subplots(figsize=(10, 16))
    sns.heatmap(
        metrics_df, annot=True, fmt='.2f', cmap=CMAP_DIVERGING,
        linewidths=0.8, linecolor='#333',
        vmin=0, vmax=1,
        cbar_kws={'label': 'Score', 'shrink': 0.6},
        ax=ax,
    )
    ax.set_title('Classification Report — Precision / Recall / F1',
                 fontweight='bold', fontsize=16, pad=20)
    ax.set_ylabel('')
    plt.yticks(rotation=0, fontsize=10)
    plt.xticks(fontsize=12, fontweight='bold')

    return _save_figure(fig, output_dir, '03_classification_report_heatmap.png')


# ═════════════════════════════════════════════════════════════════════════════
#  4. PER-CLASS ACCURACY BAR CHART
# ═════════════════════════════════════════════════════════════════════════════
def plot_per_class_accuracy(y_true, y_pred, labels, output_dir):
    """Horizontal bar chart showing accuracy for each category."""
    _setup_style()
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class_acc = []
    for i, label in enumerate(labels):
        total = cm[i].sum()
        correct = cm[i][i]
        acc = correct / total if total > 0 else 0
        per_class_acc.append(acc)

    # Sort by accuracy
    sorted_indices = np.argsort(per_class_acc)
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_acc = [per_class_acc[i] for i in sorted_indices]

    # Color gradient based on accuracy
    colors = [COLOR_ACCENT if a < 0.7 else COLOR_WARNING if a < 0.9 else COLOR_SUCCESS
              for a in sorted_acc]

    fig, ax = plt.subplots(figsize=(12, 10))
    bars = ax.barh(sorted_labels, sorted_acc, color=colors, edgecolor='#444', height=0.7)

    # Add value labels on bars
    for bar, acc in zip(bars, sorted_acc):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{acc:.1%}', va='center', fontsize=10, fontweight='bold', color='#eee')

    ax.set_xlim(0, 1.15)
    ax.set_xlabel('Accuracy', fontweight='bold', fontsize=13)
    ax.set_title('Per-Class Accuracy', fontweight='bold', fontsize=16, pad=20)
    ax.axvline(x=0.9, color=COLOR_HIGHLIGHT, linestyle='--', alpha=0.5, label='90% threshold')
    ax.legend(loc='lower right', fontsize=10)

    return _save_figure(fig, output_dir, '04_per_class_accuracy.png')


# ═════════════════════════════════════════════════════════════════════════════
#  5. CATEGORY DISTRIBUTION (Training Data)
# ═════════════════════════════════════════════════════════════════════════════
def plot_category_distribution(labels, output_dir):
    """Bar chart showing sample count per category in the full dataset."""
    _setup_style()
    unique, counts = np.unique(labels, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]

    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.magma(np.linspace(0.3, 0.85, len(unique)))
    bars = ax.bar([unique[i] for i in sorted_indices],
                  [counts[i] for i in sorted_indices],
                  color=[colors[i] for i in range(len(unique))],
                  edgecolor='#555', width=0.7)

    # Add count labels
    for bar, count in zip(bars, [counts[i] for i in sorted_indices]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom', fontsize=9, fontweight='bold', color='#eee')

    ax.set_xlabel('Category', fontweight='bold', fontsize=13)
    ax.set_ylabel('Number of Samples', fontweight='bold', fontsize=13)
    ax.set_title('Dataset — Category Distribution', fontweight='bold', fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=9)

    return _save_figure(fig, output_dir, '05_category_distribution.png')


# ═════════════════════════════════════════════════════════════════════════════
#  6. TRAIN vs TEST SPLIT DISTRIBUTION
# ═════════════════════════════════════════════════════════════════════════════
def plot_train_test_split(y_train, y_test, output_dir):
    """Grouped bar chart comparing train and test sample counts per category."""
    _setup_style()
    train_unique, train_counts = np.unique(y_train, return_counts=True)
    test_unique, test_counts = np.unique(y_test, return_counts=True)

    all_labels = sorted(set(list(train_unique) + list(test_unique)))
    train_dict = dict(zip(train_unique, train_counts))
    test_dict = dict(zip(test_unique, test_counts))

    train_vals = [train_dict.get(l, 0) for l in all_labels]
    test_vals = [test_dict.get(l, 0) for l in all_labels]

    x = np.arange(len(all_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(16, 8))
    bars1 = ax.bar(x - width / 2, train_vals, width, label='Train', color='#0f3460', edgecolor='#555')
    bars2 = ax.bar(x + width / 2, test_vals, width, label='Test', color='#e94560', edgecolor='#555')

    ax.set_xlabel('Category', fontweight='bold', fontsize=13)
    ax.set_ylabel('Sample Count', fontweight='bold', fontsize=13)
    ax.set_title('Train vs Test Split — Per Category', fontweight='bold', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=12, loc='upper right')

    return _save_figure(fig, output_dir, '06_train_test_split.png')


# ═════════════════════════════════════════════════════════════════════════════
#  7. ROC CURVES (Multi-class One-vs-Rest)
# ═════════════════════════════════════════════════════════════════════════════
def plot_roc_curves(classifier, X_test_features, y_test, labels, output_dir):
    """Plot ROC curves for each class using One-vs-Rest strategy."""
    _setup_style()

    # Binarize true labels
    y_bin = label_binarize(y_test, classes=labels)
    n_classes = len(labels)

    # Get probability predictions
    if hasattr(classifier, 'predict_proba'):
        y_score = classifier.predict_proba(X_test_features)
    elif hasattr(classifier, 'decision_function'):
        y_score = classifier.decision_function(X_test_features)
    else:
        print('    [SKIP] Skipping ROC curves - model has no probability output')
        return None

    fig, ax = plt.subplots(figsize=(14, 10))
    colors = plt.cm.tab20(np.linspace(0, 1, n_classes))

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    for i in range(n_classes):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        ax.plot(fpr, tpr, color=colors[i], alpha=0.4, linewidth=1)

    # Mean ROC
    if tprs:
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        ax.plot(mean_fpr, mean_tpr, color=COLOR_HIGHLIGHT, linewidth=3,
                label=f'Mean ROC (AUC = {mean_auc:.3f})', zorder=10)

    ax.plot([0, 1], [0, 1], 'w--', alpha=0.3, label='Random Chance')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=13)
    ax.set_title(f'ROC Curves — One-vs-Rest ({n_classes} Classes)',
                 fontweight='bold', fontsize=16, pad=20)
    ax.legend(loc='lower right', fontsize=11)

    return _save_figure(fig, output_dir, '07_roc_curves.png')


# ═════════════════════════════════════════════════════════════════════════════
#  8. PRECISION-RECALL CURVES
# ═════════════════════════════════════════════════════════════════════════════
def plot_precision_recall_curves(classifier, X_test_features, y_test, labels, output_dir):
    """Plot precision-recall curves for each class."""
    _setup_style()

    y_bin = label_binarize(y_test, classes=labels)
    n_classes = len(labels)

    if hasattr(classifier, 'predict_proba'):
        y_score = classifier.predict_proba(X_test_features)
    elif hasattr(classifier, 'decision_function'):
        y_score = classifier.decision_function(X_test_features)
    else:
        print('    [SKIP] Skipping PR curves - model has no probability output')
        return None

    fig, ax = plt.subplots(figsize=(14, 10))
    colors = plt.cm.tab20(np.linspace(0, 1, n_classes))

    avg_precisions = []

    for i in range(n_classes):
        if y_bin[:, i].sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
        ap = average_precision_score(y_bin[:, i], y_score[:, i])
        avg_precisions.append(ap)
        ax.plot(recall, precision, color=colors[i], alpha=0.4, linewidth=1)

    if avg_precisions:
        mean_ap = np.mean(avg_precisions)
        ax.axhline(y=mean_ap, color=COLOR_HIGHLIGHT, linestyle='--', linewidth=2,
                    label=f'Mean AP = {mean_ap:.3f}')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('Recall', fontweight='bold', fontsize=13)
    ax.set_ylabel('Precision', fontweight='bold', fontsize=13)
    ax.set_title(f'Precision-Recall Curves — One-vs-Rest ({n_classes} Classes)',
                 fontweight='bold', fontsize=16, pad=20)
    ax.legend(loc='lower left', fontsize=11)

    return _save_figure(fig, output_dir, '08_precision_recall_curves.png')


# ═════════════════════════════════════════════════════════════════════════════
#  9. TOP TF-IDF FEATURES PER CATEGORY
# ═════════════════════════════════════════════════════════════════════════════
def plot_top_features(vectorizer, classifier, labels, output_dir, top_n=10):
    """Show the most important TF-IDF features (words) for each category."""
    _setup_style()
    feature_names = vectorizer.get_feature_names_out()

    # Get feature importance from the base SVM if using CalibratedClassifierCV
    base_clf = classifier
    if hasattr(classifier, 'calibrated_classifiers_'):
        # CalibratedClassifierCV stores calibrated classifiers
        try:
            inner = classifier.calibrated_classifiers_[0]
            if hasattr(inner, 'estimator') and hasattr(inner.estimator, 'coef_'):
                base_clf = inner.estimator
            elif hasattr(inner, 'base_estimator') and hasattr(inner.base_estimator, 'coef_'):
                base_clf = inner.base_estimator
            else:
                base_clf = None
        except (AttributeError, IndexError):
            base_clf = None
    elif hasattr(classifier, 'estimators_'):
        try:
            est = classifier.estimators_[0]
            if hasattr(est, 'coef_'):
                base_clf = est
            elif hasattr(est, 'estimator') and hasattr(est.estimator, 'coef_'):
                base_clf = est.estimator
            else:
                base_clf = None
        except (AttributeError, IndexError):
            base_clf = None

    if base_clf is None or not hasattr(base_clf, 'coef_'):
        print('    [SKIP] Skipping feature importance - cannot extract SVM coefficients')
        return None

    coef = base_clf.coef_
    classes = base_clf.classes_ if hasattr(base_clf, 'classes_') else labels

    # Select a subset of categories for readability (up to 9)
    n_show = min(9, len(classes))
    selected_indices = np.linspace(0, len(classes) - 1, n_show, dtype=int)

    cols = 3
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten() if n_show > 1 else [axes]

    for idx, ax in enumerate(axes):
        if idx >= n_show:
            ax.set_visible(False)
            continue

        class_idx = selected_indices[idx]
        class_name = classes[class_idx]
        top_indices = np.argsort(coef[class_idx])[-top_n:]
        top_words = [feature_names[i] for i in top_indices]
        top_scores = [coef[class_idx][i] for i in top_indices]

        colors = plt.cm.magma(np.linspace(0.3, 0.85, top_n))
        ax.barh(top_words, top_scores, color=colors, edgecolor='#555')
        ax.set_title(class_name, fontweight='bold', fontsize=12)
        ax.tick_params(axis='y', labelsize=9)

    fig.suptitle('Top TF-IDF Features by Category (SVM Coefficients)',
                 fontweight='bold', fontsize=18, y=1.02)
    plt.tight_layout()

    return _save_figure(fig, output_dir, '09_top_features_per_category.png')


# ═════════════════════════════════════════════════════════════════════════════
#  10. MODEL CONFIDENCE DISTRIBUTION
# ═════════════════════════════════════════════════════════════════════════════
def plot_confidence_distribution(confidences, y_true, y_pred, output_dir):
    """Histogram of model confidence scores, split by correct/incorrect."""
    _setup_style()

    correct_conf = [c for c, yt, yp in zip(confidences, y_true, y_pred) if yt == yp]
    wrong_conf = [c for c, yt, yp in zip(confidences, y_true, y_pred) if yt != yp]

    fig, ax = plt.subplots(figsize=(12, 7))

    bins = np.linspace(0, 1, 25)
    if correct_conf:
        ax.hist(correct_conf, bins=bins, alpha=0.7, label=f'Correct ({len(correct_conf)})',
                color=COLOR_SUCCESS, edgecolor='#333')
    if wrong_conf:
        ax.hist(wrong_conf, bins=bins, alpha=0.7, label=f'Incorrect ({len(wrong_conf)})',
                color=COLOR_ACCENT, edgecolor='#333')

    mean_conf = np.mean(confidences)
    ax.axvline(x=mean_conf, color=COLOR_HIGHLIGHT, linestyle='--', linewidth=2,
               label=f'Mean Confidence: {mean_conf:.3f}')

    ax.set_xlabel('Confidence Score', fontweight='bold', fontsize=13)
    ax.set_ylabel('Number of Predictions', fontweight='bold', fontsize=13)
    ax.set_title('Model Confidence Distribution', fontweight='bold', fontsize=16, pad=20)
    ax.legend(fontsize=12)

    return _save_figure(fig, output_dir, '10_confidence_distribution.png')


# ═════════════════════════════════════════════════════════════════════════════
#  11. PER-CATEGORY CONFIDENCE BOX PLOT
# ═════════════════════════════════════════════════════════════════════════════
def plot_confidence_by_category(confidences, y_true, labels, output_dir):
    """Box plot showing confidence score distribution per true category."""
    _setup_style()

    data = []
    for conf, label in zip(confidences, y_true):
        data.append({'Category': label, 'Confidence': conf})
    df = pd.DataFrame(data)

    # Only include categories that appear in the data
    present = [l for l in labels if l in df['Category'].values]

    fig, ax = plt.subplots(figsize=(16, 8))
    box_colors = plt.cm.magma(np.linspace(0.3, 0.85, len(present)))

    bp = ax.boxplot(
        [df[df['Category'] == cat]['Confidence'].values for cat in present],
        tick_labels=present, patch_artist=True, showmeans=True,
        meanprops=dict(marker='D', markerfacecolor=COLOR_HIGHLIGHT, markersize=6),
        medianprops=dict(color=COLOR_HIGHLIGHT, linewidth=2),
        flierprops=dict(marker='o', markerfacecolor=COLOR_ACCENT, markersize=4),
    )
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_ylabel('Confidence Score', fontweight='bold', fontsize=13)
    ax.set_title('Prediction Confidence by Category', fontweight='bold', fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    ax.axhline(y=0.5, color=COLOR_WARNING, linestyle='--', alpha=0.4, label='0.5 threshold')
    ax.legend(fontsize=10)

    return _save_figure(fig, output_dir, '11_confidence_by_category.png')


# ═════════════════════════════════════════════════════════════════════════════
#  12. LEARNING CURVE
# ═════════════════════════════════════════════════════════════════════════════
def plot_learning_curve(vectorizer, X_train, y_train, output_dir):
    """Plot learning curve showing model performance vs. training set size."""
    _setup_style()

    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline

    # Build pipeline for learning curve computation
    pipeline = Pipeline([
        ('tfidf', vectorizer),
        ('clf', CalibratedClassifierCV(LinearSVC(max_iter=10000, C=1.0, class_weight='balanced'), cv=2)),
    ])

    # Compute learning curve
    n_samples = len(X_train)
    # Create reasonable train sizes based on available data
    max_points = min(5, n_samples)
    train_sizes = np.linspace(0.3, 1.0, max_points)

    try:
        train_sizes_abs, train_scores, val_scores = learning_curve(
            pipeline, X_train, y_train,
            train_sizes=train_sizes,
            cv=min(3, min(np.unique(y_train, return_counts=True)[1])),  # cv <= min class count
            scoring='accuracy',
            n_jobs=1,
        )
    except Exception as e:
        print(f'    [SKIP] Skipping learning curve - {e}')
        return None

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color=COLOR_SUCCESS)
    ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color=COLOR_ACCENT)
    ax.plot(train_sizes_abs, train_mean, 'o-', color=COLOR_SUCCESS,
            linewidth=2, markersize=8, label='Training Score')
    ax.plot(train_sizes_abs, val_mean, 's-', color=COLOR_ACCENT,
            linewidth=2, markersize=8, label='Validation Score')

    ax.set_xlabel('Training Set Size', fontweight='bold', fontsize=13)
    ax.set_ylabel('Accuracy', fontweight='bold', fontsize=13)
    ax.set_title('Learning Curve — SVM Classifier', fontweight='bold', fontsize=16, pad=20)
    ax.legend(fontsize=12, loc='lower right')
    ax.set_ylim(0, 1.1)

    return _save_figure(fig, output_dir, '12_learning_curve.png')


# ═════════════════════════════════════════════════════════════════════════════
#  13. CROSS-VALIDATION SCORES
# ═════════════════════════════════════════════════════════════════════════════
def plot_cross_validation(vectorizer, X_train, y_train, output_dir):
    """Bar chart + box plot of cross-validation fold scores."""
    _setup_style()

    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([
        ('tfidf', vectorizer),
        ('clf', CalibratedClassifierCV(LinearSVC(max_iter=10000, C=1.0, class_weight='balanced'), cv=2)),
    ])

    n_folds = min(5, min(np.unique(y_train, return_counts=True)[1]))
    if n_folds < 2:
        print('    [SKIP] Skipping cross-validation - not enough samples per class')
        return None

    try:
        scores = cross_val_score(pipeline, X_train, y_train, cv=n_folds,
                                 scoring='accuracy', n_jobs=1)
    except Exception as e:
        print(f'    [SKIP] Skipping cross-validation - {e}')
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [3, 1]})

    # Bar chart of fold scores
    fold_colors = [COLOR_SUCCESS if s >= np.mean(scores) else COLOR_WARNING for s in scores]
    bars = ax1.bar(range(1, len(scores) + 1), scores, color=fold_colors,
                   edgecolor='#555', width=0.6)
    ax1.axhline(y=np.mean(scores), color=COLOR_HIGHLIGHT, linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(scores):.3f}')
    ax1.set_xlabel('Fold', fontweight='bold', fontsize=13)
    ax1.set_ylabel('Accuracy', fontweight='bold', fontsize=13)
    ax1.set_title(f'{n_folds}-Fold Cross-Validation Scores', fontweight='bold', fontsize=14)
    ax1.set_ylim(0, 1.15)
    ax1.legend(fontsize=11)

    # Add value labels
    for bar, score in zip(bars, scores):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{score:.3f}', ha='center', fontsize=11, fontweight='bold', color='#eee')

    # Box plot summary
    bp = ax2.boxplot(scores, patch_artist=True, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor=COLOR_HIGHLIGHT, markersize=8),
                     medianprops=dict(color=COLOR_HIGHLIGHT, linewidth=2))
    bp['boxes'][0].set_facecolor(COLOR_SECONDARY)
    bp['boxes'][0].set_alpha(0.7)
    ax2.set_title('Distribution', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Accuracy', fontweight='bold', fontsize=13)
    ax2.set_ylim(0, 1.15)

    fig.suptitle('Cross-Validation Analysis', fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()

    return _save_figure(fig, output_dir, '13_cross_validation.png')


# ═════════════════════════════════════════════════════════════════════════════
#  14. PCA — 2D FEATURE SPACE VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════
def plot_pca_visualization(X_features, labels_array, unique_labels, output_dir):
    """2D PCA scatter plot of TF-IDF feature space, colored by category."""
    _setup_style()

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_features.toarray() if hasattr(X_features, 'toarray') else X_features)

    fig, ax = plt.subplots(figsize=(16, 12))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = np.array(labels_array) == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[colors[i]], label=label,
                   alpha=0.7, s=60, edgecolors='white', linewidth=0.5)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                  fontweight='bold', fontsize=13)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)',
                  fontweight='bold', fontsize=13)
    ax.set_title('PCA — 2D Feature Space Visualization',
                 fontweight='bold', fontsize=16, pad=20)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8,
              ncol=1, framealpha=0.8)

    return _save_figure(fig, output_dir, '14_pca_feature_space.png')


# ═════════════════════════════════════════════════════════════════════════════
#  15. TRAINING SUMMARY DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
def plot_summary_dashboard(eval_metrics, y_true, y_pred, confidences, labels, output_dir):
    """Single-page dashboard summarizing all key training results."""
    _setup_style()

    fig = plt.figure(figsize=(24, 16))
    fig.patch.set_facecolor('#1a1a2e')
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    # ── Panel 1: Key Metrics Cards ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    metrics_text = (
        f"TRAINING SUMMARY\n"
        f"{'─' * 30}\n\n"
        f"  Train Accuracy:   {eval_metrics.get('train_accuracy', 0):.1%}\n\n"
        f"  Test Accuracy:    {eval_metrics.get('test_accuracy', 0):.1%}\n\n"
        f"  Total Samples:    {eval_metrics.get('n_samples', 0)}\n\n"
        f"  TF-IDF Features:  {eval_metrics.get('n_features', 0)}\n\n"
        f"  Categories:       {eval_metrics.get('n_categories', 0)}\n\n"
        f"  Mean Confidence:  {np.mean(confidences):.3f}\n"
    )
    ax1.text(0.1, 0.95, metrics_text, transform=ax1.transAxes,
             fontsize=14, verticalalignment='top', fontfamily='monospace',
             color='#eee',
             bbox=dict(boxstyle='round,pad=1', facecolor='#16213e', edgecolor=COLOR_ACCENT, linewidth=2))

    # ── Panel 2: Mini Confusion Matrix ──────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1:])
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_sum[cm_sum == 0] = 1
    cm_norm = cm.astype('float') / cm_sum
    sns.heatmap(cm_norm, annot=False, cmap='YlOrRd',
                xticklabels=labels, yticklabels=labels,
                linewidths=0.3, ax=ax2, cbar_kws={'shrink': 0.6})
    ax2.set_title('Normalized Confusion Matrix', fontweight='bold', fontsize=13)
    ax2.tick_params(axis='both', labelsize=7)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

    # ── Panel 3: Confidence Histogram ───────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    correct_conf = [c for c, yt, yp in zip(confidences, y_true, y_pred) if yt == yp]
    wrong_conf = [c for c, yt, yp in zip(confidences, y_true, y_pred) if yt != yp]
    bins = np.linspace(0, 1, 20)
    if correct_conf:
        ax3.hist(correct_conf, bins=bins, alpha=0.7, color=COLOR_SUCCESS, label='Correct')
    if wrong_conf:
        ax3.hist(wrong_conf, bins=bins, alpha=0.7, color=COLOR_ACCENT, label='Incorrect')
    ax3.set_title('Confidence Distribution', fontweight='bold', fontsize=13)
    ax3.set_xlabel('Confidence')
    ax3.legend(fontsize=9)

    # ── Panel 4: Per-class F1 bar chart ─────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    report = classification_report(y_true, y_pred, labels=labels,
                                   output_dict=True, zero_division=0)
    f1_scores = [report[l]['f1-score'] for l in labels if l in report]
    f1_labels = [l for l in labels if l in report]
    sorted_idx = np.argsort(f1_scores)
    colors_f1 = [COLOR_ACCENT if f < 0.7 else COLOR_WARNING if f < 0.9 else COLOR_SUCCESS
                 for f in [f1_scores[i] for i in sorted_idx]]
    ax4.barh([f1_labels[i] for i in sorted_idx], [f1_scores[i] for i in sorted_idx],
             color=colors_f1, edgecolor='#444')
    ax4.set_title('F1-Score by Category', fontweight='bold', fontsize=13)
    ax4.set_xlim(0, 1.1)
    ax4.tick_params(axis='y', labelsize=7)

    # ── Panel 5: Category distribution ──────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    unique, counts = np.unique(y_true, return_counts=True)
    sorted_idx2 = np.argsort(counts)[::-1]
    colors_dist = plt.cm.magma(np.linspace(0.3, 0.85, len(unique)))
    ax5.bar([unique[i] for i in sorted_idx2], [counts[i] for i in sorted_idx2],
            color=colors_dist, edgecolor='#444')
    ax5.set_title('Test Set Distribution', fontweight='bold', fontsize=13)
    plt.setp(ax5.get_xticklabels(), rotation=45, ha='right', fontsize=7)

    fig.suptitle('RBJRS — Resume Classifier Training Dashboard',
                 fontweight='bold', fontsize=22, color=COLOR_HIGHLIGHT, y=0.98)

    return _save_figure(fig, output_dir, '15_training_dashboard.png', dpi=250)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN — Generate All Visualizations
# ═════════════════════════════════════════════════════════════════════════════
def generate_all_visualizations(
    y_true, y_pred, y_train, y_test,
    X_train_raw, X_test_features, X_all_features,
    all_labels, labels_all, confidences,
    classifier, vectorizer, eval_metrics,
    model_dir,
):
    """
    Master function — generates all 15 visualizations.

    Args:
        y_true: True labels for the test set.
        y_pred: Predicted labels for the test set.
        y_train: Training set labels.
        y_test: Test set labels (same as y_true).
        X_train_raw: Raw training texts (for learning curve).
        X_test_features: TF-IDF features for the test set.
        X_all_features: TF-IDF features for all data.
        all_labels: Sorted list of unique category labels.
        labels_all: Full list of labels for all data.
        confidences: List of confidence scores for test predictions.
        classifier: Trained classifier object.
        vectorizer: Fitted TF-IDF vectorizer.
        eval_metrics: Dictionary of training/test metrics.
        model_dir: Root model directory.
    """
    output_dir = os.path.join(model_dir, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)

    print(f'\n  Generating visualizations in: {output_dir}')
    print(f'  {"-" * 50}')

    saved = []

    # 1 & 2: Confusion Matrices
    saved.append(plot_confusion_matrix(y_true, y_pred, all_labels, output_dir))
    saved.append(plot_normalized_confusion_matrix(y_true, y_pred, all_labels, output_dir))

    # 3: Classification Report
    saved.append(plot_classification_report(y_true, y_pred, all_labels, output_dir))

    # 4: Per-class Accuracy
    saved.append(plot_per_class_accuracy(y_true, y_pred, all_labels, output_dir))

    # 5: Category Distribution
    saved.append(plot_category_distribution(labels_all, output_dir))

    # 6: Train vs Test Split
    saved.append(plot_train_test_split(y_train, y_test, output_dir))

    # 7: ROC Curves
    saved.append(plot_roc_curves(classifier, X_test_features, y_test, all_labels, output_dir))

    # 8: Precision-Recall Curves
    saved.append(plot_precision_recall_curves(classifier, X_test_features, y_test,
                                              all_labels, output_dir))

    # 9: Top Features
    saved.append(plot_top_features(vectorizer, classifier, all_labels, output_dir))

    # 10: Confidence Distribution
    saved.append(plot_confidence_distribution(confidences, y_true, y_pred, output_dir))

    # 11: Confidence by Category
    saved.append(plot_confidence_by_category(confidences, y_true, all_labels, output_dir))

    # 12: Learning Curve
    saved.append(plot_learning_curve(vectorizer, X_train_raw, y_train, output_dir))

    # 13: Cross-Validation
    saved.append(plot_cross_validation(vectorizer, X_train_raw, y_train, output_dir))

    # 14: PCA Visualization
    saved.append(plot_pca_visualization(X_all_features, labels_all, all_labels, output_dir))

    # 15: Summary Dashboard
    saved.append(plot_summary_dashboard(eval_metrics, y_true, y_pred, confidences,
                                         all_labels, output_dir))

    generated = [s for s in saved if s is not None]
    print(f'\n  [DONE] {len(generated)}/{len(saved)} visualizations generated successfully!')
    print(f'  Output directory: {output_dir}')

    return output_dir
