"""
Evaluation and Diagnostic Reporting Module (DW-9, FR-4, Section 7)
------------------------------------------------------------------
This module provides functions to calculate:
- Regression metrics: MAE, RMSE, R2
- Classification metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- Diagnostic visualizations: Residual plots, Confusion matrix heatmap, Model comparison charts
Using transparent, junior-level Python logic and formulas.
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ==========================================
# 1. QUANTITATIVE METRICS (DW-9)
# ==========================================

def mean_absolute_error(y_true, y_pred):
    """
    Calculate Mean Absolute Error (MAE): (1/n) * sum(|y_true - y_pred|)
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true, y_pred):
    """
    Calculate Root Mean Squared Error (RMSE): sqrt((1/n) * sum((y_true - y_pred)^2))
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mse = np.mean((y_true - y_pred) ** 2)
    return float(math.sqrt(mse))


def r2_score(y_true, y_pred):
    """
    Calculate Coefficient of Determination (R^2): 1 - (SS_res / SS_tot)
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def compute_confusion_matrix(y_true, y_pred):
    """
    Compute 2x2 Confusion Matrix for binary classification.
    Returns:
    --------
    cm : dict containing {'TN': int, 'FP': int, 'FN': int, 'TP': int, 'matrix': [[TN, FP], [FN, TP]]}
    """
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    
    return {
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'matrix': np.array([[tn, fp], [fn, tp]])
    }


def classification_metrics(y_true, y_pred):
    """
    Compute Accuracy, Precision, Recall, and F1-Score.
    """
    cm = compute_confusion_matrix(y_true, y_pred)
    tp, tn, fp, fn = cm['TP'], cm['TN'], cm['FP'], cm['FN']
    total = tp + tn + fp + fn
    
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'confusion_matrix': cm
    }


# ==========================================
# 2. DIAGNOSTIC PLOTS (DW-9, DW-10)
# ==========================================

def plot_residuals(y_true, y_pred, model_name="Linear Regression", output_path="figures/regression_residuals.png"):
    """
    Plot Residuals vs. Predicted values and Actual vs. Predicted scatter plot.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    residuals = np.array(y_true) - np.array(y_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Actual vs Predicted
    axes[0].scatter(y_true, y_pred, color='#1f77b4', alpha=0.7, edgecolors='black', s=50)
    # Ideal identity line y = x
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal Fit (y = x)')
    axes[0].set_title(f"Actual vs. Predicted Final Grade ({model_name})", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Actual Final Grade (G3)", fontsize=10)
    axes[0].set_ylabel("Predicted Final Grade", fontsize=10)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend()
    
    # 2. Residuals vs Predicted
    axes[1].scatter(y_pred, residuals, color='#ff7f0e', alpha=0.7, edgecolors='black', s=50)
    axes[1].axhline(y=0, color='crimson', linestyle='--', linewidth=2)
    axes[1].set_title(f"Residual Plot ({model_name})", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Predicted Final Grade", fontsize=10)
    axes[1].set_ylabel("Residual Error (Actual - Predicted)", fontsize=10)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved regression diagnostic plot to '{output_path}'.")


def plot_confusion_matrix_heatmap(cm_matrix, class_names=['At-Risk (<10)', 'Passing (>=10)'], 
                                  model_name="Logistic Regression", output_path="figures/confusion_matrix.png"):
    """
    Plot 2x2 Confusion Matrix Heatmap with clear numerical and percentage annotations.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    cax = ax.matshow(cm_matrix, cmap='Blues', alpha=0.85)
    fig.colorbar(cax, shrink=0.8)
    
    total = np.sum(cm_matrix)
    for i in range(2):
        for j in range(2):
            count = cm_matrix[i, j]
            pct = (count / total) * 100
            text_color = "white" if count > total / 3.0 else "black"
            ax.text(j, i, f"{count}\n({pct:.1f}%)", ha="center", va="center", 
                    color=text_color, fontsize=12, fontweight='bold')
                    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names, fontsize=10, fontweight='bold')
    ax.set_yticklabels(class_names, fontsize=10, fontweight='bold')
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel("Actual Class", fontsize=11, fontweight='bold')
    plt.title(f"Confusion Matrix: {model_name}", fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved confusion matrix heatmap to '{output_path}'.")


def plot_model_comparison(reg_results, clf_results, output_path="figures/model_comparison.png"):
    """
    Generate side-by-side comparison charts for all evaluated models.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Regression Comparison (MAE and RMSE)
    reg_models = list(reg_results.keys())
    maes = [reg_results[m]['MAE'] for m in reg_models]
    rmses = [reg_results[m]['RMSE'] for m in reg_models]
    
    x = np.arange(len(reg_models))
    width = 0.35
    
    rects1 = axes[0].bar(x - width/2, maes, width, label='MAE (Lower is Better)', color='#4575b4', edgecolor='black')
    rects2 = axes[0].bar(x + width/2, rmses, width, label='RMSE (Lower is Better)', color='#d73027', edgecolor='black')
    
    axes[0].set_title("Regression Models Error Comparison", fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(reg_models, rotation=15, ha='right', fontsize=9.5)
    axes[0].set_ylabel("Error (Grade Points on 0-20 scale)", fontsize=10)
    axes[0].grid(axis='y', linestyle=':', alpha=0.6)
    axes[0].legend()
    
    # Add value annotations
    for rect in rects1:
        h = rect.get_height()
        axes[0].text(rect.get_x() + rect.get_width()/2., h + 0.05, f"{h:.2f}", ha='center', va='bottom', fontsize=8.5)
    for rect in rects2:
        h = rect.get_height()
        axes[0].text(rect.get_x() + rect.get_width()/2., h + 0.05, f"{h:.2f}", ha='center', va='bottom', fontsize=8.5)
        
    # 2. Classification Comparison (Accuracy and F1-Score)
    clf_models = list(clf_results.keys())
    accuracies = [clf_results[m]['accuracy'] * 100 for m in clf_models]
    f1_scores = [clf_results[m]['f1_score'] * 100 for m in clf_models]
    
    x2 = np.arange(len(clf_models))
    rects3 = axes[1].bar(x2 - width/2, accuracies, width, label='Accuracy % (Higher is Better)', color='#1b9e77', edgecolor='black')
    rects4 = axes[1].bar(x2 + width/2, f1_scores, width, label='F1-Score % (Higher is Better)', color='#7570b3', edgecolor='black')
    
    axes[1].set_title("Classification Models Performance Comparison", fontsize=12, fontweight='bold')
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(clf_models, rotation=15, ha='right', fontsize=9.5)
    axes[1].set_ylabel("Percentage (%)", fontsize=10)
    axes[1].set_ylim(0, 110)
    axes[1].grid(axis='y', linestyle=':', alpha=0.6)
    axes[1].legend()
    
    for rect in rects3:
        h = rect.get_height()
        axes[1].text(rect.get_x() + rect.get_width()/2., h + 1.2, f"{h:.1f}%", ha='center', va='bottom', fontsize=8.5)
    for rect in rects4:
        h = rect.get_height()
        axes[1].text(rect.get_x() + rect.get_width()/2., h + 1.2, f"{h:.1f}%", ha='center', va='bottom', fontsize=8.5)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved model comparison chart to '{output_path}'.")

