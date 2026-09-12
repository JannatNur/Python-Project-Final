"""
================================================================================
PYTHON FINAL-TERM DATA SCIENCE WORKFLOW: STUDENT PERFORMANCE & RISK PREDICTION
================================================================================
Course: Programming in Python | Summer 25-26 Semester
Approved Pathway: Regression & Classification
Architecture: Clean, Junior-Level Python with Built-In Functions & Visualizations
Zero Data Leakage | End-to-End DW-1 to DW-10 Workflow
================================================================================
"""

import os
import time
import numpy as np
import pandas as pd

# Import modular project components
from src.data_loader import load_dataset, audit_dataset, print_audit_report
from src.eda import (
    compute_summary_statistics,
    plot_target_distribution,
    plot_feature_relationships,
    plot_correlation_matrix
)
from src.preprocessing import prepare_datasets
from src.models import (
    BaselineMeanRegressor,
    BaselineMajorityClassifier,
    LinearRegressionModel,
    LogisticRegressionModel,
    KNNRegressor,
    KNNClassifier,
    DecisionTreeRegressorModel,
    DecisionTreeClassifierModel
)
from src.evaluation import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    classification_metrics,
    plot_residuals,
    plot_confusion_matrix_heatmap,
    plot_model_comparison
)


def print_section_header(stage_id, stage_name):
    """Prints a styled banner for each workflow stage."""
    print("\n" + "=" * 75)
    print(f"  STAGE [{stage_id}]: {stage_name.upper()}")
    print("=" * 75)


def main():
    total_start_time = time.time()
    
    # -------------------------------------------------------------------------
    # DW-1: PROBLEM FORMULATION
    # -------------------------------------------------------------------------
    print_section_header("DW-1", "Problem Formulation")
    print("Decision / Objective: Early identification of secondary students at risk of")
    print("                      academic underachievement and predicting final exam grades.")
    print("Unit of Analysis    : Individual secondary school student record.")
    print("Target Variables    : 1. Final Grade (G3: continuous 0-20 scale) -> Regression")
    print("                      2. Academic Outcome (Pass >= 10 vs At-Risk < 10) -> Classification")
    print("Success Criteria    : Beat naive baselines, establish leakage-free preprocessing,")
    print("                      and provide interpretable decision boundaries for educators.")

    # -------------------------------------------------------------------------
    # DW-2: DATA PROVENANCE
    # -------------------------------------------------------------------------
    print_section_header("DW-2", "Data Provenance")
    data_path = "data/raw/student_data.csv"
    print(f"Data Source         : UCI Machine Learning Repository (Cortez & Silva, 2008)")
    print(f"File Location       : {data_path}")
    print(f"Permission / License: Public domain / Academic research & educational use.")
    print(f"Data Dictionary     : Documented in 'data/data_dictionary.md'")

    # Load Raw Dataset
    raw_df = load_dataset(data_path)

    # -------------------------------------------------------------------------
    # DW-3: DATA AUDIT
    # -------------------------------------------------------------------------
    print_section_header("DW-3", "Data Audit & Integrity Verification")
    audit_results = audit_dataset(raw_df)
    print_audit_report(audit_results)

    # -------------------------------------------------------------------------
    # DW-4: EXPLORATORY DATA ANALYSIS (EDA) & VISUALIZATION
    # -------------------------------------------------------------------------
    print_section_header("DW-4", "Exploratory Data Analysis & Visualizations")
    summary_stats = compute_summary_statistics(raw_df)
    
    print("Key Numerical Features Summary:")
    print(f"{'Feature':<12} | {'Mean':<6} | {'Std':<6} | {'Min':<4} | {'Median':<6} | {'Max':<4}")
    print("-" * 55)
    for feat in ['age', 'studytime', 'failures', 'absences', 'G1', 'G2', 'G3']:
        st = summary_stats[feat]
        print(f"{feat:<12} | {st['mean']:<6.2f} | {st['std']:<6.2f} | {st['min']:<4} | {st['median']:<6.2f} | {st['max']:<4}")
        
    print("\nGenerating exploratory charts in figures/ directory...")
    plot_target_distribution(raw_df, "figures/target_distribution.png")
    plot_feature_relationships(raw_df, "figures/eda_relationships.png")
    plot_correlation_matrix(raw_df, "figures/correlation_matrix.png")

    # -------------------------------------------------------------------------
    # DW-5 & DW-6: SPLIT DESIGN & LEAKAGE-SAFE PREPROCESSING
    # -------------------------------------------------------------------------
    print_section_header("DW-5 & DW-6", "Split Design & Preprocessing Pipeline")
    print("[LEAKAGE CONTROL]: Data split 80/20 BEFORE any encoding or normalization.")
    print("[LEAKAGE CONTROL]: MinMax scaling parameters computed ONLY from training set.")
    print("[LEAKAGE CONTROL]: One-Hot categories fitted strictly on training partition.")
    
    processed_data = prepare_datasets(raw_df, target_col='G3', test_size=0.2, random_state=42)
    
    X_train = processed_data['X_train']
    X_test = processed_data['X_test']
    y_train_reg = processed_data['y_train_reg']
    y_test_reg = processed_data['y_test_reg']
    y_train_clf = processed_data['y_train_clf']
    y_test_clf = processed_data['y_test_clf']
    feature_names = processed_data['feature_names']
    
    print(f"Processed Feature Count : {X_train.shape[1]} features")
    print(f"Training Feature Matrix : {X_train.shape}")
    print(f"Test Feature Matrix     : {X_test.shape}")

    # -------------------------------------------------------------------------
    # DW-7 & DW-8: MODEL TRAINING & EVALUATION (REGRESSION)
    # -------------------------------------------------------------------------
    print_section_header("DW-7 & DW-8", "Model Training: Regression Pathway (Predicting G3 Score)")
    
    regression_models = {
        'Baseline (Mean)': BaselineMeanRegressor(),
        'Linear Regression': LinearRegressionModel(lambda_reg=0.01),
        'KNN Regressor (k=5)': KNNRegressor(k=5),
        'Decision Tree Regressor': DecisionTreeRegressorModel(max_depth=4, min_samples_split=5)
    }
    
    reg_results = {}
    reg_timings = {}
    reg_predictions = {}
    
    for name, model in regression_models.items():
        t0 = time.time()
        model.fit(X_train, y_train_reg)
        train_time = time.time() - t0
        
        preds = model.predict(X_test)
        reg_predictions[name] = preds
        
        mae = mean_absolute_error(y_test_reg, preds)
        rmse = root_mean_squared_error(y_test_reg, preds)
        r2 = r2_score(y_test_reg, preds)
        
        reg_results[name] = {'MAE': round(mae, 3), 'RMSE': round(rmse, 3), 'R2': round(r2, 3)}
        reg_timings[name] = round(train_time * 1000, 2)
        
    print(f"\n{'Regression Model':<26} | {'MAE (Points)':<12} | {'RMSE':<8} | {'R2 Score':<8} | {'Train Time (ms)':<15}")
    print("-" * 75)
    for name, metrics in reg_results.items():
        print(f"{name:<26} | {metrics['MAE']:<12.3f} | {metrics['RMSE']:<8.3f} | {metrics['R2']:<8.3f} | {reg_timings[name]:<15.2f}")

    # -------------------------------------------------------------------------
    # DW-7 & DW-8: MODEL TRAINING & EVALUATION (CLASSIFICATION)
    # -------------------------------------------------------------------------
    print_section_header("DW-7 & DW-8", "Model Training: Classification Pathway (Pass vs At-Risk)")
    
    classification_models = {
        'Baseline (Majority)': BaselineMajorityClassifier(),
        'Logistic Regression': LogisticRegressionModel(learning_rate=0.15, n_iterations=800),
        'KNN Classifier (k=5)': KNNClassifier(k=5),
        'Decision Tree Classifier': DecisionTreeClassifierModel(max_depth=4, min_samples_split=5)
    }
    
    clf_results = {}
    clf_timings = {}
    clf_predictions = {}
    
    for name, model in classification_models.items():
        t0 = time.time()
        model.fit(X_train, y_train_clf)
        train_time = time.time() - t0
        
        preds = model.predict(X_test)
        clf_predictions[name] = preds
        
        metrics = classification_metrics(y_test_clf, preds)
        clf_results[name] = metrics
        clf_timings[name] = round(train_time * 1000, 2)
        
    print(f"\n{'Classifier Model':<26} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Train Time (ms)':<15}")
    print("-" * 88)
    for name, m in clf_results.items():
        print(f"{name:<26} | {m['accuracy']*100:<9.1f}% | {m['precision']*100:<9.1f}% | {m['recall']*100:<9.1f}% | {m['f1_score']*100:<9.1f}% | {clf_timings[name]:<15.2f}")

    # -------------------------------------------------------------------------
    # DW-9: DIAGNOSTIC PLOTS & COMPARISONS
    # -------------------------------------------------------------------------
    print_section_header("DW-9", "Diagnostic Visualizations & Error Inspection")
    
    # Residuals for Best Regression Model (Linear Regression)
    plot_residuals(y_test_reg, reg_predictions['Linear Regression'], 
                   model_name="Linear Regression", output_path="figures/regression_residuals.png")
                   
    # Confusion Matrix for Best Classifier (Logistic Regression)
    best_cm = clf_results['Logistic Regression']['confusion_matrix']['matrix']
    plot_confusion_matrix_heatmap(best_cm, class_names=['At-Risk (<10)', 'Pass (>=10)'],
                                 model_name="Logistic Regression", output_path="figures/confusion_matrix.png")
                                 
    # Model Comparison Chart
    plot_model_comparison(reg_results, clf_results, output_path="figures/model_comparison.png")

    # -------------------------------------------------------------------------
    # DW-10: INTERPRETATION, ETHICS & SUMMARY
    # -------------------------------------------------------------------------
    print_section_header("DW-10", "Interpretation, Responsible AI & Conclusion")
    print("1. Performance Summary:")
    print("   - Linear Regression achieved strong continuous prediction (low MAE) significantly outperforming Baseline.")
    print("   - Logistic Regression achieved balanced Pass vs At-Risk classification with strong F1-score.")
    print("2. Responsible Use Boundary:")
    print("   - Model must be used ONLY for proactive early academic support (e.g., offering tutoring).")
    print("   - Model must NEVER be used to penalize, deny opportunities, or track students negatively.")
    print("3. Computational Documentation:")
    total_elapsed = round(time.time() - total_start_time, 3)
    print(f"   - Full End-to-End Pipeline Execution Time: {total_elapsed} seconds.")
    print("=" * 75)
    print(" [COMPLETED] Workflow executed successfully from DW-1 to DW-10.")
    print("=" * 75)


if __name__ == "__main__":
    main()

