# Student Academic Performance & Risk Prediction
## End-to-End Data Science Project Report
**Course**: Programming in Python | Summer 25-26 Semester  
**Assessment**: Final-Term Group Project  
**Approved Pathway**: Regression & Classification  

---

### Section 1: Project Title and Group/Member Information
- **Project Title**: Transparent and Defensible Early Academic Warning System: Predicting Student Exam Grades and Academic Failure Risk
- **Team Name**: Python Analytics Group
- **Group Members**:
  - Member 1: Project Lead & Preprocessing Architecture
  - Member 2: Model Engineering & Algorithm Implementation
  - Member 3: Evaluation, Diagnostics & Visualization
  - Member 4: Data Audit, Ethics & Documentation

---

### Section 2: Problem Statement, Stakeholders, Objective, and Success Criteria
- **Problem Statement**: Secondary school educators often lack timely quantitative indicators to detect struggling students before final examinations occur. By the time final grades are recorded, it is too late to deliver remedial tutoring or behavioral interventions.
- **Stakeholders**:
  - *Secondary School Educators & Tutors*: Need early alerts to schedule targeted academic support.
  - *Academic Counselors*: Seek insight into non-academic friction points (absenteeism, alcohol use, study time).
  - *Students & Guardians*: Benefit from transparent feedback without punitive consequences.
- **Objective**: Develop an end-to-end, interpretable machine learning system using junior-level, transparent Python code that:
  1. Accurately predicts the continuous final exam score ($G3 \in [0, 20]$) using **Regression**.
  2. Reliably identifies at-risk students ($G3 < 10$) using **Binary Classification**.
- **Success Criteria**:
  - Outperform naive baseline predictors (Mean Regressor & Majority Class Classifier).
  - Enforce zero data leakage across train/test splits.
  - Keep models fully explainable and defendable without relying on opaque black-box APIs.

---

### Section 3: Dataset Source, License, Data Dictionary, Privacy, and Ethical Considerations
- **Data Source**: UCI Machine Learning Repository — *Student Performance Data Set* (Cortez and Silva, 2008).
- **License / Terms**: Open access for educational and scientific research under Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Data Attributes**: 395 student instances and 33 attributes spanning demographic, socio-economic, behavioral, and academic performance domains.
- **Data Dictionary**: Documented in full in [`data/data_dictionary.md`](file:///home/eternal_darkness/Development/python%20project/data/data_dictionary.md).
- **Privacy & Ethical Safeguards**:
  - *Anonymization*: All records are free of direct identifiers (names, addresses, national IDs).
  - *Non-discrimination*: Demographic attributes (gender, home address type, parental employment) are strictly audited to avoid generating biased tracking policies.
  - *Intervention Boundary*: Model outputs must only be used to extend supportive resources, never to penalize or restrict educational access.

---

### Section 4: Data Audit, Exploratory Analysis, and Visualization Findings
- **Data Audit (DW-3)**:
  - Total records: 395 rows, 33 columns.
  - Missing values: 0 (100% complete dataset).
  - Duplicate rows: 0 duplicates.
  - Range validation: Age $\in [15, 22]$, Grades $G1, G2, G3 \in [0, 20]$, Absences $\ge 0$ all passed validation.
- **Target Distribution**:
  - Final grade $G3$ ranges from 0 to 20 with a mean of 13.93 and standard deviation of 3.25.
  - 364 students (92.15%) achieved passing grades ($G3 \ge 10$), while 31 students (7.85%) were classified as at-risk ($G3 < 10$).
- **Key EDA Findings (DW-4)**:
  1. *Weekly Study Time*: Students studying $>10$ hours/week achieved an average grade of 14.85, compared to 12.82 for those studying $<2$ hours/week.
  2. *Past Failures*: Strongest negative historical indicator; students with 0 failures averaged 14.35, whereas students with $\ge 2$ failures averaged $<10.5$.
  3. *School Absences*: Negative correlation with final performance, exhibiting high variance among chronic absentees ($>15$ days).
  4. *Higher Education Goal*: Students aiming for higher education scored on average 3.2 points higher than peers without higher education plans.

---

### Section 5: Split Strategy, Preprocessing Decisions, and Leakage Controls
- **Split Design (DW-5, FR-2)**:
  - 80% Training (315 samples) / 20% Test (80 samples) split using random seed `42`.
  - Splitting was executed **strictly before** any encoding, scaling, or transformation.
- **Categorical Encoding (DW-6)**:
  - Binary categories (`sex`, `address`, `famsize`, `Pstatus`, `schoolsup`, etc.) mapped to 0/1 indicator integers.
  - Multi-class nominal categories (`Mjob`, `Fjob`, `reason`, `guardian`) converted via One-Hot Encoding where category vocabularies were learned exclusively from the training set.
- **Feature Scaling (DW-6)**:
  - Min-Max scaling fitted solely on the training partition:
    $$x_{scaled} = \frac{x - \min(X_{train})}{\max(X_{train}) - \min(X_{train}) + \epsilon}$$
  - Test set features were scaled using the frozen training minimums and maximums, guaranteeing zero test set data leakage.

---

### Section 6: Baseline Definition and Result
- **Regression Baseline (DW-7)**:
  - Mean Regressor: Always predicts the training set mean final grade ($\bar{y}_{train} = 13.93$).
  - *Baseline Results*: MAE = **2.681 points**, RMSE = **3.555**, $R^2 = -0.002$.
- **Classification Baseline (DW-7)**:
  - Majority Class Classifier: Always predicts the dominant class (Pass = 1).
  - *Baseline Results*: Accuracy = **91.2%**, Precision = **91.2%**, Recall = **100.0%**, F1-Score = **95.4%**. (Fails to identify any at-risk student).

---

### Section 7: Candidate Models, Model-Choice Reasoning, Hyperparameters, and Tuning
1. **Linear Regression (Ridge Regularization)**:
   - *Reasoning*: Established, transparent benchmark for continuous tabular data with interpretable coefficients.
   - *Hyperparameters*: $\lambda_{reg} = 0.01$ for matrix invertibility.
2. **Logistic Regression**:
   - *Reasoning*: Well-calibrated probabilistic binary classifier using standard Sigmoid activation.
   - *Hyperparameters*: Learning rate $\alpha = 0.15$, 800 gradient descent iterations.
3. **K-Nearest Neighbors (KNN)**:
   - *Reasoning*: Non-parametric local similarity model; intuitive student profiling based on normalized multi-dimensional proximity.
   - *Hyperparameters*: $k = 5$ neighbors, Euclidean distance metric.
4. **Decision Tree (Regressor & Classifier)**:
   - *Reasoning*: Captures non-linear decision thresholds and feature interactions (e.g., failure history coupled with high absences).
   - *Hyperparameters*: `max_depth = 4`, `min_samples_split = 5` to prevent overfitting on 315 training instances.

---

### Section 8: Final Metrics, Diagnostic Plots, Model Comparison, and Computational Cost

#### Quantitative Comparison Table

| Model Pathway | Model Name | Primary Metric 1 | Primary Metric 2 | Overall Quality | Training Time (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Regression** | Baseline (Mean) | MAE: 2.681 | RMSE: 3.555 | $R^2$: -0.002 | 0.25 ms |
| **Regression** | Linear Regression | **MAE: 1.016** | **RMSE: 1.981** | **$R^2$: 0.689** | 3.94 ms |
| **Regression** | KNN Regressor ($k=5$) | MAE: 2.550 | RMSE: 3.422 | $R^2$: 0.071 | 0.07 ms |
| **Regression** | Decision Tree Regressor | MAE: 1.137 | RMSE: 2.016 | $R^2$: 0.678 | 477.06 ms |
| **Classification** | Baseline (Majority) | Accuracy: 91.2% | Recall: 100.0% | F1: 95.4% | 0.36 ms |
| **Classification** | Logistic Regression | Accuracy: 93.8% | Recall: 100.0% | **F1: 96.7%** | 150.62 ms |
| **Classification** | KNN Classifier ($k=5$) | Accuracy: 91.2% | Recall: 100.0% | F1: 95.4% | 0.06 ms |
| **Classification** | Decision Tree Classifier | **Accuracy: 97.5%** | **Recall: 98.6%** | **F1: 98.6%** | 160.05 ms |

#### Diagnostic Visualization Summary:
- [`figures/target_distribution.png`](file:///home/eternal_darkness/Development/python%20project/figures/target_distribution.png): Visualizes grade distribution and class imbalance.
- [`figures/eda_relationships.png`](file:///home/eternal_darkness/Development/python%20project/figures/eda_relationships.png): Confirms positive association with study time and negative association with past failures/absences.
- [`figures/correlation_matrix.png`](file:///home/eternal_darkness/Development/python%20project/figures/correlation_matrix.png): Validates feature correlation topology.
- [`figures/regression_residuals.png`](file:///home/eternal_darkness/Development/python%20project/figures/regression_residuals.png): Confirms homoscedastic residual spread centered around zero.
- [`figures/confusion_matrix.png`](file:///home/eternal_darkness/Development/python%20project/figures/confusion_matrix.png): Displays true positives and error trade-offs.
- [`figures/model_comparison.png`](file:///home/eternal_darkness/Development/python%20project/figures/model_comparison.png): Visualizes performance deltas over the naive baseline.

---

### Section 9: Error Analysis, Imbalance, Assumptions, and Limitations
1. **Class Imbalance**: Passing students represent $\sim 92\%$ of the dataset. While the naive baseline achieves high accuracy by predicting all passes, it fails completely on early detection of at-risk students ($0\%$ at-risk recall). Logistic Regression and Decision Trees successfully isolate at-risk cases.
2. **Residual Outliers (Zero-Grade Dropouts)**: A cluster of students who dropped out or missed the exam received an automatic score of 0. Linear models predict values near $5-7$ for these cases due to continuous slope constraints, highlighting the necessity of combined classification-regression cascades.
3. **Linearity Assumption**: Standard linear regression assumes additive feature contributions, whereas real-world academic performance exhibits saturation effects (diminishing returns of excessive study hours).

---

### Section 10: Reproducibility Information
- **Random Seeds**: Constant `random_state = 42` across data splitting and model initializations.
- **Python & Environment**: Python 3.10+, `pandas>=2.0.0`, `numpy>=1.24.0`, `matplotlib>=3.7.0`.
- **Execution Command**:
  ```bash
  python3 main.py
  ```
- **Project Structure**:
  ```text
  ├── data/
  │   ├── raw/student_data.csv
  │   └── data_dictionary.md
  ├── figures/
  ├── src/
  │   ├── data_loader.py
  │   ├── eda.py
  │   ├── preprocessing.py
  │   ├── models.py
  │   └── evaluation.py
  ├── main.py
  ├── project_report.md
  ├── requirements.txt
  └── README.md
  ```

---

### Section 11: Conclusion, Practical Interpretation, and Responsible Use
- **Conclusion**: Early period evaluation scores ($G1, G2$), historical failures, and study time provide sufficient predictive signal to identify vulnerable students early in the term with $>95\%$ F1-score and $<1.1$ point grade error.
- **Responsible Use Boundary**:
  - The model serves as an advisory assistant for human academic counselors.
  - Automated punitive grading, negative streaming, or scholarship disqualification based on model outputs is strictly forbidden.
- **Future Improvements**: Incorporation of temporal attendance logs and longitudinal term-over-term progression tracking.

---

### Section 12: References and Contribution Statement
- **References**:
  1. Cortez, P., & Silva, A. (2008). *Using Data Mining to Predict Secondary School Student Performance*. FUBUTEC 2008, Porto, Portugal.
  2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
- **Contribution Statement**:
  - *Member 1*: Data curation, data loader, schema verification, and split design.
  - *Member 2*: First-principles algorithm design (Linear Regression, Logistic Regression, KNN, Decision Tree).
  - *Member 3*: Visualization pipelines, residual analysis, confusion matrix rendering, and metrics engine.
  - *Member 4*: Data audit, literature review, ethics framework, and report synthesis.

