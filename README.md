# Industrial Equipment Failure Forecasting
**A Machine Learning & Predictive Maintenance Pipeline**

---

### Project Metadata
- **Project Title:** Industrial Equipment Failure Forecasting

- **Dataset:** [AI4I 2020 Predictive Maintenance Dataset (10,000 records)](https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv)
- **Author:**  Debayan Panda


---

## Abstract
Industrial equipment failures can interrupt production, increase maintenance costs, and create avoidable downtime. This project builds a machine-learning workflow to estimate whether a machine is likely to fail, using operating measurements such as air and process temperature, rotational speed, torque, and tool wear. 

Using the **AI4I 2020 Predictive Maintenance dataset** ([Source Link](https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv)), this project performs rigorous data validation, exploratory data analysis, and domain-specific feature engineering, subsequently evaluating a class-weighted **Logistic Regression baseline** and a non-linear **Multi-Layer Perceptron (MLP) classifier**. Because failures are rare (3.39% of dataset records), the evaluation focuses on **ROC-AUC** and a **cost-sensitive financial analysis** rather than relying on accuracy alone. The notebook reports a 5-fold mean ROC-AUC of **0.924** for Logistic Regression and a held-out test ROC-AUC of **0.971** for the MLP classifier. Furthermore, probability-threshold optimization reduced hypothetical operational failure costs by ~27.4%, while permutation feature importance highlighted torque, temperature difference, and tool wear as key failure drivers.

---

## 1. Project at a Glance

| Item | Project Details |
| :--- | :--- |
| **Problem** | Forecasting machine failure from sensor and operating telemetry data |
| **Dataset** | [AI4I 2020 Predictive Maintenance Dataset (10,000 records)](https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv) |
| **Tools & Libraries** | Python, Google Colab, pandas, NumPy, Matplotlib, scikit-learn |
| **Machine Learning Models** | Logistic Regression, Multi-Layer Perceptron (MLP Classifier) |
| **Key Evaluation Metrics** | ROC-AUC, Class-wise Precision / Recall / F1-Score, Custom Financial Penalty |
| **Main Reported Result** | **MLP Test ROC-AUC: 0.971** (Best Tested Decision Threshold: 0.1) |

---

## 2. Introduction & Motivation
Predictive maintenance uses equipment sensor telemetry to identify patterns that precede machinery breakdowns. Instead of depending exclusively on fixed maintenance schedules (which can cause unnecessary downtime) or reacting post-failure (which risks severe operational disruption and equipment damage), a predictive system empowers maintenance teams to prioritize high-risk inspections. Machine learning models excel in this domain because failure patterns frequently involve multi-variable interactions—such as thermal differentials and speed-torque products—rather than an isolated warning signal.

---

## 3. Project Objectives
- **Data Preparation & Inspection:** Clean tabular equipment dataset, execute missing-value checks, drop non-informative identifiers (`UDI`, `Product ID`), and apply one-hot encoding.
- **Class Imbalance Profiling:** Explore and quantify the severe imbalance between normal operating conditions and failure records.
- **Feature Engineering:** Construct domain-derived operational features (`Temperature_Difference` and `Power_Estimate`).
- **Supervised Model Training:** Train class-weighted Logistic Regression and MLP Neural Network architectures using cross-validation and standard scaling.
- **Cost-Sensitive Decision Optimization:** Interpret model feature importance via permutation importance and evaluate decision thresholds to minimize custom operational financial risk penalties.

---

## 4. Workflow & Methodology

| Stage | Process Performed | Engineering Rationale |
| :--- | :--- | :--- |
| **1. Load & Inspect** | Read CSV ([Source Link](https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv)) with `pandas`; remove `UDI` & `Product ID`. | Keep focus strictly on operating telemetry and failure labels. |
| **2. Check & Encode** | Verify 0 missing values; one-hot encode `Type` (`L/M/H`). | Prepare categorical machinery variants for linear/neural models. |
| **3. Target Profiling** | Calculate distribution and percentages of target classes. | Quantify extreme class imbalance prior to modelling. |
| **4. Feature Engineering** | Create `Temperature_Difference` & `Power_Estimate`. | Capture thermal accumulation and mechanical stress relationships. |
| **5. Outlier Detection** | Compute 1.5 × IQR bounds on rotational speed (418 outliers). | Identify statistical anomalies without discarding valid high-stress operating states. |
| **6. Predictor Selection** | Pearson correlation analysis; exclude target & failure modes. | Prevent data leakage from specific failure mode labels (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`). |
| **7. Split & Scale** | 80/20 train-test split (`random_state=42`); fit `StandardScaler`. | Isolate test set completely from scaler transformations. |
| **8. Logistic Baseline** | Class-weighted fit; 5-fold CV tuning over $C \in [0.01, 0.1, 1, 10]$. | Establish linear baseline and mitigate class imbalance. |
| **9. MLP Classifier** | Hidden layers `(64, 32)`, ReLU activation, Adam solver (max 500 iter). | Evaluate complex non-linear feature interactions. |
| **10. Cost & Importance** | Test decision thresholds (0.1–0.9); calculate permutation importance. | Align decision-making with business costs and identify top predictor inputs. |

> **Important Implementation Note:** The notebook strictly excludes failure-mode columns (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) from model predictors to avoid data leakage, ensuring models predict failures purely from telemetry inputs.

---

## 5. Data Analysis & Results

### 5.1 Dataset Balance
The dataset ([Source Link](https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv)) consists of **10,000 records**, with only **339 recorded failures** (3.39%). Accuracy is an inadequate metric here, as a trivial model predicting "No Failure" continuously would achieve **96.61% accuracy** while missing 100% of breakdowns.

| Target Class | Records | Share (%) |
| :--- | :---: | :---: |
| **Normal Operation (0)** | 9,661 | 96.61% |
| **Machine Failure (1)** | 339 | 3.39% |
| **Total** | **10,000** | **100.00%** |

### 5.2 Logistic Regression Baseline Performance
Using `class_weight='balanced'` and `StandardScaler`, grid search selected **$C = 10$**.
- **5-Fold Cross-Validation ROC-AUC:** `0.92395`

**Held-out Test Set Classification Metrics:**
| Metric (Test Set) | Normal Operation (Class 0) | Machine Failure (Class 1) |
| :--- | :---: | :---: |
| **Precision** | 0.99 | 0.15 |
| **Recall** | 0.86 | 0.79 |
| **F1-Score** | 0.92 | 0.26 |
| **Support** | 1,939 | 61 |

- *Takeaway:* The baseline captures 79% of true failures but produces a high false-alarm rate (Precision = 0.15).

### 5.3 Multi-Layer Perceptron (MLP) Classifier
The non-linear MLP architecture achieved superior overall ranking ability across operating thresholds.
- **Held-out Test Set ROC-AUC:** **`0.97117`**

### 5.4 Cost-Sensitive Threshold Analysis
To reflect real-world operational trade-offs, a custom business penalty model was applied:
$$\text{Total Cost} = (\text{False Negatives} \times \$10,000) + (\text{False Positives} \times \$500)$$

- **Default MLP Predictions (Threshold = 0.5):** $\$235,000$ penalty
- **Optimized Decision Setting (Threshold = 0.1):** **$\$170,500$ penalty**
- **Financial Savings:** **$\$64,500$ reduction (~27.4% decrease in operational loss)**

### 5.5 Feature Importance (Permutation Importance on Test Set)
Permutation importance measures the drop in test ROC-AUC when feature values are shuffled:

| Rank | Feature | Mean Permutation Importance |
| :---: | :--- | :---: |
| **1** | **Torque [Nm]** | **0.406784** |
| **2** | **Temperature_Difference** | **0.134037** |
| **3** | **Tool wear [min]** | **0.133848** |

---

## 6. Conclusion & Future Work
This project demonstrates an end-to-end, cost-aware predictive maintenance pipeline. By engineering operational domain features and training an MLP classifier, the project achieved a test **ROC-AUC of 0.971**. Optimizing the decision threshold to `0.1` successfully reduced simulated factory breakdown costs by **27.4%**, proving that threshold selection paired with cost modeling is critical when handling imbalanced industrial failure data.

### Future Enhancements:
- **Advanced Ensembles:** Evaluate XGBoost, LightGBM, and Random Forest architectures.
- **Time-Series Sequence Modeling:** Apply LSTM / GRU networks to capture temporal sequence dynamics.
- **Dynamic Cost Modeling:** Integrate variable downtime cost models linked to specific machine types.

---

## 7. References & Appendices
- [UCI Machine Learning Repository — AI4I 2020 Predictive Maintenance Dataset (CSV Download Link)](https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv)
- [UCI Machine Learning Repository — AI4I 2020 Dataset Documentation](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset)
- [Scikit-Learn Documentation — Model Evaluation & ROC-AUC](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Scikit-Learn Documentation — Multi-layer Perc
