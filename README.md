This is a comprehensive set of results. To make this an effective **README.md** for GitHub or a portfolio, I have restructured your content into a professional, navigable format. I've added a **Table of Contents**, used clear **Markdown syntax**, and organized the technical deep-dive into logical blocks.

---

# 📈 Teen Mental Health: Machine Learning Forecasting

**Analyst:** Husseim Stuck

**Date:** 2024

**Repository:** [Project_Teen_Mental_Health_ML](https://github.com/Husseim-Stuck-Portfolio/Project_Teen_Mental_Health_machine_learning_forecasting)

---

## 📖 Table of Contents

1. [Executive Summary](https://www.google.com/search?q=%23executive-summary)
2. [Dataset Overview](https://www.google.com/search?q=%231-dataset-overview)
3. [Exploratory Data Analysis (EDA)](https://www.google.com/search?q=%232-exploratory-data-analysis-eda-findings)
4. [Regression Analysis]()
5. [Classification Analysis]()
6. [Feature Importance]()
7. [Hyperparameter Tuning]()
8. [Pipeline Architecture]()
9. [Gender Equity Analysis]()
10. [Key Takeaways & Recommendations]()

---

## Executive Summary

This project forecasts adolescent mental health outcomes—including **stress, anxiety, addiction**, and **depression**—using a dataset of 1,200 records. By leveraging both regression and classification algorithms, the study identifies that **daily social media usage, academic performance, and sleep patterns** are the primary predictors of mental health stability in teenagers.

---

## 1. Dataset Overview

### Column Definitions

| Category | Feature | Description |
| --- | --- | --- |
| **Demographics** | `age` | Participant age (13–19) |
|  | `gender` | Male or Female |
| **Usage** | `daily_social_media_hours` | Avg. daily hours on platforms |
|  | `platform_usage` | Primary platform (Instagram, TikTok, etc.) |
| **Lifestyle** | `sleep_hours` | Avg. nightly sleep duration |
|  | `screen_time_before_sleep` | Hours of screen use before bed |
|  | `physical_activity` | Weekly physical activity level |
| **Academic** | `academic_performance` | Numeric score (GPA equivalent) |
| **Mental Health** | `stress_level` | Self-reported (1–10) |
| (Targets) | `anxiety_level` | Self-reported (1–10) |
|  | `addiction_level` | Perceived dependency (1–10) |
|  | `depression_label` | Binary (0: No, 1: Yes) |

> **Note on Class Imbalance:** The `depression_label` is highly imbalanced (97.4% Negative / 2.6% Positive). This required **Stratified Splitting** and **Balanced Class Weights** during modeling.

---

## 2. Exploratory Data Analysis (EDA) Findings

* **ANOVA (Stress vs. Social Interaction):** $p$-value = 0.698. No significant difference in stress levels across social interaction groups.
* **Chi-Square (Depression vs. Interaction):** $p$-value = 0.796. Depression prevalence is independent of reported social interaction levels.
* **Correlation Trends:**
* **Negative:** Sleep Hours ↔ Stress/Anxiety.
* **Positive:** Screen Time ↔ All mental health metrics.



---

## 3. Regression Analysis: Continuous Outcomes

We modeled **Stress, Anxiety, and Addiction** (Scales 1–10).

### Model Performance (MSE)

| Target | Linear Regression (LR) | Random Forest (RF) | Best Model |
| --- | --- | --- | --- |
| **Stress Level** | **7.948** | 8.595 | **LR** |
| **Anxiety Level** | **8.432** | 8.900 | **LR** |
| **Addiction Level** | **7.774** | 8.347 | **LR** |

**Insight:** Linear Regression consistently outperformed Random Forest for continuous targets, suggesting these relationships are primarily linear.

---

## 4. Classification Analysis: Depression

Given the 2.6% positive class rate, we prioritized **AUC-ROC** over accuracy.

| Model | AUC Score | Performance |
| --- | --- | --- |
| **Logistic Regression** | 0.851 | Good |
| **Random Forest Classifier** | **0.943** | **Excellent** |

**Why Random Forest?** It captures non-linear interactions (e.g., the combined "toxic" effect of low sleep + high screen time) that a linear model might miss.

---

## 5. Feature Importance Analysis

Which factors "drive" the models?

1. **Daily Social Media Hours (0.187)**: The strongest predictor.
2. **Academic Performance (0.169)**: School-related stress is a major factor.
3. **Sleep Hours (0.168)**: Essential for emotional regulation.
4. **Physical Activity (0.132)**: Acts as a protective buffer.

---

## 6. Hyperparameter Tuning & Model Optimization

Using `RandomizedSearchCV`, we optimized the Random Forest Regressor for Stress Level:

* **Optimal Config:** `max_depth: 5`, `n_estimators: 121`, `max_features: 'sqrt'`.
* **Results:** MAE of **2.435** on a 10-point scale (~24% error).

---

## 7. Data Preprocessing & Pipeline Architecture

To ensure reproducibility and prevent data leakage, we utilized a `ColumnTransformer` pipeline:

1. **Numeric Features:** Passed through (Age, Hours, GPA).
2. **Categorical Features:** `OneHotEncoder` (Gender, Platform, Interaction Level).
3. **Split:** 80% Train / 20% Test (Stratified).

---

## 8. Gender Analysis: Equity & Fairness

Statistical testing (t-tests and Chi-square) confirmed **no significant difference** between genders regarding mental health outcomes in this dataset.

* **Stress $p$-value:** 0.784
* **Anxiety $p$-value:** 0.551
* **Depression $p$-value:** 0.613

**Conclusion:** Mental health challenges appear to affect adolescents regardless of gender identity within this population.

---

## 9. Recommendations for Future Work

* **Feature Engineering:** Create interaction terms like `Social Media * Screen Time`.
* **Longitudinal Study:** Collect data over time to determine **causality** (e.g., Does social media *cause* depression, or do depressed teens use social media *more*?).
* **SHAP Implementation:** Add SHAP values to explain individual "at-risk" predictions for clinical transparency.

---

## 🛠️ Usage

```python
# To replicate the classification model:
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)
rf_clf.fit(X_train, y_train)

```

---

Source:
https://www.kaggle.com/datasets/umerhaddii/european-flights-dataset/data

**Contact:** [Husseim Stuck]()

*Data Source: Based on adolescent behavioral research datasets.*