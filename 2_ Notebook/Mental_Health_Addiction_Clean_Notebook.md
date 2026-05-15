```python

# ============================================================
# ===== IMPORT ALL LIBRARIES AND TESTS =====
# ============================================================

```


```python
import pandas as pd
import numpy as np

from scipy.stats import f_oneway, chi2_contingency, randint

import plotly.express as px
import plotly.graph_objects as go

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (train_test_split, RandomizedSearchCV, cross_val_score)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

import warnings
warnings.filterwarnings('ignore')
 
# Load data
df = pd.read_csv('../1_RawData/Teen_Mental_Health_Dataset.csv')
print("Dataset shape:", df.shape)
print("\nColumn types:")
print(df.dtypes)
print("\nDepression label distribution:")
print(df['depression_label'].value_counts(normalize=True))

# Define predictors and targets
X = df.drop(['stress_level', 'anxiety_level', 'addiction_level', 'depression_label'], axis=1)
num_cols = [
    'age',
    'daily_social_media_hours',
    'sleep_hours', 
    'screen_time_before_sleep',
    'academic_performance',
    'physical_activity'
]
cat_cols = ['gender', 'platform_usage', 'social_interaction_level']

# Preprocessor is a scikit-learn utility that automatically handles mixed data types (numeric + categorical) before feeding them into machine learning models. 
preprocessor = ColumnTransformer([
    ('num', 'passthrough', num_cols),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
])
```

    Dataset shape: (1200, 13)
    
    Column types:
    age                           int64
    gender                       object
    daily_social_media_hours    float64
    platform_usage               object
    sleep_hours                 float64
    screen_time_before_sleep    float64
    academic_performance        float64
    physical_activity           float64
    social_interaction_level     object
    stress_level                  int64
    anxiety_level                 int64
    addiction_level               int64
    depression_label              int64
    dtype: object
    
    Depression label distribution:
    depression_label
    0    0.974167
    1    0.025833
    Name: proportion, dtype: float64



```python
# ============================================================
# REGRESIION VISUALIZATION (Stress, Anxiety, Addiction, Depression) 
# ============================================================
```


```python
# ===== 1. EXPLORATORY ANALYSIS =====
print("\n" + "="*50)
print("1. EXPLORATORY ANALYSIS")
print("="*50)

# Correlation heatmap
num_analysis = num_cols + ['stress_level', 'anxiety_level', 'addiction_level']
corr = df[num_analysis].corr()
fig_corr = px.imshow(corr, color_continuous_scale='RdBu_r', title='Correlation Heatmap: Key Variables')
fig_corr.update_layout(width=800, height=600)
fig_corr.show()

# ANOVA: stress by social interaction
groups = [df[df['social_interaction_level'] == cat]['stress_level'].values 
          for cat in df['social_interaction_level'].unique()]
f_stat, p_anova = f_oneway(*groups)
print(f"ANOVA stress by social_interaction_level: F={f_stat:.3f}, p={p_anova:.3f}")

# Chi-square: depression vs social interaction
ct = pd.crosstab(df['social_interaction_level'], df['depression_label'])
chi2, p_chi, _, _ = chi2_contingency(ct)
print(f"Chi-square depression vs social_interaction: chi2={chi2:.3f}, p={p_chi:.3f}")

```

    
    ==================================================
    1. EXPLORATORY ANALYSIS
    ==================================================



    
![png](Mental_Health_Addiction_Clean_Notebook_files/Mental_Health_Addiction_Clean_Notebook_3_1.png)
    


    ANOVA stress by social_interaction_level: F=0.359, p=0.698
    Chi-square depression vs social_interaction: chi2=0.457, p=0.796



```python
# ===== 2. REGRESSION FOR CONTINUOUS TARGETS =====
print("\n" + "="*50)
print("2. REGRESSION (Stress, Anxiety, Addiction, Depression)")
print("="*50)

targets_reg = ['stress_level', 'anxiety_level', 'addiction_level']
results_reg = {}

for target in targets_reg:
    print(f"\n--- {target.upper()} ---")
    y = df[target]
    
    # Split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Linear Regression
    pipe_lr = Pipeline([('prep', preprocessor), ('model', LinearRegression())])
    pipe_lr.fit(X_tr, y_tr)
    pred_lr = pipe_lr.predict(X_te)
    mse_lr = mean_squared_error(y_te, pred_lr)
    
    # Random Forest
    pipe_rf = Pipeline([('prep', preprocessor), ('model', RandomForestRegressor(n_estimators=100, random_state=42))])
    pipe_rf.fit(X_tr, y_tr)
    pred_rf = pipe_rf.predict(X_te)
    mse_rf = mean_squared_error(y_te, pred_rf)
    
    results_reg[target] = {'LR_MSE': mse_lr, 'RF_MSE': mse_rf}
    
    print(f"Linear Regression MSE: {mse_lr:.3f}")
    print(f"Random Forest MSE: {mse_rf:.3f}")
    
    # Feature importance (top 10)
    imp = pd.Series(pipe_rf.named_steps['model'].feature_importances_,
                   index=pipe_rf.named_steps['prep'].get_feature_names_out())
    imp_top10 = imp.sort_values(ascending=False).head(10)
    fig_imp = px.bar(x=imp_top10.index, y=imp_top10.values, 
                     title=f'{target.replace("_level", "").title()} - Top 10 Predictors (RF)')
    fig_imp.update_layout(xaxis_tickangle=45, width=900, height=500)
    fig_imp.show()

# ===== 3. CLASSIFICATION FOR DEPRESSION =====
print("\n" + "="*50)
print("3. CLASSIFICATION (Depression)")
print("="*50)

y_dep = df['depression_label']
X_tr_d, X_te_d, y_tr_d, y_te_d = train_test_split(X, y_dep, test_size=0.2, 
                                                  random_state=42, stratify=y_dep)

# Logistic Regression
pipe_log = Pipeline([
    ('prep', preprocessor), 
    ('model', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])
pipe_log.fit(X_tr_d, y_tr_d)
pred_proba_log = pipe_log.predict_proba(X_te_d)[:, 1]
auc_log = roc_auc_score(y_te_d, pred_proba_log)

# Random Forest Classifier
pipe_rf_cls = Pipeline([
    ('prep', preprocessor),
    ('model', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
])
pipe_rf_cls.fit(X_tr_d, y_tr_d)
pred_proba_rf = pipe_rf_cls.predict_proba(X_te_d)[:, 1]
auc_rf = roc_auc_score(y_te_d, pred_proba_rf)

print(f"Logistic Regression AUC: {auc_log:.3f}")
print(f"Random Forest AUC: {auc_rf:.3f}")

# Depression feature importance
imp_dep = pd.Series(pipe_rf_cls.named_steps['model'].feature_importances_,
                   index=pipe_rf_cls.named_steps['prep'].get_feature_names_out())
imp_dep_top10 = imp_dep.sort_values(ascending=False).head(10)
fig_dep_imp = px.bar(x=imp_dep_top10.index, y=imp_dep_top10.values,
                     title='Depression Label - Top 10 Predictors (RF)')
fig_dep_imp.update_layout(xaxis_tickangle=45, width=900, height=500)
fig_dep_imp.show()


```

    
    ==================================================
    2. REGRESSION (Stress, Anxiety, Addiction, Depression)
    ==================================================
    
    --- STRESS_LEVEL ---
    Linear Regression MSE: 7.948
    Random Forest MSE: 8.595



    
![png](Mental_Health_Addiction_Clean_Notebook_files/Mental_Health_Addiction_Clean_Notebook_4_1.png)
    


    
    --- ANXIETY_LEVEL ---
    Linear Regression MSE: 8.432
    Random Forest MSE: 8.900



    
![png](Mental_Health_Addiction_Clean_Notebook_files/Mental_Health_Addiction_Clean_Notebook_4_3.png)
    


    
    --- ADDICTION_LEVEL ---
    Linear Regression MSE: 7.774
    Random Forest MSE: 8.347



    
![png](Mental_Health_Addiction_Clean_Notebook_files/Mental_Health_Addiction_Clean_Notebook_4_5.png)
    


    
    ==================================================
    3. CLASSIFICATION (Depression)
    ==================================================
    Logistic Regression AUC: 0.920
    Random Forest AUC: 0.917



    
![png](Mental_Health_Addiction_Clean_Notebook_files/Mental_Health_Addiction_Clean_Notebook_4_7.png)
    



```python

```


```python

```


```python
# ===== 4. SUMMARY RESULTS =====
print("\n" + "="*50)
print("4. SUMMARY RESULTS")
print("="*50)

summary_df = pd.DataFrame(results_reg).T
print("\nRegression MSE Results:")
print(summary_df.round(3))

print("\nDepression Classification AUC:")
print(f"Logistic: {auc_log:.3f}, Random Forest: {auc_rf:.3f}")
```

    
    ==================================================
    4. SUMMARY RESULTS
    ==================================================
    
    Regression MSE Results:
                     LR_MSE  RF_MSE
    stress_level      7.948   8.595
    anxiety_level     8.432   8.900
    addiction_level   7.774   8.347
    
    Depression Classification AUC:
    Logistic: 0.920, Random Forest: 0.917



```python
# ============================================================
# STEP 1: CHECK DATA
# ============================================================
print("STEP 1: DATA LOADED")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# Make sure this matches your real dataset
target_stress = "stress_level"

if target_stress not in df.columns:
    raise ValueError(f"Column '{target_stress}' not found. Available columns: {df.columns.tolist()}")

print("\nTarget summary:")
print(df[target_stress].describe())


# ============================================================
# STEP 2: DEFINE FEATURES AND TARGET
# ============================================================
feature_cols = [
    "age",
    "gender",
    "daily_social_media_hours",
    "platform_usage",
    "sleep_hours",
    "screen_time_before_sleep",
    "academic_performance",
    "physical_activity",
    "social_interaction_level"
]

missing_features = [col for col in feature_cols if col not in df.columns]
if missing_features:
    raise ValueError(f"Missing feature columns: {missing_features}")

X = df[feature_cols]
y = df[target_stress]

print("\nSTEP 2: FEATURES AND TARGET READY")
print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# STEP 3: IDENTIFY NUMERIC AND CATEGORICAL COLUMNS
# ============================================================
numeric_features = [
    "age",
    "daily_social_media_hours",
    "sleep_hours",
    "screen_time_before_sleep",
    "academic_performance",
    "physical_activity"
]

categorical_features = [
    "gender",
    "platform_usage",
    "social_interaction_level"
]


# ============================================================
# STEP 4: PREPROCESSING
# ============================================================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

print("\nSTEP 4: PREPROCESSOR BUILT")


# ============================================================
# STEP 5: TRAIN / TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nSTEP 5: TRAIN/TEST SPLIT COMPLETE")
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# ============================================================
# STEP 6: REGRESSION MODELS
# ============================================================
print("\n" + "="*60)
print("STEP 6: REGRESSION MODELS")
print("="*60)

models = {
    "LinearRegression": LinearRegression(),
    "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
    "RandomForestRegressor": RandomForestRegressor(random_state=42, n_estimators=200),
    "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=5),
    "SVR": SVR()
}

results = []

for name, model in models.items():
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results.append({
        "Model": name,
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2": round(r2, 3)
    })

results_df = pd.DataFrame(results).sort_values(by="RMSE")
print(results_df)


# ============================================================
# STEP 7: CROSS-VALIDATION
# ============================================================
print("\n" + "="*60)
print("STEP 7: CROSS-VALIDATION")
print("="*60)

rf_pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42, n_estimators=200))
])

cv_scores = cross_val_score(
    rf_pipe,
    X,
    y,
    cv=5,
    scoring="neg_root_mean_squared_error"
)

print("CV RMSE scores:", -cv_scores)
print("Mean CV RMSE:", round(-cv_scores.mean(), 3))


# ============================================================
# STEP 8: RANDOM SEARCH - RANDOM FOREST
# ============================================================
print("\n" + "="*60)
print("STEP 8: RANDOM SEARCH - RANDOM FOREST")
print("="*60)

rf_random_pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])

param_dist_rf = {
    "model__n_estimators": randint(100, 400),
    "model__max_depth": [None, 5, 10, 15, 20],
    "model__min_samples_split": randint(2, 10),
    "model__min_samples_leaf": randint(1, 5),
    "model__max_features": ["sqrt", "log2", None]
}

random_search_rf = RandomizedSearchCV(
    estimator=rf_random_pipe,
    param_distributions=param_dist_rf,
    n_iter=20,
    cv=5,
    scoring="neg_root_mean_squared_error",
    random_state=42,
    n_jobs=-1
)

random_search_rf.fit(X_train, y_train)

print("Best Parameters:")
print(random_search_rf.best_params_)
print("Best CV Score (neg RMSE):", random_search_rf.best_score_)

best_rf = random_search_rf.best_estimator_
best_rf_preds = best_rf.predict(X_test)

print("\nTest Set Results:")
print("MAE:", round(mean_absolute_error(y_test, best_rf_preds), 3))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, best_rf_preds)), 3))
print("R2:", round(r2_score(y_test, best_rf_preds), 3))


# ============================================================
# STEP 9: FEATURE IMPORTANCE
# ============================================================
print("\n" + "="*60)
print("STEP 9: FEATURE IMPORTANCE")
print("="*60)

feature_names = best_rf.named_steps["preprocessor"].get_feature_names_out()
importances = best_rf.named_steps["model"].feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print(feature_importance_df.head(15))


# ============================================================
# STEP 10: SAVE RESULTS
# ============================================================
print("\n" + "="*60)
print("STEP 10: SAVE RESULTS")
print("="*60)

results_df.to_csv("stress_regression_results.csv", index=False)
feature_importance_df.to_csv("feature_importance_stress.csv", index=False)

predictions_df = pd.DataFrame({
    "actual_stress_level": y_test.values,
    "predicted_stress_level": best_rf_preds
})
predictions_df.to_csv("stress_predictions.csv", index=False)

print("Saved:")
print("- stress_regression_results.csv")
print("- feature_importance_stress.csv")
print("- stress_predictions.csv")

```

    STEP 1: DATA LOADED
    Shape: (1200, 13)
    
    Columns:
    ['age', 'gender', 'daily_social_media_hours', 'platform_usage', 'sleep_hours', 'screen_time_before_sleep', 'academic_performance', 'physical_activity', 'social_interaction_level', 'stress_level', 'anxiety_level', 'addiction_level', 'depression_label']
    
    Target summary:
    count    1200.000000
    mean        5.445833
    std         2.903290
    min         1.000000
    25%         3.000000
    50%         5.000000
    75%         8.000000
    max        10.000000
    Name: stress_level, dtype: float64
    
    STEP 2: FEATURES AND TARGET READY
    X shape: (1200, 9)
    y shape: (1200,)
    
    STEP 4: PREPROCESSOR BUILT
    
    STEP 5: TRAIN/TEST SPLIT COMPLETE
    Train shape: (960, 9)
    Test shape: (240, 9)
    
    ============================================================
    STEP 6: REGRESSION MODELS
    ============================================================
                       Model    MAE   RMSE     R2
    0       LinearRegression  2.434  2.819 -0.025
    4                    SVR  2.495  2.884 -0.072
    2  RandomForestRegressor  2.480  2.913 -0.095
    3    KNeighborsRegressor  2.635  3.086 -0.228
    1  DecisionTreeRegressor  3.100  3.855 -0.916
    
    ============================================================
    STEP 7: CROSS-VALIDATION
    ============================================================
    CV RMSE scores: [3.02869506 2.96628413 3.04320264 3.03723766 2.98999751]
    Mean CV RMSE: 3.013
    
    ============================================================
    STEP 8: RANDOM SEARCH - RANDOM FOREST
    ============================================================
    Best Parameters:
    {'model__max_depth': 5, 'model__max_features': 'sqrt', 'model__min_samples_leaf': 1, 'model__min_samples_split': 5, 'model__n_estimators': 121}
    Best CV Score (neg RMSE): -2.9350477548492577
    
    Test Set Results:
    MAE: 2.435
    RMSE: 2.834
    R2: -0.036
    
    ============================================================
    STEP 9: FEATURE IMPORTANCE
    ============================================================
                                     Feature  Importance
    1          num__daily_social_media_hours    0.187226
    4              num__academic_performance    0.169015
    2                       num__sleep_hours    0.167792
    5                 num__physical_activity    0.132440
    3          num__screen_time_before_sleep    0.131346
    0                               num__age    0.105508
    6                       cat__gender_male    0.026433
    8             cat__platform_usage_TikTok    0.023490
    10  cat__social_interaction_level_medium    0.021582
    7          cat__platform_usage_Instagram    0.018832
    9      cat__social_interaction_level_low    0.016336
    
    ============================================================
    STEP 10: SAVE RESULTS
    ============================================================
    Saved:
    - stress_regression_results.csv
    - feature_importance_stress.csv
    - stress_predictions.csv



```python

```


```python

# ============================================================
# STRESS LEVEL PREDICTION - FULL ML WORKFLOW
# ============================================================

```


```python

```


```python

#============================================================
# EVALUATING MODELS [TESTING]
#============================================================

```


```python

```


```python

```


```python

```


```python

# ============================================================
# STEP 4: PREPROCESSING
# ============================================================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

print("\nSTEP 4: PREPROCESSOR BUILT")
```

    
    STEP 4: PREPROCESSOR BUILT



```python

```


```python

```


```python

```


```python

# ============================================================
# STEP 5: TRAIN / TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nSTEP 5: TRAIN/TEST SPLIT COMPLETE")
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)



```

    
    STEP 5: TRAIN/TEST SPLIT COMPLETE
    Train shape: (960, 9)
    Test shape: (240, 9)



```python

# ============================================================
# STEP 6: REGRESSION MODELS
# ============================================================
print("\n" + "="*60)
print("STEP 6: REGRESSION MODELS")
print("="*60)

models = {
    "LinearRegression": LinearRegression(),
    "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
    "RandomForestRegressor": RandomForestRegressor(random_state=42, n_estimators=200),
    "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=5),
    "SVR": SVR()
}

results = []

for name, model in models.items():
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results.append({
        "Model": name,
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2": round(r2, 3)
    })

results_df = pd.DataFrame(results).sort_values(by="RMSE")
print(results_df)

```

    
    ============================================================
    STEP 6: REGRESSION MODELS
    ============================================================
                       Model    MAE   RMSE     R2
    0       LinearRegression  2.434  2.819 -0.025
    4                    SVR  2.495  2.884 -0.072
    2  RandomForestRegressor  2.480  2.913 -0.095
    3    KNeighborsRegressor  2.635  3.086 -0.228
    1  DecisionTreeRegressor  3.100  3.855 -0.916



```python

# ============================================================
# STEP 7: CROSS-VALIDATION
# ============================================================
print("\n" + "="*60)
print("STEP 7: CROSS-VALIDATION")
print("="*60)

rf_pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42, n_estimators=200))
])

cv_scores = cross_val_score(
    rf_pipe,
    X,
    y,
    cv=5,
    scoring="neg_root_mean_squared_error"
)

print("CV RMSE scores:", -cv_scores)
print("Mean CV RMSE:", round(-cv_scores.mean(), 3))

```

    
    ============================================================
    STEP 7: CROSS-VALIDATION
    ============================================================
    CV RMSE scores: [3.02869506 2.96628413 3.04320264 3.03723766 2.98999751]
    Mean CV RMSE: 3.013



```python
# ============================================================
# STEP 8: RANDOM SEARCH - RANDOM FOREST
# ============================================================
print("\n" + "="*60)
print("STEP 8: RANDOM SEARCH - RANDOM FOREST")
print("="*60)

rf_random_pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])

param_dist_rf = {
    "model__n_estimators": randint(100, 400),
    "model__max_depth": [None, 5, 10, 15, 20],
    "model__min_samples_split": randint(2, 10),
    "model__min_samples_leaf": randint(1, 5),
    "model__max_features": ["sqrt", "log2", None]
}

random_search_rf = RandomizedSearchCV(
    estimator=rf_random_pipe,
    param_distributions=param_dist_rf,
    n_iter=20,
    cv=5,
    scoring="neg_root_mean_squared_error",
    random_state=42,
    n_jobs=-1
)

random_search_rf.fit(X_train, y_train)

print("Best Parameters:")
print(random_search_rf.best_params_)
print("Best CV Score (neg RMSE):", random_search_rf.best_score_)

best_rf = random_search_rf.best_estimator_
best_rf_preds = best_rf.predict(X_test)

print("\nTest Set Results:")
print("MAE:", round(mean_absolute_error(y_test, best_rf_preds), 3))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, best_rf_preds)), 3))
print("R2:", round(r2_score(y_test, best_rf_preds), 3))
```

    
    ============================================================
    STEP 8: RANDOM SEARCH - RANDOM FOREST
    ============================================================
    Best Parameters:
    {'model__max_depth': 5, 'model__max_features': 'sqrt', 'model__min_samples_leaf': 1, 'model__min_samples_split': 5, 'model__n_estimators': 121}
    Best CV Score (neg RMSE): -2.9350477548492577
    
    Test Set Results:
    MAE: 2.435
    RMSE: 2.834
    R2: -0.036



```python

# ============================================================
# STEP 9: FEATURE IMPORTANCE
# ============================================================
print("\n" + "="*60)
print("STEP 9: FEATURE IMPORTANCE")
print("="*60)

feature_names = best_rf.named_steps["preprocessor"].get_feature_names_out()
importances = best_rf.named_steps["model"].feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print(feature_importance_df.head(15))



```

    
    ============================================================
    STEP 9: FEATURE IMPORTANCE
    ============================================================
                                     Feature  Importance
    1          num__daily_social_media_hours    0.187226
    4              num__academic_performance    0.169015
    2                       num__sleep_hours    0.167792
    5                 num__physical_activity    0.132440
    3          num__screen_time_before_sleep    0.131346
    0                               num__age    0.105508
    6                       cat__gender_male    0.026433
    8             cat__platform_usage_TikTok    0.023490
    10  cat__social_interaction_level_medium    0.021582
    7          cat__platform_usage_Instagram    0.018832
    9      cat__social_interaction_level_low    0.016336



```python
# ============================================================
# STEP 10: SAVE RESULTS
# ============================================================
print("\n" + "="*60)
print("STEP 10: SAVE RESULTS")
print("="*60)

results_df.to_csv("stress_regression_results.csv", index=False)
feature_importance_df.to_csv("feature_importance_stress.csv", index=False)

predictions_df = pd.DataFrame({
    "actual_stress_level": y_test.values,
    "predicted_stress_level": best_rf_preds
})
predictions_df.to_csv("stress_predictions.csv", index=False)

print("Saved:")
print("- stress_regression_results.csv")
print("- feature_importance_stress.csv")
print("- stress_predictions.csv")
```


```python

#============================================================
#GENDER ANALYSIS
#============================================================

```


```python
# ===== 4. SUMMARY RESULTS =====
print("\n" + "="*50)
print("5. GENDER ANALYSIS")
print("="*50)

# -----------------------------
# 5. gender analysis
# -----------------------------
continuous_vars = ["stress_level", "anxiety_level", "addiction_level"]
binary_var = "depression_label"

# -----------------------------
# 6. Summary stats by gender
# -----------------------------
summary = df.groupby("gender")[continuous_vars + [binary_var]].agg(["mean", "std", "count"])
print("\nSummary statistics by gender:")
print(summary)

# -----------------------------
# 7. T-tests for continuous outcomes
# -----------------------------
from scipy.stats import ttest_ind, chi2_contingency

print("\nIndependent-samples t-tests by gender:")
for var in continuous_vars:
    male = df[df["gender"] == "male"][var]
    female = df[df["gender"] == "female"][var]

    # Welch t-test is safer when variances may differ
    result = ttest_ind(male, female, equal_var=False)

    print(f"\n{var}")
    print(f"Male mean   = {male.mean():.3f}")
    print(f"Female mean = {female.mean():.3f}")
    print(f"t-statistic = {result.statistic:.3f}")
    print(f"p-value     = {result.pvalue:.3f}")

# -----------------------------
# 8. Chi-square test for depression by gender
# -----------------------------
print("\nChi-square test: gender vs depression_label")
contingency_table = pd.crosstab(df["gender"], df["depression_label"])
print("\nContingency table:")
print(contingency_table)

chi2, p, dof, expected = chi2_contingency(contingency_table)

print(f"\nchi2    = {chi2:.3f}")
print(f"p-value = {p:.3f}")
print(f"dof     = {dof}")

print("\nExpected frequencies:")
print(pd.DataFrame(
    expected,
    index=contingency_table.index,
    columns=contingency_table.columns
))

# -----------------------------
# 9. Simple interpretation
# -----------------------------
alpha = 0.05
print("\nInterpretation:")
for var in continuous_vars:
    male = df[df["gender"] == "male"][var]
    female = df[df["gender"] == "female"][var]
    result = ttest_ind(male, female, equal_var=False)

    if result.pvalue < alpha:
        print(f"- {var}: significant gender difference (p = {result.pvalue:.3f})")
    else:
        print(f"- {var}: no significant gender difference (p = {result.pvalue:.3f})")

if p < alpha:
    print(f"- depression_label: significant association with gender (p = {p:.3f})")
else:
    print(f"- depression_label: no significant association with gender (p = {p:.3f})")


```

    
    ==================================================
    5. GENDER ANALYSIS
    ==================================================
    
    Summary statistics by gender:
           stress_level                 anxiety_level                  \
                   mean       std count          mean       std count   
    gender                                                              
    female     5.422222  2.947808   585      5.687179  2.912465   585   
    male       5.468293  2.862521   615      5.588618  2.809628   615   
    
           addiction_level                 depression_label                  
                      mean       std count             mean       std count  
    gender                                                                   
    female         5.48547  2.821191   585         0.029060  0.168118   585  
    male           5.64065  2.839801   615         0.022764  0.149272   615  
    
    Independent-samples t-tests by gender:
    
    stress_level
    Male mean   = 5.468
    Female mean = 5.422
    t-statistic = 0.274
    p-value     = 0.784
    
    anxiety_level
    Male mean   = 5.589
    Female mean = 5.687
    t-statistic = -0.596
    p-value     = 0.551
    
    addiction_level
    Male mean   = 5.641
    Female mean = 5.485
    t-statistic = 0.949
    p-value     = 0.343
    
    Chi-square test: gender vs depression_label
    
    Contingency table:
    depression_label    0   1
    gender                   
    female            568  17
    male              601  14
    
    chi2    = 0.255
    p-value = 0.613
    dof     = 1
    
    Expected frequencies:
    depression_label         0        1
    gender                             
    female            569.8875  15.1125
    male              599.1125  15.8875
    
    Interpretation:
    - stress_level: no significant gender difference (p = 0.784)
    - anxiety_level: no significant gender difference (p = 0.551)
    - addiction_level: no significant gender difference (p = 0.343)
    - depression_label: no significant association with gender (p = 0.613)



```python

```
