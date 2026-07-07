import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

print("1. Loading openFDA Adverse Drug Reactions dataset...")
df = pd.read_csv("openfda_adr_dataset.csv")
print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

print("\n2. Data Cleaning & Preprocessing...")
df_clean = df.copy()

# Impute age with median
age_median = df_clean['age'].median()
df_clean['age'] = df_clean['age'].fillna(age_median)

# Impute gender
df_clean['gender'] = df_clean['gender'].fillna(0).astype(int).astype(str)
gender_map = {'1': 'Male', '2': 'Female', '0': 'Unknown'}
df_clean['gender_name'] = df_clean['gender'].map(gender_map)

# Impute country and dosage
df_clean['country'] = df_clean['country'].fillna('Unknown')
df_clean['drug_dosage_form'] = df_clean['drug_dosage_form'].fillna('Unknown')

# Target
df_clean['severity_name'] = df_clean['adr_severity'].map({1: 'Serious', 2: 'Non-Serious'})

# Standardize dosage forms
def standardize_dosage(form):
    form_lower = str(form).lower()
    if 'tablet' in form_lower:
        return 'Tablet'
    elif 'capsule' in form_lower:
        return 'Capsule'
    elif 'injection' in form_lower or 'inj' in form_lower:
        return 'Injection'
    elif 'solution' in form_lower or 'liquid' in form_lower:
        return 'Oral Solution'
    elif 'suspension' in form_lower:
        return 'Suspension'
    elif 'unknown' in form_lower:
        return 'Unknown'
    else:
        return 'Other'

df_clean['dosage_form_clean'] = df_clean['drug_dosage_form'].apply(standardize_dosage)
top_countries = df_clean['country'].value_counts().index[:5]
df_clean['country_clean'] = df_clean['country'].apply(lambda x: x if x in top_countries else 'Other')

# Create age groups
def get_age_group(age):
    if age < 18:
        return 'Pediatric (<18)'
    elif age <= 35:
        return 'Young Adult (18-35)'
    elif age <= 50:
        return 'Adult (36-50)'
    elif age <= 65:
        return 'Middle Aged (51-65)'
    else:
        return 'Senior (>65)'

df_clean['age_group'] = df_clean['age'].apply(get_age_group)

print("Data cleaning complete.")

print("\n3. Feature Engineering...")
X_raw = df_clean[['age', 'gender_name', 'dosage_form_clean', 'country_clean', 'hospitalization', 'death']]
y = df_clean['adr_severity'].apply(lambda x: 1 if x == 1 else 0)
X = pd.get_dummies(X_raw, columns=['gender_name', 'dosage_form_clean', 'country_clean'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Features: {X.columns.tolist()}")

print("\n4. Training Baseline Decision Tree...")
dt_baseline = DecisionTreeClassifier(random_state=42)
dt_baseline.fit(X_train, y_train)
print(f"Baseline Depth: {dt_baseline.get_depth()}, Leaf Nodes: {dt_baseline.get_n_leaves()}")

print("\n5. Hyperparameter Tuning using GridSearchCV...")
param_grid = {
    'max_depth': [3, 5, 8, 10, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'criterion': ['gini', 'entropy']
}
grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
dt_tuned = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Tuned Depth: {dt_tuned.get_depth()}, Leaf Nodes: {dt_tuned.get_n_leaves()}")

print("\n6. Model Evaluation...")
y_pred_base = dt_baseline.predict(X_test)
y_pred_tuned = dt_tuned.predict(X_test)

print("\nBaseline Model Metrics:")
print(f"  Accuracy:  {accuracy_score(y_test, y_pred_base):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_base):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred_base):.4f}")
print(f"  F1-score:  {f1_score(y_test, y_pred_base):.4f}")

print("\nTuned Model Metrics:")
print(f"  Accuracy:  {accuracy_score(y_test, y_pred_tuned):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_tuned):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred_tuned):.4f}")
print(f"  F1-score:  {f1_score(y_test, y_pred_tuned):.4f}")

print("\n7. Standalone execution complete. All steps completed successfully.")