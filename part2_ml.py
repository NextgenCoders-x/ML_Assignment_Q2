"""
part2_ml.py
===========
Part 2 - Machine Learning: Regression and Classification

This script builds on the cleaned dataset from Part 1 and implements:
  1. Target definition (regression + classification)
  2. Feature encoding (label + one-hot)
  3. Train/test split
  4. Feature scaling (StandardScaler)
  5. Linear Regression + Ridge Regression
  6. Logistic Regression with class-imbalance handling
  7. Full classification evaluation (confusion matrix, ROC, AUC)
  8. Threshold sensitivity analysis
  9. Regularization experiment (C=1.0 vs C=0.01)
 10. Bootstrap confidence interval for AUC difference
"""

# ============================================================
# 1. Import Libraries
# ============================================================
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")                    # Non-interactive backend for scripts
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, r2_score,
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, classification_report,
    roc_curve, roc_auc_score
)

warnings.filterwarnings("ignore")
os.makedirs("images", exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.figsize": (10, 6), "figure.dpi": 150})

print("=" * 70)
print("  Part 2 - Machine Learning: Regression and Classification")
print("=" * 70)

# ============================================================
# 2. Load Cleaned Dataset
# ============================================================
print("\n" + "-" * 70)
print("STEP 1: Load Cleaned Dataset")
print("-" * 70)

df = pd.read_csv("dataset/cleaned_data.csv")
print(f"[OK] Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Columns: {df.columns.tolist()}")

# ============================================================
# 3. Define Targets
# ============================================================
print("\n" + "-" * 70)
print("STEP 2: Define Feature Matrix and Targets")
print("-" * 70)

# Regression target: TotalMarks
y_reg = df["TotalMarks"].copy()
print(f"\nRegression target (y_reg): TotalMarks")
print(f"   Mean = {y_reg.mean():.2f}, Std = {y_reg.std():.2f}")

# Classification target: binary based on TotalMarks median
median_marks = df["TotalMarks"].median()
y_clf = (df["TotalMarks"] > median_marks).astype(int)
print(f"\nClassification target (y_clf): TotalMarks > {median_marks}")
print(f"   Class counts:")
print(f"   0 (Below/Equal Median): {(y_clf == 0).sum()}")
print(f"   1 (Above Median):       {(y_clf == 1).sum()}")

# Feature matrix: drop target + non-predictive columns
# StudentID and Name are identifiers, not features
# CGPA and Result are derived from TotalMarks (would cause data leakage)
# PlacementStatus is a downstream outcome
drop_cols = ["TotalMarks", "StudentID", "Name", "CGPA", "Result", "PlacementStatus"]
X = df.drop(columns=drop_cols)
print(f"\nFeature matrix X shape: {X.shape}")
print(f"Features used: {X.columns.tolist()}")

# ============================================================
# 4. Encoding Categorical Columns
# ============================================================
print("\n" + "-" * 70)
print("STEP 3: Encoding Categorical Columns")
print("-" * 70)

# Detect categorical columns automatically
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nCategorical columns detected: {cat_cols}")
print(f"Numeric columns detected:     {num_cols}")

# Gender is binary/nominal -> label encode (Male=1, Female=0)
# This is acceptable for binary categories
le_gender = LabelEncoder()
X["Gender"] = le_gender.fit_transform(X["Gender"])
print(f"\n[ENCODE] Gender: Label Encoded -> {dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_)))}")

# Department is nominal (no natural order) -> one-hot encode
X = pd.get_dummies(X, columns=["Department"], drop_first=True, dtype=int)
print(f"[ENCODE] Department: One-Hot Encoded (drop_first=True)")

print(f"\nEncoded feature names ({X.shape[1]} features):")
print(f"   {X.columns.tolist()}")
print(f"\nEncoded X shape: {X.shape}")

# ============================================================
# 5. Train/Test Split
# ============================================================
print("\n" + "-" * 70)
print("STEP 4: Train/Test Split")
print("-" * 70)

# Split for regression
X_train, X_test, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.20, random_state=42
)

# Split for classification (same indices)
_, _, y_train_clf, y_test_clf = train_test_split(
    X, y_clf, test_size=0.20, random_state=42
)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_train_reg:   {y_train_reg.shape[0]} samples")
print(f"y_test_reg:    {y_test_reg.shape[0]} samples")
print(f"y_train_clf:   {y_train_clf.shape[0]} samples")
print(f"y_test_clf:    {y_test_clf.shape[0]} samples")

# ============================================================
# 6. Feature Scaling
# ============================================================
print("\n" + "-" * 70)
print("STEP 5: Feature Scaling (StandardScaler)")
print("-" * 70)

scaler = StandardScaler()

# Fit ONLY on training data to prevent data leakage
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

# Transform test data using training statistics
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

print("[OK] StandardScaler fitted on X_train only (no data leakage)")
print(f"\nTraining data stats after scaling:")
print(f"   Mean (should be ~0): {X_train_scaled.mean().mean():.6f}")
print(f"   Std  (should be ~1): {X_train_scaled.std().mean():.6f}")

# ============================================================
# 7. Regression Model 1: Linear Regression
# ============================================================
print("\n" + "-" * 70)
print("STEP 6: Linear Regression")
print("-" * 70)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train_reg)
y_pred_lr = lr.predict(X_test_scaled)

# Evaluation metrics
mse_lr = mean_squared_error(y_test_reg, y_pred_lr)
r2_lr = r2_score(y_test_reg, y_pred_lr)

print(f"\n--- Linear Regression Results ---")
print(f"   Mean Squared Error (MSE): {mse_lr:.4f}")
print(f"   R-squared (R2):           {r2_lr:.4f}")

# Coefficients with feature names
coef_df = pd.DataFrame({
    "Feature": X_train_scaled.columns,
    "Coefficient": lr.coef_
}).sort_values(by="Coefficient", ascending=False)

print(f"\n--- Coefficients ---")
print(coef_df.to_string(index=False))

# Top 3 by absolute value
top3 = coef_df.assign(AbsCoeff=coef_df["Coefficient"].abs()) \
              .nlargest(3, "AbsCoeff")[["Feature", "Coefficient"]]
print(f"\n[TOP 3] Largest absolute coefficients:")
print(top3.to_string(index=False))

# ============================================================
# 8. Regression Model 2: Ridge Regression
# ============================================================
print("\n" + "-" * 70)
print("STEP 7: Ridge Regression (alpha=1.0)")
print("-" * 70)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train_reg)
y_pred_ridge = ridge.predict(X_test_scaled)

mse_ridge = mean_squared_error(y_test_reg, y_pred_ridge)
r2_ridge = r2_score(y_test_reg, y_pred_ridge)

print(f"\n--- Ridge Regression Results ---")
print(f"   Mean Squared Error (MSE): {mse_ridge:.4f}")
print(f"   R-squared (R2):           {r2_ridge:.4f}")

# Comparison table
reg_comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Ridge Regression (alpha=1.0)"],
    "MSE": [mse_lr, mse_ridge],
    "R2": [r2_lr, r2_ridge]
})
print(f"\n--- Regression Model Comparison ---")
print(reg_comparison.to_string(index=False))

# ============================================================
# 9. Classification: Class Imbalance Check
# ============================================================
print("\n" + "-" * 70)
print("STEP 8: Class Imbalance Check")
print("-" * 70)

train_class_counts = y_train_clf.value_counts()
minority_pct = train_class_counts.min() / len(y_train_clf) * 100

print(f"\nTraining set class distribution:")
print(f"   Class 0: {train_class_counts.get(0, 0)} ({train_class_counts.get(0, 0)/len(y_train_clf)*100:.1f}%)")
print(f"   Class 1: {train_class_counts.get(1, 0)} ({train_class_counts.get(1, 0)/len(y_train_clf)*100:.1f}%)")
print(f"   Minority class: {minority_pct:.1f}%")

# Determine if class_weight='balanced' is needed
if minority_pct < 35:
    use_balanced = True
    print(f"\n[!] Minority class ({minority_pct:.1f}%) is below 35% -> using class_weight='balanced'")
else:
    use_balanced = False
    print(f"\n[OK] Class distribution is acceptable ({minority_pct:.1f}% >= 35%) -> no balancing needed")

class_weight_param = "balanced" if use_balanced else None

# ============================================================
# 10. Logistic Regression (C=1.0)
# ============================================================
print("\n" + "-" * 70)
print("STEP 9: Logistic Regression (C=1.0)")
print("-" * 70)

log_reg_c1 = LogisticRegression(
    max_iter=1000,
    C=1.0,
    class_weight=class_weight_param,
    random_state=42
)
log_reg_c1.fit(X_train_scaled, y_train_clf)

# Predict labels
y_pred_clf = log_reg_c1.predict(X_test_scaled)

# Predict probabilities
y_prob_clf = log_reg_c1.predict_proba(X_test_scaled)[:, 1]

print(f"[OK] Logistic Regression trained (C=1.0, max_iter=1000)")
print(f"     class_weight = {class_weight_param}")

# ============================================================
# 11. Classification Evaluation
# ============================================================
print("\n" + "-" * 70)
print("STEP 10: Classification Evaluation")
print("-" * 70)

# Confusion Matrix
cm = confusion_matrix(y_test_clf, y_pred_clf)
print(f"\n--- Confusion Matrix ---")
print(f"   [[TN={cm[0][0]}, FP={cm[0][1]}],")
print(f"    [FN={cm[1][0]}, TP={cm[1][1]}]]")

# Metrics
acc = accuracy_score(y_test_clf, y_pred_clf)
prec = precision_score(y_test_clf, y_pred_clf)
rec = recall_score(y_test_clf, y_pred_clf)
f1 = f1_score(y_test_clf, y_pred_clf)
auc_c1 = roc_auc_score(y_test_clf, y_prob_clf)

print(f"\n--- Metrics ---")
print(f"   Accuracy:  {acc:.4f}")
print(f"   Precision: {prec:.4f}")
print(f"   Recall:    {rec:.4f}")
print(f"   F1 Score:  {f1:.4f}")
print(f"   AUC Score: {auc_c1:.4f}")

print(f"\n--- Classification Report ---")
print(classification_report(y_test_clf, y_pred_clf, target_names=["Below Median", "Above Median"]))

# ============================================================
# 12. ROC Curve
# ============================================================
print("\n" + "-" * 70)
print("STEP 11: ROC Curve")
print("-" * 70)

fpr, tpr, thresholds_roc = roc_curve(y_test_clf, y_prob_clf)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color="#2196F3", linewidth=2.5,
        label=f"Logistic Regression (AUC = {auc_c1:.4f})")
ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1,
        label="Random Classifier (AUC = 0.5)")
ax.fill_between(fpr, tpr, alpha=0.15, color="#2196F3")
ax.set_title("ROC Curve - Logistic Regression (C=1.0)", fontsize=14, fontweight="bold")
ax.set_xlabel("False Positive Rate (FPR)", fontsize=12)
ax.set_ylabel("True Positive Rate (TPR)", fontsize=12)
ax.legend(loc="lower right", fontsize=11)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig("images/roc_curve.png", bbox_inches="tight")
plt.close()
print("   -> Saved: images/roc_curve.png")

# ============================================================
# 13. Threshold Sensitivity Analysis
# ============================================================
print("\n" + "-" * 70)
print("STEP 12: Threshold Sensitivity Analysis")
print("-" * 70)

thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
threshold_results = []

for t in thresholds:
    y_pred_t = (y_prob_clf >= t).astype(int)
    p = precision_score(y_test_clf, y_pred_t, zero_division=0)
    r = recall_score(y_test_clf, y_pred_t, zero_division=0)
    f = f1_score(y_test_clf, y_pred_t, zero_division=0)
    threshold_results.append({
        "Threshold": t,
        "Precision": round(p, 4),
        "Recall": round(r, 4),
        "F1": round(f, 4)
    })

threshold_df = pd.DataFrame(threshold_results)
print("\n--- Threshold Sensitivity Table ---")
print(threshold_df.to_string(index=False))

# Best threshold based on highest F1
best_row = threshold_df.loc[threshold_df["F1"].idxmax()]
print(f"\n[BEST] Optimal threshold = {best_row['Threshold']} (F1 = {best_row['F1']})")

# ============================================================
# 14. Regularization Experiment (C=0.01)
# ============================================================
print("\n" + "-" * 70)
print("STEP 13: Regularization Experiment (C=0.01 vs C=1.0)")
print("-" * 70)

log_reg_c001 = LogisticRegression(
    max_iter=1000,
    C=0.01,
    class_weight=class_weight_param,
    random_state=42
)
log_reg_c001.fit(X_train_scaled, y_train_clf)

y_pred_c001 = log_reg_c001.predict(X_test_scaled)
y_prob_c001 = log_reg_c001.predict_proba(X_test_scaled)[:, 1]

prec_c001 = precision_score(y_test_clf, y_pred_c001)
rec_c001 = recall_score(y_test_clf, y_pred_c001)
auc_c001 = roc_auc_score(y_test_clf, y_prob_c001)

print(f"\n--- Logistic Regression (C=0.01) Results ---")
print(f"   Precision: {prec_c001:.4f}")
print(f"   Recall:    {rec_c001:.4f}")
print(f"   AUC:       {auc_c001:.4f}")

# Comparison table
reg_clf_comparison = pd.DataFrame({
    "Model": ["Logistic Regression (C=1.0)", "Logistic Regression (C=0.01)"],
    "Precision": [prec, prec_c001],
    "Recall": [rec, rec_c001],
    "AUC": [auc_c1, auc_c001]
})
print(f"\n--- Regularization Comparison ---")
print(reg_clf_comparison.to_string(index=False))

# ============================================================
# 15. Bootstrap Confidence Interval
# ============================================================
print("\n" + "-" * 70)
print("STEP 14: Bootstrap Confidence Interval for AUC Difference")
print("-" * 70)

np.random.seed(42)
n_bootstrap = 500
auc_diffs = []

n_test = len(y_test_clf)

for i in range(n_bootstrap):
    # Sample with replacement
    indices = np.random.choice(n_test, size=n_test, replace=True)
    y_true_boot = y_test_clf.values[indices]
    y_prob_c1_boot = y_prob_clf[indices]
    y_prob_c001_boot = y_prob_c001[indices]

    # Skip if only one class in bootstrap sample
    if len(np.unique(y_true_boot)) < 2:
        continue

    auc_c1_boot = roc_auc_score(y_true_boot, y_prob_c1_boot)
    auc_c001_boot = roc_auc_score(y_true_boot, y_prob_c001_boot)
    auc_diffs.append(auc_c1_boot - auc_c001_boot)

auc_diffs = np.array(auc_diffs)
mean_diff = np.mean(auc_diffs)
lower_ci = np.percentile(auc_diffs, 2.5)
upper_ci = np.percentile(auc_diffs, 97.5)

print(f"\nBootstrap samples used: {len(auc_diffs)} / {n_bootstrap}")
print(f"\n--- AUC Difference: AUC(C=1.0) - AUC(C=0.01) ---")
print(f"   Mean Difference: {mean_diff:.4f}")
print(f"   Lower CI (2.5%): {lower_ci:.4f}")
print(f"   Upper CI (97.5%): {upper_ci:.4f}")
print(f"   95% Confidence Interval: [{lower_ci:.4f}, {upper_ci:.4f}]")

if lower_ci > 0:
    print(f"\n[RESULT] CI excludes zero -> C=1.0 has statistically significantly HIGHER AUC than C=0.01")
elif upper_ci < 0:
    print(f"\n[RESULT] CI excludes zero -> C=0.01 has statistically significantly HIGHER AUC than C=1.0")
else:
    print(f"\n[RESULT] CI includes zero -> No statistically significant difference between C=1.0 and C=0.01")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("  [OK] Part 2 - All steps completed successfully!")
print("=" * 70)
print(f"\nFiles generated:")
print(f"   -> images/roc_curve.png")
print(f"\nKey Results:")
print(f"   Linear Regression R2:   {r2_lr:.4f}")
print(f"   Ridge Regression R2:    {r2_ridge:.4f}")
print(f"   LogReg (C=1.0) AUC:    {auc_c1:.4f}")
print(f"   LogReg (C=0.01) AUC:   {auc_c001:.4f}")
print(f"   Best F1 Threshold:      {best_row['Threshold']}")
print(f"   Bootstrap AUC Diff CI:  [{lower_ci:.4f}, {upper_ci:.4f}]")
