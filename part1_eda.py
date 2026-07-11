"""
part1_eda.py
============
Part 1 - Data Acquisition, Cleaning, and Exploratory Analysis

This script performs a complete EDA workflow on the student dataset:
  1. Data loading and initial inspection
  2. Missing value analysis and imputation
  3. Duplicate detection and removal
  4. Data type correction and memory optimization
  5. Descriptive statistics and skewness analysis
  6. Outlier detection using IQR
  7. Visualizations (saved to images/)
  8. Correlation analysis (Pearson & Spearman)
  9. GroupBy aggregation
 10. Save cleaned dataset
"""

# ============================================================
# 1. Import Libraries
# ============================================================
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")                    # Non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")        # Suppress warnings for clean output

# Create output directory for images
os.makedirs("images", exist_ok=True)

# Set plot style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.figsize": (10, 6), "figure.dpi": 150})

print("=" * 70)
print("  Part 1 - Data Acquisition, Cleaning, and Exploratory Analysis")
print("=" * 70)

# ============================================================
# 2. Load Dataset
# ============================================================
print("\n" + "-" * 70)
print("STEP 2: Load Dataset")
print("-" * 70)

df = pd.read_csv("dataset/student_data.csv")
print("[OK] Dataset loaded successfully from 'dataset/student_data.csv'")

# ============================================================
# 3. Initial Inspection - head(), shape, dtypes
# ============================================================
print("\n" + "-" * 70)
print("STEP 3: Initial Inspection")
print("-" * 70)

print("\n--- First 5 Rows (head) ---")
print(df.head().to_string())

print(f"\n--- Shape ---\nRows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n--- Data Types ---")
print(df.dtypes.to_string())

# ============================================================
# 4. Missing Value Analysis
# ============================================================
print("\n" + "-" * 70)
print("STEP 4: Missing Value Analysis")
print("-" * 70)

# Compute missing count and percentage
missing_count = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    "Missing Count": missing_count,
    "Missing Percentage (%)": missing_pct
}).sort_values(by="Missing Percentage (%)", ascending=False)

print("\n--- Missing Value Summary ---")
print(missing_df.to_string())

# Identify columns with > 20% missing
cols_over_20 = missing_df[missing_df["Missing Percentage (%)"] > 20].index.tolist()
print(f"\n[!] Columns with > 20% missing values: {cols_over_20}")

# Fill numeric columns with <= 20% missing using median
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cols_under_20_numeric = [
    col for col in numeric_cols
    if 0 < missing_pct[col] <= 20
]
print(f"\nNumeric columns with <= 20% missing (to be median-imputed): {cols_under_20_numeric}")

for col in cols_under_20_numeric:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)
    print(f"   -> Filled '{col}' with median = {median_val}")

print(f"\nRemaining missing values after imputation:")
print(df.isnull().sum().to_string())

# ============================================================
# 5. Duplicate Detection and Removal
# ============================================================
print("\n" + "-" * 70)
print("STEP 5: Duplicate Detection and Removal")
print("-" * 70)

dup_count = df.duplicated().sum()
print(f"Duplicate rows detected: {dup_count}")

df = df.drop_duplicates().reset_index(drop=True)
print(f"Duplicates removed. New shape: {df.shape}")

# ============================================================
# 6. Data Type Correction
# ============================================================
print("\n" + "-" * 70)
print("STEP 6: Data Type Correction")
print("-" * 70)

# Memory usage BEFORE conversion
mem_before = df.memory_usage(deep=True).sum()
print(f"\n[MEM] Memory usage BEFORE conversion: {mem_before:,} bytes ({mem_before / 1024:.2f} KB)")
print(f"\nCurrent dtype of 'ExamMarks': {df['ExamMarks'].dtype}")
print("Sample values:", df["ExamMarks"].head(10).tolist())

# Convert ExamMarks from string to numeric
df["ExamMarks"] = pd.to_numeric(df["ExamMarks"], errors="coerce")
print(f"Converted 'ExamMarks' to: {df['ExamMarks'].dtype}")

# Fill any NaN introduced by coercion with median
exam_median = df["ExamMarks"].median()
df["ExamMarks"] = df["ExamMarks"].fillna(exam_median)
print(f"   -> Filled coerced NaN in 'ExamMarks' with median = {exam_median}")

# Convert repeated string columns to category type
for col in ["Gender", "Department", "Result", "PlacementStatus"]:
    df[col] = df[col].astype("category")
    print(f"   -> Converted '{col}' to category dtype")

# Memory usage AFTER conversion
mem_after = df.memory_usage(deep=True).sum()
print(f"\n[MEM] Memory usage AFTER conversion: {mem_after:,} bytes ({mem_after / 1024:.2f} KB)")
print(f"[SAVE] Memory saved: {mem_before - mem_after:,} bytes "
      f"({(mem_before - mem_after) / mem_before * 100:.1f}%)")

print(f"\nUpdated dtypes:")
print(df.dtypes.to_string())

# ============================================================
# 7. Descriptive Statistics
# ============================================================
print("\n" + "-" * 70)
print("STEP 7: Descriptive Statistics")
print("-" * 70)

print("\n--- describe() ---")
print(df.describe().round(2).to_string())

# ============================================================
# 8. Skewness Analysis
# ============================================================
print("\n" + "-" * 70)
print("STEP 8: Skewness Analysis")
print("-" * 70)

numeric_cols_updated = df.select_dtypes(include=[np.number]).columns.tolist()
skewness = df[numeric_cols_updated].skew().round(4)
print("\n--- Skewness for Numeric Columns ---")
print(skewness.to_string())

most_skewed_col = skewness.abs().idxmax()
most_skewed_val = skewness[most_skewed_col]
print(f"\n[RESULT] Most skewed column: '{most_skewed_col}' (skewness = {most_skewed_val})")

# ============================================================
# 9. Outlier Detection using IQR
# ============================================================
print("\n" + "-" * 70)
print("STEP 9: Outlier Detection using IQR Method")
print("-" * 70)

# Detect outliers for StudyHours and Attendance
outlier_cols = ["StudyHours", "Attendance"]

for col in outlier_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

    print(f"\n--- '{col}' ---")
    print(f"   Q1            = {Q1:.2f}")
    print(f"   Q3            = {Q3:.2f}")
    print(f"   IQR           = {IQR:.2f}")
    print(f"   Lower Bound   = {lower_bound:.2f}")
    print(f"   Upper Bound   = {upper_bound:.2f}")
    print(f"   Outlier Count = {outlier_count}")

print("\n(Outliers are NOT removed - retained for analysis.)")

# ============================================================
# 10. Visualizations
# ============================================================
print("\n" + "-" * 70)
print("STEP 10: Visualizations (saving to images/)")
print("-" * 70)

# ---- 10a. Line Plot ----
# Average marks trend by Department
dept_order = df.groupby("Department")["TotalMarks"].mean().sort_values().index
dept_means = df.groupby("Department")["TotalMarks"].mean().reindex(dept_order)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(dept_means.index, dept_means.values, marker="o", linewidth=2,
        color="#2196F3", markerfacecolor="#FF5722", markersize=8)
ax.set_title("Average Total Marks by Department", fontsize=14, fontweight="bold")
ax.set_xlabel("Department", fontsize=12)
ax.set_ylabel("Average Total Marks", fontsize=12)
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig("images/line_plot.png", bbox_inches="tight")
plt.close()
print("   -> Saved: images/line_plot.png")

# ---- 10b. Bar Chart ----
# Student count per Department
dept_counts = df["Department"].value_counts()

fig, ax = plt.subplots(figsize=(10, 6))
colors = sns.color_palette("viridis", n_colors=len(dept_counts))
ax.bar(dept_counts.index, dept_counts.values, color=colors, edgecolor="black")
ax.set_title("Number of Students per Department", fontsize=14, fontweight="bold")
ax.set_xlabel("Department", fontsize=12)
ax.set_ylabel("Student Count", fontsize=12)
ax.tick_params(axis="x", rotation=30)
# Add value labels on bars
for i, (val, name) in enumerate(zip(dept_counts.values, dept_counts.index)):
    ax.text(i, val + 1, str(val), ha="center", va="bottom", fontweight="bold")
plt.tight_layout()
plt.savefig("images/bar_chart.png", bbox_inches="tight")
plt.close()
print("   -> Saved: images/bar_chart.png")

# ---- 10c. Histogram ----
# Distribution of StudyHours (skewed column)
fig, ax = plt.subplots(figsize=(10, 6))
df["StudyHours"].dropna().hist(bins=25, ax=ax, color="#4CAF50", edgecolor="black", alpha=0.8)
ax.set_title("Distribution of Study Hours", fontsize=14, fontweight="bold")
ax.set_xlabel("Study Hours", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.axvline(df["StudyHours"].mean(), color="red", linestyle="--", label=f"Mean = {df['StudyHours'].mean():.2f}")
ax.axvline(df["StudyHours"].median(), color="blue", linestyle="--", label=f"Median = {df['StudyHours'].median():.2f}")
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("images/histogram.png", bbox_inches="tight")
plt.close()
print("   -> Saved: images/histogram.png")

# ---- 10d. Scatter Plot ----
# StudyHours vs CGPA
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(df["StudyHours"], df["CGPA"],
                     c=df["TotalMarks"], cmap="coolwarm", alpha=0.7,
                     edgecolors="black", linewidth=0.5, s=50)
plt.colorbar(scatter, ax=ax, label="Total Marks")
ax.set_title("Study Hours vs CGPA (colored by Total Marks)", fontsize=14, fontweight="bold")
ax.set_xlabel("Study Hours", fontsize=12)
ax.set_ylabel("CGPA", fontsize=12)
plt.tight_layout()
plt.savefig("images/scatter_plot.png", bbox_inches="tight")
plt.close()
print("   -> Saved: images/scatter_plot.png")

# ---- 10e. Box Plot ----
# Box plot for key numeric columns
box_cols = ["Attendance", "StudyHours", "AssignmentMarks", "ExamMarks", "ProjectMarks"]
fig, ax = plt.subplots(figsize=(12, 6))
box_data = [df[col].dropna().values for col in box_cols]
bp = ax.boxplot(box_data, tick_labels=box_cols, patch_artist=True, notch=True)
palette = sns.color_palette("Set2", n_colors=len(box_cols))
for patch, color in zip(bp["boxes"], palette):
    patch.set_facecolor(color)
ax.set_title("Box Plot - Key Numeric Columns", fontsize=14, fontweight="bold")
ax.set_xlabel("Column", fontsize=12)
ax.set_ylabel("Value", fontsize=12)
plt.tight_layout()
plt.savefig("images/box_plot.png", bbox_inches="tight")
plt.close()
print("   -> Saved: images/box_plot.png")

# ---- 10f. Correlation Heatmap ----
corr_matrix = df[numeric_cols_updated].corr()
fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdYlBu_r",
            mask=mask, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            square=True, cbar_kws={"shrink": 0.8})
ax.set_title("Pearson Correlation Heatmap", fontsize=14, fontweight="bold")
ax.set_xlabel("Features", fontsize=12)
ax.set_ylabel("Features", fontsize=12)
plt.tight_layout()
plt.savefig("images/heatmap.png", bbox_inches="tight")
plt.close()
print("   -> Saved: images/heatmap.png")

# ============================================================
# 11. Pearson Correlation
# ============================================================
print("\n" + "-" * 70)
print("STEP 11: Pearson Correlation Matrix")
print("-" * 70)

pearson_corr = df[numeric_cols_updated].corr(method="pearson")
print(pearson_corr.round(3).to_string())

# ============================================================
# 12. Spearman Correlation
# ============================================================
print("\n" + "-" * 70)
print("STEP 12: Spearman Correlation Matrix")
print("-" * 70)

spearman_corr = df[numeric_cols_updated].corr(method="spearman")
print(spearman_corr.round(3).to_string())

# ============================================================
# 13. Pearson vs Spearman Comparison
# ============================================================
print("\n" + "-" * 70)
print("STEP 13: Pearson vs Spearman Comparison")
print("-" * 70)

# Compute absolute differences
abs_diff = (pearson_corr - spearman_corr).abs()

# Create a readable comparison table (upper triangle pairs)
comparison_rows = []
cols_list = abs_diff.columns.tolist()
for i in range(len(cols_list)):
    for j in range(i + 1, len(cols_list)):
        c1, c2 = cols_list[i], cols_list[j]
        comparison_rows.append({
            "Variable 1": c1,
            "Variable 2": c2,
            "Pearson": round(pearson_corr.loc[c1, c2], 4),
            "Spearman": round(spearman_corr.loc[c1, c2], 4),
            "Absolute Difference": round(abs_diff.loc[c1, c2], 4)
        })

comparison_df = pd.DataFrame(comparison_rows).sort_values(
    by="Absolute Difference", ascending=False
).reset_index(drop=True)

print("\n--- Full Comparison Table ---")
print(comparison_df.to_string())

print("\n[TOP 3] Largest Differences (Pearson vs Spearman):")
print(comparison_df.head(3).to_string())

# ============================================================
# 14. Median Imputation for Two Most Skewed Columns
# ============================================================
print("\n" + "-" * 70)
print("STEP 14: Median Imputation for Most Skewed Columns")
print("-" * 70)

# Get top 2 most skewed columns
top2_skewed = skewness.abs().sort_values(ascending=False).head(2).index.tolist()
print(f"\nTwo most skewed columns: {top2_skewed}")

for col in top2_skewed:
    mean_val = df[col].mean()
    median_val = df[col].median()
    missing_before = df[col].isnull().sum()

    print(f"\n--- '{col}' ---")
    print(f"   Mean           = {mean_val:.4f}")
    print(f"   Median         = {median_val:.4f}")
    print(f"   Missing before = {missing_before}")

    # Impute with median
    df[col] = df[col].fillna(median_val)
    missing_after = df[col].isnull().sum()
    print(f"   Missing after  = {missing_after}")

print(f"\n[OK] Confirmation - Total missing values remaining: {df.isnull().sum().sum()}")

# ============================================================
# 15. GroupBy Aggregation
# ============================================================
print("\n" + "-" * 70)
print("STEP 15: GroupBy Aggregation (Department)")
print("-" * 70)

groupby_result = df.groupby("Department")["TotalMarks"].agg(["mean", "std", "count"])
print("\n--- GroupBy Department -> TotalMarks ---")
print(groupby_result.round(2).to_string())

# Highest mean group
highest_mean_dept = groupby_result["mean"].idxmax()
highest_mean_val = groupby_result["mean"].max()
print(f"\n[1st] Highest Mean:  {highest_mean_dept} ({highest_mean_val:.2f})")

# Highest std group
highest_std_dept = groupby_result["std"].idxmax()
highest_std_val = groupby_result["std"].max()
print(f"[STD] Highest Std:   {highest_std_dept} ({highest_std_val:.2f})")

# Mean ratio (highest / lowest)
lowest_mean_val = groupby_result["mean"].min()
lowest_mean_dept = groupby_result["mean"].idxmin()
mean_ratio = highest_mean_val / lowest_mean_val
print(f"[RATIO] Mean Ratio:  {highest_mean_dept} / {lowest_mean_dept} = {mean_ratio:.4f}")

# ============================================================
# 16. Save Cleaned Dataset
# ============================================================
print("\n" + "-" * 70)
print("STEP 16: Save Cleaned Dataset")
print("-" * 70)

df.to_csv("dataset/cleaned_data.csv", index=False)
print("[OK] Cleaned dataset saved to: dataset/cleaned_data.csv")
print(f"     Final shape: {df.shape}")

print("\n" + "=" * 70)
print("  [OK] All steps completed successfully!")
print("=" * 70)
