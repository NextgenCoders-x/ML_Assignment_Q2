"""
generate_dataset.py
-------------------
Generates a realistic synthetic student dataset with 500 rows.

Features:
- Mix of numeric and categorical columns
- Realistic missing values (one column > 20% missing)
- Duplicate rows
- One numeric column stored as text (requires type conversion)
- Skewed numeric columns
- Outliers
"""

import os
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────
np.random.seed(42)

NUM_ROWS = 500

# ──────────────────────────────────────────────────────────────
# Helper data
# ──────────────────────────────────────────────────────────────
FIRST_NAMES_MALE = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh",
    "Ayaan", "Krishna", "Ishaan", "Rohan", "Kartik", "Sahil", "Nikhil",
    "Rahul", "Amit", "Vikram", "Raj", "Dev", "Ankit", "Manish", "Suresh",
    "Karan", "Pranav", "Harsh", "Deepak", "Mohit", "Gaurav", "Varun", "Akash"
]

FIRST_NAMES_FEMALE = [
    "Ananya", "Diya", "Myra", "Sara", "Aanya", "Aadhya", "Isha", "Priya",
    "Sneha", "Pooja", "Neha", "Kavya", "Riya", "Tanya", "Meera", "Simran",
    "Nisha", "Anjali", "Divya", "Shruti", "Tanvi", "Ritika", "Swati",
    "Pallavi", "Sonal", "Komal", "Jyoti", "Megha", "Archana", "Bhavna"
]

LAST_NAMES = [
    "Sharma", "Verma", "Gupta", "Singh", "Kumar", "Patel", "Joshi",
    "Mishra", "Reddy", "Nair", "Iyer", "Rao", "Das", "Mehta", "Shah",
    "Chopra", "Malhotra", "Kapoor", "Bhat", "Pandey", "Tiwari", "Yadav",
    "Chauhan", "Saxena", "Agarwal", "Bansal", "Sinha", "Jain", "Pillai", "Menon"
]

DEPARTMENTS = ["Computer Science", "Information Technology", "Electronics",
               "Electrical", "Mechanical", "Civil"]

# ──────────────────────────────────────────────────────────────
# Generate base columns
# ──────────────────────────────────────────────────────────────

# StudentID
student_ids = [f"S{str(i).zfill(3)}" for i in range(1, NUM_ROWS + 1)]

# Gender (with slight imbalance)
genders = np.random.choice(["Male", "Female"], size=NUM_ROWS, p=[0.55, 0.45])

# Name (gender-appropriate)
names = []
for g in genders:
    if g == "Male":
        first = np.random.choice(FIRST_NAMES_MALE)
    else:
        first = np.random.choice(FIRST_NAMES_FEMALE)
    last = np.random.choice(LAST_NAMES)
    names.append(f"{first} {last}")

# Department
departments = np.random.choice(DEPARTMENTS, size=NUM_ROWS,
                                p=[0.25, 0.20, 0.15, 0.15, 0.15, 0.10])

# Age – right-skewed (most students 18-22, a few older)
age_base = np.random.gamma(shape=4, scale=1, size=NUM_ROWS)
ages = np.clip(np.round(age_base + 17), 18, 30).astype(int)
# Inject a few outlier ages
outlier_idx = np.random.choice(NUM_ROWS, size=8, replace=False)
ages[outlier_idx] = np.random.choice([28, 29, 30, 31, 32], size=8)

# Attendance (60-100 normally, some low outliers)
attendance = np.random.normal(loc=78, scale=10, size=NUM_ROWS)
attendance = np.clip(np.round(attendance, 1), 30, 100)
# Inject outliers
outlier_idx_att = np.random.choice(NUM_ROWS, size=10, replace=False)
attendance[outlier_idx_att] = np.random.uniform(25, 40, size=10).round(1)

# StudyHours – right-skewed (most study 2-6 hrs, few study 10+)
study_hours = np.random.exponential(scale=3.5, size=NUM_ROWS)
study_hours = np.clip(np.round(study_hours, 1), 0.5, 16)
# Inject outliers
outlier_idx_sh = np.random.choice(NUM_ROWS, size=7, replace=False)
study_hours[outlier_idx_sh] = np.random.uniform(14, 18, size=7).round(1)

# AssignmentMarks (0-50, roughly normal)
assignment_marks = np.random.normal(loc=34, scale=8, size=NUM_ROWS)
assignment_marks = np.clip(np.round(assignment_marks, 1), 5, 50)

# ExamMarks (0-100, roughly normal)
exam_marks = np.random.normal(loc=58, scale=15, size=NUM_ROWS)
exam_marks = np.clip(np.round(exam_marks, 1), 10, 100)

# ProjectMarks (0-50, slightly left-skewed – most do well)
project_marks_raw = np.random.beta(a=5, b=2, size=NUM_ROWS) * 50
project_marks = np.clip(np.round(project_marks_raw, 1), 5, 50)

# TotalMarks = AssignmentMarks + ExamMarks + ProjectMarks (max 200)
total_marks = np.round(assignment_marks + exam_marks + project_marks, 1)

# CGPA – correlated with TotalMarks, with noise
cgpa_raw = (total_marks / 200) * 10 + np.random.normal(0, 0.4, size=NUM_ROWS)
cgpa = np.clip(np.round(cgpa_raw, 2), 3.0, 10.0)

# Result – based on CGPA threshold
result = np.where(cgpa >= 5.0, "Pass", "Fail")

# PlacementStatus – correlated with CGPA
placement_prob = np.clip((cgpa - 4.0) / 8.0, 0.05, 0.95)
placement = np.array(["Placed" if np.random.random() < p else "Not Placed"
                       for p in placement_prob])

# ──────────────────────────────────────────────────────────────
# Assemble DataFrame
# ──────────────────────────────────────────────────────────────
df = pd.DataFrame({
    "StudentID": student_ids,
    "Name": names,
    "Gender": genders,
    "Department": departments,
    "Age": ages,
    "Attendance": attendance,
    "StudyHours": study_hours,
    "AssignmentMarks": assignment_marks,
    "ExamMarks": exam_marks,
    "ProjectMarks": project_marks,
    "TotalMarks": total_marks,
    "CGPA": cgpa,
    "Result": result,
    "PlacementStatus": placement
})

# ──────────────────────────────────────────────────────────────
# Inject missing values
# ──────────────────────────────────────────────────────────────

# StudyHours: > 20% missing  (~25%)
missing_sh = np.random.choice(NUM_ROWS, size=125, replace=False)
df.loc[missing_sh, "StudyHours"] = np.nan

# Attendance: ~12% missing
missing_att = np.random.choice(NUM_ROWS, size=60, replace=False)
df.loc[missing_att, "Attendance"] = np.nan

# AssignmentMarks: ~8% missing
missing_am = np.random.choice(NUM_ROWS, size=40, replace=False)
df.loc[missing_am, "AssignmentMarks"] = np.nan

# CGPA: ~6% missing
missing_cgpa = np.random.choice(NUM_ROWS, size=30, replace=False)
df.loc[missing_cgpa, "CGPA"] = np.nan

# ProjectMarks: ~4% missing
missing_pm = np.random.choice(NUM_ROWS, size=20, replace=False)
df.loc[missing_pm, "ProjectMarks"] = np.nan

# Age: ~3% missing
missing_age = np.random.choice(NUM_ROWS, size=15, replace=False)
df.loc[missing_age, "Age"] = np.nan

# ──────────────────────────────────────────────────────────────
# Store ExamMarks as string (requires datatype conversion later)
# ──────────────────────────────────────────────────────────────
# Mix in some string representations and a few non-numeric entries
exam_str = df["ExamMarks"].astype(str)
# Replace a few entries with text that looks wrong
bad_idx = np.random.choice(NUM_ROWS, size=5, replace=False)
for idx in bad_idx:
    exam_str.iloc[idx] = exam_str.iloc[idx] + " marks"
df["ExamMarks"] = exam_str

# ──────────────────────────────────────────────────────────────
# Insert duplicate rows (8 duplicates)
# ──────────────────────────────────────────────────────────────
dup_indices = np.random.choice(NUM_ROWS, size=8, replace=False)
duplicates = df.iloc[dup_indices].copy()
df = pd.concat([df, duplicates], ignore_index=True)

# Shuffle so duplicates are not at the end
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Update StudentID to keep sequential (optional, keeps realism)
# We intentionally do NOT update – duplicates will share StudentIDs

# ──────────────────────────────────────────────────────────────
# Save to CSV
# ──────────────────────────────────────────────────────────────
os.makedirs("dataset", exist_ok=True)
df.to_csv("dataset/student_data.csv", index=False)

print(f"[OK] Dataset generated: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"     Saved to: dataset/student_data.csv")
print(f"\nMissing value summary:")
print(df.isnull().sum())
print(f"\nDuplicate rows: {df.duplicated().sum()}")
print(f"ExamMarks dtype: {df['ExamMarks'].dtype}")
