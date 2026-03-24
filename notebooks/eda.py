# =========================
# 1. IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# =========================
# 2. LOAD DATA
# =========================
df = pd.read_csv(r"C:\Users\gomathi.srinivasan\Documents\Bank_deposit\data\bank.csv", sep=";")

print("Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())

print("\nData Info:")
print(df.info())

# =========================
# 3. TARGET ANALYSIS
# =========================
print("\nTarget Distribution:")
print(df["y"].value_counts())

print("\nTarget Distribution (Percentage):")
print(df["y"].value_counts(normalize=True) * 100)

plt.figure()
df["y"].value_counts().plot(kind="bar")
plt.title("Target Distribution")
plt.show()

# =========================
# 4. MISSING / UNKNOWN VALUES
# =========================
print("\nMissing values (NaN):")
print(df.isnull().sum())

print("\n'unknown' values in categorical columns:")
print((df == "unknown").sum())

# =========================
# 5. FEATURE TYPES
# =========================
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
cat_cols = df.select_dtypes(include=["object"]).columns

print("\nNumerical Columns:", list(num_cols))
print("\nCategorical Columns:", list(cat_cols))

# =========================
# 6. NUMERICAL DISTRIBUTIONS
# =========================
df[num_cols].hist(figsize=(12, 10))
plt.suptitle("Numerical Feature Distributions")
plt.show()

# =========================
# 7. CATEGORICAL DISTRIBUTIONS
# =========================
for col in cat_cols:
    plt.figure(figsize=(6, 4))
    df[col].value_counts().plot(kind="bar")
    plt.title(col)
    plt.xticks(rotation=45)
    plt.show()

# =========================
# 8. TARGET vs CATEGORICAL FEATURES
# =========================
for col in cat_cols:
    if col != "y":
        plt.figure(figsize=(8, 4))
        sns.countplot(x=col, hue="y", data=df)
        plt.title(f"{col} vs Target")
        plt.xticks(rotation=45)
        plt.show()

# =========================
# 9. TARGET vs NUMERICAL FEATURES
# =========================
for col in num_cols:
    if col != "y":
        plt.figure(figsize=(6, 4))
        sns.boxplot(x="y", y=col, data=df)
        plt.title(f"{col} vs Target")
        plt.show()

# =========================
# 10. OUTLIER DETECTION
# =========================
for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Outliers in {col}")
    plt.show()

# =========================
# 11. CORRELATION ANALYSIS
# =========================
plt.figure(figsize=(12, 8))
sns.heatmap(df[num_cols].corr(), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Correlation with target (if y is encoded as 0/1)
if df["y"].dtype != "object":
    corr_target = df[num_cols].corr()["y"].sort_values(ascending=False)
    print("\nCorrelation with Target:\n", corr_target)

# =========================
# 12. BUSINESS INSIGHTS (WRITE YOUR OBSERVATIONS HERE)
# =========================

# Example insights (replace with your findings):

print("\n--- BUSINESS INSIGHTS ---")
print("1. Target variable is imbalanced.")
print("2. Duration appears to influence subscription likelihood.")
print("3. Certain categorical features show strong separation between classes.")
print("4. Outliers exist in financial variables like balance.")
print("5. Previous campaign interactions may impact conversion.")
