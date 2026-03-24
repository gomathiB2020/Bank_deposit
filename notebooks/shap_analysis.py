import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(
    r"C:\Users\gomathi.srinivasan\Documents\Bank_deposit\data\bank.csv",
    sep=";"
)

# -----------------------------
# TARGET ENCODING
# -----------------------------
df["y"] = df["y"].map({"yes": 1, "no": 0})

# -----------------------------
# FEATURE ENGINEERING (MUST MATCH TRAINING)
# -----------------------------

# Age group
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 25, 35, 45, 60, 100],
    labels=["young", "adult", "mid", "senior", "old"]
)

# Engineered features
df["balance_per_age"] = df["balance"] / (df["age"] + 1)
df["campaign_intensity"] = df["campaign"] / (df["duration"] + 1)
df["duration_balance"] = df["duration"] * df["balance"]
df["prev_contact_ratio"] = df["previous"] / (df["pdays"] + 1)
df["log_balance"] = np.log1p(df["balance"].clip(lower=0))

# -----------------------------
# CLEAN DATA (FIX FOR CATEGORICAL ISSUE)
# -----------------------------

# Replace inf values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Identify column types
cat_cols = df.select_dtypes(include=["object", "category"]).columns
num_cols = df.select_dtypes(include=["int64", "float64"]).columns

# Convert categorical columns to string
df[cat_cols] = df[cat_cols].astype(str)

# Fill numeric NaNs
df[num_cols] = df[num_cols].fillna(0)

# Replace NaN strings
df[cat_cols] = df[cat_cols].replace("nan", "unknown")

# -----------------------------
# FEATURES
# -----------------------------
X = df.drop("y", axis=1)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("models/final_model.pkl")

print("Model type:", type(model))

# -----------------------------
# HANDLE PIPELINE
# -----------------------------
if hasattr(model, "named_steps"):
    print("Pipeline detected")

    steps = list(model.named_steps.values())
    preprocessor = steps[0]
    ml_model = steps[-1]

    # Transform data
    X_transformed = preprocessor.transform(X)

else:
    print("Direct model detected")
    ml_model = model
    X_transformed = X

# -----------------------------
# SAMPLE DATA FOR SHAP
# -----------------------------
X_sample = X_transformed[:500]

# -----------------------------
# SHAP EXPLAINER
# -----------------------------
explainer = shap.Explainer(ml_model)
shap_values = explainer(X_sample)

# -----------------------------
# VISUALIZATIONS
# -----------------------------

# Summary plot
shap.summary_plot(shap_values, X_sample)

# Feature importance bar plot
shap.summary_plot(shap_values, X_sample, plot_type="bar")

# Force plot for first prediction
shap.force_plot(
    explainer.expected_value,
    shap_values[0].values,
    X_sample[0],
    matplotlib=True
)

plt.show()
