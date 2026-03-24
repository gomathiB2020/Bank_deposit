import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, RocCurveDisplay

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(r"C:\Users\gomathi.srinivasan\Documents\Bank_deposit\data\bank.csv", sep=";")

df["y"] = df["y"].map({"yes": 1, "no": 0})

# -----------------------------
# FEATURE ENGINEERING (MUST MATCH TRAINING)
# -----------------------------
df["age_group"] = df["age"].apply(
    lambda x: "young" if x < 30 else ("adult" if x < 60 else "senior")
)

df["balance_per_age"] = df["balance"] / (df["age"] + 1)
df["campaign_intensity"] = df["campaign"] / (df["duration"] + 1)
df["duration_balance"] = df["duration"] * df["balance"]

# Fix infinities / NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# -----------------------------
# SPLIT
# -----------------------------
X = df.drop("y", axis=1)
y = df["y"]

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("models/final_model.pkl")

# -----------------------------
# PREDICTIONS
# -----------------------------
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(y, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# -----------------------------
# ROC CURVE
# -----------------------------
fpr, tpr, _ = roc_curve(y, y_prob)

roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
roc_display.plot()

plt.title("ROC Curve")
plt.show()
