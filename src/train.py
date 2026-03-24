import pandas as pd
import joblib
import json
import os

from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn import set_config

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from preprocessing import build_preprocessor
from feature_engineering import add_features
from models import get_models, get_stacking_model
from threshold import find_best_threshold

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Fix pandas output format
set_config(transform_output="pandas")

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "bank.csv")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_PATH, sep=";")

X = df.drop("y", axis=1)
y = df["y"].map({"yes": 1, "no": 0})

# -----------------------------
# Feature engineering
# -----------------------------
X = add_features(X)

# -----------------------------
# Train / Validation split
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -----------------------------
# Preprocessing
# -----------------------------
preprocessor = build_preprocessor(X_train)

# -----------------------------
# Models
# -----------------------------
models = get_models()
models["stacking"] = get_stacking_model()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}

# -----------------------------
# Model Comparison
# -----------------------------
for name, model in models.items():
    print(f"\nRunning model: {name}")

    try:
        pipe = ImbPipeline([
            ("prep", preprocessor),
            ("smote", SMOTE()),
            ("model", model)
        ])

        score = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc").mean()
        results[name] = score

        print(f"{name}: {score:.4f}")

    except Exception as e:
        print(f"❌ Error with {name}: {e}")

# -----------------------------
# Select Best Model
# -----------------------------
best_model_name = max(results, key=results.get)
print(f"\nBest Model: {best_model_name}")

if best_model_name == "catboost":
    from catboost import CatBoostClassifier
    best_model = CatBoostClassifier(verbose=0)

    param_grid = {
        "model__depth": [4, 6],
        "model__iterations": [100, 200]
    }

elif best_model_name == "lgbm":
    from lightgbm import LGBMClassifier
    best_model = LGBMClassifier()

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 5]
    }

else:
    best_model = models[best_model_name]

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 5]
    }

# -----------------------------
# Hyperparameter Tuning
# -----------------------------
pipe = ImbPipeline([
    ("prep", preprocessor),
    ("smote", SMOTE()),
    ("model", best_model)
])

grid = GridSearchCV(
    pipe,
    param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("\nBest Params:", grid.best_params_)
print("Best CV ROC-AUC:", grid.best_score_)

# -----------------------------
# Final Model
# -----------------------------
final_model = grid.best_estimator_

# -----------------------------
# Evaluation
# -----------------------------
y_prob = final_model.predict_proba(X_val)[:, 1]
y_pred = final_model.predict(X_val)

print("\n===== EVALUATION METRICS =====")
print("Accuracy:", accuracy_score(y_val, y_pred))
print("Precision:", precision_score(y_val, y_pred))
print("Recall:", recall_score(y_val, y_pred))
print("F1 Score:", f1_score(y_val, y_pred))
print("ROC-AUC:", roc_auc_score(y_val, y_prob))

print("\nClassification Report:\n")
print(classification_report(y_val, y_pred))

# -----------------------------
# Confusion Matrix (SAVE)
# -----------------------------
ConfusionMatrixDisplay.from_predictions(y_val, y_pred)
plt.title("Confusion Matrix")

cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()

# -----------------------------
# Threshold Optimization
# -----------------------------
best_threshold, best_f1 = find_best_threshold(final_model, X_val, y_val)

print("Best Threshold:", best_threshold)
print("Best F1 Score:", best_f1)

# -----------------------------
# Save Model + Threshold
# -----------------------------
model_path = os.path.join(MODEL_DIR, "final_model.pkl")
threshold_path = os.path.join(MODEL_DIR, "threshold.json")

joblib.dump(final_model, model_path)

# SAVE AS FLOAT (IMPORTANT)
with open(threshold_path, "w") as f:
    json.dump(best_threshold, f)

print("\n✅ Training Complete!")
