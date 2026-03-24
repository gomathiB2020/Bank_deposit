from sklearn.metrics import precision_recall_curve
import numpy as np

def find_best_threshold(model, X, y):
    probs = model.predict_proba(X)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y, probs)

    # Avoid division issues
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)

    best_idx = np.argmax(f1_scores)

    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    return best_threshold, best_f1   # ✅ RETURN BOTH
