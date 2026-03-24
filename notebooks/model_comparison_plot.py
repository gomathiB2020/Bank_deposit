import matplotlib.pyplot as plt

# Example results from your training
models = ["LogReg", "RandomForest", "XGBoost"]
roc_auc_scores = [0.9101, 0.9258, 0.9311]

plt.figure()
plt.bar(models, roc_auc_scores)
plt.title("Model Comparison (ROC-AUC)")
plt.xlabel("Models")
plt.ylabel("ROC-AUC Score")
plt.show()
