from sklearn.metrics import classification_report, roc_auc_score

def evaluate(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    print(classification_report(y, y_pred))
    print("ROC-AUC:", roc_auc_score(y, y_prob))
