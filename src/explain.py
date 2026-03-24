import shap

def shap_analysis(pipeline, X):
    model = pipeline.named_steps["model"]
    X_transformed = pipeline.named_steps["prep"].transform(X)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    shap.summary_plot(shap_values, X_transformed)

   
