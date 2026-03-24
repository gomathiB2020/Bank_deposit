from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def get_models():
    models = {
        "logistic": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "rf": RandomForestClassifier(n_estimators=200),
        "xgb": XGBClassifier(scale_pos_weight=3),
        "lgbm": LGBMClassifier(),
        "catboost": CatBoostClassifier(verbose=0)
    }
    return models


def get_stacking_model():
    base_models = [
        ("rf", RandomForestClassifier()),
        ("xgb", XGBClassifier()),
        ("lgbm", LGBMClassifier())
    ]

    meta_model = LogisticRegression()

    stack = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model
    )

    return stack
