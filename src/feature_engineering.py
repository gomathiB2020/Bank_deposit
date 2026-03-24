import pandas as pd
import numpy as np

def add_features(df):
    df = df.copy()

    # Ratio features
    df["balance_per_age"] = df["balance"] / (df["age"] + 1)

    # Interaction features
    df["duration_balance"] = df["duration"] * df["balance"]
    df["campaign_intensity"] = df["campaign"] / (df["previous"] + 1)

    # Binning
    df["age_group"] = pd.cut(df["age"], bins=[18,30,50,80], labels=["young","mid","senior"])

    return df
