import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def train_bc_baseline_model(start_season, end_season, eval_season):
    df = pd.read_parquet("data/offline_rl.parquet")
    X_test = df[df.season == eval_season][OBS_COLS].to_numpy()
    test_df = df[df.season == eval_season]
    df = df[df.season.between(start_season, end_season)]
    y = df["play_type"].to_numpy()
    X = df[OBS_COLS].to_numpy()

    model = RandomForestClassifier()
    model.fit(X, y)
    y_prob = model.predict_proba(X_test)
    for i, action in enumerate(["pass","run","punt","field_goal"]):
        test_df[f"{action}_prob"] = y_prob[:, i]

    test_df.to_parquet(f"policy/baseline_{eval_season}.parquet",index=False)
