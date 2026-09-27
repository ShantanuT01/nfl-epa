import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

if __name__ == "__main__":
    df = pd.read_parquet("evaluations/2025/0.parquet")
    y_obs = df["action"].replace({"pass":0,"run":1,"punt":2,"field_goal":3}).to_numpy()
    y_scores = df[["pass_prob","run_prob","punt_prob","field_goal_prob"]].max(axis=1).to_numpy()
    print(roc_auc_score(y_true=y_obs, y_score=y_scores,multi_class="ovo"))
    print(accuracy_score(y_obs, y_scores))