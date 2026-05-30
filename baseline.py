from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pandas as pd
from sklearn.metrics import roc_auc_score

df = pd.read_parquet("ep_labels.parquet")
train_df = df[df.season.between(2016, 2019)]
test_df = df[df.season == 2020]


prob_cols = ["opp_td_prob","opp_fg_prob","opp_safety_prob","no_score_prob","safety_prob","fg_prob","td_prob"]
test_df = test_df.dropna(subset=["opp_td_prob","opp_fg_prob","opp_safety_prob","no_score_prob","safety_prob","fg_prob","td_prob"])
test_df[prob_cols] = test_df[prob_cols].div(test_df[prob_cols].sum(axis=1),axis=0)
test_df = test_df.dropna(subset=prob_cols)
y_score = test_df[prob_cols].to_numpy()

print(roc_auc_score(y_score=y_score, y_true=test_df["next_score_label"], multi_class='ovr'))


train_cols = ["down","ydstogo","yardline_100","home_advantage", "score_diff","seconds_left_in_half"]
X_train = train_df[train_cols].to_numpy()
y_train = train_df["next_score_label"].to_numpy()

X_test = test_df[train_cols].to_numpy()
y_test = test_df["next_score_label"].to_numpy()

model = XGBClassifier()
model.fit(X_train, y_train)
y_score = model.predict_proba(X_test)
#print(y_test)
print(roc_auc_score(y_score=y_score, y_true=y_test, multi_class='ovr'))
'''
pbp.no_score_prob,
  pbp.opp_fg_prob,
  pbp.opp_safety_prob,
  pbp.opp_td_prob,
  pbp.fg_prob,
  pbp.safety_prob,
  pbp.td_prob,
'''
