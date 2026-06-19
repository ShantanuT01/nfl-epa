"""
Offline DQN — NFL play-call dataset
=====================================
Trains a Deep Q-Network purely from the q_dataset parquet produced by
the DuckDB SQL pipeline (ep_labels → joined → q_dataset).

Schema expected in the parquet
--------------------------------
Observation features (6):
    down, ydstogo, yardline_100, score_diff,
    seconds_left_in_half, game_half          (H1=0 / H2=1)

Action (SQL-encoded integer):
    play_type  →  pass=0, run=1, punt=2, other=3
    action     →  original play_type string (kept for label mapping)

Reward:
    reward     (score_diff delta, sign-adjusted for possession flip)

Next-state features (6, NULL → 0 on terminal plays):
    next_down, next_ydstogo, next_yardline_100,
    next_score_diff, next_seconds_left_in_half, next_game_half

Terminal flag:
    terminal_play  (1 = end of scoring drive / game)

Possession flip:
    change_of_possession  (+1 = same team keeps ball, -1 = opponent gets ball)
    Applied to the Bellman bootstrap: the next state's Q-value represents the
    *opponent's* best outcome after a turnover/punt, so it must be negated.

Quick-start
-----------
    python offline_dqn.py --data data/q_dataset.parquet
    python offline_dqn.py --data data/q_dataset.parquet --cql_alpha 0
    python offline_dqn.py --data data/q_dataset.parquet --epochs 200 --batch_size 128

Offline-specific notes
-----------------------
* CQL penalty toggled via --cql_alpha (0 = plain Bellman).
* Terminal next-states are zeroed out; Bellman target uses (1 - done) mask.
* Target network hard-updated every --target_update_freq gradient steps.
"""

import argparse
import random
from argparse import Namespace
import duckdb
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import yaml
import os
from pathlib import Path
# ──────────────────────────────────────────
# Column layout (must match SQL output)
# ──────────────────────────────────────────

from ep.config import get_args
from ep.model import train_dqn, QNetwork, BayesianQNetwork, get_action_values, train_bc_baseline_model
from ep.dataset import OBS_DIM, OBS_COLS



# ──────────────────────────────────────────
# 8.  Entry point
# ──────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    file_args = parser.parse_args()


    with open(file_args.config) as stream:
        args = yaml.safe_load(stream)
        
    args = Namespace(**args)
    #args = get_args()
    for seed in range(args.start_seed, args.end_seed + 1):
        args.seed = seed
        q_online = QNetwork(OBS_DIM, 4, args.hidden_dim).to(args.device)
        q_target = QNetwork(OBS_DIM, 4, args.hidden_dim).to(args.device)
        q_target.load_state_dict(q_online.state_dict())
        Path(args.checkpoint_path).mkdir(parents=True, exist_ok=True)
        trained_model, action_labels = train_dqn(args, q_online, q_target)
        data = torch.load(os.path.join(args.checkpoint_path,f"{seed}.pt"),weights_only=False)
        Path(f"policy/{args.test_season}").mkdir(parents=True, exist_ok=True)
        train_bc_baseline_model(args.train_start_season, args.train_end_season, args.test_season,f"policy/{args.test_season}/baseline_{args.seed}.parquet", args.seed)
        action_labels = data["action_labels"]
        trained_model = QNetwork(OBS_DIM, len(action_labels), args.hidden_dim)
        trained_model.load_state_dict(data["model_state"])
        trained_model = trained_model.to(args.device)

        
        rows = list()
      
        df = pd.read_parquet(f"policy/{args.test_season}/baseline_{args.seed}.parquet")
        states = df[OBS_COLS].to_numpy()#.to_list()
        for state in tqdm(states):
            sample_obs = state #np.array([1.0, 20.0, 65.0, 0.0, 900.0, 0.0], dtype=np.float32)
            device     = next(trained_model.parameters()).device
            row  = get_action_values(trained_model, sample_obs, action_labels, device)
            rows.append(row)

        q = pd.DataFrame(rows)#.to_csv("ep_4th_model.csv",index=False)
        for col in q:
            df[col] = q[col].to_list()
        df["EP"] = df["pass"] * df["pass_prob"] + df["run"] * df["run_prob"] + df["punt"] * df["punt_prob"] + df["field_goal"] * df["field_goal_prob"]
        df["seed"] = seed
        Path(f"evaluations/{args.test_season}").mkdir(parents=True, exist_ok=True)
        df.to_parquet(f"evaluations/{args.test_season}/{args.seed}.parquet")
   