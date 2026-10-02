from ep.config import get_args
from ep.model import train_dqn, QNetwork, BayesianQNetwork, get_action_values, train_bc_baseline_model, train_bc_xgboost_model
from ep.dataset import OBS_DIM, OBS_COLS
import torch
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import xgboost as xgb
from tqdm import tqdm

matplotlib.rcParams['figure.dpi'] = 300

if __name__ == "__main__":
    DOWN = 4
    YARDLINE_MAX = 90          # yards from end zone: 0..90 (per your truncation)
    YDSTOGO_MAX = 10           # yards to go: 1..10

    downs_onehot = [0, 0, 0, 0]
    downs_onehot[DOWN - 1] = 1

    action_space = ["pass", "run", "punt", "field_goal"]

    all_rows = []

    # for m in tqdm(range(20), desc="models"):
    #     data = torch.load(os.path.join(f"models/2025/{m}.pt"), weights_only=False)
    #     trained_model = QNetwork(OBS_DIM, 4, 64)
    #     trained_model.load_state_dict(data["model_state"])
    #     trained_model = trained_model.to("cuda")
    #     action_labels = data["action_labels"]

    #     policy_model = xgb.XGBClassifier()
    #     policy_model.load_model(f"policy/2025/baseline_{m}.ubj")

    #     for ydstogo in range(1, YDSTOGO_MAX + 1):
    #         for yardline in range(10, 100):
    #             if yardline < ydstogo:
    #                 continue  # impossible: can't need more yards than remain to the end zone

    #             game_state = downs_onehot + [
    #                 ydstogo / 10.0,     # ydstogo
    #                 yardline / 100.0,   # yardline_100
    #                 0.5,                # seconds_left_in_half (halfway through the half)
    #                 1,                  # game_half (1 = second half -> start of Q4, as confirmed)
    #                 1,                  # posteam_timeouts_remaining (full, /3)
    #                 1,                  # defteam_timeouts_remaining (full, /3)
    #             ]

    #             q_row = get_action_values(trained_model, np.array(game_state), action_labels, "cuda")
    #             action_probs = policy_model.predict_proba([np.array(game_state)])[0]

    #             ep = 0.0
    #             prob_sum = 0.0
    #             for ai, action in enumerate(action_space):
    #                 if action_probs[ai] >= 0.01:
    #                     ep += action_probs[ai] * q_row[action]
    #                     prob_sum += action_probs[ai]

    #             all_rows.append({
    #                 "model": m,
    #                 "ydstogo": ydstogo,
    #                 "yardline": yardline,
    #                 "ep": ep / prob_sum,
    #             })

    #df = pd.DataFrame(all_rows)
    #df.to_parquet("fourth_down_ep_grid.parquet", index=False)
    df = pd.read_parquet("fourth_down_ep_grid.parquet")
    # Variance of EP across the 20 ensemble models, per (ydstogo, yardline) cell
    var_grid = df.pivot_table(index="ydstogo", columns="yardline", values="ep", aggfunc="std")

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(
        var_grid.columns, var_grid.index, var_grid.values,
        cmap="viridis", shading="nearest"
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("SD(EP) across ensemble (20 seeds)")

    ax.set_xlabel("Yards from Opponent's End Zone")
    ax.set_ylabel("Yards to Go")
    ax.set_title("4th Down: Ensemble SD in EP by Distance and Field Position\n(Start of 4th Quarter, Full Timeouts)")
    ax.set_yticks(range(1, YDSTOGO_MAX + 1))

    plt.tight_layout()
    plt.savefig("fourth_down_ep_SD_heatmap.png")