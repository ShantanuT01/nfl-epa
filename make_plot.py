from ep.config import get_args
from ep.model import train_dqn, QNetwork, BayesianQNetwork, get_action_values, train_bc_baseline_model, train_bc_xgboost_model
from ep.dataset import OBS_DIM, OBS_COLS
import torch
import os
import pandas as pd

import numpy as np
# ──────────────────────────────────────────
# 8.  Entry point
# ──────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib
import xgboost as xgb
from tqdm import tqdm
# For classification tasks

matplotlib.rcParams['figure.dpi'] = 300

import numpy as np

if __name__ == "__main__":
    label_map = {1: "1st",2:"2nd",3:"3rd",4:"4th"}
    color_map = {1: "red",2:"green",3:"blue",4:"purple"}
    for down in range(1,5):
        df_list = list()
        for m in range(1):
            data = torch.load(os.path.join(f"models/2021/{m}.pt"),weights_only=False)
            trained_model = QNetwork(OBS_DIM, 4, 128)
            trained_model.load_state_dict(data["model_state"])
            trained_model = trained_model.to("cuda")
            action_labels = data["action_labels"]
            #["down", "ydstogo", "yardline_100",
                        # "score_diff", "seconds_left_in_half", "game_half","posteam_timeouts_remaining","defteam_timeouts_remaining"]
         
            model = xgb.XGBClassifier()
            model.load_model(f"policy/2021/baseline_{m}.ubj")
            action_space = ["pass","run","punt","field_goal"]
            rows = list()
            for i in tqdm(range(10, 100)):
                downs = [0] * 4
                downs[down - 1] = 1
                game_state = downs + [1.0,0.5,0.5, 1,1,1]
                game_state[5] = i*1.0/100.0
                        
                row  = get_action_values(trained_model, np.array(game_state), action_labels, "cuda")
                action_probs = model.predict_proba([np.array(game_state)])[0]
                ep = 0.0
                for ai, action in enumerate(action_space):
                    ep += action_probs[ai] * row[action]
                
                row["yards_away_from_end_zone"] = i
                row["ep"] = ep
                rows.append(row)
            df_list.append(pd.DataFrame(rows))#.to_parquet("test.parquet",index=False)
        print(len(df_list))
        df_average = pd.concat(df_list).groupby(level=0).mean()
        df_average.to_parquet("test.parquet")
        x = df_average["yards_away_from_end_zone"]
        y = df_average["ep"]
        df_average = pd.concat(df_list).groupby(level=0).std()
        std_devs = df_average["ep"]
        df_average.to_parquet("test_sd.parquet")

    
        # ----------------------------------------------------
        # Method 1: Discrete Error Bars for Each Point
        # ----------------------------------------------------
        # Passing the array to 'yerr' automatically maps each SD to its respective point
    

        # ----------------------------------------------------
        # Method 2: Continuous Shaded Standard Deviation Band
        # ----------------------------------------------------
        # Plot the main trend line
        
        plt.plot(x, y, '-', markersize=3, color=color_map[down], label=f'{label_map[down]}')

        # Fill the unique area between (y - SD) and (y + SD) for each point
      #  plt.fill_between(x, y - 1 * std_devs, y + 1 * std_devs, color=color_map[down], alpha=0.2, label='±1 SD Region')

    plt.title('EP by Down and Field Position - 10 Yards to Go')
    plt.xlabel('Yards to Opponent\'s End Zone')
    plt.ylabel('EP')
    plt.grid(True, alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig("EP_ovr_2017.png")