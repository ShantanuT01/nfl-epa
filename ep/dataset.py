from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import duckdb 
import numpy as np

class NFLTransitionDataset(Dataset):
    def __init__(self, obs, actions, rewards, next_obs, dones, cops,
                 device: torch.device):
        self.obs      = torch.from_numpy(obs).to(device)
        self.actions  = torch.from_numpy(actions).to(device)
        self.rewards  = torch.from_numpy(rewards).to(device)
        self.next_obs = torch.from_numpy(next_obs).to(device)
        self.dones    = torch.from_numpy(dones).to(device)
        self.cops     = torch.from_numpy(cops).to(device)   # +1 / -1

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return (self.obs[idx], self.actions[idx], self.rewards[idx],
                self.next_obs[idx], self.dones[idx], self.cops[idx])



OBS_COLS      = ["down", "ydstogo", "yardline_100", "seconds_left_in_half", "game_half","posteam_timeouts_remaining","defteam_timeouts_remaining"]
NEXT_OBS_COLS = ["next_down", "next_ydstogo", "next_yardline_100", "next_seconds_left_in_half", "next_game_half","next_posteam_timeouts_remaining","next_defteam_timeouts_remaining"]
ACTION_COL    = "action"       # play_type string → label-encoded int
REWARD_COL    = "reward"
DONE_COL      = "terminal_play"
COP_COL       = "change_of_possession"  # +1 same team, -1 opponent gets ball

OBS_DIM = len(OBS_COLS)       # 6





# ──────────────────────────────────────────
# 2.  Load dataset via DuckDB
# ──────────────────────────────────────────

def load_parquet(path: str, start_season: int, end_season: int) -> tuple[np.ndarray, ...]:
    """
    Run the q_dataset query against the saved parquet and return
    numpy arrays ready for PyTorch.

    Returns
    -------
    obs, actions, rewards, next_obs, dones, cops, action_labels
        cops          : (N,) float32  +1 same possession, -1 flipped
        action_labels : list[str]     index → play_type string
    """
    con = duckdb.connect()

    # ── read parquet (DuckDB can query it directly)
    #print(start_season, end_season)
    df = con.execute(f"SELECT * FROM read_parquet('{path}') WHERE season BETWEEN {start_season} AND {end_season};").df()
  
    con.close()

    print(f"Loaded {len(df):,} transitions from {path}")
    print(f"  play_type counts:\n{df[ACTION_COL].value_counts().to_string()}\n")

    # ── action labels: SQL encodes pass=0, run=1, punt=2, other=3
    # Derive the index→name mapping from the action string column
    action_labels = (
        df[["play_type", ACTION_COL]]
        .drop_duplicates()
        .sort_values("play_type")
        .set_index("play_type")[ACTION_COL]
        .to_dict()
    )  # {0: "pass", 1: "run", 2: "punt", 3: "..."}
    n_actions = df["play_type"].max() + 1
    print(f"  Action mapping: {action_labels}\n")

    # ── observations (game_half already 0/1/2 integer from SQL)
    obs      = df[OBS_COLS].astype(np.float32).values          # (N, 6)
    actions  = df["play_type"].astype(np.int64).values          # (N,) SQL-encoded
    rewards  = df[REWARD_COL].astype(np.float32).values         # (N,)
    dones    = df[DONE_COL].astype(np.float32).values           # (N,)

    # ── next observations — NULL on terminal rows → 0 (masked out by done flag)
    next_obs = df[NEXT_OBS_COLS].astype(np.float32).fillna(0).values  # (N, 6)

    # ── change of possession: +1 same team, -1 opponent gets ball
    cops = df[COP_COL].astype(np.float32).values                      # (N,)

    return obs, actions, rewards, next_obs, dones, cops, action_labels, n_actions