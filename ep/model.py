import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Embedding, ModuleList
import torch
import random 
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from ep.dataset import *
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os

from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
from xgboost import XGBClassifier

# Behavior Cloning
def train_bc_baseline_model(start_season, end_season, eval_season,bc_path,seed):
    df = pd.read_parquet("data/offline_rl.parquet")
    X_test = df[df.season == eval_season][OBS_COLS].to_numpy()
    test_df = df[df.season == eval_season]
    df = df[df.season.between(start_season, end_season)]
    y = df["play_type"].to_numpy()
    X = df[OBS_COLS].to_numpy()

    model = RandomForestClassifier(random_state=seed)
    model.fit(X, y)
    y_prob = model.predict_proba(X_test)
    for i, action in enumerate(["pass","run","punt","field_goal"]):
        test_df[f"{action}_prob"] = y_prob[:, i]

    test_df.to_parquet(bc_path,index=False, compression="zstd")

# Behavior Cloning XGBoost
def train_bc_xgboost_model(start_season, end_season, eval_season,bc_predictions_path,bc_model_path, seed):
    df = pd.read_parquet("data/offline_rl.parquet")
    X_test = df[df.season == eval_season][OBS_COLS].to_numpy()
    test_df = df[df.season == eval_season]
    df = df[df.season.between(start_season, end_season)]
    y = df["play_type"].to_numpy()
    X = df[OBS_COLS].to_numpy()

    model = XGBClassifier(seed=seed, booster='dart', rate_drop=0.2, skip_drop=0.2)
    model.fit(X, y)
    y_prob = model.predict_proba(X_test)
    for i, action in enumerate(["pass","run","punt","field_goal"]):
        test_df[f"{action}_prob"] = y_prob[:, i]

    test_df.to_parquet(bc_predictions_path,index=False, compression="zstd")
    model.save_model(bc_model_path)




# Q-Networks
class AttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn = nn.Linear(embed_dim, 1)

    def forward(self, x):               # (B, num_features, embed_dim)
        weights = self.attn(x).softmax(dim=1)   # (B, num_features, 1)
        return (weights * x).sum(dim=1)          # (B, embed_dim)

class Attention(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        
        self.key = Linear(in_features=input_size, out_features=input_size)
        self.value = Linear(in_features=input_size, out_features=input_size)
        self.query = Linear(in_features=input_size, out_features=input_size)
        self.d = input_size

    def forward(self, X):
    
        Q = self.query(X)
        V = self.value(X)
        K = self.key(X)
        attn = (Q @ K.mT)/np.sqrt(self.d)
        attn_weighted = F.softmax(attn, dim=-1)
        output = attn_weighted @ V
        return output + X



class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        #return (15 - (-6))/2.0 * torch.tanh(self.net(obs)) + (15 - 6)/2.0
        return torch.tanh(self.net(obs)) * 6


@variational_estimator
class BayesianQNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            BayesianLinear(obs_dim, hidden_dim),
            nn.ReLU(),
            BayesianLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            BayesianLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            BayesianLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            BayesianLinear(hidden_dim, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        #return (15 - (-6))/2.0 * torch.tanh(self.net(obs)) + (15 - 6)/2.0
        return torch.tanh(self.net(obs)) * 6

def cql_penalty(q_online: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """
    Conservative Q-Learning penalty (simplified single-step version).
    Penalises Q-values for out-of-dataset actions via logsumexp.
    """
    logsumexp = torch.logsumexp(q_online, dim=1)
    q_data    = q_online.gather(1, actions.unsqueeze(1)).squeeze(1)
    return (logsumexp - q_data).mean()





def train_dqn(args, q_online, q_target, bayesian=False):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}\n")

    # ── data
    obs, actions, rewards, next_obs, dones, cops, action_labels, n_actions = load_parquet(args.data, args.train_start_season, args.train_end_season)

    dataset    = NFLTransitionDataset(obs, actions, rewards, next_obs, dones, cops, device)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            drop_last=True)

    # ── networks
   # q_online = QNetwork(OBS_DIM, n_actions, args.hidden_dim).to(device)
   # q_target = QNetwork(OBS_DIM, n_actions, args.hidden_dim).to(device)
   # q_target.load_state_dict(q_online.state_dict())
    q_target.eval()

    optimizer = Adam(q_online.parameters(), lr=args.lr)
    grad_step = 0

    print(f"obs_dim={OBS_DIM}  n_actions={n_actions}  "
          f"hidden={args.hidden_dim}  dataset={len(dataset):,}\n")

    # ── training
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        epoch_cql  = 0.0

        for obs_b, act_b, rew_b, nobs_b, done_b, cop_b in tqdm(dataloader):

            # Bellman target — negate max_next when possession flips (+1/-1)
            with torch.no_grad():
                max_next = q_target(nobs_b).max(dim=1).values
                target   = rew_b + args.gamma * (1.0 - done_b) * cop_b * max_next

            q_all  = q_online(obs_b)
            q_pred = q_all.gather(1, act_b.unsqueeze(1)).squeeze(1)

            bellman_loss = nn.functional.mse_loss(q_pred, target)
            cql_loss     = cql_penalty(q_all, act_b) if args.cql_alpha > 0 else 0.0
            if not bayesian:
                loss         = bellman_loss + args.cql_alpha * cql_loss
            else:
                loss = bellman_loss + args.cql_alpha * cql_loss + q_online.nn_kl_divergence() * 1.0/args.batch_size

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_online.parameters(), max_norm=10.0)
            optimizer.step()

            grad_step  += 1
            epoch_loss += bellman_loss.item()
            if args.cql_alpha > 0:
                epoch_cql += cql_loss.item()

            if grad_step % args.target_update_freq == 0:
                q_target.load_state_dict(q_online.state_dict())

        n_batches = len(dataloader)
        cql_str   = (f"  CQL: {epoch_cql / n_batches:.4f}"
                     if args.cql_alpha > 0 else "")
        print(f"Epoch {epoch:4d}/{args.epochs}  "
              f"Bellman: {epoch_loss / n_batches:.4f}{cql_str}")

    print(f"\nTraining complete. Total gradient steps: {grad_step:,}")

    ckpt_path = os.path.join(args.checkpoint_path, f"{args.seed}.pt")
    torch.save({
        "model_state": q_online.state_dict(),
        "action_labels": action_labels,
        "obs_cols": OBS_COLS,
    }, ckpt_path)
    print(f"Saved weights + metadata → {ckpt_path}")

    return q_online, action_labels



def get_action_values(model: QNetwork, obs: np.ndarray,
                  action_labels: list[str],
                  device: torch.device) -> tuple[int, str]:
    """Return (action_index, play_type_string) for a single observation."""
    model.eval()
    with torch.no_grad():
        t   = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        y_hat = model(t)
        scores = y_hat[0].detach().cpu().numpy()
        ret = dict()
        for i in range(len(action_labels)):
            ret[action_labels[i]] = scores[i]
        #idx = int(y_hat.argmax(dim=1).item())
    return ret


