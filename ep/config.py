
import argparse

# ──────────────────────────────────────────
# 1.  Hyper-parameters
# ──────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",               type=str,   required=True,
                   help="Path to q_dataset.parquet")
    p.add_argument("--hidden_dim",         type=int,   default=128)
    p.add_argument("--batch_size",         type=int,   default=256)
    p.add_argument("--epochs",             type=int,   default=50)
    p.add_argument("--lr",                 type=float, default=3e-4)
    p.add_argument("--gamma",              type=float, default=0.8)
    p.add_argument("--cql_alpha",          type=float, default=1.0,
                   help="CQL regularisation weight (0 = plain Bellman)")
    p.add_argument("--target_update_freq", type=int,   default=5000,
                   help="Hard-copy online → target every N gradient steps")
    p.add_argument("--seed",               type=int,   default=42)
    p.add_argument("--device",             type=str,   default="mps",
                   choices=["auto", "cpu", "cuda", "mps"])
    return p.parse_args()