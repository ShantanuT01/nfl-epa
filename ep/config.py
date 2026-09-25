
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
                   help="CQL regularisation weight (0 = plain Bellman). Ignored if --lagrange_threshold is set.")
    p.add_argument("--lagrange_threshold", type=float, default=None,
                   help="If set, enables CQL-Lagrange dual optimisation: the CQL weight "
                        "(alpha) is learned via gradient ascent to keep the CQL penalty "
                        "near this target value, instead of using a fixed --cql_alpha")
    p.add_argument("--lagrange_lr",        type=float, default=3e-4,
                   help="Learning rate for the CQL-Lagrange dual variable (log_alpha)")
    p.add_argument("--target_update_freq", type=int,   default=5000,
                   help="Hard-copy online → target every N gradient steps")
    p.add_argument("--seed",               type=int,   default=42)
    p.add_argument("--device",             type=str,   default="mps",
                   choices=["auto", "cpu", "cuda", "mps"])
    return p.parse_args()