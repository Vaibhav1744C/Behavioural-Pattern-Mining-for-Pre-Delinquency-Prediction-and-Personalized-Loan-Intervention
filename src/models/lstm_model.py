"""
lstm_model.py
-------------
Phase 5: LSTM + attention temporal model for pre-delinquency prediction.

Architecture (per spec):
  Input  : (batch, 24, 7)   per-month behavioural features
  Context: (batch, 6)        sequence-level summary features (seq_len excluded)
  LSTM   : 2-layer, hidden_size=64, unidirectional, dropout=0.2
  Attn   : masked attention pooling over timesteps (padded positions -> -inf)
  Fusion : concat(attn_pooled[64], context_dense[16]) -> Linear(80->64) ->
           ReLU -> Dropout(0.3) -> Linear(64->1) -> logit
  Loss   : BCEWithLogitsLoss (no pos_weight for primary run)

Usage:
    # Full training run
    python src/models/lstm_model.py

    # Ablation run (alpha=0 sequences)
    python src/models/lstm_model.py --sequences-file lstm_sequences_dense_alpha0.npy
                                    --results-file   lstm_results_alpha0.json

    # Quick smoke test (small subset)
    python src/models/lstm_model.py --sample 5000 --epochs 2

Prerequisites:
    python src/models/prepare_lstm_data.py    (builds dense .npy + train IDs)

Outputs:
    src/models/lstm_best.pt                   best checkpoint (val AUC)
    reports/lstm_results.json                 metrics vs XGBoost baseline
    reports/lstm_attention_samples.json       attention weights for ~50 test borrowers

Requirements:
    pip install torch scikit-learn pandas pyarrow numpy
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[2]
DENSE_PATH    = ROOT / "data" / "processed" / "lstm_sequences_dense.npy"
INDEX_PATH    = ROOT / "data" / "processed" / "lstm_loan_id_index.parquet"
CONTEXT_PATH  = ROOT / "data" / "processed" / "sequence_context_features.parquet"
CLEAN_PATH    = ROOT / "data" / "processed" / "lending_club_clean.parquet"
X_TEST_PATH   = ROOT / "data" / "processed" / "X_test.parquet"
TRAIN_IDS_PATH= ROOT / "data" / "processed" / "train_loan_ids.parquet"
CHECKPOINT_OUT= ROOT / "src"  / "models"    / "lstm_best.pt"
RESULTS_OUT   = ROOT / "reports"            / "lstm_results.json"
ATTN_OUT      = ROOT / "reports"            / "lstm_attention_samples.json"

MAX_LEN    = 24
N_FEATURES = 7
N_CONTEXT  = 6   # seq_len is excluded from model input (masking only)
RANDOM_SEED= 42

# Context feature columns fed to model (seq_len excluded)
CONTEXT_COLS = [
    "salary_stability_idx",
    "salary_delay_trend",
    "savings_slope",
    "savings_volatility",
    "cashflow_compression",
    "emi_stress_count",
]


# ── Dataset ───────────────────────────────────────────────────────────────────

class LoanSequenceDataset(Dataset):
    """
    Indexes into pre-loaded in-memory arrays — no I/O per __getitem__.

    seq_arr   : (N, 24, 7)  float32 numpy array (already in RAM)
    ctx_arr   : (N, 6)      float32 numpy array
    labels    : (N,)        int8 numpy array
    seq_lens  : (N,)        int16 numpy array (actual sequence length 1..24)
    loan_ids  : (N,)        int64 numpy array (for attention sample lookup)
    """

    def __init__(
        self,
        seq_arr:  np.ndarray,
        ctx_arr:  np.ndarray,
        labels:   np.ndarray,
        seq_lens: np.ndarray,
        loan_ids: np.ndarray,
    ):
        self.seq   = torch.from_numpy(seq_arr).float()
        self.ctx   = torch.from_numpy(ctx_arr).float()
        self.labels= torch.from_numpy(labels.astype(np.float32))
        self.lens  = torch.from_numpy(seq_lens.astype(np.int64))
        self.ids   = loan_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.seq[i], self.ctx[i], self.labels[i], self.lens[i], self.ids[i]


# ── Model ─────────────────────────────────────────────────────────────────────

class LSTMPredictor(nn.Module):
    """
    LSTM + masked attention + context fusion for binary default prediction.

    Returns (logit, attn_weights):
      logit       : (batch,)    raw logit (apply sigmoid for probability)
      attn_weights: (batch, 24) attention distribution over timesteps
    """

    def __init__(
        self,
        n_features:   int = N_FEATURES,
        hidden_size:  int = 64,
        num_layers:   int = 2,
        lstm_dropout: float = 0.2,
        n_context:    int = N_CONTEXT,
        context_dim:  int = 16,
        fusion_dim:   int = 64,
        head_dropout: float = 0.3,
    ):
        super().__init__()

        self.hidden_size = hidden_size

        # LSTM — unidirectional (bidirectional breaks causality)
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
        )

        # Attention scoring: one score per timestep
        self.attn_score = nn.Linear(hidden_size, 1, bias=False)

        # Context branch
        self.context_fc = nn.Sequential(
            nn.Linear(n_context, context_dim),
            nn.ReLU(),
        )

        # Fusion + classification head
        self.head = nn.Sequential(
            nn.Linear(hidden_size + context_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(fusion_dim, 1),
        )

    def forward(
        self,
        seq:      torch.Tensor,   # (B, 24, 7)
        ctx:      torch.Tensor,   # (B, 6)
        seq_lens: torch.Tensor,   # (B,) actual lengths, int64
    ):
        B = seq.size(0)

        # ── LSTM with packing (handles variable lengths correctly) ────────────
        packed = pack_padded_sequence(
            seq, seq_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out_packed, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(
            lstm_out_packed, batch_first=True, total_length=MAX_LEN
        )
        # lstm_out: (B, 24, hidden_size)

        # ── Masked attention pooling ──────────────────────────────────────────
        scores = self.attn_score(lstm_out).squeeze(-1)      # (B, 24)

        # Build mask: True at padded positions (t >= seq_len)
        t_idx  = torch.arange(MAX_LEN, device=seq.device).unsqueeze(0)  # (1, 24)
        mask   = t_idx >= seq_lens.unsqueeze(1)              # (B, 24) — True = padded

        scores = scores.masked_fill(mask, float("-inf"))
        attn_w = torch.softmax(scores, dim=1)               # (B, 24)

        # Weighted sum over timesteps
        pooled = (attn_w.unsqueeze(-1) * lstm_out).sum(dim=1)  # (B, hidden_size)

        # ── Context branch ────────────────────────────────────────────────────
        ctx_dense = self.context_fc(ctx)                     # (B, context_dim)

        # ── Fusion → logit ────────────────────────────────────────────────────
        fused = torch.cat([pooled, ctx_dense], dim=1)       # (B, hidden+context_dim)
        logit = self.head(fused).squeeze(-1)                 # (B,)

        return logit, attn_w


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_data(dense_path: Path, sample_n: int | None = None):
    """
    Load everything into RAM once. Returns aligned arrays for all borrowers.
    """
    print("Loading dense sequence array...")
    seq_arr = np.load(dense_path)              # (N, 24, 7)
    print(f"  seq_arr shape : {seq_arr.shape}")

    print("Loading loan_id index...")
    idx_df  = pd.read_parquet(INDEX_PATH)      # loan_id, row_index
    id_to_row = dict(zip(idx_df["loan_id"], idx_df["row_index"]))

    print("Loading context features...")
    ctx_df  = pd.read_parquet(CONTEXT_PATH)
    # Ensure seq_len is present for masking but not in model input
    assert "seq_len" in ctx_df.columns, "seq_len missing from context features"
    seq_lens_map = dict(zip(ctx_df["loan_id"], ctx_df["seq_len"]))

    # Normalise context features (global z-score — context indices are comparable across borrowers)
    ctx_feat = ctx_df[CONTEXT_COLS].copy()
    for col in CONTEXT_COLS:
        ctx_feat[col] = pd.to_numeric(ctx_feat[col], errors="coerce").fillna(0.0)
    ctx_mean = ctx_feat.mean()
    ctx_std  = ctx_feat.std().replace(0, 1)
    ctx_feat = ((ctx_feat - ctx_mean) / ctx_std).fillna(0.0)
    ctx_df[CONTEXT_COLS] = ctx_feat

    print("Loading labels...")
    clean_df = pd.read_parquet(CLEAN_PATH, columns=["early_default"])
    clean_df = clean_df.reset_index().rename(columns={"index": "loan_id"})
    label_map = dict(zip(clean_df["loan_id"], clean_df["early_default"]))

    # Align: use only loan_ids present in both index and context
    all_ids = idx_df["loan_id"].tolist()
    ctx_ids = set(ctx_df["loan_id"].tolist())
    all_ids = [lid for lid in all_ids if lid in ctx_ids and lid in label_map]

    if sample_n:
        all_ids = all_ids[:sample_n]
        print(f"  Subsetting to {len(all_ids):,} borrowers (--sample)")

    print(f"  Aligned borrowers: {len(all_ids):,}")

    # Build aligned arrays
    rows     = np.array([id_to_row[lid] for lid in all_ids], dtype=np.int64)
    seq_out  = seq_arr[rows]                                      # (N, 24, 7)

    ctx_map  = ctx_df.set_index("loan_id")
    ctx_out  = ctx_map.loc[all_ids, CONTEXT_COLS].values.astype(np.float32)

    labels   = np.array([label_map[lid] for lid in all_ids], dtype=np.int8)
    seq_lens = np.array([max(1, seq_lens_map.get(lid, MAX_LEN)) for lid in all_ids],
                        dtype=np.int16)
    loan_ids = np.array(all_ids, dtype=np.int64)

    print(f"  Labels: {labels.sum():,} defaults / {(labels==0).sum():,} non-defaults "
          f"({labels.mean()*100:.1f}% default rate)")

    return seq_out, ctx_out, labels, seq_lens, loan_ids, all_ids


def split_indices(all_ids: list, loan_ids_arr: np.ndarray) -> tuple:
    """
    Split into train+val / test using the same loan_ids as Phase 2.
    Returns (train_val_mask, test_mask) boolean arrays over loan_ids_arr.
    """
    if TRAIN_IDS_PATH.exists():
        train_ids = set(pd.read_parquet(TRAIN_IDS_PATH)["loan_id"].tolist())
    else:
        # Fallback: reproduce Phase 2 split
        print("  WARNING: train_loan_ids.parquet not found — reproducing split")
        x_test     = pd.read_parquet(X_TEST_PATH)
        # X_test was built from clean data with reset_index as loan_id
        clean      = pd.read_parquet(CLEAN_PATH, columns=["early_default"])
        clean      = clean.reset_index().rename(columns={"index": "loan_id"})
        _, test_df = train_test_split(
            clean, test_size=0.30, random_state=RANDOM_SEED,
            stratify=clean["early_default"]
        )
        test_ids   = set(test_df["loan_id"].tolist())
        train_ids  = set(all_ids) - test_ids

    test_mask     = np.array([lid not in train_ids for lid in loan_ids_arr], dtype=bool)
    train_val_mask= ~test_mask
    return train_val_mask, test_mask


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for seq, ctx, labels, lens, _ in loader:
        seq, ctx, labels, lens = (
            seq.to(device), ctx.to(device), labels.to(device), lens.to(device)
        )
        optimizer.zero_grad()
        logit, _ = model(seq, ctx, lens)
        loss     = criterion(logit, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0
    for seq, ctx, labels, lens, _ in loader:
        seq, ctx, labels, lens = (
            seq.to(device), ctx.to(device), labels.to(device), lens.to(device)
        )
        logit, _ = model(seq, ctx, lens)
        loss      = criterion(logit, labels)
        total_loss += loss.item() * len(labels)
        all_logits.append(torch.sigmoid(logit).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    probs  = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    auc    = roc_auc_score(labels, probs)
    return total_loss / len(loader.dataset), auc, probs, labels


# ── Attention sample collection ───────────────────────────────────────────────

@torch.no_grad()
def collect_attention_samples(
    model, loader, device, n_samples: int = 50
) -> list:
    """
    Collect attention weights for n_samples test borrowers.
    Mix: correct predictions (default+non-default) and incorrect predictions.
    """
    model.eval()
    records = []
    for seq, ctx, labels, lens, ids in loader:
        seq, ctx, lens = seq.to(device), ctx.to(device), lens.to(device)
        logit, attn_w  = model(seq, ctx, lens)
        probs           = torch.sigmoid(logit).cpu().numpy()
        attn_np         = attn_w.cpu().numpy()
        labels_np       = labels.numpy()
        ids_np          = ids.numpy()

        for i in range(len(labels_np)):
            pred = int(probs[i] >= 0.5)
            records.append({
                "loan_id":       int(ids_np[i]),
                "true_label":    int(labels_np[i]),
                "pred_prob":     round(float(probs[i]), 4),
                "pred_label":    pred,
                "correct":       pred == int(labels_np[i]),
                "seq_len":       int(lens[i].cpu()),
                "attn_weights":  [round(float(w), 5) for w in attn_np[i]],
            })
        if len(records) >= n_samples * 4:   # collect extra, then sample
            break

    # Pick a balanced mix
    correct_def   = [r for r in records if r["correct"]  and r["true_label"] == 1][:12]
    correct_nodef = [r for r in records if r["correct"]  and r["true_label"] == 0][:13]
    wrong_def     = [r for r in records if not r["correct"] and r["true_label"] == 1][:13]
    wrong_nodef   = [r for r in records if not r["correct"] and r["true_label"] == 0][:12]
    return (correct_def + correct_nodef + wrong_def + wrong_nodef)[:n_samples]


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Phase 5: LSTM training")
    p.add_argument("--sequences-file", type=str, default=None,
                   help="Dense .npy filename in data/processed/ (default: lstm_sequences_dense.npy)")
    p.add_argument("--results-file",   type=str, default="lstm_results.json",
                   help="Output results filename in reports/ (default: lstm_results.json)")
    p.add_argument("--epochs",         type=int, default=30)
    p.add_argument("--batch-size",     type=int, default=1024)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--hidden-size",    type=int, default=64)
    p.add_argument("--sample",         type=int, default=None,
                   help="Train on N borrowers only (quick smoke test)")
    p.add_argument("--pos-weight",     action="store_true",
                   help="Use pos_weight in BCEWithLogitsLoss (secondary comparison)")
    return p.parse_args()


def main():
    args    = parse_args()
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 64)
    print("Phase 5 — LSTM Temporal Model Training")
    print(f"Device: {device}")
    print("=" * 64)

    # ── Resolve paths ─────────────────────────────────────────────────────────
    dense_path   = ROOT / "data" / "processed" / (args.sequences_file or "lstm_sequences_dense.npy")
    results_path = ROOT / "reports" / args.results_file

    if not dense_path.exists():
        raise FileNotFoundError(
            f"Dense array not found: {dense_path}\n"
            "Run: python src/models/prepare_lstm_data.py"
        )

    # ── Load data ─────────────────────────────────────────────────────────────
    seq_arr, ctx_arr, labels, seq_lens, loan_ids, all_ids = load_all_data(
        dense_path, sample_n=args.sample
    )

    # ── Split ─────────────────────────────────────────────────────────────────
    print("\nSplitting train+val / test...")
    tv_mask, test_mask = split_indices(all_ids, loan_ids)

    # Internal 90/10 val split within train
    tv_indices  = np.where(tv_mask)[0]
    tv_labels   = labels[tv_indices]
    tr_idx, val_idx = train_test_split(
        tv_indices, test_size=0.10, random_state=RANDOM_SEED, stratify=tv_labels
    )
    test_idx = np.where(test_mask)[0]

    print(f"  Train : {len(tr_idx):,}  Val: {len(val_idx):,}  Test: {len(test_idx):,}")

    def make_ds(idx):
        return LoanSequenceDataset(
            seq_arr[idx], ctx_arr[idx], labels[idx],
            seq_lens[idx], loan_ids[idx]
        )

    train_loader = DataLoader(make_ds(tr_idx),  batch_size=args.batch_size,
                              shuffle=True,  pin_memory=(device.type == "cuda"), num_workers=0)
    val_loader   = DataLoader(make_ds(val_idx), batch_size=args.batch_size,
                              shuffle=False, pin_memory=(device.type == "cuda"), num_workers=0)
    test_loader  = DataLoader(make_ds(test_idx),batch_size=args.batch_size,
                              shuffle=False, pin_memory=(device.type == "cuda"), num_workers=0)

    # ── Model + optimiser ─────────────────────────────────────────────────────
    model = LSTMPredictor(hidden_size=args.hidden_size).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")

    pos_weight = None
    if args.pos_weight:
        n_neg = (labels[tr_idx] == 0).sum()
        n_pos = (labels[tr_idx] == 1).sum()
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
        print(f"  pos_weight={pos_weight.item():.2f} (secondary comparison run)")

    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_auc  = 0.0
    patience_left = 5
    history       = []

    print(f"\nTraining for up to {args.epochs} epochs (early stop patience=5 on val AUC)...\n")

    for epoch in range(1, args.epochs + 1):
        t0        = time.time()
        train_loss= train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_auc)
        elapsed   = time.time() - t0

        print(f"  Epoch {epoch:>3}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_auc={val_auc:.4f}  ({elapsed:.1f}s)")

        history.append({"epoch": epoch, "train_loss": round(train_loss, 5),
                        "val_loss": round(val_loss, 5), "val_auc": round(val_auc, 4)})

        if val_auc > best_val_auc:
            best_val_auc  = val_auc
            patience_left = 5
            torch.save(model.state_dict(), CHECKPOINT_OUT)
            print(f"    --> New best val AUC: {best_val_auc:.4f}  (checkpoint saved)")
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"\n  Early stopping at epoch {epoch} (patience exhausted)")
                break

    # ── Test evaluation ───────────────────────────────────────────────────────
    print(f"\nLoading best checkpoint (val AUC={best_val_auc:.4f})...")
    model.load_state_dict(torch.load(CHECKPOINT_OUT, map_location=device))

    _, test_auc, test_probs, test_labels = evaluate(model, test_loader, criterion, device)
    test_preds = (test_probs >= 0.5).astype(int)

    print("\n" + "=" * 64)
    print("TEST SET RESULTS")
    print("=" * 64)
    print(f"  AUC-ROC  : {test_auc:.4f}  (XGBoost baseline: 0.7345)")
    print(f"  F1       : {f1_score(test_labels, test_preds, zero_division=0):.4f}")
    print(f"  Precision: {precision_score(test_labels, test_preds, zero_division=0):.4f}")
    print(f"  Recall   : {recall_score(test_labels, test_preds, zero_division=0):.4f}")
    print(f"  Confusion matrix:\n{confusion_matrix(test_labels, test_preds)}")
    print("=" * 64)

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "model": "LSTM",
        "sequences": dense_path.name,
        "pos_weight_used": args.pos_weight,
        "best_val_auc": round(best_val_auc, 4),
        "xgboost_baseline_auc": 0.7345,
        "results": [{
            "label":            "LSTM" + (" + pos_weight" if args.pos_weight else " vanilla"),
            "auc_roc":          round(float(test_auc), 4),
            "f1":               round(float(f1_score(test_labels, test_preds, zero_division=0)), 4),
            "precision":        round(float(precision_score(test_labels, test_preds, zero_division=0)), 4),
            "recall":           round(float(recall_score(test_labels, test_preds, zero_division=0)), 4),
            "confusion_matrix": confusion_matrix(test_labels, test_preds).tolist(),
        }],
        "training_history": history,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults -> {results_path.relative_to(ROOT)}")

    # ── Attention samples ─────────────────────────────────────────────────────
    print("Collecting attention samples for Phase 6...")
    attn_samples = collect_attention_samples(model, test_loader, device, n_samples=50)
    with open(ATTN_OUT, "w") as f:
        json.dump(attn_samples, f, indent=2)
    print(f"Attention samples -> {ATTN_OUT.relative_to(ROOT)}")
    print(f"\nDone. Best test AUC: {test_auc:.4f}")


if __name__ == "__main__":
    main()
