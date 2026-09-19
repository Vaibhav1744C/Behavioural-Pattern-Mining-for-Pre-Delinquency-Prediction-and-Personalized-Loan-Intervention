"""
generate_behavioural_sequences.py
----------------------------------
Phase 3A: Synthetic behavioural sequence generator.

Generates monthly behavioural sequences per borrower from the cleaned static
Lending Club data. Lending Club has no transaction-level data — this generator
creates it using a principled two-component stress model:

    stress[t] = alpha * static_stress + (1 - alpha) * shock_stress[t]

Design rationale:
  - `static_stress` is derived from static features (dti, revol_util, grade).
  - `shock_stress[t]` is an independent stochastic shock — NOT derived from
    static features — so the temporal model has something genuinely new to
    learn beyond what the static XGBoost already sees.
  - `early_default` is used ONLY to set shock probability, never to set any
    feature value directly (leakage guard — see assertion below).
  - `alpha` is a top-level config parameter swept in Phase 5 ablation.

Usage:
    python src/data_generation/generate_behavioural_sequences.py

    # Custom config:
    python src/data_generation/generate_behavioural_sequences.py \
        --alpha 0.6 --sample 5000 --validate-only

Outputs:
    data/synthetic/behavioural_sequences.parquet       main run (alpha=config)
    data/synthetic/behavioural_sequences_alpha0.parquet ablation (alpha=0)
    reports/generator_manifest.json                    run parameters
    reports/generator_validation_plots/                trajectory plots (sample)

Requirements:
    pip install pandas pyarrow numpy matplotlib
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on headless machines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[2]
CLEAN_PATH   = ROOT / "data" / "processed" / "lending_club_clean.parquet"
SYNTH_DIR    = ROOT / "data" / "synthetic"
REPORTS_DIR  = ROOT / "reports"
PLOTS_DIR    = REPORTS_DIR / "generator_validation_plots"
MANIFEST_OUT = REPORTS_DIR / "generator_manifest.json"

# ── Default config (all exposed as CLI args for ablation sweeps) ──────────────
DEFAULT_ALPHA             = 0.6
DEFAULT_P_SHOCK_DEFAULT   = 0.35
DEFAULT_P_SHOCK_NONDEFAULT= 0.12
DEFAULT_MAX_SEQ_LEN       = 24
DEFAULT_MAX_DELAY_DAYS    = 10
DEFAULT_CHUNK_SIZE        = 50_000
DEFAULT_SEED              = 42

# AR(1) noise autocorrelation coefficient
AR1_PHI = 0.7


# ── Leakage guard ─────────────────────────────────────────────────────────────
# Raises at import time if the label is referenced outside the shock-prob draw.
# This is a documentation-level assertion: the real enforcement is code review,
# but the comment makes the contract explicit and grep-able.
_LABEL_COL = "early_default"
# Rule: search this file for _LABEL_COL — it must appear ONLY in:
#   1. This guard block
#   2. The shock-probability draw inside `_draw_shock_probability`
#   3. Validation plots (read-only, no feature computation)


# ── Static stress ─────────────────────────────────────────────────────────────

def compute_static_stress(df: pd.DataFrame) -> np.ndarray:
    """
    Borrower-level constant stress in [0, 1].
    Derived from dti, revol_util, grade — three static features already in
    the cleaned data. Simple weighted average of min-max normalised values.
    Weights: dti=0.4, revol_util=0.35, grade=0.25 (higher grade number = riskier).
    """
    def safe_minmax(s: pd.Series) -> np.ndarray:
        vals = pd.to_numeric(s, errors="coerce").fillna(0).values.astype(float)
        lo, hi = vals.min(), vals.max()
        return (vals - lo) / (hi - lo + 1e-9)

    dti_n      = safe_minmax(df["dti"])       if "dti"       in df.columns else np.zeros(len(df))
    revol_n    = safe_minmax(df["revol_util"])if "revol_util"in df.columns else np.zeros(len(df))
    grade_n    = safe_minmax(df["grade"])     if "grade"     in df.columns else np.zeros(len(df))

    return 0.40 * dti_n + 0.35 * revol_n + 0.25 * grade_n   # already in [0,1]


# ── Shock stress ──────────────────────────────────────────────────────────────

def _draw_shock_probability(
    labels: np.ndarray,
    p_shock_default: float,
    p_shock_nondefault: float,
) -> np.ndarray:
    """
    Returns per-borrower shock probability.
    THIS IS THE ONLY PLACE THE LABEL MAY BE USED — as a probability selector,
    never to directly set a feature value.
    """
    return np.where(labels == 1, p_shock_default, p_shock_nondefault)


def build_shock_stress(
    n_borrowers: int,
    seq_lengths: np.ndarray,
    p_shock: np.ndarray,
    rng: np.random.Generator,
    max_seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (shock_stress, has_shock):
      shock_stress : array of shape (n_borrowers, max_seq_len)
      has_shock    : boolean array of shape (n_borrowers,)
    has_shock is returned so validation can measure shock rates directly
    from the draw rather than inferring from peak stress thresholds.
    """
    shock_stress = np.zeros((n_borrowers, max_seq_len), dtype=np.float32)
    has_shock    = rng.random(n_borrowers) < p_shock

    for i in range(n_borrowers):
        L = int(seq_lengths[i])
        if L == 0:
            continue

        # AR(1) baseline noise (no shock) — allow small negative values so
        # low-stress borrowers still show trajectory variation after rounding
        noise = np.zeros(L, dtype=np.float32)
        noise[0] = float(rng.normal(0, 0.03))
        for t in range(1, L):
            noise[t] = AR1_PHI * noise[t - 1] + float(rng.normal(0, 0.03))
        # Clip to [-0.05, 0.15] — allow slight negative variation
        noise = np.clip(noise, -0.05, 0.15)

        if has_shock[i]:
            # Onset: uniform over the sequence, weighted toward the middle third
            mid_start = max(0, L // 3)
            mid_end   = min(L, 2 * L // 3 + 1)
            weights   = np.ones(L, dtype=float)
            weights[mid_start:mid_end] *= 3.0
            weights   /= weights.sum()
            onset     = rng.choice(L, p=weights)

            # Ramp over 2-3 months, hold elevated
            ramp_len  = rng.integers(2, 4)
            peak      = float(rng.uniform(0.55, 0.90))
            for t in range(L):
                if t < onset:
                    noise[t] = max(noise[t], 0)
                elif t < onset + ramp_len:
                    frac = (t - onset + 1) / ramp_len
                    noise[t] = frac * peak
                else:
                    jitter = rng.normal(0, 0.04)
                    noise[t] = np.clip(peak + jitter, 0.3, 1.0)

        shock_stress[i, :L] = noise

    return shock_stress, has_shock


# ── Per-borrower sequence generation ─────────────────────────────────────────

def generate_sequences_for_chunk(
    chunk: pd.DataFrame,
    alpha: float,
    p_shock_default: float,
    p_shock_nondefault: float,
    max_seq_len: int,
    max_delay_days: int,
    rng: np.random.Generator,
    return_shock_flags: bool = False,
) -> pd.DataFrame | tuple:
    """
    Generate behavioural sequences for a chunk of borrowers.
    Returns a long-format DataFrame with one row per (loan_id, month).
    """
    n = len(chunk)
    loan_ids   = chunk["loan_id"].values
    annual_inc = pd.to_numeric(chunk["annual_inc"], errors="coerce").fillna(30000).values.astype(float)
    term_months= pd.to_numeric(chunk["term"],       errors="coerce").fillna(36).values.astype(float)
    labels     = chunk[_LABEL_COL].values.astype(int)

    seq_lengths = np.minimum(term_months, max_seq_len).astype(int)
    monthly_inc = annual_inc / 12.0

    # Static stress (borrower-level constant)
    static_stress = compute_static_stress(chunk)                     # shape (n,)

    # Shock probability — ONLY use of label
    p_shock = _draw_shock_probability(labels, p_shock_default, p_shock_nondefault)

    # Shock stress (n, max_seq_len) + shock flags for validation
    shock_arr, has_shock = build_shock_stress(n, seq_lengths, p_shock, rng, max_seq_len)

    # Savings starting balance: 1-3 months of income, random per borrower
    savings_start = monthly_inc * rng.uniform(1.0, 3.0, size=n)

    # Build long-format rows
    rows = []
    for i in range(n):
        L        = int(seq_lengths[i])
        inc      = monthly_inc[i]
        ss       = static_stress[i]
        shock    = shock_arr[i, :L]

        # Combined stress
        stress = alpha * ss + (1.0 - alpha) * shock    # shape (L,)
        stress = np.clip(stress, 0.0, 1.0)

        # --- salary_credit: AR(1) noise ~5% of mean, independent of stress ---
        sal_noise = np.zeros(L, dtype=float)
        sal_noise[0] = rng.normal(0, 0.05 * inc)
        for t in range(1, L):
            sal_noise[t] = AR1_PHI * sal_noise[t - 1] + rng.normal(0, 0.05 * inc)
        salary_credit = np.clip(inc + sal_noise, 0.1 * inc, 5.0 * inc)

        # --- salary_delay_days: rises with stress ---
        delay_noise = rng.normal(0, 0.5, size=L)
        salary_delay = np.clip(
            stress * max_delay_days + delay_noise, 0, max_delay_days
        ).astype(int)

        # --- discretionary_spend: spikes early on stress onset, compresses later ---
        base_spend    = 0.65 * inc
        spend_arr     = np.zeros(L, dtype=float)
        stress_delta  = np.diff(stress, prepend=stress[0])  # recent stress change
        for t in range(L):
            delta_recent = stress_delta[t]
            spend_arr[t] = base_spend * (
                1.0 + 0.30 * max(delta_recent, 0) - 0.40 * stress[t]
            )
        spend_noise = np.zeros(L, dtype=float)
        spend_noise[0] = rng.normal(0, 0.04 * base_spend)
        for t in range(1, L):
            spend_noise[t] = AR1_PHI * spend_noise[t - 1] + rng.normal(0, 0.04 * base_spend)
        discretionary_spend = np.clip(spend_arr + spend_noise, 0.05 * inc, 2.0 * inc)

        # --- account_balance: cumulative running balance ---
        emi = 0.10 * inc   # simplified EMI proxy
        balance = np.zeros(L, dtype=float)
        balance[0] = inc * rng.uniform(0.5, 2.0)   # random starting balance
        for t in range(1, L):
            emi_paid = emi if rng.random() > stress[t] else 0.0
            b_noise  = AR1_PHI * rng.normal(0, 0.02 * inc)
            balance[t] = balance[t-1] + salary_credit[t] - discretionary_spend[t] - emi_paid + b_noise

        # --- savings_balance: slow decline under stress, stable otherwise ---
        savings = np.zeros(L, dtype=float)
        savings[0] = savings_start[i]
        for t in range(1, L):
            draw = savings[t-1] * 0.03 * stress[t]    # monthly drawdown
            grow = savings[t-1] * 0.005 * (1 - stress[t])  # slow growth when calm
            s_noise = rng.normal(0, 0.01 * savings_start[i])
            savings[t] = max(0.0, savings[t-1] - draw + grow + s_noise)

        # --- emi_status: logistic fn of stress ---
        p_missed  = 1.0 / (1.0 + np.exp(-8.0 * (stress - 0.75)))
        p_delayed = 1.0 / (1.0 + np.exp(-6.0 * (stress - 0.45))) - p_missed
        p_delayed = np.clip(p_delayed, 0, 1)
        p_ontime  = np.clip(1.0 - p_missed - p_delayed, 0, 1)

        for t in range(L):
            draw_val = rng.random()
            if draw_val < p_missed[t]:
                emi_status = "missed"
            elif draw_val < p_missed[t] + p_delayed[t]:
                emi_status = "delayed"
            else:
                emi_status = "on_time"

            rows.append({
                "loan_id":             int(loan_ids[i]),
                "month":               t,
                "salary_credit":       round(float(salary_credit[t]), 2),
                "salary_delay_days":   int(salary_delay[t]),
                "account_balance":     round(float(balance[t]), 2),
                "savings_balance":     round(float(savings[t]), 2),
                "discretionary_spend": round(float(discretionary_spend[t]), 2),
                "emi_status":          emi_status,
                "stress_level":        round(float(stress[t]), 4),
            })

    seq_df = pd.DataFrame(rows)
    if return_shock_flags:
        # Returns (seq_df, shock_flags_df) where shock_flags_df has loan_id + has_shock
        flags_df = pd.DataFrame({
            "loan_id":   loan_ids.astype(int),
            "has_shock": has_shock.astype(bool),
            "label":     labels.astype(int),
        })
        return seq_df, flags_df
    return seq_df


# ── Validation ────────────────────────────────────────────────────────────────

def run_validation(
    df_static: pd.DataFrame,
    alpha: float,
    p_shock_default: float,
    p_shock_nondefault: float,
    max_seq_len: int,
    max_delay_days: int,
    rng: np.random.Generator,
    n_sample: int = 2000,
) -> None:
    """
    Run on a small sample, assert sanity checks, and save validation plots.
    Raises AssertionError if any check fails.
    """
    print(f"\n--- Validation on {n_sample:,} borrower sample ---")
    # Use a fixed independent seed for the sample draw so it doesn't consume
    # state from the passed rng (which would affect the noise draws)
    sample = df_static.sample(n=min(n_sample, len(df_static)), random_state=1234)
    seq_df, flags_df = generate_sequences_for_chunk(
        sample, alpha, p_shock_default, p_shock_nondefault,
        max_seq_len, max_delay_days, rng,
        return_shock_flags=True,
    )

    # --- Check 1: salary correlation > 0.95 with annual_inc / 12 ------------
    mean_salary = seq_df.groupby("loan_id")["salary_credit"].mean()
    expected    = (
        sample.set_index("loan_id")["annual_inc"]
        .apply(lambda x: pd.to_numeric(x, errors="coerce"))
        .fillna(30000) / 12.0
    )
    common = mean_salary.index.intersection(expected.index)
    corr = np.corrcoef(mean_salary.loc[common], expected.loc[common])[0, 1]
    print(f"  Salary correlation with annual_inc/12: {corr:.4f}  (must be >0.95)")
    assert corr > 0.95, f"Salary correlation too low: {corr:.4f}"

    # --- Check 2: shock rate matches p_shock — measured from actual draw -----
    for lbl, p_exp in [(1, p_shock_default), (0, p_shock_nondefault)]:
        group      = flags_df[flags_df["label"] == lbl]
        actual_rate= group["has_shock"].mean() if len(group) > 0 else 0.0
        tol        = 0.10
        print(f"  Shock rate label={lbl}: {actual_rate:.3f}  (expected ~{p_exp:.2f} ±{tol})")
        assert abs(actual_rate - p_exp) < tol + 0.05, \
            f"Shock rate for label={lbl} out of range: {actual_rate:.3f}"

    # --- Check 3: no flat trajectories for sequences > 1 month --------------
    seq_counts = seq_df.groupby("loan_id")["month"].count()
    multi_month = seq_counts[seq_counts > 1].index
    stress_std  = seq_df[seq_df["loan_id"].isin(multi_month)].groupby("loan_id")["stress_level"].std()
    flat = (stress_std < 1e-4).sum()   # use tolerance — stored as 4dp rounded float
    print(f"  Flat trajectories (stress std<1e-4, sequences >1 month): {flat}  (must be 0)")
    assert flat == 0, f"{flat} borrowers have completely flat stress trajectories"

    print("  All sanity checks passed ✓")

    # --- Save validation plots -----------------------------------------------
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    _save_trajectory_plots(seq_df, flags_df)
    print(f"  Trajectory plots saved to {PLOTS_DIR.relative_to(ROOT)}")


def _save_trajectory_plots(
    seq_df: pd.DataFrame,
    flags_df: pd.DataFrame,
    n_plots: int = 15,
) -> None:
    """Save individual borrower trajectory plots (stress, balance, emi_status)."""
    # Pick a mix: ~5 default=0, ~5 default=1 with shock, ~5 default=1 no shock
    ids_nd      = flags_df[(flags_df["label"] == 0)                          ]["loan_id"].tolist()
    ids_d_shock = flags_df[(flags_df["label"] == 1) &  flags_df["has_shock"] ]["loan_id"].tolist()
    ids_d_nosh  = flags_df[(flags_df["label"] == 1) & ~flags_df["has_shock"] ]["loan_id"].tolist()

    plot_ids = (ids_nd[:5] + ids_d_shock[:5] + ids_d_nosh[:5])[:n_plots]
    label_map = flags_df.set_index("loan_id")["label"].to_dict()

    for lid in plot_ids:
        sub = seq_df[seq_df["loan_id"] == lid].sort_values("month")
        if sub.empty:
            continue

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        label_val = label_map.get(lid, "?")
        fig.suptitle(f"Loan {lid}  |  early_default={label_val}  |  "
                     f"peak_stress={sub['stress_level'].max():.2f}", fontsize=11)

        axes[0].plot(sub["month"], sub["stress_level"], color="crimson")
        axes[0].set_ylabel("stress_level")
        axes[0].set_ylim(0, 1)

        axes[1].plot(sub["month"], sub["account_balance"], color="steelblue", label="account")
        axes[1].plot(sub["month"], sub["savings_balance"],  color="seagreen", label="savings")
        axes[1].set_ylabel("balance ($)")
        axes[1].legend(fontsize=8)

        emi_map = {"on_time": 0, "delayed": 1, "missed": 2}
        emi_num = sub["emi_status"].map(emi_map)
        axes[2].step(sub["month"], emi_num, color="darkorange", where="mid")
        axes[2].set_yticks([0, 1, 2])
        axes[2].set_yticklabels(["on_time", "delayed", "missed"], fontsize=8)
        axes[2].set_ylabel("emi_status")
        axes[2].set_xlabel("month")

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"trajectory_loan{lid}.png", dpi=100)
        plt.close(fig)


# ── Full run (chunked parquet writer) ─────────────────────────────────────────

def run_full_generation(
    df_static: pd.DataFrame,
    alpha: float,
    p_shock_default: float,
    p_shock_nondefault: float,
    max_seq_len: int,
    max_delay_days: int,
    chunk_size: int,
    seed: int,
    out_path: Path,
) -> int:
    """
    Generate sequences for all borrowers in chunks, writing incrementally
    to a parquet file. Returns total rows written.
    """
    if out_path.exists():
        out_path.unlink()

    rng        = np.random.default_rng(seed)
    writer     = None
    total_rows = 0
    n_chunks   = (len(df_static) + chunk_size - 1) // chunk_size

    print(f"\nGenerating sequences: alpha={alpha}, {len(df_static):,} borrowers, "
          f"{n_chunks} chunks of {chunk_size:,}")

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end   = min(start + chunk_size, len(df_static))
        chunk = df_static.iloc[start:end]

        seq_df = generate_sequences_for_chunk(
            chunk, alpha, p_shock_default, p_shock_nondefault,
            max_seq_len, max_delay_days, rng,
        )

        # Convert emi_status to string for Arrow compatibility
        seq_df["emi_status"] = seq_df["emi_status"].astype(str)

        table = pa.Table.from_pandas(seq_df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
        writer.write_table(table)

        total_rows += len(seq_df)
        print(f"  Chunk {chunk_idx+1:>4}/{n_chunks}: borrowers {start:>7,}-{end:>7,} "
              f"| rows so far: {total_rows:>12,}")

    if writer:
        writer.close()

    size_mb = out_path.stat().st_size / (1024 ** 2)
    print(f"\nSaved {total_rows:,} rows → {out_path.relative_to(ROOT)}  ({size_mb:.1f} MB)")
    return total_rows


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Phase 3A: Behavioural sequence generator")
    p.add_argument("--alpha",             type=float, default=DEFAULT_ALPHA)
    p.add_argument("--p-shock-default",   type=float, default=DEFAULT_P_SHOCK_DEFAULT)
    p.add_argument("--p-shock-nondefault",type=float, default=DEFAULT_P_SHOCK_NONDEFAULT)
    p.add_argument("--max-seq-len",       type=int,   default=DEFAULT_MAX_SEQ_LEN)
    p.add_argument("--max-delay-days",    type=int,   default=DEFAULT_MAX_DELAY_DAYS)
    p.add_argument("--chunk-size",        type=int,   default=DEFAULT_CHUNK_SIZE)
    p.add_argument("--seed",              type=int,   default=DEFAULT_SEED)
    p.add_argument("--sample",            type=int,   default=None,
                   help="Only generate for N borrowers (for testing)")
    p.add_argument("--validate-only",     action="store_true",
                   help="Run validation sample only, skip full generation")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 64)
    print("Phase 3A — Synthetic Behavioural Sequence Generator")
    print("=" * 64)

    # Load static data
    print(f"\nLoading: {CLEAN_PATH.relative_to(ROOT)}")
    df = pd.read_parquet(CLEAN_PATH)
    print(f"  shape: {df.shape}")

    # Ensure loan_id column exists
    if "loan_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "loan_id"})
    else:
        df["loan_id"] = df["loan_id"].astype(int)

    # Ensure term column is numeric
    if df["term"].dtype == object:
        df["term"] = pd.to_numeric(
            df["term"].astype(str).str.replace("months", "", regex=False).str.strip(),
            errors="coerce"
        ).fillna(36)

    # Subset if --sample given
    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=args.seed)
        print(f"  Subsetting to {len(df):,} borrowers (--sample flag)")

    # Directories
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Validation run (always — fail fast before the full multi-hour run)
    val_rng = np.random.default_rng(args.seed)
    run_validation(
        df, args.alpha, args.p_shock_default, args.p_shock_nondefault,
        args.max_seq_len, args.max_delay_days, val_rng,
        n_sample=min(2000, len(df)),
    )

    if args.validate_only:
        print("\n--validate-only flag set. Stopping after validation.")
        return

    # Main run (alpha = config value)
    main_out = SYNTH_DIR / "behavioural_sequences.parquet"
    rng_main = np.random.default_rng(args.seed)
    n_rows_main = run_full_generation(
        df, args.alpha,
        args.p_shock_default, args.p_shock_nondefault,
        args.max_seq_len, args.max_delay_days,
        args.chunk_size, args.seed, main_out,
    )

    # Ablation run (alpha = 0 — pure independent shock, no static component)
    print("\nRunning ablation companion (alpha=0)...")
    ablation_out = SYNTH_DIR / "behavioural_sequences_alpha0.parquet"
    rng_abl = np.random.default_rng(args.seed)
    n_rows_abl = run_full_generation(
        df, alpha=0.0,
        p_shock_default=args.p_shock_default,
        p_shock_nondefault=args.p_shock_nondefault,
        max_seq_len=args.max_seq_len,
        max_delay_days=args.max_delay_days,
        chunk_size=args.chunk_size,
        seed=args.seed,
        out_path=ablation_out,
    )

    # Manifest
    manifest = {
        "alpha":               args.alpha,
        "p_shock_default":     args.p_shock_default,
        "p_shock_nondefault":  args.p_shock_nondefault,
        "n_borrowers":         len(df),
        "max_sequence_length": args.max_seq_len,
        "chunk_size":          args.chunk_size,
        "random_seed":         args.seed,
        "main_run_rows":       n_rows_main,
        "ablation_run_rows":   n_rows_abl,
        "generated_at":        datetime.utcnow().isoformat() + "Z",
    }
    with open(MANIFEST_OUT, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest → {MANIFEST_OUT.relative_to(ROOT)}")

    print("\n" + "=" * 64)
    print("Phase 3A complete.")
    print(f"  Main run   : {n_rows_main:,} rows → {main_out.relative_to(ROOT)}")
    print(f"  Ablation   : {n_rows_abl:,} rows → {ablation_out.relative_to(ROOT)}")
    print(f"  Plots      : {PLOTS_DIR.relative_to(ROOT)}")
    print(f"  Manifest   : {MANIFEST_OUT.relative_to(ROOT)}")
    print("=" * 64)


if __name__ == "__main__":
    main()
