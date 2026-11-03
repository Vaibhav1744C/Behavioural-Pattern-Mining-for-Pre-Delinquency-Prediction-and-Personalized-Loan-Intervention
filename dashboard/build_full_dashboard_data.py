"""Inference-only builder for full-run dashboard fixtures."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import DATA_ROOT

OUT = ROOT / "dashboard" / "data_full"
PROCESSED = DATA_ROOT / "processed"
SYNTHETIC = DATA_ROOT / "synthetic"
REPORTS = ROOT / "reports" / "full_run"
CONTEXT_COLS = ["salary_stability_idx", "salary_delay_trend", "savings_slope", "savings_volatility", "cashflow_compression", "emi_stress_count", "seq_len"]
BASE_COLS = ["grade", "dti", "annual_inc", "loan_amnt", "term", "int_rate"]
TIER_NAMES = ["Very Low", "Low", "Medium", "High", "Very High"]
CENTERS = np.array([0., .25, .5, .75, 1.])


def required_inputs() -> list[Path]:
    return [
        PROCESSED / "lending_club_clean.parquet",
        PROCESSED / "sequence_context_features.parquet",
        SYNTHETIC / "behavioural_sequences.parquet",
        PROCESSED / "lstm_sequences_dense.npy",
        PROCESSED / "lstm_loan_id_index.parquet",
        PROCESSED / "train_loan_ids.parquet",
        ROOT / "src/models/xgb_baseline_pipeline.pkl",
        ROOT / "src/models/xgb_behavioural_pipeline.pkl",
        ROOT / "src/models/lstm_best.pt",
        ROOT / "reports/full_metrics_comparison.json",
    ]


def _tier(pct: float) -> str:
    return TIER_NAMES[int(np.argmax(np.maximum(0., 1. - np.abs(CENTERS - pct) / .25)))]


def build() -> None:
    missing = [p for p in required_inputs() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Full profile prerequisites missing:\n  " + "\n  ".join(map(str, missing)))
    outputs = [OUT / n for n in ("test_predictions.parquet", "sample_borrowers.parquet", "sample_sequences.parquet", "sensitivity.json", "reference.json", "model_metrics.json")]
    if any(p.exists() for p in outputs):
        raise FileExistsError("Refusing to overwrite full dashboard artifacts: " + ", ".join(str(p) for p in outputs if p.exists()))
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "src/models"))
    sys.path.insert(0, str(ROOT / "src/explainability"))
    from baseline_xgboost import preprocess_features as preprocess_static
    from xgboost_behavioural import preprocess_features as preprocess_behavioural

    clean = pd.read_parquet(PROCESSED / "lending_club_clean.parquet")
    if "loan_id" not in clean.columns:
        clean = clean.reset_index().rename(columns={"index": "loan_id"})
    clean["loan_id"] = clean.loan_id.astype("int64")
    _, test = train_test_split(clean, test_size=.30, random_state=42, stratify=clean.early_default)
    test_ids = test.loan_id.astype("int64").tolist()
    if len(test_ids) != 563_771:
        raise ValueError(f"Expected 563,771 test IDs, got {len(test_ids):,}")
    y = test.early_default.astype("int8").to_numpy()

    context = pd.read_parquet(PROCESSED / "sequence_context_features.parquet")
    if "loan_id" not in context.columns:
        context = context.reset_index().rename(columns={"index": "loan_id"})
    context.loan_id = context.loan_id.astype("int64")
    context = context.set_index("loan_id").loc[test_ids].reset_index()
    merged = test.merge(context[["loan_id"] + CONTEXT_COLS], on="loan_id", validate="one_to_one")
    Xs = preprocess_static(test.drop(columns=["loan_id", "early_default"]))
    Xb = preprocess_behavioural(merged.drop(columns=["loan_id", "early_default"]))
    with (ROOT / "src/models/xgb_baseline_pipeline.pkl").open("rb") as f: static_pipe = pickle.load(f)
    with (ROOT / "src/models/xgb_behavioural_pipeline.pkl").open("rb") as f: behav_pipe = pickle.load(f)
    p_static = static_pipe.predict_proba(Xs)[:, 1]
    p_behav = behav_pipe.predict_proba(Xb)[:, 1]

    # Existing checkpoint inference only. load_all_data preserves the trained model's
    # original global context normalization and ID alignment; no fitting occurs.
    import torch
    from torch.utils.data import DataLoader
    from lstm_model import LSTMPredictor, LoanSequenceDataset
    seq = np.load(PROCESSED / "lstm_sequences_dense.npy", mmap_mode="r")
    index_df = pd.read_parquet(PROCESSED / "lstm_loan_id_index.parquet")
    all_ids = index_df.loan_id.astype("int64").tolist()
    row_by_id = dict(zip(index_df.loan_id.astype("int64"), index_df.row_index.astype("int64")))
    ctx_df = pd.read_parquet(PROCESSED / "sequence_context_features.parquet")
    if "loan_id" not in ctx_df.columns:
        ctx_df = ctx_df.reset_index().rename(columns={"index": "loan_id"})
    ctx_df.loan_id = ctx_df.loan_id.astype("int64")
    ctx_features = ctx_df[ ["loan_id"] + [c for c in CONTEXT_COLS if c != "seq_len"] ].copy()
    for col in CONTEXT_COLS[:-1]:
        ctx_features[col] = pd.to_numeric(ctx_features[col], errors="coerce").fillna(0.0)
        std = ctx_features[col].std()
        ctx_features[col] = ((ctx_features[col] - ctx_features[col].mean()) / (std if std else 1.0)).fillna(0.0)
    ctx_by_id = ctx_features.set_index("loan_id")
    ctx = ctx_by_id.loc[all_ids, CONTEXT_COLS[:-1]].to_numpy(dtype=np.float32)
    labels_by_id = clean.set_index("loan_id").early_default.astype("int8")
    labels = labels_by_id.loc[all_ids].to_numpy(dtype=np.int8)
    lengths = pd.to_numeric(ctx_df.set_index("loan_id").seq_len, errors="coerce").fillna(24).clip(lower=1).astype("int16")
    lens = lengths.loc[all_ids].to_numpy(dtype=np.int16)
    loan_ids = np.asarray(all_ids, dtype=np.int64)
    train_ids = set(pd.read_parquet(PROCESSED / "train_loan_ids.parquet").loan_id.astype("int64"))
    test_positions = np.asarray([i for i, lid in enumerate(all_ids) if lid not in train_ids], dtype=np.int64)
    full_test = LoanSequenceDataset(seq[[row_by_id[int(loan_ids[i])] for i in test_positions]], ctx[test_positions], labels[test_positions], lens[test_positions], loan_ids[test_positions])
    loader = DataLoader(full_test, batch_size=2048, shuffle=False, num_workers=0)
    model = LSTMPredictor().cpu()
    model.load_state_dict(torch.load(ROOT / "src/models/lstm_best.pt", map_location="cpu", weights_only=True))
    model.eval()
    p_lstm_parts = []
    with torch.no_grad():
        for sq, cx, _, ln, _ in loader:
            logits, _ = model(sq, cx, ln)
            p_lstm_parts.extend(torch.sigmoid(logits).numpy().tolist())
    id_to_lstm = dict(zip(loan_ids[test_positions].astype("int64").tolist(), p_lstm_parts))
    if set(test_ids) - set(id_to_lstm):
        raise ValueError("LSTM index does not cover all reproduced test IDs")
    p_lstm = np.array([id_to_lstm[lid] for lid in test_ids], dtype=np.float64)
    prediction = pd.DataFrame({"loan_id": test_ids, "y_true": y, "p_static": p_static, "p_behav": p_behav, "p_lstm": p_lstm})
    prediction["behavioural_percentile"] = prediction.p_behav.rank(method="average", pct=True)
    prediction["risk_tier"] = prediction.behavioural_percentile.map(_tier)
    prediction.to_parquet(OUT / "test_predictions.parquet", index=False)

    labels_df = prediction[["loan_id", "y_true", "risk_tier", "behavioural_percentile"]]
    sample = pd.concat([labels_df.loc[labels_df.y_true == 1].sample(100, random_state=42), labels_df.loc[labels_df.y_true == 0].sample(100, random_state=42)], ignore_index=True)
    sample_ids = sample.loan_id.astype("int64").tolist()
    sample_positions = [test_ids.index(lid) for lid in sample_ids]
    shap = __import__("shap")
    from shap_explainer import shap_compatible_booster
    prep = behav_pipe.named_steps["prep"]
    transformed = prep.transform(Xb.iloc[sample_positions])
    explainer = shap.TreeExplainer(shap_compatible_booster(behav_pipe.named_steps["clf"]))
    values = np.asarray(explainer.shap_values(transformed))
    if values.ndim == 3: values = values[:, :, -1]
    names = list(prep.get_feature_names_out())
    detail = test.loc[test.loan_id.isin(sample_ids), ["loan_id"] + BASE_COLS].set_index("loan_id").loc[sample_ids].reset_index().merge(context[["loan_id"] + CONTEXT_COLS], on="loan_id", validate="one_to_one").merge(sample[["loan_id", "risk_tier", "behavioural_percentile"]], on="loan_id", validate="one_to_one")
    positions = {lid: i for i, lid in enumerate(sample_ids)}
    raw, display, attention = [], [], []
    def is_date(name: str) -> bool:
        n = name.casefold()
        return any(t in n for t in ("issue_d", "earliest_cr_line", "months_since_", "mths_since_", "mo_sin_", "credit_history_age", "cr_line_age", "loan_vintage", "last_pymnt_d", "last_credit_pull_d"))
    # Re-run only the selected 200 examples to save their actual attention vectors.
    pick = np.array([all_ids.index(lid) for lid in sample_ids], dtype=np.int64)
    sample_ds = LoanSequenceDataset(seq[[row_by_id[int(lid)] for lid in sample_ids]], ctx[pick], labels[pick], lens[pick], loan_ids[pick])
    attn_loader = DataLoader(sample_ds, batch_size=200, shuffle=False, num_workers=0)
    with torch.no_grad():
        for sq, cx, _, ln, _ in attn_loader:
            _, att = model(sq, cx, ln)
            attention = att.numpy().tolist()
    for lid in detail.loan_id.astype("int64"):
        ranked = sorted(zip(names, values[positions[int(lid)]].tolist()), key=lambda z: abs(z[1]), reverse=True)
        raw.append([{"feature": n, "shap_value": float(v)} for n, v in ranked[:5]])
        display.append([{"feature": n, "shap_value": float(v)} for n, v in (r for r in ranked if not is_date(r[0]))][:5])
    detail["shap_top5_raw"] = raw
    detail["display_drivers"] = display
    detail["y_true"] = detail.loan_id.map(dict(zip(test_ids, y))).astype("int8")
    detail["p_static"] = detail.loan_id.map(dict(zip(test_ids, p_static)))
    detail["p_behav"] = detail.loan_id.map(dict(zip(test_ids, p_behav)))
    detail["p_lstm"] = detail.loan_id.map(id_to_lstm)
    detail["attention_weights"] = attention
    detail.to_parquet(OUT / "sample_borrowers.parquet", index=False)

    ds_seq = ds.dataset(SYNTHETIC / "behavioural_sequences.parquet", format="parquet")
    batches = ds_seq.scanner(columns=["loan_id", "month", "salary_delay_days", "account_balance", "savings_balance", "discretionary_spend", "emi_status"], filter=ds.field("loan_id").isin(sample_ids), batch_size=65536).to_batches()
    table = __import__("pyarrow").Table.from_batches(batches)
    sample_seq = table.to_pandas().sort_values(["loan_id", "month"])
    if sample_seq.loan_id.nunique() != 200 or sample_seq.groupby("loan_id").size().ne(24).any():
        raise ValueError("Filtered full-run sequence sample must contain 24 months for each of 200 borrowers")
    pq.write_table(table, OUT / "sample_sequences.parquet")

    def metrics(labels, pred): return {"auc_roc": float(roc_auc_score(labels, pred)), "pr_auc": float(average_precision_score(labels, pred))}
    results = {"scope": "Full dataset, 563,771 test", "static": metrics(y, p_static), "behavioural_main": metrics(y, p_behav), "lstm": metrics(y, p_lstm)}
    (OUT / "model_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    metrics_path = ROOT / "reports/full_metrics_comparison.json"
    rows = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else []
    def roc(prefix): return next((float(r["roc_auc"]) for r in rows if r["label"].casefold().startswith(prefix)), None)
    ref_source = REPORTS / "reference.json"
    refs = json.loads(ref_source.read_text(encoding="utf-8")) if ref_source.is_file() else {"scope": "Full dataset, 563,771 test", "static_roc_auc": roc("static xgboost"), "behavioural_roc_auc": roc("behav xgboost"), "lstm_roc_auc": roc("lstm")}
    (OUT / "reference.json").write_text(json.dumps(refs, indent=2), encoding="utf-8")
    sensitivity = next((p for p in (REPORTS / "sensitivity.json", REPORTS / "sensitivity_sweep.json") if p.is_file()), None)
    sens = json.loads(sensitivity.read_text(encoding="utf-8")) if sensitivity else {"rows": [], "seed_note": "No full-run sensitivity analysis was present."}
    if "rows" not in sens: sens = {"rows": sens if isinstance(sens, list) else [], "seed_note": "Full-run sensitivity source."}
    (OUT / "sensitivity.json").write_text(json.dumps({"scope": "Full dataset, 563,771 test", **sens}, indent=2), encoding="utf-8")
    print(f"Test rows: {len(prediction):,}; static ROC-AUC {results['static']['auc_roc']:.4f}; behavioural ROC-AUC {results['behavioural_main']['auc_roc']:.4f}; LSTM ROC-AUC {results['lstm']['auc_roc']:.4f}")


if __name__ == "__main__":
    build()
