from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass(slots=True)
class SelectorArtifacts:
    model: Pipeline
    report: pd.DataFrame
    feature_importance: pd.DataFrame
    comparison: pd.DataFrame
    test_seeds: list[int]


def action_name_from_log_row(row):
    if int(row.get("dropped", 0)) == 1 or pd.isna(row.get("model")) or pd.isna(row.get("location")):
        return "drop"
    return f"{row['location']}_{row['model']}"


def oracle_utility_from_log_row(row):
    timely = float(bool(row.get("timely_correct", False)))
    latency = float(row.get("freshness_ms", np.nan)) if pd.notna(row.get("freshness_ms", np.nan)) else 250.0
    load_ms = float(row.get("load_ms", 0.0)) if pd.notna(row.get("load_ms", np.nan)) else 0.0
    comm_ms = float(row.get("comm_ms", 0.0)) if pd.notna(row.get("comm_ms", np.nan)) else 0.0
    dropped = int(row.get("dropped", 0))
    return 6.0 * timely - 0.035 * latency - 0.050 * load_ms - 0.020 * comm_ms - 6.0 * dropped


def build_selector_features(logs_df, preview_conf_lookup):
    out = logs_df.copy()
    if "preview_conf" not in out.columns:
        out["preview_conf"] = np.nan
    missing_mask = out["preview_conf"].isna()
    if missing_mask.any():
        out.loc[missing_mask, "preview_conf"] = out.loc[missing_mask, "sid"].map(preview_conf_lookup)
    out["difficulty_proxy"] = 1.0 - out["preview_conf"].fillna(out["preview_conf"].median())
    group_cols = [
        "policy",
        "seed",
        "arrival_mode",
        "network_profile",
        "rate_sps",
        "deadline_budget_ms",
        "edge_memory_budget_mb",
        "model_load_bandwidth_mb_per_ms",
        "model_evict_penalty_ms",
    ]
    out["arrival_gap_ms"] = out.groupby(group_cols)["arrival_ms"].diff().fillna(0.0)
    sample_group_cols = [
        "seed",
        "arrival_mode",
        "network_profile",
        "rate_sps",
        "deadline_budget_ms",
        "edge_memory_budget_mb",
        "model_load_bandwidth_mb_per_ms",
        "model_evict_penalty_ms",
    ]
    max_sid = out.groupby(sample_group_cols)["sid"].transform("max").replace(0, 1)
    out["sample_progress"] = out["sid"] / max_sid
    out["action_name"] = out.apply(action_name_from_log_row, axis=1)
    out["oracle_utility"] = out.apply(oracle_utility_from_log_row, axis=1)
    return out


def build_selector_dataset(main_logs_df, *, preview_conf_lookup, art_dir: Path):
    use_logs = main_logs_df[
        (main_logs_df["arrival_mode"] == "bursty")
        & (main_logs_df["network_profile"] == "congested")
        & (main_logs_df["deadline_budget_ms"] == 20.0)
        & (main_logs_df["rate_sps"].isin([100, 150, 250]))
    ].copy()
    if len(use_logs) == 0:
        return None, None, None
    use_logs = build_selector_features(use_logs, preview_conf_lookup)

    group_cols = [
        "seed",
        "arrival_mode",
        "network_profile",
        "rate_sps",
        "deadline_budget_ms",
        "edge_memory_budget_mb",
        "model_load_bandwidth_mb_per_ms",
        "model_evict_penalty_ms",
        "sid",
    ]
    oracle_idx = use_logs.groupby(group_cols)["oracle_utility"].idxmax()
    oracle_df = use_logs.loc[oracle_idx].copy().reset_index(drop=True)
    oracle_df = oracle_df.rename(columns={"action_name": "oracle_action"})

    feat_df = use_logs.groupby(group_cols, as_index=False).agg(
        {
            "arrival_ms": "first",
            "arrival_gap_ms": "first",
            "sample_progress": "first",
            "preview_conf": "first",
            "difficulty_proxy": "first",
        }
    )
    selector_df = feat_df.merge(
        oracle_df[group_cols + ["oracle_action", "oracle_utility", "freshness_ms", "load_ms", "comm_ms", "timely_correct"]],
        on=group_cols,
        how="left",
    )
    selector_df = selector_df.rename(
        columns={
            "freshness_ms": "oracle_latency_ms",
            "load_ms": "oracle_load_ms",
            "comm_ms": "oracle_comm_ms",
            "timely_correct": "oracle_timely_correct",
        }
    )
    action_counts = selector_df["oracle_action"].value_counts()
    keep_actions = action_counts[action_counts >= 20].index.tolist()
    selector_df["oracle_action"] = selector_df["oracle_action"].where(selector_df["oracle_action"].isin(keep_actions), "other")
    selector_df.to_csv(art_dir / "selector_training_data.csv", index=False)
    oracle_df.to_csv(art_dir / "selector_oracle_rows.csv", index=False)
    action_counts.to_csv(art_dir / "selector_action_counts.csv", header=["count"])
    return selector_df, oracle_df, use_logs


def train_and_evaluate_selector(
    selector_df,
    candidate_logs_df,
    *,
    art_dir: Path,
    fig_dir: Path,
    max_train_rows=50000,
    max_candidate_rows=250000,
    n_estimators=120,
    make_plots=False,
):
    if selector_df is None or len(selector_df) == 0:
        return None

    feature_cols = [
        "arrival_ms",
        "arrival_gap_ms",
        "sample_progress",
        "preview_conf",
        "difficulty_proxy",
        "rate_sps",
        "deadline_budget_ms",
        "edge_memory_budget_mb",
        "model_load_bandwidth_mb_per_ms",
        "model_evict_penalty_ms",
        "network_profile",
    ]
    target_col = "oracle_action"
    work = selector_df.dropna(subset=[target_col]).copy()

    if max_train_rows is not None and len(work) > int(max_train_rows):
        per_class = max(1, int(max_train_rows) // max(1, work[target_col].nunique()))
        parts = []
        for _, group in work.groupby(target_col, observed=True):
            parts.append(group.sample(n=min(len(group), per_class), random_state=42))
        work = pd.concat(parts, ignore_index=True)

    unique_seeds = sorted(work["seed"].unique())
    n_test = max(1, len(unique_seeds) // 3)
    test_seeds = unique_seeds[-n_test:]
    train_df = work[~work["seed"].isin(test_seeds)].copy()
    test_df = work[work["seed"].isin(test_seeds)].copy()
    if len(train_df) == 0 or len(test_df) == 0:
        train_df, test_df = train_test_split(work, test_size=0.25, random_state=42, stratify=work[target_col])

    numeric_cols = [c for c in feature_cols if c != "network_profile"]
    categorical_cols = ["network_profile"]
    preprocessor = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=8,
        min_samples_leaf=6,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    pipe = Pipeline([("pre", preprocessor), ("clf", classifier)])
    pipe.fit(train_df[feature_cols], train_df[target_col])
    pred = pipe.predict(test_df[feature_cols])

    report = pd.DataFrame(classification_report(test_df[target_col], pred, output_dict=True)).T
    report.to_csv(art_dir / "selector_classification_report.csv", index=True)
    labels_sorted = sorted(pd.Index(work[target_col].unique()).astype(str).tolist())
    cm = confusion_matrix(test_df[target_col], pred, labels=labels_sorted)
    cm_df = pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)
    cm_df.to_csv(art_dir / "selector_confusion_matrix.csv", index=True)
    joblib.dump(pipe, art_dir / "learned_selector.pkl")

    rf = pipe.named_steps["clf"]
    ohe = pipe.named_steps["pre"].named_transformers_["cat"].named_steps["onehot"]
    feature_names = numeric_cols + list(ohe.get_feature_names_out(categorical_cols))
    fi = pd.DataFrame({"feature": feature_names, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
    fi.to_csv(art_dir / "selector_feature_importance.csv", index=False)

    test_scored = test_df.copy()
    test_scored["pred_action"] = pred
    key_cols = [
        "seed",
        "arrival_mode",
        "network_profile",
        "rate_sps",
        "deadline_budget_ms",
        "edge_memory_budget_mb",
        "model_load_bandwidth_mb_per_ms",
        "model_evict_penalty_ms",
        "sid",
    ]
    cands = candidate_logs_df.copy()
    if max_candidate_rows is not None and len(cands) > int(max_candidate_rows):
        keep_idx = test_scored[key_cols].drop_duplicates()
        cands = cands.merge(keep_idx, on=key_cols, how="inner")
        if len(cands) > int(max_candidate_rows):
            cands = cands.sample(n=int(max_candidate_rows), random_state=42)

    dropped = cands["dropped"].fillna(0).astype(np.int8)
    invalid_action = (dropped == 1) | cands["model"].isna() | cands["location"].isna()
    cands["action_name"] = "drop"
    valid_mask = ~invalid_action
    cands.loc[valid_mask, "action_name"] = cands.loc[valid_mask, "location"].astype(str) + "_" + cands.loc[valid_mask, "model"].astype(str)
    timely = cands["timely_correct"].fillna(False).astype(bool).astype(float)
    latency = pd.to_numeric(cands["freshness_ms"], errors="coerce").fillna(250.0)
    load_ms = pd.to_numeric(cands["load_ms"], errors="coerce").fillna(0.0)
    comm_ms = pd.to_numeric(cands["comm_ms"], errors="coerce").fillna(0.0)
    cands["oracle_utility"] = 6.0 * timely - 0.035 * latency - 0.050 * load_ms - 0.020 * comm_ms - 6.0 * dropped

    baseline_rows = []
    for policy in ["fifo", "edf", "srpt", "adaptive", "adaptive_cascade"]:
        group = cands[cands["policy"] == policy].copy()
        if len(group) == 0:
            continue
        idx = group.groupby(key_cols, observed=True)["oracle_utility"].idxmax()
        h = group.loc[idx, key_cols + ["action_name", "oracle_utility", "freshness_ms", "timely_correct", "deadline_met"]].copy()
        h = h.rename(
            columns={
                "oracle_utility": f"{policy}_utility",
                "freshness_ms": f"{policy}_latency_ms",
                "timely_correct": f"{policy}_timely_correct",
                "deadline_met": f"{policy}_deadline_met",
                "action_name": f"{policy}_action",
            }
        )
        baseline_rows.append(h)

    compare_df = test_scored[key_cols + ["pred_action", "oracle_action", "oracle_utility", "freshness_ms", "oracle_timely_correct"]].copy()
    compare_df = compare_df.rename(columns={"freshness_ms": "oracle_latency_ms"})
    for h in baseline_rows:
        compare_df = compare_df.merge(h, on=key_cols, how="left")

    pred_key = test_scored[key_cols + ["pred_action"]].drop_duplicates()
    cands_pred = cands.merge(pred_key, on=key_cols, how="inner")
    cands_pred = cands_pred[cands_pred["action_name"] == cands_pred["pred_action"]]
    pred_best = cands_pred.sort_values("oracle_utility", ascending=False).groupby(key_cols, as_index=False, observed=True).first()
    worst_fallback = cands.sort_values("oracle_utility", ascending=True).groupby(key_cols, as_index=False, observed=True).first()
    pred_choice = worst_fallback.merge(
        pred_best[key_cols + ["oracle_utility", "freshness_ms", "timely_correct", "deadline_met"]],
        on=key_cols,
        how="left",
        suffixes=("_fallback", "_pred"),
    )
    has_pred = pred_choice["oracle_utility_pred"].notna()
    pred_choice["learned_selector_action_missing"] = (~has_pred).astype(int)
    pred_choice["learned_selector_utility"] = np.where(has_pred, pred_choice["oracle_utility_pred"], pred_choice["oracle_utility_fallback"])
    pred_choice["learned_selector_latency_ms"] = np.where(has_pred, pred_choice["freshness_ms_pred"], pred_choice["freshness_ms_fallback"])
    pred_choice["learned_selector_timely_correct"] = np.where(has_pred, pred_choice["timely_correct_pred"], pred_choice["timely_correct_fallback"]).astype(float)
    pred_choice["learned_selector_deadline_met"] = np.where(has_pred, pred_choice["deadline_met_pred"], pred_choice["deadline_met_fallback"]).astype(float)

    pred_df = pred_choice[key_cols + ["learned_selector_action_missing", "learned_selector_utility", "learned_selector_latency_ms", "learned_selector_timely_correct", "learned_selector_deadline_met"]]
    compare_df = compare_df.merge(pred_df, on=key_cols, how="left")
    compare_df.to_csv(art_dir / "selector_vs_baselines_offline.csv", index=False)

    for policy in ["fifo", "edf", "srpt", "adaptive", "adaptive_cascade"]:
        for suffix in ["timely_correct", "deadline_met", "latency_ms"]:
            column = f"{policy}_{suffix}"
            if column not in compare_df.columns:
                compare_df[column] = np.nan

    plot_compare = compare_df.groupby("rate_sps", as_index=False).agg(
        {
            "learned_selector_timely_correct": "mean",
            "learned_selector_deadline_met": "mean",
            "learned_selector_latency_ms": "mean",
            "fifo_timely_correct": "mean",
            "edf_timely_correct": "mean",
            "srpt_timely_correct": "mean",
            "adaptive_timely_correct": "mean",
            "adaptive_cascade_timely_correct": "mean",
            "fifo_deadline_met": "mean",
            "edf_deadline_met": "mean",
            "srpt_deadline_met": "mean",
            "adaptive_deadline_met": "mean",
            "adaptive_cascade_deadline_met": "mean",
            "fifo_latency_ms": "mean",
            "edf_latency_ms": "mean",
            "srpt_latency_ms": "mean",
            "adaptive_latency_ms": "mean",
            "adaptive_cascade_latency_ms": "mean",
        }
    )
    plot_compare.to_csv(art_dir / "selector_story_focus_subset.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(cm_df.values, cmap="Blues", aspect="auto")
    ax.set_title("Learned Selector Confusion Matrix")
    ax.set_xlabel("Predicted action")
    ax.set_ylabel("True oracle action")
    ax.set_xticks(np.arange(len(cm_df.columns)))
    ax.set_xticklabels(cm_df.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(cm_df.index)))
    ax.set_yticklabels(cm_df.index, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(fig_dir / "selector_confusion_matrix.png", bbox_inches="tight")
    if make_plots:
        plt.show()
    else:
        plt.close(fig)

    top_fi = fi.head(12).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_fi["feature"], top_fi["importance"])
    ax.set_title("Learned Selector Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(fig_dir / "selector_feature_importance.png", bbox_inches="tight")
    if make_plots:
        plt.show()
    else:
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(plot_compare["rate_sps"], plot_compare["learned_selector_timely_correct"], marker="o", label="Learned Selector")
    for policy in ["fifo", "edf", "srpt", "adaptive", "adaptive_cascade"]:
        ax.plot(plot_compare["rate_sps"], plot_compare[f"{policy}_timely_correct"], marker="o", label=policy)
    ax.set_title("Learned Selector vs Baselines: Timely-Correct Ratio")
    ax.set_xlabel("Arrival rate (samples/s)")
    ax.set_ylabel("Ratio")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "selector_vs_baselines_timely_correct.png", bbox_inches="tight")
    if make_plots:
        plt.show()
    else:
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(plot_compare["rate_sps"], plot_compare["learned_selector_deadline_met"], marker="o", label="Learned Selector")
    for policy in ["fifo", "edf", "srpt", "adaptive", "adaptive_cascade"]:
        ax.plot(plot_compare["rate_sps"], plot_compare[f"{policy}_deadline_met"], marker="o", label=policy)
    ax.set_title("Learned Selector vs Baselines: Deadline-Meet Ratio")
    ax.set_xlabel("Arrival rate (samples/s)")
    ax.set_ylabel("Ratio")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "selector_vs_baselines_deadline_meet.png", bbox_inches="tight")
    if make_plots:
        plt.show()
    else:
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(plot_compare["rate_sps"], plot_compare["learned_selector_latency_ms"], marker="o", label="Learned Selector")
    for policy in ["fifo", "edf", "srpt", "adaptive", "adaptive_cascade"]:
        ax.plot(plot_compare["rate_sps"], plot_compare[f"{policy}_latency_ms"], marker="o", label=policy)
    ax.set_title("Learned Selector vs Baselines: Mean Latency")
    ax.set_xlabel("Arrival rate (samples/s)")
    ax.set_ylabel("Mean latency (ms)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "selector_vs_baselines_mean_latency.png", bbox_inches="tight")
    if make_plots:
        plt.show()
    else:
        plt.close(fig)

    return SelectorArtifacts(model=pipe, report=report, feature_importance=fi, comparison=compare_df, test_seeds=test_seeds)
