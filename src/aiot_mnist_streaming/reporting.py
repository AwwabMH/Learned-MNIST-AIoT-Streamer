from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from .scheduling import policy_library


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def draw_box(ax, x, y, w, h, text, fc="#eef5ff", ec="#4c78a8", fontsize=10):
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02", linewidth=1.4, facecolor=fc, edgecolor=ec)
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def save_architecture_diagram(fig_dir: Path) -> None:
    _ensure_dir(fig_dir)
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    main_boxes = [
        (0.33, 0.88, 0.25, 0.07, "MNIST Sensor Stream"),
        (0.33, 0.77, 0.25, 0.07, "Arrival Process\n(periodic / poisson / bursty)"),
        (0.33, 0.65, 0.25, 0.08, "Bounded Queue + Admission Control\n(stale-drop / replace / drop-tail)"),
        (0.33, 0.50, 0.25, 0.10, "Scheduler\nFIFO / EDF / SRPT / Adaptive / Cascade"),
        (0.33, 0.34, 0.25, 0.10, "Execution\nEdge cache + cloud offloading"),
        (0.33, 0.20, 0.25, 0.07, "Decision Output"),
    ]
    for box in main_boxes:
        draw_box(ax, *box)

    left_boxes = [
        (0.05, 0.76, 0.20, 0.10, "Model Bank\nfull / tiny / pruned / quantized"),
        (0.05, 0.58, 0.20, 0.12, "Dynamic Model Loading\ncache hits / misses / evictions"),
        (0.05, 0.38, 0.20, 0.12, "Confidence Cascade\npreview -> escalate if uncertain"),
    ]
    for box in left_boxes:
        draw_box(ax, *box, fc="#f4fbf4", ec="#59a14f")

    right_boxes = [
        (0.69, 0.74, 0.24, 0.12, "System Parameters\ndeadline, stale threshold, queue size"),
        (0.69, 0.55, 0.24, 0.12, "Network Profiles\ngood / congested"),
        (0.69, 0.33, 0.24, 0.16, "Quality Analysis\nlatency breakdown, freshness, queue,\naccuracy, timely correctness, drops"),
    ]
    for box in right_boxes:
        draw_box(ax, *box, fc="#fff6eb", ec="#f28e2b")

    arrowprops = dict(arrowstyle="->", lw=1.4, color="#444")
    for y0, y1 in [(0.88, 0.84), (0.77, 0.73), (0.65, 0.60), (0.50, 0.44), (0.34, 0.27)]:
        ax.annotate("", xy=(0.455, y1), xytext=(0.455, y0), arrowprops=arrowprops)
    ax.annotate("", xy=(0.25, 0.43), xytext=(0.33, 0.54), arrowprops=arrowprops)
    ax.annotate("", xy=(0.25, 0.64), xytext=(0.33, 0.68), arrowprops=arrowprops)
    ax.annotate("", xy=(0.25, 0.81), xytext=(0.33, 0.81), arrowprops=arrowprops)
    ax.annotate("", xy=(0.69, 0.61), xytext=(0.58, 0.55), arrowprops=arrowprops)
    ax.annotate("", xy=(0.69, 0.40), xytext=(0.58, 0.39), arrowprops=arrowprops)
    ax.annotate("", xy=(0.81, 0.55), xytext=(0.81, 0.49), arrowprops=arrowprops)

    ax.set_title("Upgraded Streaming AIoT System Architecture", fontsize=16, pad=16)
    fig.tight_layout()
    fig.savefig(fig_dir / "architecture_v2.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_training_curves(hist_full, hist_tiny, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hist_full["epoch"], hist_full["test_acc"], marker="o", label="full")
    axes[0].plot(hist_tiny["epoch"], hist_tiny["test_acc"], marker="o", label="tiny")
    axes[0].set_title("Test Accuracy During Training")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(hist_full["epoch"], hist_full["train_loss"], marker="o", label="full train")
    axes[1].plot(hist_full["epoch"], hist_full["test_loss"], marker="o", label="full test")
    axes[1].plot(hist_tiny["epoch"], hist_tiny["train_loss"], marker="s", label="tiny train")
    axes[1].plot(hist_tiny["epoch"], hist_tiny["test_loss"], marker="s", label="tiny test")
    axes[1].set_title("Training / Test Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "training_curves_v2.png", bbox_inches="tight")
    plt.close(fig)


def save_model_tradeoff_plot(model_catalog: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for _, row in model_catalog.iterrows():
        ax.scatter(row["edge_mean_ms"], row["accuracy"], s=85)
        ax.annotate(row["model"], (row["edge_mean_ms"], row["accuracy"]), xytext=(6, 5), textcoords="offset points", fontsize=8)
    ax.set_title("Accuracy vs Measured Edge Latency")
    ax.set_xlabel("Edge latency (ms)")
    ax.set_ylabel("Accuracy")
    fig.tight_layout()
    fig.savefig(fig_dir / "model_accuracy_vs_latency_v2.png", bbox_inches="tight")
    plt.close(fig)


def save_main_metrics_plots(main_agg_df: pd.DataFrame, policies: dict, policy_order: list[str], fig_dir: Path) -> None:
    for arrival_mode in sorted(main_agg_df["arrival_mode"].unique()):
        for network_profile in sorted(main_agg_df["network_profile"].unique()):
            sdf = main_agg_df[
                (main_agg_df["arrival_mode"] == arrival_mode)
                & (main_agg_df["network_profile"] == network_profile)
                & (main_agg_df["policy"].isin(policy_order))
            ].copy()
            if len(sdf) == 0:
                continue
            fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
            for policy in policy_order:
                g = sdf[sdf["policy"] == policy].sort_values("rate_sps")
                if len(g) == 0:
                    continue
                label = policies[policy]["label"]
                axes[0].plot(g["rate_sps"], g["mean_latency_ms_mean"], marker="o", label=label)
                axes[1].plot(g["rate_sps"], g["deadline_meet_ratio_mean"], marker="o", label=label)
                axes[2].plot(g["rate_sps"], g["timely_correct_ratio_mean"], marker="o", label=label)
            axes[0].set_title(f"Mean Latency | {arrival_mode} | {network_profile}")
            axes[0].set_xlabel("Arrival rate (samples/s)")
            axes[0].set_ylabel("ms")
            axes[1].set_title(f"Deadline Meet Ratio | {arrival_mode} | {network_profile}")
            axes[1].set_xlabel("Arrival rate (samples/s)")
            axes[1].set_ylabel("ratio")
            axes[1].set_ylim(0, 1.05)
            axes[2].set_title(f"Timely Correct Ratio | {arrival_mode} | {network_profile}")
            axes[2].set_xlabel("Arrival rate (samples/s)")
            axes[2].set_ylabel("ratio")
            axes[2].set_ylim(0, 1.05)
            axes[2].legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(fig_dir / f"main_metrics_{arrival_mode}_{network_profile}.png", bbox_inches="tight")
            plt.close(fig)


def save_queue_evolution_plot(main_qtraces_df: pd.DataFrame, leaderboard: pd.DataFrame, fig_dir: Path) -> None:
    overloaded_row = leaderboard.sort_values(["rate_sps", "drop_ratio_mean"], ascending=[False, False]).iloc[0]
    mask = (
        (main_qtraces_df["policy"] == overloaded_row["policy"])
        & (main_qtraces_df["arrival_mode"] == overloaded_row["arrival_mode"])
        & (main_qtraces_df["network_profile"] == overloaded_row["network_profile"])
        & (main_qtraces_df["rate_sps"] == overloaded_row["rate_sps"])
    )
    qdf = main_qtraces_df[mask].copy()
    fig, ax = plt.subplots(figsize=(10, 4))
    for seed, group in qdf.groupby("seed"):
        ax.plot(group["time_ms"], group["queue_len"], lw=1.2, alpha=0.8, label=f"seed {seed}")
    ax.set_title(f"Queue Evolution | overloaded scenario | {overloaded_row['policy_label']}")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Queue length")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "queue_evolution_overloaded_v2.png", bbox_inches="tight")
    plt.close(fig)


def save_latency_breakdown_plot(main_logs_df: pd.DataFrame, leaderboard: pd.DataFrame, fig_dir: Path) -> None:
    plot_target = leaderboard.iloc[0]
    proc = main_logs_df[
        (main_logs_df["policy"] == plot_target["policy"])
        & (main_logs_df["arrival_mode"] == plot_target["arrival_mode"])
        & (main_logs_df["network_profile"] == plot_target["network_profile"])
        & (main_logs_df["rate_sps"] == plot_target["rate_sps"])
        & (main_logs_df["processed"] == 1)
    ].copy()
    comp_break = pd.DataFrame(
        {
            "Queue wait": [proc["queue_wait_ms"].mean()],
            "Preview": [proc["preview_ms"].mean()],
            "Load": [proc["load_ms"].mean()],
            "Compute": [proc["compute_ms"].mean()],
            "Communication": [proc["comm_ms"].mean()],
        }
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = 0.0
    for column in comp_break.columns:
        value = float(comp_break[column].iloc[0])
        ax.bar([0], [value], bottom=[bottom], label=column)
        bottom += value
    ax.set_xticks([0])
    ax.set_xticklabels([f"{plot_target['policy_label']}\n{plot_target['arrival_mode']} / {plot_target['network_profile']} / {int(plot_target['rate_sps'])} sps"])
    ax.set_ylabel("Mean latency contribution (ms)")
    ax.set_title("Latency Component Breakdown")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "latency_component_breakdown_v2.png", bbox_inches="tight")
    plt.close(fig)


def save_cascade_plots(main_logs_df: pd.DataFrame, policy_label: str, fig_dir: Path, art_dir: Path) -> None:
    cascade_proc = main_logs_df[(main_logs_df["policy"] == "adaptive_cascade") & (main_logs_df["processed"] == 1)].copy()
    if len(cascade_proc) == 0:
        return
    comp = cascade_proc.groupby(["arrival_mode", "network_profile", "rate_sps", "location", "model"]).size().reset_index(name="count")
    comp.to_csv(art_dir / "cascade_dispatch_composition.csv", index=False)
    for arrival_mode in sorted(cascade_proc["arrival_mode"].unique()):
        for network_profile in sorted(cascade_proc["network_profile"].unique()):
            g = cascade_proc[(cascade_proc["arrival_mode"] == arrival_mode) & (cascade_proc["network_profile"] == network_profile)]
            if len(g) == 0:
                continue
            s = g.groupby("rate_sps")["cascade_accepted"].mean().reset_index()
            fig, ax = plt.subplots(figsize=(6.5, 4))
            ax.plot(s["rate_sps"], s["cascade_accepted"], marker="o")
            ax.set_title(f"Cascade Acceptance Ratio | {arrival_mode} | {network_profile}")
            ax.set_xlabel("Arrival rate (samples/s)")
            ax.set_ylabel("ratio")
            ax.set_ylim(0, 1.05)
            fig.tight_layout()
            fig.savefig(fig_dir / f"cascade_accept_ratio_{arrival_mode}_{network_profile}.png", bbox_inches="tight")
            plt.close(fig)

            gp = comp[(comp["arrival_mode"] == arrival_mode) & (comp["network_profile"] == network_profile)]
            piv = gp.pivot_table(index="rate_sps", columns=["location", "model"], values="count", aggfunc="sum", fill_value=0).sort_index()
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(piv.index))
            bottom = np.zeros(len(piv.index))
            for column in piv.columns:
                values = piv[column].values
                ax.bar(x, values, bottom=bottom, label=f"{column[0]} | {column[1]}")
                bottom += values
            ax.set_xticks(x)
            ax.set_xticklabels(piv.index)
            ax.set_title(f"Cascade Policy: Model and Location Composition | {arrival_mode} | {network_profile}")
            ax.set_xlabel("Arrival rate (samples/s)")
            ax.set_ylabel("Processed samples")
            ax.legend(fontsize=7, ncol=2)
            fig.tight_layout()
            fig.savefig(fig_dir / f"cascade_composition_{arrival_mode}_{network_profile}.png", bbox_inches="tight")
            plt.close(fig)


def save_ablation_plot(ablation_df: pd.DataFrame, fig_dir: Path) -> None:
    abl_mean = ablation_df.groupby(["policy", "policy_label"], as_index=False).mean(numeric_only=True)
    abl_mean = abl_mean.sort_values("timely_correct_ratio", ascending=False)
    metrics = ["timely_correct_ratio", "deadline_meet_ratio", "drop_ratio", "mean_latency_ms"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.3))
    for ax, metric in zip(axes, metrics):
        axes_val = abl_mean[["policy_label", metric]].sort_values(metric, ascending=(metric in ["drop_ratio", "mean_latency_ms"]))
        ax.barh(axes_val["policy_label"], axes_val[metric])
        ax.set_title(metric)
    fig.tight_layout()
    fig.savefig(fig_dir / "ablation_comparison_v2.png", bbox_inches="tight")
    plt.close(fig)


def save_phase_transition_figures(main_agg_df: pd.DataFrame, fig_dir: Path, art_dir: Path, policies: dict, policy_order: list[str]) -> pd.DataFrame | None:
    max_mem = float(max(main_agg_df["edge_memory_budget_mb"].dropna().unique()))
    max_bw = float(max(main_agg_df["model_load_bandwidth_mb_per_ms"].dropna().unique()))
    min_evict = float(min(main_agg_df["model_evict_penalty_ms"].dropna().unique()))
    base = main_agg_df[
        (main_agg_df["arrival_mode"] == "bursty")
        & (main_agg_df["network_profile"] == "congested")
        & (main_agg_df["edge_memory_budget_mb"] == max_mem)
        & (main_agg_df["model_load_bandwidth_mb_per_ms"] == max_bw)
        & (main_agg_df["model_evict_penalty_ms"] == min_evict)
        & (main_agg_df["policy"].isin(policy_order))
    ].copy()
    base.to_csv(art_dir / "step1_overload_story_subset.csv", index=False)
    if len(base) == 0:
        return None

    d20 = base[base["deadline_budget_ms"] == float(min(base["deadline_budget_ms"].unique()))].copy()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    plot_specs = [
        ("mean_latency_ms_mean", "Mean Latency (ms)", "Mean latency | bursty | congested | baseline cache"),
        ("deadline_meet_ratio_mean", "Ratio", "Deadline Meet Ratio | bursty | congested | baseline cache"),
        ("timely_correct_ratio_mean", "Ratio", "Timely Correct Ratio | bursty | congested | baseline cache"),
    ]
    for ax, (metric, ylabel, title) in zip(axes, plot_specs):
        for policy in policy_order:
            g = d20[d20["policy"] == policy].sort_values("rate_sps")
            if len(g) == 0:
                continue
            ax.plot(g["rate_sps"], g[metric], marker="o", label=policies[policy]["label"])
        ax.set_title(title)
        ax.set_xlabel("Arrival rate (samples/s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "step1_phase_transition_bursty_congested_deadline20.png", bbox_inches="tight")
    plt.close(fig)

    adap = base[base["policy"] == "adaptive"].copy()
    if len(adap):
        piv = adap.pivot_table(index="deadline_budget_ms", columns="rate_sps", values="deadline_meet_ratio_mean")
        fig, ax = plt.subplots(figsize=(8, 4.6))
        im = ax.imshow(piv.values, aspect="auto", cmap="viridis", vmin=np.nanmin(piv.values), vmax=np.nanmax(piv.values))
        ax.set_title("Adaptive Phase Transition Heatmap\n(deadline meet ratio)")
        ax.set_xlabel("Arrival rate (samples/s)")
        ax.set_ylabel("Deadline budget (ms)")
        ax.set_xticks(np.arange(len(piv.columns)))
        ax.set_xticklabels([int(x) for x in piv.columns])
        ax.set_yticks(np.arange(len(piv.index)))
        ax.set_yticklabels([int(x) for x in piv.index])
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                value = piv.values[i, j]
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="white" if value < np.nanmean(piv.values) else "black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(fig_dir / "step1_adaptive_rate_deadline_heatmap.png", bbox_inches="tight")
        plt.close(fig)
    return base


def save_dynamic_loading_figures(main_agg_df: pd.DataFrame, main_results_df: pd.DataFrame, fig_dir: Path, art_dir: Path, policy_order: list[str]) -> pd.DataFrame | None:
    anchor = main_agg_df[
        (main_agg_df["arrival_mode"] == "bursty")
        & (main_agg_df["network_profile"] == "congested")
        & (main_agg_df["deadline_budget_ms"] == 20.0)
        & (main_agg_df["rate_sps"].isin([100, 150, 250]))
        & (main_agg_df["policy"].isin(policy_order))
    ].copy()
    anchor.to_csv(art_dir / "step2_dynamic_loading_story_subset.csv", index=False)
    if len(anchor) == 0:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    tmp = anchor.groupby(["edge_memory_budget_mb", "policy_label"], as_index=False)["cache_hit_rate_mean"].mean()
    for policy_label, group in tmp.groupby("policy_label"):
        ax.plot(group["edge_memory_budget_mb"], group["cache_hit_rate_mean"], marker="o", label=policy_label)
    ax.set_title("Cache Hit Rate vs Memory Budget")
    ax.set_xlabel("Edge memory budget (MB)")
    ax.set_ylabel("Mean cache hit rate")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "step2_cache_hit_rate_vs_memory_budget.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    tmp = anchor.groupby(["edge_memory_budget_mb", "policy_label"], as_index=False)["mean_latency_ms_mean"].mean()
    for policy_label, group in tmp.groupby("policy_label"):
        ax.plot(group["edge_memory_budget_mb"], group["mean_latency_ms_mean"], marker="o", label=policy_label)
    ax.set_title("Mean Latency vs Memory Budget")
    ax.set_xlabel("Edge memory budget (MB)")
    ax.set_ylabel("Mean latency (ms)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "step2_latency_vs_memory_budget.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    tmp = anchor.groupby(["edge_memory_budget_mb", "policy_label"], as_index=False)["timely_correct_ratio_mean"].mean()
    for policy_label, group in tmp.groupby("policy_label"):
        ax.plot(group["edge_memory_budget_mb"], group["timely_correct_ratio_mean"], marker="o", label=policy_label)
    ax.set_title("Timely-Correct Ratio vs Memory Budget")
    ax.set_xlabel("Edge memory budget (MB)")
    ax.set_ylabel("Timely-correct ratio")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "step2_timely_correct_vs_memory_budget.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    tmp = anchor.groupby(["model_load_bandwidth_mb_per_ms", "policy_label"], as_index=False)[["mean_latency_ms_mean", "load_mean_ms_mean"]].mean()
    for policy_label, group in tmp.groupby("policy_label"):
        ax.plot(group["model_load_bandwidth_mb_per_ms"], group["mean_latency_ms_mean"], marker="o", label=f"{policy_label} latency")
        ax.plot(group["model_load_bandwidth_mb_per_ms"], group["load_mean_ms_mean"], marker="s", linestyle="--", label=f"{policy_label} load")
    ax.set_title("Load-bandwidth sensitivity")
    ax.set_xlabel("Model load bandwidth (MB/ms)")
    ax.set_ylabel("Milliseconds")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "step2_loading_breakdown_vs_memory_budget.png", bbox_inches="tight")
    plt.close(fig)

    if "cache_eviction_count" in main_results_df.columns:
        anchor_ps = main_results_df[
            (main_results_df["arrival_mode"] == "bursty")
            & (main_results_df["network_profile"] == "congested")
            & (main_results_df["deadline_budget_ms"] == 20.0)
            & (main_results_df["rate_sps"].isin([100, 150, 250]))
            & (main_results_df["policy"].isin(policy_order))
        ].copy()
        fig, ax = plt.subplots(figsize=(8, 5))
        tmp = anchor_ps.groupby(["edge_memory_budget_mb", "policy_label"], as_index=False)["cache_eviction_count"].mean()
        for policy_label, group in tmp.groupby("policy_label"):
            ax.plot(group["edge_memory_budget_mb"], group["cache_eviction_count"], marker="o", label=policy_label)
        ax.set_title("Mean Cache Evictions vs Memory Budget")
        ax.set_xlabel("Edge memory budget (MB)")
        ax.set_ylabel("Mean eviction count")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "step2_cache_evictions_vs_memory_budget.png", bbox_inches="tight")
        plt.close(fig)
    return anchor


def write_report_notes(leaderboard: pd.DataFrame, output_path: Path) -> None:
    best = leaderboard.iloc[0]
    report = f"""
# Report Notes (Generated from Notebook Outputs)

## Best overall scenario from main leaderboard
- Policy: {best['policy_label']}
- Arrival mode: {best['arrival_mode']}
- Network profile: {best['network_profile']}
- Arrival rate: {int(best['rate_sps'])} samples/s
- Timely correct ratio: {best['timely_correct_ratio_mean']:.3f}
- Deadline meet ratio: {best['deadline_meet_ratio_mean']:.3f}
- Effective accuracy: {best['effective_accuracy_mean']:.3f}
- Mean latency: {best['mean_latency_ms_mean']:.2f} ms
- P95 latency: {best['p95_latency_ms_mean']:.2f} ms
- Drop ratio: {best['drop_ratio_mean']:.3f}

## Recommended figures for the final report
1. architecture_v2.png
2. model_accuracy_vs_latency_v2.png
3. one main_metrics_* figure for your chosen workload
4. queue_evolution_overloaded_v2.png
5. latency_component_breakdown_v2.png
6. ablation_comparison_v2.png

## What to emphasize in the write-up
- Cascaded inference improves timeliness by solving easy samples cheaply.
- Dynamic model loading is not free; cache misses and memory limits create additional latency.
- FIFO / EDF / SRPT give useful baselines, but the adaptive policy can better balance deadline slack, expected accuracy, and offloading cost.
- Under bursty traffic and congested networks, queueing delay and communication delay become dominant contributors.
- Ablation results demonstrate which system features matter most.
"""
    output_path.write_text(report.strip() + "\n", encoding="utf-8")
