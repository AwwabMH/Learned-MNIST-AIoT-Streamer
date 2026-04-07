from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import torch

from .benchmarking import build_model_metrics_table
from .catalog import build_catalog_bundle
from .compression import apply_dynamic_quantization, apply_global_pruning, clone_to_cpu
from .config import build_project_config, set_global_seed, write_config_snapshot
from .data import MnistDataModule
from .learning import build_selector_dataset, train_and_evaluate_selector
from .models import build_model_bank
from .paths import create_run_paths
from .reporting import (
    save_ablation_plot,
    save_architecture_diagram,
    save_cascade_plots,
    save_dynamic_loading_figures,
    save_latency_breakdown_plot,
    save_main_metrics_plots,
    save_model_tradeoff_plot,
    save_queue_evolution_plot,
    save_phase_transition_figures,
    save_training_curves,
    write_report_notes,
)
from .scheduling import policy_library
from .simulation import StreamingSimulator


def run_project(*, quick_mode: bool = True, output_root: str | Path = "outputs_aiot_project_v2", make_plots: bool = True):
    config = build_project_config(quick_mode=quick_mode)
    config.output_root = Path(output_root)
    set_global_seed(config.seed)
    run_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    paths = create_run_paths(config.output_root, run_name)
    write_config_snapshot(config, paths.run_dir / "config.json")

    data_module = MnistDataModule(
        data_dir=config.data_dir,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        train_limit=config.training.train_limit,
        test_limit=config.training.test_limit,
    )
    train_loader, test_loader = data_module.build_loaders()
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    device = torch.device(config.device)
    model_bank = build_model_bank()
    full_model = model_bank["full_fp32"].to(device)
    tiny_model = model_bank["tiny_fp32"].to(device)

    from .training import fit_model

    full_result = fit_model(full_model, train_loader, test_loader, config.training.epochs_full, "full", config.training, device)
    tiny_result = fit_model(tiny_model, train_loader, test_loader, config.training.epochs_tiny, "tiny", config.training, device)
    hist_full = full_result.history
    hist_tiny = tiny_result.history
    torch.save(full_model.state_dict(), paths.checkpoints / "full_model.pt")
    torch.save(tiny_model.state_dict(), paths.checkpoints / "tiny_model.pt")
    hist_full.to_csv(paths.artifacts / "history_full.csv", index=False)
    hist_tiny.to_csv(paths.artifacts / "history_tiny.csv", index=False)

    compressed_bank = {
        "full_fp32": clone_to_cpu(full_model),
        "tiny_fp32": clone_to_cpu(tiny_model),
    }
    compressed_bank["full_pruned40"] = apply_global_pruning(clone_to_cpu(full_model), config.compression.prune_full_amount)
    compressed_bank["tiny_pruned30"] = apply_global_pruning(clone_to_cpu(tiny_model), config.compression.prune_tiny_amount)
    compressed_bank["full_quant"] = apply_dynamic_quantization(clone_to_cpu(full_model))
    compressed_bank["tiny_quant"] = apply_dynamic_quantization(clone_to_cpu(tiny_model))

    catalog, static_df, acc_df, lat_df, variant_outputs, latency_samples = build_model_metrics_table(
        compressed_bank,
        test_loader,
        quick_mode=config.quick_mode,
        cloud_speedup=config.compression.cloud_speedup,
        load_bandwidth_mb_per_ms=config.runtime.model_load_bandwidth_mb_per_ms[0],
    )
    catalog_bundle = build_catalog_bundle(catalog, config.compression.preview_model)
    model_catalog = catalog_bundle.catalog
    model_catalog.to_csv(paths.artifacts / "model_catalog.csv", index=False)
    static_df.to_csv(paths.artifacts / "model_static_metrics.csv", index=False)
    acc_df.to_csv(paths.artifacts / "model_accuracy.csv", index=False)
    lat_df.to_csv(paths.artifacts / "model_latency_benchmarks.csv", index=False)

    preview_conf_lookup = {int(i): float(v) for i, v in enumerate(variant_outputs[catalog_bundle.preview_model]["conf"])}

    save_architecture_diagram(paths.figures)
    save_training_curves(hist_full, hist_tiny, paths.figures)
    save_model_tradeoff_plot(model_catalog, paths.figures)

    network_profiles = {
        "good": {"one_way_mean_ms": 8.0, "one_way_std_ms": 2.0},
        "congested": {"one_way_mean_ms": 25.0, "one_way_std_ms": 10.0},
    }
    simulator = StreamingSimulator(
        model_lookup=catalog_bundle.model_lookup,
        variant_outputs=variant_outputs,
        preview_model=catalog_bundle.preview_model,
        strong_models=catalog_bundle.strong_models,
        latency_samples=latency_samples,
        cloud_speedup=config.compression.cloud_speedup,
        network_profiles=network_profiles,
        runtime_config=config.runtime,
    )
    policies = policy_library(config.runtime, config.compression)

    main_results = []
    main_logs = []
    main_qtraces = []
    deadline_sweep = config.runtime.deadline_ms
    memory_budget_sweep = config.runtime.edge_memory_budget_mb
    load_bandwidth_sweep = config.runtime.model_load_bandwidth_mb_per_ms
    evict_penalty_sweep = config.runtime.model_evict_penalty_ms

    for seed in config.experiment.seeds:
        for arrival_mode in config.experiment.arrival_modes:
            for rate in config.experiment.rates_sps:
                for network_profile in config.experiment.network_profiles:
                    for deadline_ms in deadline_sweep:
                        for memory_budget_mb in memory_budget_sweep:
                            for load_bandwidth in load_bandwidth_sweep:
                                for evict_penalty_ms in evict_penalty_sweep:
                                    for policy_name in ["fifo", "edf", "srpt", "adaptive", "adaptive_cascade"]:
                                        print(
                                            f"[MAIN] seed={seed} | mode={arrival_mode} | rate={rate} | net={network_profile} | "
                                            f"deadline={deadline_ms} | mem={memory_budget_mb} | bw={load_bandwidth} | evict={evict_penalty_ms} | policy={policy_name}"
                                        )
                                        logs_df, qtrace_df, cache_stats = simulator.run_scenario(
                                            policy_name,
                                            policies[policy_name],
                                            rate,
                                            arrival_mode,
                                            network_profile,
                                            seed,
                                            config.experiment.samples_per_scenario,
                                            memory_budget_mb=memory_budget_mb,
                                            deadline_budget_ms=deadline_ms,
                                            model_load_bandwidth_mb_per_ms=load_bandwidth,
                                            model_evict_penalty_ms=evict_penalty_ms,
                                        )
                                        summary = simulator.summarize_logs(logs_df, cache_stats)
                                        summary.update(
                                            {
                                                "policy": policy_name,
                                                "policy_label": policies[policy_name]["label"],
                                                "seed": seed,
                                                "arrival_mode": arrival_mode,
                                                "rate_sps": rate,
                                                "network_profile": network_profile,
                                                "deadline_budget_ms": float(deadline_ms),
                                                "edge_memory_budget_mb": float(memory_budget_mb),
                                                "model_load_bandwidth_mb_per_ms": float(load_bandwidth),
                                                "model_evict_penalty_ms": float(evict_penalty_ms),
                                            }
                                        )
                                        main_results.append(summary)
                                        main_logs.append(logs_df)
                                        main_qtraces.append(qtrace_df)

    main_results_df = pd.DataFrame(main_results)
    main_logs_df = pd.concat(main_logs, ignore_index=True) if main_logs else pd.DataFrame()
    main_qtraces_df = pd.concat(main_qtraces, ignore_index=True) if main_qtraces else pd.DataFrame()
    main_results_df.to_csv(paths.artifacts / "main_results_per_seed.csv", index=False)
    main_logs_df.to_csv(paths.artifacts / "main_logs_all.csv", index=False)
    main_qtraces_df.to_csv(paths.artifacts / "main_queue_traces_all.csv", index=False)

    agg_keys = ["policy", "policy_label", "arrival_mode", "rate_sps", "network_profile", "deadline_budget_ms", "edge_memory_budget_mb", "model_load_bandwidth_mb_per_ms", "model_evict_penalty_ms"]
    metric_cols = [
        "drop_ratio",
        "mean_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "deadline_meet_ratio",
        "effective_accuracy",
        "timely_correct_ratio",
        "queue_wait_mean_ms",
        "preview_mean_ms",
        "load_mean_ms",
        "compute_mean_ms",
        "comm_mean_ms",
        "cloud_ratio",
        "edge_ratio",
        "cascade_accept_ratio",
        "cache_hit_rate",
    ]
    agg_rows = []
    for key, group in main_results_df.groupby(agg_keys):
        row = dict(zip(agg_keys, key))
        for metric in metric_cols:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=0)
        row["num_seeds"] = group["seed"].nunique()
        agg_rows.append(row)
    main_agg_df = pd.DataFrame(agg_rows)
    main_agg_df.to_csv(paths.artifacts / "main_results_aggregated.csv", index=False)
    leaderboard = main_agg_df.sort_values(["timely_correct_ratio_mean", "deadline_meet_ratio_mean", "effective_accuracy_mean"], ascending=[False, False, False]).reset_index(drop=True)
    leaderboard.to_csv(paths.artifacts / "leaderboard_main.csv", index=False)

    save_phase_transition_figures(main_agg_df, paths.figures, paths.artifacts, policies, ["fifo", "edf", "srpt", "adaptive", "adaptive_cascade"])

    ablation_results = []
    ablation_logs = []
    ablation_fixed = {
        "arrival_mode": "bursty" if "bursty" in config.experiment.arrival_modes else config.experiment.arrival_modes[-1],
        "rate_sps": max(config.experiment.rates_sps),
        "network_profile": "congested",
    }
    for seed in config.experiment.seeds:
        for deadline_ms in deadline_sweep:
            for memory_budget_mb in memory_budget_sweep:
                for load_bandwidth in load_bandwidth_sweep:
                    for evict_penalty_ms in evict_penalty_sweep:
                        for policy_name in ["adaptive_cascade", "ablation_no_cascade", "ablation_no_cloud", "ablation_no_cache"]:
                            logs_df, qtrace_df, cache_stats = simulator.run_scenario(
                                policy_name,
                                policies[policy_name],
                                ablation_fixed["rate_sps"],
                                ablation_fixed["arrival_mode"],
                                ablation_fixed["network_profile"],
                                seed,
                                config.experiment.samples_per_scenario,
                                memory_budget_mb=memory_budget_mb,
                                deadline_budget_ms=deadline_ms,
                                model_load_bandwidth_mb_per_ms=load_bandwidth,
                                model_evict_penalty_ms=evict_penalty_ms,
                            )
                            summary = simulator.summarize_logs(logs_df, cache_stats)
                            summary.update(
                                {
                                    "policy": policy_name,
                                    "policy_label": policies[policy_name]["label"],
                                    "seed": seed,
                                    "deadline_budget_ms": float(deadline_ms),
                                    "edge_memory_budget_mb": float(memory_budget_mb),
                                    "model_load_bandwidth_mb_per_ms": float(load_bandwidth),
                                    "model_evict_penalty_ms": float(evict_penalty_ms),
                                    **ablation_fixed,
                                }
                            )
                            ablation_results.append(summary)
                            ablation_logs.append(logs_df)
    ablation_df = pd.DataFrame(ablation_results)
    ablation_logs_df = pd.concat(ablation_logs, ignore_index=True) if ablation_logs else pd.DataFrame()
    ablation_df.to_csv(paths.artifacts / "ablation_results_per_seed.csv", index=False)
    ablation_logs_df.to_csv(paths.artifacts / "ablation_logs_all.csv", index=False)

    save_dynamic_loading_figures(main_agg_df, main_results_df, paths.figures, paths.artifacts, ["fifo", "edf", "srpt", "adaptive", "adaptive_cascade"])

    save_main_metrics_plots(main_agg_df, policies, ["fifo", "edf", "srpt", "adaptive", "adaptive_cascade"], paths.figures)
    save_queue_evolution_plot(main_qtraces_df, leaderboard, paths.figures)
    save_latency_breakdown_plot(main_logs_df, leaderboard, paths.figures)
    save_cascade_plots(main_logs_df, policies["adaptive_cascade"]["label"], paths.figures, paths.artifacts)
    save_ablation_plot(ablation_df, paths.figures)

    if len(main_logs_df):
        main_logs_df["preview_conf"] = main_logs_df["sid"].map(preview_conf_lookup)
    selector_df, oracle_df, selector_candidate_logs_df = build_selector_dataset(main_logs_df, preview_conf_lookup=preview_conf_lookup, art_dir=paths.artifacts)
    selector_artifacts = train_and_evaluate_selector(
        selector_df,
        selector_candidate_logs_df,
        art_dir=paths.artifacts,
        fig_dir=paths.figures,
        max_train_rows=50000,
        max_candidate_rows=250000,
        n_estimators=120,
        make_plots=make_plots,
    )

    write_report_notes(leaderboard, paths.run_dir / "report_notes_generated.md")
    self_checks = pd.DataFrame(
        [
            {"check": "at_least_one_overloaded_scenario", "pass": bool((main_results_df["drop_ratio"] > 0).any())},
            {"check": "adaptive_cascade_used_cloud", "pass": bool(((main_logs_df["policy"] == "adaptive_cascade") & (main_logs_df["location"] == "cloud")).any()) if len(main_logs_df) else False},
            {"check": "adaptive_cascade_switched_models", "pass": bool(main_logs_df[main_logs_df["policy"] == "adaptive_cascade"]["model"].dropna().nunique() >= 2) if len(main_logs_df) else False},
            {"check": "cache_miss_observed", "pass": bool((main_results_df["cache_miss_count"] > 0).any()) if len(main_results_df) else False},
            {"check": "step1_phase_transition_exported", "pass": (paths.figures / "step1_phase_transition_bursty_congested_deadline20.png").exists()},
            {"check": "step2_dynamic_loading_exports", "pass": (paths.figures / "step2_cache_hit_rate_vs_memory_budget.png").exists() and (paths.figures / "step2_latency_vs_memory_budget.png").exists()},
            {"check": "selector_artifacts_exported", "pass": (paths.artifacts / "selector_training_data.csv").exists() and (paths.artifacts / "selector_classification_report.csv").exists() and (paths.figures / "selector_vs_baselines_timely_correct.png").exists()},
            {"check": "all_required_figure_files_exist", "pass": all((paths.figures / name).exists() for name in ["architecture_v2.png", "training_curves_v2.png", "model_accuracy_vs_latency_v2.png", "queue_evolution_overloaded_v2.png", "latency_component_breakdown_v2.png", "ablation_comparison_v2.png"])},
        ]
    )
    self_checks.to_csv(paths.artifacts / "self_checks.csv", index=False)

    return {
        "config": config,
        "paths": paths,
        "model_catalog": model_catalog,
        "main_results": main_results_df,
        "main_aggregated": main_agg_df,
        "ablation": ablation_df,
        "leaderboard": leaderboard,
        "selector": selector_artifacts,
        "self_checks": self_checks,
    }
