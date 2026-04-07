from __future__ import annotations

import copy
from typing import Any

import numpy as np

from .config import CompressionConfig, RuntimeConfig, normalize_positive_sweep


def policy_library(runtime_cfg: RuntimeConfig, compression_cfg: CompressionConfig) -> dict[str, dict[str, Any]]:
    deadline_values = normalize_positive_sweep(runtime_cfg.deadline_ms, "deadline_ms")
    base = {
        "deadline_ms": float(deadline_values[0]),
        "stale_ms": float(runtime_cfg.stale_ms),
        "queue_capacity": int(runtime_cfg.queue_capacity),
        "scan_window": int(runtime_cfg.queue_scan_window),
        "confidence_threshold": float(compression_cfg.confidence_threshold),
        "cache_penalty": True,
        "allow_cloud": True,
        "cascade": False,
        "priority_mode": "adaptive",
        "label": "adaptive",
    }
    return {
        "fifo": {**base, "priority_mode": "fifo", "label": "FIFO"},
        "edf": {**base, "priority_mode": "edf", "label": "EDF"},
        "srpt": {**base, "priority_mode": "srpt", "label": "SRPT"},
        "adaptive": {**base, "priority_mode": "adaptive", "label": "Adaptive"},
        "adaptive_cascade": {**base, "priority_mode": "adaptive", "cascade": True, "label": "Adaptive + Cascade"},
        "ablation_no_cascade": {**base, "priority_mode": "adaptive", "cascade": False, "label": "Ablation: no cascade"},
        "ablation_no_cloud": {**base, "priority_mode": "adaptive", "cascade": True, "allow_cloud": False, "label": "Ablation: no cloud"},
        "ablation_no_cache": {**base, "priority_mode": "adaptive", "cascade": True, "cache_penalty": False, "label": "Ablation: no cache penalty"},
    }


def sample_empirical_runtime_ms(model_name, location, rng, latency_samples, cloud_speedup):
    draw = float(latency_samples[model_name][int(rng.integers(0, len(latency_samples[model_name])) )])
    return draw if location == "edge" else max(0.05, draw / float(cloud_speedup))


def sample_network_delay_ms(profile_name, rng, network_profiles):
    prof = network_profiles[profile_name]
    tx = max(0.0, rng.normal(prof["one_way_mean_ms"], prof["one_way_std_ms"]))
    rx = max(0.0, rng.normal(prof["one_way_mean_ms"], prof["one_way_std_ms"]))
    return float(tx + rx)


def estimate_plan(
    sample,
    now_ms,
    queue_len,
    policy_cfg,
    network_profile,
    rng,
    cache,
    resource_kind,
    model_load_bandwidth_mb_per_ms,
    *,
    model_lookup,
    variant_outputs,
    preview_model,
    strong_models,
    latency_samples,
    cloud_speedup,
    network_profiles,
):
    age_ms = now_ms - sample.arrival_ms
    if age_ms > policy_cfg["stale_ms"]:
        return None

    allow_cloud = bool(policy_cfg["allow_cloud"]) and resource_kind == "cloud"
    allow_edge = resource_kind == "edge"
    cascade = bool(policy_cfg["cascade"])
    model_load_bandwidth_mb_per_ms = float(model_load_bandwidth_mb_per_ms)

    def model_load_ms(model_name):
        return float(model_lookup[model_name]["memory_mb"]) / model_load_bandwidth_mb_per_ms

    def utility(model_name, pred_latency_ms, load_ms, comm_ms, preview_ms):
        acc = float(model_lookup[model_name]["accuracy"])
        slack = policy_cfg["deadline_ms"] - pred_latency_ms
        return (
            4.5 * acc
            - 0.013 * pred_latency_ms
            - 0.028 * max(0.0, -slack)
            - 0.0035 * queue_len * float(model_lookup[model_name]["edge_mean_ms"])
            - 0.010 * load_ms
            - 0.008 * comm_ms
            - 0.004 * preview_ms
        )

    def basic_actions():
        actions = []
        for model_name in model_lookup:
            if allow_edge:
                load_ms = 0.0 if model_name in cache.loaded or not policy_cfg["cache_penalty"] else model_load_ms(model_name)
                comp_ms = sample_empirical_runtime_ms(model_name, "edge", rng, latency_samples, cloud_speedup)
                total = age_ms + load_ms + comp_ms
                actions.append(
                    {
                        "final_model": model_name,
                        "location": "edge",
                        "preview_model": None,
                        "preview_ms": 0.0,
                        "load_ms": load_ms,
                        "compute_ms": comp_ms,
                        "comm_ms": 0.0,
                        "pred_latency_ms": total,
                    }
                )
            if allow_cloud:
                comp_ms = sample_empirical_runtime_ms(model_name, "cloud", rng, latency_samples, cloud_speedup)
                comm_ms = sample_network_delay_ms(network_profile, rng, network_profiles)
                total = age_ms + comp_ms + comm_ms
                actions.append(
                    {
                        "final_model": model_name,
                        "location": "cloud",
                        "preview_model": None,
                        "preview_ms": 0.0,
                        "load_ms": 0.0,
                        "compute_ms": comp_ms,
                        "comm_ms": comm_ms,
                        "pred_latency_ms": total,
                    }
                )
        return actions

    def cascaded_actions():
        actions = []
        preview_conf = float(variant_outputs[preview_model]["conf"][sample.sid])
        preview_load_ms = 0.0 if preview_model in cache.loaded or not policy_cfg["cache_penalty"] else model_load_ms(preview_model)
        preview_compute_ms = sample_empirical_runtime_ms(preview_model, "edge", rng, latency_samples, cloud_speedup)
        preview_total_local = preview_load_ms + preview_compute_ms

        if allow_edge:
            direct_total = age_ms + preview_total_local
            actions.append(
                {
                    "final_model": preview_model,
                    "location": "edge",
                    "preview_model": preview_model,
                    "preview_ms": preview_compute_ms,
                    "load_ms": preview_load_ms,
                    "compute_ms": 0.0,
                    "comm_ms": 0.0,
                    "pred_latency_ms": direct_total,
                    "cascade_conf": preview_conf,
                    "cascade_accepted": int(preview_conf >= policy_cfg["confidence_threshold"]),
                }
            )

        for model_name in strong_models:
            if allow_edge:
                add_load_ms = 0.0 if model_name in cache.loaded or not policy_cfg["cache_penalty"] else model_load_ms(model_name)
                comp_ms = sample_empirical_runtime_ms(model_name, "edge", rng, latency_samples, cloud_speedup)
                total = age_ms + preview_total_local + add_load_ms + comp_ms
                actions.append(
                    {
                        "final_model": model_name,
                        "location": "edge",
                        "preview_model": preview_model,
                        "preview_ms": preview_compute_ms,
                        "load_ms": preview_load_ms + add_load_ms,
                        "compute_ms": comp_ms,
                        "comm_ms": 0.0,
                        "pred_latency_ms": total,
                        "cascade_conf": preview_conf,
                        "cascade_accepted": 0,
                    }
                )
            if allow_cloud:
                comp_ms = sample_empirical_runtime_ms(model_name, "cloud", rng, latency_samples, cloud_speedup)
                comm_ms = sample_network_delay_ms(network_profile, rng, network_profiles)
                total = age_ms + preview_total_local + comp_ms + comm_ms
                actions.append(
                    {
                        "final_model": model_name,
                        "location": "cloud",
                        "preview_model": preview_model,
                        "preview_ms": preview_compute_ms,
                        "load_ms": preview_load_ms,
                        "compute_ms": comp_ms,
                        "comm_ms": comm_ms,
                        "pred_latency_ms": total,
                        "cascade_conf": preview_conf,
                        "cascade_accepted": 0,
                    }
                )
        return actions

    candidate_actions = cascaded_actions() if cascade else basic_actions()

    for action in candidate_actions:
        action["score"] = utility(
            action["final_model"],
            action["pred_latency_ms"],
            action["load_ms"],
            action["comm_ms"],
            action["preview_ms"],
        )
        if cascade and action.get("preview_model") is not None:
            conf = action.get("cascade_conf", 0.0)
            if action["final_model"] == preview_model and conf < policy_cfg["confidence_threshold"]:
                action["score"] -= 0.60
            if action["final_model"] != preview_model and conf >= policy_cfg["confidence_threshold"]:
                action["score"] -= 0.15

    if policy_cfg["priority_mode"] == "srpt":
        candidate_actions = sorted(candidate_actions, key=lambda item: item["pred_latency_ms"])
        return candidate_actions[0] if candidate_actions else None
    return max(candidate_actions, key=lambda item: item["score"]) if candidate_actions else None


def pick_queue_index(
    queue,
    now_ms,
    policy_cfg,
    network_profile,
    rng,
    cache,
    resource_kind,
    model_load_bandwidth_mb_per_ms,
    *,
    model_lookup,
    variant_outputs,
    preview_model,
    strong_models,
    latency_samples,
    cloud_speedup,
    network_profiles,
):
    if not queue:
        return None, None
    window = min(len(queue), policy_cfg["scan_window"])
    if policy_cfg["priority_mode"] == "fifo":
        indices = list(range(window))
        indices.sort(key=lambda i: queue[i].arrival_ms)
    elif policy_cfg["priority_mode"] == "edf":
        indices = list(range(window))
        indices.sort(key=lambda i: queue[i].deadline_ms)
    elif policy_cfg["priority_mode"] == "srpt":
        scored = []
        for index in range(window):
            plan = estimate_plan(
                queue[index],
                now_ms,
                len(queue),
                policy_cfg,
                network_profile,
                rng,
                cache,
                resource_kind,
                model_load_bandwidth_mb_per_ms,
                model_lookup=model_lookup,
                variant_outputs=variant_outputs,
                preview_model=preview_model,
                strong_models=strong_models,
                latency_samples=latency_samples,
                cloud_speedup=cloud_speedup,
                network_profiles=network_profiles,
            )
            if plan is not None:
                scored.append((plan["pred_latency_ms"], index, plan))
        if not scored:
            return None, None
        scored.sort(key=lambda item: item[0])
        return scored[0][1], scored[0][2]
    else:
        indices = list(range(window))

    best_idx, best_plan, best_score = None, None, -1e18
    for index in indices:
        plan = estimate_plan(
            queue[index],
            now_ms,
            len(queue),
            policy_cfg,
            network_profile,
            rng,
            cache,
            resource_kind,
            model_load_bandwidth_mb_per_ms,
            model_lookup=model_lookup,
            variant_outputs=variant_outputs,
            preview_model=preview_model,
            strong_models=strong_models,
            latency_samples=latency_samples,
            cloud_speedup=cloud_speedup,
            network_profiles=network_profiles,
        )
        if plan is None:
            continue
        age_bonus = 0.001 * (now_ms - queue[index].arrival_ms)
        score = plan.get("score", 0.0) + age_bonus
        if score > best_score:
            best_idx, best_plan, best_score = index, plan, score
    return best_idx, best_plan
