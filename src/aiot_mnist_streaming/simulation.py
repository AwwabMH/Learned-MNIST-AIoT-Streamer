from __future__ import annotations

import copy
import heapq
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .arrivals import generate_arrival_times
from .scheduling import pick_queue_index


@dataclass(order=True)
class Event:
    time_ms: float
    order: int
    kind: str = field(compare=False)
    payload: dict[str, Any] = field(compare=False, default_factory=dict)


@dataclass(slots=True)
class Sample:
    sid: int
    arrival_ms: float
    deadline_ms: float
    label: int


class EdgeCache:
    def __init__(self, memory_budget_mb, model_load_bandwidth_mb_per_ms, model_lookup, *, enable_cache_penalty=True, evict_penalty_ms=0.8):
        self.memory_budget_mb = float(memory_budget_mb)
        self.model_load_bandwidth_mb_per_ms = float(model_load_bandwidth_mb_per_ms)
        self.enable_cache_penalty = bool(enable_cache_penalty)
        self.evict_penalty_ms = float(evict_penalty_ms)
        self.model_lookup = model_lookup
        self.loaded: dict[str, float] = {}
        self.used_mb = 0.0
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0

    def touch(self, model_name, now_ms):
        self.loaded[model_name] = float(now_ms)

    def ensure_loaded(self, model_name, now_ms):
        profile = self.model_lookup[model_name]
        model_mb = float(profile["memory_mb"])
        base_load_ms = 0.0 if not self.enable_cache_penalty else (model_mb / self.model_load_bandwidth_mb_per_ms)

        if model_name in self.loaded:
            self.hit_count += 1
            self.touch(model_name, now_ms)
            return 0.0, []

        self.miss_count += 1
        evicted = []
        if not self.enable_cache_penalty:
            self.loaded[model_name] = float(now_ms)
            return 0.0, evicted

        while self.used_mb + model_mb > self.memory_budget_mb and self.loaded:
            lru_model = min(self.loaded.items(), key=lambda kv: kv[1])[0]
            self.used_mb -= float(self.model_lookup[lru_model]["memory_mb"])
            del self.loaded[lru_model]
            evicted.append(lru_model)
            self.eviction_count += 1

        self.loaded[model_name] = float(now_ms)
        self.used_mb += model_mb
        load_ms = base_load_ms + len(evicted) * self.evict_penalty_ms
        return load_ms, evicted


class StreamingSimulator:
    def __init__(
        self,
        *,
        model_lookup,
        variant_outputs,
        preview_model,
        strong_models,
        latency_samples,
        cloud_speedup,
        network_profiles,
        runtime_config,
    ):
        self.model_lookup = model_lookup
        self.variant_outputs = variant_outputs
        self.preview_model = preview_model
        self.strong_models = strong_models
        self.latency_samples = latency_samples
        self.cloud_speedup = float(cloud_speedup)
        self.network_profiles = network_profiles
        self.runtime_config = runtime_config

    def _sample_empirical_runtime_ms(self, model_name, location, rng):
        draw = float(self.latency_samples[model_name][int(rng.integers(0, len(self.latency_samples[model_name])))])
        return draw if location == "edge" else max(0.05, draw / self.cloud_speedup)

    def _sample_network_delay_ms(self, profile_name, rng):
        prof = self.network_profiles[profile_name]
        tx = max(0.0, rng.normal(prof["one_way_mean_ms"], prof["one_way_std_ms"]))
        rx = max(0.0, rng.normal(prof["one_way_mean_ms"], prof["one_way_std_ms"]))
        return float(tx + rx)

    def _prune_stale(self, queue, now_ms, stale_ms, logs, scenario_meta):
        keep = []
        for sample in queue:
            if now_ms - sample.arrival_ms > stale_ms:
                logs.append(
                    {
                        "sid": sample.sid,
                        "arrival_ms": sample.arrival_ms,
                        "dispatch_ms": np.nan,
                        "finish_ms": np.nan,
                        "freshness_ms": np.nan,
                        "queue_wait_ms": np.nan,
                        "preview_ms": np.nan,
                        "load_ms": np.nan,
                        "compute_ms": np.nan,
                        "comm_ms": np.nan,
                        "processed": 0,
                        "dropped": 1,
                        "drop_reason": "stale_in_queue",
                        "deadline_met": 0,
                        "correct": np.nan,
                        "pred": np.nan,
                        "model": None,
                        "location": None,
                        "preview_model": None,
                        "cache_hit": np.nan,
                        "cache_evictions": np.nan,
                        **scenario_meta,
                    }
                )
            else:
                keep.append(sample)
        return keep

    def summarize_logs(self, logs_df, cache_stats=None):
        processed = logs_df[logs_df["processed"] == 1].copy()
        dropped = logs_df[logs_df["dropped"] == 1].copy()
        return {
            "arrivals": len(logs_df),
            "processed_count": int(len(processed)),
            "dropped_count": int(len(dropped)),
            "drop_ratio": float(len(dropped) / len(logs_df)) if len(logs_df) else np.nan,
            "mean_latency_ms": float(processed["freshness_ms"].mean()) if len(processed) else np.nan,
            "p95_latency_ms": float(processed["freshness_ms"].quantile(0.95)) if len(processed) else np.nan,
            "p99_latency_ms": float(processed["freshness_ms"].quantile(0.99)) if len(processed) else np.nan,
            "mean_freshness_ms": float(processed["freshness_ms"].mean()) if len(processed) else np.nan,
            "deadline_meet_ratio": float(processed["deadline_met"].mean()) if len(processed) else np.nan,
            "effective_accuracy": float(processed["correct"].mean()) if len(processed) else np.nan,
            "timely_correct_ratio": float(processed["timely_correct"].mean()) if len(processed) else np.nan,
            "queue_wait_mean_ms": float(processed["queue_wait_ms"].mean()) if len(processed) else np.nan,
            "preview_mean_ms": float(processed["preview_ms"].mean()) if len(processed) else np.nan,
            "load_mean_ms": float(processed["load_ms"].mean()) if len(processed) else np.nan,
            "compute_mean_ms": float(processed["compute_ms"].mean()) if len(processed) else np.nan,
            "comm_mean_ms": float(processed["comm_ms"].mean()) if len(processed) else np.nan,
            "cloud_ratio": float((processed["location"] == "cloud").mean()) if len(processed) else np.nan,
            "edge_ratio": float((processed["location"] == "edge").mean()) if len(processed) else np.nan,
            "cascade_accept_ratio": float(processed["cascade_accepted"].fillna(0).mean()) if len(processed) else np.nan,
            "cache_hit_rate": cache_stats["cache_hit_rate"] if cache_stats is not None else np.nan,
            "cache_miss_count": cache_stats["cache_miss_count"] if cache_stats is not None else np.nan,
            "cache_eviction_count": cache_stats["cache_eviction_count"] if cache_stats is not None else np.nan,
        }

    def run_scenario(
        self,
        policy_name,
        policy_cfg,
        rate_sps,
        arrival_mode,
        network_profile,
        seed,
        n_samples,
        *,
        memory_budget_mb,
        deadline_budget_ms=None,
        model_load_bandwidth_mb_per_ms=None,
        model_evict_penalty_ms=None,
    ):
        policy_cfg = copy.deepcopy(policy_cfg)
        if deadline_budget_ms is not None:
            policy_cfg["deadline_ms"] = float(deadline_budget_ms)
        memory_budget = float(memory_budget_mb)
        load_bandwidth = float(model_load_bandwidth_mb_per_ms if model_load_bandwidth_mb_per_ms is not None else self.runtime_config.model_load_bandwidth_mb_per_ms[0])
        evict_penalty = float(model_evict_penalty_ms if model_evict_penalty_ms is not None else self.runtime_config.model_evict_penalty_ms[0])
        cache = EdgeCache(
            memory_budget_mb=memory_budget,
            model_load_bandwidth_mb_per_ms=load_bandwidth,
            model_lookup=self.model_lookup,
            enable_cache_penalty=policy_cfg["cache_penalty"],
            evict_penalty_ms=evict_penalty,
        )
        rng = np.random.default_rng(seed)
        labels = self.variant_outputs["full_fp32"]["label"]
        n_samples = min(int(n_samples), len(labels))
        arrivals = generate_arrival_times(rate_sps, n_samples, arrival_mode, seed)

        queue = []
        logs = []
        qtrace = []
        events = []
        edge_busy = False
        cloud_busy = False
        event_counter = 0

        scenario_meta = {
            "policy": policy_name,
            "policy_label": policy_cfg["label"],
            "rate_sps": rate_sps,
            "arrival_mode": arrival_mode,
            "network_profile": network_profile,
            "seed": seed,
            "deadline_budget_ms": policy_cfg["deadline_ms"],
            "edge_memory_budget_mb": memory_budget,
            "model_load_bandwidth_mb_per_ms": load_bandwidth,
            "model_evict_penalty_ms": evict_penalty,
        }

        def push_event(t, kind, payload):
            nonlocal event_counter
            heapq.heappush(events, Event(float(t), event_counter, kind, payload))
            event_counter += 1

        for sid in range(n_samples):
            push_event(arrivals[sid], "arrival", {"sid": sid})

        def admit_sample(sample, now_ms):
            nonlocal queue
            queue = self._prune_stale(queue, now_ms, policy_cfg["stale_ms"], logs, scenario_meta)
            if len(queue) < policy_cfg["queue_capacity"]:
                queue.append(sample)
                return
            utilities = []
            for index, queued in enumerate(queue):
                urgency = (now_ms - queued.arrival_ms) / max(1.0, policy_cfg["stale_ms"])
                utilities.append((urgency, index))
            utilities.sort(reverse=True)
            worst_idx = utilities[0][1]
            worst = queue[worst_idx]
            new_slack = sample.deadline_ms - now_ms
            old_slack = worst.deadline_ms - now_ms
            if new_slack > old_slack:
                dropped = queue.pop(worst_idx)
                logs.append({"sid": dropped.sid, "arrival_ms": dropped.arrival_ms, "dispatch_ms": np.nan, "finish_ms": np.nan, "freshness_ms": np.nan, "queue_wait_ms": np.nan, "preview_ms": np.nan, "load_ms": np.nan, "compute_ms": np.nan, "comm_ms": np.nan, "processed": 0, "dropped": 1, "drop_reason": "replace_stalest", "deadline_met": 0, "correct": np.nan, "pred": np.nan, "model": None, "location": None, "preview_model": None, "cache_hit": np.nan, "cache_evictions": np.nan, **scenario_meta})
                queue.append(sample)
            else:
                logs.append({"sid": sample.sid, "arrival_ms": sample.arrival_ms, "dispatch_ms": np.nan, "finish_ms": np.nan, "freshness_ms": np.nan, "queue_wait_ms": np.nan, "preview_ms": np.nan, "load_ms": np.nan, "compute_ms": np.nan, "comm_ms": np.nan, "processed": 0, "dropped": 1, "drop_reason": "drop_tail", "deadline_met": 0, "correct": np.nan, "pred": np.nan, "model": None, "location": None, "preview_model": None, "cache_hit": np.nan, "cache_evictions": np.nan, **scenario_meta})

        def dispatch_if_possible(now_ms):
            nonlocal edge_busy, cloud_busy, queue
            queue = self._prune_stale(queue, now_ms, policy_cfg["stale_ms"], logs, scenario_meta)
            made_progress = True
            while made_progress:
                made_progress = False
                qtrace.append({"time_ms": now_ms, "queue_len": len(queue), **scenario_meta})
                if queue and not edge_busy:
                    idx, plan = pick_queue_index(
                        queue,
                        now_ms,
                        policy_cfg,
                        network_profile,
                        rng,
                        cache,
                        resource_kind="edge",
                        model_load_bandwidth_mb_per_ms=load_bandwidth,
                        model_lookup=self.model_lookup,
                        variant_outputs=self.variant_outputs,
                        preview_model=self.preview_model,
                        strong_models=self.strong_models,
                        latency_samples=self.latency_samples,
                        cloud_speedup=self.cloud_speedup,
                        network_profiles=self.network_profiles,
                    )
                    if idx is not None and plan is not None:
                        sample = queue.pop(idx)
                        cache_hit_before = int(plan["final_model"] in cache.loaded)
                        load_ms_final, evicted_models = cache.ensure_loaded(plan["final_model"], now_ms)
                        load_preview_ms = 0.0
                        preview_hit = np.nan
                        if plan.get("preview_model") is not None:
                            preview_hit = int(plan["preview_model"] in cache.loaded)
                            preview_load_ms, preview_evicted = cache.ensure_loaded(plan["preview_model"], now_ms)
                            load_preview_ms += preview_load_ms
                            evicted_models += preview_evicted
                        total_load_ms = plan["load_ms"] if policy_cfg["cache_penalty"] else 0.0
                        total_load_ms = max(total_load_ms, load_ms_final + load_preview_ms)
                        service_ms = plan["preview_ms"] + total_load_ms + plan["compute_ms"]
                        finish_ms = now_ms + service_ms
                        push_event(
                            finish_ms,
                            "edge_done",
                            {
                                "sample": sample,
                                "plan": plan,
                                "dispatch_ms": now_ms,
                                "finish_ms": finish_ms,
                                "cache_hit": cache_hit_before,
                                "preview_hit": preview_hit,
                                "cache_evictions": len(evicted_models),
                            },
                        )
                        edge_busy = True
                        made_progress = True

                queue = self._prune_stale(queue, now_ms, policy_cfg["stale_ms"], logs, scenario_meta)
                if queue and policy_cfg["allow_cloud"] and not cloud_busy:
                    idx, plan = pick_queue_index(
                        queue,
                        now_ms,
                        policy_cfg,
                        network_profile,
                        rng,
                        cache,
                        resource_kind="cloud",
                        model_load_bandwidth_mb_per_ms=load_bandwidth,
                        model_lookup=self.model_lookup,
                        variant_outputs=self.variant_outputs,
                        preview_model=self.preview_model,
                        strong_models=self.strong_models,
                        latency_samples=self.latency_samples,
                        cloud_speedup=self.cloud_speedup,
                        network_profiles=self.network_profiles,
                    )
                    if idx is not None and plan is not None and plan["location"] == "cloud":
                        sample = queue.pop(idx)
                        preview_hit = np.nan
                        preview_load_ms = 0.0
                        if plan.get("preview_model") is not None:
                            preview_hit = int(plan["preview_model"] in cache.loaded)
                            preview_load_ms, _ = cache.ensure_loaded(plan["preview_model"], now_ms)
                        total_load_ms = plan["load_ms"] if policy_cfg["cache_penalty"] else 0.0
                        total_load_ms = max(total_load_ms, preview_load_ms)
                        service_ms = plan["preview_ms"] + total_load_ms + plan["compute_ms"] + plan["comm_ms"]
                        finish_ms = now_ms + service_ms
                        push_event(
                            finish_ms,
                            "cloud_done",
                            {
                                "sample": sample,
                                "plan": plan,
                                "dispatch_ms": now_ms,
                                "finish_ms": finish_ms,
                                "cache_hit": np.nan,
                                "preview_hit": preview_hit,
                                "cache_evictions": 0,
                            },
                        )
                        cloud_busy = True
                        made_progress = True

        while events:
            ev = heapq.heappop(events)
            now_ms = ev.time_ms
            queue = self._prune_stale(queue, now_ms, policy_cfg["stale_ms"], logs, scenario_meta)

            if ev.kind == "arrival":
                sid = ev.payload["sid"]
                sample = Sample(sid=sid, arrival_ms=float(arrivals[sid]), deadline_ms=float(arrivals[sid] + policy_cfg["deadline_ms"]), label=int(labels[sid]))
                admit_sample(sample, now_ms)
                dispatch_if_possible(now_ms)
            elif ev.kind in {"edge_done", "cloud_done"}:
                edge_busy = edge_busy and ev.kind != "edge_done"
                cloud_busy = cloud_busy and ev.kind != "cloud_done"
                sample = ev.payload["sample"]
                plan = ev.payload["plan"]
                dispatch_ms = ev.payload["dispatch_ms"]
                finish_ms = ev.payload["finish_ms"]
                queue_wait_ms = dispatch_ms - sample.arrival_ms
                freshness_ms = finish_ms - sample.arrival_ms
                deadline_met = int(freshness_ms <= policy_cfg["deadline_ms"])
                final_model = plan["final_model"]
                pred = int(self.variant_outputs[final_model]["pred"][sample.sid])
                correct = int(self.variant_outputs[final_model]["correct"][sample.sid])
                logs.append(
                    {
                        "sid": sample.sid,
                        "arrival_ms": sample.arrival_ms,
                        "dispatch_ms": dispatch_ms,
                        "finish_ms": finish_ms,
                        "freshness_ms": freshness_ms,
                        "queue_wait_ms": queue_wait_ms,
                        "preview_ms": float(plan["preview_ms"]),
                        "load_ms": float(plan["load_ms"] if policy_cfg["cache_penalty"] else 0.0),
                        "compute_ms": float(plan["compute_ms"]),
                        "comm_ms": float(plan["comm_ms"]),
                        "processed": 1,
                        "dropped": 0,
                        "drop_reason": None,
                        "deadline_met": deadline_met,
                        "correct": correct,
                        "pred": pred,
                        "model": final_model,
                        "location": plan["location"],
                        "preview_model": plan.get("preview_model"),
                        "preview_conf": float(plan.get("cascade_conf", np.nan)),
                        "cascade_accepted": int(plan.get("cascade_accepted", 0)) if "cascade_accepted" in plan else np.nan,
                        "cache_hit": ev.payload["cache_hit"],
                        "preview_hit": ev.payload["preview_hit"],
                        "cache_evictions": ev.payload["cache_evictions"],
                        **scenario_meta,
                    }
                )
                dispatch_if_possible(now_ms)

        logs_df = pd.DataFrame(logs)
        qtrace_df = pd.DataFrame(qtrace)
        if len(logs_df):
            logs_df["total_service_ms"] = logs_df[[c for c in ["preview_ms", "load_ms", "compute_ms", "comm_ms"] if c in logs_df.columns]].fillna(0).sum(axis=1)
            logs_df["timely_correct"] = (logs_df["processed"] == 1) & (logs_df["correct"] == 1) & (logs_df["deadline_met"] == 1)
        cache_stats = {
            "cache_hit_rate": cache.hit_count / max(1, cache.hit_count + cache.miss_count),
            "cache_miss_count": cache.miss_count,
            "cache_eviction_count": cache.eviction_count,
        }
        return logs_df, qtrace_df, cache_stats
