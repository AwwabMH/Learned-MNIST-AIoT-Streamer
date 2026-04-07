from __future__ import annotations

import numpy as np


def generate_arrival_times(rate_sps, n_samples, mode, seed):
    rng = np.random.default_rng(seed)
    rate_sps = float(rate_sps)
    if mode == "periodic":
        inter = 1000.0 / rate_sps
        return np.arange(n_samples, dtype=float) * inter
    if mode == "poisson":
        inter = rng.exponential(scale=1000.0 / rate_sps, size=n_samples)
        return np.cumsum(inter) - inter[0]
    if mode == "bursty":
        t = 0.0
        times = []
        phase = "on"
        remaining = int(rng.integers(20, 60))
        while len(times) < n_samples:
            local_rate = rate_sps * 2.6 if phase == "on" else max(1.0, rate_sps * 0.35)
            inter = rng.exponential(scale=1000.0 / local_rate)
            t += inter
            times.append(t)
            remaining -= 1
            if remaining <= 0:
                phase = "off" if phase == "on" else "on"
                remaining = int(rng.integers(20, 60))
        return np.array(times, dtype=float)
    raise ValueError(f"Unknown arrival mode: {mode}")
