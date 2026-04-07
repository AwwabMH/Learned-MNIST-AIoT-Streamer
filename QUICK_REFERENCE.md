# Quick Reference Guide

Fast lookup for common tasks.

## Installation (3 steps, ~2 min)

```bash
# 1. Activate virtual environment (Windows)
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify (should print no errors)
python -c "import torch, torchvision, sklearn; print('OK')"
```

## Running the Notebook

### Fastest Way: Jupyter GUI
```bash
# Activate venv, then:
jupyter notebook aiot_streaming_mnist_project_v3_final.ipynb
```

### In VS Code
1. Open workspace
2. Select Python interpreter: `.\.venv\Scripts\python.exe`
3. Open notebook file → Click "Run All"

### Command-line (Non-interactive)
```bash
# Requires papermill: pip install papermill
papermill aiot_streaming_mnist_project_v3_final.ipynb output.ipynb
```

## Configuration Cheat Sheet

Located in **Cell 1**:

```python
# Choose execution mode
QUICK_MODE = True    # 4-6 min, smallest outputs
QUICK_MODE = False   # 25-35 min, heavy experiment

# If QUICK_MODE is too slow, reduce further:
TRAIN_CFG.batch_size = 64          # Reduce from 128
TRAIN_CFG.epochs_full = 2          # Reduce from 3
EXP_CFG.samples_per_scenario = 300  # Reduce from 700

# If QUICK_MODE is too fast / you want heavier results:
QUICK_MODE = False
EXP_CFG.seeds = (11, 22, 33, 44, 55, 66)  # More seeds
EXP_CFG.rates_sps = (10, 20, 35, 60, 100, 150, 250)  # More rates
```

## Expected Runtime

| Mode | Time | Output Size | Use Case |
|------|------|-------------|----------|
| QUICK_MODE=True | 4-6 min | 200-300 MB | Debugging & testing |
| QUICK_MODE=False | 25-35 min | 1-2 GB | Final submission |
| CPU-only | 2-3× slower | Same | No GPU available |

## Key Output Files

After run completes, check `outputs_aiot_project_v2/run_<TIMESTAMP>/`:

```
Latest run contains:
✓ config.json                          # Run configuration (JSON)
✓ report_notes_generated_v3.md          # Auto-generated report (Markdown)
✓ artifacts/main_logs_all.csv           # [LARGE] All samples (~500MB+)
✓ artifacts/main_results_aggregated.csv # Summary stats (policy comparison)
✓ artifacts/leaderboard_main.csv        # Policy rankings
✓ artifacts/learned_selector.pkl        # Trained scheduler model
✓ figures/main_metrics_*.png            # Main results plots
✓ figures/selector_*.png                # Scheduler evaluation plots
✓ checkpoints/full_model.pt             # Trained CNN weights
✓ checkpoints/learned_selector.pkl      # RandomForest model
```

## Quick Data Analysis

```python
import pandas as pd
import joblib

# Load final ranking
results = pd.read_csv('outputs_aiot_project_v2/run_<TS>/artifacts/leaderboard_main.csv')
print(results.head())  # Which policy wins?

# Load all logs (warning: large file ~500MB+)
logs = pd.read_csv('artifacts/main_logs_all.csv', low_memory=False)
print(logs['correct'].mean())  # Overall accuracy
print(logs.groupby('policy')['latency_ms'].mean())  # Latency by policy

# Load selector model  
selector = joblib.load('checkpoints/learned_selector.pkl')
print(selector)  # Inspect trained model
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "CUDA out of memory" | Set `QUICK_MODE=True` or reduce batch_size to 64 |
| "Jupyter kernel crashed" | Restart kernel; close other apps; check RAM |
| "MNIST not found" | First run auto-downloads (~50MB); wait & check internet |
| "Slow execution (CPU)" | Verify "Device: cuda" in Cell 1 output; reinstall torch with GPU |
| "No outputs generated" | Check write permissions on `outputs_aiot_project_v2/`; verify free disk space |
| "Long stall during cell 16" | This is normal; main experiments take 10-15 min in FULL mode |

## Device Detection

**To check GPU availability**:

```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("GPU Memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
```

If GPU not detected:
1. Install NVIDIA drivers (latest)
2. Install CUDA toolkit matching your GPU
3. Reinstall PyTorch:
   ```bash
   pip uninstall torch torchvision -y
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

## File Management

### Keep (Important for submission)
```
✓ aiot_streaming_mnist_project_v3_final.ipynb
✓ requirements.txt
✓ README.md
✓ outputs_aiot_project_v2/run_<LATEST>/
  ├── config.json
  ├── report_notes_generated_v3.md
  ├── artifacts/
  └── figures/
```

### Can Delete (To save space)
```
✗ outputs_aiot_project_v2/run_<OLD>/    # Old runs
✗ data_mnist/MNIST/raw/                 # MNIST auto-re-downloads
✗ outputs_aiot_project_v2/run_*/artifacts/main_logs_all.csv  # If space critical
```

## Environment Variables

**Override default device** (if needed):

```python
# In Cell 1, before running:
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0 only

# Or force CPU:
DEVICE = torch.device("cpu")
```

## Interpreting Results

### What does a "good" run look like?

```
✓ All epochs converge (accuracy increases, loss decreases)
✓ self_checks_final_v3.csv has all True values
✓ main_results_aggregated.csv shows clear policy ranking
✓ Learned selector utility ≥ EDF baseline utility
✓ No "NaN" or "Inf" values in outputs
✓ All PNG figures display without corruption
```

### Key Metrics to Compare

1. **Accuracy** (top-1 %)
   - Baseline: ~98% on full model
   - Acceptable: ≥95% even with compression

2. **Latency** (ms)
   - Baseline: 10-30 ms per sample
   - Under load: may increase to 50-200 ms

3. **Deadline-met ratio** (%)
   - GOOD: ≥95% at nominal rates
   - FAIR: 70-90% under high load
   - Policy selection crucial at high load

4. **Drop rate** (%)
   - GOOD: <5% under all conditions
   - FAIR: <10% under extreme load

## Useful Pandas Commands

```python
import pandas as pd

# Load main results
df = pd.read_csv('artifacts/main_results_aggregated.csv')

# Filter by policy
edf_results = df[df['policy'] == 'edf']

# Group by rate
by_rate = df.groupby('rate_sps')['deadline_met_ratio'].mean()

# Export subset
df[['policy', 'latency_ms_mean', 'deadline_met_ratio']].to_csv('summary.csv', index=False)

# Find best policy per rate
best_by_rate = df.loc[df.groupby('rate_sps')['deadline_met_ratio'].idxmax()]
```

## Useful PyTorch Commands

```python
import torch
from pathlib import Path

# Load trained model
state = torch.load('checkpoints/full_model.pt')
model = LeNetFull()
model.load_state_dict(state)
model.eval()  # Inference mode

# Get inference time
x = torch.randn(1, 1, 28, 28)
import time
t0 = time.perf_counter()
with torch.no_grad():
    y = model(x)
t1 = time.perf_counter()
print(f"Latency: {(t1-t0)*1000:.2f} ms")
```

## Common Report Queries

**"Which policy is fastest?"**
```python
logs = pd.read_csv('artifacts/main_logs_all.csv')
print(logs.groupby('policy')['latency_ms'].mean().sort_values())
```

**"At what rate does deadline miss-ratio exceed 5%?"**
```python
results = pd.read_csv('artifacts/main_results_aggregated.csv')
miss_by_rate = 1 - results.groupby('rate_sps')['deadline_met_ratio'].mean()
critical_rate = miss_by_rate[miss_by_rate > 0.05].index.min()
print(f"Critical rate: {critical_rate} samples/sec")
```

**"Did cascade actually help?"**
```python
ablation = pd.read_csv('artifacts/ablation_results_per_seed.csv')
cascade_on = ablation[ablation['cascade'] == 1]['deadline_met_ratio'].mean()
cascade_off = ablation[ablation['cascade'] == 0]['deadline_met_ratio'].mean()
print(f"Impact: {(cascade_on - cascade_off)*100:.1f}% improvement")
```

---

## First-Time Checklist

- [ ] Python 3.12+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] GPU available (optional but recommended)
- [ ] Set `QUICK_MODE = True` for first test run
- [ ] Run Cell 1 to verify configuration
- [ ] Run "Run All" cells
- [ ] Check `outputs_aiot_project_v2/run_<TS>/` folder created
- [ ] Verify `self_checks_final_v3.csv` all True
- [ ] Switch to `QUICK_MODE = False` for final submission
- [ ] Re-run for production results

---

**See README.md for full documentation | See ARTIFACTS_INVENTORY.md for output file descriptions**
