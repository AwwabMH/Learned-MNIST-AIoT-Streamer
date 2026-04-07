# AIoT Streaming MNIST

An end-to-end, GitHub-ready Python project for streaming MNIST classification in an AIoT edge/cloud setting. The repo packages the original notebook workflow into a modular codebase with reusable training, compression, benchmarking, simulation, reporting, and learned-policy components.

## What’s included

- LeNet-style MNIST models for a full and preview path.
- Global pruning and dynamic quantization variants.
- Empirical latency benchmarking and a merged model catalog.
- Event-driven edge/cloud simulation with cache pressure, deadlines, and cascade inference.
- FIFO, EDF, SRPT, and adaptive policy baselines.
- Ablation runs and a learned selector prototype.
- Generated figures, CSV artifacts, and report notes.

## Project Layout

```text
src/aiot_mnist_streaming/
  config.py          Project configuration and reproducibility helpers
  data.py            MNIST loading and data module
  models.py          LeNetFull and LeNetTiny
  training.py        Model training and evaluation
  compression.py     Pruning and quantization utilities
  benchmarking.py    Prediction, metrics, and latency benchmarking
  catalog.py         Model catalog assembly
  arrivals.py        Synthetic arrival process generators
  scheduling.py      Policy library and action scoring
  simulation.py      Event-driven simulator and cache model
  learning.py        Learned selector dataset and training
  reporting.py       Figures and report-note generation
  pipeline.py        End-to-end orchestration
  cli.py             Command-line entrypoint
main.py              Convenience launcher
aiot_streaming_mnist_project_v3_final.ipynb  Original notebook reference
```

## Quick Start

### 1. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

For a package-style install with the local CLI, you can also use:

```powershell
pip install -e .
```

### 3. Run the project

Quick mode for development:

```powershell
python main.py --quick
```

Module entrypoint:

```powershell
python -m aiot_mnist_streaming --quick
```

Full run:

```powershell
python main.py --full
```

## Outputs

Each run writes to `outputs_aiot_project_v2/run_<timestamp>/` with:

- `config.json` for reproducibility.
- `checkpoints/` for trained weights and selector artifacts.
- `artifacts/` for CSV exports and learned selector outputs.
- `figures/` for architecture, training, experiment, and selector plots.

The repository intentionally ignores these outputs so the Git history stays lightweight.

## Notes

- The original notebook is kept in the repo for reference, but the package code is the source of truth for maintainable runs.
- The project is configured for Python 3.12+.
- MNIST is downloaded automatically into `data_mnist/` when the pipeline runs.

## Development Tips

- Use `--quick` first to confirm your environment.
- Switch to `--full` only when you want the full experiment sweep.
- If you only want the notebook view, open `aiot_streaming_mnist_project_v3_final.ipynb` directly in VS Code or Jupyter.

#### 4. **Slow execution (not using GPU)**
   - Verify: Check output message "Device: cuda" or "Device: cpu"
   - If CPU: Install GPU drivers and matching CUDA toolkit; reinstall PyTorch with GPU support

#### 5. **Outputs not generating**
   - Check that `output_aiot_project_v2/` directory has write permissions
   - On Windows: Run Jupyter as Administrator if permission denied
   - Verify free disk space: outputs require ~500MB (QUICK) to 3GB (FULL)

### Debugging Tips

- **Progress tracking**: Each major section prints status updates
- **Intermediate checks**: Cell 27 (`self_checks_final_v3.csv`) validates all artifacts exist
- **Verbose logging**: Add `pd.set_option('display.max_columns', None)` to view full DataFrames
- **Memory usage**: Monitor with `import psutil; psutil.virtual_memory()`

## Project Structure

```
├── aiot_streaming_mnist_project_v3_final.ipynb  # Main notebook (all-in-one)
├── requirements.txt                              # Python dependencies
├── README.md                                     # This file
├── data_mnist/                                   # MNIST dataset (auto-generated)
│   └── MNIST/raw/                               # Raw ubyte files
├── outputs_aiot_project_v2/                      # Output directory
│   ├── run_20260407_210314/                      # Latest run
│   │   ├── config.json                           # Config snapshot
│   │   ├── artifacts/                            # CSV/PKL exports
│   │   ├── checkpoints/                          # Model weights
│   │   └── figures/                              # PNG plots
│   └── report_notes_generated_v3.md              # Final report
├── currentnote.py                                # (Optional note files)
└── test_kernel.py                                # (Optional test files)
```

## Key Methodologies

### 1. Cascaded Inference
```
Input Sample
    ↓
Lightweight Model (Preview)
    ↓ (confidence ≥ threshold?)
   YES → Return prediction (low latency) ✓
    ↓ (confidence < threshold?)
    NO → Full Model (Edge)
    ↓ (still uncertain?)
    NO → Cloud (stronger model, higher latency)
    ↓
Return best prediction
```

### 2. Dynamic Model Loading
- Device has fixed memory budget (e.g., 2.5 MB)
- Models loaded on-demand when selected by policy
- Eviction of LRU (least-recently-used) model if memory exhausted
- Loading latency: `load_time = model_size / bandwidth_mbs`

### 3. Scheduling Policies
- **FIFO**: First-in-first-out queue (baseline)
- **EDF**: Earliest-deadline-first (real-time scheduling)
- **SRPT**: Shortest-remaining-process-time (minimize mean latency)
- **Adaptive**: EDF with deadline slack awareness + cloud escalation
- **Adaptive+Cascade**: Adaptive + preview model cascade for fast-path decisions
- **Learned Selector**: Random forest predicts best policy per sample

### 4. Latency Components
For each sample:
- **Arrival** → **Queue Wait** → **Model Loading** → **Compute** → **Communication** → **Latency (ms)**

### 5. Learned Selector Training
- **Features**: arrival_gap_ms, arrival_rate, queue_depth, memory_usage, last_deadline_miss, ...
- **Target**: Oracle policy (chosen offline based on hindsight results)
- **Model**: RandomForest with 120 trees, max_depth=8, stratified sampling
- **Evaluation**: Held-out test set (different seeds), utility comparison vs baselines

## Performance Expectations

### Typical Runtimes (by `QUICK_MODE`)

| Phase | QUICK_MODE=True | QUICK_MODE=False |
|-------|-----------------|------------------|
| Setup | <1 min | <1 min |
| Model Training | 2-3 min | 8-12 min |
| Main Experiments | 1-2 min | 10-15 min |
| Ablation Study | 30 sec | 3-5 min |
| Step 1-3 Analysis | 30 sec | 1-2 min |
| **Total** | **~4-6 min** | **~25-35 min** |

### Typical Output Sizes

| Component | QUICK | FULL |
|-----------|-------|------|
| Artifacts (CSV/PKL) | 80-150 MB | 800-1200 MB |
| Figures (PNG) | 40-60 MB | 100-150 MB |
| Checkpoints (PT) | 50-100 MB | TBD |
| **Total** | **~200-300 MB** | **~1-2 GB** |

## Citation & Acknowledgments

This work combines:
- **MNIST CNN Training**: PyTorch tutorial architecture
- **Model Compression**: PyTorch pruning + quantization
- **Streaming Simulator**: Custom Python event-driven architecture
- **Scheduling Policies**: Academic baselines (FIFO, EDF, SRPT)
- **Learned Control**: Scikit-learn RandomForest classifier

## Support & Questions

If you encounter issues:

1. **Check reproducibility**: Run with `QUICK_MODE=True` first
2. **Review logs**: Check `outputs_aiot_project_v2/run_<TS>/report_notes_generated_v3.md`
3. **Inspect artifacts**: Use `pandas.read_csv()` to load and inspect logs
4. **Reduce scale**: Lower seeds, rates, samples_per_scenario in ExperimentConfig

## Version History

- **V3 (Current)**: Full streaming simulator with cascaded inference, dynamic loading, scheduling policies, ablation study, learned selector
- **V2**: Baseline multi-seed evaluation without cascade
- **V1**: Single policy, single seed proof-of-concept

---

**Last Updated**: April 7, 2026  
**Python**: 3.12.0  
**Key Packages**: PyTorch 2.5.1, torchvision 0.20.1, scikit-learn 1.8.0, pandas 3.0.2, numpy 2.4.4
