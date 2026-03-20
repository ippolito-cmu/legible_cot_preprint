# ICLR 2025 -- Legi-Val Supplemental Materials

This codebase offers *all* relevant code and data to reproduce and observe findings from `"Measuring Reasoning Trace Legibility: Can Those Who Understand Teach?"`

**NOTE**: to load the dashboard, you will need access to the database. Only reward model scores are maintained in pickle files, all else is accessible via [this huggingface repo](https://huggingface.co/datasets/anonymousicml26/sub_data/tree/main). 

You will need to unzip this file, which takes around 20GB of space to store all relevant data. When ready, move it to the outputs dir, install requirements, and load the dashboard via:

```bash
python3 -m src.viz.app_db
```

Which should run locally and give you access to all panels.

## Introduction


This repository contains:

- Scripts to generate *reasoning traces* (pickled) for several datasets.
- Analysis pipelines that compute per-trace / per-model metrics (efficiency, redundancy, reward-model scoring, pedagogical utility, backtracking).
- A small Flask dashboard that reads a SQLite database at `outputs/traces.db`.

The repo is designed so you can either:

1) Use precomputed artifacts (recommended for quick inspection), or
2) Regenerate traces + analyses (more compute intensive).


## Repository layout (high level)

- `outputs/`
	- Generated artifacts (traces, analysis outputs, and `traces.db`).
- `src/scripts/`
	- Core Python entrypoints for generation + analysis.
- `src/utils/`
	- Shared utilities (clients, graders, logging helpers, parsing, etc.).
- `src/viz/`
	- Flask dashboard (`app_db.py`) and DB access layer.


## Quickstart (use existing artifacts)

If `outputs/traces.db` and the corresponding `outputs/` artifacts are present, you can launch the dashboard directly.

1) Create a Python environment and install dependencies.

2) Ensure the SQLite DB exists (see above) (build it if needed):

```bash
python3 migrate_database.py --all
```

3) Launch the dashboard:

```bash
python3 -m src.viz.app_db
```

Then open `http://localhost:8080`.


## End-to-end reproduction (regenerate artifacts)

This is the “from scratch” path. It is compute intensive and some steps assume access to GPUs (and in one case, a SLURM cluster).

### Step 1 — Generate traces (teacher models)

You can generate traces with:

```bash
python3 -u -m src.scripts.generate_traces --model_name <provider/model> --dataset_name <math|gpqa|connections> --verbose
```

For your convenience, we have shared our wrappers as well:

- `generate_traces_math.sh`
- `generate_traces_gpqa.sh`
- `generate_traces_connections.sh`

Traces are written under:

- `outputs/traces/<dataset>/traces_<provider_model>.pkl`


### Step 2 — Run analysis pipelines

Each pipeline has a `run_*.sh` wrapper that handles sharding and merges.

1) Efficiency

```bash
bash run_efficiency_analysis.sh
```

Outputs under `outputs/efficiency_analysis/<dataset>/<model>/...`.

2) Redundancy (sentence-transformer based)

```bash
bash run_redundancy_analysis.sh
```

Outputs under `outputs/redundancy_analysis/<dataset>/<model>/...`.

3) Reward-model scoring

```bash
bash run_reward_model_analysis.sh
```

Outputs under `outputs/reward_model_analysis/<dataset>/<teacher>/...`.

4) Pedagogical utility

```bash
bash run_pedagogical_utility.sh
```

Outputs under `outputs/pu/<dataset>/<teacher>/...`.

5) Backtracking analysis (cluster/Scheduler)

```bash
bash run_backtracking_analysis.sh
```

This wrapper uses `srun` and is intended for environments with SLURM. If you do not have SLURM, you can skip this step.


### Step 3 — Optional: metric-correlation analysis

The dashboard expects a correlation JSON at `analysis/experiments/metric_correlations/cross_correlations.json`.
You can generate it from the `outputs/` artifacts via:

```bash
python3 -m src.scripts.analyze_metric_correlations --outputs_dir outputs --out_dir analysis/experiments/metric_correlations
```


### Step 4 — Build / rebuild the SQLite DB

To build the database consumed by the dashboard:

```bash
python3 migrate_database.py --all
```

To rebuild from a clean slate (backs up the existing DB first):

```bash
bash recreate_database.sh
```


### Step 5 — Launch the dashboard

```bash
python3 src/viz/app_db.py
```


## Key entrypoints (what to read first)

- `migrate_database.py`
	- Unified migration script that builds/updates `outputs/traces.db` from the artifacts in `outputs/`.

- `src/viz/app_db.py`
	- Flask dashboard entrypoint. Reads `outputs/traces.db` and serves the UI + API.

- `src/viz/db_backend.py`
	- DB access layer (SQLite). Used by the Flask app.

- `src/scripts/generate_traces.py`
	- Main trace-generation entrypoint (model + dataset + YAML config-driven overrides).

- `src/scripts/analyze_efficiency.py`, `src/scripts/analyze_redundancy.py`, `src/scripts/analyze_reward_models.py`, `src/scripts/pedagogical_utility.py`, `src/scripts/analyze_backtracking.py`
	- Core analysis implementations.
