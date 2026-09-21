# TelemetryX

TelemetryX builds lap-by-lap Formula 1 race datasets from [FastF1](https://docs.fastf1.dev/) data and uses them to estimate each driver's probability of winning at a race snapshot. A snapshot represents the information available after a completed leader lap, so the feature pipeline can be evaluated without using the final race result as an input.

The repository currently contains the data-ingestion and dataset pipeline, leakage-safe feature engineering, and a baseline logistic-regression winner model. Race data is stored as Parquet; manifests and validation reports are JSON.

## Requirements

- Python 3.12
- Internet access the first time a FastF1 session is loaded, unless the session is already in the local cache
- A platform supported by FastF1 and its dependencies

The project deliberately constrains Python to `>=3.12,<3.13` in `pyproject.toml`.

## Installation

From the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The optional dependency groups are:

```powershell
python -m pip install -e ".[ml]"    # CatBoost and joblib
python -m pip install -e ".[app]"   # Streamlit and Plotly
```

The main package dependencies are FastF1, NumPy, pandas, PyArrow, PyYAML, and scikit-learn. The `dev` group also installs pytest, coverage, Ruff, and mypy.

## Quick start

All commands below are run from the repository root. The default configuration uses the 2024 Bahrain Grand Prix race as its sample race and stores FastF1's cache under `data/raw/fastf1_cache`.

Inspect the tables available in a FastF1 session:

```powershell
python -m telemetryx.data.inspect_race
```

Prepare one race by loading, validating, cleaning, and saving its lap data:

```powershell
python -m telemetryx.data.prepare_race --season 2024 --event Bahrain --session-type R
```

Build a processed driver-lap dataset for the race:

```powershell
python -m telemetryx.data.build_dataset --season 2024 --event Bahrain --session-type R
```

Build datasets for a complete season, or for selected championship rounds:

```powershell
python -m telemetryx.data.build_season --season 2024
python -m telemetryx.data.build_season --season 2023 --rounds 1,2,3
```

Combine the saved race datasets into one validated corpus:

```powershell
python -m telemetryx.data.build_corpus
```

Existing artifacts are protected by default. Add `--overwrite` to a command when replacing an artifact is intentional. Use `--help` on any module for all available options, including lap limits and custom output directories.

## Pipeline

The normal order of operations is:

1. **Inspect** a FastF1 session and record its available tables and schema.
2. **Prepare** one race: clean lap data and write a validation report.
3. **Build** one race dataset from the cleaned session data. Each row represents a driver at a completed leader-lap snapshot and includes the `WonRace` target.
4. **Build a season** when several races are needed. Season selection follows the FastF1 schedule and keeps races in championship order.
5. **Build a corpus** from the individual race datasets.
6. **Engineer features** from the corpus and split at race level in chronological order before fitting a model.

The split configuration in `config/settings.yaml` uses the final five races of the 2023 season for validation and the 2024 season for testing. This avoids placing snapshots from the same race in different splits.

## Generated data

The configured directories have distinct roles:

| Directory | Contents |
| --- | --- |
| `data/raw/fastf1_cache` | FastF1's cached session data |
| `data/interim/inventories` | JSON inventories from `inspect_race` |
| `data/interim/cleaned_races` | Cleaned lap Parquet files and validation JSON |
| `data/processed/race_datasets` | One Parquet dataset and one JSON manifest per race |
| `data/processed/corpora` | The combined corpus Parquet file and manifest |
| `models` | Configured location for model artifacts |
| `artifacts` | Configured location for other durable artifacts |

The repository includes representative artifacts for the 2023 Bahrain and Saudi Arabian races and the 2024 Bahrain race. Generated caches and additional artifacts should remain outside source control unless they are intentionally being used as fixtures.

Race datasets use names such as:

```text
2024_01_bahrain_grand_prix_dataset.parquet
2024_01_bahrain_grand_prix_manifest.json
```

The corpus builder writes:

```text
telemetryx_race_corpus.parquet
telemetryx_race_corpus_manifest.json
```

Manifests contain counts, race metadata, and checksums so saved artifacts can be checked after writing or loading.

## Configuration

The default configuration is [config/settings.yaml](config/settings.yaml). `telemetryx.config.load_settings()` loads it, resolves relative paths from the repository root, validates individual values, and applies cross-section validation rules.

Important settings include:

- `data.seasons`: seasons used by data workflows
- `data.default_session_type`: normally `R` for race sessions
- `data.load_laps`, `data.load_weather`, and `data.load_race_control_messages`: tables loaded from FastF1
- `data.load_telemetry`: disabled by default because the current pipeline is lap-based
- `data.cache_enabled`: whether FastF1 uses the configured local cache
- `dataset.snapshot_reference`: currently `leader_lap_completion`
- `dataset.processed_file_format`: currently `parquet`
- `splitting.strategy`: currently `chronological_race_level`

Relative paths in this file are resolved against the project root, not the current shell directory. A different YAML file can be passed to `load_settings(config_path=...)` by library callers.

## Features and baseline model

Feature engineering is exposed through `telemetryx.features.engineering.engineer_race_features`. The model input contains:

- categorical values: `Driver`, `Compound`, and `TrackStatus`
- numeric race-state and pace values such as position, completion fraction, tyre life, lap times, and delta to the leader
- boolean race-state values: `IsLeader`, `IsLapped`, and `IsTopThree`

Known post-race or outcome-related columns, including `FinalPosition`, `ClassifiedPosition`, `Points`, `Status`, and `GridPosition`, are explicitly disallowed as model features.

The baseline API is in `telemetryx.modeling.baseline`:

```python
from telemetryx.modeling.baseline import (
    predict_winner_probabilities,
    train_baseline_winner_model,
)
from telemetryx.features.engineering import engineer_race_features

features = engineer_race_features(corpus)
model = train_baseline_winner_model(training_features)
predictions = predict_winner_probabilities(model, features)
```

`train_baseline_winner_model` fits preprocessing only on the supplied training frame and trains a scikit-learn logistic-regression classifier. `predict_winner_probabilities` normalizes the per-driver probabilities within each `RaceId` and `SnapshotLap`, producing a distribution over the drivers represented in that snapshot.

## Useful package modules

| Module | Purpose |
| --- | --- |
| `telemetryx.config` | Typed YAML configuration loading and validation |
| `telemetryx.data.load` | FastF1 cache configuration and session loading |
| `telemetryx.data.inspect_race` | Session inventory generation |
| `telemetryx.data.prepare_race` | Cleaning and validation artifacts for one race |
| `telemetryx.data.build_dataset` | One-race processed dataset creation |
| `telemetryx.data.build_season` | Chronological season dataset creation |
| `telemetryx.data.build_corpus` | Multi-race corpus creation and validation |
| `telemetryx.data.replay` | Race-state snapshot reconstruction |
| `telemetryx.data.targets` | Winner target construction |
| `telemetryx.features.engineering` | Leakage-safe model feature creation |
| `telemetryx.features.leakage` | Model-column and target safeguards |
| `telemetryx.modeling.preprocessing` | Feature encoding and transformation |
| `telemetryx.modeling.baseline` | Baseline model training and prediction |

## Tests and checks

Run the test suite with:

```powershell
python -m pytest
```

The project configuration also defines the following checks:

```powershell
ruff check src tests
ruff format --check src tests
python -m mypy
```

Tests cover configuration validation, FastF1 loading boundaries, cleaning, replay, target construction, artifact manifests and checksums, chronological splitting, feature leakage protection, preprocessing, and the baseline model.

## Repository layout

```text
config/                 YAML project configuration
data/                   Raw, interim, and processed data artifacts
models/                 Model output directory
notebooks/              Exploration notebooks
src/telemetryx/         Installable Python package
tests/                  Automated tests
pyproject.toml          Packaging, dependencies, and tool configuration
```
