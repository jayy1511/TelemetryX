# TelemetryX

TelemetryX is a leakage-safe, lap-by-lap Formula 1 winner-probability system that reconstructs historical races as temporal snapshots and estimates each driver's probability of eventually winning using information available up to that point.

## Why TelemetryX

Race winner prediction is a temporal modeling problem: a useful prediction must change as the race develops, and must not use information that only becomes known after the race. TelemetryX makes that constraint explicit in its replay, target, split, and feature contracts, then evaluates probability quality at the complete race-snapshot level.

## What It Does

- Loads historical sessions through FastF1 and validates and cleans lap data.
- Reconstructs each race at completed leader-lap boundaries.
- Builds the eventual-winner target from final results, then creates per-race and season datasets and a multi-race corpus.
- Splits races chronologically, engineers audited features, fits a logistic-regression baseline, and normalizes driver probabilities within each race snapshot.
- Reports validation metrics, reference benchmarks, early/middle/late race-stage analysis, and a `Driver` feature ablation.

The current experiment is batch-oriented and uses historical lap/session data. It is not a live-race prediction service.

## Example Output

Each prediction row identifies a race, snapshot lap, driver, and normalized probability:

```text
RaceId | SnapshotLap | Driver | WinnerProbability
...    | ...         | ...    | p_driver
```

Within every `(RaceId, SnapshotLap)` group, probabilities sum to 1.0 across the drivers represented in that snapshot. `p_driver` above is a schema placeholder, not a reported prediction.

## Architecture

```text
FastF1 session data
    -> validation and cleaning
    -> race preparation
    -> temporal replay
    -> winner targets from final results
    -> per-race datasets
    -> season datasets
    -> race corpus
    -> chronological train / validation / test split
    -> leakage-safe feature engineering
    -> preprocessing fitted on training data
    -> logistic-regression baseline
    -> normalized winner probabilities
    -> validation evaluation and benchmarks
    -> race-stage analysis and Driver ablation
```

## Leakage Prevention

- A replay snapshot at lap `N` uses lap records only through lap `N`; later-lap records are excluded.
- The `Position` feature is the driver's current position in that snapshot. Final race `Position` is used only to create `WonRace`; it is not carried into the model feature frame. Final-result fields such as `Status`, `ClassifiedPosition`, `Points`, and `GridPosition` are rejected as model features.
- `WonRace` is the supervised target derived from the final result, not a predictor.
- Splitting happens at complete-race boundaries before feature engineering. No race is shared across train, validation, and test.
- Imputation, one-hot encoding, missing-value indicators, and scaling are fitted using training features only. Validation features use the fitted training transformations.
- The 2024 season is reserved as the final test holdout. The current experiment does not transform, predict, or evaluate its rows.

## Dataset and Split

The current local corpus manifest describes 46 races: 22 from 2023 and 24 from 2024, with 54,891 driver-snapshot rows, 2,769 race snapshots, and 25 drivers. These are locally generated processed artifacts; their presence in a local workspace does not mean they are committed to git.

| Split | Races | Rows | Assignment |
| --- | ---: | ---: | --- |
| Train | 17 | 20,210 | First 17 races of 2023 |
| Validation | 5 | 6,049 | Final 5 races of 2023 |
| Test holdout | 24 | 28,632 | All of 2024; untouched |

Validation races are the 2023 United States, Mexico City, Sao Paulo, Las Vegas, and Abu Dhabi Grands Prix. The split defaults are defined in [config/settings.yaml](config/settings.yaml) and the experiment CLI.

## Feature Set

The model feature schema in `telemetryx.features.engineering` is:

| Type | Features |
| --- | --- |
| Categorical | `Driver`, `Compound`, `TrackStatus` |
| Numeric | `SnapshotLap`, `Position`, `FieldSize`, `PositionFraction`, `CompletedLaps`, `CompletionFraction`, `LapsBehindLeader`, `Stint`, `TyreLife`, `LastLapTimeSeconds`, `AverageLapTimeSeconds`, `LeaderLastLapTimeSeconds`, `LastLapDeltaToLeaderSeconds` |
| Boolean | `IsLeader`, `IsLapped`, `IsTopThree` |

Derived race-state features are computed within the temporal replay snapshot. The `Driver` column is both a categorical feature in the baseline and the feature removed by the ablation experiment.

## Modeling Approach

The baseline is scikit-learn logistic regression. Preprocessing imputes categorical missing values with a sentinel and one-hot encodes them; numeric and Boolean values use median imputation with missing-value indicators, followed by scaling. Unknown categories are ignored by the encoder.

The classifier first produces independent per-driver probabilities. TelemetryX then normalizes those scores within each `(RaceId, SnapshotLap)` so they form one distribution over the drivers represented in that snapshot.

## Evaluation Metrics

Metrics are computed once per race snapshot, not once per driver row. Log loss measures the probability assigned to the actual winner; the Brier score measures the squared error across the snapshot's driver probabilities. Lower is better for both. Top-1 accuracy records whether the actual winner has the unique highest probability. Mean winner probability is the average probability assigned to the actual winner.

Top-1 alone is insufficient for a probability model: it ignores how much probability the model assigns to the winner and whether the distribution is overconfident. Log loss and Brier score therefore accompany ranking accuracy and mean winner probability. Calibration analysis is not yet part of the current evaluation.

## Current Validation Results

Results below are from the five-race 2023 validation set (306 snapshots). They are validation evidence, not final generalization results; the 2024 test holdout has not been evaluated.

| Model | Log loss | Brier score | Top-1 accuracy | Mean winner probability |
| --- | ---: | ---: | ---: | ---: |
| TelemetryX logistic | 0.1147 | 0.0610 | 97.06% | 91.33% |

The CLI prints log loss and Brier score at full precision: `0.114718` and `0.060974`.

## Benchmark Comparison

All predictors are compared on the same 306 validation snapshots. The historical driver prior estimates smoothed win frequency from training races only, with one target observation per driver and race.

| Predictor | Log loss | Brier score | Top-1 accuracy | Mean winner probability |
| --- | ---: | ---: | ---: | ---: |
| Uniform | 2.9838 | 0.9494 | 0.00% | 5.06% |
| Historical driver prior | 0.9991 | 0.4305 | 100.00% | 36.82% |
| TelemetryX logistic | 0.1147 | 0.0610 | 97.06% | 91.33% |

The historical prior illustrates why top-1 is not enough: it ranked the winner first in every validation snapshot but assigned substantially less probability to the winner on average and had much worse log loss and Brier score than the logistic baseline.

## Race-Stage Analysis

Stages are assigned per race using `SnapshotLap / maximum SnapshotLap`: Early is at most one-third progress, Middle is greater than one-third through two-thirds, and Late is greater than two-thirds. The final snapshot is used only to group evaluation results, not as a model feature.

| Stage | Snapshots | Log loss | Brier score | Top-1 accuracy | Mean winner probability | Mean winner rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Early | 99 | 0.1039 | 0.0505 | 98.99% | 91.48% | 1.01 |
| Middle | 103 | 0.2041 | 0.1211 | 93.20% | 85.63% | 1.07 |
| Late | 104 | 0.0365 | 0.0114 | 99.04% | 96.82% | 1.01 |

## Driver Feature Ablation

The two runs use the same split, estimator, preprocessing, and validation snapshots; the only intended difference is whether `Driver` is included as a model feature.

| Feature set | Log loss | Brier score | Top-1 accuracy | Mean winner probability |
| --- | ---: | ---: | ---: | ---: |
| With `Driver` | 0.1147 | 0.0610 | 97.06% | 91.33% |
| Without `Driver` | 1.0884 | 0.4219 | 76.47% | 69.96% |

| Feature set | Stage | Snapshots | Log loss | Brier score | Top-1 accuracy | Mean winner probability | Mean winner rank |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| With `Driver` | Early | 99 | 0.1039 | 0.0505 | 98.99% | 91.48% | 1.01 |
| With `Driver` | Middle | 103 | 0.2041 | 0.1211 | 93.20% | 85.63% | 1.07 |
| With `Driver` | Late | 104 | 0.0365 | 0.0114 | 99.04% | 96.82% | 1.01 |
| Without `Driver` | Early | 99 | 1.2976 | 0.4814 | 73.74% | 67.88% | 1.87 |
| Without `Driver` | Middle | 103 | 1.6897 | 0.6646 | 62.14% | 56.92% | 2.21 |
| Without `Driver` | Late | 104 | 0.2937 | 0.1250 | 93.27% | 84.85% | 1.10 |

On these five validation races, driver identity contributes strongly in early and middle stages. Race-state features are more sufficient late in the race, where the model without `Driver` improves substantially. This is a small validation sample, not evidence of final generalization.

## Project Structure

```text
config/                 Validated YAML settings
data/raw/               FastF1 cache
data/interim/           Inspection, cleaned-race, and validation artifacts
data/processed/         Per-race datasets and the combined corpus (generated locally)
src/telemetryx/data/    Loading, cleaning, replay, targets, datasets, splits, corpus
src/telemetryx/features/ Leakage checks and feature engineering
src/telemetryx/modeling/ Preprocessing, baseline, experiment, evaluation, benchmarks
tests/                  Automated tests
pyproject.toml          Dependencies and tool configuration
```

## Setup and Installation

Python `>=3.12,<3.13` is required. From the repository root, create an environment and install the package with its development tools:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

FastF1 may need internet access to retrieve sessions not already cached. Its cache is configured under `data/raw/fastf1_cache` by default.

## Reproducing the Processed Corpus

Build the 2023 and 2024 race datasets, then combine them into the corpus used by the experiment. Commands run from the repository root:

```powershell
python -m telemetryx.data.build_season --season 2023
python -m telemetryx.data.build_season --season 2024
python -m telemetryx.data.build_corpus
```

The corpus builder defaults to `data/processed/corpora/telemetryx_race_corpus.parquet`, which is also the experiment's default input. Season builds default to `data/processed/race_datasets`. Use each command's `--help` for output-directory, round-selection, lap-range, and overwrite options. These commands generate local processed data; the README does not imply those generated files are tracked in git.

## Running the Experiment CLI

With the corpus available at its default path:

```powershell
python -m telemetryx.modeling.run_experiment
```

The CLI runs the baseline validation experiment and Driver ablation, then prints validation benchmarks and race-stage summaries. It leaves 2024 rows untouched. Available options, with defaults from `run_experiment.py`, are:

| Option | Default | Purpose |
| --- | --- | --- |
| `--corpus PATH` | `data/processed/corpora/telemetryx_race_corpus.parquet` | Input corpus |
| `--validation-season INT` | `2023` | Season containing the validation tail |
| `--validation-last-races INT` | `5` | Number of final races reserved for validation |
| `--test-seasons SEASONS` | `2024` | Comma-separated complete holdout seasons |
| `--max-iterations INT` | `2000` | Logistic-regression iteration limit |

Run `python -m telemetryx.modeling.run_experiment --help` for parser help.

## Tests and Quality Checks

The project quality gate is:

```powershell
ruff format .
ruff check .
mypy
pytest
```

The automated suite covers data validation and cleaning, replay and targets, artifact integrity, chronological splitting, leakage safeguards, preprocessing, model evaluation, benchmarks, race-stage analysis, and ablation.

## Current Project Status

TelemetryX is in active development. The data pipeline, corpus split, logistic baseline, validation benchmarks, stage analysis, and Driver ablation are implemented. The reported results are from five 2023 validation races. The 24-race 2024 season remains the untouched final holdout and has not been evaluated.

## Known Limitations

- Validation covers only five races from one season; results may not generalize to other seasons or race conditions.
- The Driver ablation shows that the current model depends substantially on driver identity, especially early and mid-race.
- The current estimator is a logistic-regression baseline; calibration analysis and a stronger alternative model are not yet included in the reported validation workflow.
- The workflow consumes historical FastF1 session data and produces batch experiment output; it does not provide live inference or a production service.

## Planned Next Steps

- Add a current-position / leader-based benchmark and further diagnostics.
- Analyze probability calibration and consider one stronger alternative model.
- Freeze model-selection decisions before the final one-time evaluation on the untouched 2024 holdout.
- Prepare a final report, visualizations, and portfolio presentation.

## Tech Stack

Python 3.12, FastF1, pandas, NumPy, PyArrow, scikit-learn, PyYAML, pytest, Ruff, and mypy.
