from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from telemetryx.features.engineering import MODEL_FEATURE_COLUMNS
from telemetryx.modeling.experiment import (
    BaselineExperimentResult,
    run_baseline_experiment,
)
from telemetryx.modeling.stage_evaluation import summarize_race_stages

DRIVER_FEATURE_COLUMN = "Driver"

NO_DRIVER_MODEL_COLUMNS: tuple[str, ...] = tuple(
    column for column in MODEL_FEATURE_COLUMNS if column != DRIVER_FEATURE_COLUMN
)


class AblationExperimentError(RuntimeError):
    """Raised when an ablation experiment cannot be compared safely."""


@dataclass(frozen=True)
class DriverAblationResult:
    """Validation results with and without driver identity."""

    with_driver: BaselineExperimentResult
    without_driver: BaselineExperimentResult
    comparison: pd.DataFrame


def run_driver_ablation(
    corpus: pd.DataFrame,
    *,
    validation_season: int,
    validation_last_races: int,
    test_seasons: Sequence[int],
    max_iterations: int = 2000,
) -> DriverAblationResult:
    """
    Compare the baseline model with and without driver identity.

    Both experiments use the same race split, preprocessing strategy,
    logistic-regression configuration, and validation data. The only
    intentional difference is whether ``Driver`` is available as a
    predictive feature.
    """
    with_driver = run_baseline_experiment(
        corpus,
        validation_season=validation_season,
        validation_last_races=validation_last_races,
        test_seasons=test_seasons,
        model_columns=MODEL_FEATURE_COLUMNS,
        max_iterations=max_iterations,
    )

    without_driver = run_baseline_experiment(
        corpus,
        validation_season=validation_season,
        validation_last_races=validation_last_races,
        test_seasons=test_seasons,
        model_columns=NO_DRIVER_MODEL_COLUMNS,
        max_iterations=max_iterations,
    )

    _validate_comparable_experiments(
        with_driver,
        without_driver,
    )

    comparison = _build_driver_ablation_comparison(
        with_driver,
        without_driver,
    )

    return DriverAblationResult(
        with_driver=with_driver,
        without_driver=without_driver,
        comparison=comparison,
    )


def _build_driver_ablation_comparison(
    with_driver: BaselineExperimentResult,
    without_driver: BaselineExperimentResult,
) -> pd.DataFrame:
    evaluations = (
        ("With Driver", with_driver.validation_evaluation),
        ("Without Driver", without_driver.validation_evaluation),
    )

    rows: list[dict[str, object]] = []

    for variant, evaluation in evaluations:
        summary = evaluation.summary

        rows.append(
            {
                "Variant": variant,
                "Snapshots": summary.snapshot_count,
                "Races": summary.race_count,
                "LogLoss": summary.mean_log_loss,
                "BrierScore": summary.mean_brier_score,
                "TopOneAccuracy": summary.top_one_accuracy,
                "MeanWinnerProbability": (summary.mean_actual_winner_probability),
            }
        )

    return pd.DataFrame(rows)


def build_driver_ablation_stage_comparison(
    result: DriverAblationResult,
) -> pd.DataFrame:
    """Compare Driver ablation performance across race stages."""
    with_driver_summary = summarize_race_stages(
        result.with_driver.validation_evaluation.snapshots
    )

    without_driver_summary = summarize_race_stages(
        result.without_driver.validation_evaluation.snapshots
    )

    _validate_comparable_stage_summaries(
        with_driver_summary,
        without_driver_summary,
    )

    frames: list[pd.DataFrame] = []

    for variant, summary in (
        ("With Driver", with_driver_summary),
        ("Without Driver", without_driver_summary),
    ):
        variant_summary = summary.copy(deep=True)
        variant_summary.insert(
            0,
            "Variant",
            variant,
        )
        frames.append(variant_summary)

    return pd.concat(
        frames,
        ignore_index=True,
    )


def _validate_comparable_stage_summaries(
    with_driver: pd.DataFrame,
    without_driver: pd.DataFrame,
) -> None:
    """Require both ablation variants to summarize identical race stages."""
    comparison_columns = [
        "RaceStage",
        "SnapshotCount",
        "RaceCount",
    ]

    with_structure = with_driver.loc[
        :,
        comparison_columns,
    ].reset_index(drop=True)

    without_structure = without_driver.loc[
        :,
        comparison_columns,
    ].reset_index(drop=True)

    if not with_structure.equals(without_structure):
        raise AblationExperimentError(
            "Driver ablation stage summaries must represent "
            "identical validation stages."
        )


def _validate_comparable_experiments(
    with_driver: BaselineExperimentResult,
    without_driver: BaselineExperimentResult,
) -> None:
    with_snapshots = with_driver.validation_evaluation.snapshots.loc[
        :,
        [
            "RaceId",
            "SnapshotLap",
        ],
    ].reset_index(drop=True)

    without_snapshots = without_driver.validation_evaluation.snapshots.loc[
        :,
        [
            "RaceId",
            "SnapshotLap",
        ],
    ].reset_index(drop=True)

    with_snapshots = with_snapshots.sort_values(
        by=[
            "RaceId",
            "SnapshotLap",
        ],
        kind="stable",
    ).reset_index(drop=True)

    without_snapshots = without_snapshots.sort_values(
        by=[
            "RaceId",
            "SnapshotLap",
        ],
        kind="stable",
    ).reset_index(drop=True)

    if not with_snapshots.equals(without_snapshots):
        raise AblationExperimentError(
            "Driver ablation experiments must evaluate identical validation snapshots."
        )
