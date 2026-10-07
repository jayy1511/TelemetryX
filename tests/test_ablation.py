from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import telemetryx.modeling.ablation as ablation
from telemetryx.features.engineering import MODEL_FEATURE_COLUMNS
from telemetryx.modeling.ablation import (
    DRIVER_FEATURE_COLUMN,
    NO_DRIVER_MODEL_COLUMNS,
    AblationExperimentError,
    run_driver_ablation,
)
from telemetryx.modeling.ablation import (
    AblationExperimentError,
    DriverAblationResult,
    run_driver_ablation,
)


def make_fake_experiment(
    *,
    log_loss: float,
    brier_score: float,
    top_one_accuracy: float,
    winner_probability: float,
    snapshot_laps: tuple[int, ...] = (1, 2, 3),
) -> Any:
    """Create a minimal experiment result for ablation tests."""
    snapshots = pd.DataFrame(
        {
            "RaceId": ["2023_18_united_states_grand_prix" for _ in snapshot_laps],
            "SnapshotLap": list(snapshot_laps),
        }
    )

    summary = SimpleNamespace(
        snapshot_count=len(snapshot_laps),
        race_count=1,
        mean_log_loss=log_loss,
        mean_brier_score=brier_score,
        top_one_accuracy=top_one_accuracy,
        mean_actual_winner_probability=winner_probability,
    )

    evaluation = SimpleNamespace(
        summary=summary,
        snapshots=snapshots,
    )

    return SimpleNamespace(
        validation_evaluation=evaluation,
    )


def test_no_driver_columns_exclude_driver_identity() -> None:
    """The ablated feature set must not expose Driver to the model."""
    assert DRIVER_FEATURE_COLUMN == "Driver"
    assert DRIVER_FEATURE_COLUMN not in NO_DRIVER_MODEL_COLUMNS


def test_no_driver_columns_preserve_every_other_feature() -> None:
    """Ablation should remove Driver and nothing else."""
    expected = tuple(column for column in MODEL_FEATURE_COLUMNS if column != "Driver")

    assert NO_DRIVER_MODEL_COLUMNS == expected

    assert len(NO_DRIVER_MODEL_COLUMNS) == (len(MODEL_FEATURE_COLUMNS) - 1)


def test_driver_ablation_runs_identical_experiments_except_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both runs should differ only in their model feature columns."""
    calls: list[dict[str, Any]] = []

    with_driver = make_fake_experiment(
        log_loss=0.10,
        brier_score=0.05,
        top_one_accuracy=0.95,
        winner_probability=0.90,
    )

    without_driver = make_fake_experiment(
        log_loss=0.25,
        brier_score=0.15,
        top_one_accuracy=0.80,
        winner_probability=0.75,
    )

    results = iter(
        [
            with_driver,
            without_driver,
        ]
    )

    def fake_run_baseline_experiment(
        corpus: pd.DataFrame,
        *,
        validation_season: int,
        validation_last_races: int,
        test_seasons: tuple[int, ...],
        model_columns: tuple[str, ...],
        max_iterations: int,
    ) -> Any:
        calls.append(
            {
                "corpus": corpus,
                "validation_season": validation_season,
                "validation_last_races": validation_last_races,
                "test_seasons": test_seasons,
                "model_columns": model_columns,
                "max_iterations": max_iterations,
            }
        )

        return next(results)

    monkeypatch.setattr(
        ablation,
        "run_baseline_experiment",
        fake_run_baseline_experiment,
    )

    corpus = pd.DataFrame(
        {
            "RaceId": [
                "2023_01_bahrain_grand_prix",
            ]
        }
    )

    result = run_driver_ablation(
        corpus,
        validation_season=2023,
        validation_last_races=5,
        test_seasons=(2024,),
        max_iterations=1500,
    )

    assert len(calls) == 2

    assert calls[0]["corpus"] is corpus
    assert calls[1]["corpus"] is corpus

    for call in calls:
        assert call["validation_season"] == 2023
        assert call["validation_last_races"] == 5
        assert call["test_seasons"] == (2024,)
        assert call["max_iterations"] == 1500

    assert calls[0]["model_columns"] == MODEL_FEATURE_COLUMNS
    assert calls[1]["model_columns"] == NO_DRIVER_MODEL_COLUMNS

    assert result.with_driver is with_driver
    assert result.without_driver is without_driver


def test_driver_ablation_comparison_contains_both_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The result should expose comparable validation metrics."""
    with_driver = make_fake_experiment(
        log_loss=0.10,
        brier_score=0.05,
        top_one_accuracy=0.95,
        winner_probability=0.90,
    )

    without_driver = make_fake_experiment(
        log_loss=0.30,
        brier_score=0.20,
        top_one_accuracy=0.75,
        winner_probability=0.65,
    )

    results = iter(
        [
            with_driver,
            without_driver,
        ]
    )

    monkeypatch.setattr(
        ablation,
        "run_baseline_experiment",
        lambda *args, **kwargs: next(results),
    )

    result = run_driver_ablation(
        pd.DataFrame({"RaceId": ["race"]}),
        validation_season=2023,
        validation_last_races=5,
        test_seasons=(2024,),
    )

    assert result.comparison["Variant"].tolist() == [
        "With Driver",
        "Without Driver",
    ]

    assert result.comparison.columns.tolist() == [
        "Variant",
        "Snapshots",
        "Races",
        "LogLoss",
        "BrierScore",
        "TopOneAccuracy",
        "MeanWinnerProbability",
    ]


def test_driver_ablation_comparison_propagates_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Comparison values should come from each validation evaluation."""
    with_driver = make_fake_experiment(
        log_loss=0.11,
        brier_score=0.06,
        top_one_accuracy=0.97,
        winner_probability=0.91,
    )

    without_driver = make_fake_experiment(
        log_loss=0.42,
        brier_score=0.24,
        top_one_accuracy=0.70,
        winner_probability=0.58,
    )

    results = iter(
        [
            with_driver,
            without_driver,
        ]
    )

    monkeypatch.setattr(
        ablation,
        "run_baseline_experiment",
        lambda *args, **kwargs: next(results),
    )

    result = run_driver_ablation(
        pd.DataFrame({"RaceId": ["race"]}),
        validation_season=2023,
        validation_last_races=5,
        test_seasons=(2024,),
    )

    with_row = result.comparison.iloc[0]
    without_row = result.comparison.iloc[1]

    assert with_row["LogLoss"] == pytest.approx(0.11)
    assert with_row["BrierScore"] == pytest.approx(0.06)
    assert with_row["TopOneAccuracy"] == pytest.approx(0.97)
    assert with_row["MeanWinnerProbability"] == pytest.approx(0.91)

    assert without_row["LogLoss"] == pytest.approx(0.42)
    assert without_row["BrierScore"] == pytest.approx(0.24)
    assert without_row["TopOneAccuracy"] == pytest.approx(0.70)
    assert without_row["MeanWinnerProbability"] == pytest.approx(0.58)


def test_driver_ablation_requires_identical_validation_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ablation runs cannot be compared on different snapshots."""
    with_driver = make_fake_experiment(
        log_loss=0.10,
        brier_score=0.05,
        top_one_accuracy=0.95,
        winner_probability=0.90,
        snapshot_laps=(1, 2, 3),
    )

    without_driver = make_fake_experiment(
        log_loss=0.20,
        brier_score=0.10,
        top_one_accuracy=0.85,
        winner_probability=0.80,
        snapshot_laps=(1, 2, 4),
    )

    results = iter(
        [
            with_driver,
            without_driver,
        ]
    )

    monkeypatch.setattr(
        ablation,
        "run_baseline_experiment",
        lambda *args, **kwargs: next(results),
    )

    with pytest.raises(
        AblationExperimentError,
        match="must evaluate identical validation snapshots",
    ):
        run_driver_ablation(
            pd.DataFrame({"RaceId": ["race"]}),
            validation_season=2023,
            validation_last_races=5,
            test_seasons=(2024,),
        )


def test_snapshot_order_does_not_affect_comparability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equal snapshot sets should compare safely despite row ordering."""
    with_driver = make_fake_experiment(
        log_loss=0.10,
        brier_score=0.05,
        top_one_accuracy=0.95,
        winner_probability=0.90,
        snapshot_laps=(1, 2, 3),
    )

    without_driver = make_fake_experiment(
        log_loss=0.20,
        brier_score=0.10,
        top_one_accuracy=0.85,
        winner_probability=0.80,
        snapshot_laps=(3, 1, 2),
    )

    results = iter(
        [
            with_driver,
            without_driver,
        ]
    )

    monkeypatch.setattr(
        ablation,
        "run_baseline_experiment",
        lambda *args, **kwargs: next(results),
    )

    result = run_driver_ablation(
        pd.DataFrame({"RaceId": ["race"]}),
        validation_season=2023,
        validation_last_races=5,
        test_seasons=(2024,),
    )

    assert len(result.comparison) == 2
