from __future__ import annotations

import pandas as pd
import pytest

import telemetryx.modeling.diagnostics as diagnostics
from telemetryx.modeling.diagnostics import (
    ModelDiagnosticsError,
    summarize_race_stage_performance,
)
from telemetryx.modeling.evaluation import ModelEvaluationError


def make_staged_snapshots() -> pd.DataFrame:
    """Build deterministic staged snapshot metrics for two races."""
    return pd.DataFrame(
        {
            "RaceId": [
                "race_b",
                "race_a",
                "race_a",
                "race_a",
                "race_b",
            ],
            "RaceStage": [
                "Middle",
                "Early",
                "Early",
                "Middle",
                "Early",
            ],
            "SnapshotLogLoss": [
                0.80,
                0.10,
                0.30,
                0.50,
                0.20,
            ],
            "SnapshotBrierScore": [
                0.40,
                0.05,
                0.15,
                0.20,
                0.10,
            ],
            "TopOneCorrect": [
                False,
                True,
                False,
                True,
                True,
            ],
            "ActualWinnerProbability": [
                0.40,
                0.90,
                0.70,
                0.60,
                0.80,
            ],
            "ActualWinnerRank": [
                2,
                1,
                2,
                1,
                1,
            ],
        }
    )


def test_race_stage_summary_aggregates_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metrics should be averaged within each race-stage group."""
    staged = make_staged_snapshots()
    raw = pd.DataFrame({"placeholder": [1]})

    monkeypatch.setattr(
        diagnostics,
        "validate_snapshot_evaluation",
        lambda frame: None,
    )

    monkeypatch.setattr(
        diagnostics,
        "add_race_stages",
        lambda frame: staged.copy(deep=True),
    )

    summary = summarize_race_stage_performance(raw)

    race_a_early = summary.loc[
        summary["RaceId"].eq("race_a") & summary["RaceStage"].eq("Early")
    ].iloc[0]

    assert race_a_early["SnapshotCount"] == 2
    assert race_a_early["MeanLogLoss"] == pytest.approx(0.20)
    assert race_a_early["MeanBrierScore"] == pytest.approx(0.10)
    assert race_a_early["TopOneAccuracy"] == pytest.approx(0.50)
    assert race_a_early["MeanWinnerProbability"] == pytest.approx(0.80)
    assert race_a_early["MeanWinnerRank"] == pytest.approx(1.50)


def test_race_stage_summary_uses_expected_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Diagnostics should expose a stable race-stage summary schema."""
    staged = make_staged_snapshots()

    monkeypatch.setattr(
        diagnostics,
        "validate_snapshot_evaluation",
        lambda frame: None,
    )

    monkeypatch.setattr(
        diagnostics,
        "add_race_stages",
        lambda frame: staged.copy(deep=True),
    )

    summary = summarize_race_stage_performance(pd.DataFrame({"placeholder": [1]}))

    assert summary.columns.tolist() == [
        "RaceId",
        "RaceStage",
        "SnapshotCount",
        "MeanLogLoss",
        "MeanBrierScore",
        "TopOneAccuracy",
        "MeanWinnerProbability",
        "MeanWinnerRank",
    ]


def test_race_stage_summary_sorts_races_and_stage_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Output should be deterministic by race and Early/Middle/Late order."""
    staged = make_staged_snapshots()

    monkeypatch.setattr(
        diagnostics,
        "validate_snapshot_evaluation",
        lambda frame: None,
    )

    monkeypatch.setattr(
        diagnostics,
        "add_race_stages",
        lambda frame: staged.copy(deep=True),
    )

    summary = summarize_race_stage_performance(pd.DataFrame({"placeholder": [1]}))

    assert list(
        zip(
            summary["RaceId"],
            summary["RaceStage"],
            strict=True,
        )
    ) == [
        ("race_a", "Early"),
        ("race_a", "Middle"),
        ("race_b", "Early"),
        ("race_b", "Middle"),
    ]


def test_race_stage_summary_wraps_invalid_snapshot_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid evaluation frames should receive diagnostic context."""

    def fail_validation(
        frame: pd.DataFrame,
    ) -> None:
        raise ModelEvaluationError("synthetic validation failure")

    monkeypatch.setattr(
        diagnostics,
        "validate_snapshot_evaluation",
        fail_validation,
    )

    with pytest.raises(
        ModelDiagnosticsError,
        match="invalid for race-stage diagnostics",
    ):
        summarize_race_stage_performance(pd.DataFrame({"placeholder": [1]}))


def test_race_stage_summary_rejects_unknown_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected stage labels should never silently enter diagnostics."""
    staged = make_staged_snapshots()

    staged.loc[
        staged.index[0],
        "RaceStage",
    ] = "Unknown"

    monkeypatch.setattr(
        diagnostics,
        "validate_snapshot_evaluation",
        lambda frame: None,
    )

    monkeypatch.setattr(
        diagnostics,
        "add_race_stages",
        lambda frame: staged.copy(deep=True),
    )

    with pytest.raises(
        ModelDiagnosticsError,
        match="unknown race stage",
    ):
        summarize_race_stage_performance(pd.DataFrame({"placeholder": [1]}))
