from __future__ import annotations

import pandas as pd
import pytest

import telemetryx.modeling.diagnostics as diagnostics
from telemetryx.features.engineering import MODEL_FEATURE_COLUMNS
from telemetryx.modeling.diagnostics import (
    ModelDiagnosticsError,
    add_predicted_winner_details,
    select_driver_snapshot_features,
    select_worst_snapshots,
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
            "SnapshotLap": [
                20,
                5,
                10,
                25,
                8,
            ],
            "ActualWinner": [
                "VER",
                "VER",
                "VER",
                "VER",
                "VER",
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


def test_select_worst_snapshots_orders_by_log_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worst snapshots should appear first."""
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

    worst = select_worst_snapshots(
        raw,
        limit=3,
    )

    assert worst["SnapshotLogLoss"].tolist() == [
        0.80,
        0.50,
        0.30,
    ]


def test_select_worst_snapshots_filters_race_and_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Diagnostics should support focused race-stage investigation."""
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

    worst = select_worst_snapshots(
        raw,
        race_id="race_a",
        race_stage="Early",
        limit=10,
    )

    assert worst["RaceId"].tolist() == [
        "race_a",
        "race_a",
    ]

    assert worst["RaceStage"].tolist() == [
        "Early",
        "Early",
    ]

    assert worst["SnapshotLogLoss"].tolist() == [
        0.30,
        0.10,
    ]


@pytest.mark.parametrize(
    "limit",
    [
        0,
        -1,
    ],
)
def test_select_worst_snapshots_rejects_non_positive_limit(
    monkeypatch: pytest.MonkeyPatch,
    limit: int,
) -> None:
    """Snapshot diagnostic limits must be positive."""
    monkeypatch.setattr(
        diagnostics,
        "validate_snapshot_evaluation",
        lambda frame: None,
    )

    with pytest.raises(
        ModelDiagnosticsError,
        match="greater than zero",
    ):
        select_worst_snapshots(
            pd.DataFrame({"placeholder": [1]}),
            limit=limit,
        )


def test_select_worst_snapshots_rejects_boolean_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Booleans must not be accepted as integer limits."""
    monkeypatch.setattr(
        diagnostics,
        "validate_snapshot_evaluation",
        lambda frame: None,
    )

    with pytest.raises(
        TypeError,
        match="limit must be an integer",
    ):
        select_worst_snapshots(
            pd.DataFrame({"placeholder": [1]}),
            limit=True,
        )


def test_select_worst_snapshots_rejects_unknown_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown race-stage filters should fail explicitly."""
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

    with pytest.raises(
        ModelDiagnosticsError,
        match="Unknown race stage",
    ):
        select_worst_snapshots(
            pd.DataFrame({"placeholder": [1]}),
            race_stage="Qualifying",
        )


def test_select_worst_snapshots_rejects_empty_filter_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A filter matching no snapshots should fail clearly."""
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

    with pytest.raises(
        ModelDiagnosticsError,
        match="No snapshots matched",
    ):
        select_worst_snapshots(
            pd.DataFrame({"placeholder": [1]}),
            race_id="race_that_does_not_exist",
        )


def test_add_predicted_winner_details_adds_unique_favorite() -> None:
    """Selected snapshots should receive the model's top predicted driver."""
    snapshots = pd.DataFrame(
        {
            "RaceId": [
                "race_a",
                "race_a",
            ],
            "SnapshotLap": [
                10,
                20,
            ],
            "ActualWinner": [
                "VER",
                "VER",
            ],
        }
    )

    predictions = pd.DataFrame(
        {
            "RaceId": [
                "race_a",
                "race_a",
                "race_a",
                "race_a",
            ],
            "SnapshotLap": [
                10,
                10,
                20,
                20,
            ],
            "Driver": [
                "VER",
                "LEC",
                "VER",
                "LEC",
            ],
            "WinnerProbability": [
                0.30,
                0.70,
                0.80,
                0.20,
            ],
        }
    )

    enriched = add_predicted_winner_details(
        snapshots,
        predictions,
    )

    assert enriched["PredictedWinner"].tolist() == [
        "LEC",
        "VER",
    ]

    assert enriched["PredictedWinnerProbability"].tolist() == pytest.approx(
        [
            0.70,
            0.80,
        ]
    )


def test_add_predicted_winner_details_preserves_snapshot_columns() -> None:
    """Enrichment should preserve the original snapshot diagnostics."""
    snapshots = pd.DataFrame(
        {
            "RaceId": [
                "race_a",
            ],
            "SnapshotLap": [
                22,
            ],
            "ActualWinner": [
                "VER",
            ],
            "SnapshotLogLoss": [
                1.5,
            ],
        }
    )

    predictions = pd.DataFrame(
        {
            "RaceId": [
                "race_a",
                "race_a",
            ],
            "SnapshotLap": [
                22,
                22,
            ],
            "Driver": [
                "VER",
                "LEC",
            ],
            "WinnerProbability": [
                0.20,
                0.80,
            ],
        }
    )

    enriched = add_predicted_winner_details(
        snapshots,
        predictions,
    )

    assert enriched["ActualWinner"].tolist() == [
        "VER",
    ]

    assert enriched["SnapshotLogLoss"].tolist() == pytest.approx(
        [
            1.5,
        ]
    )


def test_add_predicted_winner_details_rejects_tied_favorite() -> None:
    """A snapshot must have one unique predicted winner."""
    snapshots = pd.DataFrame(
        {
            "RaceId": [
                "race_a",
            ],
            "SnapshotLap": [
                22,
            ],
        }
    )

    predictions = pd.DataFrame(
        {
            "RaceId": [
                "race_a",
                "race_a",
            ],
            "SnapshotLap": [
                22,
                22,
            ],
            "Driver": [
                "VER",
                "LEC",
            ],
            "WinnerProbability": [
                0.50,
                0.50,
            ],
        }
    )

    with pytest.raises(
        ModelDiagnosticsError,
        match="one unique predicted winner",
    ):
        add_predicted_winner_details(
            snapshots,
            predictions,
        )


def test_add_predicted_winner_details_rejects_unmatched_snapshots() -> None:
    """Selected snapshots must have matching prediction rows."""
    snapshots = pd.DataFrame(
        {
            "RaceId": [
                "race_a",
            ],
            "SnapshotLap": [
                22,
            ],
        }
    )

    predictions = pd.DataFrame(
        {
            "RaceId": [
                "race_b",
            ],
            "SnapshotLap": [
                22,
            ],
            "Driver": [
                "VER",
            ],
            "WinnerProbability": [
                1.0,
            ],
        }
    )

    with pytest.raises(
        ModelDiagnosticsError,
        match="No predictions matched",
    ):
        add_predicted_winner_details(
            snapshots,
            predictions,
        )


def test_add_predicted_winner_details_does_not_modify_inputs() -> None:
    """Prediction enrichment should leave source frames unchanged."""
    snapshots = pd.DataFrame(
        {
            "RaceId": [
                "race_a",
            ],
            "SnapshotLap": [
                22,
            ],
        }
    )

    predictions = pd.DataFrame(
        {
            "RaceId": [
                "race_a",
                "race_a",
            ],
            "SnapshotLap": [
                22,
                22,
            ],
            "Driver": [
                "VER",
                "LEC",
            ],
            "WinnerProbability": [
                0.20,
                0.80,
            ],
        }
    )

    original_snapshots = snapshots.copy(deep=True)
    original_predictions = predictions.copy(deep=True)

    add_predicted_winner_details(
        snapshots,
        predictions,
    )

    pd.testing.assert_frame_equal(
        snapshots,
        original_snapshots,
    )

    pd.testing.assert_frame_equal(
        predictions,
        original_predictions,
    )


def make_diagnostic_features() -> pd.DataFrame:
    """Build deterministic feature rows across races, laps and drivers."""
    features = pd.DataFrame(
        {
            "RaceId": [
                "race_a",
                "race_a",
                "race_a",
                "race_a",
                "race_a",
                "race_b",
            ],
            "SnapshotLap": [
                20,
                10,
                20,
                10,
                10,
                10,
            ],
            "Driver": [
                "VER",
                "PER",
                "PER",
                "VER",
                "LEC",
                "VER",
            ],
            "Position": [
                1,
                1,
                2,
                2,
                3,
                1,
            ],
        }
    )

    for column in MODEL_FEATURE_COLUMNS:
        if column not in features.columns:
            features[column] = 0

    return features


def test_select_driver_snapshot_features_filters_requested_rows() -> None:
    """Only requested race, lap and driver combinations should be returned."""
    features = make_diagnostic_features()

    selected = select_driver_snapshot_features(
        features,
        race_id="race_a",
        snapshot_laps=[10, 20],
        drivers=["VER", "PER"],
    )

    assert len(selected) == 4

    assert selected["RaceId"].tolist() == [
        "race_a",
        "race_a",
        "race_a",
        "race_a",
    ]

    assert selected["SnapshotLap"].tolist() == [
        10,
        10,
        20,
        20,
    ]

    assert selected["Driver"].tolist() == [
        "PER",
        "VER",
        "PER",
        "VER",
    ]

    assert selected["Position"].tolist() == [
        1,
        2,
        2,
        1,
    ]


def test_select_driver_snapshot_features_uses_expected_schema() -> None:
    """Diagnostics should return only identity and model feature columns."""
    features = make_diagnostic_features()
    features["WonRace"] = False

    selected = select_driver_snapshot_features(
        features,
        race_id="race_a",
        snapshot_laps=[10],
        drivers=["VER", "PER"],
    )

    expected_columns = list(
        dict.fromkeys(
            [
                "RaceId",
                "SnapshotLap",
                "Driver",
                *MODEL_FEATURE_COLUMNS,
            ]
        )
    )

    assert selected.columns.tolist() == expected_columns
    assert "WonRace" not in selected.columns


def test_select_driver_snapshot_features_rejects_missing_rows() -> None:
    """Every requested driver-lap combination must exist."""
    features = make_diagnostic_features()

    with pytest.raises(
        ModelDiagnosticsError,
        match="Some requested driver-snapshot feature rows are missing",
    ):
        select_driver_snapshot_features(
            features,
            race_id="race_a",
            snapshot_laps=[10, 20],
            drivers=["VER", "LEC"],
        )


def test_select_driver_snapshot_features_rejects_duplicate_rows() -> None:
    """Duplicate driver-snapshot identities must be rejected."""
    features = make_diagnostic_features()

    features = pd.concat(
        [
            features,
            features.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ModelDiagnosticsError,
        match="duplicate driver-snapshot rows",
    ):
        select_driver_snapshot_features(
            features,
            race_id="race_a",
            snapshot_laps=[20],
            drivers=["VER", "PER"],
        )


def test_select_driver_snapshot_features_rejects_empty_laps() -> None:
    """At least one snapshot lap must be requested."""
    with pytest.raises(
        ModelDiagnosticsError,
        match="At least one snapshot lap",
    ):
        select_driver_snapshot_features(
            make_diagnostic_features(),
            race_id="race_a",
            snapshot_laps=[],
            drivers=["VER"],
        )


def test_select_driver_snapshot_features_rejects_empty_drivers() -> None:
    """At least one driver must be requested."""
    with pytest.raises(
        ModelDiagnosticsError,
        match="At least one driver",
    ):
        select_driver_snapshot_features(
            make_diagnostic_features(),
            race_id="race_a",
            snapshot_laps=[10],
            drivers=[],
        )


def test_select_driver_snapshot_features_rejects_missing_columns() -> None:
    """Required model feature columns must be present."""
    features = make_diagnostic_features().drop(columns=["Position"])

    with pytest.raises(
        ModelDiagnosticsError,
        match="Features are missing required diagnostic columns",
    ):
        select_driver_snapshot_features(
            features,
            race_id="race_a",
            snapshot_laps=[10],
            drivers=["VER"],
        )


def test_select_driver_snapshot_features_rejects_unmatched_filters() -> None:
    """Filters matching no rows should fail explicitly."""
    with pytest.raises(
        ModelDiagnosticsError,
        match="No feature rows matched",
    ):
        select_driver_snapshot_features(
            make_diagnostic_features(),
            race_id="unknown_race",
            snapshot_laps=[10],
            drivers=["VER"],
        )


def test_select_driver_snapshot_features_does_not_modify_input() -> None:
    """Feature inspection must preserve the original feature frame."""
    features = make_diagnostic_features()
    original = features.copy(deep=True)

    selected = select_driver_snapshot_features(
        features,
        race_id="race_a",
        snapshot_laps=[10, 20],
        drivers=["VER", "PER"],
    )

    selected.loc[0, "Position"] = 99

    pd.testing.assert_frame_equal(
        features,
        original,
    )
