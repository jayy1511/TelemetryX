"""Tests for TelemetryX winner-probability model evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from telemetryx.data.dataset import build_race_dataset
from telemetryx.features.engineering import engineer_race_features
from telemetryx.modeling.baseline import (
    WINNER_PROBABILITY_COLUMN,
)
from telemetryx.modeling.evaluation import (
    ACTUAL_WINNER_COLUMN,
    ACTUAL_WINNER_PROBABILITY_COLUMN,
    ACTUAL_WINNER_RANK_COLUMN,
    SNAPSHOT_BRIER_SCORE_COLUMN,
    SNAPSHOT_EVALUATION_COLUMNS,
    SNAPSHOT_LOG_LOSS_COLUMN,
    TOP_ONE_CORRECT_COLUMN,
    EvaluationSummary,
    ModelEvaluationError,
    WinnerModelEvaluation,
    evaluate_winner_probabilities,
    validate_snapshot_evaluation,
)


def make_cleaned_laps() -> pd.DataFrame:
    """Return a deterministic three-driver, three-lap race."""
    return pd.DataFrame(
        {
            "Driver": [
                "VER",
                "VER",
                "VER",
                "NOR",
                "NOR",
                "NOR",
                "BOT",
                "BOT",
                "BOT",
            ],
            "LapNumber": [
                1,
                2,
                3,
                1,
                2,
                3,
                1,
                2,
                3,
            ],
            "Position": [
                1,
                1,
                1,
                2,
                2,
                2,
                3,
                3,
                3,
            ],
            "Stint": [
                1,
                1,
                2,
                1,
                1,
                2,
                1,
                1,
                2,
            ],
            "Compound": [
                "SOFT",
                "SOFT",
                "MEDIUM",
                "SOFT",
                "SOFT",
                "MEDIUM",
                "SOFT",
                "SOFT",
                "MEDIUM",
            ],
            "TyreLife": [
                1.0,
                2.0,
                1.0,
                1.0,
                2.0,
                1.0,
                1.0,
                2.0,
                1.0,
            ],
            "TrackStatus": [
                "1",
                "1",
                "1",
                "1",
                "1",
                "1",
                "1",
                "1",
                "1",
            ],
            "LapTimeSeconds": [
                90.0,
                89.0,
                88.0,
                91.0,
                90.0,
                89.0,
                92.0,
                91.0,
                90.0,
            ],
        }
    )


def make_results() -> pd.DataFrame:
    """Return final results with VER as the race winner."""
    return pd.DataFrame(
        {
            "Abbreviation": [
                "VER",
                "NOR",
                "BOT",
            ],
            "Position": [
                1,
                2,
                3,
            ],
            "Status": [
                "Finished",
                "Finished",
                "Finished",
            ],
        }
    )


def make_features(
    *,
    season: int = 2023,
    round_number: int = 1,
    event_name: str = "Bahrain Grand Prix",
) -> pd.DataFrame:
    """Return a valid engineered feature frame."""
    race_dataset = build_race_dataset(
        make_cleaned_laps(),
        make_results(),
        season=season,
        round_number=round_number,
        event_name=event_name,
        session_name="Race",
    )

    return engineer_race_features(race_dataset)


def make_predictions(
    features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Return hand-authored probabilities for three race snapshots.

    Snapshot 1:
        VER 0.60, NOR 0.30, BOT 0.10
        winner rank = 1

    Snapshot 2:
        VER 0.20, NOR 0.70, BOT 0.10
        winner rank = 2

    Snapshot 3:
        VER 0.40, NOR 0.40, BOT 0.20
        VER and NOR tie for rank 1
    """
    active_features = make_features() if features is None else features

    probability_by_snapshot = {
        1: {
            "VER": 0.60,
            "NOR": 0.30,
            "BOT": 0.10,
        },
        2: {
            "VER": 0.20,
            "NOR": 0.70,
            "BOT": 0.10,
        },
        3: {
            "VER": 0.40,
            "NOR": 0.40,
            "BOT": 0.20,
        },
    }

    predictions = active_features.loc[
        :,
        [
            "RaceId",
            "SnapshotLap",
            "Driver",
        ],
    ].copy(deep=True)

    predictions[WINNER_PROBABILITY_COLUMN] = [
        probability_by_snapshot[int(snapshot_lap)][str(driver)]
        for snapshot_lap, driver in zip(
            predictions["SnapshotLap"],
            predictions["Driver"],
            strict=True,
        )
    ]

    return predictions


def make_evaluation() -> WinnerModelEvaluation:
    """Return evaluation for the hand-authored probability fixture."""
    features = make_features()

    return evaluate_winner_probabilities(
        features,
        make_predictions(features),
    )


def test_evaluate_returns_complete_result() -> None:
    """Evaluation should return both summary and snapshot diagnostics."""
    evaluation = make_evaluation()

    assert isinstance(
        evaluation,
        WinnerModelEvaluation,
    )

    assert isinstance(
        evaluation.summary,
        EvaluationSummary,
    )

    assert isinstance(
        evaluation.snapshots,
        pd.DataFrame,
    )


def test_snapshot_evaluation_uses_required_schema() -> None:
    """Per-snapshot output should follow the public evaluation contract."""
    snapshots = make_evaluation().snapshots

    assert tuple(snapshots.columns) == SNAPSHOT_EVALUATION_COLUMNS


def test_evaluation_contains_one_row_per_snapshot() -> None:
    """Driver rows should collapse into one metric row per race snapshot."""
    evaluation = make_evaluation()

    assert len(evaluation.snapshots) == 3

    assert evaluation.summary.snapshot_count == 3
    assert evaluation.summary.row_count == 9
    assert evaluation.summary.race_count == 1


def test_actual_winner_is_recorded_per_snapshot() -> None:
    """Every snapshot should identify the true eventual winner."""
    snapshots = make_evaluation().snapshots

    assert set(snapshots[ACTUAL_WINNER_COLUMN]) == {
        "VER",
    }


def test_actual_winner_probabilities_match_fixture() -> None:
    """Evaluation should extract the probability assigned to VER."""
    snapshots = make_evaluation().snapshots

    np.testing.assert_allclose(
        snapshots[ACTUAL_WINNER_PROBABILITY_COLUMN].to_numpy(dtype=float),
        np.array(
            [
                0.60,
                0.20,
                0.40,
            ]
        ),
        rtol=1e-12,
        atol=1e-12,
    )


def test_snapshot_log_loss_matches_hand_calculation() -> None:
    """Snapshot log loss should equal negative log winner probability."""
    snapshots = make_evaluation().snapshots

    expected = np.array(
        [
            -np.log(0.60),
            -np.log(0.20),
            -np.log(0.40),
        ]
    )

    np.testing.assert_allclose(
        snapshots[SNAPSHOT_LOG_LOSS_COLUMN].to_numpy(dtype=float),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_snapshot_brier_score_matches_hand_calculation() -> None:
    """Brier score should evaluate the full snapshot distribution."""
    snapshots = make_evaluation().snapshots

    expected = np.array(
        [
            ((0.60 - 1.0) ** 2 + (0.30 - 0.0) ** 2 + (0.10 - 0.0) ** 2),
            ((0.20 - 1.0) ** 2 + (0.70 - 0.0) ** 2 + (0.10 - 0.0) ** 2),
            ((0.40 - 1.0) ** 2 + (0.40 - 0.0) ** 2 + (0.20 - 0.0) ** 2),
        ]
    )

    np.testing.assert_allclose(
        snapshots[SNAPSHOT_BRIER_SCORE_COLUMN].to_numpy(dtype=float),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        expected,
        np.array(
            [
                0.26,
                1.14,
                0.56,
            ]
        ),
        rtol=1e-12,
        atol=1e-12,
    )


def test_actual_winner_rank_matches_probability_order() -> None:
    """The true winner's rank should follow descending probability."""
    snapshots = make_evaluation().snapshots

    assert snapshots[ACTUAL_WINNER_RANK_COLUMN].tolist() == [
        1,
        2,
        1,
    ]


def test_probability_tie_receives_rank_one() -> None:
    """A winner tied for highest probability should receive rank one."""
    snapshots = make_evaluation().snapshots

    lap_three = snapshots.loc[snapshots["SnapshotLap"].eq(3)].iloc[0]

    assert lap_three[ACTUAL_WINNER_RANK_COLUMN] == 1

    assert bool(lap_three[TOP_ONE_CORRECT_COLUMN]) is True


def test_top_one_correct_matches_winner_rank() -> None:
    """Top-one correctness should be true exactly when winner rank is one."""
    snapshots = make_evaluation().snapshots

    assert snapshots[TOP_ONE_CORRECT_COLUMN].tolist() == [
        True,
        False,
        True,
    ]


def test_summary_mean_log_loss_matches_snapshot_mean() -> None:
    """Aggregate log loss should be the arithmetic mean across snapshots."""
    evaluation = make_evaluation()

    expected = float(
        np.mean(
            [
                -np.log(0.60),
                -np.log(0.20),
                -np.log(0.40),
            ]
        )
    )

    assert evaluation.summary.mean_log_loss == pytest.approx(expected)


def test_summary_mean_brier_score_matches_snapshot_mean() -> None:
    """Aggregate Brier score should average snapshot-level scores."""
    evaluation = make_evaluation()

    expected = float(
        np.mean(
            [
                0.26,
                1.14,
                0.56,
            ]
        )
    )

    assert evaluation.summary.mean_brier_score == pytest.approx(expected)


def test_summary_top_one_accuracy_matches_fixture() -> None:
    """Two of three snapshots rank the eventual winner first."""
    evaluation = make_evaluation()

    assert evaluation.summary.top_one_accuracy == pytest.approx(2.0 / 3.0)


def test_summary_mean_winner_probability_matches_fixture() -> None:
    """Aggregate winner confidence should average winner probabilities."""
    evaluation = make_evaluation()

    assert evaluation.summary.mean_actual_winner_probability == pytest.approx(
        (0.60 + 0.20 + 0.40) / 3.0
    )


def test_prediction_row_order_does_not_affect_metrics() -> None:
    """Evaluation alignment must use keys rather than DataFrame row order."""
    features = make_features()

    predictions = (
        make_predictions(features)
        .sample(
            frac=1.0,
            random_state=42,
        )
        .reset_index(drop=True)
    )

    evaluation = evaluate_winner_probabilities(
        features,
        predictions,
    )

    expected = make_evaluation()

    pd.testing.assert_frame_equal(
        evaluation.snapshots,
        expected.snapshots,
    )

    assert evaluation.summary == (expected.summary)


def test_evaluation_does_not_modify_inputs() -> None:
    """Evaluation must leave source features and predictions unchanged."""
    features = make_features()
    predictions = make_predictions(features)

    original_features = features.copy(deep=True)

    original_predictions = predictions.copy(deep=True)

    evaluate_winner_probabilities(
        features,
        predictions,
    )

    pd.testing.assert_frame_equal(
        features,
        original_features,
    )

    pd.testing.assert_frame_equal(
        predictions,
        original_predictions,
    )


def test_evaluation_rejects_non_dataframe_features() -> None:
    """Evaluation features must be provided as a DataFrame."""
    invalid_features: Any = []

    with pytest.raises(
        TypeError,
        match=("features must be provided as a pandas DataFrame"),
    ):
        evaluate_winner_probabilities(
            invalid_features,
            make_predictions(),
        )


def test_evaluation_rejects_non_dataframe_predictions() -> None:
    """Evaluation predictions must be provided as a DataFrame."""
    invalid_predictions: Any = []

    with pytest.raises(
        TypeError,
        match=("predictions must be provided as a pandas DataFrame"),
    ):
        evaluate_winner_probabilities(
            make_features(),
            invalid_predictions,
        )


def test_evaluation_wraps_invalid_feature_frame() -> None:
    """Malformed engineered features must fail before metric calculation."""
    features = make_features().drop(
        columns=[
            "PositionFraction",
        ]
    )

    predictions = make_predictions()

    with pytest.raises(
        ModelEvaluationError,
        match=("feature frame failed validation before evaluation"),
    ):
        evaluate_winner_probabilities(
            features,
            predictions,
        )


def test_evaluation_wraps_invalid_prediction_frame() -> None:
    """Malformed probabilities must fail before target alignment."""
    features = make_features()

    predictions = make_predictions(features)

    predictions.loc[
        predictions["SnapshotLap"].eq(1),
        WINNER_PROBABILITY_COLUMN,
    ] = 0.1

    with pytest.raises(
        ModelEvaluationError,
        match=("prediction frame failed validation before evaluation"),
    ):
        evaluate_winner_probabilities(
            features,
            predictions,
        )


def test_evaluation_rejects_missing_prediction_row() -> None:
    """Every supervised feature row must have one prediction."""
    features = make_features()

    predictions = make_predictions(features)

    missing_mask = predictions["SnapshotLap"].eq(1) & predictions["Driver"].eq("VER")

    predictions = predictions.loc[~missing_mask].reset_index(drop=True)

    snapshot_mask = predictions["SnapshotLap"].eq(1)

    snapshot_total = float(
        predictions.loc[
            snapshot_mask,
            WINNER_PROBABILITY_COLUMN,
        ].sum()
    )

    predictions.loc[
        snapshot_mask,
        WINNER_PROBABILITY_COLUMN,
    ] = (
        predictions.loc[
            snapshot_mask,
            WINNER_PROBABILITY_COLUMN,
        ]
        / snapshot_total
    )

    with pytest.raises(
        ModelEvaluationError,
        match=("Feature and prediction keys do not match exactly"),
    ):
        evaluate_winner_probabilities(
            features,
            predictions,
        )


def test_evaluation_rejects_unexpected_prediction_row() -> None:
    """Predictions cannot contain observations absent from the features."""
    features = make_features()

    predictions = make_predictions(features)

    extra = predictions.iloc[[0]].copy(deep=True)

    extra["Driver"] = "PIA"

    extra[WINNER_PROBABILITY_COLUMN] = 0.0

    malformed = pd.concat(
        [
            predictions,
            extra,
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ModelEvaluationError,
        match=("Feature and prediction keys do not match exactly"),
    ):
        evaluate_winner_probabilities(
            features,
            malformed,
        )


def test_evaluation_rejects_prediction_for_wrong_race() -> None:
    """Race identity must match exactly between features and predictions."""
    features = make_features()

    predictions = make_predictions(features)

    snapshot_mask = predictions["SnapshotLap"].eq(1)

    predictions.loc[
        snapshot_mask,
        "RaceId",
    ] = "2099_01_future_grand_prix"

    with pytest.raises(
        ModelEvaluationError,
        match=("Feature and prediction keys do not match exactly"),
    ):
        evaluate_winner_probabilities(
            features,
            predictions,
        )


def test_validate_snapshot_evaluation_accepts_valid_frame() -> None:
    """Generated snapshot diagnostics should pass standalone validation."""
    validate_snapshot_evaluation(make_evaluation().snapshots)


def test_validate_snapshot_evaluation_rejects_non_dataframe() -> None:
    """Snapshot validation requires a pandas DataFrame."""
    invalid_snapshots: Any = []

    with pytest.raises(
        TypeError,
        match=("snapshots must be provided as a pandas DataFrame"),
    ):
        validate_snapshot_evaluation(invalid_snapshots)


def test_validate_snapshot_evaluation_rejects_empty_frame() -> None:
    """An empty evaluation artifact is invalid."""
    snapshots = make_evaluation().snapshots.iloc[0:0].copy()

    with pytest.raises(
        ModelEvaluationError,
        match="contains no rows",
    ):
        validate_snapshot_evaluation(snapshots)


def test_validate_snapshot_evaluation_rejects_wrong_schema() -> None:
    """Snapshot diagnostics must use the exact public schema."""
    snapshots = make_evaluation().snapshots

    snapshots["ExtraColumn"] = 1

    with pytest.raises(
        ModelEvaluationError,
        match=("does not match the required evaluation schema"),
    ):
        validate_snapshot_evaluation(snapshots)


def test_validate_snapshot_evaluation_rejects_duplicate_snapshot() -> None:
    """Each race snapshot may appear only once in evaluation output."""
    snapshots = make_evaluation().snapshots

    duplicate = snapshots.iloc[[0]].copy(deep=True)

    malformed = pd.concat(
        [
            snapshots,
            duplicate,
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ModelEvaluationError,
        match="duplicate race snapshots",
    ):
        validate_snapshot_evaluation(malformed)


def test_validate_snapshot_evaluation_rejects_missing_identity() -> None:
    """Snapshot identity fields cannot be missing."""
    snapshots = make_evaluation().snapshots

    snapshots.loc[
        0,
        ACTUAL_WINNER_COLUMN,
    ] = pd.NA

    with pytest.raises(
        ModelEvaluationError,
        match=("identity columns cannot contain missing values"),
    ):
        validate_snapshot_evaluation(snapshots)


def test_validate_snapshot_evaluation_rejects_missing_winner_probability() -> None:
    """The true winner must have an evaluable probability."""
    snapshots = make_evaluation().snapshots

    snapshots.loc[
        0,
        ACTUAL_WINNER_PROBABILITY_COLUMN,
    ] = np.nan

    with pytest.raises(
        ModelEvaluationError,
        match=("Actual winner probabilities cannot contain missing values"),
    ):
        validate_snapshot_evaluation(snapshots)


@pytest.mark.parametrize(
    "invalid_probability",
    [
        -0.1,
        1.1,
    ],
)
def test_validate_snapshot_evaluation_rejects_winner_probability_outside_range(
    invalid_probability: float,
) -> None:
    """True winner probability must remain inside [0, 1]."""
    snapshots = make_evaluation().snapshots

    snapshots.loc[
        0,
        ACTUAL_WINNER_PROBABILITY_COLUMN,
    ] = invalid_probability

    with pytest.raises(
        ModelEvaluationError,
        match=("Actual winner probabilities must remain between 0 and 1"),
    ):
        validate_snapshot_evaluation(snapshots)


def test_validate_snapshot_evaluation_rejects_invalid_rank() -> None:
    """Winner rank cannot be zero or negative."""
    snapshots = make_evaluation().snapshots

    snapshots.loc[
        0,
        ACTUAL_WINNER_RANK_COLUMN,
    ] = 0

    with pytest.raises(
        ModelEvaluationError,
        match=("Actual winner ranks must be at least one"),
    ):
        validate_snapshot_evaluation(snapshots)


@pytest.mark.parametrize(
    "column",
    [
        SNAPSHOT_LOG_LOSS_COLUMN,
        SNAPSHOT_BRIER_SCORE_COLUMN,
    ],
)
def test_validate_snapshot_evaluation_rejects_negative_metric(
    column: str,
) -> None:
    """Probability error metrics cannot be negative."""
    snapshots = make_evaluation().snapshots

    snapshots.loc[
        0,
        column,
    ] = -0.1

    with pytest.raises(
        ModelEvaluationError,
        match=(rf"{column} cannot contain negative values"),
    ):
        validate_snapshot_evaluation(snapshots)


def test_validate_snapshot_evaluation_requires_boolean_top_one() -> None:
    """TopOneCorrect should remain a Boolean diagnostic."""
    snapshots = make_evaluation().snapshots

    snapshots[TOP_ONE_CORRECT_COLUMN] = snapshots[TOP_ONE_CORRECT_COLUMN].astype(
        "Int64"
    )

    with pytest.raises(
        ModelEvaluationError,
        match=("TopOneCorrect must use a Boolean dtype"),
    ):
        validate_snapshot_evaluation(snapshots)
