from __future__ import annotations

import pandas as pd
import pytest

from telemetryx.modeling.benchmarks import (
    BenchmarkPredictionError,
    predict_driver_prior_winner_probabilities,
    predict_uniform_winner_probabilities,
)


def make_prediction_features() -> pd.DataFrame:
    """Build prediction snapshots with different field sizes."""
    return pd.DataFrame(
        {
            "RaceId": [
                "validation_race",
                "validation_race",
                "validation_race",
                "validation_race",
                "validation_race",
            ],
            "SnapshotLap": [
                1,
                1,
                1,
                2,
                2,
            ],
            "Driver": [
                "VER",
                "NOR",
                "LEC",
                "VER",
                "NOR",
            ],
        }
    )


def make_training_features() -> pd.DataFrame:
    """Build repeated snapshot rows for two historical races."""
    return pd.DataFrame(
        {
            "RaceId": [
                "race_1",
                "race_1",
                "race_1",
                "race_1",
                "race_2",
                "race_2",
                "race_2",
                "race_2",
            ],
            "SnapshotLap": [
                1,
                1,
                2,
                2,
                1,
                1,
                2,
                2,
            ],
            "Driver": [
                "VER",
                "NOR",
                "VER",
                "NOR",
                "VER",
                "NOR",
                "VER",
                "NOR",
            ],
            "WonRace": pd.Series(
                [
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                ],
                dtype="boolean",
            ),
        }
    )


def test_uniform_predictions_are_equal_within_snapshot() -> None:
    """Uniform benchmark should divide probability equally among drivers."""
    features = make_prediction_features()

    predictions = predict_uniform_winner_probabilities(features)

    lap_one = predictions.loc[predictions["SnapshotLap"].eq(1)]

    lap_two = predictions.loc[predictions["SnapshotLap"].eq(2)]

    assert lap_one["WinnerProbability"].tolist() == pytest.approx(
        [
            1 / 3,
            1 / 3,
            1 / 3,
        ]
    )

    assert lap_two["WinnerProbability"].tolist() == pytest.approx(
        [
            0.5,
            0.5,
        ]
    )


def test_uniform_predictions_sum_to_one_per_snapshot() -> None:
    """Every uniform snapshot should contain one unit of probability."""
    predictions = predict_uniform_winner_probabilities(make_prediction_features())

    totals = predictions.groupby(
        [
            "RaceId",
            "SnapshotLap",
        ]
    )["WinnerProbability"].sum()

    assert totals.tolist() == pytest.approx(
        [
            1.0,
            1.0,
        ]
    )


def test_uniform_prediction_preserves_identity_columns() -> None:
    """Benchmark output should retain the row identities used by evaluation."""
    features = make_prediction_features()

    predictions = predict_uniform_winner_probabilities(features)

    assert predictions.columns.tolist() == [
        "RaceId",
        "SnapshotLap",
        "Driver",
        "WinnerProbability",
    ]

    pd.testing.assert_frame_equal(
        predictions[
            [
                "RaceId",
                "SnapshotLap",
                "Driver",
            ]
        ],
        features[
            [
                "RaceId",
                "SnapshotLap",
                "Driver",
            ]
        ],
    )


def test_uniform_prediction_does_not_modify_input() -> None:
    """Benchmark generation must not mutate prediction features."""
    features = make_prediction_features()

    original = features.copy(deep=True)

    predict_uniform_winner_probabilities(features)

    pd.testing.assert_frame_equal(
        features,
        original,
    )


def test_driver_prior_uses_training_race_win_frequency() -> None:
    """Historical prior should reflect driver results from training races."""
    training = make_training_features()

    prediction = pd.DataFrame(
        {
            "RaceId": [
                "validation_race",
                "validation_race",
            ],
            "SnapshotLap": [
                1,
                1,
            ],
            "Driver": [
                "VER",
                "NOR",
            ],
        }
    )

    probabilities = predict_driver_prior_winner_probabilities(
        training,
        prediction,
    )

    assert probabilities["WinnerProbability"].tolist() == pytest.approx(
        [
            0.75,
            0.25,
        ]
    )


def test_repeated_training_snapshots_do_not_change_driver_prior() -> None:
    """Each historical race should count once regardless of snapshot count."""
    training = make_training_features()

    prediction = pd.DataFrame(
        {
            "RaceId": [
                "validation_race",
                "validation_race",
            ],
            "SnapshotLap": [
                1,
                1,
            ],
            "Driver": [
                "VER",
                "NOR",
            ],
        }
    )

    original_predictions = predict_driver_prior_winner_probabilities(
        training,
        prediction,
    )

    duplicated_training = pd.concat(
        [
            training,
            training,
            training,
        ],
        ignore_index=True,
    )

    duplicated_predictions = predict_driver_prior_winner_probabilities(
        duplicated_training,
        prediction,
    )

    pd.testing.assert_frame_equal(
        original_predictions,
        duplicated_predictions,
    )


def test_driver_prior_handles_unseen_driver_with_smoothing() -> None:
    """An unseen validation driver should still receive non-zero probability."""
    training = make_training_features()

    prediction = pd.DataFrame(
        {
            "RaceId": [
                "validation_race",
                "validation_race",
                "validation_race",
            ],
            "SnapshotLap": [
                1,
                1,
                1,
            ],
            "Driver": [
                "VER",
                "NOR",
                "PIA",
            ],
        }
    )

    probabilities = predict_driver_prior_winner_probabilities(
        training,
        prediction,
    )

    assert probabilities["WinnerProbability"].tolist() == pytest.approx(
        [
            0.60,
            0.20,
            0.20,
        ]
    )


def test_driver_prior_probabilities_sum_to_one_per_snapshot() -> None:
    """Historical priors should be normalized within every snapshot."""
    training = make_training_features()

    predictions = predict_driver_prior_winner_probabilities(
        training,
        make_prediction_features(),
    )

    totals = predictions.groupby(
        [
            "RaceId",
            "SnapshotLap",
        ]
    )["WinnerProbability"].sum()

    assert totals.tolist() == pytest.approx(
        [
            1.0,
            1.0,
        ]
    )


def test_driver_prior_does_not_modify_inputs() -> None:
    """Historical benchmark generation must preserve both input frames."""
    training = make_training_features()
    prediction = make_prediction_features()

    original_training = training.copy(deep=True)

    original_prediction = prediction.copy(deep=True)

    predict_driver_prior_winner_probabilities(
        training,
        prediction,
    )

    pd.testing.assert_frame_equal(
        training,
        original_training,
    )

    pd.testing.assert_frame_equal(
        prediction,
        original_prediction,
    )


@pytest.mark.parametrize(
    "smoothing",
    [
        0.0,
        -1.0,
    ],
)
def test_driver_prior_rejects_non_positive_smoothing(
    smoothing: float,
) -> None:
    """Additive smoothing must have positive probability mass."""
    with pytest.raises(
        BenchmarkPredictionError,
        match="smoothing must be greater than zero",
    ):
        predict_driver_prior_winner_probabilities(
            make_training_features(),
            make_prediction_features(),
            smoothing=smoothing,
        )


def test_driver_prior_rejects_non_numeric_smoothing() -> None:
    """Smoothing must be numeric rather than arbitrary input."""
    with pytest.raises(
        TypeError,
        match="smoothing must be a positive number",
    ):
        predict_driver_prior_winner_probabilities(
            make_training_features(),
            make_prediction_features(),
            smoothing="invalid",  # type: ignore[arg-type]
        )


def test_driver_prior_rejects_non_boolean_target() -> None:
    """Historical winner labels must retain boolean target semantics."""
    training = make_training_features()

    training["WonRace"] = training["WonRace"].astype("Int64")

    with pytest.raises(
        BenchmarkPredictionError,
        match="WonRace must use a boolean dtype",
    ):
        predict_driver_prior_winner_probabilities(
            training,
            make_prediction_features(),
        )


def test_driver_prior_rejects_inconsistent_race_driver_target() -> None:
    """One driver's race target cannot change between snapshots."""
    training = make_training_features()

    training.loc[
        (
            training["RaceId"].eq("race_1")
            & training["Driver"].eq("VER")
            & training["SnapshotLap"].eq(2)
        ),
        "WonRace",
    ] = False

    with pytest.raises(
        BenchmarkPredictionError,
        match="one stable winner target",
    ):
        predict_driver_prior_winner_probabilities(
            training,
            make_prediction_features(),
        )


def test_driver_prior_requires_exactly_one_winner_per_training_race() -> None:
    """Every historical training race must identify one winner."""
    training = make_training_features()

    training.loc[
        training["RaceId"].eq("race_1"),
        "WonRace",
    ] = False

    with pytest.raises(
        BenchmarkPredictionError,
        match="exactly one winner",
    ):
        predict_driver_prior_winner_probabilities(
            training,
            make_prediction_features(),
        )


def test_uniform_rejects_duplicate_prediction_identity() -> None:
    """A driver can appear only once in a race snapshot."""
    features = make_prediction_features()

    duplicated = pd.concat(
        [
            features,
            features.iloc[
                [
                    0,
                ]
            ],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        BenchmarkPredictionError,
        match="duplicate",
    ):
        predict_uniform_winner_probabilities(duplicated)


def test_uniform_rejects_missing_required_column() -> None:
    """Benchmark prediction requires race, lap and driver identity."""
    features = make_prediction_features().drop(
        columns=[
            "Driver",
        ]
    )

    with pytest.raises(
        BenchmarkPredictionError,
        match="missing required columns",
    ):
        predict_uniform_winner_probabilities(features)
