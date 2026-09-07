from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from telemetryx.data.dataset import build_race_dataset
from telemetryx.data.targets import TARGET_COLUMN
from telemetryx.features.engineering import engineer_race_features
from telemetryx.modeling.baseline import (
    PREDICTION_COLUMNS,
    WINNER_PROBABILITY_COLUMN,
    BaselineModelError,
    BaselineWinnerModel,
    predict_winner_probabilities,
    train_baseline_winner_model,
    validate_winner_probability_frame,
)


def make_cleaned_laps(
    *,
    drivers: tuple[str, str, str] = (
        "VER",
        "NOR",
        "BOT",
    ),
    lap_time_offset: float = 0.0,
) -> pd.DataFrame:
    """Return a small deterministic three-driver cleaned race."""
    rows: list[dict[str, object]] = []

    for position, driver in enumerate(
        drivers,
        start=1,
    ):
        base_lap_time = 89.0 + position + lap_time_offset

        for lap_number in (
            1,
            2,
            3,
        ):
            rows.append(
                {
                    "Driver": driver,
                    "LapNumber": lap_number,
                    "Position": position,
                    "Stint": (1 if lap_number < 3 else 2),
                    "Compound": ("SOFT" if lap_number < 3 else "MEDIUM"),
                    "TyreLife": (float(lap_number) if lap_number < 3 else 1.0),
                    "TrackStatus": "1",
                    "LapTimeSeconds": (base_lap_time - (lap_number - 1)),
                }
            )

    return pd.DataFrame(rows)


def make_results(
    *,
    drivers: tuple[str, str, str] = (
        "VER",
        "NOR",
        "BOT",
    ),
) -> pd.DataFrame:
    """Return final results with the first supplied driver winning."""
    return pd.DataFrame(
        {
            "Abbreviation": list(drivers),
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
    drivers: tuple[str, str, str] = (
        "VER",
        "NOR",
        "BOT",
    ),
    lap_time_offset: float = 0.0,
) -> pd.DataFrame:
    """Return a valid engineered synthetic race."""
    race_dataset = build_race_dataset(
        make_cleaned_laps(
            drivers=drivers,
            lap_time_offset=lap_time_offset,
        ),
        make_results(
            drivers=drivers,
        ),
        season=season,
        round_number=round_number,
        event_name=event_name,
        session_name="Race",
    )

    return engineer_race_features(race_dataset)


def make_training_features() -> pd.DataFrame:
    """Return multiple races suitable for baseline training."""
    first = make_features(
        season=2023,
        round_number=1,
        event_name="Bahrain Grand Prix",
        drivers=(
            "VER",
            "NOR",
            "BOT",
        ),
        lap_time_offset=0.0,
    )

    second = make_features(
        season=2023,
        round_number=2,
        event_name="Saudi Arabian Grand Prix",
        drivers=(
            "NOR",
            "VER",
            "BOT",
        ),
        lap_time_offset=5.0,
    )

    third = make_features(
        season=2023,
        round_number=3,
        event_name="Australian Grand Prix",
        drivers=(
            "BOT",
            "VER",
            "NOR",
        ),
        lap_time_offset=10.0,
    )

    return pd.concat(
        [
            first,
            second,
            third,
        ],
        ignore_index=True,
    )


def make_validation_features() -> pd.DataFrame:
    """Return future features containing one unseen training driver."""
    return make_features(
        season=2024,
        round_number=1,
        event_name="Bahrain Grand Prix",
        drivers=(
            "PIA",
            "NOR",
            "BOT",
        ),
        lap_time_offset=20.0,
    )


def make_predictions() -> pd.DataFrame:
    """Return valid baseline predictions for one future race."""
    model = train_baseline_winner_model(make_training_features())

    return predict_winner_probabilities(
        model,
        make_validation_features(),
    )


def test_train_baseline_returns_fitted_model() -> None:
    """Training should return reusable preprocessing and classifier state."""
    model = train_baseline_winner_model(make_training_features())

    assert isinstance(
        model,
        BaselineWinnerModel,
    )

    assert isinstance(
        model.estimator,
        LogisticRegression,
    )

    assert model.transformed_feature_count > 0

    assert model.transformed_feature_count == len(model.transformed_feature_names)

    assert len(model.model_columns) > 0


def test_fitted_estimator_contains_binary_classes() -> None:
    """The baseline classifier must learn winner and non-winner classes."""
    model = train_baseline_winner_model(make_training_features())

    assert set(int(value) for value in model.estimator.classes_) == {
        0,
        1,
    }

    assert int(model.estimator.classes_[model.positive_class_index]) == 1


def test_training_accepts_safe_model_feature_subset() -> None:
    """Experiments may train using a smaller audited input set."""
    model = train_baseline_winner_model(
        make_training_features(),
        model_columns=(
            "Position",
            "TyreLife",
            "IsLeader",
        ),
    )

    assert model.model_columns == (
        "Position",
        "TyreLife",
        "IsLeader",
    )


def test_training_rejects_non_dataframe() -> None:
    """Baseline training requires an engineered DataFrame."""
    invalid_features: Any = []

    with pytest.raises(
        TypeError,
        match=("training_features must be provided as a pandas DataFrame"),
    ):
        train_baseline_winner_model(invalid_features)


@pytest.mark.parametrize(
    "max_iterations",
    [
        0,
        -1,
        True,
    ],
)
def test_training_rejects_invalid_max_iterations(
    max_iterations: int,
) -> None:
    """Optimizer iteration limits must be positive integers."""
    with pytest.raises(
        BaselineModelError,
        match="max_iterations must be a positive integer",
    ):
        train_baseline_winner_model(
            make_training_features(),
            max_iterations=max_iterations,
        )


def test_training_rejects_target_leakage() -> None:
    """WonRace cannot be deliberately inserted into model inputs."""
    with pytest.raises(
        BaselineModelError,
        match=("Training data failed validation before baseline model fitting"),
    ):
        train_baseline_winner_model(
            make_training_features(),
            model_columns=(
                "Position",
                TARGET_COLUMN,
            ),
        )


def test_training_rejects_no_positive_target() -> None:
    """Training requires at least one winner observation."""
    features = make_training_features()

    features[TARGET_COLUMN] = pd.Series(
        False,
        index=features.index,
        dtype="boolean",
    )

    with pytest.raises(
        BaselineModelError,
        match="contains no winner observations",
    ):
        train_baseline_winner_model(features)


def test_training_rejects_no_negative_target() -> None:
    """Training requires non-winner observations as well as winners."""
    features = make_training_features()

    features[TARGET_COLUMN] = pd.Series(
        True,
        index=features.index,
        dtype="boolean",
    )

    with pytest.raises(
        BaselineModelError,
        match="contains no non-winner observations",
    ):
        train_baseline_winner_model(features)


def test_predict_returns_required_schema() -> None:
    """Prediction output should contain keys plus winner probability."""
    predictions = make_predictions()

    assert tuple(predictions.columns) == PREDICTION_COLUMNS

    assert tuple(predictions.columns) == (
        "RaceId",
        "SnapshotLap",
        "Driver",
        WINNER_PROBABILITY_COLUMN,
    )


def test_prediction_row_count_matches_feature_rows() -> None:
    """Every scored driver snapshot should receive one probability."""
    model = train_baseline_winner_model(make_training_features())

    features = make_validation_features()

    predictions = predict_winner_probabilities(
        model,
        features,
    )

    assert len(predictions) == len(features)


def test_probabilities_remain_between_zero_and_one() -> None:
    """Every normalized winner probability must be valid."""
    predictions = make_predictions()

    probabilities = predictions[WINNER_PROBABILITY_COLUMN]

    assert bool(probabilities.ge(0.0).all())

    assert bool(probabilities.le(1.0).all())


def test_probabilities_are_finite() -> None:
    """Prediction output cannot contain NaN or infinity."""
    predictions = make_predictions()

    values = predictions[WINNER_PROBABILITY_COLUMN].to_numpy(dtype=float)

    assert bool(np.isfinite(values).all())


def test_probabilities_sum_to_one_per_snapshot() -> None:
    """Each race snapshot must form one coherent winner distribution."""
    predictions = make_predictions()

    sums = predictions.groupby(
        [
            "RaceId",
            "SnapshotLap",
        ]
    )[WINNER_PROBABILITY_COLUMN].sum()

    np.testing.assert_allclose(
        sums.to_numpy(dtype=float),
        np.ones(
            len(sums),
            dtype=float,
        ),
        rtol=1e-9,
        atol=1e-9,
    )


def test_each_driver_receives_one_probability_per_snapshot() -> None:
    """Prediction keys must remain unique."""
    predictions = make_predictions()

    duplicates = predictions.duplicated(
        subset=[
            "RaceId",
            "SnapshotLap",
            "Driver",
        ],
        keep=False,
    )

    assert bool(duplicates.any()) is False


def test_prediction_supports_unseen_driver() -> None:
    """Future drivers absent from training should still be scoreable."""
    training_features = make_training_features()

    validation_features = make_validation_features()

    assert "PIA" not in set(training_features["Driver"])

    assert "PIA" in set(validation_features["Driver"])

    model = train_baseline_winner_model(training_features)

    predictions = predict_winner_probabilities(
        model,
        validation_features,
    )

    pia_predictions = predictions.loc[predictions["Driver"].eq("PIA")]

    assert len(pia_predictions) == 3

    assert bool(pia_predictions[WINNER_PROBABILITY_COLUMN].notna().all())


def test_prediction_does_not_refit_preprocessor() -> None:
    """Future inference must preserve training preprocessing statistics."""
    training_features = make_training_features()

    validation_features = make_validation_features()

    model = train_baseline_winner_model(training_features)

    scaler = model.preprocessor.transformer.named_transformers_["numeric"].named_steps[
        "scaler"
    ]

    mean_before = scaler.mean_.copy()
    scale_before = scaler.scale_.copy()

    predict_winner_probabilities(
        model,
        validation_features,
    )

    np.testing.assert_array_equal(
        scaler.mean_,
        mean_before,
    )

    np.testing.assert_array_equal(
        scaler.scale_,
        scale_before,
    )


def test_prediction_does_not_depend_on_target_column() -> None:
    """WonRace values must not influence inference probabilities."""
    training_features = make_training_features()

    validation_features = make_validation_features()

    model = train_baseline_winner_model(training_features)

    original_predictions = predict_winner_probabilities(
        model,
        validation_features,
    )

    changed_target = validation_features.copy(deep=True)

    changed_target[TARGET_COLUMN] = ~changed_target[TARGET_COLUMN]

    changed_predictions = predict_winner_probabilities(
        model,
        changed_target,
    )

    pd.testing.assert_frame_equal(
        original_predictions,
        changed_predictions,
    )


def test_repeated_prediction_is_deterministic() -> None:
    """Fixed fitted state and inputs should produce identical predictions."""
    model = train_baseline_winner_model(make_training_features())

    features = make_validation_features()

    first = predict_winner_probabilities(
        model,
        features,
    )

    second = predict_winner_probabilities(
        model,
        features,
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )


def test_repeated_training_produces_same_predictions() -> None:
    """The deterministic baseline should reproduce the same probabilities."""
    training_features = make_training_features()

    validation_features = make_validation_features()

    first_model = train_baseline_winner_model(training_features)

    second_model = train_baseline_winner_model(training_features)

    first_predictions = predict_winner_probabilities(
        first_model,
        validation_features,
    )

    second_predictions = predict_winner_probabilities(
        second_model,
        validation_features,
    )

    np.testing.assert_allclose(
        first_predictions[WINNER_PROBABILITY_COLUMN].to_numpy(dtype=float),
        second_predictions[WINNER_PROBABILITY_COLUMN].to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )


def test_prediction_rejects_non_model() -> None:
    """Inference requires an actual fitted TelemetryX baseline."""
    invalid_model: Any = {}

    with pytest.raises(
        TypeError,
        match=("model must be provided as a BaselineWinnerModel"),
    ):
        predict_winner_probabilities(
            invalid_model,
            make_validation_features(),
        )


def test_prediction_rejects_non_dataframe() -> None:
    """Inference features must be provided as a DataFrame."""
    model = train_baseline_winner_model(make_training_features())

    invalid_features: Any = []

    with pytest.raises(
        TypeError,
        match=("features must be provided as a pandas DataFrame"),
    ):
        predict_winner_probabilities(
            model,
            invalid_features,
        )


def test_prediction_rejects_empty_feature_frame() -> None:
    """There must be observations to score."""
    features = make_validation_features()

    empty = features.iloc[0:0].copy()

    model = train_baseline_winner_model(make_training_features())

    with pytest.raises(
        BaselineModelError,
        match="Features cannot be empty",
    ):
        predict_winner_probabilities(
            model,
            empty,
        )


def test_prediction_wraps_invalid_feature_frame() -> None:
    """Inference cannot bypass preprocessing feature validation."""
    model = train_baseline_winner_model(make_training_features())

    malformed = make_validation_features().drop(
        columns=[
            "PositionFraction",
        ]
    )

    with pytest.raises(
        BaselineModelError,
        match=("Features failed preprocessing before baseline prediction"),
    ):
        predict_winner_probabilities(
            model,
            malformed,
        )


def test_validate_prediction_frame_accepts_valid_predictions() -> None:
    """Valid baseline output should pass standalone validation."""
    validate_winner_probability_frame(make_predictions())


def test_validate_prediction_frame_rejects_non_dataframe() -> None:
    """Prediction validation requires a pandas DataFrame."""
    invalid_predictions: Any = []

    with pytest.raises(
        TypeError,
        match=("predictions must be provided as a pandas DataFrame"),
    ):
        validate_winner_probability_frame(invalid_predictions)


def test_validate_prediction_frame_rejects_empty_frame() -> None:
    """An empty probability table is invalid."""
    predictions = make_predictions().iloc[0:0].copy()

    with pytest.raises(
        BaselineModelError,
        match="contains no rows",
    ):
        validate_winner_probability_frame(predictions)


def test_validate_prediction_frame_rejects_wrong_schema() -> None:
    """Prediction columns must exactly match the public contract."""
    predictions = make_predictions()

    predictions["ExtraColumn"] = 1

    with pytest.raises(
        BaselineModelError,
        match=("does not match the required prediction schema"),
    ):
        validate_winner_probability_frame(predictions)


def test_validate_prediction_frame_rejects_missing_key() -> None:
    """Prediction identity columns cannot contain missing values."""
    predictions = make_predictions()

    predictions.loc[
        0,
        "Driver",
    ] = pd.NA

    with pytest.raises(
        BaselineModelError,
        match=("Prediction key columns cannot contain missing values"),
    ):
        validate_winner_probability_frame(predictions)


def test_validate_prediction_frame_rejects_duplicate_key() -> None:
    """A driver may appear only once in each race snapshot."""
    predictions = make_predictions()

    duplicate = predictions.iloc[[0]].copy()

    malformed = pd.concat(
        [
            predictions,
            duplicate,
        ],
        ignore_index=True,
    )

    with pytest.raises(
        BaselineModelError,
        match=("duplicate race-snapshot-driver rows"),
    ):
        validate_winner_probability_frame(malformed)


def test_validate_prediction_frame_rejects_missing_probability() -> None:
    """Winner probabilities cannot contain missing values."""
    predictions = make_predictions()

    predictions.loc[
        0,
        WINNER_PROBABILITY_COLUMN,
    ] = np.nan

    with pytest.raises(
        BaselineModelError,
        match=("Winner probabilities cannot contain missing values"),
    ):
        validate_winner_probability_frame(predictions)


def test_validate_prediction_frame_rejects_infinite_probability() -> None:
    """Winner probabilities must always be finite."""
    predictions = make_predictions()

    predictions.loc[
        0,
        WINNER_PROBABILITY_COLUMN,
    ] = np.inf

    with pytest.raises(
        BaselineModelError,
        match=("Winner probabilities must contain only finite values"),
    ):
        validate_winner_probability_frame(predictions)


@pytest.mark.parametrize(
    "invalid_probability",
    [
        -0.1,
        1.1,
    ],
)
def test_validate_prediction_frame_rejects_probability_outside_range(
    invalid_probability: float,
) -> None:
    """Winner probabilities must remain in [0, 1]."""
    predictions = make_predictions()

    predictions.loc[
        0,
        WINNER_PROBABILITY_COLUMN,
    ] = invalid_probability

    with pytest.raises(
        BaselineModelError,
        match=("Winner probabilities must remain between 0 and 1"),
    ):
        validate_winner_probability_frame(predictions)


def test_validate_prediction_frame_rejects_non_unit_snapshot_sum() -> None:
    """Each snapshot probability distribution must sum to one."""
    predictions = make_predictions()

    snapshot_mask = predictions["SnapshotLap"].eq(1)

    predictions.loc[
        snapshot_mask,
        WINNER_PROBABILITY_COLUMN,
    ] = 0.1

    with pytest.raises(
        BaselineModelError,
        match=("Winner probabilities must sum to one within every race snapshot"),
    ):
        validate_winner_probability_frame(predictions)
