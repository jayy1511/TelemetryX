from __future__ import annotations

from typing import Final

import pandas as pd

from telemetryx.data.targets import TARGET_COLUMN

WINNER_PROBABILITY_COLUMN: Final[str] = "WinnerProbability"

PREDICTION_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "RaceId",
    "SnapshotLap",
    "Driver",
)

UNIFORM_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (*PREDICTION_KEY_COLUMNS,)

DRIVER_PRIOR_TRAIN_COLUMNS: Final[tuple[str, ...]] = (
    "RaceId",
    "Driver",
    TARGET_COLUMN,
)

DEFAULT_PRIOR_SMOOTHING: Final[float] = 1.0

CURRENT_LEADER_COLUMN: Final[str] = "IsLeader"

CURRENT_LEADER_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    *PREDICTION_KEY_COLUMNS,
    CURRENT_LEADER_COLUMN,
)

DEFAULT_LEADER_PROBABILITY: Final[float] = 0.5


class BenchmarkPredictionError(ValueError):
    """Raised when a benchmark prediction cannot be constructed."""


def predict_uniform_winner_probabilities(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign equal winner probability to every driver in each snapshot.

    A snapshot containing 20 drivers therefore assigns probability
    1 / 20 to every driver.
    """
    _validate_prediction_features(features)

    predictions = features.loc[
        :,
        list(PREDICTION_KEY_COLUMNS),
    ].copy(deep=True)

    field_size = (
        predictions.groupby(
            [
                "RaceId",
                "SnapshotLap",
            ],
            sort=False,
        )["Driver"]
        .transform("size")
        .astype("Float64")
    )

    if bool(field_size.le(0).any()):
        raise BenchmarkPredictionError(
            "Every snapshot must contain at least one driver."
        )

    predictions[WINNER_PROBABILITY_COLUMN] = (1.0 / field_size).astype("Float64")

    _validate_prediction_probabilities(predictions)

    return predictions


def predict_current_leader_winner_probabilities(
    features: pd.DataFrame,
    *,
    leader_probability: float = DEFAULT_LEADER_PROBABILITY,
) -> pd.DataFrame:
    """
    Predict race winners using only the current snapshot leader.

    The current leader receives a fixed probability mass. All remaining
    probability mass is distributed equally across the other drivers.
    """
    _validate_prediction_features(features)

    missing_columns = [
        column
        for column in CURRENT_LEADER_REQUIRED_COLUMNS
        if column not in features.columns
    ]

    if missing_columns:
        raise BenchmarkPredictionError(
            f"Current-leader benchmark is missing required columns: {missing_columns}."
        )

    if isinstance(leader_probability, bool) or not isinstance(
        leader_probability,
        int | float,
    ):
        raise TypeError("leader_probability must be a number between zero and one.")

    leader_probability_value = float(leader_probability)

    if not 0.0 < leader_probability_value < 1.0:
        raise BenchmarkPredictionError(
            "leader_probability must be strictly between zero and one."
        )

    predictions = features.loc[
        :,
        [
            *PREDICTION_KEY_COLUMNS,
            CURRENT_LEADER_COLUMN,
        ],
    ].copy(deep=True)

    leaders = predictions[CURRENT_LEADER_COLUMN].astype("boolean")

    if bool(leaders.isna().any()):
        raise BenchmarkPredictionError("IsLeader cannot contain missing values.")

    predictions["_IsLeader"] = leaders

    snapshot_group = predictions.groupby(
        [
            "RaceId",
            "SnapshotLap",
        ],
        sort=False,
        dropna=False,
    )

    leader_count = snapshot_group["_IsLeader"].transform("sum").astype("Int64")

    if bool(leader_count.ne(1).any()):
        raise BenchmarkPredictionError(
            "Every snapshot must contain exactly one current leader."
        )

    field_size = snapshot_group["Driver"].transform("size").astype("Int64")

    probabilities = pd.Series(
        index=predictions.index,
        dtype="Float64",
    )

    single_driver_snapshot = field_size.eq(1)

    probabilities.loc[single_driver_snapshot] = 1.0

    multi_driver_snapshot = ~single_driver_snapshot

    non_leader_probability = (1.0 - leader_probability_value) / (
        field_size.astype("Float64") - 1.0
    )

    probabilities.loc[multi_driver_snapshot & predictions["_IsLeader"]] = (
        leader_probability_value
    )

    probabilities.loc[multi_driver_snapshot & ~predictions["_IsLeader"]] = (
        non_leader_probability.loc[multi_driver_snapshot & ~predictions["_IsLeader"]]
    )

    predictions[WINNER_PROBABILITY_COLUMN] = probabilities

    predictions = predictions.drop(
        columns=[
            CURRENT_LEADER_COLUMN,
            "_IsLeader",
        ]
    )

    _validate_prediction_probabilities(predictions)

    return predictions


def predict_driver_prior_winner_probabilities(
    training_features: pd.DataFrame,
    prediction_features: pd.DataFrame,
    *,
    smoothing: float = DEFAULT_PRIOR_SMOOTHING,
) -> pd.DataFrame:
    """
    Predict using only historical driver win frequency.

    The prior is learned from training races only. Repeated lap snapshots
    do not increase a driver's weight because training data is reduced to
    one target observation per race and driver before frequencies are
    calculated.

    Additive smoothing gives every driver a non-zero prior, including
    drivers unseen in the training set. Priors are then normalized within
    each prediction snapshot so winner probabilities sum to one.
    """
    _validate_driver_prior_training_frame(training_features)

    _validate_prediction_features(prediction_features)

    if isinstance(
        smoothing,
        bool,
    ) or not isinstance(
        smoothing,
        int | float,
    ):
        raise TypeError("smoothing must be a positive number.")

    smoothing_value = float(smoothing)

    if smoothing_value <= 0.0:
        raise BenchmarkPredictionError("smoothing must be greater than zero.")

    race_driver_targets = training_features.loc[
        :,
        list(DRIVER_PRIOR_TRAIN_COLUMNS),
    ].copy(deep=True)

    _validate_stable_race_driver_targets(race_driver_targets)

    race_driver_targets = race_driver_targets.drop_duplicates(
        subset=[
            "RaceId",
            "Driver",
        ],
        keep="first",
    )

    race_driver_targets[TARGET_COLUMN] = (
        race_driver_targets[TARGET_COLUMN].astype("boolean").astype("Int64")
    )

    driver_statistics = race_driver_targets.groupby(
        "Driver",
        sort=False,
    ).agg(
        RaceAppearances=(
            "RaceId",
            "nunique",
        ),
        RaceWins=(
            TARGET_COLUMN,
            "sum",
        ),
    )

    driver_statistics["HistoricalWinRate"] = (
        driver_statistics["RaceWins"].astype("Float64") + smoothing_value
    ) / (
        driver_statistics["RaceAppearances"].astype("Float64") + (2.0 * smoothing_value)
    )

    predictions = prediction_features.loc[
        :,
        list(PREDICTION_KEY_COLUMNS),
    ].copy(deep=True)

    predictions["_DriverPrior"] = predictions["Driver"].map(
        driver_statistics["HistoricalWinRate"]
    )

    fallback_prior = _calculate_unseen_driver_prior(
        race_driver_targets,
        smoothing=smoothing_value,
    )

    predictions["_DriverPrior"] = (
        pd.to_numeric(
            predictions["_DriverPrior"],
            errors="coerce",
        )
        .astype("Float64")
        .fillna(fallback_prior)
    )

    snapshot_prior_total = (
        predictions.groupby(
            [
                "RaceId",
                "SnapshotLap",
            ],
            sort=False,
        )["_DriverPrior"]
        .transform("sum")
        .astype("Float64")
    )

    if bool(snapshot_prior_total.le(0.0).any()):
        raise BenchmarkPredictionError(
            "Driver priors must have positive mass in every snapshot."
        )

    predictions[WINNER_PROBABILITY_COLUMN] = (
        predictions["_DriverPrior"] / snapshot_prior_total
    ).astype("Float64")

    predictions = predictions.drop(
        columns=[
            "_DriverPrior",
        ]
    )

    _validate_prediction_probabilities(predictions)

    return predictions


def _calculate_unseen_driver_prior(
    race_driver_targets: pd.DataFrame,
    *,
    smoothing: float,
) -> float:
    """Return a smoothed fallback prior for unseen drivers."""
    total_races = int(race_driver_targets["RaceId"].nunique())

    if total_races <= 0:
        raise BenchmarkPredictionError("Training data must contain at least one race.")

    return smoothing / (float(total_races) + (2.0 * smoothing))


def _validate_prediction_features(
    features: pd.DataFrame,
) -> None:
    """Validate the identity columns required for benchmark prediction."""
    if not isinstance(
        features,
        pd.DataFrame,
    ):
        raise TypeError("features must be a pandas DataFrame.")

    if features.empty:
        raise BenchmarkPredictionError("Prediction features cannot be empty.")

    missing_columns = [
        column for column in UNIFORM_REQUIRED_COLUMNS if column not in features.columns
    ]

    if missing_columns:
        raise BenchmarkPredictionError(
            "Prediction features are missing required columns: "
            f"{', '.join(missing_columns)}."
        )

    if bool(
        features.loc[
            :,
            list(PREDICTION_KEY_COLUMNS),
        ]
        .isna()
        .any()
        .any()
    ):
        raise BenchmarkPredictionError(
            "Prediction identity columns cannot contain missing values."
        )

    duplicate_rows = features.duplicated(
        subset=list(PREDICTION_KEY_COLUMNS),
        keep=False,
    )

    if bool(duplicate_rows.any()):
        raise BenchmarkPredictionError(
            "Prediction features contain duplicate RaceId/SnapshotLap/Driver rows."
        )


def _validate_driver_prior_training_frame(
    training_features: pd.DataFrame,
) -> None:
    """Validate the training data required for the driver-prior benchmark."""
    if not isinstance(
        training_features,
        pd.DataFrame,
    ):
        raise TypeError("training_features must be a pandas DataFrame.")

    if training_features.empty:
        raise BenchmarkPredictionError(
            "Driver-prior training features cannot be empty."
        )

    missing_columns = [
        column
        for column in DRIVER_PRIOR_TRAIN_COLUMNS
        if column not in training_features.columns
    ]

    if missing_columns:
        raise BenchmarkPredictionError(
            "Driver-prior training features are missing required columns: "
            f"{', '.join(missing_columns)}."
        )

    if bool(
        training_features.loc[
            :,
            [
                "RaceId",
                "Driver",
                TARGET_COLUMN,
            ],
        ]
        .isna()
        .any()
        .any()
    ):
        raise BenchmarkPredictionError(
            "Driver-prior training columns cannot contain missing values."
        )

    target_values = training_features[TARGET_COLUMN]

    if not pd.api.types.is_bool_dtype(target_values.dtype):
        raise BenchmarkPredictionError(f"{TARGET_COLUMN} must use a boolean dtype.")


def _validate_stable_race_driver_targets(
    race_driver_targets: pd.DataFrame,
) -> None:
    """Require the target to be constant across snapshots of one driver."""
    target_counts = race_driver_targets.groupby(
        [
            "RaceId",
            "Driver",
        ],
        sort=False,
    )[TARGET_COLUMN].nunique(dropna=False)

    if bool(target_counts.ne(1).any()):
        raise BenchmarkPredictionError(
            "Each race-driver pair must have one stable winner target."
        )

    race_targets = race_driver_targets.drop_duplicates(
        subset=[
            "RaceId",
            "Driver",
        ],
        keep="first",
    )

    winners_per_race = race_targets.groupby(
        "RaceId",
        sort=False,
    )[TARGET_COLUMN].sum()

    if bool(winners_per_race.ne(1).any()):
        raise BenchmarkPredictionError(
            "Each training race must contain exactly one winner."
        )


def _validate_prediction_probabilities(
    predictions: pd.DataFrame,
) -> None:
    """Require finite per-snapshot probabilities that sum to one."""
    probabilities = pd.to_numeric(
        predictions[WINNER_PROBABILITY_COLUMN],
        errors="coerce",
    ).astype("Float64")

    if bool(probabilities.isna().any()):
        raise BenchmarkPredictionError(
            "Winner probabilities cannot contain missing values."
        )

    if bool((probabilities.lt(0.0) | probabilities.gt(1.0)).any()):
        raise BenchmarkPredictionError(
            "Winner probabilities must remain between 0 and 1."
        )

    totals = (
        predictions.assign(
            **{
                WINNER_PROBABILITY_COLUMN: probabilities,
            }
        )
        .groupby(
            [
                "RaceId",
                "SnapshotLap",
            ],
            sort=False,
        )[WINNER_PROBABILITY_COLUMN]
        .sum()
    )

    invalid_totals = totals.sub(1.0).abs().gt(1e-9)

    if bool(invalid_totals.any()):
        raise BenchmarkPredictionError(
            "Winner probabilities must sum to 1 within every race snapshot."
        )
