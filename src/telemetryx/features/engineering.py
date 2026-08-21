from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from telemetryx.data.corpus import validate_race_corpus
from telemetryx.data.targets import TARGET_COLUMN

FEATURE_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "RaceId",
    "SnapshotLap",
    "Driver",
)

FEATURE_METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "RaceId",
    "Season",
    "RoundNumber",
    "EventName",
    "SessionName",
)

CATEGORICAL_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "Driver",
    "Compound",
    "TrackStatus",
)

NUMERIC_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "SnapshotLap",
    "Position",
    "FieldSize",
    "PositionFraction",
    "CompletedLaps",
    "CompletionFraction",
    "LapsBehindLeader",
    "Stint",
    "TyreLife",
    "LastLapTimeSeconds",
    "AverageLapTimeSeconds",
    "LeaderLastLapTimeSeconds",
    "LastLapDeltaToLeaderSeconds",
)

BOOLEAN_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "IsLeader",
    "IsLapped",
    "IsTopThree",
)

MODEL_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    *CATEGORICAL_FEATURE_COLUMNS,
    *NUMERIC_FEATURE_COLUMNS,
    *BOOLEAN_FEATURE_COLUMNS,
)

FEATURE_FRAME_COLUMNS: Final[tuple[str, ...]] = (
    *FEATURE_METADATA_COLUMNS,
    "SnapshotLap",
    "Driver",
    "Position",
    "FieldSize",
    "PositionFraction",
    "CompletedLaps",
    "CompletionFraction",
    "LapsBehindLeader",
    "Stint",
    "Compound",
    "TyreLife",
    "TrackStatus",
    "LastLapTimeSeconds",
    "AverageLapTimeSeconds",
    "LeaderLastLapTimeSeconds",
    "LastLapDeltaToLeaderSeconds",
    "IsLeader",
    "IsLapped",
    "IsTopThree",
    TARGET_COLUMN,
)

DISALLOWED_FEATURE_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "FinalPosition",
        "ClassifiedPosition",
        "Points",
        "Status",
        "GridPosition",
    }
)


class FeatureEngineeringError(ValueError):
    """Raised when leakage-safe features cannot be produced."""


def engineer_race_features(
    corpus: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create model-ready features from a validated historical race corpus.

    Every derived value uses only information already present in the temporal
    replay row or in other drivers' rows from the same race snapshot.

    Parameters
    ----------
    corpus:
        Valid TelemetryX multi-race corpus.

    Returns
    -------
    pd.DataFrame
        Feature table containing race metadata, model features and WonRace.

    Raises
    ------
    TypeError
        If ``corpus`` is not a pandas DataFrame.
    FeatureEngineeringError
        If the source corpus or generated feature table is invalid.
    """
    if not isinstance(
        corpus,
        pd.DataFrame,
    ):
        raise TypeError("corpus must be provided as a pandas DataFrame.")

    try:
        validate_race_corpus(corpus)
    except (TypeError, ValueError) as exc:
        raise FeatureEngineeringError(
            "The race corpus failed validation before feature engineering."
        ) from exc

    working = corpus.copy(deep=True)

    _reject_disallowed_source_columns(working)

    snapshot_groups = [
        working["RaceId"],
        working["SnapshotLap"],
    ]

    working["FieldSize"] = (
        working.groupby(
            snapshot_groups,
            sort=False,
            dropna=False,
        )["Driver"]
        .transform("size")
        .astype("Int64")
    )

    working["PositionFraction"] = _calculate_position_fraction(
        position=working["Position"],
        field_size=working["FieldSize"],
    )

    working["CompletionFraction"] = _calculate_completion_fraction(
        completed_laps=working["CompletedLaps"],
        snapshot_lap=working["SnapshotLap"],
    )

    working["IsLapped"] = working["LapsBehindLeader"].gt(0).astype("boolean")

    working["IsTopThree"] = working["Position"].le(3).astype("boolean")

    working["AverageLapTimeSeconds"] = _calculate_average_lap_time(
        cumulative_lap_time=working["CumulativeLapTimeSeconds"],
        completed_laps=working["CompletedLaps"],
    )

    working["LeaderLastLapTimeSeconds"] = _calculate_leader_last_lap_time(working)

    working["LastLapDeltaToLeaderSeconds"] = _calculate_last_lap_delta_to_leader(
        last_lap_time=working["LastLapTimeSeconds"],
        leader_last_lap_time=working["LeaderLastLapTimeSeconds"],
    )

    features = working.loc[
        :,
        list(FEATURE_FRAME_COLUMNS),
    ].copy(deep=True)

    _normalize_feature_dtypes(features)

    validate_feature_frame(features)

    return features


def validate_feature_frame(
    features: pd.DataFrame,
) -> None:
    """
    Validate the structural and temporal contract of engineered features.

    Parameters
    ----------
    features:
        Feature table produced by ``engineer_race_features``.

    Raises
    ------
    TypeError
        If ``features`` is not a pandas DataFrame.
    FeatureEngineeringError
        If feature invariants are violated.
    """
    if not isinstance(
        features,
        pd.DataFrame,
    ):
        raise TypeError("features must be provided as a pandas DataFrame.")

    if features.empty:
        raise FeatureEngineeringError("The feature frame contains no rows.")

    duplicate_columns = _duplicate_column_names(features)

    if duplicate_columns:
        raise FeatureEngineeringError(
            "The feature frame contains duplicate column names: "
            f"{', '.join(duplicate_columns)}."
        )

    missing_columns = [
        column for column in FEATURE_FRAME_COLUMNS if column not in features.columns
    ]

    if missing_columns:
        raise FeatureEngineeringError(
            "The feature frame is missing required columns: "
            f"{', '.join(missing_columns)}."
        )

    disallowed_columns = sorted(
        DISALLOWED_FEATURE_COLUMNS.intersection(
            set(str(column) for column in features.columns)
        )
    )

    if disallowed_columns:
        raise FeatureEngineeringError(
            "The feature frame contains prohibited post-race columns: "
            f"{', '.join(disallowed_columns)}."
        )

    duplicate_keys = features.duplicated(
        subset=list(FEATURE_KEY_COLUMNS),
        keep=False,
    )

    if bool(duplicate_keys.any()):
        raise FeatureEngineeringError(
            "The feature frame contains duplicate race-snapshot-driver rows."
        )

    _validate_field_size(features)

    _validate_position_fraction(features)

    _validate_completion_fraction(features)

    _validate_boolean_features(features)

    _validate_lap_delta(features)

    _validate_target(features)

    _validate_numeric_finiteness(features)


def _calculate_position_fraction(
    *,
    position: pd.Series,
    field_size: pd.Series,
) -> pd.Series:
    """
    Scale current position to the range [0, 1].

    Position 1 maps to 0. A last-place driver maps to 1 when at least two
    drivers are represented in the snapshot.
    """
    position_numeric = pd.to_numeric(
        position,
        errors="coerce",
    ).astype("Float64")

    field_size_numeric = pd.to_numeric(
        field_size,
        errors="coerce",
    ).astype("Float64")

    denominator = field_size_numeric - 1.0

    result = (position_numeric - 1.0).div(denominator)

    single_driver_snapshot = field_size_numeric.eq(1.0)

    result.loc[single_driver_snapshot] = 0.0

    return result.astype("Float64")


def _calculate_completion_fraction(
    *,
    completed_laps: pd.Series,
    snapshot_lap: pd.Series,
) -> pd.Series:
    """
    Measure how closely a driver matches the leader's completed lap count.

    A driver on the leader lap has value 1.0. A lapped driver's value is
    below 1.0.
    """
    completed_numeric = pd.to_numeric(
        completed_laps,
        errors="coerce",
    ).astype("Float64")

    snapshot_numeric = pd.to_numeric(
        snapshot_lap,
        errors="coerce",
    ).astype("Float64")

    result = completed_numeric.div(snapshot_numeric)

    return result.astype("Float64")


def _calculate_average_lap_time(
    *,
    cumulative_lap_time: pd.Series,
    completed_laps: pd.Series,
) -> pd.Series:
    """Return each driver's observed average lap time through the snapshot."""
    cumulative_numeric = pd.to_numeric(
        cumulative_lap_time,
        errors="coerce",
    ).astype("Float64")

    completed_numeric = pd.to_numeric(
        completed_laps,
        errors="coerce",
    ).astype("Float64")

    valid_denominator = completed_numeric.gt(0)

    result = pd.Series(
        pd.NA,
        index=cumulative_numeric.index,
        dtype="Float64",
    )

    result.loc[valid_denominator] = cumulative_numeric.loc[valid_denominator].div(
        completed_numeric.loc[valid_denominator]
    )

    return result


def _calculate_leader_last_lap_time(
    frame: pd.DataFrame,
) -> pd.Series:
    """
    Broadcast the current snapshot leader's last-lap time to every driver.

    The leader is determined from the temporal replay's IsLeader field at the
    same race snapshot. No future result information is consulted.
    """
    leader_times = (
        frame["LastLapTimeSeconds"]
        .where(frame["IsLeader"])
        .groupby(
            [
                frame["RaceId"],
                frame["SnapshotLap"],
            ],
            sort=False,
            dropna=False,
        )
        .transform("max")
    )

    return pd.to_numeric(
        leader_times,
        errors="coerce",
    ).astype("Float64")


def _calculate_last_lap_delta_to_leader(
    *,
    last_lap_time: pd.Series,
    leader_last_lap_time: pd.Series,
) -> pd.Series:
    """Return current last-lap pace relative to the snapshot leader."""
    driver_time = pd.to_numeric(
        last_lap_time,
        errors="coerce",
    ).astype("Float64")

    leader_time = pd.to_numeric(
        leader_last_lap_time,
        errors="coerce",
    ).astype("Float64")

    return (driver_time - leader_time).astype("Float64")


def _normalize_feature_dtypes(
    features: pd.DataFrame,
) -> None:
    """Normalize generated features to stable pandas extension dtypes."""
    for column in (
        "FieldSize",
        "CompletedLaps",
        "LapsBehindLeader",
        "Stint",
        "SnapshotLap",
        "Position",
    ):
        features[column] = pd.to_numeric(
            features[column],
            errors="coerce",
        ).astype("Int64")

    for column in (
        "PositionFraction",
        "CompletionFraction",
        "TyreLife",
        "LastLapTimeSeconds",
        "AverageLapTimeSeconds",
        "LeaderLastLapTimeSeconds",
        "LastLapDeltaToLeaderSeconds",
    ):
        features[column] = pd.to_numeric(
            features[column],
            errors="coerce",
        ).astype("Float64")

    for column in (
        "IsLeader",
        "IsLapped",
        "IsTopThree",
        TARGET_COLUMN,
    ):
        features[column] = features[column].astype("boolean")

    for column in (
        "RaceId",
        "EventName",
        "SessionName",
        "Driver",
        "Compound",
        "TrackStatus",
    ):
        features[column] = features[column].astype("string")


def _validate_field_size(
    features: pd.DataFrame,
) -> None:
    """Ensure every snapshot has a consistent positive field size."""
    invalid = features["FieldSize"].isna() | features["FieldSize"].le(0)

    if bool(invalid.any()):
        raise FeatureEngineeringError("FieldSize must contain positive values.")

    unique_sizes = features.groupby(
        [
            "RaceId",
            "SnapshotLap",
        ],
        sort=False,
        dropna=False,
    )["FieldSize"].nunique(dropna=False)

    if bool(unique_sizes.ne(1).any()):
        raise FeatureEngineeringError(
            "FieldSize must be constant within each race snapshot."
        )


def _validate_position_fraction(
    features: pd.DataFrame,
) -> None:
    """Require normalized position values to remain inside [0, 1]."""
    values = features["PositionFraction"]

    invalid = values.isna() | values.lt(0.0) | values.gt(1.0)

    if bool(invalid.any()):
        raise FeatureEngineeringError("PositionFraction must remain between 0 and 1.")


def _validate_completion_fraction(
    features: pd.DataFrame,
) -> None:
    """Require completion fractions to describe observed lap progress."""
    values = features["CompletionFraction"]

    invalid = values.isna() | values.lt(0.0) | values.gt(1.0)

    if bool(invalid.any()):
        raise FeatureEngineeringError("CompletionFraction must remain between 0 and 1.")


def _validate_boolean_features(
    features: pd.DataFrame,
) -> None:
    """Require generated indicator features to use Boolean dtypes."""
    for column in BOOLEAN_FEATURE_COLUMNS:
        if not pd.api.types.is_bool_dtype(features[column].dtype):
            raise FeatureEngineeringError(f"{column} must use a Boolean dtype.")

        if bool(features[column].isna().any()):
            raise FeatureEngineeringError(f"{column} cannot contain missing values.")


def _validate_lap_delta(
    features: pd.DataFrame,
) -> None:
    """Require each leader to have zero last-lap delta when pace is known."""
    leaders = features.loc[features["IsLeader"]]

    known_leader_deltas = leaders["LastLapDeltaToLeaderSeconds"].dropna()

    non_zero = known_leader_deltas.abs().gt(1e-9)

    if bool(non_zero.any()):
        raise FeatureEngineeringError(
            "A leader with known lap time must have zero LastLapDeltaToLeaderSeconds."
        )


def _validate_target(
    features: pd.DataFrame,
) -> None:
    """Ensure the outcome remains a Boolean label, not an input feature."""
    target = features[TARGET_COLUMN]

    if bool(target.isna().any()):
        raise FeatureEngineeringError(f"{TARGET_COLUMN} cannot contain missing values.")

    if not pd.api.types.is_bool_dtype(target.dtype):
        raise FeatureEngineeringError(f"{TARGET_COLUMN} must use a Boolean dtype.")

    if TARGET_COLUMN in MODEL_FEATURE_COLUMNS:
        raise FeatureEngineeringError(
            f"{TARGET_COLUMN} cannot be included in model feature columns."
        )


def _validate_numeric_finiteness(
    features: pd.DataFrame,
) -> None:
    """Reject positive or negative infinity in numeric model features."""
    for column in NUMERIC_FEATURE_COLUMNS:
        numeric = pd.to_numeric(
            features[column],
            errors="coerce",
        )

        non_missing = numeric.dropna()

        if non_missing.empty:
            continue

        values = non_missing.to_numpy(
            dtype=float,
        )

        if not bool(np.isfinite(values).all()):
            raise FeatureEngineeringError(
                f"{column} contains non-finite numeric values."
            )


def _reject_disallowed_source_columns(
    corpus: pd.DataFrame,
) -> None:
    """Reject known post-race fields before feature derivation."""
    disallowed = sorted(
        DISALLOWED_FEATURE_COLUMNS.intersection(
            set(str(column) for column in corpus.columns)
        )
    )

    if disallowed:
        raise FeatureEngineeringError(
            "Feature engineering received prohibited post-race columns: "
            f"{', '.join(disallowed)}."
        )


def _duplicate_column_names(
    frame: pd.DataFrame,
) -> tuple[str, ...]:
    """Return duplicated DataFrame column names."""
    duplicated = frame.columns[frame.columns.duplicated()]

    return tuple(str(column) for column in duplicated)
