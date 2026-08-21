from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from telemetryx.data.corpus import combine_race_datasets
from telemetryx.data.dataset import build_race_dataset
from telemetryx.data.targets import TARGET_COLUMN
from telemetryx.features.engineering import (
    BOOLEAN_FEATURE_COLUMNS,
    CATEGORICAL_FEATURE_COLUMNS,
    FEATURE_FRAME_COLUMNS,
    FEATURE_KEY_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    FeatureEngineeringError,
    engineer_race_features,
    validate_feature_frame,
)


def make_cleaned_laps(
    *,
    lap_time_offset: float = 0.0,
    change_future_lap: bool = False,
) -> pd.DataFrame:
    """
    Return cleaned lap data for a four-driver synthetic race.

    BOT completes only two laps and is therefore lapped in snapshot three.
    ALB gives us a fourth-place driver for testing IsTopThree.

    When ``change_future_lap`` is true, only lap-three observations are
    changed. Earlier laps remain identical so temporal-leakage tests can
    verify that snapshots one and two are unaffected.
    """
    ver_lap_three = 150.0 if change_future_lap else 88.0

    nor_lap_three = 160.0 if change_future_lap else 89.0

    alb_lap_three = 170.0 if change_future_lap else 91.0

    future_compound = "HARD" if change_future_lap else "MEDIUM"

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
                "ALB",
                "ALB",
                "ALB",
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
                4,
                4,
                4,
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
                1,
                1,
                2,
            ],
            "Compound": [
                "SOFT",
                "SOFT",
                future_compound,
                "SOFT",
                "SOFT",
                future_compound,
                "SOFT",
                "SOFT",
                "SOFT",
                "SOFT",
                future_compound,
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
                "1",
                "1",
            ],
            "LapTimeSeconds": [
                90.0 + lap_time_offset,
                89.0 + lap_time_offset,
                ver_lap_three + lap_time_offset,
                91.0 + lap_time_offset,
                90.0 + lap_time_offset,
                nor_lap_three + lap_time_offset,
                92.0 + lap_time_offset,
                91.0 + lap_time_offset,
                93.0 + lap_time_offset,
                92.0 + lap_time_offset,
                alb_lap_three + lap_time_offset,
            ],
        }
    )


def make_results(
    *,
    winner: str = "VER",
) -> pd.DataFrame:
    """Return final results containing exactly one winner."""
    drivers = [
        "VER",
        "NOR",
        "BOT",
        "ALB",
    ]

    ordered_drivers = [
        winner,
        *[driver for driver in drivers if driver != winner],
    ]

    return pd.DataFrame(
        {
            "Abbreviation": ordered_drivers,
            "Position": [
                1,
                2,
                3,
                4,
            ],
            "Status": [
                "Finished",
                "Finished",
                "+1 Lap",
                "Finished",
            ],
        }
    )


def make_race_dataset(
    *,
    season: int = 2023,
    round_number: int = 1,
    event_name: str = "Bahrain Grand Prix",
    winner: str = "VER",
    lap_time_offset: float = 0.0,
    change_future_lap: bool = False,
) -> pd.DataFrame:
    """Return one valid synthetic race dataset."""
    return build_race_dataset(
        make_cleaned_laps(
            lap_time_offset=lap_time_offset,
            change_future_lap=change_future_lap,
        ),
        make_results(
            winner=winner,
        ),
        season=season,
        round_number=round_number,
        event_name=event_name,
        session_name="Race",
    )


def make_corpus() -> pd.DataFrame:
    """Return a valid two-race corpus."""
    first_race = make_race_dataset(
        season=2023,
        round_number=1,
        event_name="Bahrain Grand Prix",
    )

    second_race = make_race_dataset(
        season=2023,
        round_number=2,
        event_name="Saudi Arabian Grand Prix",
        lap_time_offset=10.0,
    )

    return combine_race_datasets(
        [
            second_race,
            first_race,
        ]
    )


def make_features() -> pd.DataFrame:
    """Return a valid engineered feature frame."""
    return engineer_race_features(make_corpus())


def select_feature_row(
    features: pd.DataFrame,
    *,
    race_id: str,
    snapshot_lap: int,
    driver: str,
) -> pd.Series:
    """Return exactly one driver row from one race snapshot."""
    selected = features.loc[
        features["RaceId"].eq(race_id)
        & features["SnapshotLap"].eq(snapshot_lap)
        & features["Driver"].eq(driver)
    ]

    assert len(selected) == 1

    return selected.iloc[0]


def test_engineering_uses_documented_feature_schema() -> None:
    """Engineered data should use the deterministic feature-frame schema."""
    features = make_features()

    assert tuple(features.columns) == FEATURE_FRAME_COLUMNS

    assert len(features) == 24


def test_feature_keys_are_unique() -> None:
    """Every race-snapshot-driver observation must remain unique."""
    features = make_features()

    duplicates = features.duplicated(
        subset=list(FEATURE_KEY_COLUMNS),
        keep=False,
    )

    assert bool(duplicates.any()) is False

    assert FEATURE_KEY_COLUMNS == (
        "RaceId",
        "SnapshotLap",
        "Driver",
    )


def test_field_size_is_calculated_per_snapshot() -> None:
    """Every driver should observe the number of represented competitors."""
    features = make_features()

    assert set(features["FieldSize"].unique()) == {
        4,
    }

    snapshot_sizes = features.groupby(
        [
            "RaceId",
            "SnapshotLap",
        ]
    )["FieldSize"].unique()

    assert all(values.tolist() == [4] for values in snapshot_sizes)


def test_position_fraction_scales_position_to_field() -> None:
    """Current position should be normalized relative to field size."""
    features = make_features()

    race_id = "2023_01_bahrain_grand_prix"

    ver = select_feature_row(
        features,
        race_id=race_id,
        snapshot_lap=1,
        driver="VER",
    )

    nor = select_feature_row(
        features,
        race_id=race_id,
        snapshot_lap=1,
        driver="NOR",
    )

    bot = select_feature_row(
        features,
        race_id=race_id,
        snapshot_lap=1,
        driver="BOT",
    )

    alb = select_feature_row(
        features,
        race_id=race_id,
        snapshot_lap=1,
        driver="ALB",
    )

    assert ver["PositionFraction"] == pytest.approx(0.0)

    assert nor["PositionFraction"] == pytest.approx(1.0 / 3.0)

    assert bot["PositionFraction"] == pytest.approx(2.0 / 3.0)

    assert alb["PositionFraction"] == pytest.approx(1.0)


def test_completion_fraction_identifies_lapped_driver() -> None:
    """A driver one lap down should have completion below one."""
    features = make_features()

    bot = select_feature_row(
        features,
        race_id="2023_01_bahrain_grand_prix",
        snapshot_lap=3,
        driver="BOT",
    )

    assert bot["CompletedLaps"] == 2

    assert bot["CompletionFraction"] == pytest.approx(2.0 / 3.0)

    assert bool(bot["IsLapped"]) is True


def test_lead_lap_drivers_have_full_completion_fraction() -> None:
    """Drivers on the leader lap should have completion fraction one."""
    features = make_features()

    ver = select_feature_row(
        features,
        race_id="2023_01_bahrain_grand_prix",
        snapshot_lap=3,
        driver="VER",
    )

    nor = select_feature_row(
        features,
        race_id="2023_01_bahrain_grand_prix",
        snapshot_lap=3,
        driver="NOR",
    )

    assert ver["CompletionFraction"] == pytest.approx(1.0)

    assert nor["CompletionFraction"] == pytest.approx(1.0)

    assert bool(ver["IsLapped"]) is False


def test_top_three_feature_uses_current_position() -> None:
    """IsTopThree should describe current replay position only."""
    features = make_features()

    bot = select_feature_row(
        features,
        race_id="2023_01_bahrain_grand_prix",
        snapshot_lap=1,
        driver="BOT",
    )

    alb = select_feature_row(
        features,
        race_id="2023_01_bahrain_grand_prix",
        snapshot_lap=1,
        driver="ALB",
    )

    assert bool(bot["IsTopThree"]) is True

    assert bool(alb["IsTopThree"]) is False


def test_average_lap_time_uses_observed_cumulative_time() -> None:
    """Average pace should use only completed laps through the snapshot."""
    features = make_features()

    ver = select_feature_row(
        features,
        race_id="2023_01_bahrain_grand_prix",
        snapshot_lap=3,
        driver="VER",
    )

    bot = select_feature_row(
        features,
        race_id="2023_01_bahrain_grand_prix",
        snapshot_lap=3,
        driver="BOT",
    )

    assert ver["AverageLapTimeSeconds"] == pytest.approx((90.0 + 89.0 + 88.0) / 3.0)

    assert bot["AverageLapTimeSeconds"] == pytest.approx((92.0 + 91.0) / 2.0)


def test_leader_last_lap_time_is_broadcast_within_snapshot() -> None:
    """Every driver should receive the same current leader lap time."""
    features = make_features()

    snapshot = features.loc[
        features["RaceId"].eq("2023_01_bahrain_grand_prix")
        & features["SnapshotLap"].eq(3)
    ]

    assert set(snapshot["LeaderLastLapTimeSeconds"].tolist()) == {
        88.0,
    }


def test_leader_lap_time_does_not_cross_race_boundary() -> None:
    """Leader pace must be grouped by both race and snapshot."""
    features = make_features()

    first_race_leader = select_feature_row(
        features,
        race_id="2023_01_bahrain_grand_prix",
        snapshot_lap=1,
        driver="VER",
    )

    second_race_leader = select_feature_row(
        features,
        race_id=("2023_02_saudi_arabian_grand_prix"),
        snapshot_lap=1,
        driver="VER",
    )

    assert first_race_leader["LeaderLastLapTimeSeconds"] == pytest.approx(90.0)

    assert second_race_leader["LeaderLastLapTimeSeconds"] == pytest.approx(100.0)


def test_last_lap_delta_is_relative_to_current_leader() -> None:
    """Driver pace deltas should use the leader of the same snapshot."""
    features = make_features()

    race_id = "2023_01_bahrain_grand_prix"

    ver = select_feature_row(
        features,
        race_id=race_id,
        snapshot_lap=3,
        driver="VER",
    )

    nor = select_feature_row(
        features,
        race_id=race_id,
        snapshot_lap=3,
        driver="NOR",
    )

    bot = select_feature_row(
        features,
        race_id=race_id,
        snapshot_lap=3,
        driver="BOT",
    )

    assert ver["LastLapDeltaToLeaderSeconds"] == pytest.approx(0.0)

    assert nor["LastLapDeltaToLeaderSeconds"] == pytest.approx(1.0)

    assert bot["LastLapDeltaToLeaderSeconds"] == pytest.approx(3.0)


def test_target_is_preserved_but_not_a_model_feature() -> None:
    """WonRace must remain available only as the supervised target."""
    features = make_features()

    assert TARGET_COLUMN in features.columns

    assert TARGET_COLUMN not in MODEL_FEATURE_COLUMNS

    winner_rows = features.loc[features[TARGET_COLUMN]]

    assert set(winner_rows["Driver"]) == {
        "VER",
    }


def test_model_feature_columns_exist_in_feature_frame() -> None:
    """Every declared model input should be available after engineering."""
    features = make_features()

    assert all(column in features.columns for column in MODEL_FEATURE_COLUMNS)


def test_post_race_result_fields_are_not_features() -> None:
    """Engineered model data must not expose final-result information."""
    features = make_features()

    for column in (
        "FinalPosition",
        "ClassifiedPosition",
        "Points",
        "Status",
        "GridPosition",
    ):
        assert column not in features.columns

    assert "CumulativeLapTimeSeconds" not in MODEL_FEATURE_COLUMNS


def test_feature_dtypes_are_normalized() -> None:
    """Generated feature groups should use stable pandas dtypes."""
    features = make_features()

    for column in (
        "SnapshotLap",
        "Position",
        "FieldSize",
        "CompletedLaps",
        "LapsBehindLeader",
        "Stint",
    ):
        assert str(features[column].dtype) == "Int64"

    for column in (
        "PositionFraction",
        "CompletionFraction",
        "TyreLife",
        "LastLapTimeSeconds",
        "AverageLapTimeSeconds",
        "LeaderLastLapTimeSeconds",
        "LastLapDeltaToLeaderSeconds",
    ):
        assert str(features[column].dtype) == "Float64"

    for column in BOOLEAN_FEATURE_COLUMNS:
        assert str(features[column].dtype) == "boolean"

    assert str(features[TARGET_COLUMN].dtype) == "boolean"

    for column in CATEGORICAL_FEATURE_COLUMNS:
        assert str(features[column].dtype) == "string"


def test_declared_numeric_features_are_present() -> None:
    """All numeric model features should exist in the output."""
    features = make_features()

    assert all(column in features.columns for column in NUMERIC_FEATURE_COLUMNS)


def test_engineering_does_not_modify_source_corpus() -> None:
    """Feature engineering must leave the canonical corpus unchanged."""
    corpus = make_corpus()

    original = corpus.copy(deep=True)

    engineer_race_features(corpus)

    pd.testing.assert_frame_equal(
        corpus,
        original,
    )


def test_future_lap_changes_do_not_affect_earlier_features() -> None:
    """
    Changing lap three must not alter features available through lap two.

    This is the critical temporal leakage regression test.
    """
    original_race = make_race_dataset(change_future_lap=False)

    modified_future_race = make_race_dataset(change_future_lap=True)

    original_features = engineer_race_features(original_race)

    modified_features = engineer_race_features(modified_future_race)

    original_early = original_features.loc[
        original_features["SnapshotLap"].le(2)
    ].reset_index(drop=True)

    modified_early = modified_features.loc[
        modified_features["SnapshotLap"].le(2)
    ].reset_index(drop=True)

    pd.testing.assert_frame_equal(
        original_early,
        modified_early,
    )


def test_future_lap_changes_do_affect_future_snapshot() -> None:
    """The temporal regression fixture must actually alter lap-three data."""
    original_features = engineer_race_features(
        make_race_dataset(change_future_lap=False)
    )

    modified_features = engineer_race_features(
        make_race_dataset(change_future_lap=True)
    )

    original_lap_three = original_features.loc[original_features["SnapshotLap"].eq(3)][
        "LastLapTimeSeconds"
    ].reset_index(drop=True)

    modified_lap_three = modified_features.loc[modified_features["SnapshotLap"].eq(3)][
        "LastLapTimeSeconds"
    ].reset_index(drop=True)

    assert not original_lap_three.equals(modified_lap_three)


def test_engineer_features_rejects_non_dataframe() -> None:
    """Feature engineering requires a pandas DataFrame."""
    invalid_corpus: Any = []

    with pytest.raises(
        TypeError,
        match="corpus must be provided as a pandas DataFrame",
    ):
        engineer_race_features(invalid_corpus)


def test_engineer_features_wraps_invalid_corpus() -> None:
    """Malformed corpus structure should fail before derivation."""
    corpus = make_corpus().drop(
        columns=[
            "CompletedLaps",
        ]
    )

    with pytest.raises(
        FeatureEngineeringError,
        match="failed validation before feature engineering",
    ):
        engineer_race_features(corpus)


def test_validate_feature_frame_accepts_valid_features() -> None:
    """Correctly engineered data should pass standalone validation."""
    validate_feature_frame(make_features())


def test_validate_feature_frame_rejects_non_dataframe() -> None:
    """Feature validation requires a pandas DataFrame."""
    invalid_features: Any = []

    with pytest.raises(
        TypeError,
        match="features must be provided as a pandas DataFrame",
    ):
        validate_feature_frame(invalid_features)


def test_validate_feature_frame_rejects_empty_table() -> None:
    """An empty feature table cannot be used for modeling."""
    features = make_features().iloc[0:0]

    with pytest.raises(
        FeatureEngineeringError,
        match="contains no rows",
    ):
        validate_feature_frame(features)


def test_validate_feature_frame_reports_missing_columns() -> None:
    """Missing engineered columns should fail validation."""
    features = make_features().drop(
        columns=[
            "PositionFraction",
            "CompletionFraction",
        ]
    )

    with pytest.raises(
        FeatureEngineeringError,
        match="missing required columns",
    ) as exception_info:
        validate_feature_frame(features)

    message = str(exception_info.value)

    assert "PositionFraction" in message
    assert "CompletionFraction" in message


def test_validate_feature_frame_rejects_duplicate_columns() -> None:
    """Duplicate feature names should be rejected."""
    features = make_features()

    malformed = pd.concat(
        [
            features,
            features[["Driver"]],
        ],
        axis=1,
    )

    with pytest.raises(
        FeatureEngineeringError,
        match="duplicate column names",
    ):
        validate_feature_frame(malformed)


def test_validate_feature_frame_rejects_duplicate_keys() -> None:
    """A race-snapshot-driver observation cannot appear twice."""
    features = make_features()

    duplicate_row = features.iloc[[0]].copy()

    malformed = pd.concat(
        [
            features,
            duplicate_row,
        ],
        ignore_index=True,
    )

    with pytest.raises(
        FeatureEngineeringError,
        match="duplicate race-snapshot-driver rows",
    ):
        validate_feature_frame(malformed)


def test_validate_feature_frame_rejects_post_race_column() -> None:
    """Known post-race fields must never enter the feature frame."""
    features = make_features()

    features["Points"] = 25.0

    with pytest.raises(
        FeatureEngineeringError,
        match="prohibited post-race columns",
    ):
        validate_feature_frame(features)


def test_validate_feature_frame_rejects_invalid_field_size() -> None:
    """Field sizes must remain positive."""
    features = make_features()

    features.loc[
        0,
        "FieldSize",
    ] = 0

    with pytest.raises(
        FeatureEngineeringError,
        match="FieldSize must contain positive values",
    ):
        validate_feature_frame(features)


def test_validate_feature_frame_rejects_inconsistent_field_size() -> None:
    """One snapshot cannot advertise multiple field sizes."""
    features = make_features()

    features.loc[
        0,
        "FieldSize",
    ] = 99

    with pytest.raises(
        FeatureEngineeringError,
        match="constant within each race snapshot",
    ):
        validate_feature_frame(features)


@pytest.mark.parametrize(
    "invalid_value",
    [
        -0.1,
        1.1,
    ],
)
def test_validate_feature_frame_rejects_position_fraction_outside_range(
    invalid_value: float,
) -> None:
    """PositionFraction must remain inside the normalized range."""
    features = make_features()

    features.loc[
        0,
        "PositionFraction",
    ] = invalid_value

    with pytest.raises(
        FeatureEngineeringError,
        match="PositionFraction must remain between 0 and 1",
    ):
        validate_feature_frame(features)


@pytest.mark.parametrize(
    "invalid_value",
    [
        -0.1,
        1.1,
    ],
)
def test_validate_feature_frame_rejects_completion_fraction_outside_range(
    invalid_value: float,
) -> None:
    """CompletionFraction must remain inside the normalized range."""
    features = make_features()

    features.loc[
        0,
        "CompletionFraction",
    ] = invalid_value

    with pytest.raises(
        FeatureEngineeringError,
        match="CompletionFraction must remain between 0 and 1",
    ):
        validate_feature_frame(features)


def test_validate_feature_frame_requires_boolean_indicators() -> None:
    """Indicator features should use Boolean dtypes."""
    features = make_features()

    features["IsLapped"] = features["IsLapped"].astype("Int64")

    with pytest.raises(
        FeatureEngineeringError,
        match="IsLapped must use a Boolean dtype",
    ):
        validate_feature_frame(features)


def test_validate_feature_frame_rejects_missing_boolean() -> None:
    """Generated indicator features cannot contain unknown values."""
    features = make_features()

    features.loc[
        0,
        "IsTopThree",
    ] = pd.NA

    with pytest.raises(
        FeatureEngineeringError,
        match="IsTopThree cannot contain missing values",
    ):
        validate_feature_frame(features)


def test_validate_feature_frame_requires_zero_leader_delta() -> None:
    """A leader's known lap delta to itself must be zero."""
    features = make_features()

    leader_mask = features["IsLeader"] & features["LastLapDeltaToLeaderSeconds"].notna()

    leader_index = features.index[leader_mask][0]

    features.loc[
        leader_index,
        "LastLapDeltaToLeaderSeconds",
    ] = 1.0

    with pytest.raises(
        FeatureEngineeringError,
        match="leader with known lap time must have zero",
    ):
        validate_feature_frame(features)


def test_validate_feature_frame_rejects_missing_target() -> None:
    """The supervised target cannot be missing."""
    features = make_features()

    features.loc[
        0,
        TARGET_COLUMN,
    ] = pd.NA

    with pytest.raises(
        FeatureEngineeringError,
        match="WonRace cannot contain missing values",
    ):
        validate_feature_frame(features)


def test_validate_feature_frame_requires_boolean_target() -> None:
    """WonRace should remain a Boolean supervised label."""
    features = make_features()

    features[TARGET_COLUMN] = features[TARGET_COLUMN].astype("Int64")

    with pytest.raises(
        FeatureEngineeringError,
        match="WonRace must use a Boolean dtype",
    ):
        validate_feature_frame(features)


def test_validate_feature_frame_rejects_infinite_numeric_value() -> None:
    """Model features must never contain positive or negative infinity."""
    features = make_features()

    features.loc[
        0,
        "AverageLapTimeSeconds",
    ] = np.inf

    with pytest.raises(
        FeatureEngineeringError,
        match="contains non-finite numeric values",
    ):
        validate_feature_frame(features)
