from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from telemetryx.data.dataset import (
    DISALLOWED_POST_RACE_COLUMNS,
    RACE_DATASET_COLUMNS,
    RACE_DATASET_KEY_COLUMNS,
    RaceDatasetError,
    RaceMetadata,
    build_race_dataset,
    validate_race_dataset,
)
from telemetryx.data.targets import TARGET_COLUMN


def make_cleaned_laps() -> pd.DataFrame:
    """
    Return cleaned lap data for a small three-driver race.

    VER and NOR complete three laps. BOT completes two laps and is therefore
    one lap behind in the third replay snapshot.
    """
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
            ],
            "Stint": [
                1,
                1,
                2,
                1,
                1,
                1,
                1,
                1,
            ],
            "Compound": [
                "SOFT",
                "SOFT",
                "MEDIUM",
                "SOFT",
                "SOFT",
                "SOFT",
                "SOFT",
                "SOFT",
            ],
            "TyreLife": [
                1.0,
                2.0,
                1.0,
                1.0,
                2.0,
                3.0,
                1.0,
                2.0,
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
            ],
        }
    )


def make_results() -> pd.DataFrame:
    """Return final results containing one winner and extra result fields."""
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
                "+1 Lap",
            ],
            "Points": [
                25.0,
                18.0,
                15.0,
            ],
            "GridPosition": [
                1,
                2,
                3,
            ],
        }
    )


def make_race_dataset() -> pd.DataFrame:
    """Return a valid assembled race dataset."""
    return build_race_dataset(
        make_cleaned_laps(),
        make_results(),
        season=2024,
        round_number=1,
        event_name="Bahrain Grand Prix",
        session_name="Race",
    )


def test_build_race_dataset_uses_documented_schema() -> None:
    """The assembled dataset should use the deterministic column schema."""
    dataset = make_race_dataset()

    assert tuple(dataset.columns) == RACE_DATASET_COLUMNS
    assert len(dataset) == 9

    assert dataset["SnapshotLap"].tolist() == [
        1,
        1,
        1,
        2,
        2,
        2,
        3,
        3,
        3,
    ]


def test_build_race_dataset_adds_normalized_metadata() -> None:
    """Race metadata should be normalized and repeated on every row."""
    dataset = build_race_dataset(
        make_cleaned_laps(),
        make_results(),
        season=2024,
        round_number=1,
        event_name="  Bahrain Grand Prix  ",
        session_name="  Race  ",
    )

    assert dataset["RaceId"].unique().tolist() == ["2024_01_bahrain_grand_prix"]

    assert dataset["Season"].unique().tolist() == [2024]

    assert dataset["RoundNumber"].unique().tolist() == [1]

    assert dataset["EventName"].unique().tolist() == ["Bahrain Grand Prix"]

    assert dataset["SessionName"].unique().tolist() == ["Race"]


def test_race_metadata_creates_deterministic_identifier() -> None:
    """Equivalent metadata should always produce the same race identifier."""
    metadata = RaceMetadata(
        season=2024,
        round_number=21,
        event_name="  São Paulo Grand Prix!  ",
        session_name=" Race ",
    )

    assert metadata.event_name == "São Paulo Grand Prix!"
    assert metadata.session_name == "Race"

    assert metadata.race_id == ("2024_21_são_paulo_grand_prix")


def test_dataset_has_unique_race_snapshot_driver_keys() -> None:
    """Each driver should appear once per race snapshot."""
    dataset = make_race_dataset()

    duplicate_keys = dataset.duplicated(
        subset=list(RACE_DATASET_KEY_COLUMNS),
        keep=False,
    )

    assert bool(duplicate_keys.any()) is False

    assert tuple(RACE_DATASET_KEY_COLUMNS) == (
        "RaceId",
        "SnapshotLap",
        "Driver",
    )


def test_dataset_contains_one_positive_target_per_snapshot() -> None:
    """Every snapshot should contain exactly one eventual race winner."""
    dataset = make_race_dataset()

    winner_counts = (
        dataset.groupby(
            [
                "RaceId",
                "SnapshotLap",
            ]
        )[TARGET_COLUMN]
        .sum()
        .to_dict()
    )

    assert winner_counts == {
        (
            "2024_01_bahrain_grand_prix",
            1,
        ): 1,
        (
            "2024_01_bahrain_grand_prix",
            2,
        ): 1,
        (
            "2024_01_bahrain_grand_prix",
            3,
        ): 1,
    }


def test_dataset_retains_lapped_driver_state() -> None:
    """A lapped driver should remain represented by their latest state."""
    dataset = make_race_dataset()

    bot_lap_three = dataset.loc[
        dataset["SnapshotLap"].eq(3) & dataset["Driver"].eq("BOT")
    ].iloc[0]

    assert bot_lap_three["CompletedLaps"] == 2
    assert bot_lap_three["LapsBehindLeader"] == 1
    assert bot_lap_three["LastLapTimeSeconds"] == pytest.approx(91.0)

    assert bool(bot_lap_three[TARGET_COLUMN]) is False


def test_post_race_result_columns_do_not_enter_dataset() -> None:
    """Final result fields must not leak into model observations."""
    dataset = make_race_dataset()

    assert "FinalPosition" not in dataset.columns
    assert "ClassifiedPosition" not in dataset.columns
    assert "Points" not in dataset.columns
    assert "Status" not in dataset.columns
    assert "GridPosition" not in dataset.columns

    assert DISALLOWED_POST_RACE_COLUMNS.intersection(set(dataset.columns)) == set()


def test_build_race_dataset_preserves_input_dataframes() -> None:
    """Dataset construction must not modify lap or result inputs."""
    laps = make_cleaned_laps()
    results = make_results()

    original_laps = laps.copy(deep=True)
    original_results = results.copy(deep=True)

    build_race_dataset(
        laps,
        results,
        season=2024,
        round_number=1,
        event_name="Bahrain Grand Prix",
    )

    pd.testing.assert_frame_equal(
        laps,
        original_laps,
    )

    pd.testing.assert_frame_equal(
        results,
        original_results,
    )


def test_requested_lap_range_is_applied() -> None:
    """Dataset construction should forward replay lap boundaries."""
    dataset = build_race_dataset(
        make_cleaned_laps(),
        make_results(),
        season=2024,
        round_number=1,
        event_name="Bahrain Grand Prix",
        start_lap=2,
        end_lap=2,
    )

    assert dataset["SnapshotLap"].unique().tolist() == [2]

    assert len(dataset) == 3

    assert int(dataset[TARGET_COLUMN].sum()) == 1


@pytest.mark.parametrize(
    ("season", "round_number", "expected_message"),
    [
        (
            0,
            1,
            "season must be a positive integer",
        ),
        (
            -1,
            1,
            "season must be a positive integer",
        ),
        (
            True,
            1,
            "season must be a positive integer",
        ),
        (
            2024,
            0,
            "round_number must be a positive integer",
        ),
        (
            2024,
            -1,
            "round_number must be a positive integer",
        ),
        (
            2024,
            False,
            "round_number must be a positive integer",
        ),
    ],
)
def test_invalid_numeric_metadata_is_rejected(
    season: int,
    round_number: int,
    expected_message: str,
) -> None:
    """Season and round number must be positive integers."""
    with pytest.raises(
        RaceDatasetError,
        match=expected_message,
    ):
        RaceMetadata(
            season=season,
            round_number=round_number,
            event_name="Bahrain Grand Prix",
        )


@pytest.mark.parametrize(
    ("event_name", "session_name", "expected_message"),
    [
        (
            "",
            "Race",
            "event_name cannot be blank",
        ),
        (
            "   ",
            "Race",
            "event_name cannot be blank",
        ),
        (
            "Bahrain Grand Prix",
            "",
            "session_name cannot be blank",
        ),
        (
            "Bahrain Grand Prix",
            "   ",
            "session_name cannot be blank",
        ),
    ],
)
def test_blank_text_metadata_is_rejected(
    event_name: str,
    session_name: str,
    expected_message: str,
) -> None:
    """Race names and session names cannot be blank."""
    with pytest.raises(
        RaceDatasetError,
        match=expected_message,
    ):
        RaceMetadata(
            season=2024,
            round_number=1,
            event_name=event_name,
            session_name=session_name,
        )


def test_non_string_metadata_is_rejected() -> None:
    """Race text metadata must use strings."""
    invalid_event_name: Any = 123

    with pytest.raises(
        RaceDatasetError,
        match="event_name must be a string",
    ):
        RaceMetadata(
            season=2024,
            round_number=1,
            event_name=invalid_event_name,
        )


def test_build_race_dataset_rejects_non_dataframe_inputs() -> None:
    """Dataset construction requires lap and result DataFrames."""
    invalid_laps: Any = []
    invalid_results: Any = []

    with pytest.raises(
        TypeError,
        match="laps must be provided as a pandas DataFrame",
    ):
        build_race_dataset(
            invalid_laps,
            make_results(),
            season=2024,
            round_number=1,
            event_name="Bahrain Grand Prix",
        )

    with pytest.raises(
        TypeError,
        match="results must be provided as a pandas DataFrame",
    ):
        build_race_dataset(
            make_cleaned_laps(),
            invalid_results,
            season=2024,
            round_number=1,
            event_name="Bahrain Grand Prix",
        )


def test_validate_race_dataset_accepts_valid_dataset() -> None:
    """A correctly assembled race dataset should pass validation."""
    validate_race_dataset(make_race_dataset())


def test_validate_race_dataset_rejects_non_dataframe() -> None:
    """Race-dataset validation requires a pandas DataFrame."""
    invalid_dataset: Any = []

    with pytest.raises(
        TypeError,
        match="dataset must be provided as a pandas DataFrame",
    ):
        validate_race_dataset(invalid_dataset)


def test_validate_race_dataset_rejects_empty_table() -> None:
    """An empty race dataset cannot be used for model training."""
    dataset = make_race_dataset().iloc[0:0]

    with pytest.raises(
        RaceDatasetError,
        match="contains no rows",
    ):
        validate_race_dataset(dataset)


def test_validate_race_dataset_reports_missing_columns() -> None:
    """Missing schema columns should be identified clearly."""
    dataset = make_race_dataset().drop(
        columns=[
            "CumulativeLapTimeSeconds",
            "TrackStatus",
        ]
    )

    with pytest.raises(
        RaceDatasetError,
        match="missing required columns",
    ) as exception_info:
        validate_race_dataset(dataset)

    message = str(exception_info.value)

    assert "CumulativeLapTimeSeconds" in message
    assert "TrackStatus" in message


def test_validate_race_dataset_rejects_duplicate_columns() -> None:
    """Ambiguous duplicate dataset columns should be rejected."""
    dataset = make_race_dataset()

    dataset = pd.concat(
        [
            dataset,
            dataset[["RaceId"]],
        ],
        axis=1,
    )

    with pytest.raises(
        RaceDatasetError,
        match="duplicate column names",
    ):
        validate_race_dataset(dataset)


@pytest.mark.parametrize(
    "column",
    sorted(DISALLOWED_POST_RACE_COLUMNS),
)
def test_validate_race_dataset_rejects_post_race_columns(
    column: str,
) -> None:
    """Prohibited final-result columns should trigger leakage protection."""
    dataset = make_race_dataset()
    dataset[column] = 1

    with pytest.raises(
        RaceDatasetError,
        match="prohibited post-race columns",
    ):
        validate_race_dataset(dataset)


def test_validate_race_dataset_rejects_duplicate_keys() -> None:
    """Race, snapshot and driver keys must uniquely identify rows."""
    dataset = make_race_dataset()

    duplicated_row = dataset.iloc[[0]].copy()

    dataset = pd.concat(
        [
            dataset,
            duplicated_row,
        ],
        ignore_index=True,
    )

    with pytest.raises(
        RaceDatasetError,
        match="duplicate race-snapshot-driver rows",
    ):
        validate_race_dataset(dataset)


def test_validate_race_dataset_rejects_missing_metadata() -> None:
    """Every dataset row must contain complete race identity metadata."""
    dataset = make_race_dataset()

    dataset["EventName"] = dataset["EventName"].astype("object")

    dataset.loc[
        0,
        "EventName",
    ] = None

    with pytest.raises(
        RaceDatasetError,
        match="EventName contains missing race metadata",
    ):
        validate_race_dataset(dataset)


def test_validate_race_dataset_rejects_inconsistent_metadata() -> None:
    """One race dataset cannot contain multiple event identities."""
    dataset = make_race_dataset()

    dataset.loc[
        0,
        "EventName",
    ] = "Different Grand Prix"

    with pytest.raises(
        RaceDatasetError,
        match="EventName must contain exactly one value",
    ):
        validate_race_dataset(dataset)


def test_validate_race_dataset_rejects_future_completed_laps() -> None:
    """A row cannot use a completed lap beyond its snapshot cutoff."""
    dataset = make_race_dataset()

    dataset.loc[
        0,
        "CompletedLaps",
    ] = (
        int(
            dataset.loc[
                0,
                "SnapshotLap",
            ]
        )
        + 1
    )

    with pytest.raises(
        RaceDatasetError,
        match="completed laps beyond their snapshot cutoff",
    ):
        validate_race_dataset(dataset)


def test_validate_race_dataset_rejects_invalid_availability_cutoff() -> None:
    """Availability metadata must match the row's snapshot lap."""
    dataset = make_race_dataset()

    dataset.loc[
        0,
        "DataAvailableThroughLap",
    ] = 99

    with pytest.raises(
        RaceDatasetError,
        match="DataAvailableThroughLap must equal SnapshotLap",
    ):
        validate_race_dataset(dataset)


def test_validate_race_dataset_rejects_inconsistent_lap_deficit() -> None:
    """Lap deficit must equal snapshot lap minus completed laps."""
    dataset = make_race_dataset()

    dataset.loc[
        0,
        "LapsBehindLeader",
    ] = 5

    with pytest.raises(
        RaceDatasetError,
        match="LapsBehindLeader is inconsistent",
    ):
        validate_race_dataset(dataset)


def test_validate_race_dataset_rejects_missing_target() -> None:
    """Winner targets must be complete on every row."""
    dataset = make_race_dataset()

    dataset.loc[
        0,
        TARGET_COLUMN,
    ] = pd.NA

    with pytest.raises(
        RaceDatasetError,
        match="WonRace contains missing values",
    ):
        validate_race_dataset(dataset)


def test_validate_race_dataset_rejects_non_boolean_target() -> None:
    """Winner targets must use a Boolean dtype."""
    dataset = make_race_dataset()

    dataset[TARGET_COLUMN] = dataset[TARGET_COLUMN].astype("Int64")

    with pytest.raises(
        RaceDatasetError,
        match="WonRace must use a Boolean dtype",
    ):
        validate_race_dataset(dataset)


def test_validate_race_dataset_rejects_snapshot_without_winner() -> None:
    """Each snapshot must contain exactly one positive winner label."""
    dataset = make_race_dataset()

    missing_winner = dataset["SnapshotLap"].eq(2) & dataset["Driver"].eq("VER")

    dataset.loc[
        missing_winner,
        TARGET_COLUMN,
    ] = False

    with pytest.raises(
        RaceDatasetError,
        match="exactly one winner target",
    ):
        validate_race_dataset(dataset)


def test_validate_race_dataset_rejects_snapshot_with_two_winners() -> None:
    """A snapshot cannot contain two positive winner labels."""
    dataset = make_race_dataset()

    second_winner = dataset["SnapshotLap"].eq(2) & dataset["Driver"].eq("NOR")

    dataset.loc[
        second_winner,
        TARGET_COLUMN,
    ] = True

    with pytest.raises(
        RaceDatasetError,
        match="exactly one winner target",
    ):
        validate_race_dataset(dataset)
