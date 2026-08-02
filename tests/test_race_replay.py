from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from telemetryx.data.replay import (
    REPLAY_COLUMNS,
    RaceReplayError,
    build_race_replay,
    select_replay_snapshot,
)


def make_replay_laps() -> pd.DataFrame:
    """
    Return cleaned lap data for a small three-driver race.

    VER and NOR complete three laps. BOT completes only two laps, making BOT
    one lap behind the leader in the lap-three snapshot.
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
                1,
                2,
                1,
                1,
                2,
                3,
                1,
                2,
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


def test_build_replay_creates_one_row_per_snapshot_and_driver() -> None:
    """Every available driver should appear once in each leader snapshot."""
    replay = build_race_replay(make_replay_laps())

    assert tuple(replay.columns) == REPLAY_COLUMNS

    assert replay["SnapshotLap"].tolist() == [
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

    snapshot_sizes = replay.groupby("SnapshotLap").size().to_dict()

    assert snapshot_sizes == {
        1: 3,
        2: 3,
        3: 3,
    }

    duplicate_rows = replay.duplicated(
        subset=[
            "SnapshotLap",
            "Driver",
        ]
    )

    assert bool(duplicate_rows.any()) is False


def test_replay_does_not_modify_input_dataframe() -> None:
    """Replay construction must preserve its source lap table."""
    laps = make_replay_laps()
    original = laps.copy(deep=True)

    build_race_replay(laps)

    pd.testing.assert_frame_equal(
        laps,
        original,
    )


def test_snapshot_contains_latest_observed_driver_state() -> None:
    """Each snapshot should contain the latest row available for each driver."""
    replay = build_race_replay(make_replay_laps())

    snapshot = select_replay_snapshot(
        replay,
        snapshot_lap=2,
    )

    ver = snapshot.loc[snapshot["Driver"].eq("VER")].iloc[0]

    assert ver["SnapshotLap"] == 2
    assert ver["CompletedLaps"] == 2
    assert ver["Position"] == 1
    assert ver["Stint"] == 1
    assert ver["Compound"] == "SOFT"
    assert ver["TyreLife"] == pytest.approx(2.0)
    assert ver["LastLapTimeSeconds"] == pytest.approx(89.0)


def test_lapped_driver_retains_latest_known_state() -> None:
    """A lapped driver should remain present using their latest completed lap."""
    replay = build_race_replay(make_replay_laps())

    snapshot = select_replay_snapshot(
        replay,
        snapshot_lap=3,
    )

    bot = snapshot.loc[snapshot["Driver"].eq("BOT")].iloc[0]

    assert bot["SnapshotLap"] == 3
    assert bot["CompletedLaps"] == 2
    assert bot["LapsBehindLeader"] == 1
    assert bot["Position"] == 3
    assert bot["LastLapTimeSeconds"] == pytest.approx(91.0)
    assert bot["CumulativeLapTimeSeconds"] == pytest.approx(183.0)
    assert bool(bot["IsLeader"]) is False


def test_each_snapshot_contains_exactly_one_leader() -> None:
    """Every replay snapshot should identify exactly one current leader."""
    replay = build_race_replay(make_replay_laps())

    leader_counts = (
        replay.loc[replay["IsLeader"].fillna(False)]
        .groupby("SnapshotLap")
        .size()
        .to_dict()
    )

    assert leader_counts == {
        1: 1,
        2: 1,
        3: 1,
    }


def test_cumulative_lap_times_are_calculated_per_driver() -> None:
    """Cumulative times should use only each driver's completed laps."""
    replay = build_race_replay(make_replay_laps())

    snapshot = select_replay_snapshot(
        replay,
        snapshot_lap=3,
    )

    cumulative_times = dict(
        zip(
            snapshot["Driver"],
            snapshot["CumulativeLapTimeSeconds"],
            strict=True,
        )
    )

    assert cumulative_times["VER"] == pytest.approx(267.0)
    assert cumulative_times["NOR"] == pytest.approx(270.0)
    assert cumulative_times["BOT"] == pytest.approx(183.0)


def test_missing_lap_time_invalidates_later_cumulative_time() -> None:
    """A missing historical lap must not be treated as a zero-second lap."""
    laps = make_replay_laps()

    missing_mask = laps["Driver"].eq("NOR") & laps["LapNumber"].eq(2)

    laps.loc[
        missing_mask,
        "LapTimeSeconds",
    ] = None

    replay = build_race_replay(laps)

    lap_two_snapshot = select_replay_snapshot(
        replay,
        snapshot_lap=2,
    )

    lap_three_snapshot = select_replay_snapshot(
        replay,
        snapshot_lap=3,
    )

    nor_lap_two = lap_two_snapshot.loc[lap_two_snapshot["Driver"].eq("NOR")].iloc[0]

    nor_lap_three = lap_three_snapshot.loc[lap_three_snapshot["Driver"].eq("NOR")].iloc[
        0
    ]

    assert pd.isna(nor_lap_two["LastLapTimeSeconds"])

    assert pd.isna(nor_lap_two["CumulativeLapTimeSeconds"])

    assert nor_lap_three["LastLapTimeSeconds"] == pytest.approx(89.0)

    assert pd.isna(nor_lap_three["CumulativeLapTimeSeconds"])


def test_future_lap_changes_do_not_affect_earlier_snapshots() -> None:
    """Changing lap-three values must not alter snapshots through lap two."""
    original_laps = make_replay_laps()
    modified_laps = original_laps.copy(deep=True)

    ver_future_row = modified_laps["Driver"].eq("VER") & modified_laps["LapNumber"].eq(
        3
    )

    nor_future_row = modified_laps["Driver"].eq("NOR") & modified_laps["LapNumber"].eq(
        3
    )

    modified_laps.loc[
        ver_future_row,
        [
            "Position",
            "Stint",
            "Compound",
            "TyreLife",
            "TrackStatus",
            "LapTimeSeconds",
        ],
    ] = [
        2,
        4,
        "HARD",
        50,
        "5",
        500.0,
    ]

    modified_laps.loc[
        nor_future_row,
        "Position",
    ] = 1

    original_replay = build_race_replay(
        original_laps,
        end_lap=2,
    )

    modified_replay = build_race_replay(
        modified_laps,
        end_lap=2,
    )

    pd.testing.assert_frame_equal(
        original_replay,
        modified_replay,
    )


def test_replay_never_contains_laps_beyond_snapshot_cutoff() -> None:
    """Completed laps must always be less than or equal to the snapshot lap."""
    replay = build_race_replay(make_replay_laps())

    future_rows = replay[replay["CompletedLaps"] > replay["SnapshotLap"]]

    assert future_rows.empty

    assert (replay["DataAvailableThroughLap"] == replay["SnapshotLap"]).all()


def test_requested_lap_range_is_applied() -> None:
    """The start and end boundaries should restrict replay snapshots."""
    replay = build_race_replay(
        make_replay_laps(),
        start_lap=2,
        end_lap=2,
    )

    assert replay["SnapshotLap"].unique().tolist() == [2]

    assert len(replay) == 3


def test_input_row_order_does_not_change_replay_output() -> None:
    """Replay output should be deterministic for shuffled source rows."""
    laps = make_replay_laps()

    shuffled = laps.sample(
        frac=1.0,
        random_state=42,
    ).reset_index(drop=True)

    expected = build_race_replay(laps)
    actual = build_race_replay(shuffled)

    pd.testing.assert_frame_equal(
        actual,
        expected,
    )


def test_driver_values_are_normalized() -> None:
    """Driver identifiers should be trimmed and converted to uppercase."""
    laps = make_replay_laps()

    laps.loc[
        laps["Driver"].eq("VER"),
        "Driver",
    ] = " ver "

    replay = build_race_replay(laps)

    assert set(replay["Driver"]) == {
        "VER",
        "NOR",
        "BOT",
    }


def test_optional_text_values_are_normalized() -> None:
    """Compound and track-status values should be trimmed safely."""
    laps = make_replay_laps()

    target_row = laps["Driver"].eq("VER") & laps["LapNumber"].eq(1)

    laps.loc[
        target_row,
        "Compound",
    ] = " soft "

    laps.loc[
        target_row,
        "TrackStatus",
    ] = " 1 "

    replay = build_race_replay(
        laps,
        end_lap=1,
    )

    ver = replay.loc[replay["Driver"].eq("VER")].iloc[0]

    assert ver["Compound"] == "SOFT"
    assert ver["TrackStatus"] == "1"


def test_select_snapshot_returns_independent_copy() -> None:
    """Changing a selected snapshot must not modify the full replay."""
    replay = build_race_replay(make_replay_laps())

    snapshot = select_replay_snapshot(
        replay,
        snapshot_lap=2,
    )

    snapshot.loc[
        snapshot["Driver"].eq("VER"),
        "Position",
    ] = 99

    original_position = replay.loc[
        replay["SnapshotLap"].eq(2) & replay["Driver"].eq("VER"),
        "Position",
    ].iloc[0]

    assert original_position == 1


def test_non_dataframe_input_is_rejected() -> None:
    """Replay construction should reject non-DataFrame input."""
    invalid_input: Any = [
        {
            "Driver": "VER",
            "LapNumber": 1,
        }
    ]

    with pytest.raises(
        TypeError,
        match="laps must be provided as a pandas DataFrame",
    ):
        build_race_replay(invalid_input)


def test_empty_lap_table_is_rejected() -> None:
    """An empty table cannot produce race snapshots."""
    laps = make_replay_laps().iloc[0:0]

    with pytest.raises(
        RaceReplayError,
        match="empty lap table",
    ):
        build_race_replay(laps)


def test_missing_required_columns_are_reported() -> None:
    """Replay construction should identify required missing columns."""
    laps = make_replay_laps().drop(
        columns=[
            "LapTimeSeconds",
            "TrackStatus",
        ]
    )

    with pytest.raises(
        RaceReplayError,
        match="missing required columns",
    ) as exception_info:
        build_race_replay(laps)

    message = str(exception_info.value)

    assert "LapTimeSeconds" in message
    assert "TrackStatus" in message


def test_duplicate_driver_lap_rows_are_rejected() -> None:
    """A driver cannot have multiple records for the same completed lap."""
    laps = make_replay_laps()

    duplicate = laps.iloc[[0]].copy()

    laps = pd.concat(
        [
            laps,
            duplicate,
        ],
        ignore_index=True,
    )

    with pytest.raises(
        RaceReplayError,
        match="duplicate Driver and LapNumber",
    ):
        build_race_replay(laps)


@pytest.mark.parametrize(
    ("column", "invalid_value", "expected_message"),
    [
        (
            "LapNumber",
            0,
            "LapNumber must contain positive values",
        ),
        (
            "LapNumber",
            1.5,
            "LapNumber must contain whole numbers",
        ),
        (
            "LapNumber",
            "invalid",
            "LapNumber contains non-numeric values",
        ),
        (
            "Position",
            -1,
            "Position must contain positive values",
        ),
        (
            "Stint",
            1.5,
            "Stint must contain whole numbers",
        ),
        (
            "TyreLife",
            -1,
            "TyreLife contains negative values",
        ),
        (
            "LapTimeSeconds",
            "invalid",
            "LapTimeSeconds contains non-numeric values",
        ),
        (
            "LapTimeSeconds",
            0,
            "LapTimeSeconds must be greater than zero",
        ),
    ],
)
def test_invalid_numeric_values_are_rejected(
    column: str,
    invalid_value: object,
    expected_message: str,
) -> None:
    """Replay numeric fields must satisfy their documented constraints."""
    laps = make_replay_laps()
    laps[column] = laps[column].astype("object")
    laps.loc[0, column] = invalid_value

    with pytest.raises(
        RaceReplayError,
        match=expected_message,
    ):
        build_race_replay(laps)


@pytest.mark.parametrize(
    "invalid_driver",
    [
        None,
        "",
        "   ",
    ],
)
def test_missing_driver_values_are_rejected(
    invalid_driver: object,
) -> None:
    """Every replay row must have a usable driver identifier."""
    laps = make_replay_laps()
    laps["Driver"] = laps["Driver"].astype("object")
    laps.loc[0, "Driver"] = invalid_driver

    with pytest.raises(
        RaceReplayError,
        match="Driver contains missing or blank values",
    ):
        build_race_replay(laps)


def test_snapshot_with_multiple_leaders_is_rejected() -> None:
    """Each completed leader lap must identify exactly one leader."""
    laps = make_replay_laps()

    second_leader = laps["Driver"].eq("NOR") & laps["LapNumber"].eq(1)

    laps.loc[
        second_leader,
        "Position",
    ] = 1

    with pytest.raises(
        RaceReplayError,
        match="exactly one leader",
    ):
        build_race_replay(
            laps,
            end_lap=1,
        )


def test_range_without_completed_leader_laps_is_rejected() -> None:
    """A requested interval must contain at least one leader lap."""
    with pytest.raises(
        RaceReplayError,
        match="No completed leader laps",
    ):
        build_race_replay(
            make_replay_laps(),
            start_lap=10,
        )


@pytest.mark.parametrize(
    ("start_lap", "end_lap", "expected_message"),
    [
        (
            0,
            None,
            "start_lap must be a positive integer",
        ),
        (
            -1,
            None,
            "start_lap must be a positive integer",
        ),
        (
            2,
            0,
            "end_lap must be a positive integer",
        ),
        (
            3,
            2,
            "end_lap cannot be smaller than start_lap",
        ),
    ],
)
def test_invalid_replay_ranges_are_rejected(
    start_lap: int,
    end_lap: int | None,
    expected_message: str,
) -> None:
    """Replay boundaries must describe a valid positive interval."""
    with pytest.raises(
        RaceReplayError,
        match=expected_message,
    ):
        build_race_replay(
            make_replay_laps(),
            start_lap=start_lap,
            end_lap=end_lap,
        )


def test_select_snapshot_rejects_unavailable_lap() -> None:
    """Selecting a lap absent from the replay should fail clearly."""
    replay = build_race_replay(make_replay_laps())

    with pytest.raises(
        RaceReplayError,
        match="snapshot lap 10 is unavailable",
    ):
        select_replay_snapshot(
            replay,
            snapshot_lap=10,
        )


def test_select_snapshot_rejects_invalid_lap() -> None:
    """Snapshot selection requires a positive lap number."""
    replay = build_race_replay(make_replay_laps())

    with pytest.raises(
        RaceReplayError,
        match="snapshot_lap must be a positive integer",
    ):
        select_replay_snapshot(
            replay,
            snapshot_lap=0,
        )


def test_select_snapshot_requires_replay_columns() -> None:
    """Snapshot selection should reject incomplete replay tables."""
    incomplete_replay = pd.DataFrame(
        {
            "SnapshotLap": [1],
            "Driver": ["VER"],
        }
    )

    with pytest.raises(
        RaceReplayError,
        match="missing required columns",
    ):
        select_replay_snapshot(
            incomplete_replay,
            snapshot_lap=1,
        )
