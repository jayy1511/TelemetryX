from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from telemetryx.data.targets import (
    TARGET_COLUMN,
    WINNER_TARGET_COLUMNS,
    RaceTargetError,
    attach_winner_targets,
    build_winner_targets,
)


def make_results() -> pd.DataFrame:
    """Return a minimal FastF1-like final results table."""
    return pd.DataFrame(
        {
            "Abbreviation": [
                "VER",
                "NOR",
                "BOT",
            ],
            "Position": [
                1.0,
                2.0,
                3.0,
            ],
            "Status": [
                "Finished",
                "Finished",
                "Finished",
            ],
        }
    )


def make_replay() -> pd.DataFrame:
    """Return a small temporally safe replay table."""
    return pd.DataFrame(
        {
            "SnapshotLap": [
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
            "Driver": [
                "VER",
                "NOR",
                "BOT",
                "VER",
                "NOR",
                "BOT",
                "VER",
                "NOR",
                "BOT",
            ],
            "Position": [
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
            "CompletedLaps": [
                1,
                1,
                1,
                2,
                2,
                2,
                3,
                3,
                2,
            ],
        }
    )


def test_build_targets_creates_one_binary_target_per_driver() -> None:
    """Final positions should become one Boolean winner target per driver."""
    targets = build_winner_targets(make_results())

    assert tuple(targets.columns) == WINNER_TARGET_COLUMNS

    assert targets["Driver"].tolist() == [
        "VER",
        "NOR",
        "BOT",
    ]

    assert targets[TARGET_COLUMN].tolist() == [
        True,
        False,
        False,
    ]

    assert str(targets[TARGET_COLUMN].dtype) == "boolean"


def test_final_position_is_not_returned_as_a_target_feature() -> None:
    """Post-race finishing position must not leak into the target table."""
    targets = build_winner_targets(make_results())

    assert "Position" not in targets.columns
    assert "Status" not in targets.columns

    assert tuple(targets.columns) == (
        "Driver",
        "WonRace",
    )


def test_build_targets_does_not_modify_results() -> None:
    """Target construction must preserve the original results table."""
    results = make_results()
    original = results.copy(deep=True)

    build_winner_targets(results)

    pd.testing.assert_frame_equal(
        results,
        original,
    )


def test_driver_identifiers_are_trimmed_and_normalized() -> None:
    """Result driver identifiers should become uppercase abbreviations."""
    results = make_results()

    results["Abbreviation"] = [
        " ver ",
        " nor ",
        " bot ",
    ]

    targets = build_winner_targets(results)

    assert targets["Driver"].tolist() == [
        "VER",
        "NOR",
        "BOT",
    ]


def test_driver_column_is_supported_as_fallback() -> None:
    """Compatible results may use Driver instead of Abbreviation."""
    results = make_results().rename(
        columns={
            "Abbreviation": "Driver",
        }
    )

    targets = build_winner_targets(results)

    assert targets["Driver"].tolist() == [
        "VER",
        "NOR",
        "BOT",
    ]

    assert targets[TARGET_COLUMN].sum() == 1


def test_non_dataframe_results_are_rejected() -> None:
    """Winner targets require a pandas DataFrame."""
    invalid_results: Any = [
        {
            "Abbreviation": "VER",
            "Position": 1,
        }
    ]

    with pytest.raises(
        TypeError,
        match="results must be provided as a pandas DataFrame",
    ):
        build_winner_targets(invalid_results)


def test_empty_results_are_rejected() -> None:
    """An empty result table cannot identify a race winner."""
    with pytest.raises(
        RaceTargetError,
        match="empty results table",
    ):
        build_winner_targets(pd.DataFrame())


def test_missing_driver_column_is_rejected() -> None:
    """Results must contain a supported driver identifier column."""
    results = make_results().rename(
        columns={
            "Abbreviation": "DriverCode",
        }
    )

    with pytest.raises(
        RaceTargetError,
        match="no supported driver column",
    ):
        build_winner_targets(results)


def test_missing_position_column_is_rejected() -> None:
    """Final position is required to identify the winner."""
    results = make_results().drop(columns=["Position"])

    with pytest.raises(
        RaceTargetError,
        match="missing required columns",
    ):
        build_winner_targets(results)


def test_duplicate_result_columns_are_rejected() -> None:
    """Ambiguous duplicate result columns must not be accepted."""
    results = make_results()

    results.columns = [
        "Abbreviation",
        "Position",
        "Position",
    ]

    with pytest.raises(
        RaceTargetError,
        match="duplicate column names",
    ):
        build_winner_targets(results)


def test_duplicate_result_drivers_are_rejected() -> None:
    """Each driver may appear only once in the final results."""
    results = make_results()

    results.loc[
        2,
        "Abbreviation",
    ] = "VER"

    with pytest.raises(
        RaceTargetError,
        match="duplicate driver rows",
    ):
        build_winner_targets(results)


@pytest.mark.parametrize(
    ("invalid_driver", "expected_message"),
    [
        (
            None,
            "missing or blank driver identifiers",
        ),
        (
            "",
            "missing or blank driver identifiers",
        ),
        (
            "   ",
            "missing or blank driver identifiers",
        ),
    ],
)
def test_missing_result_driver_values_are_rejected(
    invalid_driver: object,
    expected_message: str,
) -> None:
    """Every final result row must contain a usable driver identifier."""
    results = make_results()
    results["Abbreviation"] = results["Abbreviation"].astype("object")

    results.loc[
        1,
        "Abbreviation",
    ] = invalid_driver

    with pytest.raises(
        RaceTargetError,
        match=expected_message,
    ):
        build_winner_targets(results)


@pytest.mark.parametrize(
    ("invalid_position", "expected_message"),
    [
        (
            None,
            "Position contains missing values",
        ),
        (
            "not-a-position",
            "Position contains non-numeric values",
        ),
        (
            0,
            "Position must contain positive values",
        ),
        (
            -1,
            "Position must contain positive values",
        ),
        (
            2.5,
            "Position must contain whole numbers",
        ),
    ],
)
def test_invalid_final_positions_are_rejected(
    invalid_position: object,
    expected_message: str,
) -> None:
    """Final positions must be present, numeric, positive, and whole."""
    results = make_results()
    results["Position"] = results["Position"].astype("object")

    results.loc[
        1,
        "Position",
    ] = invalid_position

    with pytest.raises(
        RaceTargetError,
        match=expected_message,
    ):
        build_winner_targets(results)


def test_results_without_winner_are_rejected() -> None:
    """A result table must contain one position-one driver."""
    results = make_results()

    results["Position"] = [
        2,
        3,
        4,
    ]

    with pytest.raises(
        RaceTargetError,
        match="exactly one position-one driver; found 0",
    ):
        build_winner_targets(results)


def test_results_with_multiple_winners_are_rejected() -> None:
    """Two position-one drivers cannot define a valid winner target."""
    results = make_results()

    results["Position"] = [
        1,
        1,
        3,
    ]

    with pytest.raises(
        RaceTargetError,
        match="exactly one position-one driver; found 2",
    ):
        build_winner_targets(results)


def test_attach_targets_adds_winner_to_every_replay_row() -> None:
    """Each replay row should receive its driver's final winner label."""
    replay = make_replay()

    targets = build_winner_targets(make_results())

    training_rows = attach_winner_targets(
        replay,
        targets,
    )

    assert TARGET_COLUMN in training_rows.columns
    assert len(training_rows) == len(replay)

    ver_targets = training_rows.loc[
        training_rows["Driver"].eq("VER"),
        TARGET_COLUMN,
    ]

    nor_targets = training_rows.loc[
        training_rows["Driver"].eq("NOR"),
        TARGET_COLUMN,
    ]

    bot_targets = training_rows.loc[
        training_rows["Driver"].eq("BOT"),
        TARGET_COLUMN,
    ]

    assert ver_targets.tolist() == [
        True,
        True,
        True,
    ]

    assert nor_targets.tolist() == [
        False,
        False,
        False,
    ]

    assert bot_targets.tolist() == [
        False,
        False,
        False,
    ]


def test_each_snapshot_contains_exactly_one_positive_target() -> None:
    """Every replay snapshot should contain the eventual winner once."""
    training_rows = attach_winner_targets(
        make_replay(),
        build_winner_targets(make_results()),
    )

    winner_counts = training_rows.groupby("SnapshotLap")[TARGET_COLUMN].sum().to_dict()

    assert winner_counts == {
        1: 1,
        2: 1,
        3: 1,
    }


def test_attach_targets_preserves_source_dataframes() -> None:
    """Target attachment must not mutate replay or target inputs."""
    replay = make_replay()
    targets = build_winner_targets(make_results())

    original_replay = replay.copy(deep=True)
    original_targets = targets.copy(deep=True)

    attach_winner_targets(
        replay,
        targets,
    )

    pd.testing.assert_frame_equal(
        replay,
        original_replay,
    )

    pd.testing.assert_frame_equal(
        targets,
        original_targets,
    )


def test_attach_targets_normalizes_driver_identifiers() -> None:
    """Equivalent driver identifiers should match after normalization."""
    replay = make_replay()

    replay["Driver"] = replay["Driver"].replace(
        {
            "VER": " ver ",
            "NOR": "nor",
            "BOT": " BOT ",
        }
    )

    targets = pd.DataFrame(
        {
            "Driver": [
                "ver",
                " nor ",
                "bot",
            ],
            TARGET_COLUMN: [
                True,
                False,
                False,
            ],
        }
    )

    training_rows = attach_winner_targets(
        replay,
        targets,
    )

    assert set(training_rows["Driver"]) == {
        "VER",
        "NOR",
        "BOT",
    }

    assert int(training_rows[TARGET_COLUMN].sum()) == 3


def test_missing_target_driver_is_rejected() -> None:
    """Every replay driver must have a corresponding winner target."""
    replay = make_replay()

    targets = pd.DataFrame(
        {
            "Driver": [
                "VER",
                "NOR",
            ],
            TARGET_COLUMN: [
                True,
                False,
            ],
        }
    )

    with pytest.raises(
        RaceTargetError,
        match="missing for replay drivers: BOT",
    ):
        attach_winner_targets(
            replay,
            targets,
        )


def test_duplicate_target_drivers_are_rejected() -> None:
    """A target table may contain only one row per driver."""
    targets = pd.DataFrame(
        {
            "Driver": [
                "VER",
                "VER",
                "NOR",
                "BOT",
            ],
            TARGET_COLUMN: [
                True,
                False,
                False,
                False,
            ],
        }
    )

    with pytest.raises(
        RaceTargetError,
        match="duplicate drivers",
    ):
        attach_winner_targets(
            make_replay(),
            targets,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        1,
        0,
        "True",
        "False",
    ],
)
def test_non_boolean_target_values_are_rejected(
    invalid_value: object,
) -> None:
    """Winner targets must contain actual Boolean values."""
    targets = pd.DataFrame(
        {
            "Driver": [
                "VER",
                "NOR",
                "BOT",
            ],
            TARGET_COLUMN: [
                invalid_value,
                False,
                False,
            ],
        }
    )

    with pytest.raises(
        RaceTargetError,
        match="must contain only boolean values",
    ):
        attach_winner_targets(
            make_replay(),
            targets,
        )


def test_missing_target_value_is_rejected() -> None:
    """Winner labels may not contain missing values."""
    targets = pd.DataFrame(
        {
            "Driver": [
                "VER",
                "NOR",
                "BOT",
            ],
            TARGET_COLUMN: pd.Series(
                [
                    True,
                    None,
                    False,
                ],
                dtype="boolean",
            ),
        }
    )

    with pytest.raises(
        RaceTargetError,
        match="WonRace contains missing values",
    ):
        attach_winner_targets(
            make_replay(),
            targets,
        )


@pytest.mark.parametrize(
    "target_values",
    [
        [
            False,
            False,
            False,
        ],
        [
            True,
            True,
            False,
        ],
    ],
)
def test_target_table_requires_exactly_one_winner(
    target_values: list[bool],
) -> None:
    """A target table must identify exactly one race winner."""
    targets = pd.DataFrame(
        {
            "Driver": [
                "VER",
                "NOR",
                "BOT",
            ],
            TARGET_COLUMN: target_values,
        }
    )

    with pytest.raises(
        RaceTargetError,
        match="must contain exactly one winner",
    ):
        attach_winner_targets(
            make_replay(),
            targets,
        )


def test_snapshot_without_winner_is_rejected() -> None:
    """Every snapshot must include the driver who eventually won."""
    replay = make_replay()

    replay = replay.loc[
        ~(replay["SnapshotLap"].eq(2) & replay["Driver"].eq("VER"))
    ].reset_index(drop=True)

    targets = build_winner_targets(make_results())

    with pytest.raises(
        RaceTargetError,
        match="exactly one positive winner target",
    ):
        attach_winner_targets(
            replay,
            targets,
        )


def test_existing_target_column_in_replay_is_rejected() -> None:
    """Target attachment must not silently replace an existing target."""
    replay = make_replay()
    replay[TARGET_COLUMN] = False

    with pytest.raises(
        RaceTargetError,
        match="already contains WonRace",
    ):
        attach_winner_targets(
            replay,
            build_winner_targets(make_results()),
        )


@pytest.mark.parametrize(
    ("replay_value", "target_value", "expected_message"),
    [
        (
            [],
            pd.DataFrame(
                {
                    "Driver": ["VER"],
                    TARGET_COLUMN: [True],
                }
            ),
            "replay must be provided as a pandas DataFrame",
        ),
        (
            pd.DataFrame(
                {
                    "SnapshotLap": [1],
                    "Driver": ["VER"],
                }
            ),
            [],
            "targets must be provided as a pandas DataFrame",
        ),
    ],
)
def test_attach_targets_rejects_non_dataframe_inputs(
    replay_value: object,
    target_value: object,
    expected_message: str,
) -> None:
    """Target attachment requires two pandas DataFrames."""
    with pytest.raises(
        TypeError,
        match=expected_message,
    ):
        attach_winner_targets(
            replay_value,  # type: ignore[arg-type]
            target_value,  # type: ignore[arg-type]
        )


def test_empty_replay_is_rejected() -> None:
    """An empty replay cannot become a supervised-learning table."""
    targets = build_winner_targets(make_results())

    with pytest.raises(
        RaceTargetError,
        match="empty replay table",
    ):
        attach_winner_targets(
            make_replay().iloc[0:0],
            targets,
        )


def test_empty_target_table_is_rejected() -> None:
    """An empty target table cannot label replay rows."""
    with pytest.raises(
        RaceTargetError,
        match="empty winner-target table",
    ):
        attach_winner_targets(
            make_replay(),
            pd.DataFrame(),
        )


def test_missing_replay_columns_are_rejected() -> None:
    """Replay tables require SnapshotLap and Driver."""
    replay = make_replay().drop(columns=["SnapshotLap"])

    with pytest.raises(
        RaceTargetError,
        match="missing required columns",
    ):
        attach_winner_targets(
            replay,
            build_winner_targets(make_results()),
        )


def test_missing_target_columns_are_rejected() -> None:
    """Target tables require Driver and WonRace."""
    targets = pd.DataFrame(
        {
            "Driver": [
                "VER",
                "NOR",
                "BOT",
            ]
        }
    )

    with pytest.raises(
        RaceTargetError,
        match="missing required columns",
    ):
        attach_winner_targets(
            make_replay(),
            targets,
        )


def test_missing_replay_driver_value_is_rejected() -> None:
    """Replay rows must contain usable driver identifiers."""
    replay = make_replay()
    replay["Driver"] = replay["Driver"].astype("object")

    replay.loc[
        0,
        "Driver",
    ] = None

    with pytest.raises(
        RaceTargetError,
        match="missing or blank driver identifiers",
    ):
        attach_winner_targets(
            replay,
            build_winner_targets(make_results()),
        )


def test_duplicate_replay_columns_are_rejected() -> None:
    """Ambiguous replay column names must be rejected."""
    replay = make_replay()

    replay.columns = [
        "SnapshotLap",
        "Driver",
        "Position",
        "Driver",
    ]

    with pytest.raises(
        RaceTargetError,
        match="replay table contains duplicate column names",
    ):
        attach_winner_targets(
            replay,
            build_winner_targets(make_results()),
        )


def test_duplicate_target_columns_are_rejected() -> None:
    """Ambiguous target column names must be rejected."""
    targets = pd.DataFrame(
        [
            [
                "VER",
                True,
                True,
            ],
            [
                "NOR",
                False,
                False,
            ],
            [
                "BOT",
                False,
                False,
            ],
        ],
        columns=[
            "Driver",
            TARGET_COLUMN,
            TARGET_COLUMN,
        ],
    )

    with pytest.raises(
        RaceTargetError,
        match="target table contains duplicate column names",
    ):
        attach_winner_targets(
            make_replay(),
            targets,
        )
