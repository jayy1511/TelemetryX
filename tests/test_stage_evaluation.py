from __future__ import annotations

import pandas as pd
import pytest

from telemetryx.modeling.stage_evaluation import (
    EARLY_STAGE,
    LATE_STAGE,
    MIDDLE_STAGE,
    RACE_PROGRESS_COLUMN,
    RACE_STAGE_COLUMN,
    StageEvaluationError,
    add_race_stages,
    summarize_race_stages,
)


def make_evaluation_snapshots() -> pd.DataFrame:
    """Build a small deterministic snapshot-evaluation frame."""
    return pd.DataFrame(
        {
            "RaceId": [
                "race_a",
                "race_a",
                "race_a",
                "race_a",
                "race_a",
                "race_a",
            ],
            "SnapshotLap": [
                1,
                2,
                3,
                4,
                5,
                6,
            ],
            "ActualWinnerProbability": [
                0.20,
                0.40,
                0.50,
                0.60,
                0.80,
                1.00,
            ],
            "ActualWinnerRank": [
                3,
                2,
                2,
                1,
                1,
                1,
            ],
            "SnapshotLogLoss": [
                1.60,
                0.90,
                0.70,
                0.50,
                0.20,
                0.00,
            ],
            "SnapshotBrierScore": [
                0.80,
                0.60,
                0.50,
                0.30,
                0.10,
                0.00,
            ],
            "TopOneCorrect": [
                False,
                False,
                False,
                True,
                True,
                True,
            ],
        }
    )


def test_add_race_stages_assigns_expected_boundaries() -> None:
    """One-third and two-thirds progress belong to early and middle."""
    snapshots = make_evaluation_snapshots()

    staged = add_race_stages(snapshots)

    assert staged[RACE_PROGRESS_COLUMN].tolist() == pytest.approx(
        [
            1 / 6,
            2 / 6,
            3 / 6,
            4 / 6,
            5 / 6,
            1.0,
        ]
    )

    assert staged[RACE_STAGE_COLUMN].tolist() == [
        EARLY_STAGE,
        EARLY_STAGE,
        MIDDLE_STAGE,
        MIDDLE_STAGE,
        LATE_STAGE,
        LATE_STAGE,
    ]


def test_add_race_stages_uses_each_races_own_final_snapshot() -> None:
    """Race progress must be normalized independently for each race."""
    snapshots = pd.concat(
        [
            make_evaluation_snapshots(),
            pd.DataFrame(
                {
                    "RaceId": [
                        "race_b",
                        "race_b",
                        "race_b",
                    ],
                    "SnapshotLap": [
                        1,
                        2,
                        3,
                    ],
                    "ActualWinnerProbability": [
                        0.3,
                        0.6,
                        0.9,
                    ],
                    "ActualWinnerRank": [
                        3,
                        2,
                        1,
                    ],
                    "SnapshotLogLoss": [
                        1.0,
                        0.5,
                        0.1,
                    ],
                    "SnapshotBrierScore": [
                        0.7,
                        0.3,
                        0.05,
                    ],
                    "TopOneCorrect": [
                        False,
                        False,
                        True,
                    ],
                }
            ),
        ],
        ignore_index=True,
    )

    staged = add_race_stages(snapshots)

    race_b = staged.loc[staged["RaceId"].eq("race_b")]

    assert race_b[RACE_PROGRESS_COLUMN].tolist() == pytest.approx(
        [
            1 / 3,
            2 / 3,
            1.0,
        ]
    )

    assert race_b[RACE_STAGE_COLUMN].tolist() == [
        EARLY_STAGE,
        MIDDLE_STAGE,
        LATE_STAGE,
    ]


def test_add_race_stages_does_not_modify_input() -> None:
    """Stage evaluation must not mutate the caller's evaluation frame."""
    snapshots = make_evaluation_snapshots()

    original = snapshots.copy(deep=True)

    add_race_stages(snapshots)

    pd.testing.assert_frame_equal(
        snapshots,
        original,
    )


def test_summarize_race_stages_calculates_expected_metrics() -> None:
    """Stage summaries should average the underlying snapshot metrics."""
    snapshots = make_evaluation_snapshots()

    summary = summarize_race_stages(snapshots)

    early = summary.loc[summary[RACE_STAGE_COLUMN].eq(EARLY_STAGE)].iloc[0]

    middle = summary.loc[summary[RACE_STAGE_COLUMN].eq(MIDDLE_STAGE)].iloc[0]

    late = summary.loc[summary[RACE_STAGE_COLUMN].eq(LATE_STAGE)].iloc[0]

    assert early["SnapshotCount"] == 2
    assert early["RaceCount"] == 1

    assert early["MeanLogLoss"] == pytest.approx(1.25)

    assert early["MeanBrierScore"] == pytest.approx(0.70)

    assert early["TopOneAccuracy"] == pytest.approx(0.0)

    assert early["MeanWinnerProbability"] == pytest.approx(0.30)

    assert early["MeanWinnerRank"] == pytest.approx(2.5)

    assert middle["MeanLogLoss"] == pytest.approx(0.60)

    assert middle["TopOneAccuracy"] == pytest.approx(0.5)

    assert middle["MeanWinnerProbability"] == pytest.approx(0.55)

    assert late["MeanLogLoss"] == pytest.approx(0.10)

    assert late["MeanBrierScore"] == pytest.approx(0.05)

    assert late["TopOneAccuracy"] == pytest.approx(1.0)

    assert late["MeanWinnerProbability"] == pytest.approx(0.90)

    assert late["MeanWinnerRank"] == pytest.approx(1.0)


def test_summarize_race_stages_preserves_stage_order() -> None:
    """Stage summaries should always appear early, middle, then late."""
    summary = summarize_race_stages(make_evaluation_snapshots())

    assert summary[RACE_STAGE_COLUMN].tolist() == [
        EARLY_STAGE,
        MIDDLE_STAGE,
        LATE_STAGE,
    ]


def test_add_race_stages_rejects_empty_frame() -> None:
    """Evaluation cannot operate on an empty snapshot frame."""
    snapshots = pd.DataFrame()

    with pytest.raises(
        StageEvaluationError,
        match="cannot be empty",
    ):
        add_race_stages(snapshots)


def test_add_race_stages_rejects_missing_required_column() -> None:
    """Evaluation input must contain every required snapshot metric."""
    snapshots = make_evaluation_snapshots().drop(
        columns=[
            "SnapshotLogLoss",
        ]
    )

    with pytest.raises(
        StageEvaluationError,
        match="missing required columns",
    ):
        add_race_stages(snapshots)


@pytest.mark.parametrize(
    "snapshot_lap",
    [
        0,
        -1,
    ],
)
def test_add_race_stages_rejects_non_positive_snapshot_laps(
    snapshot_lap: int,
) -> None:
    """Snapshot laps must represent positive completed race progress."""
    snapshots = make_evaluation_snapshots()

    snapshots.loc[
        0,
        "SnapshotLap",
    ] = snapshot_lap

    with pytest.raises(
        StageEvaluationError,
        match="SnapshotLap must contain positive values",
    ):
        add_race_stages(snapshots)


def test_add_race_stages_rejects_missing_snapshot_lap() -> None:
    """Missing race progress cannot be assigned to a stage."""
    snapshots = make_evaluation_snapshots()

    snapshots.loc[
        0,
        "SnapshotLap",
    ] = pd.NA

    with pytest.raises(
        StageEvaluationError,
        match="SnapshotLap cannot contain missing",
    ):
        add_race_stages(snapshots)


def test_add_race_stages_rejects_missing_race_id() -> None:
    """Every evaluation snapshot must belong to a known race."""
    snapshots = make_evaluation_snapshots()

    snapshots.loc[
        0,
        "RaceId",
    ] = pd.NA

    with pytest.raises(
        StageEvaluationError,
        match="RaceId cannot contain missing values",
    ):
        add_race_stages(snapshots)


def test_add_race_stages_rejects_empty_race_id() -> None:
    """Whitespace-only race identifiers are invalid."""
    snapshots = make_evaluation_snapshots()

    snapshots.loc[
        0,
        "RaceId",
    ] = "   "

    with pytest.raises(
        StageEvaluationError,
        match="RaceId cannot contain empty values",
    ):
        add_race_stages(snapshots)


def test_summarize_race_stages_requires_all_three_stages() -> None:
    """A summary should fail when a dataset cannot represent every stage."""
    snapshots = (
        make_evaluation_snapshots()
        .iloc[
            [
                0,
            ]
        ]
        .copy()
    )

    with pytest.raises(
        StageEvaluationError,
        match="contains no snapshots",
    ):
        summarize_race_stages(snapshots)
