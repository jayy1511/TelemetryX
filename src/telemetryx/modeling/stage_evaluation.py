from __future__ import annotations

from typing import Final

import pandas as pd

RACE_PROGRESS_COLUMN: Final[str] = "RaceProgressFraction"
RACE_STAGE_COLUMN: Final[str] = "RaceStage"

EARLY_STAGE: Final[str] = "Early"
MIDDLE_STAGE: Final[str] = "Middle"
LATE_STAGE: Final[str] = "Late"

RACE_STAGE_ORDER: Final[tuple[str, ...]] = (
    EARLY_STAGE,
    MIDDLE_STAGE,
    LATE_STAGE,
)

REQUIRED_EVALUATION_COLUMNS: Final[tuple[str, ...]] = (
    "RaceId",
    "SnapshotLap",
    "ActualWinnerProbability",
    "ActualWinnerRank",
    "SnapshotLogLoss",
    "SnapshotBrierScore",
    "TopOneCorrect",
)


class StageEvaluationError(ValueError):
    """Raised when race-stage evaluation cannot be performed."""


def add_race_stages(
    snapshots: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add relative race progress and race-stage labels.

    Race progress is calculated independently for each race:

        SnapshotLap / maximum SnapshotLap for that race

    Stages are defined as:

        Early:  progress <= 1/3
        Middle: 1/3 < progress <= 2/3
        Late:   progress > 2/3

    The race's final lap is used only for evaluation grouping.
    It is not introduced as a model feature.
    """
    _validate_snapshot_frame(snapshots)

    working = snapshots.copy(deep=True)

    snapshot_laps = pd.to_numeric(
        working["SnapshotLap"],
        errors="coerce",
    ).astype("Float64")

    if bool(snapshot_laps.isna().any()):
        raise StageEvaluationError(
            "SnapshotLap cannot contain missing or non-numeric values."
        )

    if bool(snapshot_laps.le(0).any()):
        raise StageEvaluationError("SnapshotLap must contain positive values.")

    maximum_laps = working.groupby(
        "RaceId",
        sort=False,
    )["SnapshotLap"].transform("max")

    maximum_laps = pd.to_numeric(
        maximum_laps,
        errors="coerce",
    ).astype("Float64")

    if bool(maximum_laps.isna().any()):
        raise StageEvaluationError(
            "Unable to determine maximum SnapshotLap for one or more races."
        )

    if bool(maximum_laps.le(0).any()):
        raise StageEvaluationError("Race maximum SnapshotLap must be positive.")

    progress = (snapshot_laps / maximum_laps).astype("Float64")

    if bool(progress.isna().any()):
        raise StageEvaluationError("Race progress cannot contain missing values.")

    if bool((progress.le(0.0) | progress.gt(1.0)).any()):
        raise StageEvaluationError("Race progress must remain inside (0, 1].")

    stages = pd.Series(
        LATE_STAGE,
        index=working.index,
        dtype="string",
    )

    stages.loc[progress.le(1.0 / 3.0)] = EARLY_STAGE

    stages.loc[progress.gt(1.0 / 3.0) & progress.le(2.0 / 3.0)] = MIDDLE_STAGE

    working[RACE_PROGRESS_COLUMN] = progress

    working[RACE_STAGE_COLUMN] = stages

    return working


def summarize_race_stages(
    snapshots: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize prediction quality for early, middle and late race stages.

    Returns one row per stage with the same core metrics used by the
    baseline experiment evaluation.
    """
    staged = add_race_stages(snapshots)

    summary_rows: list[dict[str, object]] = []

    for stage in RACE_STAGE_ORDER:
        stage_frame = staged.loc[staged[RACE_STAGE_COLUMN].eq(stage)]

        if stage_frame.empty:
            raise StageEvaluationError(f"Race stage {stage!r} contains no snapshots.")

        summary_rows.append(
            {
                RACE_STAGE_COLUMN: stage,
                "SnapshotCount": len(stage_frame),
                "RaceCount": int(stage_frame["RaceId"].nunique()),
                "MeanLogLoss": float(
                    pd.to_numeric(
                        stage_frame["SnapshotLogLoss"],
                        errors="raise",
                    ).mean()
                ),
                "MeanBrierScore": float(
                    pd.to_numeric(
                        stage_frame["SnapshotBrierScore"],
                        errors="raise",
                    ).mean()
                ),
                "TopOneAccuracy": float(
                    stage_frame["TopOneCorrect"].astype("Float64").mean()
                ),
                "MeanWinnerProbability": float(
                    pd.to_numeric(
                        stage_frame["ActualWinnerProbability"],
                        errors="raise",
                    ).mean()
                ),
                "MeanWinnerRank": float(
                    pd.to_numeric(
                        stage_frame["ActualWinnerRank"],
                        errors="raise",
                    ).mean()
                ),
            }
        )

    return pd.DataFrame(summary_rows)


def _validate_snapshot_frame(
    snapshots: pd.DataFrame,
) -> None:
    """Validate the evaluation frame required for stage analysis."""
    if not isinstance(
        snapshots,
        pd.DataFrame,
    ):
        raise TypeError("snapshots must be a pandas DataFrame.")

    if snapshots.empty:
        raise StageEvaluationError("Snapshot evaluation cannot be empty.")

    missing_columns = [
        column
        for column in REQUIRED_EVALUATION_COLUMNS
        if column not in snapshots.columns
    ]

    if missing_columns:
        raise StageEvaluationError(
            "Snapshot evaluation is missing required columns: "
            f"{', '.join(missing_columns)}."
        )

    if bool(snapshots["RaceId"].isna().any()):
        raise StageEvaluationError("RaceId cannot contain missing values.")

    race_ids = snapshots["RaceId"].astype("string").str.strip()

    if bool(race_ids.eq("").any()):
        raise StageEvaluationError("RaceId cannot contain empty values.")
