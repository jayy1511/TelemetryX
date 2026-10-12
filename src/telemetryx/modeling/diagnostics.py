from __future__ import annotations

import pandas as pd

from telemetryx.modeling.evaluation import (
    ModelEvaluationError,
    validate_snapshot_evaluation,
)
from telemetryx.modeling.stage_evaluation import add_race_stages

RACE_STAGE_ORDER: tuple[str, ...] = (
    "Early",
    "Middle",
    "Late",
)


class ModelDiagnosticsError(RuntimeError):
    """Raised when model diagnostics cannot be constructed."""


def summarize_race_stage_performance(
    snapshot_evaluation: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize validation performance for every race and race stage.

    This is a post-hoc diagnostic over already evaluated snapshots. It does
    not participate in model fitting or feature engineering.
    """
    try:
        validate_snapshot_evaluation(snapshot_evaluation)
        staged = add_race_stages(snapshot_evaluation)
    except ModelEvaluationError as exc:
        raise ModelDiagnosticsError(
            "Snapshot evaluation is invalid for race-stage diagnostics."
        ) from exc

    rows: list[dict[str, object]] = []

    grouped = staged.groupby(
        [
            "RaceId",
            "RaceStage",
        ],
        sort=False,
        observed=True,
        dropna=False,
    )

    for (race_id, race_stage), group in grouped:
        rows.append(
            {
                "RaceId": str(race_id),
                "RaceStage": str(race_stage),
                "SnapshotCount": int(len(group)),
                "MeanLogLoss": float(group["SnapshotLogLoss"].mean()),
                "MeanBrierScore": float(group["SnapshotBrierScore"].mean()),
                "TopOneAccuracy": float(group["TopOneCorrect"].mean()),
                "MeanWinnerProbability": float(group["ActualWinnerProbability"].mean()),
                "MeanWinnerRank": float(group["ActualWinnerRank"].mean()),
            }
        )

    summary = pd.DataFrame.from_records(rows)

    if summary.empty:
        raise ModelDiagnosticsError("Race-stage diagnostics cannot be empty.")

    summary["_StageOrder"] = summary["RaceStage"].map(
        {stage: index for index, stage in enumerate(RACE_STAGE_ORDER)}
    )

    if bool(summary["_StageOrder"].isna().any()):
        raise ModelDiagnosticsError(
            "Race-stage diagnostics contain an unknown race stage."
        )

    summary = (
        summary.sort_values(
            [
                "RaceId",
                "_StageOrder",
            ],
            kind="stable",
        )
        .drop(columns="_StageOrder")
        .reset_index(drop=True)
    )

    return summary
