from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from telemetryx.features.engineering import MODEL_FEATURE_COLUMNS
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


def select_worst_snapshots(
    snapshot_evaluation: pd.DataFrame,
    *,
    race_id: str | None = None,
    race_stage: str | None = None,
    limit: int = 10,
) -> pd.DataFrame:
    """
    Return the worst validation snapshots ranked by log loss.

    Optional filters can restrict the diagnostic to one race and/or race stage.
    """
    validate_snapshot_evaluation(snapshot_evaluation)

    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError("limit must be an integer.")

    if limit <= 0:
        raise ModelDiagnosticsError("limit must be greater than zero.")

    staged = add_race_stages(snapshot_evaluation)

    filtered = staged.copy(deep=True)

    if race_id is not None:
        filtered = filtered.loc[filtered["RaceId"].eq(race_id)]

    if race_stage is not None:
        if race_stage not in RACE_STAGE_ORDER:
            raise ModelDiagnosticsError(f"Unknown race stage: {race_stage}.")

        filtered = filtered.loc[filtered["RaceStage"].eq(race_stage)]

    if filtered.empty:
        raise ModelDiagnosticsError(
            "No snapshots matched the requested diagnostic filters."
        )

    columns = [
        "RaceId",
        "SnapshotLap",
        "RaceStage",
        "ActualWinner",
        "ActualWinnerProbability",
        "ActualWinnerRank",
        "SnapshotLogLoss",
        "SnapshotBrierScore",
        "TopOneCorrect",
    ]

    return (
        filtered.sort_values(
            "SnapshotLogLoss",
            ascending=False,
            kind="stable",
        )
        .loc[:, columns]
        .head(limit)
        .reset_index(drop=True)
    )


def add_predicted_winner_details(
    snapshots: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add the model's top predicted driver to snapshot diagnostics.

    Prediction probabilities are matched by RaceId and SnapshotLap.
    """
    required_snapshot_columns = (
        "RaceId",
        "SnapshotLap",
    )

    required_prediction_columns = (
        "RaceId",
        "SnapshotLap",
        "Driver",
        "WinnerProbability",
    )

    missing_snapshot_columns = [
        column
        for column in required_snapshot_columns
        if column not in snapshots.columns
    ]

    if missing_snapshot_columns:
        raise ModelDiagnosticsError(
            "Snapshot diagnostics are missing required columns: "
            f"{missing_snapshot_columns}."
        )

    missing_prediction_columns = [
        column
        for column in required_prediction_columns
        if column not in predictions.columns
    ]

    if missing_prediction_columns:
        raise ModelDiagnosticsError(
            f"Predictions are missing required columns: {missing_prediction_columns}."
        )

    if snapshots.empty:
        raise ModelDiagnosticsError("Snapshot diagnostics cannot be empty.")

    if predictions.empty:
        raise ModelDiagnosticsError("Predictions cannot be empty.")

    selected_keys = snapshots.loc[
        :,
        [
            "RaceId",
            "SnapshotLap",
        ],
    ].drop_duplicates()

    selected_predictions = predictions.merge(
        selected_keys,
        on=[
            "RaceId",
            "SnapshotLap",
        ],
        how="inner",
        validate="many_to_one",
    )

    if selected_predictions.empty:
        raise ModelDiagnosticsError("No predictions matched the selected snapshots.")

    top_probability = selected_predictions.groupby(
        [
            "RaceId",
            "SnapshotLap",
        ],
        sort=False,
        dropna=False,
    )["WinnerProbability"].transform("max")

    favorites = selected_predictions.loc[
        selected_predictions["WinnerProbability"].eq(top_probability)
    ].copy(deep=True)

    favorite_counts = favorites.groupby(
        [
            "RaceId",
            "SnapshotLap",
        ],
        sort=False,
        dropna=False,
    )["Driver"].transform("size")

    if bool(favorite_counts.ne(1).any()):
        raise ModelDiagnosticsError(
            "Every selected snapshot must have one unique predicted winner."
        )

    favorites = favorites.rename(
        columns={
            "Driver": "PredictedWinner",
            "WinnerProbability": "PredictedWinnerProbability",
        }
    )

    favorites = favorites.loc[
        :,
        [
            "RaceId",
            "SnapshotLap",
            "PredictedWinner",
            "PredictedWinnerProbability",
        ],
    ]

    enriched = snapshots.merge(
        favorites,
        on=[
            "RaceId",
            "SnapshotLap",
        ],
        how="left",
        validate="one_to_one",
    )

    if bool(enriched["PredictedWinner"].isna().any()):
        raise ModelDiagnosticsError(
            "Some selected snapshots are missing prediction details."
        )

    return enriched


def select_driver_snapshot_features(
    features: pd.DataFrame,
    *,
    race_id: str,
    snapshot_laps: Sequence[int],
    drivers: Sequence[str],
) -> pd.DataFrame:
    """
    Extract model features for selected drivers and race snapshots.

    This is a post-hoc diagnostic used to investigate model predictions.
    It does not modify the source features or participate in model training.
    """
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame.")

    if features.empty:
        raise ModelDiagnosticsError("Feature data cannot be empty.")

    if not snapshot_laps:
        raise ModelDiagnosticsError("At least one snapshot lap must be provided.")

    if not drivers:
        raise ModelDiagnosticsError("At least one driver must be provided.")

    columns = list(
        dict.fromkeys(
            [
                "RaceId",
                "SnapshotLap",
                "Driver",
                *MODEL_FEATURE_COLUMNS,
            ]
        )
    )

    missing_columns = [column for column in columns if column not in features.columns]

    if missing_columns:
        raise ModelDiagnosticsError(
            f"Features are missing required diagnostic columns: {missing_columns}."
        )

    selected = features.loc[
        features["RaceId"].eq(race_id)
        & features["SnapshotLap"].isin(snapshot_laps)
        & features["Driver"].isin(drivers),
        columns,
    ].copy(deep=True)

    if selected.empty:
        raise ModelDiagnosticsError(
            "No feature rows matched the requested diagnostic filters."
        )

    if bool(
        selected.duplicated(
            subset=[
                "RaceId",
                "SnapshotLap",
                "Driver",
            ]
        ).any()
    ):
        raise ModelDiagnosticsError(
            "Selected features contain duplicate driver-snapshot rows."
        )

    expected_row_count = len(set(snapshot_laps)) * len(set(drivers))

    if len(selected) != expected_row_count:
        raise ModelDiagnosticsError(
            "Some requested driver-snapshot feature rows are missing."
        )

    return selected.sort_values(
        [
            "SnapshotLap",
            "Driver",
        ],
        kind="stable",
    ).reset_index(drop=True)
