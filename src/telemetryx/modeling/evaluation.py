"""Evaluate TelemetryX winner-probability predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from telemetryx.data.targets import TARGET_COLUMN
from telemetryx.features.engineering import (
    FEATURE_KEY_COLUMNS,
    FeatureEngineeringError,
    validate_feature_frame,
)
from telemetryx.modeling.baseline import (
    WINNER_PROBABILITY_COLUMN,
    BaselineModelError,
    validate_winner_probability_frame,
)

ACTUAL_WINNER_COLUMN: Final[str] = "ActualWinner"
ACTUAL_WINNER_PROBABILITY_COLUMN: Final[str] = "ActualWinnerProbability"
ACTUAL_WINNER_RANK_COLUMN: Final[str] = "ActualWinnerRank"
SNAPSHOT_LOG_LOSS_COLUMN: Final[str] = "SnapshotLogLoss"
SNAPSHOT_BRIER_SCORE_COLUMN: Final[str] = "SnapshotBrierScore"
TOP_ONE_CORRECT_COLUMN: Final[str] = "TopOneCorrect"

SNAPSHOT_EVALUATION_COLUMNS: Final[tuple[str, ...]] = (
    "RaceId",
    "Season",
    "RoundNumber",
    "SnapshotLap",
    ACTUAL_WINNER_COLUMN,
    ACTUAL_WINNER_PROBABILITY_COLUMN,
    ACTUAL_WINNER_RANK_COLUMN,
    SNAPSHOT_LOG_LOSS_COLUMN,
    SNAPSHOT_BRIER_SCORE_COLUMN,
    TOP_ONE_CORRECT_COLUMN,
)


class ModelEvaluationError(ValueError):
    """Raised when winner-probability evaluation cannot be completed."""


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """Aggregate probability and ranking metrics."""

    snapshot_count: int
    race_count: int
    row_count: int
    mean_log_loss: float
    mean_brier_score: float
    top_one_accuracy: float
    mean_actual_winner_probability: float


@dataclass(frozen=True, slots=True)
class WinnerModelEvaluation:
    """Complete TelemetryX model-evaluation result."""

    summary: EvaluationSummary
    snapshots: pd.DataFrame


def evaluate_winner_probabilities(
    features: pd.DataFrame,
    predictions: pd.DataFrame,
) -> WinnerModelEvaluation:
    """
    Evaluate winner probabilities at complete race-snapshot level.

    The supervised feature frame supplies the true eventual winner.
    Predictions supply one normalized probability distribution for each race
    snapshot.

    Metrics are calculated per snapshot rather than as ordinary row-level
    classification accuracy.

    Parameters
    ----------
    features:
        Engineered feature frame containing ``WonRace``.
    predictions:
        Winner probability frame produced by the baseline model.

    Returns
    -------
    WinnerModelEvaluation
        Aggregate metrics plus one evaluation row per race snapshot.

    Raises
    ------
    TypeError
        If either input is not a pandas DataFrame.
    ModelEvaluationError
        If the feature and prediction tables cannot be safely aligned.
    """
    if not isinstance(
        features,
        pd.DataFrame,
    ):
        raise TypeError("features must be provided as a pandas DataFrame.")

    if not isinstance(
        predictions,
        pd.DataFrame,
    ):
        raise TypeError("predictions must be provided as a pandas DataFrame.")

    try:
        validate_feature_frame(features)
    except (
        TypeError,
        FeatureEngineeringError,
    ) as exc:
        raise ModelEvaluationError(
            "The feature frame failed validation before evaluation."
        ) from exc

    try:
        validate_winner_probability_frame(predictions)
    except (
        TypeError,
        BaselineModelError,
    ) as exc:
        raise ModelEvaluationError(
            "The prediction frame failed validation before evaluation."
        ) from exc

    evaluation_rows = _align_features_and_predictions(
        features=features,
        predictions=predictions,
    )

    _validate_one_winner_per_snapshot(evaluation_rows)

    snapshot_evaluation = _build_snapshot_evaluation(evaluation_rows)

    validate_snapshot_evaluation(snapshot_evaluation)

    summary = _summarize_evaluation(
        snapshot_evaluation=snapshot_evaluation,
        row_count=len(evaluation_rows),
    )

    return WinnerModelEvaluation(
        summary=summary,
        snapshots=snapshot_evaluation,
    )


def validate_snapshot_evaluation(
    snapshots: pd.DataFrame,
) -> None:
    """
    Validate the per-snapshot evaluation artifact.

    Parameters
    ----------
    snapshots:
        Table containing one evaluation observation per race snapshot.

    Raises
    ------
    TypeError
        If ``snapshots`` is not a pandas DataFrame.
    ModelEvaluationError
        If evaluation invariants are violated.
    """
    if not isinstance(
        snapshots,
        pd.DataFrame,
    ):
        raise TypeError("snapshots must be provided as a pandas DataFrame.")

    if snapshots.empty:
        raise ModelEvaluationError("The snapshot evaluation contains no rows.")

    if tuple(snapshots.columns) != SNAPSHOT_EVALUATION_COLUMNS:
        raise ModelEvaluationError(
            "The snapshot evaluation does not match the required evaluation schema."
        )

    if bool(
        snapshots[
            [
                "RaceId",
                "Season",
                "RoundNumber",
                "SnapshotLap",
                ACTUAL_WINNER_COLUMN,
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise ModelEvaluationError(
            "Snapshot evaluation identity columns cannot contain missing values."
        )

    duplicate_snapshots = snapshots.duplicated(
        subset=[
            "RaceId",
            "SnapshotLap",
        ],
        keep=False,
    )

    if bool(duplicate_snapshots.any()):
        raise ModelEvaluationError("The evaluation contains duplicate race snapshots.")

    winner_probabilities = pd.to_numeric(
        snapshots[ACTUAL_WINNER_PROBABILITY_COLUMN],
        errors="coerce",
    )

    if bool(winner_probabilities.isna().any()):
        raise ModelEvaluationError(
            "Actual winner probabilities cannot contain missing values."
        )

    winner_probability_array = winner_probabilities.to_numpy(
        dtype=np.float64,
    )

    if not bool(np.isfinite(winner_probability_array).all()):
        raise ModelEvaluationError("Actual winner probabilities must be finite.")

    if bool((winner_probability_array < 0.0).any()) or bool(
        (winner_probability_array > 1.0).any()
    ):
        raise ModelEvaluationError(
            "Actual winner probabilities must remain between 0 and 1."
        )

    ranks = pd.to_numeric(
        snapshots[ACTUAL_WINNER_RANK_COLUMN],
        errors="coerce",
    )

    if bool(ranks.isna().any()):
        raise ModelEvaluationError("Actual winner ranks cannot contain missing values.")

    rank_array = ranks.to_numpy(
        dtype=np.float64,
    )

    if not bool(np.isfinite(rank_array).all()):
        raise ModelEvaluationError("Actual winner ranks must be finite.")

    if bool((rank_array < 1.0).any()):
        raise ModelEvaluationError("Actual winner ranks must be at least one.")

    for column in (
        SNAPSHOT_LOG_LOSS_COLUMN,
        SNAPSHOT_BRIER_SCORE_COLUMN,
    ):
        values = pd.to_numeric(
            snapshots[column],
            errors="coerce",
        )

        if bool(values.isna().any()):
            raise ModelEvaluationError(f"{column} cannot contain missing values.")

        value_array = values.to_numpy(
            dtype=np.float64,
        )

        if not bool(np.isfinite(value_array).all()):
            raise ModelEvaluationError(f"{column} must contain only finite values.")

        if bool((value_array < 0.0).any()):
            raise ModelEvaluationError(f"{column} cannot contain negative values.")

    top_one = snapshots[TOP_ONE_CORRECT_COLUMN]

    if not pd.api.types.is_bool_dtype(top_one.dtype):
        raise ModelEvaluationError(
            f"{TOP_ONE_CORRECT_COLUMN} must use a Boolean dtype."
        )

    if bool(top_one.isna().any()):
        raise ModelEvaluationError(
            f"{TOP_ONE_CORRECT_COLUMN} cannot contain missing values."
        )


def _align_features_and_predictions(
    *,
    features: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Align supervised targets and predicted probabilities by exact key."""
    feature_columns = [
        *FEATURE_KEY_COLUMNS,
        "Season",
        "RoundNumber",
        TARGET_COLUMN,
    ]

    targets = features.loc[
        :,
        feature_columns,
    ].copy(deep=True)

    try:
        merged = targets.merge(
            predictions,
            on=list(FEATURE_KEY_COLUMNS),
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
    except (
        KeyError,
        pd.errors.MergeError,
        ValueError,
    ) as exc:
        raise ModelEvaluationError(
            "Could not align feature targets with model predictions."
        ) from exc

    complete_alignment = merged["_merge"].eq("both")

    if not bool(complete_alignment.all()):
        missing_predictions = int(merged["_merge"].eq("left_only").sum())

        unexpected_predictions = int(merged["_merge"].eq("right_only").sum())

        raise ModelEvaluationError(
            "Feature and prediction keys do not match exactly: "
            f"{missing_predictions} feature rows are missing predictions "
            f"and {unexpected_predictions} prediction rows have no "
            "matching feature row."
        )

    if len(merged) != len(features):
        raise ModelEvaluationError(
            "Evaluation alignment changed the expected row count."
        )

    return merged.drop(
        columns=[
            "_merge",
        ]
    )


def _validate_one_winner_per_snapshot(
    evaluation_rows: pd.DataFrame,
) -> None:
    """Require exactly one true winner in every race snapshot."""
    winner_counts = evaluation_rows.groupby(
        [
            "RaceId",
            "SnapshotLap",
        ],
        sort=False,
        dropna=False,
    )[TARGET_COLUMN].sum()

    if not bool(winner_counts.eq(1).all()):
        raise ModelEvaluationError(
            "Every race snapshot must contain exactly one true winner."
        )


def _build_snapshot_evaluation(
    evaluation_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Convert driver-level predictions into snapshot-level metrics."""
    working = evaluation_rows.copy(deep=True)

    probabilities = pd.to_numeric(
        working[WINNER_PROBABILITY_COLUMN],
        errors="raise",
    ).astype("float64")

    target_numeric = working[TARGET_COLUMN].astype("int64")

    working["_SquaredProbabilityError"] = (probabilities - target_numeric).pow(2)

    working["_ProbabilityRank"] = probabilities.groupby(
        [
            working["RaceId"],
            working["SnapshotLap"],
        ],
        sort=False,
        dropna=False,
    ).rank(
        method="min",
        ascending=False,
    )

    working["_TopProbabilityCount"] = (
        working["_ProbabilityRank"]
        .eq(1.0)
        .groupby(
            [
                working["RaceId"],
                working["SnapshotLap"],
            ],
            sort=False,
            dropna=False,
        )
        .transform("sum")
    )

    brier_scores = (
        working.groupby(
            [
                "RaceId",
                "SnapshotLap",
            ],
            sort=False,
            dropna=False,
        )["_SquaredProbabilityError"]
        .sum()
        .rename(SNAPSHOT_BRIER_SCORE_COLUMN)
        .reset_index()
    )

    winner_rows = working.loc[working[TARGET_COLUMN].astype(bool)].copy(deep=True)

    winner_probabilities = pd.to_numeric(
        winner_rows[WINNER_PROBABILITY_COLUMN],
        errors="raise",
    ).astype("float64")

    clipped_winner_probabilities = winner_probabilities.clip(
        lower=np.finfo(np.float64).tiny,
        upper=1.0,
    )

    winner_rows[SNAPSHOT_LOG_LOSS_COLUMN] = -np.log(clipped_winner_probabilities)

    winner_rows[TOP_ONE_CORRECT_COLUMN] = (
        winner_rows["_ProbabilityRank"].eq(1.0)
        & winner_rows["_TopProbabilityCount"].eq(1)
    ).astype("boolean")

    winner_rows[ACTUAL_WINNER_RANK_COLUMN] = pd.to_numeric(
        winner_rows["_ProbabilityRank"],
        errors="raise",
    ).astype("Int64")

    snapshot_evaluation = winner_rows.rename(
        columns={
            "Driver": ACTUAL_WINNER_COLUMN,
            WINNER_PROBABILITY_COLUMN: (ACTUAL_WINNER_PROBABILITY_COLUMN),
        }
    ).loc[
        :,
        [
            "RaceId",
            "Season",
            "RoundNumber",
            "SnapshotLap",
            ACTUAL_WINNER_COLUMN,
            ACTUAL_WINNER_PROBABILITY_COLUMN,
            ACTUAL_WINNER_RANK_COLUMN,
            SNAPSHOT_LOG_LOSS_COLUMN,
            TOP_ONE_CORRECT_COLUMN,
        ],
    ]

    snapshot_evaluation = snapshot_evaluation.merge(
        brier_scores,
        on=[
            "RaceId",
            "SnapshotLap",
        ],
        how="left",
        validate="one_to_one",
    )

    snapshot_evaluation = snapshot_evaluation.loc[
        :,
        list(SNAPSHOT_EVALUATION_COLUMNS),
    ]

    snapshot_evaluation = snapshot_evaluation.sort_values(
        by=[
            "Season",
            "RoundNumber",
            "SnapshotLap",
        ],
        kind="stable",
    ).reset_index(drop=True)

    return snapshot_evaluation


def _summarize_evaluation(
    *,
    snapshot_evaluation: pd.DataFrame,
    row_count: int,
) -> EvaluationSummary:
    """Aggregate snapshot metrics into one model-evaluation summary."""
    snapshot_count = len(snapshot_evaluation)

    race_count = int(snapshot_evaluation["RaceId"].nunique())

    mean_log_loss = float(snapshot_evaluation[SNAPSHOT_LOG_LOSS_COLUMN].mean())

    mean_brier_score = float(snapshot_evaluation[SNAPSHOT_BRIER_SCORE_COLUMN].mean())

    top_one_accuracy = float(
        snapshot_evaluation[TOP_ONE_CORRECT_COLUMN].astype("float64").mean()
    )

    mean_actual_winner_probability = float(
        snapshot_evaluation[ACTUAL_WINNER_PROBABILITY_COLUMN].mean()
    )

    return EvaluationSummary(
        snapshot_count=snapshot_count,
        race_count=race_count,
        row_count=row_count,
        mean_log_loss=mean_log_loss,
        mean_brier_score=mean_brier_score,
        top_one_accuracy=top_one_accuracy,
        mean_actual_winner_probability=(mean_actual_winner_probability),
    )
