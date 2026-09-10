"""Run the leakage-safe TelemetryX baseline modeling experiment."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from telemetryx.data.split import (
    CorpusSplit,
    CorpusSplitError,
    split_race_corpus,
)
from telemetryx.features.engineering import (
    MODEL_FEATURE_COLUMNS,
    FeatureEngineeringError,
    engineer_race_features,
)
from telemetryx.modeling.baseline import (
    BaselineModelError,
    BaselineWinnerModel,
    predict_winner_probabilities,
    train_baseline_winner_model,
)
from telemetryx.modeling.evaluation import (
    ModelEvaluationError,
    WinnerModelEvaluation,
    evaluate_winner_probabilities,
)


class BaselineExperimentError(ValueError):
    """Raised when the baseline experiment cannot be completed safely."""


@dataclass(frozen=True, slots=True)
class BaselineExperimentSummary:
    """High-level structure and validation metrics for one experiment."""

    train_race_count: int
    validation_race_count: int
    test_race_count: int
    train_row_count: int
    validation_row_count: int
    test_row_count: int
    validation_snapshot_count: int
    validation_mean_log_loss: float
    validation_mean_brier_score: float
    validation_top_one_accuracy: float
    validation_mean_winner_probability: float


@dataclass(frozen=True, slots=True)
class BaselineExperimentResult:
    """
    Result of one chronological TelemetryX baseline experiment.

    Test data remains present only in the race split. It is deliberately not
    feature-engineered, transformed, predicted or evaluated by this workflow.
    """

    split: CorpusSplit
    model: BaselineWinnerModel
    validation_features: pd.DataFrame
    validation_predictions: pd.DataFrame
    validation_evaluation: WinnerModelEvaluation
    summary: BaselineExperimentSummary


def run_baseline_experiment(
    corpus: pd.DataFrame,
    *,
    validation_season: int,
    validation_last_races: int,
    test_seasons: Sequence[int],
    model_columns: Sequence[str] = MODEL_FEATURE_COLUMNS,
    max_iterations: int = 2000,
) -> BaselineExperimentResult:
    """
    Run the first leakage-safe TelemetryX modeling experiment.

    The corpus is split chronologically before feature engineering. Training
    features are used to fit preprocessing and logistic regression.
    Validation features are transformed using that fitted state and evaluated.

    The test split is deliberately left untouched so it remains a genuine
    final holdout.

    Parameters
    ----------
    corpus:
        Valid TelemetryX multi-race corpus.
    validation_season:
        Season whose final races form the validation set.
    validation_last_races:
        Number of final races from ``validation_season`` to reserve.
    test_seasons:
        Complete future seasons reserved as the final test holdout.
    model_columns:
        Audited predictive feature columns.
    max_iterations:
        Maximum logistic-regression optimization iterations.

    Returns
    -------
    BaselineExperimentResult
        Split state, fitted model, validation predictions and metrics.

    Raises
    ------
    TypeError
        If ``corpus`` is not a pandas DataFrame.
    BaselineExperimentError
        If splitting, feature engineering, training, prediction or evaluation
        fails.
    """
    if not isinstance(corpus, pd.DataFrame):
        raise TypeError("corpus must be provided as a pandas DataFrame.")

    try:
        split = split_race_corpus(
            corpus,
            validation_season=validation_season,
            validation_last_races=validation_last_races,
            test_seasons=test_seasons,
        )
    except (TypeError, CorpusSplitError) as exc:
        raise BaselineExperimentError(
            "The corpus could not be split for the baseline experiment."
        ) from exc

    try:
        training_features = engineer_race_features(split.train)

        validation_features = engineer_race_features(split.validation)
    except (TypeError, FeatureEngineeringError) as exc:
        raise BaselineExperimentError(
            "Feature engineering failed during the baseline experiment."
        ) from exc

    try:
        model = train_baseline_winner_model(
            training_features,
            model_columns=model_columns,
            max_iterations=max_iterations,
        )
    except (TypeError, BaselineModelError) as exc:
        raise BaselineExperimentError("Baseline model training failed.") from exc

    try:
        validation_predictions = predict_winner_probabilities(
            model,
            validation_features,
        )
    except (TypeError, BaselineModelError) as exc:
        raise BaselineExperimentError("Baseline validation prediction failed.") from exc

    try:
        validation_evaluation = evaluate_winner_probabilities(
            validation_features,
            validation_predictions,
        )
    except (TypeError, ModelEvaluationError) as exc:
        raise BaselineExperimentError("Baseline validation evaluation failed.") from exc

    _validate_experiment_boundaries(
        split=split,
        validation_features=validation_features,
        validation_predictions=validation_predictions,
    )

    summary = _build_experiment_summary(
        split=split,
        validation_evaluation=validation_evaluation,
    )

    return BaselineExperimentResult(
        split=split,
        model=model,
        validation_features=validation_features,
        validation_predictions=validation_predictions,
        validation_evaluation=validation_evaluation,
        summary=summary,
    )


def _validate_experiment_boundaries(
    *,
    split: CorpusSplit,
    validation_features: pd.DataFrame,
    validation_predictions: pd.DataFrame,
) -> None:
    """Verify that validation output never crosses race-split boundaries."""
    train_race_ids = _race_ids(split.train)

    validation_race_ids = _race_ids(split.validation)

    test_race_ids = _race_ids(split.test)

    feature_race_ids = _race_ids(validation_features)

    prediction_race_ids = _race_ids(validation_predictions)

    if feature_race_ids != validation_race_ids:
        raise BaselineExperimentError(
            "Validation feature races do not match the chronological validation split."
        )

    if prediction_race_ids != validation_race_ids:
        raise BaselineExperimentError(
            "Validation prediction races do not match the "
            "chronological validation split."
        )

    if train_race_ids.intersection(validation_race_ids):
        raise BaselineExperimentError("Training and validation races overlap.")

    if train_race_ids.intersection(test_race_ids):
        raise BaselineExperimentError("Training and test races overlap.")

    if validation_race_ids.intersection(test_race_ids):
        raise BaselineExperimentError("Validation and test races overlap.")


def _build_experiment_summary(
    *,
    split: CorpusSplit,
    validation_evaluation: WinnerModelEvaluation,
) -> BaselineExperimentSummary:
    """Create a compact experiment summary from validated results."""
    evaluation_summary = validation_evaluation.summary

    return BaselineExperimentSummary(
        train_race_count=split.train_race_count,
        validation_race_count=(split.validation_race_count),
        test_race_count=split.test_race_count,
        train_row_count=len(split.train),
        validation_row_count=len(split.validation),
        test_row_count=len(split.test),
        validation_snapshot_count=(evaluation_summary.snapshot_count),
        validation_mean_log_loss=(evaluation_summary.mean_log_loss),
        validation_mean_brier_score=(evaluation_summary.mean_brier_score),
        validation_top_one_accuracy=(evaluation_summary.top_one_accuracy),
        validation_mean_winner_probability=(
            evaluation_summary.mean_actual_winner_probability
        ),
    )


def _race_ids(
    frame: pd.DataFrame,
) -> set[str]:
    """Return normalized race identifiers from a DataFrame."""
    return {
        str(value) for value in (frame["RaceId"].astype("string").dropna().tolist())
    }
