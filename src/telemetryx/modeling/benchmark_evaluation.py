from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from telemetryx.modeling.benchmarks import (
    BenchmarkPredictionError,
    predict_driver_prior_winner_probabilities,
    predict_uniform_winner_probabilities,
)
from telemetryx.modeling.evaluation import (
    ModelEvaluationError,
    WinnerModelEvaluation,
    evaluate_winner_probabilities,
)


class BenchmarkEvaluationError(RuntimeError):
    """Raised when validation benchmark evaluation cannot be completed."""


@dataclass(frozen=True)
class ValidationBenchmarkResult:
    """Predictions and evaluations for TelemetryX validation benchmarks."""

    uniform_predictions: pd.DataFrame
    uniform_evaluation: WinnerModelEvaluation
    driver_prior_predictions: pd.DataFrame
    driver_prior_evaluation: WinnerModelEvaluation


def evaluate_validation_benchmarks(
    training_corpus: pd.DataFrame,
    validation_features: pd.DataFrame,
) -> ValidationBenchmarkResult:
    """
    Evaluate reference predictors on the validation partition.

    The uniform benchmark uses only the drivers present in each validation
    snapshot.

    The driver-prior benchmark learns historical win frequencies strictly
    from the training partition, then applies them to validation snapshots.

    No test rows are accepted or evaluated by this function.
    """
    if not isinstance(
        training_corpus,
        pd.DataFrame,
    ):
        raise TypeError("training_corpus must be a pandas DataFrame.")

    if not isinstance(
        validation_features,
        pd.DataFrame,
    ):
        raise TypeError("validation_features must be a pandas DataFrame.")

    try:
        uniform_predictions = predict_uniform_winner_probabilities(validation_features)

        uniform_evaluation = evaluate_winner_probabilities(
            validation_features,
            uniform_predictions,
        )

        driver_prior_predictions = predict_driver_prior_winner_probabilities(
            training_corpus,
            validation_features,
        )

        driver_prior_evaluation = evaluate_winner_probabilities(
            validation_features,
            driver_prior_predictions,
        )

    except (
        BenchmarkPredictionError,
        ModelEvaluationError,
    ) as exc:
        raise BenchmarkEvaluationError(
            "Failed to evaluate validation benchmarks."
        ) from exc

    return ValidationBenchmarkResult(
        uniform_predictions=uniform_predictions,
        uniform_evaluation=uniform_evaluation,
        driver_prior_predictions=driver_prior_predictions,
        driver_prior_evaluation=driver_prior_evaluation,
    )


def build_validation_model_comparison(
    baseline_evaluation: WinnerModelEvaluation,
    benchmarks: ValidationBenchmarkResult,
) -> pd.DataFrame:
    """Build an apples-to-apples validation metric comparison table."""
    if not isinstance(
        baseline_evaluation,
        WinnerModelEvaluation,
    ):
        raise TypeError("baseline_evaluation must be a WinnerModelEvaluation.")

    if not isinstance(
        benchmarks,
        ValidationBenchmarkResult,
    ):
        raise TypeError("benchmarks must be a ValidationBenchmarkResult.")

    evaluations = (
        (
            "Uniform",
            benchmarks.uniform_evaluation,
        ),
        (
            "Driver prior",
            benchmarks.driver_prior_evaluation,
        ),
        (
            "TelemetryX logistic",
            baseline_evaluation,
        ),
    )

    rows: list[dict[str, object]] = []

    for model_name, evaluation in evaluations:
        summary = evaluation.summary

        rows.append(
            {
                "Model": model_name,
                "Snapshots": summary.snapshot_count,
                "Races": summary.race_count,
                "LogLoss": summary.mean_log_loss,
                "BrierScore": summary.mean_brier_score,
                "TopOneAccuracy": summary.top_one_accuracy,
                "MeanWinnerProbability": (summary.mean_actual_winner_probability),
            }
        )

    comparison = pd.DataFrame(rows)

    _validate_comparable_evaluations(comparison)

    return comparison


def _validate_comparable_evaluations(
    comparison: pd.DataFrame,
) -> None:
    """Require every comparison row to cover the same validation sample."""
    if comparison.empty:
        raise BenchmarkEvaluationError("Validation model comparison cannot be empty.")

    if comparison["Snapshots"].nunique() != 1:
        raise BenchmarkEvaluationError(
            "Compared models must evaluate the same snapshots."
        )

    if comparison["Races"].nunique() != 1:
        raise BenchmarkEvaluationError("Compared models must evaluate the same races.")
