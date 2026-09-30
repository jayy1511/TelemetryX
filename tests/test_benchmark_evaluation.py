from __future__ import annotations

import pandas as pd
import pytest

import telemetryx.modeling.benchmark_evaluation as benchmark_evaluation
from telemetryx.modeling.benchmark_evaluation import (
    BenchmarkEvaluationError,
    ValidationBenchmarkResult,
    build_validation_model_comparison,
    evaluate_validation_benchmarks,
)
from telemetryx.modeling.benchmarks import BenchmarkPredictionError
from telemetryx.modeling.evaluation import (
    EvaluationSummary,
    ModelEvaluationError,
    WinnerModelEvaluation,
)


def make_evaluation(
    *,
    snapshot_count: int = 3,
    race_count: int = 1,
    row_count: int = 6,
    mean_log_loss: float = 0.5,
    mean_brier_score: float = 0.3,
    top_one_accuracy: float = 0.6,
    mean_winner_probability: float = 0.7,
) -> WinnerModelEvaluation:
    """Build a small evaluation result for comparison tests."""
    summary = EvaluationSummary(
        snapshot_count=snapshot_count,
        race_count=race_count,
        row_count=row_count,
        mean_log_loss=mean_log_loss,
        mean_brier_score=mean_brier_score,
        top_one_accuracy=top_one_accuracy,
        mean_actual_winner_probability=mean_winner_probability,
    )

    snapshots = pd.DataFrame(
        {
            "RaceId": [
                "race_a",
            ]
            * snapshot_count,
            "SnapshotLap": list(
                range(
                    1,
                    snapshot_count + 1,
                )
            ),
        }
    )

    return WinnerModelEvaluation(
        summary=summary,
        snapshots=snapshots,
    )


def make_benchmark_result() -> ValidationBenchmarkResult:
    """Build deterministic benchmark evaluations."""
    uniform_predictions = pd.DataFrame(
        {
            "RaceId": [
                "race_a",
            ],
            "SnapshotLap": [
                1,
            ],
            "Driver": [
                "VER",
            ],
            "WinnerProbability": [
                1.0,
            ],
        }
    )

    driver_prior_predictions = uniform_predictions.copy(deep=True)

    return ValidationBenchmarkResult(
        uniform_predictions=uniform_predictions,
        uniform_evaluation=make_evaluation(
            mean_log_loss=2.0,
            mean_brier_score=0.8,
            top_one_accuracy=0.1,
            mean_winner_probability=0.2,
        ),
        driver_prior_predictions=driver_prior_predictions,
        driver_prior_evaluation=make_evaluation(
            mean_log_loss=0.9,
            mean_brier_score=0.4,
            top_one_accuracy=0.5,
            mean_winner_probability=0.6,
        ),
    )


def test_evaluate_validation_benchmarks_uses_validation_for_uniform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Uniform prediction should operate on validation features only."""
    training = pd.DataFrame(
        {
            "source": [
                "train",
            ]
        }
    )

    validation = pd.DataFrame(
        {
            "source": [
                "validation",
            ]
        }
    )

    uniform_predictions = pd.DataFrame(
        {
            "kind": [
                "uniform",
            ]
        }
    )

    prior_predictions = pd.DataFrame(
        {
            "kind": [
                "prior",
            ]
        }
    )

    uniform_evaluation = make_evaluation()
    prior_evaluation = make_evaluation()

    def fake_uniform_predictor(
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        assert features is validation
        return uniform_predictions

    def fake_prior_predictor(
        training_features: pd.DataFrame,
        prediction_features: pd.DataFrame,
    ) -> pd.DataFrame:
        assert training_features is training
        assert prediction_features is validation
        return prior_predictions

    evaluations: list[
        tuple[
            pd.DataFrame,
            pd.DataFrame,
        ]
    ] = []

    def fake_evaluator(
        features: pd.DataFrame,
        predictions: pd.DataFrame,
    ) -> WinnerModelEvaluation:
        evaluations.append(
            (
                features,
                predictions,
            )
        )

        if predictions is uniform_predictions:
            return uniform_evaluation

        return prior_evaluation

    monkeypatch.setattr(
        benchmark_evaluation,
        "predict_uniform_winner_probabilities",
        fake_uniform_predictor,
    )

    monkeypatch.setattr(
        benchmark_evaluation,
        "predict_driver_prior_winner_probabilities",
        fake_prior_predictor,
    )

    monkeypatch.setattr(
        benchmark_evaluation,
        "evaluate_winner_probabilities",
        fake_evaluator,
    )

    result = evaluate_validation_benchmarks(
        training,
        validation,
    )

    assert result.uniform_predictions is uniform_predictions
    assert result.driver_prior_predictions is prior_predictions
    assert result.uniform_evaluation is uniform_evaluation
    assert result.driver_prior_evaluation is prior_evaluation

    assert evaluations == [
        (
            validation,
            uniform_predictions,
        ),
        (
            validation,
            prior_predictions,
        ),
    ]


def test_evaluate_validation_benchmarks_does_not_accept_test_partition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Benchmark orchestration should require only train and validation."""
    training = pd.DataFrame(
        {
            "split": [
                "train",
            ]
        }
    )

    validation = pd.DataFrame(
        {
            "split": [
                "validation",
            ]
        }
    )

    observed_frames: list[pd.DataFrame] = []

    def fake_uniform_predictor(
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        observed_frames.append(features)
        return pd.DataFrame(
            {
                "prediction": [
                    1,
                ]
            }
        )

    def fake_prior_predictor(
        training_features: pd.DataFrame,
        prediction_features: pd.DataFrame,
    ) -> pd.DataFrame:
        observed_frames.extend(
            [
                training_features,
                prediction_features,
            ]
        )

        return pd.DataFrame(
            {
                "prediction": [
                    2,
                ]
            }
        )

    def fake_evaluator(
        features: pd.DataFrame,
        predictions: pd.DataFrame,
    ) -> WinnerModelEvaluation:
        observed_frames.append(features)
        return make_evaluation()

    monkeypatch.setattr(
        benchmark_evaluation,
        "predict_uniform_winner_probabilities",
        fake_uniform_predictor,
    )

    monkeypatch.setattr(
        benchmark_evaluation,
        "predict_driver_prior_winner_probabilities",
        fake_prior_predictor,
    )

    monkeypatch.setattr(
        benchmark_evaluation,
        "evaluate_winner_probabilities",
        fake_evaluator,
    )

    evaluate_validation_benchmarks(
        training,
        validation,
    )

    assert all(frame is training or frame is validation for frame in observed_frames)


def test_evaluate_validation_benchmarks_wraps_prediction_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Benchmark prediction errors should receive orchestration context."""

    def fail_uniform(
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        raise BenchmarkPredictionError("synthetic prediction failure")

    monkeypatch.setattr(
        benchmark_evaluation,
        "predict_uniform_winner_probabilities",
        fail_uniform,
    )

    with pytest.raises(
        BenchmarkEvaluationError,
        match="Failed to evaluate validation benchmarks",
    ):
        evaluate_validation_benchmarks(
            pd.DataFrame(
                {
                    "train": [
                        1,
                    ]
                }
            ),
            pd.DataFrame(
                {
                    "validation": [
                        1,
                    ]
                }
            ),
        )


def test_evaluate_validation_benchmarks_wraps_evaluation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model-evaluation errors should receive benchmark context."""
    predictions = pd.DataFrame(
        {
            "prediction": [
                1,
            ]
        }
    )

    monkeypatch.setattr(
        benchmark_evaluation,
        "predict_uniform_winner_probabilities",
        lambda features: predictions,
    )

    def fail_evaluation(
        features: pd.DataFrame,
        prediction_frame: pd.DataFrame,
    ) -> WinnerModelEvaluation:
        raise ModelEvaluationError("synthetic evaluation failure")

    monkeypatch.setattr(
        benchmark_evaluation,
        "evaluate_winner_probabilities",
        fail_evaluation,
    )

    with pytest.raises(
        BenchmarkEvaluationError,
        match="Failed to evaluate validation benchmarks",
    ):
        evaluate_validation_benchmarks(
            pd.DataFrame(
                {
                    "train": [
                        1,
                    ]
                }
            ),
            pd.DataFrame(
                {
                    "validation": [
                        1,
                    ]
                }
            ),
        )


def test_evaluate_validation_benchmarks_rejects_non_dataframe_training() -> None:
    """Training input must be represented as a DataFrame."""
    with pytest.raises(
        TypeError,
        match="training_corpus must be a pandas DataFrame",
    ):
        evaluate_validation_benchmarks(
            None,  # type: ignore[arg-type]
            pd.DataFrame(
                {
                    "validation": [
                        1,
                    ]
                }
            ),
        )


def test_evaluate_validation_benchmarks_rejects_non_dataframe_validation() -> None:
    """Validation input must be represented as a DataFrame."""
    with pytest.raises(
        TypeError,
        match="validation_features must be a pandas DataFrame",
    ):
        evaluate_validation_benchmarks(
            pd.DataFrame(
                {
                    "train": [
                        1,
                    ]
                }
            ),
            None,  # type: ignore[arg-type]
        )


def test_model_comparison_contains_all_three_models() -> None:
    """Comparison should include both benchmarks and TelemetryX."""
    baseline = make_evaluation(
        mean_log_loss=0.1,
        mean_brier_score=0.05,
        top_one_accuracy=0.95,
        mean_winner_probability=0.9,
    )

    comparison = build_validation_model_comparison(
        baseline,
        make_benchmark_result(),
    )

    assert comparison["Model"].tolist() == [
        "Uniform",
        "Driver prior",
        "TelemetryX logistic",
    ]


def test_model_comparison_uses_expected_schema() -> None:
    """Comparison output should expose the shared validation metrics."""
    comparison = build_validation_model_comparison(
        make_evaluation(),
        make_benchmark_result(),
    )

    assert comparison.columns.tolist() == [
        "Model",
        "Snapshots",
        "Races",
        "LogLoss",
        "BrierScore",
        "TopOneAccuracy",
        "MeanWinnerProbability",
    ]


def test_model_comparison_preserves_summary_metrics() -> None:
    """Each model row should contain metrics from its own evaluation."""
    baseline = make_evaluation(
        mean_log_loss=0.12,
        mean_brier_score=0.06,
        top_one_accuracy=0.97,
        mean_winner_probability=0.91,
    )

    comparison = build_validation_model_comparison(
        baseline,
        make_benchmark_result(),
    )

    logistic = comparison.loc[comparison["Model"].eq("TelemetryX logistic")].iloc[0]

    assert logistic["LogLoss"] == pytest.approx(0.12)

    assert logistic["BrierScore"] == pytest.approx(0.06)

    assert logistic["TopOneAccuracy"] == pytest.approx(0.97)

    assert logistic["MeanWinnerProbability"] == pytest.approx(0.91)


def test_model_comparison_requires_same_snapshot_count() -> None:
    """Every model must be evaluated over the same snapshots."""
    benchmarks = make_benchmark_result()

    baseline = make_evaluation(
        snapshot_count=4,
    )

    with pytest.raises(
        BenchmarkEvaluationError,
        match="same snapshots",
    ):
        build_validation_model_comparison(
            baseline,
            benchmarks,
        )


def test_model_comparison_requires_same_race_count() -> None:
    """Every model must be evaluated over the same races."""
    benchmarks = make_benchmark_result()

    baseline = make_evaluation(
        race_count=2,
    )

    with pytest.raises(
        BenchmarkEvaluationError,
        match="same races",
    ):
        build_validation_model_comparison(
            baseline,
            benchmarks,
        )


def test_model_comparison_rejects_invalid_baseline_type() -> None:
    """The logistic result must use the common evaluation representation."""
    with pytest.raises(
        TypeError,
        match="baseline_evaluation must be a WinnerModelEvaluation",
    ):
        build_validation_model_comparison(
            None,  # type: ignore[arg-type]
            make_benchmark_result(),
        )


def test_model_comparison_rejects_invalid_benchmark_type() -> None:
    """Benchmark results must use the benchmark result container."""
    with pytest.raises(
        TypeError,
        match="benchmarks must be a ValidationBenchmarkResult",
    ):
        build_validation_model_comparison(
            make_evaluation(),
            None,  # type: ignore[arg-type]
        )
