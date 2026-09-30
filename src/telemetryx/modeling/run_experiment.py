from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from telemetryx.data.build_corpus import (
    RaceCorpusBuildError,
    load_race_corpus_artifact,
)
from telemetryx.modeling.benchmark_evaluation import (
    build_validation_model_comparison,
    evaluate_validation_benchmarks,
)
from telemetryx.modeling.experiment import (
    BaselineExperimentError,
    BaselineExperimentResult,
    run_baseline_experiment,
)
from telemetryx.modeling.stage_evaluation import summarize_race_stages

DEFAULT_CORPUS_PATH: Final[Path] = Path(
    "data/processed/corpora/telemetryx_race_corpus.parquet"
)

DEFAULT_VALIDATION_SEASON: Final[int] = 2023
DEFAULT_VALIDATION_LAST_RACES: Final[int] = 5

DEFAULT_TEST_SEASONS: Final[tuple[int, ...]] = (2024,)


class ExperimentRunnerError(RuntimeError):
    """Raised when the command-line baseline experiment cannot run."""


def run_experiment_from_corpus(
    corpus_path: Path,
    *,
    validation_season: int = DEFAULT_VALIDATION_SEASON,
    validation_last_races: int = DEFAULT_VALIDATION_LAST_RACES,
    test_seasons: Sequence[int] = DEFAULT_TEST_SEASONS,
    max_iterations: int = 2000,
) -> BaselineExperimentResult:
    """
    Load a processed corpus and run the baseline validation experiment.

    Parameters
    ----------
    corpus_path:
        Path to the processed TelemetryX race corpus Parquet file.
    validation_season:
        Season whose final races are reserved for validation.
    validation_last_races:
        Number of final validation-season races to reserve.
    test_seasons:
        Complete future seasons reserved as the untouched test holdout.
    max_iterations:
        Maximum logistic-regression optimization iterations.

    Returns
    -------
    BaselineExperimentResult
        Fitted model, chronological split and validation evaluation.

    Raises
    ------
    TypeError
        If ``corpus_path`` is not a pathlib Path.
    ExperimentRunnerError
        If corpus loading or experiment execution fails.
    """
    if not isinstance(
        corpus_path,
        Path,
    ):
        raise TypeError("corpus_path must be provided as a pathlib Path.")

    try:
        corpus = load_race_corpus_artifact(corpus_path)
    except (
        FileNotFoundError,
        OSError,
        TypeError,
        ValueError,
        RaceCorpusBuildError,
    ) as exc:
        raise ExperimentRunnerError(
            "Failed to load the processed TelemetryX race corpus."
        ) from exc

    try:
        return run_baseline_experiment(
            corpus,
            validation_season=validation_season,
            validation_last_races=validation_last_races,
            test_seasons=test_seasons,
            max_iterations=max_iterations,
        )
    except (
        TypeError,
        BaselineExperimentError,
    ) as exc:
        raise ExperimentRunnerError(
            "Failed to run the TelemetryX baseline experiment."
        ) from exc


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for baseline experiments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the TelemetryX chronological baseline winner-probability experiment."
        )
    )

    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help=(
            "Path to the processed race corpus Parquet file. "
            f"Default: {DEFAULT_CORPUS_PATH}"
        ),
    )

    parser.add_argument(
        "--validation-season",
        type=int,
        default=DEFAULT_VALIDATION_SEASON,
        help=(
            "Season whose final races are reserved for validation. "
            f"Default: {DEFAULT_VALIDATION_SEASON}"
        ),
    )

    parser.add_argument(
        "--validation-last-races",
        type=int,
        default=DEFAULT_VALIDATION_LAST_RACES,
        help=(
            "Number of final validation-season races to reserve. "
            f"Default: {DEFAULT_VALIDATION_LAST_RACES}"
        ),
    )

    parser.add_argument(
        "--test-seasons",
        type=_parse_seasons,
        default=DEFAULT_TEST_SEASONS,
        help=(
            "Comma-separated complete seasons reserved for final testing. "
            "Example: 2024 or 2024,2025. "
            "Default: 2024"
        ),
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=2000,
        help=("Maximum logistic-regression optimization iterations. Default: 2000"),
    )

    return parser


def print_experiment_result(
    result: BaselineExperimentResult,
) -> None:
    """Print a human-readable validation experiment report."""
    summary = result.summary

    print()
    print("TelemetryX baseline experiment")
    print("=" * 60)

    print(f"Train races:      {summary.train_race_count}")
    print(f"Validation races: {summary.validation_race_count}")
    print(f"Test races:       {summary.test_race_count}")

    print()

    print(f"Train rows:       {summary.train_row_count:,}")
    print(f"Validation rows:  {summary.validation_row_count:,}")
    print(f"Test rows:        {summary.test_row_count:,}")

    print()

    print(f"Validation snapshots: {summary.validation_snapshot_count:,}")

    print(f"Validation log loss: {summary.validation_mean_log_loss:.6f}")

    print(f"Validation Brier score: {summary.validation_mean_brier_score:.6f}")

    print(f"Validation top-1 accuracy: {summary.validation_top_one_accuracy:.2%}")

    print(
        "Mean actual-winner probability: "
        f"{summary.validation_mean_winner_probability:.2%}"
    )

    print()

    print("Validation races:")
    print(
        result.validation_features.loc[
            :,
            [
                "RaceId",
                "Season",
                "RoundNumber",
            ],
        ]
        .drop_duplicates()
        .sort_values(
            by=[
                "Season",
                "RoundNumber",
            ],
            kind="stable",
        )
        .to_string(index=False)
    )

    benchmarks = evaluate_validation_benchmarks(
        result.split.train,
        result.validation_features,
    )

    comparison = build_validation_model_comparison(
        result.validation_evaluation,
        benchmarks,
    )

    print()
    print("Validation benchmark comparison:")
    print(
        "Model                Snapshots  Races  LogLoss   Brier     Top-1    WinnerProb"
    )

    for row in comparison.itertuples(index=False):
        print(
            f"{row.Model:<20}"
            f"{row.Snapshots:>9}  "
            f"{row.Races:>5}  "
            f"{row.LogLoss:>7.4f}  "
            f"{row.BrierScore:>7.4f}  "
            f"{row.TopOneAccuracy:>7.2%}  "
            f"{row.MeanWinnerProbability:>10.2%}"
        )
    stage_summary = summarize_race_stages(result.validation_evaluation.snapshots)

    print()
    print("Validation performance by race stage:")
    print(
        "Stage       Snapshots  Races  "
        "LogLoss   Brier     Top-1    "
        "WinnerProb  WinnerRank"
    )

    for row in stage_summary.itertuples(index=False):
        print(
            f"{row.RaceStage:<11}"
            f"{row.SnapshotCount:>9}  "
            f"{row.RaceCount:>5}  "
            f"{row.MeanLogLoss:>7.4f}  "
            f"{row.MeanBrierScore:>7.4f}  "
            f"{row.TopOneAccuracy:>7.2%}  "
            f"{row.MeanWinnerProbability:>10.2%}  "
            f"{row.MeanWinnerRank:>10.2f}"
        )

    print()

    print("Final 10 validation snapshots:")
    print(result.validation_evaluation.snapshots.tail(10).to_string(index=False))


def main(
    argv: Sequence[str] | None = None,
) -> None:
    """Run the TelemetryX baseline experiment CLI."""
    parser = build_argument_parser()

    args = parser.parse_args(argv)

    try:
        result = run_experiment_from_corpus(
            args.corpus,
            validation_season=(args.validation_season),
            validation_last_races=(args.validation_last_races),
            test_seasons=args.test_seasons,
            max_iterations=(args.max_iterations),
        )
    except ExperimentRunnerError as exc:
        parser.error(str(exc))

    print_experiment_result(result)


def _parse_seasons(
    value: str,
) -> tuple[int, ...]:
    """Parse a comma-separated list of positive season integers."""
    parts = [part.strip() for part in value.split(",") if part.strip()]

    if not parts:
        raise argparse.ArgumentTypeError("At least one test season is required.")

    seasons: list[int] = []

    for part in parts:
        try:
            season = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "Test seasons must be comma-separated integers."
            ) from exc

        if season <= 0:
            raise argparse.ArgumentTypeError("Test seasons must be positive integers.")

        seasons.append(season)

    if len(seasons) != len(set(seasons)):
        raise argparse.ArgumentTypeError("Test seasons cannot contain duplicates.")

    return tuple(seasons)


if __name__ == "__main__":
    main()
