from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import telemetryx.modeling.run_experiment as runner_module
from telemetryx.modeling.experiment import BaselineExperimentError
from telemetryx.modeling.run_experiment import (
    DEFAULT_CORPUS_PATH,
    DEFAULT_TEST_SEASONS,
    DEFAULT_VALIDATION_LAST_RACES,
    DEFAULT_VALIDATION_SEASON,
    ExperimentRunnerError,
    build_argument_parser,
    main,
    print_experiment_result,
    run_experiment_from_corpus,
)


def make_fake_result() -> Any:
    """Return a minimal experiment-like object for CLI report testing."""
    summary = SimpleNamespace(
        train_race_count=17,
        validation_race_count=5,
        test_race_count=24,
        train_row_count=20210,
        validation_row_count=6049,
        test_row_count=28632,
        validation_snapshot_count=306,
        validation_mean_log_loss=0.114718,
        validation_mean_brier_score=0.060974,
        validation_top_one_accuracy=0.9706,
        validation_mean_winner_probability=0.9133,
    )

    validation_features = pd.DataFrame(
        {
            "RaceId": [
                "2023_18_united_states_grand_prix",
                "2023_19_mexico_city_grand_prix",
            ],
            "Season": [
                2023,
                2023,
            ],
            "RoundNumber": [
                18,
                19,
            ],
        }
    )

    snapshots = pd.DataFrame(
        {
            "RaceId": [
                "2023_19_mexico_city_grand_prix",
            ],
            "Season": [
                2023,
            ],
            "RoundNumber": [
                19,
            ],
            "SnapshotLap": [
                71,
            ],
            "ActualWinner": [
                "VER",
            ],
            "ActualWinnerProbability": [
                0.95,
            ],
            "ActualWinnerRank": [
                1,
            ],
            "SnapshotLogLoss": [
                0.05,
            ],
            "SnapshotBrierScore": [
                0.01,
            ],
            "TopOneCorrect": [
                True,
            ],
        }
    )

    evaluation = SimpleNamespace(snapshots=snapshots)

    return SimpleNamespace(
        summary=summary,
        validation_features=validation_features,
        validation_evaluation=evaluation,
    )


def test_default_experiment_configuration() -> None:
    """CLI defaults should encode the TelemetryX MVP split."""
    assert DEFAULT_CORPUS_PATH == Path(
        "data/processed/corpora/telemetryx_race_corpus.parquet"
    )

    assert DEFAULT_VALIDATION_SEASON == 2023

    assert DEFAULT_VALIDATION_LAST_RACES == 5

    assert DEFAULT_TEST_SEASONS == (2024,)


def test_argument_parser_uses_default_configuration() -> None:
    """Running without arguments should use the MVP experiment policy."""
    parser = build_argument_parser()

    args = parser.parse_args([])

    assert args.corpus == DEFAULT_CORPUS_PATH

    assert args.validation_season == DEFAULT_VALIDATION_SEASON

    assert args.validation_last_races == DEFAULT_VALIDATION_LAST_RACES

    assert args.test_seasons == (2024,)

    assert args.max_iterations == 2000


def test_argument_parser_accepts_overrides() -> None:
    """Experiment settings should be configurable from the command line."""
    parser = build_argument_parser()

    args = parser.parse_args(
        [
            "--corpus",
            "custom/corpus.parquet",
            "--validation-season",
            "2024",
            "--validation-last-races",
            "3",
            "--test-seasons",
            "2025,2026",
            "--max-iterations",
            "5000",
        ]
    )

    assert args.corpus == Path("custom/corpus.parquet")

    assert args.validation_season == 2024
    assert args.validation_last_races == 3

    assert args.test_seasons == (
        2025,
        2026,
    )

    assert args.max_iterations == 5000


@pytest.mark.parametrize(
    "value",
    [
        "",
        ",",
        "   ",
    ],
)
def test_argument_parser_rejects_empty_test_seasons(
    value: str,
) -> None:
    """At least one explicit test season must be supplied."""
    parser = build_argument_parser()

    with pytest.raises(
        SystemExit,
    ):
        parser.parse_args(
            [
                "--test-seasons",
                value,
            ]
        )


@pytest.mark.parametrize(
    "value",
    [
        "abc",
        "2024,abc",
        "2024.5",
    ],
)
def test_argument_parser_rejects_non_integer_test_seasons(
    value: str,
) -> None:
    """Test seasons must be integer calendar years."""
    parser = build_argument_parser()

    with pytest.raises(
        SystemExit,
    ):
        parser.parse_args(
            [
                "--test-seasons",
                value,
            ]
        )


@pytest.mark.parametrize(
    "value",
    [
        "0",
        "-1",
        "2024,0",
    ],
)
def test_argument_parser_rejects_non_positive_test_seasons(
    value: str,
) -> None:
    """Test-season values must be positive."""
    parser = build_argument_parser()

    with pytest.raises(
        SystemExit,
    ):
        parser.parse_args(
            [
                "--test-seasons",
                value,
            ]
        )


def test_argument_parser_rejects_duplicate_test_seasons() -> None:
    """The same test season cannot be configured twice."""
    parser = build_argument_parser()

    with pytest.raises(
        SystemExit,
    ):
        parser.parse_args(
            [
                "--test-seasons",
                "2024,2024",
            ]
        )


def test_run_experiment_requires_path() -> None:
    """The programmatic runner requires pathlib Path input."""
    invalid_path: Any = "corpus.parquet"

    with pytest.raises(
        TypeError,
        match=("corpus_path must be provided as a pathlib Path"),
    ):
        run_experiment_from_corpus(invalid_path)


def test_run_experiment_loads_corpus_and_forwards_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner should connect corpus loading to experiment execution."""
    corpus = pd.DataFrame(
        {
            "example": [
                1,
            ]
        }
    )

    expected_result = object()

    loaded_paths: list[Path] = []

    received_arguments: list[dict[str, Any]] = []

    def fake_loader(
        path: Path,
    ) -> pd.DataFrame:
        loaded_paths.append(path)

        return corpus

    def fake_experiment(
        received_corpus: pd.DataFrame,
        **kwargs: Any,
    ) -> Any:
        assert received_corpus is corpus

        received_arguments.append(kwargs)

        return expected_result

    monkeypatch.setattr(
        runner_module,
        "load_race_corpus_artifact",
        fake_loader,
    )

    monkeypatch.setattr(
        runner_module,
        "run_baseline_experiment",
        fake_experiment,
    )

    path = Path("example.parquet")

    result = run_experiment_from_corpus(
        path,
        validation_season=2022,
        validation_last_races=4,
        test_seasons=(
            2023,
            2024,
        ),
        max_iterations=1234,
    )

    assert result is expected_result

    assert loaded_paths == [
        path,
    ]

    assert received_arguments == [
        {
            "validation_season": 2022,
            "validation_last_races": 4,
            "test_seasons": (
                2023,
                2024,
            ),
            "max_iterations": 1234,
        }
    ]


def test_run_experiment_wraps_corpus_load_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Corpus loading failures should receive CLI-specific context."""

    def failing_loader(
        path: Path,
    ) -> pd.DataFrame:
        raise FileNotFoundError(path)

    monkeypatch.setattr(
        runner_module,
        "load_race_corpus_artifact",
        failing_loader,
    )

    with pytest.raises(
        ExperimentRunnerError,
        match=("Failed to load the processed TelemetryX race corpus"),
    ):
        run_experiment_from_corpus(Path("missing.parquet"))


def test_run_experiment_wraps_baseline_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Experiment failures should receive runner-specific context."""
    corpus = pd.DataFrame(
        {
            "example": [
                1,
            ]
        }
    )

    def fake_loader(
        path: Path,
    ) -> pd.DataFrame:
        return corpus

    def failing_experiment(
        received_corpus: pd.DataFrame,
        **kwargs: Any,
    ) -> Any:
        raise BaselineExperimentError("synthetic failure")

    monkeypatch.setattr(
        runner_module,
        "load_race_corpus_artifact",
        fake_loader,
    )

    monkeypatch.setattr(
        runner_module,
        "run_baseline_experiment",
        failing_experiment,
    )

    with pytest.raises(
        ExperimentRunnerError,
        match=("Failed to run the TelemetryX baseline experiment"),
    ):
        run_experiment_from_corpus(Path("corpus.parquet"))


def test_print_experiment_result_contains_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI report should expose split sizes and validation metrics."""
    print_experiment_result(make_fake_result())

    output = capsys.readouterr().out

    assert "TelemetryX baseline experiment" in output

    assert "Train races:      17" in output

    assert "Validation races: 5" in output

    assert "Test races:       24" in output

    assert "Train rows:       20,210" in output

    assert "Validation rows:  6,049" in output

    assert "Test rows:        28,632" in output

    assert "Validation snapshots: 306" in output

    assert "Validation log loss: 0.114718" in output

    assert "Validation Brier score: 0.060974" in output

    assert "Validation top-1 accuracy: 97.06%" in output

    assert "Mean actual-winner probability: 91.33%" in output


def test_print_experiment_result_lists_validation_races(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The report should identify races used for model validation."""
    print_experiment_result(make_fake_result())

    output = capsys.readouterr().out

    assert "2023_18_united_states_grand_prix" in output

    assert "2023_19_mexico_city_grand_prix" in output


def test_main_forwards_command_line_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """main() should pass parsed arguments to the programmatic runner."""
    expected_result = object()

    received_arguments: list[
        tuple[
            Path,
            dict[str, Any],
        ]
    ] = []

    printed_results: list[object] = []

    def fake_run(
        corpus_path: Path,
        **kwargs: Any,
    ) -> Any:
        received_arguments.append(
            (
                corpus_path,
                kwargs,
            )
        )

        return expected_result

    def fake_print(
        result: Any,
    ) -> None:
        printed_results.append(result)

    monkeypatch.setattr(
        runner_module,
        "run_experiment_from_corpus",
        fake_run,
    )

    monkeypatch.setattr(
        runner_module,
        "print_experiment_result",
        fake_print,
    )

    main(
        [
            "--corpus",
            "example.parquet",
            "--validation-season",
            "2022",
            "--validation-last-races",
            "4",
            "--test-seasons",
            "2023,2024",
            "--max-iterations",
            "3000",
        ]
    )

    assert received_arguments == [
        (
            Path("example.parquet"),
            {
                "validation_season": 2022,
                "validation_last_races": 4,
                "test_seasons": (
                    2023,
                    2024,
                ),
                "max_iterations": 3000,
            },
        )
    ]

    assert printed_results == [
        expected_result,
    ]


def test_main_uses_default_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running the CLI without arguments should use the MVP defaults."""
    expected_result = object()

    received_arguments: list[
        tuple[
            Path,
            dict[str, Any],
        ]
    ] = []

    def fake_run(
        corpus_path: Path,
        **kwargs: Any,
    ) -> Any:
        received_arguments.append(
            (
                corpus_path,
                kwargs,
            )
        )

        return expected_result

    monkeypatch.setattr(
        runner_module,
        "run_experiment_from_corpus",
        fake_run,
    )

    monkeypatch.setattr(
        runner_module,
        "print_experiment_result",
        lambda result: None,
    )

    main([])

    assert received_arguments == [
        (
            DEFAULT_CORPUS_PATH,
            {
                "validation_season": (DEFAULT_VALIDATION_SEASON),
                "validation_last_races": (DEFAULT_VALIDATION_LAST_RACES),
                "test_seasons": (DEFAULT_TEST_SEASONS),
                "max_iterations": 2000,
            },
        )
    ]
