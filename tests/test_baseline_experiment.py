"""Tests for the end-to-end TelemetryX baseline experiment."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

import telemetryx.modeling.experiment as experiment_module
from telemetryx.data.corpus import combine_race_datasets
from telemetryx.data.dataset import build_race_dataset
from telemetryx.features.engineering import engineer_race_features
from telemetryx.modeling.baseline import (
    BaselineWinnerModel,
)
from telemetryx.modeling.evaluation import (
    WinnerModelEvaluation,
)
from telemetryx.modeling.experiment import (
    BaselineExperimentError,
    BaselineExperimentResult,
    BaselineExperimentSummary,
    run_baseline_experiment,
)


def make_cleaned_laps(
    *,
    drivers: tuple[str, str, str],
    lap_time_offset: float = 0.0,
) -> pd.DataFrame:
    """Return a deterministic three-driver, three-lap cleaned race."""
    rows: list[dict[str, object]] = []

    for position, driver in enumerate(
        drivers,
        start=1,
    ):
        base_lap_time = 89.0 + position + lap_time_offset

        for lap_number in (
            1,
            2,
            3,
        ):
            rows.append(
                {
                    "Driver": driver,
                    "LapNumber": lap_number,
                    "Position": position,
                    "Stint": (1 if lap_number < 3 else 2),
                    "Compound": ("SOFT" if lap_number < 3 else "MEDIUM"),
                    "TyreLife": (float(lap_number) if lap_number < 3 else 1.0),
                    "TrackStatus": "1",
                    "LapTimeSeconds": (base_lap_time - (lap_number - 1)),
                }
            )

    return pd.DataFrame(rows)


def make_results(
    *,
    drivers: tuple[str, str, str],
) -> pd.DataFrame:
    """Return final results with the first supplied driver winning."""
    return pd.DataFrame(
        {
            "Abbreviation": list(drivers),
            "Position": [
                1,
                2,
                3,
            ],
            "Status": [
                "Finished",
                "Finished",
                "Finished",
            ],
        }
    )


def make_race_dataset(
    *,
    season: int,
    round_number: int,
    event_name: str,
    drivers: tuple[str, str, str],
    lap_time_offset: float = 0.0,
) -> pd.DataFrame:
    """Return one valid synthetic TelemetryX race dataset."""
    return build_race_dataset(
        make_cleaned_laps(
            drivers=drivers,
            lap_time_offset=lap_time_offset,
        ),
        make_results(
            drivers=drivers,
        ),
        season=season,
        round_number=round_number,
        event_name=event_name,
        session_name="Race",
    )


def make_experiment_corpus(
    *,
    test_lap_time_offset: float = 0.0,
) -> pd.DataFrame:
    """
    Return a corpus with explicit train, validation and test eras.

    Expected split:

    2022 R1-R2       -> train
    2023 R1-R2       -> train
    2023 R3-R4       -> validation
    2024 R1-R2       -> test
    """
    races = [
        make_race_dataset(
            season=2024,
            round_number=2,
            event_name="Saudi Arabian Grand Prix",
            drivers=(
                "SAI",
                "PER",
                "TSU",
            ),
            lap_time_offset=(40.0 + test_lap_time_offset),
        ),
        make_race_dataset(
            season=2023,
            round_number=4,
            event_name="Azerbaijan Grand Prix",
            drivers=(
                "LEC",
                "VER",
                "NOR",
            ),
            lap_time_offset=25.0,
        ),
        make_race_dataset(
            season=2022,
            round_number=1,
            event_name="Bahrain Grand Prix",
            drivers=(
                "VER",
                "NOR",
                "BOT",
            ),
            lap_time_offset=0.0,
        ),
        make_race_dataset(
            season=2023,
            round_number=1,
            event_name="Bahrain Grand Prix",
            drivers=(
                "BOT",
                "VER",
                "NOR",
            ),
            lap_time_offset=10.0,
        ),
        make_race_dataset(
            season=2024,
            round_number=1,
            event_name="Bahrain Grand Prix",
            drivers=(
                "HAM",
                "RUS",
                "ALO",
            ),
            lap_time_offset=(35.0 + test_lap_time_offset),
        ),
        make_race_dataset(
            season=2022,
            round_number=2,
            event_name="Saudi Arabian Grand Prix",
            drivers=(
                "NOR",
                "VER",
                "BOT",
            ),
            lap_time_offset=5.0,
        ),
        make_race_dataset(
            season=2023,
            round_number=3,
            event_name="Australian Grand Prix",
            drivers=(
                "PIA",
                "NOR",
                "BOT",
            ),
            lap_time_offset=20.0,
        ),
        make_race_dataset(
            season=2023,
            round_number=2,
            event_name="Saudi Arabian Grand Prix",
            drivers=(
                "VER",
                "BOT",
                "NOR",
            ),
            lap_time_offset=15.0,
        ),
    ]

    return combine_race_datasets(races)


def run_standard_experiment(
    corpus: pd.DataFrame | None = None,
) -> BaselineExperimentResult:
    """Run the standard synthetic baseline experiment."""
    active_corpus = make_experiment_corpus() if corpus is None else corpus

    return run_baseline_experiment(
        active_corpus,
        validation_season=2023,
        validation_last_races=2,
        test_seasons=(2024,),
    )


def race_ids(
    frame: pd.DataFrame,
) -> set[str]:
    """Return unique race identifiers represented by a frame."""
    return {
        str(value) for value in (frame["RaceId"].astype("string").dropna().tolist())
    }


def test_experiment_returns_complete_result() -> None:
    """The workflow should return all expected experiment components."""
    result = run_standard_experiment()

    assert isinstance(
        result,
        BaselineExperimentResult,
    )

    assert isinstance(
        result.summary,
        BaselineExperimentSummary,
    )

    assert isinstance(
        result.model,
        BaselineWinnerModel,
    )

    assert isinstance(
        result.validation_evaluation,
        WinnerModelEvaluation,
    )

    assert isinstance(
        result.validation_features,
        pd.DataFrame,
    )

    assert isinstance(
        result.validation_predictions,
        pd.DataFrame,
    )


def test_experiment_uses_expected_race_split_counts() -> None:
    """The experiment must preserve the configured chronological split."""
    result = run_standard_experiment()

    assert result.split.train_race_count == 4
    assert result.split.validation_race_count == 2
    assert result.split.test_race_count == 2


def test_experiment_summary_uses_expected_race_counts() -> None:
    """Summary race counts should match the underlying split."""
    summary = run_standard_experiment().summary

    assert summary.train_race_count == 4
    assert summary.validation_race_count == 2
    assert summary.test_race_count == 2


def test_experiment_summary_uses_expected_row_counts() -> None:
    """Every synthetic race contributes nine replay observations."""
    summary = run_standard_experiment().summary

    assert summary.train_row_count == 36
    assert summary.validation_row_count == 18
    assert summary.test_row_count == 18


def test_validation_snapshot_count_matches_reserved_races() -> None:
    """Two three-lap validation races should produce six snapshots."""
    result = run_standard_experiment()

    assert result.summary.validation_snapshot_count == 6

    assert result.validation_evaluation.summary.snapshot_count == 6


def test_validation_features_contain_only_validation_races() -> None:
    """Feature engineering output must not cross split boundaries."""
    result = run_standard_experiment()

    assert race_ids(result.validation_features) == {
        "2023_03_australian_grand_prix",
        "2023_04_azerbaijan_grand_prix",
    }


def test_validation_predictions_contain_only_validation_races() -> None:
    """Prediction output should contain no train or test races."""
    result = run_standard_experiment()

    assert race_ids(result.validation_predictions) == {
        "2023_03_australian_grand_prix",
        "2023_04_azerbaijan_grand_prix",
    }


def test_train_validation_and_test_races_are_disjoint() -> None:
    """No race may participate in more than one experiment partition."""
    result = run_standard_experiment()

    train_races = race_ids(result.split.train)

    validation_races = race_ids(result.split.validation)

    test_races = race_ids(result.split.test)

    assert train_races.isdisjoint(validation_races)

    assert train_races.isdisjoint(test_races)

    assert validation_races.isdisjoint(test_races)


def test_validation_metrics_are_finite() -> None:
    """All aggregate validation metrics should be numerically usable."""
    summary = run_standard_experiment().summary

    values = np.array(
        [
            summary.validation_mean_log_loss,
            summary.validation_mean_brier_score,
            summary.validation_top_one_accuracy,
            (summary.validation_mean_winner_probability),
        ],
        dtype=float,
    )

    assert bool(np.isfinite(values).all())


def test_validation_log_loss_is_non_negative() -> None:
    """Probability log loss cannot be negative."""
    summary = run_standard_experiment().summary

    assert summary.validation_mean_log_loss >= 0.0


def test_validation_brier_score_is_non_negative() -> None:
    """Brier error cannot be negative."""
    summary = run_standard_experiment().summary

    assert summary.validation_mean_brier_score >= 0.0


def test_validation_top_one_accuracy_is_probability_like() -> None:
    """Top-one accuracy should remain inside [0, 1]."""
    summary = run_standard_experiment().summary

    assert 0.0 <= summary.validation_top_one_accuracy <= 1.0


def test_mean_winner_probability_is_probability_like() -> None:
    """Average true-winner confidence should remain inside [0, 1]."""
    summary = run_standard_experiment().summary

    assert 0.0 <= (summary.validation_mean_winner_probability) <= 1.0


def test_validation_prediction_probabilities_sum_to_one() -> None:
    """Every validation snapshot must form one winner distribution."""
    result = run_standard_experiment()

    probability_sums = result.validation_predictions.groupby(
        [
            "RaceId",
            "SnapshotLap",
        ]
    )["WinnerProbability"].sum()

    np.testing.assert_allclose(
        probability_sums.to_numpy(dtype=float),
        np.ones(
            len(probability_sums),
            dtype=float,
        ),
        rtol=1e-9,
        atol=1e-9,
    )


def test_feature_engineering_is_not_called_for_test_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The final holdout must remain untouched during validation work."""
    engineered_race_sets: list[set[str]] = []

    original_engineer = experiment_module.engineer_race_features

    def recording_engineer(
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        engineered_race_sets.append(race_ids(frame))

        return original_engineer(frame)

    monkeypatch.setattr(
        experiment_module,
        "engineer_race_features",
        recording_engineer,
    )

    run_standard_experiment()

    assert len(engineered_race_sets) == 2

    assert engineered_race_sets[0] == {
        "2022_01_bahrain_grand_prix",
        "2022_02_saudi_arabian_grand_prix",
        "2023_01_bahrain_grand_prix",
        "2023_02_saudi_arabian_grand_prix",
    }

    assert engineered_race_sets[1] == {
        "2023_03_australian_grand_prix",
        "2023_04_azerbaijan_grand_prix",
    }

    all_engineered_races = set().union(*engineered_race_sets)

    assert "2024_01_bahrain_grand_prix" not in all_engineered_races

    assert "2024_02_saudi_arabian_grand_prix" not in all_engineered_races


def test_model_training_receives_only_training_races(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fitting function must receive only chronological training races."""
    training_calls: list[set[str]] = []

    original_train = experiment_module.train_baseline_winner_model

    def recording_train(
        training_features: pd.DataFrame,
        **kwargs: Any,
    ) -> BaselineWinnerModel:
        training_calls.append(race_ids(training_features))

        return original_train(
            training_features,
            **kwargs,
        )

    monkeypatch.setattr(
        experiment_module,
        "train_baseline_winner_model",
        recording_train,
    )

    run_standard_experiment()

    assert len(training_calls) == 1

    assert training_calls[0] == {
        "2022_01_bahrain_grand_prix",
        "2022_02_saudi_arabian_grand_prix",
        "2023_01_bahrain_grand_prix",
        "2023_02_saudi_arabian_grand_prix",
    }


def test_prediction_receives_only_validation_races(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validation scoring must not include the final test holdout."""
    prediction_calls: list[set[str]] = []

    original_predict = experiment_module.predict_winner_probabilities

    def recording_predict(
        model: BaselineWinnerModel,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        prediction_calls.append(race_ids(features))

        return original_predict(
            model,
            features,
        )

    monkeypatch.setattr(
        experiment_module,
        "predict_winner_probabilities",
        recording_predict,
    )

    run_standard_experiment()

    assert prediction_calls == [
        {
            "2023_03_australian_grand_prix",
            "2023_04_azerbaijan_grand_prix",
        }
    ]


def test_changing_test_data_does_not_change_validation_predictions() -> None:
    """
    The untouched test holdout must have zero influence on validation output.

    Only the 2024 lap times differ between these two corpora.
    """
    original_corpus = make_experiment_corpus(test_lap_time_offset=0.0)

    altered_test_corpus = make_experiment_corpus(test_lap_time_offset=1000.0)

    original_result = run_standard_experiment(original_corpus)

    altered_result = run_standard_experiment(altered_test_corpus)

    pd.testing.assert_frame_equal(
        original_result.validation_predictions,
        altered_result.validation_predictions,
    )

    pd.testing.assert_frame_equal(
        (original_result.validation_evaluation.snapshots),
        (altered_result.validation_evaluation.snapshots),
    )

    assert original_result.summary == (altered_result.summary)


def test_experiment_does_not_modify_source_corpus() -> None:
    """Running an experiment must leave the canonical corpus unchanged."""
    corpus = make_experiment_corpus()

    original = corpus.copy(deep=True)

    run_standard_experiment(corpus)

    pd.testing.assert_frame_equal(
        corpus,
        original,
    )


def test_result_test_split_remains_raw_corpus_data() -> None:
    """The result should retain the untouched test partition for later use."""
    result = run_standard_experiment()

    assert race_ids(result.split.test) == {
        "2024_01_bahrain_grand_prix",
        "2024_02_saudi_arabian_grand_prix",
    }

    assert "PositionFraction" not in result.split.test.columns

    assert "WinnerProbability" not in result.split.test.columns


def test_experiment_accepts_safe_feature_subset() -> None:
    """The workflow should support audited baseline feature experiments."""
    result = run_baseline_experiment(
        make_experiment_corpus(),
        validation_season=2023,
        validation_last_races=2,
        test_seasons=(2024,),
        model_columns=(
            "Position",
            "TyreLife",
            "IsLeader",
        ),
    )

    assert result.model.model_columns == (
        "Position",
        "TyreLife",
        "IsLeader",
    )


def test_experiment_is_deterministic() -> None:
    """Fixed data and configuration should reproduce validation output."""
    corpus = make_experiment_corpus()

    first = run_standard_experiment(corpus)

    second = run_standard_experiment(corpus)

    pd.testing.assert_frame_equal(
        first.validation_predictions,
        second.validation_predictions,
    )

    pd.testing.assert_frame_equal(
        first.validation_evaluation.snapshots,
        second.validation_evaluation.snapshots,
    )

    assert first.summary == second.summary


def test_experiment_rejects_non_dataframe() -> None:
    """The experiment entry point requires a pandas DataFrame."""
    invalid_corpus: Any = []

    with pytest.raises(
        TypeError,
        match=("corpus must be provided as a pandas DataFrame"),
    ):
        run_baseline_experiment(
            invalid_corpus,
            validation_season=2023,
            validation_last_races=2,
            test_seasons=(2024,),
        )


def test_experiment_wraps_split_failure() -> None:
    """Invalid chronological split configuration should be contextualized."""
    corpus = make_experiment_corpus()

    with pytest.raises(
        BaselineExperimentError,
        match=("corpus could not be split for the baseline experiment"),
    ):
        run_baseline_experiment(
            corpus,
            validation_season=2023,
            validation_last_races=99,
            test_seasons=(2024,),
        )


def test_experiment_wraps_invalid_model_columns() -> None:
    """Unsafe predictive columns should fail during model training."""
    with pytest.raises(
        BaselineExperimentError,
        match="Baseline model training failed",
    ):
        run_baseline_experiment(
            make_experiment_corpus(),
            validation_season=2023,
            validation_last_races=2,
            test_seasons=(2024,),
            model_columns=(
                "Position",
                "WonRace",
            ),
        )


def test_experiment_wraps_invalid_max_iterations() -> None:
    """Invalid optimizer configuration should fail through model training."""
    with pytest.raises(
        BaselineExperimentError,
        match="Baseline model training failed",
    ):
        run_baseline_experiment(
            make_experiment_corpus(),
            validation_season=2023,
            validation_last_races=2,
            test_seasons=(2024,),
            max_iterations=0,
        )


def test_validation_features_equal_direct_feature_engineering() -> None:
    """The workflow's validation features should match direct engineering."""
    result = run_standard_experiment()

    expected = engineer_race_features(result.split.validation)

    pd.testing.assert_frame_equal(
        result.validation_features,
        expected,
    )
