from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from telemetryx.data.dataset import build_race_dataset
from telemetryx.data.targets import TARGET_COLUMN
from telemetryx.features.engineering import (
    MODEL_FEATURE_COLUMNS,
    engineer_race_features,
)
from telemetryx.features.leakage import (
    DECLARED_MODEL_FEATURES,
    IDENTIFIER_COLUMNS,
    POLICY_METADATA_COLUMNS,
    POST_RACE_COLUMNS,
    PROHIBITED_MODEL_COLUMNS,
    TARGET_COLUMNS,
    FeatureLeakageError,
    LeakageAuditReport,
    LeakageIssue,
    LeakageIssueCode,
    audit_feature_frame,
    audit_model_feature_columns,
    select_model_features,
    select_model_target,
    validate_model_feature_columns,
)


def make_cleaned_laps() -> pd.DataFrame:
    """Return a small valid cleaned race for feature tests."""
    return pd.DataFrame(
        {
            "Driver": [
                "VER",
                "VER",
                "VER",
                "NOR",
                "NOR",
                "NOR",
                "BOT",
                "BOT",
            ],
            "LapNumber": [
                1,
                2,
                3,
                1,
                2,
                3,
                1,
                2,
            ],
            "Position": [
                1,
                1,
                1,
                2,
                2,
                2,
                3,
                3,
            ],
            "Stint": [
                1,
                1,
                2,
                1,
                1,
                1,
                1,
                1,
            ],
            "Compound": [
                "SOFT",
                "SOFT",
                "MEDIUM",
                "SOFT",
                "SOFT",
                "SOFT",
                "SOFT",
                "SOFT",
            ],
            "TyreLife": [
                1.0,
                2.0,
                1.0,
                1.0,
                2.0,
                3.0,
                1.0,
                2.0,
            ],
            "TrackStatus": [
                "1",
                "1",
                "1",
                "1",
                "1",
                "1",
                "1",
                "1",
            ],
            "LapTimeSeconds": [
                90.0,
                89.0,
                88.0,
                91.0,
                90.0,
                89.0,
                92.0,
                91.0,
            ],
        }
    )


def make_results() -> pd.DataFrame:
    """Return final results containing exactly one race winner."""
    return pd.DataFrame(
        {
            "Abbreviation": [
                "VER",
                "NOR",
                "BOT",
            ],
            "Position": [
                1,
                2,
                3,
            ],
            "Status": [
                "Finished",
                "Finished",
                "+1 Lap",
            ],
            "Points": [
                25.0,
                18.0,
                15.0,
            ],
        }
    )


def make_features() -> pd.DataFrame:
    """Return a valid engineered feature frame."""
    race_dataset = build_race_dataset(
        make_cleaned_laps(),
        make_results(),
        season=2023,
        round_number=1,
        event_name="Bahrain Grand Prix",
        session_name="Race",
    )

    return engineer_race_features(race_dataset)


def issue_codes(
    report: LeakageAuditReport,
) -> set[LeakageIssueCode]:
    """Return all issue codes represented by one audit."""
    return {issue.code for issue in report.issues}


def issues_for_column(
    report: LeakageAuditReport,
    column: str,
) -> tuple[LeakageIssue, ...]:
    """Return audit issues associated with one column."""
    return tuple(issue for issue in report.issues if issue.column == column)


def test_declared_model_features_pass_column_audit() -> None:
    """The canonical TelemetryX feature allowlist should be safe."""
    report = audit_model_feature_columns(MODEL_FEATURE_COLUMNS)

    assert report.model_columns == (MODEL_FEATURE_COLUMNS)

    assert report.issues == ()
    assert report.has_leakage is False
    assert report.issue_count == 0


def test_declared_features_do_not_overlap_prohibited_columns() -> None:
    """The allowlist and explicit prohibition sets must remain disjoint."""
    assert DECLARED_MODEL_FEATURES.intersection(PROHIBITED_MODEL_COLUMNS) == set()


def test_target_column_is_detected_as_leakage() -> None:
    """WonRace must never be accepted as a predictive input."""
    report = audit_model_feature_columns(
        [
            "Position",
            TARGET_COLUMN,
        ]
    )

    assert report.has_leakage is True

    target_issues = issues_for_column(
        report,
        TARGET_COLUMN,
    )

    assert len(target_issues) == 1

    assert target_issues[0].code == (LeakageIssueCode.TARGET_FEATURE)


@pytest.mark.parametrize(
    "column",
    sorted(POST_RACE_COLUMNS),
)
def test_post_race_columns_are_detected(
    column: str,
) -> None:
    """Known post-race information must be rejected."""
    report = audit_model_feature_columns(
        [
            "Position",
            column,
        ]
    )

    matching = issues_for_column(
        report,
        column,
    )

    assert len(matching) == 1

    assert matching[0].code == (LeakageIssueCode.POST_RACE_FEATURE)


@pytest.mark.parametrize(
    "column",
    sorted(IDENTIFIER_COLUMNS | POLICY_METADATA_COLUMNS),
)
def test_identifier_and_policy_metadata_are_rejected(
    column: str,
) -> None:
    """Race identity and chronology metadata stay outside the MVP model."""
    report = audit_model_feature_columns(
        [
            column,
        ]
    )

    matching = issues_for_column(
        report,
        column,
    )

    assert len(matching) == 1

    assert matching[0].code == (LeakageIssueCode.IDENTIFIER_FEATURE)


def test_undeclared_feature_is_rejected() -> None:
    """A new feature is unsafe until deliberately added to the allowlist."""
    report = audit_model_feature_columns(
        [
            "Position",
            "MysteryFeature",
        ]
    )

    matching = issues_for_column(
        report,
        "MysteryFeature",
    )

    assert len(matching) == 1

    assert matching[0].code == (LeakageIssueCode.UNDECLARED_FEATURE)


def test_duplicate_feature_is_reported() -> None:
    """The same model column must not be supplied twice."""
    report = audit_model_feature_columns(
        [
            "Position",
            "Driver",
            "Position",
        ]
    )

    duplicate_issues = [
        issue
        for issue in report.issues
        if issue.code == LeakageIssueCode.DUPLICATE_FEATURE
    ]

    assert len(duplicate_issues) == 1
    assert duplicate_issues[0].column == "Position"


@pytest.mark.parametrize(
    "column",
    [
        "",
        " ",
        "   ",
    ],
)
def test_blank_feature_name_is_reported(
    column: str,
) -> None:
    """Whitespace-only model feature names are invalid."""
    report = audit_model_feature_columns(
        [
            column,
        ]
    )

    assert report.has_leakage is True
    assert report.issue_count == 1

    assert report.issues[0].code == (LeakageIssueCode.BLANK_FEATURE)


def test_feature_names_are_normalized_before_audit() -> None:
    """Leading and trailing whitespace should not alter safe columns."""
    report = audit_model_feature_columns(
        [
            " Position ",
            " Driver ",
        ]
    )

    assert report.model_columns == (
        "Position",
        "Driver",
    )

    assert report.has_leakage is False


def test_non_string_feature_name_is_rejected() -> None:
    """All feature names must be strings."""
    invalid_columns: Any = [
        "Position",
        123,
    ]

    with pytest.raises(
        TypeError,
        match="Every model feature name must be a string",
    ):
        audit_model_feature_columns(invalid_columns)


def test_report_raise_for_leakage_does_nothing_when_safe() -> None:
    """A safe audit report should not raise."""
    report = audit_model_feature_columns(
        [
            "Driver",
            "Position",
        ]
    )

    report.raise_for_leakage()


def test_report_raise_for_leakage_contains_issue_details() -> None:
    """Raised audit errors should explain the offending columns."""
    report = audit_model_feature_columns(
        [
            TARGET_COLUMN,
            "Points",
        ]
    )

    with pytest.raises(
        FeatureLeakageError,
        match="Unsafe model features were detected",
    ) as exception_info:
        report.raise_for_leakage()

    message = str(exception_info.value)

    assert "WonRace" in message
    assert "target_feature" in message

    assert "Points" in message
    assert "post_race_feature" in message


def test_validate_model_feature_columns_returns_normalized_names() -> None:
    """Validated model columns should preserve requested ordering."""
    columns = validate_model_feature_columns(
        [
            " Position ",
            "Driver",
            "TyreLife",
        ]
    )

    assert columns == (
        "Position",
        "Driver",
        "TyreLife",
    )


def test_validate_model_feature_columns_rejects_leakage() -> None:
    """Unsafe requested model columns should raise immediately."""
    with pytest.raises(
        FeatureLeakageError,
        match="Unsafe model features were detected",
    ):
        validate_model_feature_columns(
            [
                "Position",
                TARGET_COLUMN,
            ]
        )


def test_validate_model_feature_columns_requires_feature() -> None:
    """A model cannot be trained with an empty feature set."""
    with pytest.raises(
        FeatureLeakageError,
        match="At least one model feature is required",
    ):
        validate_model_feature_columns([])


def test_audit_feature_frame_accepts_default_model_features() -> None:
    """The canonical feature frame should pass the complete leakage audit."""
    features = make_features()

    report = audit_feature_frame(features)

    assert report.has_leakage is False
    assert report.issue_count == 0
    assert report.model_columns == MODEL_FEATURE_COLUMNS


def test_audit_feature_frame_accepts_safe_subset() -> None:
    """Model experiments may request a safe subset of declared features."""
    report = audit_feature_frame(
        make_features(),
        model_columns=(
            "Position",
            "TyreLife",
            "IsLeader",
        ),
    )

    assert report.has_leakage is False

    assert report.model_columns == (
        "Position",
        "TyreLife",
        "IsLeader",
    )


def test_audit_feature_frame_detects_target_request() -> None:
    """Requesting WonRace through the feature API must fail audit."""
    report = audit_feature_frame(
        make_features(),
        model_columns=(
            "Position",
            TARGET_COLUMN,
        ),
    )

    assert LeakageIssueCode.TARGET_FEATURE in issue_codes(report)


def test_audit_feature_frame_detects_missing_undeclared_column() -> None:
    """A nonexistent unsafe feature should report both policy problems."""
    report = audit_feature_frame(
        make_features(),
        model_columns=(
            "Position",
            "FutureWinnerProbability",
        ),
    )

    matching = issues_for_column(
        report,
        "FutureWinnerProbability",
    )

    assert {issue.code for issue in matching} == {
        LeakageIssueCode.UNDECLARED_FEATURE,
        LeakageIssueCode.MISSING_FEATURE,
    }


def test_audit_feature_frame_rejects_non_dataframe() -> None:
    """Frame auditing requires a pandas DataFrame."""
    invalid_features: Any = []

    with pytest.raises(
        TypeError,
        match="features must be provided as a pandas DataFrame",
    ):
        audit_feature_frame(invalid_features)


def test_audit_feature_frame_wraps_feature_validation_failure() -> None:
    """Malformed feature frames should fail before model-column auditing."""
    features = make_features().drop(
        columns=[
            "PositionFraction",
        ]
    )

    with pytest.raises(
        FeatureLeakageError,
        match="failed validation before leakage auditing",
    ):
        audit_feature_frame(features)


def test_select_model_features_returns_only_allowlisted_columns() -> None:
    """The selected model matrix should contain no metadata or target."""
    features = make_features()

    model_features = select_model_features(features)

    assert tuple(model_features.columns) == MODEL_FEATURE_COLUMNS

    assert TARGET_COLUMN not in model_features.columns
    assert "RaceId" not in model_features.columns
    assert "EventName" not in model_features.columns
    assert "Season" not in model_features.columns
    assert "RoundNumber" not in model_features.columns


def test_select_model_features_preserves_requested_order() -> None:
    """Safe feature subsets should preserve experiment-defined ordering."""
    model_features = select_model_features(
        make_features(),
        model_columns=(
            "IsLeader",
            "Position",
            "Driver",
        ),
    )

    assert tuple(model_features.columns) == (
        "IsLeader",
        "Position",
        "Driver",
    )


def test_select_model_features_rejects_target_leakage() -> None:
    """The X-selection boundary must reject direct target leakage."""
    with pytest.raises(
        FeatureLeakageError,
        match="Unsafe model features were detected",
    ):
        select_model_features(
            make_features(),
            model_columns=(
                "Position",
                TARGET_COLUMN,
            ),
        )


def test_select_model_features_rejects_post_race_leakage() -> None:
    """Post-race fields cannot be requested through the model boundary."""
    with pytest.raises(
        FeatureLeakageError,
        match="Points",
    ):
        select_model_features(
            make_features(),
            model_columns=(
                "Position",
                "Points",
            ),
        )


def test_selected_model_features_are_independent_copy() -> None:
    """Mutating X must not change the canonical engineered feature frame."""
    features = make_features()

    model_features = select_model_features(
        features,
        model_columns=(
            "Driver",
            "Position",
        ),
    )

    original_driver = str(
        features.loc[
            0,
            "Driver",
        ]
    )

    model_features.loc[
        0,
        "Driver",
    ] = "XXX"

    assert (
        str(
            features.loc[
                0,
                "Driver",
            ]
        )
        == original_driver
    )


def test_select_model_target_returns_boolean_target() -> None:
    """The y boundary should expose only the winner outcome."""
    features = make_features()

    target = select_model_target(features)

    assert target.name == TARGET_COLUMN

    assert pd.api.types.is_bool_dtype(target.dtype)

    assert len(target) == len(features)

    assert int(target.sum()) == 3


def test_selected_target_is_independent_copy() -> None:
    """Mutating y must not alter the engineered feature frame."""
    features = make_features()

    target = select_model_target(features)

    original_value = bool(
        features.loc[
            0,
            TARGET_COLUMN,
        ]
    )

    target.loc[0] = not original_value

    assert (
        bool(
            features.loc[
                0,
                TARGET_COLUMN,
            ]
        )
        is original_value
    )


def test_select_model_target_rejects_non_dataframe() -> None:
    """Target extraction requires a pandas DataFrame."""
    invalid_features: Any = []

    with pytest.raises(
        TypeError,
        match="features must be provided as a pandas DataFrame",
    ):
        select_model_target(invalid_features)


def test_select_model_target_wraps_invalid_feature_frame() -> None:
    """Target selection must not bypass feature-frame validation."""
    features = make_features().drop(
        columns=[
            "CompletionFraction",
        ]
    )

    with pytest.raises(
        FeatureLeakageError,
        match="failed validation before target selection",
    ):
        select_model_target(features)


def test_policy_sets_contain_expected_sensitive_columns() -> None:
    """Leakage policy constants should document our current boundary."""
    assert TARGET_COLUMN in TARGET_COLUMNS

    assert {
        "RaceId",
        "EventName",
        "SessionName",
    }.issubset(IDENTIFIER_COLUMNS)

    assert {
        "Season",
        "RoundNumber",
    }.issubset(POLICY_METADATA_COLUMNS)

    assert {
        "FinalPosition",
        "ClassifiedPosition",
        "Points",
        "Status",
        "GridPosition",
    }.issubset(POST_RACE_COLUMNS)
