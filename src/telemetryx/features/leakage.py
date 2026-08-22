from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import pandas as pd

from telemetryx.data.targets import TARGET_COLUMN
from telemetryx.features.engineering import (
    DISALLOWED_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    validate_feature_frame,
)


class LeakageIssueCode(StrEnum):
    """Supported feature-leakage audit issue categories."""

    DUPLICATE_FEATURE = "duplicate_feature"
    BLANK_FEATURE = "blank_feature"
    TARGET_FEATURE = "target_feature"
    POST_RACE_FEATURE = "post_race_feature"
    IDENTIFIER_FEATURE = "identifier_feature"
    UNDECLARED_FEATURE = "undeclared_feature"
    MISSING_FEATURE = "missing_feature"


class FeatureLeakageError(ValueError):
    """Raised when unsafe model-input columns are detected."""


@dataclass(frozen=True, slots=True)
class LeakageIssue:
    """One structural problem found during a model-feature audit."""

    code: LeakageIssueCode
    column: str
    message: str


@dataclass(frozen=True, slots=True)
class LeakageAuditReport:
    """Result of auditing a collection of intended model features."""

    model_columns: tuple[str, ...]
    issues: tuple[LeakageIssue, ...]

    @property
    def has_leakage(self) -> bool:
        """Return whether any unsafe model-input configuration was found."""
        return bool(self.issues)

    @property
    def issue_count(self) -> int:
        """Return the total number of audit issues."""
        return len(self.issues)

    def raise_for_leakage(self) -> None:
        """Raise when the audit contains one or more issues."""
        if not self.issues:
            return

        descriptions = [
            (f"{issue.code.value}: {issue.column}: {issue.message}")
            for issue in self.issues
        ]

        raise FeatureLeakageError(
            f"Unsafe model features were detected: {'; '.join(descriptions)}."
        )


IDENTIFIER_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "RaceId",
        "EventName",
        "SessionName",
    }
)

POLICY_METADATA_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "Season",
        "RoundNumber",
    }
)

TARGET_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        TARGET_COLUMN,
    }
)

POST_RACE_COLUMNS: Final[frozenset[str]] = frozenset(DISALLOWED_FEATURE_COLUMNS)

PROHIBITED_MODEL_COLUMNS: Final[frozenset[str]] = (
    IDENTIFIER_COLUMNS | POLICY_METADATA_COLUMNS | TARGET_COLUMNS | POST_RACE_COLUMNS
)

DECLARED_MODEL_FEATURES: Final[frozenset[str]] = frozenset(MODEL_FEATURE_COLUMNS)


def audit_model_feature_columns(
    columns: Sequence[str],
) -> LeakageAuditReport:
    """
    Audit intended model-input column names.

    TelemetryX uses an explicit allowlist defined by
    ``MODEL_FEATURE_COLUMNS``. Target, post-race, identifier and undeclared
    columns are rejected.

    Parameters
    ----------
    columns:
        Columns intended to be supplied to a predictive model.

    Returns
    -------
    LeakageAuditReport
        Structured audit result. The report does not raise automatically.
    """
    normalized_columns: list[str] = []

    issues: list[LeakageIssue] = []

    seen_columns: set[str] = set()

    for raw_column in columns:
        if not isinstance(
            raw_column,
            str,
        ):
            raise TypeError("Every model feature name must be a string.")

        column = raw_column.strip()

        if not column:
            issues.append(
                LeakageIssue(
                    code=LeakageIssueCode.BLANK_FEATURE,
                    column=raw_column,
                    message=("Model feature names cannot be blank."),
                )
            )

            continue

        normalized_columns.append(column)

        if column in seen_columns:
            issues.append(
                LeakageIssue(
                    code=LeakageIssueCode.DUPLICATE_FEATURE,
                    column=column,
                    message=("A model feature cannot be supplied more than once."),
                )
            )

            continue

        seen_columns.add(column)

        _audit_single_model_column(
            column=column,
            issues=issues,
        )

    return LeakageAuditReport(
        model_columns=tuple(normalized_columns),
        issues=tuple(issues),
    )


def validate_model_feature_columns(
    columns: Sequence[str],
) -> tuple[str, ...]:
    """
    Validate model columns and return their normalized names.

    Parameters
    ----------
    columns:
        Intended predictive model inputs.

    Returns
    -------
    tuple[str, ...]
        Validated feature names in their original order.

    Raises
    ------
    FeatureLeakageError
        If any requested model column violates the feature contract.
    """
    report = audit_model_feature_columns(columns)

    report.raise_for_leakage()

    if not report.model_columns:
        raise FeatureLeakageError("At least one model feature is required.")

    return report.model_columns


def audit_feature_frame(
    features: pd.DataFrame,
    *,
    model_columns: Sequence[str] = MODEL_FEATURE_COLUMNS,
) -> LeakageAuditReport:
    """
    Audit model columns against an engineered feature DataFrame.

    The complete feature frame is first validated using the feature
    engineering contract. Requested model features are then audited against
    the explicit TelemetryX allowlist and checked for availability.

    Parameters
    ----------
    features:
        Valid engineered TelemetryX feature frame.
    model_columns:
        Columns intended to be supplied to the predictive model.

    Returns
    -------
    LeakageAuditReport
        Structured audit including unsafe or missing model columns.

    Raises
    ------
    TypeError
        If ``features`` is not a pandas DataFrame.
    FeatureLeakageError
        If the feature frame itself is invalid.
    """
    if not isinstance(
        features,
        pd.DataFrame,
    ):
        raise TypeError("features must be provided as a pandas DataFrame.")

    try:
        validate_feature_frame(features)
    except (TypeError, ValueError) as exc:
        raise FeatureLeakageError(
            "The feature frame failed validation before leakage auditing."
        ) from exc

    report = audit_model_feature_columns(model_columns)

    issues = list(report.issues)

    available_columns = {str(column) for column in features.columns}

    for column in report.model_columns:
        if column not in available_columns:
            issues.append(
                LeakageIssue(
                    code=LeakageIssueCode.MISSING_FEATURE,
                    column=column,
                    message=(
                        "The requested model feature is not present "
                        "in the feature frame."
                    ),
                )
            )

    return LeakageAuditReport(
        model_columns=report.model_columns,
        issues=tuple(issues),
    )


def select_model_features(
    features: pd.DataFrame,
    *,
    model_columns: Sequence[str] = MODEL_FEATURE_COLUMNS,
) -> pd.DataFrame:
    """
    Return a validated model-input matrix containing only safe features.

    This function provides the final structural boundary before a model sees
    the data. Metadata and the supervised target remain outside the returned
    DataFrame.

    Parameters
    ----------
    features:
        Engineered TelemetryX feature frame.
    model_columns:
        Requested predictive input columns.

    Returns
    -------
    pd.DataFrame
        Independent DataFrame containing only audited model features.

    Raises
    ------
    FeatureLeakageError
        If unsafe or missing model features are detected.
    """
    report = audit_feature_frame(
        features,
        model_columns=model_columns,
    )

    report.raise_for_leakage()

    if not report.model_columns:
        raise FeatureLeakageError("At least one model feature is required.")

    return features.loc[
        :,
        list(report.model_columns),
    ].copy(deep=True)


def select_model_target(
    features: pd.DataFrame,
) -> pd.Series:
    """
    Return an independent copy of the supervised winner target.

    Separating this function from ``select_model_features`` makes the X/y
    boundary explicit.

    Parameters
    ----------
    features:
        Valid engineered TelemetryX feature frame.

    Returns
    -------
    pd.Series
        Boolean ``WonRace`` target.

    Raises
    ------
    TypeError
        If ``features`` is not a pandas DataFrame.
    FeatureLeakageError
        If the feature frame is invalid.
    """
    if not isinstance(
        features,
        pd.DataFrame,
    ):
        raise TypeError("features must be provided as a pandas DataFrame.")

    try:
        validate_feature_frame(features)
    except (TypeError, ValueError) as exc:
        raise FeatureLeakageError(
            "The feature frame failed validation before target selection."
        ) from exc

    target = features[TARGET_COLUMN].copy(deep=True)

    if not pd.api.types.is_bool_dtype(target.dtype):
        raise FeatureLeakageError(f"{TARGET_COLUMN} must use a Boolean dtype.")

    if bool(target.isna().any()):
        raise FeatureLeakageError(f"{TARGET_COLUMN} cannot contain missing values.")

    return target


def _audit_single_model_column(
    *,
    column: str,
    issues: list[LeakageIssue],
) -> None:
    """Audit one requested model-input column."""
    if column in TARGET_COLUMNS:
        issues.append(
            LeakageIssue(
                code=LeakageIssueCode.TARGET_FEATURE,
                column=column,
                message=("The supervised outcome cannot be used as an input."),
            )
        )

        return

    if column in POST_RACE_COLUMNS:
        issues.append(
            LeakageIssue(
                code=LeakageIssueCode.POST_RACE_FEATURE,
                column=column,
                message=("Post-race information is unavailable at prediction time."),
            )
        )

        return

    if column in IDENTIFIER_COLUMNS or column in POLICY_METADATA_COLUMNS:
        issues.append(
            LeakageIssue(
                code=LeakageIssueCode.IDENTIFIER_FEATURE,
                column=column,
                message=(
                    "Race identity or chronology metadata is excluded "
                    "from the MVP model feature set."
                ),
            )
        )

        return

    if column not in DECLARED_MODEL_FEATURES:
        issues.append(
            LeakageIssue(
                code=LeakageIssueCode.UNDECLARED_FEATURE,
                column=column,
                message=(
                    "The feature is not present in the explicit "
                    "TelemetryX model-feature allowlist."
                ),
            )
        )
