from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import fastf1
import pytest

from telemetryx.config import Settings, load_settings
from telemetryx.data import (
    RaceLoadError,
    configure_fastf1_cache,
    load_race_session,
)


class FakeSession:
    """Small test double for a FastF1 Session."""

    def __init__(self) -> None:
        """Initialize an empty record of load calls."""
        self.load_calls: list[dict[str, bool]] = []

    def load(
        self,
        *,
        laps: bool,
        telemetry: bool,
        weather: bool,
        messages: bool,
    ) -> None:
        """Record the options supplied by TelemetryX."""
        self.load_calls.append(
            {
                "laps": laps,
                "telemetry": telemetry,
                "weather": weather,
                "messages": messages,
            }
        )


def settings_with_temporary_cache(
    tmp_path: Path,
) -> Settings:
    """Return project settings using a temporary FastF1 cache."""
    settings = load_settings()

    temporary_paths = replace(
        settings.paths,
        fastf1_cache_dir=tmp_path / "fastf1-cache",
    )

    return replace(
        settings,
        paths=temporary_paths,
    )


def test_configure_fastf1_cache_creates_and_enables_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache setup should create the directory and enable FastF1 caching."""
    settings = settings_with_temporary_cache(tmp_path)
    enabled_paths: list[str] = []

    def fake_enable_cache(cache_directory: str) -> None:
        enabled_paths.append(cache_directory)

    monkeypatch.setattr(
        fastf1.Cache,
        "enable_cache",
        fake_enable_cache,
    )

    result = configure_fastf1_cache(settings)

    assert result == tmp_path / "fastf1-cache"
    assert result.is_dir()
    assert enabled_paths == [str(result)]


def test_load_default_sample_race_uses_configured_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default loader should use the configured sample race and flags."""
    settings = settings_with_temporary_cache(tmp_path)
    fake_session = FakeSession()
    get_session_calls: list[tuple[int, str | int, str]] = []

    def fake_enable_cache(cache_directory: str) -> None:
        assert cache_directory == str(settings.paths.fastf1_cache_dir)

    def fake_get_session(
        season: int,
        event: str | int,
        session_type: str,
    ) -> Any:
        get_session_calls.append(
            (
                season,
                event,
                session_type,
            )
        )
        return fake_session

    monkeypatch.setattr(
        fastf1.Cache,
        "enable_cache",
        fake_enable_cache,
    )
    monkeypatch.setattr(
        fastf1,
        "get_session",
        fake_get_session,
    )

    result = load_race_session(settings)

    assert result is fake_session
    assert get_session_calls == [
        (
            2024,
            "Bahrain",
            "R",
        )
    ]

    assert fake_session.load_calls == [
        {
            "laps": True,
            "telemetry": False,
            "weather": True,
            "messages": True,
        }
    ]


def test_race_selection_can_be_overridden(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller should be able to select another configured race."""
    settings = settings_with_temporary_cache(tmp_path)
    fake_session = FakeSession()
    get_session_calls: list[tuple[int, str | int, str]] = []

    monkeypatch.setattr(
        fastf1.Cache,
        "enable_cache",
        lambda cache_directory: None,
    )

    def fake_get_session(
        season: int,
        event: str | int,
        session_type: str,
    ) -> Any:
        get_session_calls.append(
            (
                season,
                event,
                session_type,
            )
        )
        return fake_session

    monkeypatch.setattr(
        fastf1,
        "get_session",
        fake_get_session,
    )

    load_race_session(
        settings,
        season=2023,
        event="  Monza  ",
        session_type="r",
    )

    assert get_session_calls == [
        (
            2023,
            "Monza",
            "R",
        )
    ]


def test_round_number_can_be_used_as_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FastF1 event selection should support a positive round number."""
    settings = settings_with_temporary_cache(tmp_path)
    fake_session = FakeSession()
    selected_events: list[str | int] = []

    monkeypatch.setattr(
        fastf1.Cache,
        "enable_cache",
        lambda cache_directory: None,
    )

    def fake_get_session(
        season: int,
        event: str | int,
        session_type: str,
    ) -> Any:
        selected_events.append(event)
        return fake_session

    monkeypatch.setattr(
        fastf1,
        "get_session",
        fake_get_session,
    )

    load_race_session(
        settings,
        season=2024,
        event=1,
    )

    assert selected_events == [1]


@pytest.mark.parametrize(
    ("invalid_event", "expected_message"),
    [
        ("", "event name must not be empty"),
        ("   ", "event name must not be empty"),
        (0, "round number must be greater than zero"),
        (-1, "round number must be greater than zero"),
        (True, "non-empty name or positive round number"),
    ],
)
def test_invalid_event_is_rejected_before_fastf1_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_event: Any,
    expected_message: str,
) -> None:
    """Invalid event identifiers should fail before any API selection."""
    settings = settings_with_temporary_cache(tmp_path)
    get_session_was_called = False

    def fake_get_session(
        season: int,
        event: str | int,
        session_type: str,
    ) -> Any:
        nonlocal get_session_was_called
        get_session_was_called = True
        return FakeSession()

    monkeypatch.setattr(
        fastf1,
        "get_session",
        fake_get_session,
    )

    with pytest.raises(
        RaceLoadError,
        match=expected_message,
    ):
        load_race_session(
            settings,
            event=invalid_event,
        )

    assert get_session_was_called is False


def test_season_outside_configured_scope_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A race outside configured seasons should not trigger FastF1."""
    settings = settings_with_temporary_cache(tmp_path)
    get_session_was_called = False

    def fake_get_session(
        season: int,
        event: str | int,
        session_type: str,
    ) -> Any:
        nonlocal get_session_was_called
        get_session_was_called = True
        return FakeSession()

    monkeypatch.setattr(
        fastf1,
        "get_session",
        fake_get_session,
    )

    with pytest.raises(
        RaceLoadError,
        match="Season 2022 is outside the configured data scope",
    ):
        load_race_session(
            settings,
            season=2022,
        )

    assert get_session_was_called is False


def test_unsupported_session_type_is_rejected(
    tmp_path: Path,
) -> None:
    """Unknown session identifiers should produce a clear error."""
    settings = settings_with_temporary_cache(tmp_path)

    with pytest.raises(
        RaceLoadError,
        match="Unsupported session type",
    ):
        load_race_session(
            settings,
            session_type="warmup",
        )


def test_fastf1_selection_error_is_wrapped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Errors from FastF1 session selection should retain their cause."""
    settings = settings_with_temporary_cache(tmp_path)

    monkeypatch.setattr(
        fastf1.Cache,
        "enable_cache",
        lambda cache_directory: None,
    )

    def failing_get_session(
        season: int,
        event: str | int,
        session_type: str,
    ) -> Any:
        raise OSError("Simulated FastF1 failure")

    monkeypatch.setattr(
        fastf1,
        "get_session",
        failing_get_session,
    )

    with pytest.raises(
        RaceLoadError,
        match="Could not load FastF1 session",
    ) as exception_info:
        load_race_session(settings)

    assert isinstance(
        exception_info.value.__cause__,
        OSError,
    )


def test_fastf1_load_error_is_wrapped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Errors while populating a session should retain their cause."""
    settings = settings_with_temporary_cache(tmp_path)

    class FailingSession:
        """A session test double whose load operation fails."""

        def load(
            self,
            *,
            laps: bool,
            telemetry: bool,
            weather: bool,
            messages: bool,
        ) -> None:
            raise ValueError("Simulated session parsing failure")

    monkeypatch.setattr(
        fastf1.Cache,
        "enable_cache",
        lambda cache_directory: None,
    )
    monkeypatch.setattr(
        fastf1,
        "get_session",
        lambda season, event, session_type: FailingSession(),
    )

    with pytest.raises(
        RaceLoadError,
        match="Could not load FastF1 session",
    ) as exception_info:
        load_race_session(settings)

    assert isinstance(
        exception_info.value.__cause__,
        ValueError,
    )
