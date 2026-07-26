from __future__ import annotations

from pathlib import Path

import fastf1
from fastf1.core import Session

from telemetryx.config import Settings, load_settings


EventIdentifier = str | int

_SUPPORTED_SESSION_TYPES = frozenset(
    {
        "R",
        "Q",
        "S",
        "FP1",
        "FP2",
        "FP3",
    }
)


class RaceLoadError(RuntimeError):
    """Raised when a requested FastF1 session cannot be loaded."""


def configure_fastf1_cache(settings: Settings) -> Path:
    """
    Create and enable the configured FastF1 cache directory.

    Parameters
    ----------
    settings:
        Validated TelemetryX project settings.

    Returns
    -------
    Path
        Absolute path to the enabled FastF1 cache directory.

    Raises
    ------
    RaceLoadError
        If caching is disabled, the directory cannot be created, or FastF1
        rejects the cache configuration.
    """
    if not settings.data.cache_enabled:
        raise RaceLoadError(
            "FastF1 caching must remain enabled for the TelemetryX MVP."
        )

    cache_directory = settings.paths.fastf1_cache_dir

    try:
        cache_directory.mkdir(
            parents=True,
            exist_ok=True,
        )
    except OSError as exc:
        raise RaceLoadError(
            f"Could not create FastF1 cache directory: {cache_directory}"
        ) from exc

    try:
        fastf1.Cache.enable_cache(str(cache_directory))
    except Exception as exc:
        raise RaceLoadError(
            f"Could not enable FastF1 cache: {cache_directory}"
        ) from exc

    return cache_directory


def load_race_session(
    settings: Settings | None = None,
    *,
    season: int | None = None,
    event: EventIdentifier | None = None,
    session_type: str | None = None,
) -> Session:
    """
    Load a historical Formula 1 session using TelemetryX settings.

    When no race overrides are supplied, the configured ``sample_race`` is
    loaded. Explicit overrides are useful for inspection scripts and future
    command-line tools.

    Parameters
    ----------
    settings:
        Validated TelemetryX settings. The default project settings are loaded
        automatically when this argument is omitted.
    season:
        Optional championship season override.
    event:
        Optional event override. FastF1 accepts either an event name or a
        one-based championship round number.
    session_type:
        Optional session identifier override, such as ``"R"`` or ``"Q"``.

    Returns
    -------
    Session
        A fully loaded FastF1 session.

    Raises
    ------
    RaceLoadError
        If the selection is invalid, cache setup fails, or FastF1 cannot load
        the session.
    """
    active_settings = settings if settings is not None else load_settings()

    selected_season, selected_event, selected_session_type = (
        _resolve_session_selection(
            settings=active_settings,
            season=season,
            event=event,
            session_type=session_type,
        )
    )

    configure_fastf1_cache(active_settings)

    try:
        session = fastf1.get_session(
            selected_season,
            selected_event,
            selected_session_type,
        )

        session.load(
            laps=active_settings.data.load_laps,
            telemetry=active_settings.data.load_telemetry,
            weather=active_settings.data.load_weather,
            messages=active_settings.data.load_race_control_messages,
        )
    except Exception as exc:
        raise RaceLoadError(
            "Could not load FastF1 session "
            f"(season={selected_season}, "
            f"event={selected_event!r}, "
            f"session_type={selected_session_type!r})."
        ) from exc

    return session


def _resolve_session_selection(
    settings: Settings,
    season: int | None,
    event: EventIdentifier | None,
    session_type: str | None,
) -> tuple[int, EventIdentifier, str]:
    """Resolve and validate the requested race-session identifiers."""
    selected_season = (
        settings.sample_race.season
        if season is None
        else season
    )

    selected_event: EventIdentifier = (
        settings.sample_race.event
        if event is None
        else event
    )

    selected_session_type = (
        settings.sample_race.session_type
        if session_type is None
        else session_type
    )

    _validate_season(
        season=selected_season,
        configured_seasons=settings.data.seasons,
    )

    normalised_event = _normalise_event(selected_event)
    normalised_session_type = _normalise_session_type(
        selected_session_type
    )

    return (
        selected_season,
        normalised_event,
        normalised_session_type,
    )


def _validate_season(
    season: int,
    configured_seasons: tuple[int, ...],
) -> None:
    """Validate that a season is an integer in the configured data scope."""
    if isinstance(season, bool) or not isinstance(season, int):
        raise RaceLoadError("The requested season must be an integer.")

    if season not in configured_seasons:
        allowed = ", ".join(
            str(value) for value in configured_seasons
        )
        raise RaceLoadError(
            f"Season {season} is outside the configured data scope. "
            f"Allowed seasons: {allowed}."
        )


def _normalise_event(event: EventIdentifier) -> EventIdentifier:
    """Validate and normalize a FastF1 event name or round number."""
    if isinstance(event, bool):
        raise RaceLoadError(
            "The event must be a non-empty name or positive round number."
        )

    if isinstance(event, int):
        if event < 1:
            raise RaceLoadError(
                "An event round number must be greater than zero."
            )

        return event

    if isinstance(event, str):
        normalised = event.strip()

        if not normalised:
            raise RaceLoadError(
                "An event name must not be empty."
            )

        return normalised

    raise RaceLoadError(
        "The event must be a non-empty name or positive round number."
    )


def _normalise_session_type(session_type: str) -> str:
    """Validate and normalize a FastF1 session abbreviation."""
    if not isinstance(session_type, str):
        raise RaceLoadError(
            "The session type must be a string."
        )

    normalised = session_type.strip().upper()

    if normalised not in _SUPPORTED_SESSION_TYPES:
        supported = ", ".join(sorted(_SUPPORTED_SESSION_TYPES))
        raise RaceLoadError(
            f"Unsupported session type {session_type!r}. "
            f"Supported values: {supported}."
        )

    return normalised