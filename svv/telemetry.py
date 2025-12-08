"""
Lightweight telemetry wrapper for svVascularize.

This module provides optional integration with Sentry for crash and error
reporting, primarily for the GUI. It is designed to be safe when Sentry is
not installed or when no DSN is configured.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

_ENABLED: bool = False
_EXCEPTHOOK_INSTALLED: bool = False

# Default DSN for the svVascularize GUI. This is used only when no explicit
# DSN is provided by the caller or via environment variables. It is safe for
# Sentry DSNs to be embedded in client applications, but environment variables
# still take precedence when present.
_DEFAULT_GUI_DSN: str = (
    "https://c76b4f3a871ce81fb7d5b55df436e6fc@"
    "o4510450670174208.ingest.us.sentry.io/4510450824577024"
)


def _before_send(event, hint):
    """
    Scrub potentially sensitive fields before sending to Sentry.

    This removes obvious user/request payloads. Additional scrubbing can be
    added here if needed (e.g., trimming filesystem paths).
    """
    try:
        event.pop("user", None)
        event.pop("request", None)
    except Exception:
        pass
    return event


def init_telemetry(dsn: Optional[str], release: str, environment: str = "gui") -> None:
    """
    Initialize Sentry-based telemetry if a DSN is provided.

    Parameters
    ----------
    dsn : str | None
        Sentry DSN. When ``None`` or empty, telemetry is disabled.
    release : str
        Application/library release string (e.g., svv.__version__).
    environment : str
        Environment label (e.g., "gui", "dev", "staging").
    """
    global _ENABLED, _EXCEPTHOOK_INSTALLED

    if _ENABLED:
        return

    # Allow environment variable override to hard-disable telemetry
    if os.environ.get("SVV_TELEMETRY_DISABLED", "").strip() == "1":
        return

    # Use provided DSN when available; otherwise fall back to the built-in
    # default DSN for the svVascularize GUI.
    if not dsn:
        dsn = _DEFAULT_GUI_DSN
    if not dsn:
        return

    try:
        import sentry_sdk  # type: ignore[import]
        from sentry_sdk.integrations.logging import LoggingIntegration  # type: ignore[import]
        try:
            # Qt integration is optional – if unavailable, we still proceed
            from sentry_sdk.integrations.qt import QtIntegration  # type: ignore[import]
        except Exception:
            QtIntegration = None  # type: ignore[assignment]
        try:
            # Faulthandler integration can capture hard crashes (e.g. segfaults)
            from sentry_sdk.integrations.faulthandler import (  # type: ignore[import]
                FaulthandlerIntegration,
            )
        except Exception:
            FaulthandlerIntegration = None  # type: ignore[assignment]
    except Exception:
        # Sentry SDK not installed
        return

    logging_integration = LoggingIntegration(
        level=None,        # Do not auto-capture all logs
        event_level=None,  # Only explicit captures / unhandled exceptions
    )

    integrations = [logging_integration]
    try:
        if QtIntegration is not None:
            integrations.append(QtIntegration())
    except Exception:
        # If QtIntegration fails for any reason, continue with logging only
        pass
    try:
        if FaulthandlerIntegration is not None:
            integrations.append(FaulthandlerIntegration())
    except Exception:
        # Hard-crash capture is best-effort; never break startup
        pass

    try:
        sentry_sdk.init(
            dsn=dsn,
            release=release,
            environment=environment,
            integrations=integrations,
            before_send=_before_send,
            attach_stacktrace=True,
            send_default_pii=False,
        )

        # Tag events as originating from the svVascularize GUI
        with sentry_sdk.configure_scope() as scope:
            scope.set_tag("application", "svVascularize")
            scope.set_tag("component", "gui")

        _ENABLED = True

        # Install a simple global excepthook as a last-resort safety net.
        if not _EXCEPTHOOK_INSTALLED:
            _install_global_excepthook()
            _EXCEPTHOOK_INSTALLED = True
    except Exception:
        # Any failure here should not break the application
        _ENABLED = False


def _install_global_excepthook() -> None:
    """Install a global sys.excepthook that reports to Sentry when enabled."""
    try:
        import sentry_sdk  # type: ignore[import]
    except Exception:
        return

    original_hook = sys.excepthook

    def _hook(exc_type, exc_value, exc_traceback):
        # Avoid swallowing KeyboardInterrupt
        if exc_type is KeyboardInterrupt:
            if original_hook:
                original_hook(exc_type, exc_value, exc_traceback)
            return
        try:
            if _ENABLED:
                sentry_sdk.capture_exception(exc_value)
        except Exception:
            pass
        if original_hook:
            original_hook(exc_type, exc_value, exc_traceback)

    sys.excepthook = _hook


def capture_exception(exc: BaseException) -> None:
    """
    Manually capture an exception and send to telemetry backend, if enabled.
    """
    if not _ENABLED:
        return
    try:
        import sentry_sdk  # type: ignore[import]

        sentry_sdk.capture_exception(exc)
        # Flush to ensure the event is sent immediately
        sentry_sdk.flush(timeout=2.0)
    except Exception:
        pass


def capture_message(message: str, level: str = "info", **tags) -> None:
    """
    Manually capture a message (breadcrumb/event) if telemetry is enabled.

    Parameters
    ----------
    message : str
        Message text to record.
    level : str
        Sentry level string, e.g., "info", "warning", "error".
    **tags :
        Optional key/value tags to attach to the event.
    """
    if not _ENABLED:
        return
    try:
        import sentry_sdk  # type: ignore[import]

        with sentry_sdk.push_scope() as scope:
            for key, value in tags.items():
                scope.set_tag(key, value)
            sentry_sdk.capture_message(message, level=level)
        # Flush to ensure the event is sent immediately
        sentry_sdk.flush(timeout=2.0)
    except Exception:
        pass


def telemetry_enabled() -> bool:
    """
    Return True if telemetry/Sentry has been initialized for this process.

    This is primarily intended for GUI debug helpers so they can inform the
    user when a telemetry test event could not be sent because telemetry is
    disabled or the Sentry SDK is unavailable.
    """
    return _ENABLED
