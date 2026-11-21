"""Structured logging setup: JSON lines with ISO UTC timestamps."""

import logging

import structlog


class _NamedPrintLoggerFactory:
    """PrintLogger that remembers the ``get_logger(name)`` argument for ``add_logger_name``."""

    def __call__(self, *args: object) -> structlog.PrintLogger:
        logger = structlog.PrintLogger()
        logger.name = str(args[0]) if args else "root"  # type: ignore[attr-defined]  # read by add_logger_name
        return logger


def configure_logging(json: bool = True, level: str = "INFO") -> None:
    """Configure structlog; ``json=False`` gives human-readable console output."""
    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,  # emits the key "logger"
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[level.upper()]
        ),
        logger_factory=_NamedPrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )
