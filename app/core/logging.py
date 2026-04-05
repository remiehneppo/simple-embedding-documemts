import logging
import logging.handlers
import sys
from pathlib import Path

import structlog
from structlog.processors import CallsiteParameter, CallsiteParameterAdder


def setup_logging(log_level: str = "INFO", log_path: Path | None = None) -> None:
    """Configure stdlib logging + structlog with JSON output (file) and console (tty)."""
    level_int = getattr(logging, log_level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_path:
        log_path.mkdir(parents=True, exist_ok=True)
        # Rotating file handler: 10 MB per file, keep 5 backups
        file_handler = logging.handlers.RotatingFileHandler(
            log_path / "app.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)  # Always capture DEBUG in file
        handlers.append(file_handler)

    logging.basicConfig(
        level=logging.DEBUG,  # Root logger captures everything; handlers filter
        handlers=handlers,
        force=True,
        format="%(message)s",
    )
    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Use human-friendly console renderer in a TTY, JSON otherwise
    if sys.stdout.isatty():
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            # Capture source location (file, function, line) for every log call
            CallsiteParameterAdder(
                [
                    CallsiteParameter.FILENAME,
                    CallsiteParameter.FUNC_NAME,
                    CallsiteParameter.LINENO,
                ]
            ),
            # Render exc_info as a formatted traceback string before serialising
            structlog.processors.ExceptionRenderer(),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,  # Allow level changes at runtime
    )


def get_logger(name: str = __name__) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger().bind(logger=name)
