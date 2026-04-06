"""Outputs package."""

from localvisionai.config import OutputConfig
from .base import AbstractOutputHandler
from .console_output import ConsoleOutput
from .json_output import JSONOutput


def build_handlers(config: OutputConfig, job_id: str) -> list[AbstractOutputHandler]:
    """Factory: create all output handlers specified in config."""
    handlers: list[AbstractOutputHandler] = []

    for fmt in config.formats:
        if fmt == "console":
            handlers.append(ConsoleOutput(verbose=False))
        elif fmt == "json":
            handlers.append(JSONOutput(output_dir=config.output_dir))
        elif fmt == "srt":
            from .srt_output import SRTOutput
            handlers.append(SRTOutput(output_dir=config.output_dir))
        elif fmt == "csv":
            from .csv_output import CSVOutput
            handlers.append(CSVOutput(output_dir=config.output_dir))
        elif fmt == "sqlite":
            from .sqlite_output import SQLiteOutput
            handlers.append(SQLiteOutput(output_dir=config.output_dir))

    return handlers


__all__ = [
    "AbstractOutputHandler",
    "ConsoleOutput",
    "JSONOutput",
    "build_handlers",
]
