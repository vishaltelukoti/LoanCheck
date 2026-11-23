"""Shared logging configuration for LoanCheck command-line entry points."""

import logging


def configure_logging() -> None:
    """Configure concise, consistent console logging for project scripts.

    This operation is idempotent: calling it from multiple entry points does not
    add duplicate handlers.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
