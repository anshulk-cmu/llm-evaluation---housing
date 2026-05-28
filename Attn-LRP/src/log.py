# -*- coding: utf-8 -*-
"""
Centralised logging for the Attn-LRP pipeline.

- One configured logger tree under the name "attnlrp".
- Logs to BOTH the console and a timestamped file in results/logs/, so long
  GPU runs (Phase 1 ~1000 backward passes, Phase 2 across 120 permutations)
  leave a recoverable record.
- `get_logger(__name__)` in any module returns a child logger; handlers are
  installed once on the root "attnlrp" logger (idempotent).
- `banner(...)` logs a visually separated section header.

Environment:
  ATTNLRP_LOG_LEVEL   (default INFO)  e.g. DEBUG, WARNING
  ATTNLRP_LOG_FILE    (default auto)  explicit log file path override
"""

from __future__ import annotations
import logging
import os
import sys
from datetime import datetime

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.dirname(_SRC_DIR)
LOG_DIR = os.path.join(_PKG_DIR, "results", "logs")

_ROOT_NAME = "attnlrp"
_CONFIGURED = False
_LOG_PATH: str | None = None

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _configure_root() -> None:
    global _CONFIGURED, _LOG_PATH
    if _CONFIGURED:
        return
    os.makedirs(LOG_DIR, exist_ok=True)
    level = getattr(logging, os.environ.get("ATTNLRP_LOG_LEVEL", "INFO").upper(), logging.INFO)

    root = logging.getLogger(_ROOT_NAME)
    root.setLevel(level)
    root.propagate = False
    fmt = logging.Formatter(_FMT, datefmt=_DATEFMT)

    # Make the console robust to non-ASCII on Windows cp1252 terminals so a
    # stray unicode char can never crash a long run via UnicodeEncodeError.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    _LOG_PATH = os.environ.get(
        "ATTNLRP_LOG_FILE",
        os.path.join(LOG_DIR, f"run_{datetime.now():%Y%m%d_%H%M%S}.log"),
    )
    fh = logging.FileHandler(_LOG_PATH, encoding="utf-8")
    fh.setLevel(logging.DEBUG)            # file keeps everything
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Route Python warnings through logging too.
    logging.captureWarnings(True)
    _CONFIGURED = True
    root.info(f"logging initialised -> {_LOG_PATH}")


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger under the 'attnlrp' tree (configures handlers on first use)."""
    _configure_root()
    if not name or name in ("__main__", _ROOT_NAME):
        return logging.getLogger(_ROOT_NAME)
    short = name.split(".")[-1]
    return logging.getLogger(f"{_ROOT_NAME}.{short}")


def banner(msg: str, logger: logging.Logger | None = None) -> None:
    log = logger or get_logger()
    line = "=" * 78
    log.info(line)
    log.info(msg)
    log.info(line)


def log_path() -> str | None:
    """Path of the active log file (after first get_logger call)."""
    return _LOG_PATH
