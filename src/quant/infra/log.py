from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-5s %(name)-22s %(message)s"
_configured = False


def setup(level: str = "INFO", log_file: Path | None = None) -> None:
    global _configured
    root = logging.getLogger()
    if _configured:
        root.setLevel(level.upper())
        return
    root.setLevel(level.upper())
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(console)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(fh)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
