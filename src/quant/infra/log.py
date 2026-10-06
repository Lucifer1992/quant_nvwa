from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-5s %(name)-22s %(message)s"
_console_added = False
_file_targets: set[str] = set()


def _attach_file(root: logging.Logger, log_file: Path | None) -> None:
    if log_file is None:
        return
    key = str(Path(log_file))
    if key in _file_targets:
        return
    Path(key).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(key, encoding="utf-8")
    fh.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(fh)
    _file_targets.add(key)


def setup(level: str = "INFO", log_file: Path | None = None) -> None:
    """幂等配置根 logger：可重复调用，级别与新的日志文件都会生效。

    第二个入口（例如 CLI 之后 paper 服务改写到自己的日志文件）如果只更新级别，
    日志会静默留在旧目标里，排障时找不到文件。
    """
    global _console_added
    root = logging.getLogger()
    root.setLevel(level.upper())
    if not _console_added:
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(console)
        _console_added = True
    _attach_file(root, log_file)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
