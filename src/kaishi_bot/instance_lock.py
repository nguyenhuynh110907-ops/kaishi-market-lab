from __future__ import annotations

import fcntl
import os
from pathlib import Path
from typing import IO


class DashboardInstanceLock:
    """Hold an advisory lock for one dashboard database."""

    def __init__(self, state_path: Path) -> None:
        self.lock_path = state_path.with_suffix(f"{state_path.suffix}.lock")
        self._file: IO[str] | None = None

    @property
    def acquired(self) -> bool:
        return self._file is not None

    def acquire(self) -> None:
        if self.acquired:
            return
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = self.lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock_file.close()
            raise RuntimeError(
                f"Kaishi dashboard is already running for {self.lock_path.with_suffix('')}"
            ) from exc
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"{os.getpid()}\n")
        lock_file.flush()
        self._file = lock_file

    def release(self) -> None:
        if self._file is None:
            return
        fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        self._file.close()
        self._file = None

    def __enter__(self) -> DashboardInstanceLock:
        self.acquire()
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()
