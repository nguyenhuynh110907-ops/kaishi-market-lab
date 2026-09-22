from pathlib import Path
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from kaishi_bot.cli import app
from kaishi_bot.instance_lock import DashboardInstanceLock

runner = CliRunner()


def test_status_reads_empty_state(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["status", "--state", str(tmp_path / "state.sqlite3")],
    )

    assert result.exit_code == 0
    assert "entries=0 fills=0 take_profits=0" in result.stdout


def test_validate_prints_demo_and_strategy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("series: KXBTC15M\n", encoding="utf-8")
    key = tmp_path / "demo.key"
    key.write_text("private", encoding="utf-8")
    monkeypatch.setenv("KALSHI_KEY_ID", "demo-id")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", str(key))

    result = runner.invoke(app, ["validate", "--config", str(config)])

    assert result.exit_code == 0
    assert "DEMO" in result.stdout
    assert "entry=0.25" in result.stdout
    assert "tp=0.40" in result.stdout


def test_validate_rejects_missing_credentials(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("series: KXBTC15M\n", encoding="utf-8")
    monkeypatch.delenv("KALSHI_KEY_ID", raising=False)
    monkeypatch.delenv("KALSHI_PRIVATE_KEY_PATH", raising=False)

    result = runner.invoke(app, ["validate", "--config", str(config)])

    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "KALSHI_KEY_ID is required" in str(result.exception)


def test_installed_bot_entrypoint_can_import_package() -> None:
    bot = Path(sys.executable).with_name("bot")

    result = subprocess.run(
        [str(bot), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Kalshi Demo BTC 15-minute bot" in result.stdout


def test_dashboard_defaults_to_loopback_port_5714(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    def fake_run(app, *, host, port, log_level):
        captured.update(host=host, port=port, log_level=log_level)

    monkeypatch.setattr("uvicorn.run", fake_run)
    result = runner.invoke(app, ["dashboard", "--state", str(tmp_path / "dashboard.sqlite3")])

    assert result.exit_code == 0
    assert captured == {"host": "127.0.0.1", "port": 5714, "log_level": "info"}


def test_dashboard_rejects_non_loopback_host(tmp_path: Path) -> None:
    result = runner.invoke(app, [
        "dashboard", "--state", str(tmp_path / "dashboard.sqlite3"), "--host", "0.0.0.0"
    ])

    assert result.exit_code != 0
    assert "127.0.0.1" in result.stdout


def test_dashboard_instance_lock_prevents_two_writers(tmp_path: Path) -> None:
    state = tmp_path / "dashboard.sqlite3"
    first = DashboardInstanceLock(state)
    second = DashboardInstanceLock(state)

    with first:
        with pytest.raises(RuntimeError, match="already running"):
            second.acquire()

    with second:
        assert second.acquired


def test_dashboard_does_not_misreport_runtime_failure_as_lock_conflict(
    tmp_path: Path, monkeypatch,
) -> None:
    def fail_run(*_args, **_kwargs):
        raise RuntimeError("server failed")

    monkeypatch.setattr("uvicorn.run", fail_run)
    result = runner.invoke(
        app, ["dashboard", "--state", str(tmp_path / "dashboard.sqlite3")]
    )

    assert isinstance(result.exception, RuntimeError)
    assert "already running" not in result.stdout
