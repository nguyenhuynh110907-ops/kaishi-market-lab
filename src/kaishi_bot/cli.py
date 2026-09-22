from __future__ import annotations

import asyncio
import os
from pathlib import Path

import typer
import uvicorn

from kaishi_bot.config import BotConfig, Credentials
from kaishi_bot.exchange import KalshiDemoAdapter
from kaishi_bot.instance_lock import DashboardInstanceLock
from kaishi_bot.logging_setup import configure_logging
from kaishi_bot.service import BotService
from kaishi_bot.store import StateStore

app = typer.Typer(no_args_is_help=True, help="Kalshi Demo BTC 15-minute bot")


def load_settings(config_path: Path) -> tuple[BotConfig, Credentials]:
    return BotConfig.load(config_path), Credentials.load(os.environ)


def build_adapter(credentials: Credentials) -> KalshiDemoAdapter:
    return KalshiDemoAdapter.create(credentials)


@app.command()
def validate(
    config: Path = typer.Option(Path("config.yaml"), "--config"),
) -> None:
    settings, credentials = load_settings(config)
    typer.echo("KALSHI DEMO — configuration is valid")
    typer.echo(
        f"series={settings.series} entry={settings.entry_price} "
        f"tp={settings.take_profit_price} count={settings.contracts}"
    )
    typer.echo(f"UP={settings.trade_up} DOWN={settings.trade_down}")
    typer.echo(f"key_file={credentials.private_key_path}")


@app.command()
def check(
    config: Path = typer.Option(Path("config.yaml"), "--config"),
    state: Path = typer.Option(Path("state.sqlite3"), "--state"),
) -> None:
    """Read Demo market data without placing an order."""

    settings, credentials = load_settings(config)

    async def command() -> None:
        exchange = build_adapter(credentials)
        try:
            with StateStore(state) as store:
                ticker, quotes = await BotService(exchange, store, settings).check()
            typer.echo(
                f"DEMO {ticker}: UP ask={quotes.up_ask}, "
                f"DOWN ask={quotes.down_ask}"
            )
        finally:
            await exchange.close()

    asyncio.run(command())


@app.command()
def run(
    config: Path = typer.Option(Path("config.yaml"), "--config"),
    state: Path = typer.Option(Path("state.sqlite3"), "--state"),
    log: Path = typer.Option(Path("logs/bot.jsonl"), "--log"),
) -> None:
    """Start automated trading with Demo funds only."""

    settings, credentials = load_settings(config)
    configure_logging(log)
    typer.echo("=== KALSHI DEMO — NO PRODUCTION ORDERS ===")
    typer.echo(
        f"entry={settings.entry_price} tp={settings.take_profit_price} "
        f"count={settings.contracts}"
    )

    async def command() -> None:
        exchange = build_adapter(credentials)
        try:
            with StateStore(state) as store:
                await BotService(exchange, store, settings).run()
        finally:
            await exchange.close()

    try:
        asyncio.run(command())
    except KeyboardInterrupt:
        typer.echo("Stopped. Resting Demo take-profit orders were left unchanged.")


@app.command()
def status(
    state: Path = typer.Option(Path("state.sqlite3"), "--state"),
) -> None:
    with StateStore(state) as store:
        summary = store.summary()
    typer.echo(
        f"entries={summary['entries']} fills={summary['fills']} "
        f"take_profits={summary['take_profits']}"
    )


@app.command()
def dashboard(
    state: Path = typer.Option(Path("dashboard.sqlite3"), "--state"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(5714, "--port", min=1, max=65535),
    research_config: Path = typer.Option(
        Path("research_capture.yaml"), "--research-config"
    ),
) -> None:
    """Run the local Paper, Strategy Lab, and API control dashboard."""

    if host != "127.0.0.1":
        typer.echo("Dashboard is local-only; --host must be 127.0.0.1")
        raise typer.Exit(code=2)
    from kaishi_bot.web import create_app

    instance_lock = DashboardInstanceLock(state)
    try:
        instance_lock.acquire()
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    try:
        typer.echo(f"Kaishi dashboard: http://{host}:{port}")
        uvicorn.run(
            create_app(state, research_config_path=research_config),
            host=host, port=port, log_level="info",
        )
    finally:
        instance_lock.release()
