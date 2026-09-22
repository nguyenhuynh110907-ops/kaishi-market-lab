from __future__ import annotations

import asyncio
import json
from collections import deque
from datetime import UTC, datetime
from decimal import Decimal, ROUND_FLOOR
from uuid import uuid4
from zoneinfo import ZoneInfo
from typing import Any

from kaishi_bot.dashboard_models import AssetSnapshot, DashboardSettings, QuotePoint, RuntimeMode
from kaishi_bot.dashboard_store import DashboardStore
from kaishi_bot.lab import StrategyLab
from kaishi_bot.market_data import PublicKalshiMarketData
from kaishi_bot.paper import PaperBroker
from kaishi_bot.safety import SafetyGate
from kaishi_bot.domain import OrderRequest, Side
from kaishi_bot.pricing import to_v2_entry, to_v2_exit
from kaishi_bot.fees import FeeSchedule, fractional_contract_size, fractional_entry_cost
from kaishi_bot.entry_guard import GuardQuote, GuardReason, evaluate_entry
from kaishi_bot.market_data import OrderBookSnapshot
from kaishi_bot.production import OrderConfirmationPending


def json_safe(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return json_safe(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


class DashboardRuntime:
    def __init__(
        self, store: DashboardStore, market_data: PublicKalshiMarketData | None,
        gate: SafetyGate, production: object | None = None,
    ) -> None:
        self.store = store
        self.market_data = market_data
        self.gate = gate
        self.production = production
        self.paper = PaperBroker(store)
        self.paper_model_shadow: object | None = None
        self.lab = StrategyLab(store)
        self.assets = {asset: AssetSnapshot(asset=asset) for asset in store.load_settings().assets}
        self.latest_quotes: dict[str, QuotePoint] = {}
        self.version = 0
        self.subscribers: set[asyncio.Queue[str]] = set()
        self.task: asyncio.Task[None] | None = None
        self.account: dict[str, object] | None = None
        self.fee_schedules: dict[str, FeeSchedule] = {}
        self.lab_tasks: set[asyncio.Task[None]] = set()
        self.lab_task_ids: set[int] = set()
        self.last_account_refresh: datetime | None = None
        self.live_alert: str | None = None
        self.account_refresh_task: asyncio.Task[None] | None = None
        self.positions_refresh_task: asyncio.Task[None] | None = None
        self.lab_realtime_queue: asyncio.Queue[
            tuple[int, object, QuotePoint, int]
        ] = asyncio.Queue(maxsize=500)
        self.lab_realtime_task: asyncio.Task[None] | None = None
        self.lab_realtime_store: DashboardStore | None = None
        self.lab_realtime: StrategyLab | None = None
        self.live_stream_task: asyncio.Task[None] | None = None
        self.live_stream_connected = False
        self.last_full_publish: datetime | None = None
        self.last_quote_publish: dict[str, datetime] = {}
        self._snapshot_cache: dict[str, object] | None = None

    async def start(self) -> None:
        self._snapshot_cache = self.snapshot()
        for run in self.lab.list_runs():
            if run["status"] == "backfilling":
                self.start_lab_backfill(int(run["id"]))
        if self.market_data and not self.task:
            self.lab_realtime_store = DashboardStore(self.store.path)
            self.lab_realtime = StrategyLab(self.lab_realtime_store)
            self.lab_realtime_task = asyncio.create_task(
                self._lab_realtime_loop(), name="strategy-lab-realtime-worker"
            )
            self.task = asyncio.create_task(self._loop(), name="kalshi-dashboard-market-loop")
            if self.production and hasattr(self.production, "quote_stream"):
                self.live_stream_task = asyncio.create_task(
                    self._live_stream_loop(), name="kalshi-live-websocket"
                )

    async def close(self) -> None:
        for lab_task in self.lab_tasks:
            lab_task.cancel()
        if self.lab_tasks:
            await asyncio.gather(*self.lab_tasks, return_exceptions=True)
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        if self.account_refresh_task:
            self.account_refresh_task.cancel()
            await asyncio.gather(self.account_refresh_task, return_exceptions=True)
        if self.positions_refresh_task:
            self.positions_refresh_task.cancel()
            await asyncio.gather(self.positions_refresh_task, return_exceptions=True)
        if self.lab_realtime_task:
            self.lab_realtime_task.cancel()
            await asyncio.gather(self.lab_realtime_task, return_exceptions=True)
        if self.lab_realtime_store:
            self.lab_realtime_store.close()
        if self.live_stream_task:
            self.live_stream_task.cancel()
            await asyncio.gather(self.live_stream_task, return_exceptions=True)
        if self.market_data:
            await self.market_data.close()
        if self.production and hasattr(self.production, "close"):
            await self.production.close()

    def start_lab_backfill(self, run_id: int) -> None:
        if run_id in self.lab_task_ids:
            return
        task = asyncio.create_task(
            self._run_lab_backfill(run_id), name=f"strategy-lab-history-{run_id}"
        )
        self.lab_tasks.add(task)
        self.lab_task_ids.add(run_id)

        def finished(done: asyncio.Task[None]) -> None:
            self.lab_tasks.discard(done)
            self.lab_task_ids.discard(run_id)

        task.add_done_callback(finished)

    def stop_active_labs(self) -> list[int]:
        stopped: list[int] = []
        for run in self.lab.list_runs():
            if run["status"] in {"running", "backfilling"}:
                run_id = int(run["id"])
                if self.lab.stop(run_id):
                    stopped.append(run_id)
        return stopped

    async def _run_lab_backfill(self, run_id: int) -> None:
        run = self.store.connection.execute(
            "SELECT * FROM lab_runs WHERE id=?", (run_id,)
        ).fetchone()
        if run is None or run["status"] != "backfilling":
            return
        try:
            assets = json.loads(run["assets"])
            requested = int(run["history_cycles_requested"])
            selection_time = datetime.fromisoformat(str(run["started_at"]))
            candidates = self.lab.history_cycle_candidates(
                assets, int(run["history_cutoff_event_id"] or 0), selection_time,
            )
            loaded = {asset: 0 for asset in assets}
            for item in candidates:
                asset, ticker = item["asset"], item["ticker"]
                if loaded[asset] >= requested:
                    continue
                cached = self.store.market_result(ticker)
                if cached is not None:
                    loaded[asset] += 1
                    continue
                if self.market_data is None or not hasattr(self.market_data, "result"):
                    continue
                try:
                    winning_side = await self.market_data.result(ticker)
                except Exception:
                    await asyncio.sleep(0)
                    continue
                if winning_side:
                    self.store.save_market_result(
                        ticker, winning_side, datetime.fromisoformat(item["close_time"])
                    )
                    loaded[asset] += 1
                await asyncio.sleep(0)
            await self.lab.backfill(run_id)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            with self.store.connection:
                self.store.connection.execute(
                    """UPDATE lab_runs SET status='failed',error_message=?
                    WHERE id=? AND status='backfilling'""", (str(error), run_id),
                )

    async def _loop(self) -> None:
        while True:
            started = datetime.now(UTC)
            try:
                await self.poll_once(started)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self.gate.disarm()
                self.account = {"error": f"Live runtime recovered after error: {error}"}
                await self.publish()
            elapsed = (datetime.now(UTC) - started).total_seconds()
            await asyncio.sleep(max(0.1, 1.0 - elapsed))

    def _stream_tickers(self) -> tuple[str, ...]:
        return tuple(sorted(
            state.market.ticker for state in self.assets.values()
            if state.market is not None and state.status != "disabled"
        ))

    async def _live_stream_loop(self) -> None:
        while True:
            tickers = self._stream_tickers()
            if not tickers:
                self.live_stream_connected = False
                await asyncio.sleep(0.25)
                continue
            try:
                async for ticker, quote in self.production.quote_stream(list(tickers)):
                    self.live_stream_connected = True
                    if self._stream_tickers() != tickers:
                        break
                    await self._handle_live_stream_quote(ticker, quote)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self.live_stream_connected = False
                self.live_alert = f"WebSocket giá đang nối lại: {error}"
                await asyncio.sleep(1)
            finally:
                self.live_stream_connected = False

    async def _restart_live_stream(self) -> None:
        """Reconnect immediately when the active 15-minute tickers roll over."""
        previous = self.live_stream_task
        self.live_stream_connected = False
        if previous is not None and previous is not asyncio.current_task():
            previous.cancel()
            # Some SDK websocket versions can block in context-manager cleanup.
            # Never let that cleanup hold the market loop and REST fallback.
            def consume_result(done: asyncio.Task[None]) -> None:
                try:
                    done.exception()
                except asyncio.CancelledError:
                    pass

            previous.add_done_callback(consume_result)
        if self.production and hasattr(self.production, "quote_stream"):
            self.live_stream_task = asyncio.create_task(
                self._live_stream_loop(), name="kalshi-live-websocket"
            )

    async def _handle_live_stream_quote(
        self, ticker: str, quote: QuotePoint,
    ) -> None:
        state = next(
            (
                item for item in self.assets.values()
                if item.market is not None and item.market.ticker == ticker
            ),
            None,
        )
        if state is None or state.market is None:
            return
        settings = self.store.load_settings()
        active_runs = (
            [] if settings.mode is RuntimeMode.LIVE
            else [item for item in self.lab.list_runs() if item["status"] == "running"]
        )
        if self.live_alert and self.live_alert.startswith("WebSocket giá đang nối lại:"):
            self.live_alert = None
        previous_quote = self.latest_quotes.get(ticker)
        state.quote = quote
        # Trading sees every ticker change; the visual chart only needs one
        # point per second and remains a compact 15-minute series.
        if (
            state.chart
            and int(state.chart[-1].observed_at.timestamp())
            == int(quote.observed_at.timestamp())
        ):
            state.chart[-1] = quote
        else:
            state.chart = (state.chart + [quote])[-900:]
        state.status = "live"
        state.error = None
        self.latest_quotes[ticker] = quote
        event_id = None
        if settings.log_quotes or active_runs:
            event_id = self.store.record_quote(
                state.asset, ticker, quote, state.market.close_time
            )
        schedule = self.fee_schedules.get(state.market.series)
        if schedule is None and hasattr(self.market_data, "fee_schedule"):
            try:
                schedule = await self.market_data.fee_schedule(state.market.series)
                self.fee_schedules[state.market.series] = schedule
            except Exception as error:
                if settings.mode is RuntimeMode.PAPER:
                    state.error = f"Fee metadata unavailable: {error}"
                schedule = None
        if settings.mode is RuntimeMode.PAPER and schedule is not None:
            paper_settings = (
                settings.model_copy(update={"bot_enabled": False})
                if self.paper_model_shadow is not None
                and self.paper_model_shadow.active
                else settings
            )
            self.paper.on_quote(
                state.market, quote, paper_settings, quote.observed_at,
                fee_schedule=schedule, previous_quote=previous_quote, depth=None,
                include_model_positions=not (
                    self.paper_model_shadow is not None
                    and self.paper_model_shadow.active
                ),
            )
            if self.paper_model_shadow is not None:
                self.paper_model_shadow.on_quote(state.market, quote, schedule)
        elif settings.mode is RuntimeMode.LIVE:
            # Existing positions are always protected before new entries.
            await self._live_exits(settings, quote.observed_at)
        if (
            settings.mode is RuntimeMode.LIVE and settings.bot_enabled
            and self.gate.armed
        ):
            await self._live_entries(
                state.market, quote, settings, previous_quote, schedule, None
            )
        for run in active_runs:
            assert event_id is not None
            item = (int(run["id"]), state.market, quote, event_id)
            if self.lab_realtime_task is None:
                self.lab.on_quote(*item)
                continue
            try:
                self.lab_realtime_queue.put_nowait(item)
            except asyncio.QueueFull:
                self.live_alert = (
                    "Strategy Lab Ä‘ang quÃ¡ táº£i; Live váº«n Ä‘Æ°á»£c Æ°u tiÃªn"
                )
        previous_publish = self.last_quote_publish.get(ticker)
        if (
            previous_publish is None
            or (quote.observed_at - previous_publish).total_seconds() >= 0.25
        ):
            self.last_quote_publish[ticker] = quote.observed_at
            await self.publish_quote(state.asset, ticker, quote)

    async def poll_once(self, now: datetime | None = None) -> None:
        if not self.market_data:
            return
        observed_at = now or datetime.now(UTC)
        settings = self.store.load_settings()
        active_runs = (
            [] if settings.mode is RuntimeMode.LIVE
            else [item for item in self.lab.list_runs() if item["status"] == "running"]
        )
        active_run_ids = [int(item["id"]) for item in active_runs]
        paper_assets = {
            str(position["asset"]) for position in self.store.open_positions()
        }
        live_tickers = {
            str(position.get("ticker", ""))
            for position in (self.account or {}).get("positions", [])
            if Decimal(str(position.get("quantity", "0"))) > 0
        }
        live_assets = {
            asset for asset, asset_settings in settings.assets.items()
            if any(ticker.startswith(asset_settings.series) for ticker in live_tickers)
        }
        required_assets = {
            asset for asset, asset_settings in settings.assets.items()
            if asset_settings.enabled
        } | paper_assets | live_assets
        current_tickers = {
            state.market.ticker for state in self.assets.values() if state.market is not None
        }
        expired_tickers = {
            state.market.ticker for state in self.assets.values()
            if state.market is not None and state.market.close_time <= observed_at
        }
        pending_paper_tickers = {
            str(position["ticker"]) for position in self.store.open_positions()
            if str(position["ticker"]) not in current_tickers
            or str(position["ticker"]) in expired_tickers
        }
        if active_run_ids:
            placeholders = ",".join("?" for _ in active_run_ids)
            lab_positions = list(self.store.connection.execute(
                f"""SELECT DISTINCT run_id,ticker FROM lab_positions
                WHERE status='open' AND run_id IN ({placeholders})""",
                active_run_ids,
            ))
        else:
            # Avoid a full scan of historical lab positions every second.
            lab_positions = []
        pending_lab: dict[str, list[int]] = {}
        for position in lab_positions:
            ticker = str(position["ticker"])
            if ticker not in current_tickers or ticker in expired_tickers:
                pending_lab.setdefault(ticker, []).append(int(position["run_id"]))
        pending_run_ids = {
            run_id for run_ids in pending_lab.values() for run_id in run_ids
        }
        awaiting_run_ids = {
            int(item["id"]) for item in active_runs
            if item.get("settlement_status") == "awaiting_settlement"
        }
        unresolved_run_ids: set[int] = set()
        for ticker in sorted(pending_paper_tickers | set(pending_lab)):
            try:
                winning_side = await self.market_data.result(ticker)
                if winning_side:
                    self.store.save_market_result(ticker, winning_side, observed_at)
                    self.paper.settle_ticker(ticker, winning_side, observed_at)
                    for run_id in pending_lab.get(ticker, []):
                        self.lab.settle_ticker(run_id, ticker, winning_side, observed_at)
                else:
                    unresolved_run_ids.update(pending_lab.get(ticker, []))
            except Exception:
                unresolved_run_ids.update(pending_lab.get(ticker, []))
        with self.store.connection:
            for run_id in pending_run_ids | awaiting_run_ids:
                status = (
                    "awaiting_settlement" if run_id in unresolved_run_ids else "ready"
                )
                self.store.connection.execute(
                    "UPDATE lab_runs SET settlement_status=? WHERE id=?",
                    (status, run_id),
                )
        stream_tickers_changed = False
        for asset, asset_settings in settings.assets.items():
            state = self.assets[asset]
            if asset not in required_assets:
                if state.market is not None:
                    self.latest_quotes.pop(state.market.ticker, None)
                state.market = None
                state.quote = None
                state.chart = []
                state.status = "disabled"
                state.error = "Đã tắt · không tải giá"
                continue
            if state.market is not None and state.market.close_time > observed_at:
                continue
            previous_ticker = state.market.ticker if state.market is not None else None
            try:
                state.market = await self.market_data.discover(asset, asset_settings.series, observed_at)
                if previous_ticker and previous_ticker != state.market.ticker:
                    self.latest_quotes.pop(previous_ticker, None)
                    stream_tickers_changed = True
                state.chart = []
                state.status = "connecting"
                state.error = None
            except Exception as error:
                old_ticker = state.market.ticker if state.market is not None else previous_ticker
                state.market = None
                state.quote = None
                state.chart = []
                state.status = "unavailable"
                state.error = str(error)
                if old_ticker:
                    self.latest_quotes.pop(old_ticker, None)
        if stream_tickers_changed:
            await self._restart_live_stream()
        market_by_ticker = {
            state.market.ticker: state for state in self.assets.values() if state.market is not None
        }
        streaming_live = (
            self.live_stream_connected
            and bool(market_by_ticker)
            and all(
                state.quote is not None
                and (observed_at - state.quote.observed_at).total_seconds() <= 3
                for state in market_by_ticker.values()
            )
        )
        if streaming_live:
            quotes: dict[str, QuotePoint] = {}
        else:
            try:
                quotes = await self.market_data.quotes(list(market_by_ticker))
            except Exception as error:
                for state in market_by_ticker.values():
                    state.status = "stale"
                    state.error = str(error)
                self.gate.mark_data_stale()
                await self.publish()
                return
        lab_events: list[tuple[int, object, QuotePoint, int]] = []
        missing_stale = False
        for ticker, state in market_by_ticker.items():
            if streaming_live:
                continue
            if ticker in quotes:
                continue
            quote_age = (
                (observed_at - state.quote.observed_at).total_seconds()
                if state.quote is not None else float("inf")
            )
            if quote_age > 3:
                state.status = "stale"
                state.error = "Kalshi did not return a fresh quote"
                missing_stale = True
        if missing_stale:
            self.gate.mark_data_stale()
        for ticker, quote in quotes.items():
            state = market_by_ticker.get(ticker)
            if not state or not state.market:
                continue
            previous_quote = self.latest_quotes.get(ticker)
            state.quote = quote
            state.chart = (state.chart + [quote])[-900:]
            state.status = "live"
            state.error = None
            self.latest_quotes[ticker] = quote
            event_id = None
            if settings.log_quotes or active_runs:
                event_id = self.store.record_quote(
                    state.asset, ticker, quote, state.market.close_time
                )
            if settings.mode is RuntimeMode.PAPER:
                schedule = self.fee_schedules.get(state.market.series)
                if schedule is None and hasattr(self.market_data, "fee_schedule"):
                    try:
                        schedule = await self.market_data.fee_schedule(state.market.series)
                        self.fee_schedules[state.market.series] = schedule
                    except Exception as error:
                        state.error = f"Fee metadata unavailable: {error}"
                if schedule is not None:
                    paper_settings = (
                        settings.model_copy(update={"bot_enabled": False})
                        if self.paper_model_shadow is not None
                        and self.paper_model_shadow.active
                        else settings
                    )
                    self.paper.on_quote(
                        state.market, quote, paper_settings, quote.observed_at,
                        fee_schedule=schedule, previous_quote=previous_quote, depth=None,
                        include_model_positions=not (
                            self.paper_model_shadow is not None
                            and self.paper_model_shadow.active
                        ),
                    )
                    if self.paper_model_shadow is not None:
                        self.paper_model_shadow.on_quote(
                            state.market, quote, schedule
                        )
            elif (
                settings.mode is RuntimeMode.LIVE and settings.bot_enabled
                and self.gate.armed and self.production
            ):
                schedule = self.fee_schedules.get(state.market.series)
                if schedule is None and hasattr(self.market_data, "fee_schedule"):
                    try:
                        schedule = await self.market_data.fee_schedule(state.market.series)
                        self.fee_schedules[state.market.series] = schedule
                    except Exception:
                        schedule = None
                await self._live_entries(
                    state.market, quote, settings, previous_quote, schedule, None
                )
            for run in active_runs:
                assert event_id is not None
                lab_events.append((int(run["id"]), state.market, quote, event_id))
        if settings.mode in (RuntimeMode.READ_ONLY, RuntimeMode.LIVE) and self.production:
            if self.account is None:
                self.account = {
                    "positions": [], "orders": [], "fills": [], "settlements": [],
                }
            if settings.mode is RuntimeMode.LIVE:
                await self._live_exits(settings, observed_at)
            if self.positions_refresh_task is None or self.positions_refresh_task.done():
                self.positions_refresh_task = asyncio.create_task(
                    self._refresh_positions(settings, observed_at),
                    name="kalshi-positions-refresh",
                )
            if (
                self.last_account_refresh is None
                or (observed_at - self.last_account_refresh).total_seconds() >= 10
            ):
                self.last_account_refresh = observed_at
                if self.account_refresh_task is None or self.account_refresh_task.done():
                    self.account_refresh_task = asyncio.create_task(
                        self._refresh_account(), name="kalshi-account-refresh"
                    )
        for item in lab_events:
            if self.lab_realtime_task is None:
                self.lab.on_quote(*item)
                continue
            try:
                self.lab_realtime_queue.put_nowait(item)
            except asyncio.QueueFull:
                self.live_alert = "Strategy Lab đang quá tải; Live vẫn được ưu tiên"
        if not streaming_live or (
            self.last_full_publish is None
            or (observed_at - self.last_full_publish).total_seconds() >= 10
        ):
            await self.publish()
            self.last_full_publish = observed_at

    async def _lab_realtime_loop(self) -> None:
        while True:
            item = await self.lab_realtime_queue.get()
            try:
                assert self.lab_realtime is not None
                await asyncio.to_thread(self.lab_realtime.on_quote, *item)
            except asyncio.CancelledError:
                raise
            except Exception:
                # Lab errors must never stop or disarm live trading.
                pass
            finally:
                self.lab_realtime_queue.task_done()

    async def _refresh_positions(
        self, settings: DashboardSettings, observed_at: datetime,
    ) -> None:
        try:
            if hasattr(self.production, "positions_snapshot"):
                positions = await self.production.positions_snapshot()
            else:
                positions = list((await self.production.account_snapshot()).get("positions", []))
            if self.account is None:
                self.account = {}
            self.account["positions"] = positions
            self.account.pop("error", None)
            self.store.reconcile_live_positions(
                positions, observed_at,
            )
            if (
                not self.store.live_protection_status()
                and self.live_alert
                and "đang chờ Kalshi xác nhận" in self.live_alert
            ):
                self.live_alert = None
            if settings.mode is RuntimeMode.LIVE:
                await self._live_exits(settings, datetime.now(UTC))
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.live_alert = f"Lỗi đọc vị thế Live: {error}"
            self.gate.disarm()

    async def _refresh_account(self) -> None:
        try:
            full = await self.production.account_snapshot()
            external_activity = self.store.reconcile_live_account(
                list(full.get("fills", [])), list(full.get("orders", [])),
                list(full.get("positions", [])), list(full.get("settlements", [])),
            )
            if external_activity:
                self.gate.disarm()
                tickers = ", ".join(sorted({
                    str(item.get("ticker", "")) for item in external_activity
                }))
                self.live_alert = (
                    f"Phát hiện lệnh ngoài runtime ({tickers}); Live đã tự khóa"
                )
            if self.account is not None and "positions" in self.account:
                full["positions"] = self.account["positions"]
            self.account = full
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.live_alert = f"Không làm mới được lịch sử tài khoản: {error}"

    async def _live_exits(self, settings: DashboardSettings, observed_at: datetime) -> None:
        for position in (self.account or {}).get("positions", []):
            ticker = str(position.get("ticker", ""))
            quote = self.latest_quotes.get(ticker)
            if quote is None or position.get("side") not in {"up", "down"}:
                continue
            side = Side(str(position["side"]))
            quantity = Decimal(str(position.get("quantity", "0")))
            bid = quote.up_bid if side is Side.UP else quote.down_bid
            if quantity <= 0:
                continue
            protection = self.store.live_exit_lock(ticker, side.value)
            if protection and protection.get("last_error") == "fill_confirmed":
                # Kalshi has filled the IOC. Keep the lock until the positions
                # endpoint confirms flat so a stale snapshot cannot trigger a
                # duplicate TP/SL and overwrite the successful status.
                continue
            if protection and str(protection.get("last_error", "")).startswith(
                "confirmation_pending:"
            ):
                placed_at = datetime.fromisoformat(str(protection["placed_at"]))
                if (observed_at - placed_at).total_seconds() < 2:
                    # Do not resend the same client id while Kalshi's order/fill
                    # views are converging. Position reconciliation will remove
                    # the lock as soon as the IOC fill is visible.
                    continue
                # IOC orders cannot remain resting. If the position is still
                # present, retry safely with a fresh id; reduce_only prevents an
                # over-close if the previous response was merely delayed.
                self.store.release_live_exit(str(protection["client_order_id"]))
                protection = None
            # Kalshi only permits reduce_only with IOC. Remove any legacy GTC
            # protection created by older versions, then protect locally from
            # the websocket quote stream.
            if (
                protection and protection.get("reason") == "take_profit"
                and protection.get("order_kind", "legacy") == "legacy"
            ):
                order_id = str(protection.get("order_id") or "")
                try:
                    if order_id:
                        await self.production.cancel_order(order_id)
                except Exception as error:
                    self.live_alert = f"Không hủy được TP cũ {ticker}: {error}"
                    self.gate.disarm()
                    continue
                self.store.release_live_exit(str(protection["client_order_id"]))
                protection = None
            if bid <= settings.stop_loss:
                self.store.lock_live_trade_exit(ticker, side.value)
                await self._submit_live_exit(
                    ticker, side, quantity, bid, "stop_loss", observed_at, protection
                )
                continue
            if bid >= settings.take_profit:
                self.store.lock_live_trade_exit(ticker, side.value)
                await self._submit_live_exit(
                    ticker, side, quantity, bid, "take_profit", observed_at, protection
                )

    async def _submit_live_exit(
        self, ticker: str, side: Side, quantity: Decimal, bid: Decimal,
        reason: str, observed_at: datetime, existing: dict[str, object] | None = None,
    ) -> None:
        client_order_id = str(existing["client_order_id"]) if existing else str(uuid4())
        if existing is None and not self.store.reserve_live_exit(
            ticker, side.value, client_order_id, reason, observed_at, "ioc"
        ):
            return
        book_side, yes_price = to_v2_exit(side, bid)
        request = OrderRequest(
            ticker=ticker, client_order_id=client_order_id, book_side=book_side,
            count=quantity, yes_price=yes_price, reduce_only=True,
            time_in_force="immediate_or_cancel",
        )
        self.store.record_live_order_intent(
            client_order_id, ticker, side.value, "exit", reason,
            quantity, bid, observed_at,
            str((self.store.active_live_trade(ticker) or {}).get("trade_id") or "") or None,
        )
        day = observed_at.astimezone(ZoneInfo("America/New_York")).date().isoformat()
        try:
            result = await self.production.place_guarded_order(
                request, Decimal("0"), self.store.live_daily_spend(day)
            )
        except OrderConfirmationPending as error:
            self.store.record_live_exit_error(
                client_order_id, f"confirmation_pending: {error}"
            )
            self.live_alert = f"{reason.upper()} đang chờ Kalshi xác nhận {ticker}"
            return
        except Exception as error:
            self.store.record_live_exit_error(client_order_id, str(error))
            self.live_alert = f"{reason.upper()} chưa gửi được {ticker}: {error}"
            self.gate.disarm()
            return
        if result.fill_count is not None and result.fill_count <= 0:
            self.store.record_live_order_result(client_order_id, result.order_id)
            self.store.release_live_exit(client_order_id)
            self.live_alert = f"{reason.upper()} không khớp {ticker}; sẽ thử lại nhịp sau"
        elif result.fill_count is None:
            self.store.record_live_order_result(client_order_id, result.order_id)
            self.store.record_live_exit(client_order_id, result.order_id)
            self.live_alert = f"{reason.upper()} đang chờ Kalshi xác nhận {ticker}"
        else:
            self.store.record_live_order_result(client_order_id, result.order_id)
            filled = min(quantity, result.fill_count)
            positions = list((self.account or {}).get("positions", []))
            for item in positions:
                if str(item.get("ticker")) != ticker or str(item.get("side")) != side.value:
                    continue
                remaining = max(Decimal("0"), quantity - filled)
                item["quantity"] = str(remaining)
                item["position"] = str(remaining if side is Side.UP else -remaining)
            if self.account is not None:
                self.account["positions"] = [
                    item for item in positions
                    if Decimal(str(item.get("quantity", "0"))) > 0
                ]
            if filled >= quantity:
                self.store.mark_live_exit_filled(client_order_id, result.order_id)
            else:
                self.store.release_live_exit(client_order_id)
            self.live_alert = None

    async def _live_entries(
        self, market: object, quote: QuotePoint, settings: DashboardSettings,
        previous_quote: QuotePoint | None = None,
        fee_schedule: FeeSchedule | None = None,
        depth: OrderBookSnapshot | None = None,
    ) -> None:
        asset_settings = settings.assets[market.asset]
        if not asset_settings.enabled:
            return
        elapsed = (quote.observed_at - market.open_time).total_seconds()
        if not settings.entry_start_seconds <= elapsed < settings.entry_end_seconds:
            return
        day = quote.observed_at.astimezone(ZoneInfo("America/New_York")).date().isoformat()
        for side, enabled, ask, bid in (
            (Side.UP, asset_settings.trade_up, quote.up_ask, quote.up_bid),
            (Side.DOWN, asset_settings.trade_down, quote.down_ask, quote.down_bid),
        ):
            if not enabled:
                continue
            # One logical trade owns a ticker. It may accumulate through many
            # IOC child fills, but UP and DOWN can never overlap.
            trade = self.store.active_live_trade(market.ticker)
            if trade is not None and (
                str(trade["side"]) != side.value
                or bool(trade["exit_locked"])
                or str(trade["phase"]) == "exit_locked"
            ):
                continue
            if any(
                str(item.get("ticker")) == market.ticker
                and str(item.get("side")) != side.value
                and Decimal(str(item.get("quantity", "0"))) > 0
                for item in (self.account or {}).get("positions", [])
            ):
                continue
            if trade is None and any(
                str(item.get("ticker")) == market.ticker
                and Decimal(str(item.get("quantity", "0"))) > 0
                for item in (self.account or {}).get("positions", [])
            ):
                # Never adopt or top up a position that lacks our durable
                # trade_id (for example a manual Kalshi position).
                continue
            used = self.store.live_daily_spend(day)
            target_budget = (
                Decimal(str(trade["target_budget"])) if trade is not None
                else min(settings.entry_amount, Decimal("200"))
            )
            filled_outlay = Decimal("0")
            if trade is not None:
                _, filled_outlay = self.store.live_trade_entry_totals(
                    str(trade["trade_id"])
                )
            remaining_target = max(Decimal("0"), target_budget - filled_outlay)
            budget = min(remaining_target, settings.daily_cap - used)
            if budget < Decimal("0.01"):
                if trade is not None and remaining_target < Decimal("0.01"):
                    self.store.set_live_trade_phase(str(trade["trade_id"]), "holding")
                continue
            if fee_schedule is None:
                self.store.increment_guard_counter(
                    "live", "account", GuardReason.DEPTH_UNAVAILABLE,
                    quote.observed_at,
                )
                continue
            previous = None
            if previous_quote is not None:
                previous = GuardQuote(
                    bid=previous_quote.up_bid if side is Side.UP else previous_quote.down_bid,
                    ask=previous_quote.up_ask if side is Side.UP else previous_quote.down_ask,
                    observed_at=previous_quote.observed_at,
                )
            decision = evaluate_entry(
                previous=previous,
                current=GuardQuote(bid=bid, ask=ask, observed_at=quote.observed_at),
                entry_min=settings.entry_min,
                entry_price=settings.entry_price, stop_loss=settings.stop_loss,
                take_profit=settings.take_profit, budget=budget,
                settings=settings.entry_guard, fee_schedule=fee_schedule,
                cooldown_until=None,
                fractional=True,
            )
            if not decision.eligible:
                self.store.increment_guard_counter(
                    "live", "account", decision.reason, quote.observed_at
                )
                continue
            if (
                depth is None
                or abs((depth.observed_at - quote.observed_at).total_seconds()) > 3
            ):
                if hasattr(self.market_data, "orderbook"):
                    try:
                        depth = await self.market_data.orderbook(market.ticker)
                    except Exception:
                        depth = None
                if depth is None:
                    self.store.increment_guard_counter(
                        "live", "account", GuardReason.DEPTH_UNAVAILABLE,
                        quote.observed_at,
                    )
                    continue
            # The orderbook fetch yields control. Re-read durable state before
            # reserving money so a concurrent TP/SL lock or completed child
            # entry cannot be followed by a stale top-up.
            if trade is not None:
                refreshed_trade = self.store.active_live_trade(market.ticker)
                if (
                    refreshed_trade is None
                    or str(refreshed_trade["trade_id"]) != str(trade["trade_id"])
                    or bool(refreshed_trade["exit_locked"])
                ):
                    continue
                trade = refreshed_trade
                _, refreshed_outlay = self.store.live_trade_entry_totals(
                    str(trade["trade_id"])
                )
                remaining_target = max(
                    Decimal("0"), Decimal(str(trade["target_budget"])) - refreshed_outlay
                )
                budget = min(
                    remaining_target, settings.daily_cap - self.store.live_daily_spend(day)
                )
                if budget < Decimal("0.01"):
                    if remaining_target < Decimal("0.01"):
                        self.store.set_live_trade_phase(str(trade["trade_id"]), "holding")
                    continue
            visible = depth.available(side.value, ask)
            fresh_quantity, _, _ = fractional_contract_size(
                fee_schedule, ask, budget
            )
            quantity = min(fresh_quantity, visible)
            quantity = (quantity / Decimal("0.01")).to_integral_value(
                rounding=ROUND_FLOOR
            ) * Decimal("0.01")
            if quantity <= 0:
                self.store.increment_guard_counter(
                    "live", "account", GuardReason.INSUFFICIENT_DEPTH,
                    quote.observed_at,
                )
                continue
            _, entry_fee, cost = fractional_entry_cost(fee_schedule, quantity, ask)
            if cost > budget:
                quantity, _, entry_fee = fractional_contract_size(
                    fee_schedule, ask, budget
                )
                _, entry_fee, cost = fractional_entry_cost(fee_schedule, quantity, ask)
            if quantity <= 0:
                continue
            if trade is None:
                trade = self.store.create_live_trade(
                    str(uuid4()), market.asset, market.ticker, side.value,
                    target_budget, quote.observed_at,
                )
                if trade is None:
                    continue
            client_order_id = str(uuid4())
            previous_used = self.store.reserve_live_entry(
                market.ticker, side.value, client_order_id, cost, day, quote.observed_at
            )
            if previous_used is None:
                continue
            book_side, yes_price = to_v2_entry(side, ask)
            request = OrderRequest(
                ticker=market.ticker, client_order_id=client_order_id,
                book_side=book_side, count=quantity, yes_price=yes_price,
                reduce_only=False, time_in_force="immediate_or_cancel",
            )
            self.store.record_live_order_intent(
                client_order_id, market.ticker, side.value, "entry", "entry",
                quantity, ask, quote.observed_at, str(trade["trade_id"]),
            )
            try:
                result = await self.production.place_guarded_order(request, cost, previous_used)
                if result.fill_count is not None and result.fill_count <= 0:
                    self.store.release_live_entry(client_order_id, day)
                    self.store.abandon_empty_live_trade(str(trade["trade_id"]))
                elif result.fill_count is None:
                    self.store.record_live_order(client_order_id, result.order_id)
                    self.store.record_live_order_result(client_order_id, result.order_id)
                    self.live_alert = f"ENTRY đang chờ Kalshi xác nhận {market.ticker}"
                else:
                    self.store.record_live_order(client_order_id, result.order_id)
                    self.store.record_live_order_result(client_order_id, result.order_id)
                    filled = result.fill_count
                    _, actual_fee, actual_cost = fractional_entry_cost(
                        fee_schedule, filled, ask
                    )
                    self.store.record_live_order_fill(
                        client_order_id, filled, actual_fee, actual_cost
                    )
                    self.store.finalize_live_entry(client_order_id, day, actual_cost)
                    _, total_outlay = self.store.live_trade_entry_totals(
                        str(trade["trade_id"])
                    )
                    self.store.set_live_trade_phase(
                        str(trade["trade_id"]),
                        "holding" if target_budget - total_outlay < Decimal("0.01")
                        else "accumulating",
                    )
                    if self.account is None:
                        self.account = {"positions": []}
                    positions = list(self.account.get("positions", []))
                    current = next((item for item in positions if
                        str(item.get("ticker")) == market.ticker
                        and str(item.get("side")) == side.value), None)
                    if current is None:
                        positions.append({
                            "ticker": market.ticker, "side": side.value,
                            "quantity": str(filled), "position": str(
                                filled if side is Side.UP else -filled
                            ),
                            "exposure": str(cost), "realized_pnl": "0",
                            "fees_paid": "0", "updated_at": quote.observed_at.isoformat(),
                        })
                    else:
                        total_quantity, total_outlay = self.store.live_trade_entry_totals(
                            str(trade["trade_id"])
                        )
                        current["quantity"] = str(total_quantity)
                        current["position"] = str(
                            total_quantity if side is Side.UP else -total_quantity
                        )
                        current["exposure"] = str(total_outlay)
                    self.account["positions"] = positions
            except Exception:
                self.store.release_live_entry(client_order_id, day)
                self.store.abandon_empty_live_trade(str(trade["trade_id"]))
                self.gate.disarm()
                raise

    def snapshot(self) -> dict[str, object]:
        settings = self.store.load_settings()
        paper = self.paper.snapshot(self.latest_quotes)
        today = datetime.now(UTC).astimezone(__import__("zoneinfo").ZoneInfo("America/New_York")).date().isoformat()
        paper_daily_used = self.store.daily_spend(today)
        live_daily_used = self.store.live_daily_spend(today)
        using_paper = settings.mode is RuntimeMode.PAPER
        daily_used = paper_daily_used if using_paper else live_daily_used
        daily_limit = settings.paper_daily_cap if using_paper else settings.daily_cap
        account_view = None if self.account is None else dict(self.account)
        if account_view is not None:
            account_view["fills"] = self.store.live_fills(limit=100)
        return json_safe({
            "version": self.version,
            "mode": settings.mode.value,
            "bot_enabled": settings.bot_enabled,
            "armed": self.gate.armed,
            "credentials_available": self.gate.credentials_available,
            "price_transport": (
                "websocket" if self.live_stream_connected else "rest"
            ),
            "live_alert": self.live_alert,
            "live_protection": self.store.live_protection_status(),
            "live_order_intents": self.store.live_order_intents(limit=100),
            "live_trades": self.store.live_trades(limit=100),
            "settings": settings,
            "assets": self.assets,
            "paper": paper,
            "daily_used": daily_used,
            "daily_limit": daily_limit,
            "daily_remaining": max(Decimal("0"), daily_limit - daily_used),
            "paper_daily_used": paper_daily_used,
            "paper_daily_limit": settings.paper_daily_cap,
            "paper_daily_remaining": max(
                Decimal("0"), settings.paper_daily_cap - paper_daily_used
            ),
            "live_daily_used": live_daily_used,
            "live_daily_limit": settings.daily_cap,
            "live_daily_remaining": max(
                Decimal("0"), settings.daily_cap - live_daily_used
            ),
            "account": account_view,
            "events": self.store.recent_events(),
            "guard_counters": self.store.guard_counters(
                "live" if settings.mode is RuntimeMode.LIVE else "paper", "account"
            ),
            "lab_runs": self.lab.list_runs(),
        })

    def latest_snapshot(self) -> dict[str, object]:
        """Return the last published full state without re-querying SQLite."""
        if self._snapshot_cache is None:
            self._snapshot_cache = self.snapshot()
        return self._snapshot_cache

    async def publish(self) -> None:
        self.version += 1
        snapshot = self.snapshot()
        self._snapshot_cache = snapshot
        payload = json.dumps(snapshot, separators=(",", ":"))
        for queue in list(self.subscribers):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(payload)

    async def publish_quote(self, asset: str, ticker: str, quote: QuotePoint) -> None:
        self.version += 1
        if self._snapshot_cache is not None:
            self._snapshot_cache["version"] = self.version
            self._snapshot_cache["price_transport"] = "websocket"
            self._snapshot_cache["armed"] = self.gate.armed
            cached_assets = self._snapshot_cache.get("assets")
            if isinstance(cached_assets, dict):
                cached_asset = cached_assets.get(asset)
                if isinstance(cached_asset, dict):
                    cached_asset["quote"] = json_safe(quote)
                    cached_asset["chart"] = json_safe(self.assets[asset].chart)
                    cached_asset["status"] = "live"
                    cached_asset["error"] = None
        payload = json.dumps(json_safe({
            "type": "quote", "version": self.version, "asset": asset,
            "ticker": ticker, "quote": quote, "armed": self.gate.armed,
            "bot_enabled": self.store.load_settings().bot_enabled,
            "price_transport": "websocket",
            "live_alert": self.live_alert,
            "live_protection": self.store.live_protection_status(),
        }), separators=(",", ":"))
        for queue in list(self.subscribers):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(payload)

    def subscribe(self) -> asyncio.Queue[str]:
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
        self.subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[str]) -> None:
        self.subscribers.discard(queue)
