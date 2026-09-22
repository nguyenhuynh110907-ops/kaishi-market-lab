import asyncio
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from kaishi_bot.dashboard_models import AssetMarket, QuotePoint, RuntimeMode
from kaishi_bot.dashboard_runtime import DashboardRuntime
from kaishi_bot.dashboard_store import DashboardStore
from kaishi_bot.safety import SafetyGate
from kaishi_bot.domain import OrderResult
from kaishi_bot.production import OrderConfirmationPending
from kaishi_bot.fees import FeeSchedule
from kaishi_bot.market_data import DepthLevel, OrderBookSnapshot


class MissingMarkets:
    async def discover(self, asset, series, now):
        raise RuntimeError("rollover pending")

    async def quotes(self, tickers):
        return {}

    async def close(self):
        pass


class PartialQuotes:
    async def discover(self, asset, series, now):
        return AssetMarket(
            asset=asset, series=series, ticker=f"{asset}-1", title=asset,
            open_time=now - timedelta(minutes=1), close_time=now + timedelta(minutes=10),
        )

    async def quotes(self, tickers):
        now = datetime.now(UTC)
        return {"BTC-1": QuotePoint(
            observed_at=now, up_bid="0.19", up_ask="0.20", down_bid="0.80", down_ask="0.81"
        )}

    async def close(self):
        pass


class LiveGateway:
    def __init__(self):
        self.orders = []
        self.positions = []
        self.cancelled = []
        self.account_reads = 0

    async def account_snapshot(self):
        self.account_reads += 1
        return {"balance": "1000.00", "portfolio_value": "1000.00", "positions": self.positions, "orders": []}

    async def positions_snapshot(self):
        return self.positions

    async def cancel_order(self, order_id):
        self.cancelled.append(order_id)

    async def place_guarded_order(self, request, entry_cost, daily_used):
        self.orders.append((request, entry_cost, daily_used))
        return OrderResult(
            f"order-{len(self.orders)}", request.client_order_id, request.count
        )

    async def close(self):
        pass


class ConfirmationPendingGateway(LiveGateway):
    async def place_guarded_order(self, request, entry_cost, daily_used):
        self.orders.append((request, entry_cost, daily_used))
        raise OrderConfirmationPending(request.client_order_id)


class ExternalExitGateway(LiveGateway):
    def __init__(self, observed_at):
        super().__init__()
        self.observed_at = observed_at

    async def account_snapshot(self):
        return {
            "balance": "1000.00", "portfolio_value": "1000.00",
            "positions": [],
            "orders": [{
                "order_id": "external-order", "client_order_id": "external-client",
                "status": "executed", "remaining_count": "0",
            }],
            "fills": [{
                "fill_id": "external-fill", "order_id": "external-order",
                "ticker": "BTC-1", "side": "yes", "action": "buy", "count": "1",
                "yes_price": "0.31", "no_price": "0.69", "fee_cost": "0.01",
                "is_taker": True,
                "created_at": (self.observed_at + timedelta(seconds=1)).isoformat(),
            }],
            "settlements": [],
        }


class HangingStreamGateway(LiveGateway):
    def __init__(self):
        super().__init__()
        self.subscriptions = []

    async def quote_stream(self, tickers):
        self.subscriptions.append(tuple(tickers))
        await asyncio.Future()
        if False:
            yield None


class CompleteQuotes(PartialQuotes):
    def __init__(self):
        self.quote_calls = 0
        self.base_time = datetime.now(UTC)

    async def quotes(self, tickers):
        now = self.base_time + timedelta(seconds=self.quote_calls)
        self.quote_calls += 1
        return {
            ticker: QuotePoint(
                observed_at=now,
                up_bid="0.19" if ticker == "BTC-1" else "0.59",
                up_ask="0.20" if ticker == "BTC-1" else "0.60",
                down_bid="0.39", down_ask="0.40",
            )
            for ticker in tickers
        }

    async def fee_schedule(self, series, refresh=False):
        return FeeSchedule("quadratic", Decimal("1"), Decimal("0.07"), "test")

    async def orderbook(self, ticker):
        return OrderBookSnapshot(
            ticker=ticker,
            observed_at=self.base_time + timedelta(seconds=self.quote_calls),
            up_asks=(DepthLevel(Decimal("0.20"), Decimal("100")),),
            down_asks=(DepthLevel(Decimal("0.40"), Decimal("100")),),
        )


class TrackingQuotes(CompleteQuotes):
    def __init__(self):
        super().__init__()
        self.discovered = []
        self.requested_tickers = []

    async def discover(self, asset, series, now):
        self.discovered.append(asset)
        return await super().discover(asset, series, now)

    async def quotes(self, tickers):
        self.requested_tickers.append(list(tickers))
        return await super().quotes(tickers)


class TakeProfitQuotes(CompleteQuotes):
    async def quotes(self, tickers):
        values = await super().quotes(tickers)
        now = datetime.now(UTC)
        values["BTC-1"] = QuotePoint(
            observed_at=now, up_bid="0.40", up_ask="0.41", down_bid="0.59", down_ask="0.60"
        )
        return values


class StopLossQuotes(CompleteQuotes):
    async def quotes(self, tickers):
        values = await super().quotes(tickers)
        values["BTC-1"] = QuotePoint(
            observed_at=datetime.now(UTC), up_bid="0.10", up_ask="0.11",
            down_bid="0.89", down_ask="0.90",
        )
        return values


class LabRolloverFeed(CompleteQuotes):
    def __init__(self) -> None:
        self.result_calls: list[str] = []

    async def discover(self, asset, series, now):
        return AssetMarket(
            asset=asset, series=series, ticker=f"{asset}-NEW", title=asset,
            open_time=now - timedelta(minutes=1), close_time=now + timedelta(minutes=14),
        )

    async def result(self, ticker):
        self.result_calls.append(ticker)
        return "up"


class MixedSettlementFeed(LabRolloverFeed):
    async def result(self, ticker):
        self.result_calls.append(ticker)
        return None if ticker == "BTC-WAIT" else "up"


@pytest.mark.asyncio
async def test_rollover_discovery_failure_clears_the_old_quote(tmp_path) -> None:
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        runtime = DashboardRuntime(store, MissingMarkets(), SafetyGate(credentials_available=False))
        runtime.assets["BTC"].quote = QuotePoint(
            observed_at=datetime.now(UTC), up_bid="0.4", up_ask="0.5",
            down_bid="0.5", down_ask="0.6",
        )
        await runtime.poll_once(datetime.now(UTC))

        assert runtime.assets["BTC"].quote is None
        assert runtime.assets["BTC"].chart == []
        assert runtime.assets["BTC"].status == "unavailable"


@pytest.mark.asyncio
async def test_partial_quote_response_marks_old_assets_stale_and_disarms_live(tmp_path) -> None:
    now = datetime.now(UTC)
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        gate = SafetyGate(credentials_available=True)
        gate.set_mode(RuntimeMode.LIVE)
        gate.arm(gate.create_challenge())
        runtime = DashboardRuntime(store, PartialQuotes(), gate)
        runtime.assets["ETH"].quote = QuotePoint(
            observed_at=now - timedelta(seconds=4), up_bid="0.4", up_ask="0.5",
            down_bid="0.5", down_ask="0.6",
        )
        await runtime.poll_once(now)

        assert runtime.assets["ETH"].status == "stale"
        assert gate.armed is False


@pytest.mark.asyncio
async def test_live_rollover_restarts_stuck_websocket_and_uses_rest_snapshot(tmp_path) -> None:
    now = datetime.now(UTC)
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        store.save_settings(store.load_settings().model_copy(
            update={"mode": RuntimeMode.LIVE, "bot_enabled": False}
        ))
        gate = SafetyGate(credentials_available=True)
        gate.set_mode(RuntimeMode.LIVE)
        gateway = HangingStreamGateway()
        runtime = DashboardRuntime(store, CompleteQuotes(), gate, gateway)
        runtime.assets["BTC"].market = AssetMarket(
            asset="BTC", series="KXBTC15M", ticker="BTC-OLD", title="BTC",
            open_time=now - timedelta(minutes=16), close_time=now - timedelta(seconds=1),
        )
        async def wait_forever():
            await asyncio.Future()

        previous = asyncio.create_task(wait_forever())
        runtime.live_stream_task = previous
        runtime.live_stream_connected = True

        await runtime.poll_once(now)
        await asyncio.sleep(0)

        assert previous.cancelled()
        assert runtime.live_stream_task is not previous
        assert runtime.assets["BTC"].market.ticker == "BTC-1"
        assert runtime.assets["BTC"].quote is not None
        assert gateway.subscriptions
        runtime.live_stream_task.cancel()
        await asyncio.gather(runtime.live_stream_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_connected_websocket_without_quotes_falls_back_to_rest(tmp_path) -> None:
    now = datetime.now(UTC)
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        store.save_settings(store.load_settings().model_copy(
            update={"mode": RuntimeMode.LIVE, "bot_enabled": False}
        ))
        gate = SafetyGate(credentials_available=True)
        gate.set_mode(RuntimeMode.LIVE)
        runtime = DashboardRuntime(store, CompleteQuotes(), gate, LiveGateway())
        runtime.live_stream_connected = True

        await runtime.poll_once(now)

        assert runtime.assets["BTC"].status == "live"
        assert runtime.assets["BTC"].quote is not None


def test_snapshot_reports_the_paper_daily_budget(tmp_path) -> None:
    today = datetime.now(UTC).astimezone(
        __import__("zoneinfo").ZoneInfo("America/New_York")
    ).date().isoformat()
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        with store.connection:
            store.connection.execute(
                "INSERT INTO paper_daily_spend(day,amount) VALUES (?,?)",
                (today, "50.0000"),
            )
        runtime = DashboardRuntime(store, None, SafetyGate(credentials_available=False))

        state = runtime.snapshot()

    assert state["daily_used"] == "50.0000"
    assert state["daily_limit"] == "1000.00"
    assert state["daily_remaining"] == "950.0000"


def test_live_snapshot_uses_only_the_real_order_budget(tmp_path) -> None:
    today = datetime.now(UTC).astimezone(
        __import__("zoneinfo").ZoneInfo("America/New_York")
    ).date().isoformat()
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        store.save_settings(store.load_settings().model_copy(
            update={"mode": RuntimeMode.LIVE}
        ))
        with store.connection:
            store.connection.execute(
                "INSERT INTO paper_daily_spend(day,amount) VALUES (?,?)",
                (today, "900.00"),
            )
            store.connection.execute(
                "INSERT INTO live_daily_spend(day,amount) VALUES (?,?)",
                (today, "12.50"),
            )
        runtime = DashboardRuntime(
            store, None, SafetyGate(credentials_available=True), LiveGateway()
        )

        state = runtime.snapshot()

    assert state["daily_used"] == "12.50"
    assert state["daily_limit"] == "1000.00"
    assert state["daily_remaining"] == "987.50"
    assert state["paper_daily_used"] == "900.00"
    assert state["live_daily_used"] == "12.50"


@pytest.mark.asyncio
async def test_paper_bot_fails_closed_when_verified_fee_metadata_is_missing(tmp_path) -> None:
    now = datetime.now(UTC)
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        store.save_settings(store.load_settings().model_copy(update={"bot_enabled": True}))
        runtime = DashboardRuntime(
            store, PartialQuotes(), SafetyGate(credentials_available=False)
        )
        await runtime.poll_once(now)
        await runtime.poll_once(now + timedelta(seconds=1))
        assert store.open_positions() == []


@pytest.mark.asyncio
async def test_armed_live_bot_places_guarded_entries_once_per_market_side(tmp_path) -> None:
    now = datetime.now(UTC)
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        settings = store.load_settings().model_copy(update={"mode": RuntimeMode.LIVE, "bot_enabled": True})
        store.save_settings(settings)
        gate = SafetyGate(credentials_available=True)
        gate.set_mode(RuntimeMode.LIVE)
        gate.arm(gate.create_challenge())
        gateway = LiveGateway()
        runtime = DashboardRuntime(store, CompleteQuotes(), gate, gateway)
        await runtime.poll_once(now)
        assert len(gateway.orders) == 1
        await runtime.poll_once(now + timedelta(seconds=1))

        assert len(gateway.orders) == 1
        request, cost, used = gateway.orders[0]
        assert request.ticker == "BTC-1"
        assert request.reduce_only is False
        assert request.time_in_force == "immediate_or_cancel"
        assert cost <= 10
        assert used == 0


@pytest.mark.asyncio
async def test_live_trade_accumulates_partial_ioc_fills_only_inside_entry_band(tmp_path) -> None:
    now = datetime.now(UTC)
    schedule = FeeSchedule("quadratic", Decimal("1"), Decimal("0.07"), "test")
    market = AssetMarket(
        asset="BTC", series="KXBTC15M", ticker="BTC-1", title="BTC",
        open_time=now - timedelta(minutes=1), close_time=now + timedelta(minutes=14),
    )
    depth = OrderBookSnapshot(
        ticker="BTC-1", observed_at=now,
        up_asks=(DepthLevel(Decimal("0.20"), Decimal("2")),), down_asks=(),
    )
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        settings = store.load_settings().model_copy(update={
            "mode": RuntimeMode.LIVE, "bot_enabled": True,
            "entry_amount": Decimal("1.00"), "entry_min": Decimal("0.15"),
            "entry_price": Decimal("0.25"),
            "assets": {
                key: value.model_copy(update={
                    "enabled": key == "BTC", "trade_up": key == "BTC", "trade_down": False,
                }) for key, value in store.load_settings().assets.items()
            },
        })
        gateway = LiveGateway()
        runtime = DashboardRuntime(
            store, CompleteQuotes(), SafetyGate(credentials_available=True), gateway
        )
        inside = QuotePoint(
            observed_at=now, up_bid="0.19", up_ask="0.20",
            down_bid="0.79", down_ask="0.80",
        )
        await runtime._live_entries(market, inside, settings, None, schedule, depth)
        later_depth = OrderBookSnapshot(
            ticker="BTC-1", observed_at=now + timedelta(seconds=1),
            up_asks=depth.up_asks, down_asks=(),
        )
        await runtime._live_entries(
            market, inside.model_copy(update={"observed_at": now + timedelta(seconds=1)}),
            settings, inside, schedule, later_depth,
        )
        assert len(gateway.orders) == 2
        assert all(item[0].time_in_force == "immediate_or_cancel" for item in gateway.orders)
        trade = store.active_live_trade("BTC-1")
        assert trade is not None and trade["phase"] == "accumulating"

        expanded = settings.model_copy(update={"entry_amount": Decimal("5.00")})
        third_quote = inside.model_copy(update={
            "observed_at": now + timedelta(seconds=2)
        })
        await runtime._live_entries(
            market, third_quote, expanded, inside, schedule, later_depth,
        )
        assert len(gateway.orders) == 3
        trade = store.active_live_trade("BTC-1")
        assert Decimal(str(trade["target_budget"])) == Decimal("1.00")
        assert trade["phase"] == "holding"
        await runtime._live_entries(
            market, third_quote.model_copy(update={
                "observed_at": now + timedelta(seconds=3)
            }), expanded, third_quote, schedule, later_depth,
        )
        assert len(gateway.orders) == 3

        outside = inside.model_copy(update={
            "observed_at": now + timedelta(seconds=4),
            "up_bid": Decimal("0.29"), "up_ask": Decimal("0.30"),
        })
        await runtime._live_entries(market, outside, expanded, third_quote, schedule, later_depth)
        assert len(gateway.orders) == 3

        store.lock_live_trade_exit("BTC-1", "up")
        await runtime._live_entries(
            market, inside.model_copy(update={"observed_at": now + timedelta(seconds=5)}),
            expanded, outside, schedule, later_depth,
        )
        assert len(gateway.orders) == 3


@pytest.mark.asyncio
async def test_entry_rechecks_exit_lock_after_orderbook_fetch(tmp_path) -> None:
    now = datetime.now(UTC)
    schedule = FeeSchedule("quadratic", Decimal("1"), Decimal("0.07"), "test")
    market = AssetMarket(
        asset="BTC", series="KXBTC15M", ticker="BTC-1", title="BTC",
        open_time=now - timedelta(minutes=1), close_time=now + timedelta(minutes=14),
    )
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        settings = store.load_settings().model_copy(update={
            "mode": RuntimeMode.LIVE, "bot_enabled": True,
            "entry_amount": Decimal("1"), "entry_min": Decimal("0.15"),
            "entry_price": Decimal("0.25"),
        })
        trade = store.create_live_trade(
            "trade-1", "BTC", "BTC-1", "up", Decimal("1"), now
        )
        assert trade is not None

        class LockDuringDepth(CompleteQuotes):
            async def orderbook(self, ticker):
                store.lock_live_trade_exit(ticker, "up")
                return OrderBookSnapshot(
                    ticker=ticker, observed_at=now,
                    up_asks=(DepthLevel(Decimal("0.20"), Decimal("100")),),
                    down_asks=(),
                )

        gateway = LiveGateway()
        runtime = DashboardRuntime(
            store, LockDuringDepth(), SafetyGate(credentials_available=True), gateway
        )
        quote = QuotePoint(
            observed_at=now, up_bid="0.19", up_ask="0.20",
            down_bid="0.79", down_ask="0.80",
        )
        await runtime._live_entries(market, quote, settings, None, schedule, None)
        assert gateway.orders == []
        assert store.active_live_trade("BTC-1")["phase"] == "exit_locked"


@pytest.mark.asyncio
async def test_live_entry_can_use_two_hundred_dollar_trade_budget(tmp_path) -> None:
    now = datetime.now(UTC)
    schedule = FeeSchedule("quadratic", Decimal("1"), Decimal("0.07"), "test")
    market = AssetMarket(
        asset="BTC", series="KXBTC15M", ticker="BTC-1", title="BTC",
        open_time=now - timedelta(minutes=1), close_time=now + timedelta(minutes=14),
    )
    quote = QuotePoint(
        observed_at=now, up_bid="0.19", up_ask="0.20",
        down_bid="0.79", down_ask="0.80",
    )
    depth = OrderBookSnapshot(
        ticker="BTC-1", observed_at=now,
        up_asks=(DepthLevel(Decimal("0.20"), Decimal("2000")),), down_asks=(),
    )
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        base = store.load_settings()
        settings = base.model_copy(update={
            "mode": RuntimeMode.LIVE, "bot_enabled": True,
            "entry_amount": Decimal("200"), "entry_min": Decimal("0.15"),
            "entry_price": Decimal("0.25"),
            "assets": {
                key: value.model_copy(update={
                    "enabled": key == "BTC", "trade_up": key == "BTC", "trade_down": False,
                }) for key, value in base.assets.items()
            },
        })
        gateway = LiveGateway()
        runtime = DashboardRuntime(
            store, CompleteQuotes(), SafetyGate(credentials_available=True), gateway
        )
        await runtime._live_entries(market, quote, settings, None, schedule, depth)
        assert len(gateway.orders) == 1
        _, cost, _ = gateway.orders[0]
        assert Decimal("10") < cost <= Decimal("200")
        assert Decimal(str(store.active_live_trade("BTC-1")["target_budget"])) == Decimal("200")


@pytest.mark.asyncio
async def test_disabled_assets_are_not_discovered_or_quoted(tmp_path) -> None:
    now = datetime.now(UTC)
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        settings = store.load_settings()
        settings = settings.model_copy(update={
            "assets": {
                asset: item.model_copy(update={"enabled": asset == "BTC"})
                for asset, item in settings.assets.items()
            }
        })
        store.save_settings(settings)
        feed = TrackingQuotes()
        runtime = DashboardRuntime(
            store, feed, SafetyGate(credentials_available=False), None
        )

        await runtime.poll_once(now)

        assert feed.discovered == ["BTC"]
        assert feed.requested_tickers == [["BTC-1"]]
        assert runtime.assets["ETH"].status == "disabled"
        assert runtime.assets["ETH"].quote is None


@pytest.mark.asyncio
async def test_disabled_asset_keeps_quotes_until_open_paper_position_closes(tmp_path) -> None:
    now = datetime.now(UTC)
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        settings = store.load_settings()
        settings = settings.model_copy(update={
            "assets": {
                asset: item.model_copy(update={"enabled": False})
                for asset, item in settings.assets.items()
            }
        })
        store.save_settings(settings)
        store.open_position(
            asset="ETH", ticker="ETH-1", side="up", quantity=Decimal("1"),
            entry_price=Decimal("0.20"), opened_at=now, day="2026-08-06",
        )
        feed = TrackingQuotes()
        runtime = DashboardRuntime(
            store, feed, SafetyGate(credentials_available=False), None
        )

        await runtime.poll_once(now)

        assert feed.discovered == ["ETH"]
        assert feed.requested_tickers == [["ETH-1"]]


@pytest.mark.asyncio
async def test_quote_sql_logging_can_be_disabled_without_stopping_realtime(tmp_path) -> None:
    now = datetime.now(UTC)
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        settings = store.load_settings().model_copy(update={"log_quotes": False})
        store.save_settings(settings)
        runtime = DashboardRuntime(
            store, CompleteQuotes(), SafetyGate(credentials_available=False), None
        )

        await runtime.poll_once(now)

        quote_count = store.connection.execute(
            "SELECT COUNT(*) FROM quote_events"
        ).fetchone()[0]
        assert quote_count == 0
        assert runtime.assets["BTC"].status == "live"
        assert runtime.assets["BTC"].quote is not None


@pytest.mark.asyncio
async def test_quote_stream_payload_is_compact_and_serialized_once(tmp_path) -> None:
    now = datetime.now(UTC)
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        runtime = DashboardRuntime(
            store, CompleteQuotes(), SafetyGate(credentials_available=False), None
        )
        queue = runtime.subscribe()
        quote = QuotePoint(
            observed_at=now, up_bid="0.60", up_ask="0.61",
            down_bid="0.39", down_ask="0.40",
        )

        await runtime.publish_quote("BTC", "BTC-1", quote)

        raw = await queue.get()
        payload = json.loads(raw)
        assert payload["type"] == "quote"
        assert payload["asset"] == "BTC"
        assert payload["quote"]["up_ask"] == "0.61"
        assert len(raw) < 1000


@pytest.mark.asyncio
async def test_armed_live_bot_honors_fractional_budget_and_entry_window(tmp_path) -> None:
    now = datetime.now(UTC)
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        settings = store.load_settings().model_copy(update={
            "mode": RuntimeMode.LIVE,
            "bot_enabled": True,
            "entry_amount": Decimal("0.01"),
            "entry_start_seconds": 120,
            "entry_end_seconds": 900,
        })
        store.save_settings(settings)
        gate = SafetyGate(credentials_available=True)
        gate.set_mode(RuntimeMode.LIVE)
        gate.arm(gate.create_challenge())
        gateway = LiveGateway()
        feed = CompleteQuotes()
        feed.base_time = now
        runtime = DashboardRuntime(store, feed, gate, gateway)

        await runtime.poll_once(now)
        assert gateway.orders == []

        for state in runtime.assets.values():
            if state.market is not None:
                state.market = state.market.model_copy(
                    update={"open_time": now - timedelta(seconds=121)}
                )
        await runtime.poll_once(now + timedelta(seconds=1))

        assert len(gateway.orders) == 1
        request, cost, _ = gateway.orders[0]
        assert request.count == Decimal("0.04")
        assert cost == Decimal("0.01")


@pytest.mark.asyncio
async def test_armed_live_bot_can_enter_during_the_final_minute(tmp_path) -> None:
    now = datetime.now(UTC)
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        settings = store.load_settings().model_copy(update={
            "mode": RuntimeMode.LIVE,
            "bot_enabled": True,
            "entry_amount": Decimal("0.01"),
            "entry_start_seconds": 800,
            "entry_end_seconds": 900,
            "min_seconds_before_close": 60,
        })
        store.save_settings(settings)
        gate = SafetyGate(credentials_available=True)
        gate.set_mode(RuntimeMode.LIVE)
        gate.arm(gate.create_challenge())
        gateway = LiveGateway()
        feed = CompleteQuotes()
        feed.base_time = now
        runtime = DashboardRuntime(store, feed, gate, gateway)

        await runtime.poll_once(now)
        assert gateway.orders == []

        for state in runtime.assets.values():
            if state.market is not None:
                state.market = state.market.model_copy(update={
                    "open_time": now - timedelta(seconds=870),
                    "close_time": now + timedelta(seconds=30),
                })
        await runtime.poll_once(now + timedelta(seconds=1))

        assert len(gateway.orders) == 1


@pytest.mark.asyncio
async def test_armed_live_bot_closes_real_position_at_take_profit(tmp_path) -> None:
    now = datetime.now(UTC)
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        store.save_settings(store.load_settings().model_copy(
            update={"mode": RuntimeMode.LIVE, "bot_enabled": True}
        ))
        gate = SafetyGate(credentials_available=True)
        gate.set_mode(RuntimeMode.LIVE)
        gate.arm(gate.create_challenge())
        gateway = LiveGateway()
        gateway.positions = [{"ticker": "BTC-1", "side": "up", "quantity": "3"}]
        runtime = DashboardRuntime(store, TakeProfitQuotes(), gate, gateway)
        await runtime.poll_once(now)
        await runtime.positions_refresh_task

        exits = [item for item in gateway.orders if item[0].reduce_only]
        assert len(exits) == 1
        assert exits[0][0].count == Decimal("3")
        assert exits[0][0].time_in_force == "immediate_or_cancel"
        lock = store.live_exit_lock("BTC-1", "up")
        assert lock["last_error"] == "fill_confirmed"
        assert lock["order_kind"] == "ioc"

        # A stale positions snapshot must not send another TP after the fill.
        runtime.account["positions"] = gateway.positions
        await runtime._live_exits(store.load_settings(), now + timedelta(seconds=1))
        assert len([item for item in gateway.orders if item[0].reduce_only]) == 1

        gateway.positions = []
        await runtime._refresh_positions(store.load_settings(), now + timedelta(seconds=2))
        assert store.live_exit_lock("BTC-1", "up") is None


@pytest.mark.asyncio
async def test_disarmed_live_still_protects_existing_position(tmp_path) -> None:
    now = datetime.now(UTC)
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        store.save_settings(store.load_settings().model_copy(
            update={"mode": RuntimeMode.LIVE, "bot_enabled": True}
        ))
        gate = SafetyGate(credentials_available=True)
        gate.set_mode(RuntimeMode.LIVE)
        gateway = LiveGateway()
        gateway.positions = [{"ticker": "BTC-1", "side": "up", "quantity": "0.06"}]
        runtime = DashboardRuntime(store, TakeProfitQuotes(), gate, gateway)

        await runtime.poll_once(now)
        await runtime.positions_refresh_task

        exits = [item for item in gateway.orders if item[0].reduce_only]
        entries = [item for item in gateway.orders if not item[0].reduce_only]
        assert len(exits) == 1
        assert exits[0][0].count == Decimal("0.06")
        assert entries == []


@pytest.mark.asyncio
async def test_external_fill_disarms_live_and_surfaces_alert(tmp_path) -> None:
    now = datetime.now(UTC)
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        store.save_settings(store.load_settings().model_copy(
            update={"mode": RuntimeMode.LIVE, "bot_enabled": True}
        ))
        store.create_live_trade(
            "trade-1", "BTC", "BTC-1", "down", Decimal("1"), now
        )
        gate = SafetyGate(credentials_available=True)
        gate.set_mode(RuntimeMode.LIVE)
        gate.arm(gate.create_challenge())
        runtime = DashboardRuntime(
            store, CompleteQuotes(), gate, ExternalExitGateway(now)
        )

        await runtime._refresh_account()

        assert gate.armed is False
        assert "Lệnh ngoài runtime".lower() in runtime.live_alert.lower()
        assert "BTC-1" in runtime.live_alert


@pytest.mark.asyncio
async def test_stop_loss_cancels_legacy_resting_tp_then_uses_ioc(tmp_path) -> None:
    now = datetime.now(UTC)
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        store.save_settings(store.load_settings().model_copy(
            update={"mode": RuntimeMode.LIVE, "bot_enabled": False}
        ))
        assert store.reserve_live_exit(
            "BTC-1", "up", "tp-client", "take_profit", now
        )
        store.record_live_exit("tp-client", "tp-order")
        gate = SafetyGate(credentials_available=True)
        gate.set_mode(RuntimeMode.LIVE)
        gateway = LiveGateway()
        gateway.positions = [{"ticker": "BTC-1", "side": "up", "quantity": "0.06"}]
        runtime = DashboardRuntime(store, StopLossQuotes(), gate, gateway)

        await runtime.poll_once(now)
        await runtime.positions_refresh_task

        assert gateway.cancelled == ["tp-order"]
        exits = [item[0] for item in gateway.orders if item[0].reduce_only]
        assert len(exits) == 1
        assert exits[0].time_in_force == "immediate_or_cancel"
        assert exits[0].count == Decimal("0.06")


@pytest.mark.asyncio
async def test_pending_stop_loss_confirmation_does_not_disarm_or_resend(tmp_path) -> None:
    now = datetime.now(UTC)
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        store.save_settings(store.load_settings().model_copy(
            update={"mode": RuntimeMode.LIVE, "bot_enabled": True}
        ))
        gate = SafetyGate(credentials_available=True)
        gate.set_mode(RuntimeMode.LIVE)
        gate.arm(gate.create_challenge())
        gateway = ConfirmationPendingGateway()
        gateway.positions = [{"ticker": "BTC-1", "side": "up", "quantity": "0.06"}]
        runtime = DashboardRuntime(store, StopLossQuotes(), gate, gateway)

        await runtime.poll_once(now)
        await runtime.positions_refresh_task
        await runtime._live_exits(store.load_settings(), now + timedelta(seconds=1))

        assert gate.armed is True
        assert len(gateway.orders) == 1
        assert "đang chờ Kalshi xác nhận" in runtime.live_alert
        lock = store.live_exit_lock("BTC-1", "up")
        assert str(lock["last_error"]).startswith("confirmation_pending:")


@pytest.mark.asyncio
async def test_runtime_settles_expired_lab_position_before_new_ticker_entry(tmp_path) -> None:
    now = datetime.now(UTC)
    schedule = FeeSchedule("quadratic", Decimal("1"), Decimal("0.07"), "test")
    old_market = AssetMarket(
        asset="BTC", series="KXBTC15M", ticker="BTC-OLD", title="BTC",
        open_time=now - timedelta(minutes=16), close_time=now - timedelta(seconds=1),
    )
    entry_time = now - timedelta(minutes=2)
    entry_market = old_market.model_copy(update={"close_time": now + timedelta(minutes=10)})
    entry_quote = QuotePoint(
        observed_at=entry_time, up_bid="0.19", up_ask="0.20",
        down_bid="0.79", down_ask="0.80",
    )
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        feed = LabRolloverFeed()
        runtime = DashboardRuntime(store, feed, SafetyGate(credentials_available=False))
        run_id = runtime.lab.start(["BTC"], 1, 2, 3600, {"BTC": schedule})
        store.connection.execute(
            """UPDATE lab_candidates SET side_policy='up',entry_min='0.20',entry_price='0.25',
            min_seconds=60 WHERE run_id=?""", (run_id,),
        )
        event_id = store.record_quote("BTC", old_market.ticker, entry_quote, entry_market.close_time)
        runtime.lab.on_quote(run_id, entry_market, entry_quote, event_id)
        runtime.assets["BTC"].market = old_market

        await runtime.poll_once(now)

        old_position = store.connection.execute(
            "SELECT * FROM lab_positions WHERE run_id=? ORDER BY id LIMIT 1", (run_id,)
        ).fetchone()
        assert old_position["status"] == "closed"
        assert old_position["close_reason"] == "settlement"
        assert feed.result_calls == ["BTC-OLD"]


@pytest.mark.asyncio
async def test_runtime_does_not_settle_positions_owned_by_backfill(tmp_path) -> None:
    now = datetime.now(UTC)
    schedule = FeeSchedule("quadratic", Decimal("1"), Decimal("0.07"), "test")
    old_market = AssetMarket(
        asset="BTC", series="KXBTC15M", ticker="BTC-HISTORY", title="BTC",
        open_time=now - timedelta(minutes=16), close_time=now - timedelta(seconds=1),
    )
    entry_quote = QuotePoint(
        observed_at=now - timedelta(minutes=2), up_bid="0.19", up_ask="0.20",
        down_bid="0.79", down_ask="0.80",
    )
    replay_market = old_market.model_copy(update={"close_time": now + timedelta(minutes=10)})
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        feed = LabRolloverFeed()
        runtime = DashboardRuntime(store, feed, SafetyGate(credentials_available=False))
        run_id = runtime.lab.start(["BTC"], 1, 2, 3600, {"BTC": schedule})
        store.connection.execute(
            """UPDATE lab_candidates SET side_policy='up',entry_min='0.20',
            entry_price='0.25',take_profit='0.80',stop_loss='0.15',min_seconds=60
            WHERE run_id=?""", (run_id,),
        )
        event_id = store.record_quote(
            "BTC", old_market.ticker, entry_quote, replay_market.close_time
        )
        runtime.lab.on_quote(run_id, replay_market, entry_quote, event_id)
        store.connection.execute(
            "UPDATE lab_runs SET status='backfilling' WHERE id=?", (run_id,),
        )
        runtime.assets["BTC"].market = old_market

        await runtime.poll_once(now)

        position = store.connection.execute(
            "SELECT * FROM lab_positions WHERE run_id=?", (run_id,)
        ).fetchone()

    assert position["status"] == "open"
    assert feed.result_calls == []


@pytest.mark.asyncio
async def test_run_stays_awaiting_until_every_expired_ticker_has_a_result(tmp_path) -> None:
    now = datetime.now(UTC)
    schedule = FeeSchedule("quadratic", Decimal("1"), Decimal("0.07"), "test")
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        feed = MixedSettlementFeed()
        runtime = DashboardRuntime(store, feed, SafetyGate(credentials_available=False))
        run_id = runtime.lab.start(
            ["BTC", "ETH"], 1, 2, 3600, {"BTC": schedule, "ETH": schedule}
        )
        for asset, ticker in (("BTC", "BTC-WAIT"), ("ETH", "ETH-DONE")):
            candidate_id = f"{asset}-000"
            store.connection.execute(
                "UPDATE lab_candidates SET cash='999.20',entry_count=1 WHERE run_id=? AND candidate_id=?",
                (run_id, candidate_id),
            )
            store.connection.execute(
                """INSERT INTO lab_positions(
                run_id,candidate_id,ticker,side,quantity,entry_price,entry_cost,
                entry_fee,entry_outlay,status
                ) VALUES (?,?,?,?,?,?,?,?,?,'open')""",
                (run_id, candidate_id, ticker, "up", "4", "0.20", "0.80", "0", "0.80"),
            )

        await runtime.poll_once(now)
        status = store.connection.execute(
            "SELECT settlement_status FROM lab_runs WHERE id=?", (run_id,)
        ).fetchone()[0]

    assert status == "awaiting_settlement"
    assert set(feed.result_calls) == {"BTC-WAIT", "ETH-DONE"}


@pytest.mark.asyncio
async def test_awaiting_run_resumes_when_open_ticker_is_current_after_restart(tmp_path) -> None:
    now = datetime.now(UTC)
    schedule = FeeSchedule("quadratic", Decimal("1"), Decimal("0.07"), "test")
    current = AssetMarket(
        asset="BTC", series="KXBTC15M", ticker="BTC-1", title="BTC",
        open_time=now - timedelta(minutes=1), close_time=now + timedelta(minutes=10),
    )
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        runtime = DashboardRuntime(store, CompleteQuotes(), SafetyGate(credentials_available=False))
        run_id = runtime.lab.start(["BTC"], 1, 2, 3600, {"BTC": schedule})
        store.connection.execute(
            "UPDATE lab_runs SET settlement_status='awaiting_settlement' WHERE id=?",
            (run_id,),
        )
        store.connection.execute(
            "UPDATE lab_candidates SET cash='999.20',entry_count=1 WHERE run_id=?",
            (run_id,),
        )
        store.connection.execute(
            """INSERT INTO lab_positions(
            run_id,candidate_id,ticker,side,quantity,entry_price,entry_cost,
            entry_fee,entry_outlay,status
            ) VALUES (?,?,?,?,?,?,?,?,?,'open')""",
            (run_id, "BTC-000", "BTC-1", "up", "4", "0.20", "0.80", "0", "0.80"),
        )
        runtime.assets["BTC"].market = current

        await runtime.poll_once(now)
        run = store.connection.execute(
            "SELECT settlement_status,quote_count FROM lab_runs WHERE id=?", (run_id,)
        ).fetchone()

    assert run["settlement_status"] == "ready"
    assert run["quote_count"] > 0
@pytest.mark.asyncio
async def test_runtime_resumes_interrupted_history_warmup_after_restart(tmp_path) -> None:
    store = DashboardStore(tmp_path / "state.sqlite3")
    runtime = DashboardRuntime(store, None, SafetyGate(credentials_available=False))
    run_id = runtime.lab.start(
        ["BTC"], 1, 2, 3600,
        {"BTC": FeeSchedule("quadratic", Decimal("1"), Decimal("0.07"), "test")},
        history_cycles=12, history_cutoff_event_id=0,
    )

    await runtime.start()
    for _ in range(10):
        await asyncio.sleep(0)
        status = runtime.store.connection.execute(
            "SELECT status FROM lab_runs WHERE id=?", (run_id,)
        ).fetchone()[0]
        if status == "running":
            break
    await runtime.close()
    store.close()

    assert status == "running"


@pytest.mark.asyncio
async def test_one_unavailable_historical_result_does_not_fail_the_run(tmp_path) -> None:
    class ResultUnavailable:
        async def result(self, ticker):
            raise RuntimeError("temporary result lookup failure")

        async def close(self):
            pass

    store = DashboardStore(tmp_path / "state.sqlite3")
    now = datetime.now(UTC)
    close_time = now - timedelta(minutes=15)
    item = QuotePoint(
        observed_at=close_time - timedelta(minutes=10),
        up_bid="0.19", up_ask="0.20", down_bid="0.79", down_ask="0.80",
    )
    event_id = store.record_quote("BTC", "BTC-OLD", item, close_time)
    runtime = DashboardRuntime(
        store, ResultUnavailable(), SafetyGate(credentials_available=False)
    )
    run_id = runtime.lab.start(
        ["BTC"], 1, 2, 3600,
        {"BTC": FeeSchedule("quadratic", Decimal("1"), Decimal("0.07"), "test")},
        history_cycles=1, history_cutoff_event_id=event_id,
    )

    await runtime._run_lab_backfill(run_id)
    run = runtime.store.connection.execute(
        "SELECT status,history_cycles_loaded FROM lab_runs WHERE id=?", (run_id,)
    ).fetchone()
    await runtime.close()
    store.close()

    assert run["status"] == "running"
    assert run["history_cycles_loaded"] == 0
