from decimal import Decimal
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from kaishi_bot.dashboard_models import RuntimeMode
from kaishi_bot.domain import OrderRequest
from kaishi_bot.production import (
    OrderConfirmationPending,
    ProductionGateway,
    _CompatibleOrderbookSnapshotMessage,
)
from kaishi_bot.safety import SafetyError, SafetyGate


class FakePortfolio:
    def balance(self):
        return SimpleNamespace(
            balance=100000, balance_dollars=Decimal("1000.00"),
            portfolio_value=120000, updated_ts=1,
        )

    def positions_all(self, max_pages=10):
        return iter([
            SimpleNamespace(
                ticker="BTC", position=Decimal("2.00"),
                market_exposure=Decimal("1.2000"),
                realized_pnl=Decimal("0.1000"), fees_paid=Decimal("0.0300"),
                last_updated_ts=datetime(2026, 8, 6, tzinfo=UTC),
            ),
            SimpleNamespace(
                ticker="FLAT", position=Decimal("0"), market_exposure=Decimal("0"),
                realized_pnl=Decimal("0"), fees_paid=Decimal("0"),
                last_updated_ts=datetime(2026, 8, 6, tzinfo=UTC),
            ),
        ])

    def fills_all(self, max_pages=2):
        return iter([SimpleNamespace(
            fill_id="fill-1", order_id="order-1", ticker="KXBTC15M-TEST",
            side="yes", action="buy", count=Decimal("0.06"),
            yes_price=Decimal("0.7800"), no_price=Decimal("0.2200"),
            fee_cost=Decimal("0.000800"), is_taker=True,
            created_time=datetime(2026, 8, 6, 20, 42, 20, tzinfo=UTC),
        )])

    def settlements_all(self, max_pages=2):
        return iter([SimpleNamespace(
            ticker="KXBTC15M-TEST", market_result="no",
            yes_count=Decimal("0.06"), yes_total_cost=Decimal("0.046800"),
            no_count=Decimal("0"), no_total_cost=Decimal("0"),
            revenue=Decimal("0"), fee_cost=Decimal("0.000800"),
            settled_time=datetime(2026, 8, 6, 20, 45, 11, tzinfo=UTC),
        )])


class FakeOrders:
    def __init__(self):
        self.created = 0

    def list_all(self, max_pages=10):
        return iter([SimpleNamespace(
            order_id="resting-1", ticker="BTC", status="resting",
            outcome_side="yes", book_side="bid",
            yes_price=Decimal("0.5600"), no_price=Decimal("0.4400"),
            initial_count=Decimal("2.00"), fill_count=Decimal("1.00"),
            remaining_count=Decimal("1.00"), taker_fees=Decimal("0.0100"),
            maker_fees=Decimal("0.0200"),
            created_time=datetime(2026, 8, 6, tzinfo=UTC),
        )])

    def create_v2(self, *, request):
        self.created += 1
        return SimpleNamespace(
            order_id="order-1", client_order_id=request.client_order_id,
            fill_count=request.count,
        )


class FakeClient:
    def __init__(self):
        self.portfolio = FakePortfolio()
        self.orders = FakeOrders()


class FakeWebSocket:
    def connect(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def subscribe_ticker(self, *, tickers, maxsize):
        async def messages():
            yield SimpleNamespace(
                type="ticker",
                msg=SimpleNamespace(
                    market_ticker=tickers[0],
                    yes_bid=Decimal("0.61"), yes_ask=Decimal("0.62"),
                    no_bid=Decimal("0.38"), no_ask=Decimal("0.39"),
                    ts=0, ts_ms=1_786_050_000_000,
                ),
            )
            # An identical top must not create a redundant quote.
            yield SimpleNamespace(
                type="ticker",
                msg=SimpleNamespace(
                    market_ticker=tickers[0],
                    yes_bid=Decimal("0.61"), yes_ask=Decimal("0.62"),
                    no_bid=Decimal("0.38"), no_ask=Decimal("0.39"),
                    ts=0, ts_ms=1_786_050_000_100,
                ),
            )
            # A changed top is delivered immediately.
            yield SimpleNamespace(
                type="ticker",
                msg=SimpleNamespace(
                    market_ticker=tickers[0],
                    yes_bid=Decimal("0.62"), yes_ask=Decimal("0.63"),
                    no_bid=None, no_ask=None,
                    ts=0, ts_ms=1_786_050_001_000,
                ),
            )
        return messages()


@pytest.mark.asyncio
async def test_account_snapshot_is_json_safe_and_write_needs_armed_gate() -> None:
    gate = SafetyGate(credentials_available=True)
    client = FakeClient()
    gateway = ProductionGateway(client, gate)
    snapshot = await gateway.account_snapshot()
    assert snapshot["balance"] == "1000.00"
    assert snapshot["portfolio_value"] == "1200.00"
    assert len(snapshot["positions"]) == 1
    assert snapshot["positions"][0]["position"] == "2.00"
    assert snapshot["positions"][0]["exposure"] == "1.2000"
    assert snapshot["positions"][0]["realized_pnl"] == "0.1000"
    assert snapshot["positions"][0]["fees_paid"] == "0.0300"
    assert snapshot["orders"][0]["yes_price"] == "0.5600"
    assert snapshot["orders"][0]["remaining_count"] == "1.00"
    assert snapshot["orders"][0]["fees_paid"] == "0.0300"
    assert snapshot["fills"][0]["ticker"] == "KXBTC15M-TEST"
    assert snapshot["fills"][0]["count"] == "0.06"
    assert snapshot["fills"][0]["fee_cost"] == "0.000800"
    assert snapshot["settlements"][0]["market_result"] == "no"
    assert snapshot["settlements"][0]["revenue"] == "0"

    request = OrderRequest("BTC", "client-1", "bid", Decimal("1"), Decimal("0.25"), False)
    with pytest.raises(SafetyError):
        await gateway.place_guarded_order(request, Decimal("0.25"), Decimal("0"))
    assert client.orders.created == 0

    gate.set_mode(RuntimeMode.LIVE)
    gate.arm(gate.create_challenge())
    result = await gateway.place_guarded_order(request, Decimal("0.25"), Decimal("0"))
    assert result.order_id == "order-1"
    assert result.fill_count == Decimal("1")


@pytest.mark.asyncio
async def test_live_gateway_accepts_fractional_count_and_rejects_zero() -> None:
    gate = SafetyGate(credentials_available=True)
    gate.set_mode(RuntimeMode.LIVE)
    client = FakeClient()
    gateway = ProductionGateway(client, gate)

    gate.arm(gate.create_challenge())
    fractional = OrderRequest(
        "BTC", "fractional", "bid", Decimal("0.01"), Decimal("0.81"), False
    )
    result = await gateway.place_guarded_order(
        fractional, Decimal("0.01"), Decimal("0")
    )
    assert client.orders.created == 1
    assert result.fill_count == Decimal("0.01")

    zero = OrderRequest("BTC", "zero", "bid", Decimal("0"), Decimal("0.81"), False)
    with pytest.raises(SafetyError, match="at least 0.01"):
        await gateway.place_guarded_order(zero, Decimal("0"), Decimal("0"))
    assert gate.armed is False


@pytest.mark.asyncio
async def test_quote_stream_uses_latest_wins_ticker_prices() -> None:
    gateway = ProductionGateway(
        FakeClient(), SafetyGate(credentials_available=True),
        websocket_factory=FakeWebSocket,
    )

    updates = [item async for item in gateway.quote_stream(["BTC-1"])]

    ticker, quote = updates[0]
    assert ticker == "BTC-1"
    assert quote.up_bid == Decimal("0.61")
    assert quote.up_ask == Decimal("0.62")
    assert quote.down_bid == Decimal("0.38")
    assert quote.down_ask == Decimal("0.39")
    assert len(updates) == 2
    _, widened = updates[1]
    assert widened.up_bid == Decimal("0.62")
    assert widened.up_ask == Decimal("0.63")
    assert widened.down_bid == Decimal("0.37")
    assert widened.down_ask == Decimal("0.38")
    assert widened.observed_at == datetime.fromtimestamp(
        1_786_050_001_000 / 1000, tz=UTC
    )


def test_orderbook_snapshot_accepts_current_dollars_fields() -> None:
    message = _CompatibleOrderbookSnapshotMessage.model_validate({
        "type": "orderbook_snapshot", "sid": 3, "seq": 7,
        "msg": {
            "market_ticker": "BTC-1", "market_id": "market-1",
            "yes_dollars": [["0.61", "10.00"]],
            "no_dollars": [["0.38", "9.00"]],
        },
    })

    assert message.msg.yes == {Decimal("0.61"): Decimal("10.00")}
    assert message.msg.no == {Decimal("0.38"): Decimal("9.00")}


@pytest.mark.asyncio
async def test_duplicate_client_id_becomes_confirmation_pending_without_disarming() -> None:
    class DuplicateOrders(FakeOrders):
        def create_v2(self, *, request):
            raise RuntimeError("{'code': 'order_already_exists'}")

        def list_all(self, max_pages=10):
            return iter([])

    gate = SafetyGate(credentials_available=True)
    gate.set_mode(RuntimeMode.LIVE)
    gate.arm(gate.create_challenge())
    client = FakeClient()
    client.orders = DuplicateOrders()
    gateway = ProductionGateway(client, gate)
    request = OrderRequest(
        "BTC", "duplicate", "ask", Decimal("0.06"), Decimal("0.49"), True,
        "immediate_or_cancel",
    )

    with pytest.raises(OrderConfirmationPending):
        await gateway.place_guarded_order(request, Decimal("0"), Decimal("0"))

    assert gate.armed is True
