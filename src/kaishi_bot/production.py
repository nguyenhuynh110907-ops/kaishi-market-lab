from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from collections.abc import AsyncIterator
from typing import Annotated, Any, Literal

from kalshi import CreateOrderV2Request, KalshiAuth, KalshiClient, KalshiConfig
from kalshi.ws import KalshiWebSocket
from pydantic import AliasChoices, BaseModel, BeforeValidator, Field

from kaishi_bot.config import Credentials
from kaishi_bot.dashboard_models import QuotePoint
from kaishi_bot.domain import OrderRequest, OrderResult
from kaishi_bot.safety import SafetyError, SafetyGate


class OrderConfirmationPending(RuntimeError):
    """Kalshi accepted a client id but has not exposed the order yet."""


def _orderbook_levels(value: Any) -> dict[Decimal, Decimal]:
    """Accept every snapshot field shape used by Kalshi and its SDKs."""
    if value is None:
        return {}
    rows = value.items() if isinstance(value, dict) else value
    return {
        Decimal(str(price)): Decimal(str(quantity))
        for price, quantity in rows
    }


_OrderbookLevels = Annotated[
    dict[Decimal, Decimal], BeforeValidator(_orderbook_levels),
]


class _CompatibleOrderbookSnapshotPayload(BaseModel):
    market_ticker: str
    market_id: str
    yes: _OrderbookLevels = Field(
        validation_alias=AliasChoices("yes_dollars_fp", "yes_dollars", "yes"),
    )
    no: _OrderbookLevels = Field(
        validation_alias=AliasChoices("no_dollars_fp", "no_dollars", "no"),
    )
    model_config = {"extra": "allow", "populate_by_name": True}


class _CompatibleOrderbookSnapshotMessage(BaseModel):
    type: Literal["orderbook_snapshot"] = "orderbook_snapshot"
    sid: int
    seq: int
    msg: _CompatibleOrderbookSnapshotPayload
    model_config = {"extra": "allow", "populate_by_name": True}


def _install_orderbook_snapshot_compatibility() -> None:
    """Teach kalshi-sdk 7.4 to accept the current production wire format."""
    import kalshi.ws.client as ws_client
    import kalshi.ws.dispatch as ws_dispatch

    ws_client.OrderbookSnapshotMessage = _CompatibleOrderbookSnapshotMessage
    ws_dispatch.OrderbookSnapshotMessage = _CompatibleOrderbookSnapshotMessage
    ws_dispatch.MESSAGE_MODELS["orderbook_snapshot"] = (
        _CompatibleOrderbookSnapshotMessage
    )


def _is_duplicate_order_error(error: Exception) -> bool:
    code = str(getattr(error, "code", "")).lower()
    message = str(error).lower()
    return "order_already_exists" in code or "order_already_exists" in message


def _dollars_from_cents(value: object) -> str:
    return f"{Decimal(str(value)) / Decimal('100'):.2f}"


def _decimal_string(value: object | None, default: str = "0") -> str:
    return str(Decimal(str(value))) if value is not None else default


class ProductionGateway:
    def __init__(self, client: Any, gate: SafetyGate, websocket_factory: Any = None) -> None:
        self.client = client
        self.gate = gate
        self.websocket_factory = websocket_factory

    @classmethod
    def create(cls, credentials: Credentials, gate: SafetyGate) -> "ProductionGateway":
        _install_orderbook_snapshot_compatibility()
        config = KalshiConfig.production()
        client = KalshiClient(
            key_id=credentials.key_id,
            private_key_path=credentials.private_key_path,
            config=config,
        )
        auth = KalshiAuth.from_key_path(
            credentials.key_id, credentials.private_key_path
        )
        return cls(
            client, gate,
            websocket_factory=lambda: KalshiWebSocket(auth=auth, config=config),
        )

    async def quote_stream(
        self, tickers: list[str],
    ) -> AsyncIterator[tuple[str, QuotePoint]]:
        if self.websocket_factory is None:
            raise RuntimeError("Kalshi WebSocket is unavailable")
        websocket = self.websocket_factory()
        async with websocket.connect() as session:
            # Ticker is a latest-wins feed and the SDK safely drops superseded
            # messages. It avoids reconstructing a 500+ msg/s stateful book
            # merely to obtain four executable top-of-book prices.
            stream = await session.subscribe_ticker(tickers=tickers, maxsize=1000)
            last_tops: dict[str, tuple[Decimal, Decimal, Decimal, Decimal]] = {}
            async for message in stream:
                payload = message.msg
                ticker = str(payload.market_ticker)
                if ticker not in tickers:
                    continue
                yes_bid = Decimal(str(payload.yes_bid))
                yes_ask = Decimal(str(payload.yes_ask))
                raw_no_bid = getattr(payload, "no_bid", None)
                raw_no_ask = getattr(payload, "no_ask", None)
                no_bid = (
                    Decimal(str(raw_no_bid)) if raw_no_bid is not None
                    else Decimal("1") - yes_ask
                )
                no_ask = (
                    Decimal(str(raw_no_ask)) if raw_no_ask is not None
                    else Decimal("1") - yes_bid
                )
                top = (yes_bid, yes_ask, no_bid, no_ask)
                if last_tops.get(ticker) == top:
                    continue
                last_tops[ticker] = top

                timestamp_ms = int(getattr(payload, "ts_ms", 0) or 0)
                timestamp = getattr(payload, "ts", None)
                if timestamp_ms > 0:
                    observed_at = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)
                elif isinstance(timestamp, datetime):
                    observed_at = timestamp.astimezone(UTC)
                else:
                    observed_at = datetime.now(UTC)
                yield ticker, QuotePoint(
                    observed_at=observed_at,
                    up_bid=yes_bid,
                    up_ask=yes_ask,
                    down_bid=no_bid,
                    down_ask=no_ask,
                )

    async def account_snapshot(self) -> dict[str, object]:
        def read() -> dict[str, object]:
            balance = self.client.portfolio.balance()
            positions = []
            for item in self.client.portfolio.positions_all(max_pages=10):
                signed = Decimal(str(getattr(item, "position", None) or "0"))
                if signed == 0:
                    continue
                positions.append({
                    "ticker": str(item.ticker),
                    "position": str(signed),
                    "side": "up" if signed > 0 else "down",
                    "quantity": str(abs(signed)),
                    "exposure": _decimal_string(getattr(item, "market_exposure", None)),
                    "realized_pnl": _decimal_string(getattr(item, "realized_pnl", None)),
                    "fees_paid": _decimal_string(getattr(item, "fees_paid", None)),
                    "updated_at": (
                        item.last_updated_ts.isoformat()
                        if getattr(item, "last_updated_ts", None) else None
                    ),
                })
            orders = []
            for item in self.client.orders.list_all(max_pages=10):
                orders.append({
                    "order_id": str(item.order_id),
                    "client_order_id": str(
                        getattr(item, "client_order_id", "") or ""
                    ),
                    "ticker": str(getattr(item, "ticker", "")),
                    "status": str(getattr(item, "status", "")),
                    "outcome_side": str(getattr(item, "outcome_side", "")),
                    "book_side": str(getattr(item, "book_side", "")),
                    "yes_price": _decimal_string(getattr(item, "yes_price", None)),
                    "no_price": _decimal_string(getattr(item, "no_price", None)),
                    "initial_count": _decimal_string(getattr(item, "initial_count", None)),
                    "fill_count": _decimal_string(getattr(item, "fill_count", None)),
                    "remaining_count": _decimal_string(getattr(item, "remaining_count", None)),
                    "fees_paid": _decimal_string(
                        Decimal(str(getattr(item, "taker_fees", None) or "0"))
                        + Decimal(str(getattr(item, "maker_fees", None) or "0"))
                    ),
                    "created_at": (
                        item.created_time.isoformat()
                        if getattr(item, "created_time", None) else None
                    ),
                })
            fills = []
            for item in self.client.portfolio.fills_all(max_pages=2):
                fills.append({
                    "fill_id": str(item.fill_id),
                    "order_id": str(item.order_id),
                    "ticker": str(item.ticker),
                    "side": str(item.side),
                    "action": str(item.action),
                    "count": _decimal_string(item.count),
                    "yes_price": _decimal_string(item.yes_price),
                    "no_price": _decimal_string(item.no_price),
                    "fee_cost": _decimal_string(item.fee_cost),
                    "is_taker": bool(item.is_taker),
                    "created_at": (
                        item.created_time.isoformat()
                        if getattr(item, "created_time", None) else None
                    ),
                })
            settlements = []
            for item in self.client.portfolio.settlements_all(max_pages=2):
                settlements.append({
                    "ticker": str(item.ticker),
                    "market_result": str(item.market_result),
                    "yes_count": _decimal_string(item.yes_count),
                    "yes_total_cost": _decimal_string(item.yes_total_cost),
                    "no_count": _decimal_string(item.no_count),
                    "no_total_cost": _decimal_string(item.no_total_cost),
                    "revenue": _decimal_string(item.revenue),
                    "fee_cost": _decimal_string(item.fee_cost),
                    "settled_at": (
                        item.settled_time.isoformat()
                        if getattr(item, "settled_time", None) else None
                    ),
                })
            balance_dollars = getattr(balance, "balance_dollars", None)
            portfolio_value_dollars = getattr(balance, "portfolio_value_dollars", None)
            return {
                "balance": (
                    _decimal_string(balance_dollars)
                    if balance_dollars is not None
                    else _dollars_from_cents(balance.balance)
                ),
                "portfolio_value": (
                    _decimal_string(portfolio_value_dollars)
                    if portfolio_value_dollars is not None
                    else _dollars_from_cents(balance.portfolio_value)
                ),
                "updated_at": datetime.fromtimestamp(
                    int(balance.updated_ts), tz=UTC
                ).isoformat(),
                "positions": positions,
                "orders": orders,
                "fills": fills,
                "settlements": settlements,
            }
        return await asyncio.to_thread(read)

    async def positions_snapshot(self) -> list[dict[str, object]]:
        """Fetch only open positions for the latency-sensitive protection loop."""
        def read() -> list[dict[str, object]]:
            positions: list[dict[str, object]] = []
            for item in self.client.portfolio.positions_all(max_pages=2):
                signed = Decimal(str(getattr(item, "position", None) or "0"))
                if signed == 0:
                    continue
                positions.append({
                    "ticker": str(item.ticker),
                    "position": str(signed),
                    "side": "up" if signed > 0 else "down",
                    "quantity": str(abs(signed)),
                    "exposure": _decimal_string(getattr(item, "market_exposure", None)),
                    "realized_pnl": _decimal_string(getattr(item, "realized_pnl", None)),
                    "fees_paid": _decimal_string(getattr(item, "fees_paid", None)),
                    "updated_at": (
                        item.last_updated_ts.isoformat()
                        if getattr(item, "last_updated_ts", None) else None
                    ),
                })
            return positions
        return await asyncio.to_thread(read)

    async def find_order_by_client_id(self, client_order_id: str) -> OrderResult | None:
        def find() -> OrderResult | None:
            for order in self.client.orders.list_all(max_pages=2):
                if str(getattr(order, "client_order_id", "")) == client_order_id:
                    return OrderResult(
                        str(order.order_id), client_order_id,
                        Decimal(str(getattr(order, "fill_count", "0"))),
                    )
            return None
        return await asyncio.to_thread(find)

    async def cancel_order(self, order_id: str) -> None:
        await asyncio.to_thread(self.client.orders.cancel_v2, order_id)

    async def place_guarded_order(
        self, request: OrderRequest, entry_cost: Decimal, daily_used: Decimal
    ) -> OrderResult:
        if request.count <= 0 or request.count.as_tuple().exponent < -2:
            self.gate.disarm()
            raise SafetyError("Live quantity must be at least 0.01 contracts")
        self.gate.authorize_write(
            entry_cost, daily_used, reduce_only=request.reduce_only
        )
        sdk_request = CreateOrderV2Request(
            ticker=request.ticker,
            client_order_id=request.client_order_id,
            side=request.book_side,
            count=request.count,
            price=request.yes_price,
            time_in_force=request.time_in_force,
            self_trade_prevention_type="taker_at_cross",
            cancel_order_on_pause=True,
            reduce_only=request.reduce_only,
        )
        try:
            response = await asyncio.to_thread(self.client.orders.create_v2, request=sdk_request)
            return OrderResult(
                str(response.order_id),
                str(response.client_order_id or request.client_order_id),
                Decimal(str(response.fill_count)),
            )
        except Exception as error:
            # A create request may be accepted before the order becomes visible
            # through the list endpoint.  Reusing its client id then produces
            # order_already_exists even though the protective IOC was valid.
            # Give Kalshi's read model a short window to catch up.
            for delay in (0, 0.05, 0.10, 0.20):
                if delay:
                    await asyncio.sleep(delay)
                recovered = await self.find_order_by_client_id(request.client_order_id)
                if recovered is not None:
                    return recovered
            if _is_duplicate_order_error(error):
                raise OrderConfirmationPending(request.client_order_id) from error
            raise

    async def close(self) -> None:
        close = getattr(self.client, "close", None)
        if close:
            await asyncio.to_thread(close)
