from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from kaishi_bot.dashboard_models import AssetMarket, DashboardSettings, QuotePoint
from kaishi_bot.dashboard_store import DashboardStore
from kaishi_bot.entry_guard import GuardQuote, GuardReason, evaluate_entry
from kaishi_bot.fees import FeeSchedule, taker_fee
from kaishi_bot.market_data import OrderBookSnapshot


NY = ZoneInfo("America/New_York")


class PaperBroker:
    def __init__(self, store: DashboardStore) -> None:
        self.store = store

    def on_quote(
        self, market: AssetMarket, quote: QuotePoint,
        settings: DashboardSettings, observed_at: datetime,
        *, fee_schedule: FeeSchedule | None = None,
        previous_quote: QuotePoint | None = None,
        depth: OrderBookSnapshot | None = None,
        include_model_positions: bool = True,
    ) -> list[str]:
        events: list[str] = []
        closed_sides: set[str] = set()
        for position in list(self.store.open_positions()):
            if position["ticker"] != market.ticker:
                continue
            if (
                not include_model_positions
                and str(position.get("strategy_owner", "")).startswith("model:")
            ):
                continue
            bid = quote.up_bid if position["side"] == "up" else quote.down_bid
            if bid >= settings.take_profit:
                exit_fee = (
                    taker_fee(fee_schedule, Decimal(str(position["quantity"])), bid)
                    if fee_schedule else Decimal("0")
                )
                if self.store.close_position(
                    int(position["id"]), bid, observed_at, "take_profit", exit_fee
                ):
                    events.append("take_profit")
                    closed_sides.add(str(position["side"]))
            elif bid <= settings.stop_loss:
                exit_fee = (
                    taker_fee(fee_schedule, Decimal(str(position["quantity"])), bid)
                    if fee_schedule else Decimal("0")
                )
                if self.store.close_position(
                    int(position["id"]), bid, observed_at, "stop_loss", exit_fee
                ):
                    events.append("stop_loss")
                    closed_sides.add(str(position["side"]))
        asset = settings.assets.get(market.asset)
        cycle_start = market.close_time - timedelta(minutes=15)
        elapsed = (observed_at - cycle_start).total_seconds()
        if (
            not settings.bot_enabled or asset is None or not asset.enabled
            or observed_at >= market.close_time
            or not settings.entry_start_seconds <= elapsed < settings.entry_end_seconds
        ):
            return events

        day = observed_at.astimezone(NY).date().isoformat()
        for side, enabled, ask, bid in (
            ("up", asset.trade_up, quote.up_ask, quote.up_bid),
            ("down", asset.trade_down, quote.down_ask, quote.down_bid),
        ):
            if not enabled or side in closed_sides:
                continue
            if self.store.has_market_side_lock(market.ticker, side):
                continue
            remaining = settings.paper_daily_cap - self.store.daily_spend(day)
            budget = min(settings.entry_amount, Decimal("200.00"), remaining, self.store.cash())
            entry_fee = Decimal("0")
            liquidity_status = "liquidity_unverified"
            if fee_schedule is not None:
                previous = None
                if previous_quote is not None:
                    previous = GuardQuote(
                        bid=previous_quote.up_bid if side == "up" else previous_quote.down_bid,
                        ask=previous_quote.up_ask if side == "up" else previous_quote.down_ask,
                        observed_at=previous_quote.observed_at,
                    )
                decision = evaluate_entry(
                    previous=previous,
                    current=GuardQuote(bid=bid, ask=ask, observed_at=quote.observed_at),
                    entry_min=settings.entry_min,
                    entry_price=settings.entry_price,
                    stop_loss=settings.stop_loss,
                    take_profit=settings.take_profit,
                    budget=budget,
                    settings=settings.entry_guard,
                    fee_schedule=fee_schedule,
                    cooldown_until=None,
                )
                if not decision.eligible:
                    self.store.increment_guard_counter(
                        "paper", "account", decision.reason, observed_at
                    )
                    continue
                quantity = decision.quantity
                entry_fee = decision.entry_fee
                if depth is not None:
                    if depth.available(side, ask) < quantity:
                        self.store.increment_guard_counter(
                            "paper", "account", GuardReason.INSUFFICIENT_DEPTH,
                            observed_at,
                        )
                        continue
                    liquidity_status = "verified"
            else:
                if ask < settings.entry_min or ask > settings.entry_price:
                    continue
                quantity = Decimal(int(budget / ask)) if ask > 0 else Decimal(0)
            if quantity <= 0:
                continue
            position_id = self.store.open_position(
                asset=market.asset, ticker=market.ticker, side=side,
                quantity=quantity, entry_price=ask, opened_at=observed_at, day=day,
                entry_fee=entry_fee, liquidity_status=liquidity_status,
            )
            if position_id is not None:
                events.append("entry")
        return events

    def close_position(
        self, position_id: int, quote: QuotePoint, observed_at: datetime
    ) -> bool:
        position = next(
            (item for item in self.store.open_positions() if item["id"] == position_id),
            None,
        )
        if position is None:
            return False
        bid = quote.up_bid if position["side"] == "up" else quote.down_bid
        closed = self.store.close_position(position_id, bid, observed_at, "manual")
        return closed

    def settle_ticker(self, ticker: str, winning_side: str, observed_at: datetime) -> int:
        settled = 0
        for position in list(self.store.open_positions()):
            if position["ticker"] != ticker:
                continue
            payout = Decimal("1") if position["side"] == winning_side else Decimal("0")
            if self.store.close_position(int(position["id"]), payout, observed_at, "settlement"):
                settled += 1
        return settled

    def snapshot(self, quotes: dict[str, QuotePoint]) -> dict[str, object]:
        cash = self.store.cash()
        liquidation = Decimal("0")
        unrealized = Decimal("0")
        positions = self.store.open_positions()
        for position in positions:
            quote = quotes.get(str(position["ticker"]))
            if quote is None:
                mark = position["entry_price"]
            else:
                mark = quote.up_bid if position["side"] == "up" else quote.down_bid
            value = Decimal(str(position["quantity"])) * Decimal(str(mark))
            liquidation += value
            unrealized += value - Decimal(str(position["entry_cost"]))
            position["mark"] = mark
            position["unrealized_pnl"] = value - Decimal(str(position["entry_cost"]))
        closed_positions = self.store.closed_positions()
        realized = sum(
            (Decimal(str(item["realized_pnl"])) for item in closed_positions),
            Decimal("0"),
        )
        return {
            "cash": cash,
            "equity": cash + liquidation,
            "realized_pnl": realized,
            "unrealized_pnl": unrealized,
            "positions": positions,
            "closed_positions": closed_positions[-100:],
        }
