from datetime import UTC, datetime, timedelta
from decimal import Decimal

from kaishi_bot.domain import Side
from kaishi_bot.fees import FeeSchedule
from kaishi_bot.research_replay import (
    ExecutableBook,
    ExecutionLevel,
    ExecutionSimulator,
    LatencyProfile,
    PointInTimeEventStream,
    ReplayAction,
    ReplayEvent,
    ReplayPortfolio,
    ReplayPosition,
    TrajectoryLedger,
    action_mask,
)


NOW = datetime(2026, 8, 11, 12, tzinfo=UTC)
FEE = FeeSchedule("quadratic", Decimal("1"), Decimal("0.07"), "f")


def book(at, up_asks=(), up_bids=()):
    return ExecutableBook("BTC-1", at, tuple(up_asks), tuple(up_bids))


def test_events_are_visible_only_by_available_at_with_stable_order() -> None:
    stream = PointInTimeEventStream([
        ReplayEvent(NOW + timedelta(seconds=1), 1, 2, "b", "future", None),
        ReplayEvent(NOW, 2, 1, "z", "second", None),
        ReplayEvent(NOW, 1, 9, "a", "first", None),
    ])
    assert [item.kind for item in stream.visible_until(NOW)] == ["first", "second"]
    assert stream.visible_until(NOW) == ()


def test_buy_uses_arrival_ask_and_consumes_depth_with_partial_fill() -> None:
    simulator = ExecutionSimulator([
        book(NOW, [ExecutionLevel(Decimal("0.70"), Decimal("1"))]),
        book(NOW + timedelta(milliseconds=500), [
            ExecutionLevel(Decimal("0.72"), Decimal("2")),
            ExecutionLevel(Decimal("0.75"), Decimal("1")),
        ]),
    ])
    fill = simulator.execute(
        ticker="BTC-1", side=Side.UP, operation="buy", quantity=Decimal("4"),
        decision_at=NOW, latency=LatencyProfile(network_ms=500), fee_schedule=FEE,
    )
    assert fill.arrival_at == NOW + timedelta(milliseconds=500)
    assert fill.filled_quantity == Decimal("3")
    assert fill.unfilled_quantity == Decimal("1")
    assert fill.vwap == Decimal("0.73")
    assert fill.reason == "partial_fill"


def test_sell_uses_bid_and_no_fill_when_liquidity_disappears() -> None:
    simulator = ExecutionSimulator([
        book(NOW, up_bids=[ExecutionLevel(Decimal("0.80"), Decimal("2"))]),
        book(NOW + timedelta(seconds=1)),
    ])
    early = simulator.execute(
        ticker="BTC-1", side=Side.UP, operation="sell", quantity=Decimal("1"),
        decision_at=NOW, latency=LatencyProfile(), fee_schedule=FEE,
    )
    late = simulator.execute(
        ticker="BTC-1", side=Side.UP, operation="sell", quantity=Decimal("1"),
        decision_at=NOW, latency=LatencyProfile(exchange_ms=1000), fee_schedule=FEE,
    )
    assert early.vwap == Decimal("0.80")
    assert late.reason == "no_liquidity"


def test_settlement_is_idempotent_and_has_no_exit_fee() -> None:
    portfolio = ReplayPortfolio(
        Decimal("9"), ReplayPosition(Side.UP, Decimal("1"), Decimal("0.70"))
    )
    assert portfolio.settle(
        episode_id="e", ticker="BTC-1", settlement_version="s1",
        winning_side=Side.UP,
    ) is True
    assert portfolio.cash == Decimal("10")
    assert portfolio.fees == 0
    assert portfolio.settle(
        episode_id="e", ticker="BTC-1", settlement_version="s1",
        winning_side=Side.UP,
    ) is False
    assert portfolio.cash == Decimal("10")


def test_action_masks_cover_flat_position_and_quality_states() -> None:
    flat = action_mask(
        position=None, data_complete=True, stale=False,
        market_open=True, entry_allowed=True,
    )
    assert {ReplayAction.BUY_UP, ReplayAction.BUY_DOWN} <= flat
    blocked = action_mask(
        position=None, data_complete=True, stale=True,
        market_open=True, entry_allowed=True,
    )
    assert blocked == {ReplayAction.WAIT}
    held = action_mask(
        position=ReplayPosition(Side.UP, Decimal("1"), Decimal("0.7")),
        data_complete=True, stale=False, market_open=True, entry_allowed=True,
        add_count=1, max_add_count=1,
    )
    assert ReplayAction.EXIT_ALL in held and ReplayAction.ADD not in held


def test_same_seed_and_trajectory_have_same_hash() -> None:
    left = TrajectoryLedger(42)
    right = TrajectoryLedger(42)
    left.append(action="buy_up", price=Decimal("0.70"))
    right.append(action="buy_up", price=Decimal("0.70"))
    assert left.digest() == right.digest()
