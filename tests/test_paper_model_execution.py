from datetime import UTC, datetime, timedelta
from decimal import Decimal

from kaishi_bot.agentic_strategy import Action, Decision
from kaishi_bot.dashboard_models import AssetMarket, QuotePoint, RuntimeMode
from kaishi_bot.dashboard_store import DashboardStore
from kaishi_bot.domain import Side
from kaishi_bot.fees import FeeSchedule
from kaishi_bot.paper import PaperBroker
from kaishi_bot.paper_model_shadow import PaperModelShadow


SCHEDULE = FeeSchedule("quadratic", Decimal("1"), Decimal("0.07"), "test")


def _market(now: datetime) -> AssetMarket:
    return AssetMarket(
        asset="BTC", series="KXBTC15M", ticker="BTC-MODEL-1",
        open_time=now - timedelta(minutes=5),
        close_time=now + timedelta(minutes=10), target="100",
    )


def _quote(now: datetime, up_bid="0.19", up_ask="0.20") -> QuotePoint:
    return QuotePoint(
        observed_at=now, up_bid=Decimal(up_bid), up_ask=Decimal(up_ask),
        down_bid=Decimal("0.79"), down_ask=Decimal("0.80"),
    )


def _decision(action: Action, *, reason="positive_settlement_edge") -> Decision:
    return Decision(
        action=action, side=Side.UP, fraction=Decimal("1"),
        probability=Decimal("0.95"), edge=Decimal("0.20"), reason=reason,
    )


def test_model_buy_and_exit_mutate_only_paper_ledger(tmp_path) -> None:
    now = datetime(2026, 8, 12, 16, tzinfo=UTC)
    with DashboardStore(tmp_path / "paper.sqlite3") as store:
        model = PaperModelShadow(None, None, store, None)  # type: ignore[arg-type]
        model.model_id = "xgb-test"

        entry = model._execute_decision(
            _decision(Action.BUY), _market(now), _quote(now), SCHEDULE
        )

        assert entry["executed"] is True
        position = store.open_positions()[0]
        assert position["strategy_owner"] == "model:xgb-test"
        assert position["entry_price"] == Decimal("0.20")
        assert store.cash() < Decimal("1000")

        exit_result = model._execute_decision(
            _decision(Action.EXIT_ALL, reason="final_take_profit"),
            _market(now), _quote(now, up_bid="0.40", up_ask="0.41"), SCHEDULE,
        )

        assert exit_result["executed"] is True
        assert store.open_positions() == []
        closed = store.closed_positions()[0]
        assert closed["close_reason"] == "take_profit"
        assert closed["realized_pnl"] > 0


def test_model_execution_is_blocked_outside_paper_mode(tmp_path) -> None:
    now = datetime(2026, 8, 12, 16, tzinfo=UTC)
    with DashboardStore(tmp_path / "paper.sqlite3") as store:
        store.save_settings(store.load_settings().model_copy(update={
            "mode": RuntimeMode.LIVE,
        }))
        model = PaperModelShadow(None, None, store, None)  # type: ignore[arg-type]
        model.model_id = "xgb-test"

        result = model._execute_decision(
            _decision(Action.BUY), _market(now), _quote(now), SCHEDULE
        )

        assert result["executed"] is False
        assert result["result"] == "not_paper_mode"
        assert store.open_positions() == []


def test_control_deck_does_not_close_model_position_while_model_is_active(tmp_path) -> None:
    now = datetime(2026, 8, 12, 16, tzinfo=UTC)
    with DashboardStore(tmp_path / "paper.sqlite3") as store:
        store.open_position(
            asset="BTC", ticker="BTC-MODEL-1", side="up",
            quantity=Decimal("1"), entry_price=Decimal("0.20"),
            opened_at=now, day="2026-08-12", strategy_owner="model:xgb-test",
        )
        settings = store.load_settings().model_copy(update={"bot_enabled": False})

        PaperBroker(store).on_quote(
            _market(now), _quote(now, up_bid="0.50", up_ask="0.51"),
            settings, now, fee_schedule=SCHEDULE,
            include_model_positions=False,
        )

        assert len(store.open_positions()) == 1
