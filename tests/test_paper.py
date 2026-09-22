from datetime import UTC, datetime, timedelta
from decimal import Decimal

from kaishi_bot.dashboard_models import AssetMarket, AssetSettings, DashboardSettings, QuotePoint
from kaishi_bot.dashboard_store import DashboardStore
from kaishi_bot.paper import PaperBroker
from kaishi_bot.fees import FeeSchedule


SCHEDULE = FeeSchedule("quadratic", Decimal("1"), Decimal("0.07"), "test")


def market(now: datetime, ticker: str = "BTC-1") -> AssetMarket:
    return AssetMarket(
        asset="BTC", series="KXBTC15M", ticker=ticker, title="BTC",
        open_time=now - timedelta(minutes=1), close_time=now + timedelta(minutes=10),
    )


def quote(now: datetime, up_ask: str, up_bid: str | None = None) -> QuotePoint:
    ask = Decimal(up_ask)
    bid = Decimal(up_bid) if up_bid else ask - Decimal("0.01")
    return QuotePoint(
        observed_at=now, up_bid=bid, up_ask=ask,
        down_bid=Decimal("1") - ask,
        down_ask=Decimal("1") - bid,
    )


def settings(**changes) -> DashboardSettings:
    base = DashboardSettings(bot_enabled=True).model_dump()
    base.update(changes)
    base["assets"]["BTC"] = AssetSettings(
        series="KXBTC15M", trade_down=False
    )
    return DashboardSettings.model_validate(base)


def test_entry_sizes_whole_contracts_and_marks_equity(tmp_path) -> None:
    now = datetime(2026, 8, 3, 16, 0, tzinfo=UTC)
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        broker = PaperBroker(store)
        events = broker.on_quote(market(now), quote(now, "0.24", "0.23"), settings(), now)

        assert events == ["entry"]
        position = store.open_positions()[0]
        assert position["quantity"] == Decimal("41")
        assert position["entry_cost"] == Decimal("9.84")
        assert store.cash() == Decimal("990.16")
        assert broker.snapshot({"BTC-1": quote(now, "0.24", "0.23")})["equity"] == Decimal("999.59")


def test_paper_can_enter_during_the_final_second(tmp_path) -> None:
    now = datetime(2026, 8, 3, 16, 0, tzinfo=UTC)
    closing_market = market(now).model_copy(
        update={"close_time": now + timedelta(seconds=1)}
    )
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        broker = PaperBroker(store)
        events = broker.on_quote(
            closing_market, quote(now, "0.24", "0.23"), settings(), now
        )

    assert events == ["entry"]


def test_paper_does_not_enter_after_market_close(tmp_path) -> None:
    now = datetime(2026, 8, 3, 16, 0, tzinfo=UTC)
    closed_market = market(now).model_copy(
        update={"close_time": now - timedelta(milliseconds=1)}
    )
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        events = PaperBroker(store).on_quote(
            closed_market, quote(now, "0.24", "0.23"), settings(), now
        )

    assert events == []


def test_paper_only_opens_inside_the_selected_entry_window(tmp_path) -> None:
    cycle_start = datetime(2026, 8, 3, 16, 0, tzinfo=UTC)
    cfg = settings(entry_start_seconds=300, entry_end_seconds=600)
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        broker = PaperBroker(store)
        assert broker.on_quote(
            market(cycle_start).model_copy(update={
                "open_time": cycle_start,
                "close_time": cycle_start + timedelta(minutes=15),
            }),
            quote(cycle_start + timedelta(seconds=299), "0.24", "0.23"),
            cfg,
            cycle_start + timedelta(seconds=299),
        ) == []
        assert broker.on_quote(
            market(cycle_start).model_copy(update={
                "open_time": cycle_start,
                "close_time": cycle_start + timedelta(minutes=15),
            }),
            quote(cycle_start + timedelta(seconds=300), "0.24", "0.23"),
            cfg,
            cycle_start + timedelta(seconds=300),
        ) == ["entry"]


def test_old_market_position_is_not_marked_with_next_market_quote(tmp_path) -> None:
    now = datetime(2026, 8, 3, 16, 0, tzinfo=UTC)
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        broker = PaperBroker(store)
        broker.on_quote(market(now), quote(now, "0.20", "0.19"), settings(), now)
        snapshot = broker.snapshot({"BTC-2": quote(now, "0.90", "0.89")})
    assert snapshot["equity"] == Decimal("1000.00")
    assert snapshot["positions"][0]["mark"] == Decimal("0.20")


def test_take_profit_and_stop_loss_close_at_executable_bid(tmp_path) -> None:
    now = datetime(2026, 8, 3, 16, 0, tzinfo=UTC)
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        broker = PaperBroker(store)
        broker.on_quote(market(now), quote(now, "0.20", "0.19"), settings(), now)
        assert broker.on_quote(market(now), quote(now, "0.42", "0.40"), settings(), now)[0] == "take_profit"
        assert store.closed_positions()[0]["exit_price"] == Decimal("0.40")

        broker.on_quote(market(now, "BTC-2"), quote(now, "0.20", "0.19"), settings(), now)
        assert broker.on_quote(market(now, "BTC-2"), quote(now, "0.16", "0.15"), settings(), now)[0] == "stop_loss"


def test_paper_daily_cap_and_market_side_lock_block_reentry(tmp_path) -> None:
    now = datetime(2026, 8, 3, 16, 0, tzinfo=UTC)
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        broker = PaperBroker(store)
        cfg = settings(entry_amount="10", paper_daily_cap="10")
        assert broker.on_quote(market(now), quote(now, "0.25"), cfg, now) == ["entry"]
        broker.on_quote(market(now), quote(now, "0.40", "0.40"), cfg, now)
        assert broker.on_quote(market(now), quote(now, "0.20"), cfg, now) == []
        assert broker.on_quote(market(now, "BTC-2"), quote(now, "0.20"), cfg, now) == []


def test_paper_can_continue_after_fifty_dollars_of_daily_spend(tmp_path) -> None:
    now = datetime(2026, 8, 3, 16, 0, tzinfo=UTC)
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        with store.connection:
            store.connection.execute(
                "INSERT INTO paper_daily_spend(day,amount) VALUES (?,?)",
                ("2026-08-03", "50.00"),
            )
        broker = PaperBroker(store)
        cfg = settings(entry_amount="1", paper_daily_cap="1000", daily_cap="50")

        assert broker.on_quote(market(now), quote(now, "0.20"), cfg, now) == ["entry"]
        assert store.daily_spend("2026-08-03") == Decimal("51.00")


def test_expired_market_settles_open_position_to_authoritative_result(tmp_path) -> None:
    now = datetime(2026, 8, 3, 16, 0, tzinfo=UTC)
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        broker = PaperBroker(store)
        broker.on_quote(market(now), quote(now, "0.20", "0.19"), settings(), now)
        assert broker.settle_ticker("BTC-1", "up", now + timedelta(minutes=15)) == 1
        closed = store.closed_positions()[0]
    assert closed["exit_price"] == Decimal("1")
    assert closed["realized_pnl"] == Decimal("40.00")


def test_guard_enters_on_one_tick_and_rejects_below_floor(tmp_path) -> None:
    now = datetime(2026, 8, 3, 16, 0, tzinfo=UTC)
    first = quote(now, "0.20", "0.19")
    second = quote(now + timedelta(seconds=1), "0.20", "0.19")
    jumped = quote(now + timedelta(seconds=2), "0.14", "0.13")
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        broker = PaperBroker(store)
        assert broker.on_quote(
            market(now), first, settings(), now,
            fee_schedule=SCHEDULE, previous_quote=None,
        ) == ["entry"]
        assert broker.on_quote(
            market(now), second, settings(), second.observed_at,
            fee_schedule=SCHEDULE, previous_quote=first,
        ) == []
        broker.close_position(1, quote(now, "0.20", "0.19"), now + timedelta(seconds=3))
        assert broker.on_quote(
            market(now), jumped, settings(), jumped.observed_at,
            fee_schedule=SCHEDULE, previous_quote=second,
        ) == []
        assert store.guard_counters("paper", "account")["below_floor"] == 1


def test_guarded_paper_reenters_on_first_fresh_valid_quote_without_cooldown(tmp_path) -> None:
    now = datetime(2026, 8, 3, 16, 0, tzinfo=UTC)
    first = quote(now, "0.20", "0.19")
    second = quote(now + timedelta(seconds=1), "0.20", "0.19")
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        broker = PaperBroker(store)
        broker.on_quote(market(now), first, settings(), now, fee_schedule=SCHEDULE)
        broker.on_quote(
            market(now), second, settings(), second.observed_at,
            fee_schedule=SCHEDULE, previous_quote=first,
        )
        close_quote = quote(now + timedelta(seconds=2), "0.42", "0.40")
        assert broker.on_quote(
            market(now), close_quote, settings(), close_quote.observed_at,
            fee_schedule=SCHEDULE, previous_quote=second,
        ) == ["take_profit"]
        during = quote(now + timedelta(seconds=5), "0.20", "0.19")
        assert broker.on_quote(
            market(now), during, settings(), during.observed_at,
            fee_schedule=SCHEDULE, previous_quote=close_quote,
        ) == ["entry"]
        after_one = quote(now + timedelta(seconds=13), "0.20", "0.19")
        after_two = quote(now + timedelta(seconds=14), "0.20", "0.19")
        assert broker.on_quote(
            market(now), after_one, settings(), after_one.observed_at,
            fee_schedule=SCHEDULE, previous_quote=during,
        ) == []
        assert broker.on_quote(
            market(now), after_two, settings(), after_two.observed_at,
            fee_schedule=SCHEDULE, previous_quote=after_one,
        ) == []
        assert len(store.open_positions()) == 1
