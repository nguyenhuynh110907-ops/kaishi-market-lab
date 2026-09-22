from datetime import UTC, datetime, timedelta
from decimal import Decimal
import asyncio
import pytest

from kaishi_bot.dashboard_models import AssetMarket, QuotePoint
from kaishi_bot.dashboard_store import DashboardStore
from kaishi_bot.fees import FeeSchedule
from kaishi_bot.lab import StrategyLab, generate_candidates
from kaishi_bot.entry_guard import EntryGuardSettings


SCHEDULE = FeeSchedule("quadratic", Decimal("1"), Decimal("0.07"), "2026-02-05")


def fee_schedules(*assets: str) -> dict[str, FeeSchedule]:
    return {asset: SCHEDULE for asset in assets}


def market(now: datetime) -> AssetMarket:
    return AssetMarket(
        asset="BTC", series="KXBTC15M", ticker="BTC-1", title="BTC",
        open_time=now - timedelta(minutes=1), close_time=now + timedelta(minutes=10),
    )


def quote(now: datetime, up_ask: str) -> QuotePoint:
    ask = Decimal(up_ask)
    return QuotePoint(
        observed_at=now, up_bid=ask - Decimal("0.01"), up_ask=ask,
        down_bid=Decimal("1") - ask, down_ask=Decimal("1.01") - ask,
    )


def test_candidate_generation_is_reproducible_and_within_ranges() -> None:
    first = generate_candidates("BTC", 100, 42)
    second = generate_candidates("BTC", 100, 42)

    assert first == second
    assert len(first) == 100
    assert len({item.candidate_id for item in first}) == 100
    assert any(item.entry > Decimal("0.50") for item in first)
    for item in first:
        assert Decimal("0.10") <= item.entry <= Decimal("0.85")
        assert item.stop_loss < item.entry_min <= item.entry
        assert Decimal("0.03") <= item.entry - item.entry_min <= Decimal("0.20")
        assert item.entry_min - item.stop_loss >= Decimal("0.05")
        assert item.entry + Decimal("0.05") <= item.take_profit <= Decimal("0.95")
        assert Decimal("0.02") <= item.stop_loss
        assert item.min_seconds_before_close == 0
        assert item.side_policy in {"up", "down", "both"}
    assert [item.min_seconds_before_close for item in generate_candidates("BTC", 3, 42)] == [0, 0, 0]


def test_lab_candidate_uses_explicit_entry_band(tmp_path) -> None:
    now = datetime.now(UTC)
    market_data = market(now)
    below = quote(now, "0.19")
    at_min = quote(now + timedelta(seconds=1), "0.20")
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        lab = StrategyLab(store)
        run_id = lab.start(["BTC"], 1, 2, 3600, fee_schedules("BTC"))
        store.connection.execute(
            """UPDATE lab_candidates SET side_policy='up',entry_min='0.20',
            entry_price='0.25',take_profit='0.40',stop_loss='0.15',min_seconds=60
            WHERE run_id=?""", (run_id,),
        )
        for item in (below, at_min):
            event_id = store.record_quote(
                "BTC", market_data.ticker, item, market_data.close_time
            )
            lab.on_quote(run_id, market_data, item, event_id)
        position = store.connection.execute(
            "SELECT * FROM lab_positions WHERE run_id=?", (run_id,)
        ).fetchone()
        board = lab.leaderboard(run_id)["candidates"][0]

    assert position["entry_event_id"] == 2
    assert board["entry_min"] == "0.20"


def test_eligible_cycles_count_once_per_ticker_before_guard_checks(tmp_path) -> None:
    now = datetime.now(UTC)
    first_market = market(now)
    second_market = AssetMarket(
        asset="BTC", series="KXBTC15M", ticker="BTC-2", title="BTC",
        open_time=now, close_time=now + timedelta(minutes=15),
    )
    guard = EntryGuardSettings(
        confirmation_ticks=2, minimum_reward_risk=Decimal("0.01")
    )
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        lab = StrategyLab(store)
        run_id = lab.start(
            ["BTC"], 1, 2, 3600, fee_schedules("BTC"), guard_settings=guard
        )
        store.connection.execute(
            """UPDATE lab_candidates SET side_policy='up',entry_min='0.20',
            entry_price='0.25',take_profit='0.40',stop_loss='0.15',min_seconds=60
            WHERE run_id=?""", (run_id,),
        )

        first = quote(now, "0.22")
        first_id = store.record_quote("BTC", first_market.ticker, first, first_market.close_time)
        lab.on_quote(run_id, first_market, first, first_id)
        assert store.connection.execute(
            "SELECT COUNT(*) FROM lab_positions WHERE run_id=?", (run_id,)
        ).fetchone()[0] == 1
        assert store.connection.execute(
            "SELECT COUNT(*) FROM lab_candidate_eligible_cycles WHERE run_id=?",
            (run_id,),
        ).fetchone()[0] == 1

        confirmed = quote(now + timedelta(seconds=1), "0.22")
        confirmed_id = store.record_quote(
            "BTC", first_market.ticker, confirmed, first_market.close_time
        )
        lab.on_quote(run_id, first_market, confirmed, confirmed_id)
        close = quote(now + timedelta(seconds=2), "0.50")
        close_id = store.record_quote("BTC", first_market.ticker, close, first_market.close_time)
        lab.on_quote(run_id, first_market, close, close_id)

        next_cycle = quote(now + timedelta(seconds=3), "0.22")
        next_id = store.record_quote(
            "BTC", second_market.ticker, next_cycle, second_market.close_time
        )
        lab.on_quote(run_id, second_market, next_cycle, next_id)
        board = lab.leaderboard(run_id)["candidates"][0]

    assert board["eligible_cycles"] == 2


def test_pre_band_legacy_run_keeps_its_original_no_floor_entry_rule(tmp_path) -> None:
    now = datetime.now(UTC)
    market_data = market(now)
    very_low = QuotePoint(
        observed_at=now, up_bid="0.00", up_ask="0.01",
        down_bid="0.98", down_ask="0.99",
    )
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        lab = StrategyLab(store)
        run_id = lab.start(["BTC"], 1, 2, 3600, fee_schedules("BTC"))
        store.connection.execute(
            """UPDATE lab_candidates SET side_policy='up',entry_min=NULL,
            entry_price='0.25',take_profit='0.40',stop_loss='0.005',min_seconds=60
            WHERE run_id=?""", (run_id,),
        )
        event_id = store.record_quote(
            "BTC", market_data.ticker, very_low, market_data.close_time
        )
        lab.on_quote(run_id, market_data, very_low, event_id)
        count = store.connection.execute(
            "SELECT COUNT(*) FROM lab_positions WHERE run_id=?", (run_id,)
        ).fetchone()[0]
        displayed_min = lab.leaderboard(run_id)["candidates"][0]["entry_min"]

    assert count == 1
    assert displayed_min == "0.1500"


def test_history_warmup_uses_latest_completed_cycles_then_starts_realtime(tmp_path) -> None:
    realtime_start = datetime(2026, 8, 3, 18, 0, tzinfo=UTC)
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        lab = StrategyLab(store)
        for number, close_time in (
            (1, realtime_start - timedelta(minutes=30)),
            (2, realtime_start - timedelta(minutes=15)),
        ):
            ticker = f"BTC-{number}"
            for offset in (600, 599):
                observed = close_time - timedelta(seconds=offset)
                item = quote(observed, "0.20")
                store.record_quote("BTC", ticker, item, close_time)
            store.save_market_result(ticker, "up", close_time)
        cutoff = store.connection.execute("SELECT MAX(id) FROM quote_events").fetchone()[0]
        run_id = lab.start(
            ["BTC"], 1, 2, 3600, fee_schedules("BTC"),
            history_cycles=1, history_cutoff_event_id=cutoff,
        )
        store.connection.execute(
            """UPDATE lab_candidates SET side_policy='up',entry_min='0.20',
            entry_price='0.25',take_profit='0.80',stop_loss='0.15',min_seconds=60
            WHERE run_id=?""", (run_id,),
        )

        asyncio.run(lab.backfill(run_id, now=realtime_start))

        run = next(item for item in lab.list_runs() if item["id"] == run_id)
        board = lab.leaderboard(run_id)["candidates"][0]
        seen_tickers = {
            row[0] for row in store.connection.execute(
                """SELECT DISTINCT q.ticker FROM quote_events q
                JOIN lab_seen_events s ON s.quote_event_id=q.id WHERE s.run_id=?""",
                (run_id,),
            )
        }

    assert run["status"] == "running"
    assert datetime.fromisoformat(run["started_at"]) == realtime_start
    assert run["history_cycles_loaded"] == 1
    assert run["history_events_total"] == run["history_events_processed"] == 2
    assert seen_tickers == {"BTC-2"}
    assert board["settlement_count"] == 1


def test_stopped_history_warmup_never_transitions_to_realtime(tmp_path) -> None:
    now = datetime(2026, 8, 3, 18, 0, tzinfo=UTC)
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        lab = StrategyLab(store)
        run_id = lab.start(
            ["BTC"], 1, 2, 3600, fee_schedules("BTC"),
            history_cycles=12, history_cutoff_event_id=0,
        )
        assert lab.stop(run_id) is True
        asyncio.run(lab.backfill(run_id, now=now))
        run = next(item for item in lab.list_runs() if item["id"] == run_id)

    assert run["status"] == "stopped"
    assert run["history_events_processed"] == 0


def test_realtime_clock_starts_after_history_processing(tmp_path, monkeypatch) -> None:
    import kaishi_bot.lab_vectorized as vectorized_module

    before = datetime(2026, 8, 3, 17, 0, tzinfo=UTC)
    after = datetime(2026, 8, 3, 18, 0, tzinfo=UTC)
    class FakeDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return after

    with DashboardStore(tmp_path / "state.sqlite3") as store:
        lab = StrategyLab(store)
        close_time = before - timedelta(minutes=15)
        item = quote(close_time - timedelta(minutes=10), "0.20")
        event_id = store.record_quote("BTC", "BTC-HISTORY", item, close_time)
        store.save_market_result("BTC-HISTORY", "up", close_time)
        run_id = lab.start(
            ["BTC"], 1, 2, 3600, fee_schedules("BTC"),
            history_cycles=1, history_cutoff_event_id=event_id,
        )
        monkeypatch.setattr(vectorized_module, "datetime", FakeDateTime)
        asyncio.run(lab.backfill(run_id))
        run = next(row for row in lab.list_runs() if row["id"] == run_id)

    assert datetime.fromisoformat(run["started_at"]) == after


def test_history_falls_back_to_older_settled_cycles(tmp_path) -> None:
    now = datetime(2026, 8, 3, 18, 0, tzinfo=UTC)
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        lab = StrategyLab(store)
        for number in (1, 2, 3):
            close_time = now - timedelta(minutes=15 * (4 - number))
            item = quote(close_time - timedelta(minutes=10), "0.20")
            store.record_quote("BTC", f"BTC-{number}", item, close_time)
            if number != 3:
                store.save_market_result(f"BTC-{number}", "up", close_time)
        cutoff = store.connection.execute("SELECT MAX(id) FROM quote_events").fetchone()[0]
        run_id = lab.start(
            ["BTC"], 1, 2, 3600, fee_schedules("BTC"),
            history_cycles=2, history_cutoff_event_id=cutoff,
        )
        asyncio.run(lab.backfill(run_id, now=now))
        tickers = {
            row[0] for row in store.connection.execute(
                """SELECT DISTINCT q.ticker FROM quote_events q
                JOIN lab_seen_events s ON s.quote_event_id=q.id WHERE s.run_id=?""",
                (run_id,),
            )
        }

    assert tickers == {"BTC-1", "BTC-2"}


def test_history_backfill_does_not_call_candidate_sql_engine(tmp_path, monkeypatch) -> None:
    now = datetime(2026, 8, 3, 18, 0, tzinfo=UTC)
    close_time = now - timedelta(minutes=15)
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        lab = StrategyLab(store)
        entry = quote(close_time - timedelta(minutes=10), "0.20")
        event_id = store.record_quote("BTC", "BTC-BATCH", entry, close_time)
        store.save_market_result("BTC-BATCH", "up", close_time)
        run_id = lab.start(
            ["BTC"], 2, 2, 3600, fee_schedules("BTC"),
            history_cycles=1, history_cutoff_event_id=event_id,
        )
        store.connection.execute(
            """UPDATE lab_candidates SET side_policy='up',entry_min='0.20',
            entry_price='0.25',take_profit='0.80',stop_loss='0.15',min_seconds=60
            WHERE run_id=?""", (run_id,),
        )

        def fail_per_candidate(*_args, **_kwargs):
            raise AssertionError("historical backfill used per-candidate SQL engine")

        monkeypatch.setattr(lab, "_apply_candidate", fail_per_candidate)
        asyncio.run(lab.backfill(run_id, now=now))
        board = lab.leaderboard(run_id)

    assert [item["entry_count"] for item in board["candidates"]] == [1, 1]
    assert [item["settlement_count"] for item in board["candidates"]] == [1, 1]


def test_vectorized_history_matches_legacy_trade_results(tmp_path) -> None:
    now = datetime(2026, 8, 3, 18, 0, tzinfo=UTC)
    close_time = now - timedelta(minutes=15)

    def run(engine: str, database: str) -> tuple[dict[str, object], list[tuple[object, ...]]]:
        with DashboardStore(tmp_path / database) as store:
            lab = StrategyLab(store)
            quotes = [
                quote(close_time - timedelta(minutes=10), "0.20"),
                QuotePoint(
                    observed_at=close_time - timedelta(minutes=9, seconds=59),
                    up_bid="0.41", up_ask="0.42", down_bid="0.58", down_ask="0.59",
                ),
                quote(close_time - timedelta(minutes=9, seconds=58), "0.20"),
                QuotePoint(
                    observed_at=close_time - timedelta(minutes=9, seconds=57),
                    up_bid="0.14", up_ask="0.15", down_bid="0.84", down_ask="0.85",
                ),
            ]
            last_id = 0
            for item in quotes:
                last_id = store.record_quote("BTC", "BTC-EQUIV", item, close_time)
            store.save_market_result("BTC-EQUIV", "up", close_time)
            run_id = lab.start(
                ["BTC"], 1, 2, 3600, fee_schedules("BTC"),
                history_cycles=1, history_cutoff_event_id=last_id,
            )
            store.connection.execute(
                """UPDATE lab_runs SET backfill_engine=? WHERE id=?""", (engine, run_id)
            )
            store.connection.execute(
                """UPDATE lab_candidates SET side_policy='up',entry_min='0.20',
                entry_price='0.25',take_profit='0.40',stop_loss='0.15',min_seconds=60
                WHERE run_id=?""", (run_id,),
            )
            asyncio.run(lab.backfill(run_id, now=now))
            candidate = dict(store.connection.execute(
                "SELECT * FROM lab_candidates WHERE run_id=?", (run_id,)
            ).fetchone())
            trades = [tuple(row) for row in store.connection.execute(
                """SELECT ticker,side,quantity,entry_price,entry_cost,status,exit_price,pnl,
                entry_fee,exit_fee,entry_outlay,gross_proceeds,net_proceeds,
                entry_event_id,exit_event_id,close_reason FROM lab_positions
                WHERE run_id=? ORDER BY id""", (run_id,)
            )]
            return candidate, trades

    legacy, legacy_trades = run("legacy", "legacy.sqlite3")
    vectorized, vectorized_trades = run("vectorized", "vectorized.sqlite3")
    for key in (
        "cash", "realized_pnl", "closed_trades", "wins", "total_fees",
        "entry_count", "tp_count", "sl_count", "settlement_count",
        "peak_equity", "max_drawdown",
    ):
        assert Decimal(str(vectorized[key])) == Decimal(str(legacy[key]))
    assert vectorized_trades == legacy_trades


def test_vectorized_guard_accepts_exact_one_cent_spread(tmp_path) -> None:
    now = datetime(2026, 8, 3, 18, 0, tzinfo=UTC)
    close_time = now - timedelta(minutes=15)
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        lab = StrategyLab(store)
        item = quote(close_time - timedelta(minutes=10), "0.20")
        event_id = store.record_quote("BTC", "BTC-GUARD", item, close_time)
        store.save_market_result("BTC-GUARD", "up", close_time)
        run_id = lab.start(
            ["BTC"], 1, 2, 3600, fee_schedules("BTC"), EntryGuardSettings(),
            history_cycles=1, history_cutoff_event_id=event_id,
        )
        store.connection.execute(
            """UPDATE lab_candidates SET side_policy='up',entry_min='0.20',
            entry_price='0.25',take_profit='0.40',stop_loss='0.15',min_seconds=60
            WHERE run_id=?""", (run_id,),
        )
        asyncio.run(lab.backfill(run_id, now=now))
        candidate = store.connection.execute(
            "SELECT * FROM lab_candidates WHERE run_id=?", (run_id,)
        ).fetchone()

    assert candidate["entry_count"] == 1
    assert candidate["settlement_count"] == 1
    assert candidate["guard_spread_too_wide"] == 0


def test_vectorized_history_batches_candidate_updates(tmp_path) -> None:
    now = datetime(2026, 8, 3, 18, 0, tzinfo=UTC)
    close_time = now - timedelta(minutes=15)
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        lab = StrategyLab(store)
        last_id = 0
        for seconds in range(5):
            item = quote(close_time - timedelta(minutes=10) + timedelta(seconds=seconds), "0.20")
            last_id = store.record_quote("BTC", "BTC-BATCH", item, close_time)
        store.save_market_result("BTC-BATCH", "up", close_time)
        run_id = lab.start(
            ["BTC"], 100, 2, 3600, fee_schedules("BTC"),
            history_cycles=1, history_cutoff_event_id=last_id,
        )
        statements: list[str] = []
        store.connection.set_trace_callback(statements.append)
        asyncio.run(lab.backfill(run_id, now=now))
        store.connection.set_trace_callback(None)

    candidate_updates = [
        statement for statement in statements
        if statement.startswith("UPDATE lab_candidates SET cash=")
    ]
    assert len(candidate_updates) <= 200


def test_vectorized_history_resumes_from_atomic_checkpoint(tmp_path, monkeypatch) -> None:
    import kaishi_bot.lab_vectorized as vectorized_module

    now = datetime(2026, 8, 3, 18, 0, tzinfo=UTC)
    close_time = now - timedelta(minutes=15)
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        lab = StrategyLab(store)
        events = [
            quote(close_time - timedelta(minutes=10), "0.20"),
            QuotePoint(
                observed_at=close_time - timedelta(minutes=9, seconds=59),
                up_bid="0.41", up_ask="0.42", down_bid="0.58", down_ask="0.59",
            ),
        ]
        last_id = 0
        for item in events:
            last_id = store.record_quote("BTC", "BTC-RESUME", item, close_time)
        store.save_market_result("BTC-RESUME", "up", close_time)
        run_id = lab.start(
            ["BTC"], 1, 2, 3600, fee_schedules("BTC"),
            history_cycles=1, history_cutoff_event_id=last_id,
        )
        store.connection.execute(
            """UPDATE lab_candidates SET side_policy='up',entry_min='0.20',
            entry_price='0.25',take_profit='0.40',stop_loss='0.15',min_seconds=60
            WHERE run_id=?""", (run_id,),
        )
        calls = 0

        async def stop_after_first_quote(_delay):
            nonlocal calls
            calls += 1
            if calls == 1:
                lab.stop(run_id)

        with monkeypatch.context() as patcher:
            patcher.setattr(vectorized_module.asyncio, "sleep", stop_after_first_quote)
            asyncio.run(lab.backfill(run_id, now=now))
        assert store.connection.execute(
            "SELECT history_events_processed FROM lab_runs WHERE id=?", (run_id,)
        ).fetchone()[0] == 1
        store.connection.execute(
            "UPDATE lab_runs SET status='backfilling' WHERE id=?", (run_id,)
        )
        asyncio.run(lab.backfill(run_id, now=now))
        candidate = store.connection.execute(
            "SELECT * FROM lab_candidates WHERE run_id=?", (run_id,)
        ).fetchone()
        positions = list(store.connection.execute(
            "SELECT * FROM lab_positions WHERE run_id=?", (run_id,)
        ))

    assert candidate["entry_count"] == 1
    assert candidate["tp_count"] == 1
    assert len(positions) == 1
    assert positions[0]["close_reason"] == "take_profit"


def test_history_uses_cached_settlement_time_in_event_order(tmp_path) -> None:
    now = datetime(2026, 8, 3, 18, 0, tzinfo=UTC)
    first_close = now - timedelta(minutes=30)
    second_close = now - timedelta(minutes=15)
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        lab = StrategyLab(store)
        first_quote = quote(first_close - timedelta(minutes=10), "0.20")
        first_id = store.record_quote("BTC", "BTC-1", first_quote, first_close)
        second_early = quote(first_close + timedelta(seconds=1), "0.20")
        early_id = store.record_quote("BTC", "BTC-2", second_early, second_close)
        second_late = quote(first_close + timedelta(seconds=3), "0.20")
        late_id = store.record_quote("BTC", "BTC-2", second_late, second_close)
        store.save_market_result("BTC-1", "up", first_close + timedelta(seconds=2))
        store.save_market_result("BTC-2", "up", second_close)
        run_id = lab.start(
            ["BTC"], 1, 2, 3600, fee_schedules("BTC"),
            history_cycles=2, history_cutoff_event_id=late_id,
        )
        store.connection.execute(
            """UPDATE lab_candidates SET side_policy='up',entry_min='0.20',
            entry_price='0.25',take_profit='0.80',stop_loss='0.15',min_seconds=60
            WHERE run_id=?""", (run_id,),
        )
        asyncio.run(lab.backfill(run_id, now=now))
        entries = list(store.connection.execute(
            """SELECT ticker,entry_event_id FROM lab_positions
            WHERE run_id=? ORDER BY id""", (run_id,),
        ))

    assert first_id < early_id < late_id
    assert [(row["ticker"], row["entry_event_id"]) for row in entries][:2] == [
        ("BTC-1", first_id), ("BTC-2", late_id)
    ]


def test_stopping_at_last_history_yield_prevents_late_settlement(
    tmp_path, monkeypatch
) -> None:
    import kaishi_bot.lab as lab_module

    now = datetime(2026, 8, 3, 18, 0, tzinfo=UTC)
    close_time = now - timedelta(minutes=15)
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        lab = StrategyLab(store)
        item = quote(close_time - timedelta(minutes=10), "0.20")
        event_id = store.record_quote("BTC", "BTC-1", item, close_time)
        store.save_market_result("BTC-1", "up", close_time)
        run_id = lab.start(
            ["BTC"], 1, 2, 3600, fee_schedules("BTC"),
            history_cycles=1, history_cutoff_event_id=event_id,
        )
        store.connection.execute(
            """UPDATE lab_candidates SET side_policy='up',entry_min='0.20',
            entry_price='0.25',take_profit='0.80',stop_loss='0.15',min_seconds=60
            WHERE run_id=?""", (run_id,),
        )

        async def stop_on_yield(_):
            lab.stop(run_id)

        monkeypatch.setattr(lab_module.asyncio, "sleep", stop_on_yield)
        asyncio.run(lab.backfill(run_id, now=now))
        position = store.connection.execute(
            "SELECT status FROM lab_positions WHERE run_id=?", (run_id,)
        ).fetchone()
        run = next(row for row in lab.list_runs() if row["id"] == run_id)

    assert run["status"] == "stopped"
    assert position["status"] == "open"


def test_replay_does_not_apply_realtime_duration_to_stored_events(tmp_path) -> None:
    now = datetime(2026, 8, 3, 18, 0, tzinfo=UTC)
    close_time = now - timedelta(minutes=15)
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        lab = StrategyLab(store)
        entry = quote(close_time - timedelta(minutes=12), "0.20")
        tp = QuotePoint(
            observed_at=entry.observed_at + timedelta(seconds=61),
            up_bid="0.41", up_ask="0.42", down_bid="0.58", down_ask="0.59",
        )
        first_id = store.record_quote("BTC", "BTC-1", entry, close_time)
        second_id = store.record_quote("BTC", "BTC-1", tp, close_time)
        store.save_market_result("BTC-1", "down", close_time)
        run_id = lab.start(
            ["BTC"], 1, 2, 60, fee_schedules("BTC"),
            history_cycles=1, history_cutoff_event_id=second_id,
        )
        store.connection.execute(
            """UPDATE lab_candidates SET side_policy='up',entry_min='0.20',
            entry_price='0.25',take_profit='0.40',stop_loss='0.15',min_seconds=60
            WHERE run_id=?""", (run_id,),
        )
        asyncio.run(lab.backfill(run_id, now=now))
        source = lab.leaderboard(run_id)["candidates"][0]
        lab.stop(run_id)
        replay_id = lab.replay(run_id)
        replayed = lab.leaderboard(replay_id)["candidates"][0]

    assert first_id < second_id
    assert replayed["net_pnl"] == source["net_pnl"]
    assert replayed["tp_count"] == source["tp_count"] == 1


def test_lab_allows_five_thousand_candidates_per_coin(tmp_path) -> None:
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        lab = StrategyLab(store)
        assets = ["BTC", "ETH", "SOL", "XRP", "DOGE"]
        run_id = lab.start(
            assets, 5000, 42, 10800, fee_schedules(*assets)
        )
        total = store.connection.execute(
            "SELECT COUNT(*) FROM lab_candidates WHERE run_id=?", (run_id,)
        ).fetchone()[0]
        assert total == 25000

        with pytest.raises(ValueError, match="stop the current run"):
            lab.start(["BTC"], 1, 43, 10800, fee_schedules("BTC"))

        assert lab.stop(run_id) is True
        with pytest.raises(ValueError, match="25000"):
            lab.start(["BTC"], 25001, 44, 10800, fee_schedules("BTC"))


def test_lab_fans_one_quote_to_independent_candidates_and_ranks(tmp_path) -> None:
    now = datetime(2026, 8, 3, 16, 0, tzinfo=UTC)
    market = AssetMarket(
        asset="BTC", series="KXBTC15M", ticker="BTC-1", title="BTC",
        open_time=now - timedelta(minutes=1), close_time=now + timedelta(minutes=10),
    )
    low = QuotePoint(
        observed_at=now, up_bid="0.09", up_ask="0.10", down_bid="0.89", down_ask="0.90"
    )
    high = QuotePoint(
        observed_at=now + timedelta(seconds=1), up_bid="0.80", up_ask="0.81",
        down_bid="0.19", down_ask="0.20",
    )
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        lab = StrategyLab(store)
        run_id = lab.start(["BTC"], 10, 7, 10800, fee_schedules("BTC"))
        event_id = store.record_quote("BTC", market.ticker, low, market.close_time)
        lab.on_quote(run_id, market, low, event_id)
        high_event_id = store.record_quote("BTC", market.ticker, high, market.close_time)
        lab.on_quote(run_id, market, high, high_event_id)
        board = lab.leaderboard(run_id)
        replay_statuses = []
        original_on_quote = lab.on_quote

        def observe_replay_statuses(*args, **kwargs):
            replay_statuses.append([
                row["status"] for row in store.connection.execute(
                    "SELECT status FROM lab_runs ORDER BY id"
                )
            ])
            return original_on_quote(*args, **kwargs)

        lab.on_quote = observe_replay_statuses
        replay_id = lab.replay(run_id)
        replay_board = lab.leaderboard(replay_id)

    assert len(board["candidates"]) == 10
    assert board["candidates"][0]["cash"] != board["candidates"][-1]["cash"] or any(
        item["closed_trades"] == 0 for item in board["candidates"]
    )
    assert board["preliminary"] is True
    assert [item["net_pnl"] for item in replay_board["candidates"]] == [
        item["net_pnl"] for item in board["candidates"]
    ]
    assert replay_statuses
    assert all(statuses.count("running") == 1 for statuses in replay_statuses)
    assert all("replaying" in statuses for statuses in replay_statuses)


def test_replay_failure_does_not_leave_a_running_run(tmp_path) -> None:
    now = datetime.now(UTC)
    market_data = market(now)
    q = quote(now, "0.20")
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        lab = StrategyLab(store)
        run_id = lab.start(["BTC"], 1, 2, 3600, fee_schedules("BTC"))
        event_id = store.record_quote("BTC", market_data.ticker, q, market_data.close_time)
        lab.on_quote(run_id, market_data, q, event_id)
        with store.connection:
            store.connection.execute(
                "UPDATE quote_events SET observed_at='invalid' WHERE id=?", (event_id,)
            )

        with pytest.raises(ValueError):
            lab.replay(run_id)

        replay = store.connection.execute(
            "SELECT status FROM lab_runs WHERE id=(SELECT MAX(id) FROM lab_runs)"
        ).fetchone()
        assert replay["status"] == "failed"


def test_lab_stops_automatically_after_duration(tmp_path) -> None:
    now = datetime.now(UTC)
    market_data = market(now)
    q = quote(now + timedelta(seconds=61), "0.20")
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        lab = StrategyLab(store)
        run_id = lab.start(["BTC"], 1, 1, 60, fee_schedules("BTC"))
        lab.on_quote(run_id, market_data, q, 1)
        run = lab.list_runs()[0]
    assert run["status"] == "completed"
    assert run["quote_count"] == 0


def test_both_side_signal_opens_only_lower_ask_with_one_dollar_all_in(tmp_path) -> None:
    now = datetime.now(UTC)
    market_data = market(now)
    q = QuotePoint(
        observed_at=now, up_bid="0.19", up_ask="0.20", down_bid="0.19", down_ask="0.20"
    )
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        lab = StrategyLab(store)
        run_id = lab.start(["BTC"], 1, 2, 3600, fee_schedules("BTC"))
        store.connection.execute(
            """UPDATE lab_candidates SET side_policy='both',entry_min='0.20',
            entry_price='0.25' WHERE run_id=?""",
            (run_id,),
        )
        event_id = store.record_quote("BTC", market_data.ticker, q)
        lab.on_quote(run_id, market_data, q, event_id)
        spend = store.connection.execute(
            "SELECT amount FROM lab_daily_spend WHERE run_id=?", (run_id,)
        ).fetchone()[0]
        positions = list(store.connection.execute(
            "SELECT * FROM lab_positions WHERE run_id=?", (run_id,)
        ))
    assert len(positions) == 1
    assert positions[0]["side"] == "up"
    assert Decimal(positions[0]["entry_outlay"]) <= Decimal("1.00")
    assert Decimal(spend) == Decimal(positions[0]["entry_outlay"])


def test_candidate_reenters_same_ticker_side_only_on_later_event(tmp_path) -> None:
    now = datetime.now(UTC)
    market_data = market(now)
    entry = QuotePoint(
        observed_at=now, up_bid="0.19", up_ask="0.20",
        down_bid="0.79", down_ask="0.80",
    )
    take_profit = QuotePoint(
        observed_at=now + timedelta(seconds=1), up_bid="0.41", up_ask="0.42",
        down_bid="0.58", down_ask="0.59",
    )
    reentry = QuotePoint(
        observed_at=now + timedelta(seconds=2), up_bid="0.19", up_ask="0.20",
        down_bid="0.79", down_ask="0.80",
    )
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        lab = StrategyLab(store)
        run_id = lab.start(["BTC"], 1, 2, 3600, fee_schedules("BTC"))
        store.connection.execute(
            """UPDATE lab_candidates SET side_policy='up',entry_min='0.20',entry_price='0.25',
            take_profit='0.40',stop_loss='0.15',min_seconds=60 WHERE run_id=?""",
            (run_id,),
        )
        events = []
        for item in (entry, take_profit, reentry):
            event_id = store.record_quote("BTC", market_data.ticker, item, market_data.close_time)
            events.append(event_id)
            lab.on_quote(run_id, market_data, item, event_id)
        positions = list(store.connection.execute(
            "SELECT * FROM lab_positions WHERE run_id=? ORDER BY id", (run_id,)
        ))
    assert [item["status"] for item in positions] == ["closed", "open"]
    assert [item["entry_event_id"] for item in positions] == [events[0], events[2]]
    assert positions[0]["exit_event_id"] == events[1]


def test_open_position_blocks_opposite_side_and_dca(tmp_path) -> None:
    now = datetime.now(UTC)
    market_data = market(now)
    first = QuotePoint(
        observed_at=now, up_bid="0.19", up_ask="0.20",
        down_bid="0.29", down_ask="0.30",
    )
    opposite = QuotePoint(
        observed_at=now + timedelta(seconds=1), up_bid="0.20", up_ask="0.21",
        down_bid="0.09", down_ask="0.10",
    )
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        lab = StrategyLab(store)
        run_id = lab.start(["BTC"], 1, 2, 3600, fee_schedules("BTC"))
        store.connection.execute(
            """UPDATE lab_candidates SET side_policy='both',entry_min='0.10',entry_price='0.35',
            take_profit='0.80',stop_loss='0.05',min_seconds=60 WHERE run_id=?""",
            (run_id,),
        )
        for item in (first, opposite):
            event_id = store.record_quote("BTC", market_data.ticker, item, market_data.close_time)
            lab.on_quote(run_id, market_data, item, event_id)
        positions = list(store.connection.execute(
            "SELECT * FROM lab_positions WHERE run_id=?", (run_id,)
        ))
    assert len(positions) == 1
    assert positions[0]["side"] == "up"


def test_lab_daily_cap_counts_all_in_outlay_and_never_exceeds_fifty(tmp_path) -> None:
    now = datetime.now(UTC)
    market_data = market(now)
    too_large = QuotePoint(
        observed_at=now, up_bid="0.49", up_ask="0.50",
        down_bid="0.49", down_ask="0.50",
    )
    fits = QuotePoint(
        observed_at=now + timedelta(seconds=1), up_bid="0.09", up_ask="0.10",
        down_bid="0.89", down_ask="0.90",
    )
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        lab = StrategyLab(store)
        run_id = lab.start(["BTC"], 1, 2, 3600, fee_schedules("BTC"))
        store.connection.execute(
            """UPDATE lab_candidates SET side_policy='up',entry_min='0.10',entry_price='0.60',
            min_seconds=60 WHERE run_id=?""", (run_id,),
        )
        day = now.astimezone(__import__("zoneinfo").ZoneInfo("America/New_York")).date().isoformat()
        store.connection.execute(
            "INSERT INTO lab_daily_spend VALUES (?,?,?,?)",
            (run_id, "BTC-000", day, "49.60"),
        )
        first_id = store.record_quote("BTC", market_data.ticker, too_large, market_data.close_time)
        lab.on_quote(run_id, market_data, too_large, first_id)
        assert store.connection.execute(
            "SELECT COUNT(*) FROM lab_positions WHERE run_id=?", (run_id,)
        ).fetchone()[0] == 0
        second_id = store.record_quote("BTC", market_data.ticker, fits, market_data.close_time)
        lab.on_quote(run_id, market_data, fits, second_id)
        spent = Decimal(store.connection.execute(
            "SELECT amount FROM lab_daily_spend WHERE run_id=?", (run_id,)
        ).fetchone()[0])
        blocked = store.connection.execute(
            "SELECT blocked_daily_cap FROM lab_candidates WHERE run_id=?", (run_id,)
        ).fetchone()[0]
    assert spent <= Decimal("50.00")
    assert blocked == 1


def test_expired_lab_position_settles_to_authoritative_result_without_exit_fee(tmp_path) -> None:
    now = datetime.now(UTC)
    market_data = market(now)
    q = QuotePoint(
        observed_at=now, up_bid="0.19", up_ask="0.20",
        down_bid="0.79", down_ask="0.80",
    )
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        lab = StrategyLab(store)
        run_id = lab.start(["BTC"], 1, 2, 3600, fee_schedules("BTC"))
        store.connection.execute(
            """UPDATE lab_candidates SET side_policy='up',entry_min='0.20',entry_price='0.25',
            min_seconds=60 WHERE run_id=?""", (run_id,),
        )
        event_id = store.record_quote("BTC", market_data.ticker, q, market_data.close_time)
        lab.on_quote(run_id, market_data, q, event_id)
        assert lab.settle_ticker(run_id, market_data.ticker, "up", now + timedelta(minutes=15)) == 1
        position = store.connection.execute(
            "SELECT * FROM lab_positions WHERE run_id=?", (run_id,)
        ).fetchone()
        candidate = store.connection.execute(
            "SELECT * FROM lab_candidates WHERE run_id=?", (run_id,)
        ).fetchone()
    assert position["status"] == "closed"
    assert position["close_reason"] == "settlement"
    assert Decimal(position["exit_fee"]) == Decimal("0")
    assert Decimal(position["exit_price"]) == Decimal("1")
    assert candidate["settlement_count"] == 1


def test_settlement_is_applied_only_once_per_run_and_ticker(tmp_path) -> None:
    now = datetime.now(UTC)
    market_data = market(now)
    first_quote = QuotePoint(
        observed_at=now, up_bid="0.19", up_ask="0.20",
        down_bid="0.79", down_ask="0.80",
    )
    second_quote = first_quote.model_copy(
        update={"observed_at": now + timedelta(seconds=1)}
    )
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        lab = StrategyLab(store)
        run_id = lab.start(["BTC"], 1, 2, 3600, fee_schedules("BTC"))
        store.connection.execute(
            """UPDATE lab_candidates SET side_policy='up',entry_min='0.20',
            entry_price='0.25',take_profit='0.80',stop_loss='0.15',min_seconds=60
            WHERE run_id=?""", (run_id,),
        )
        first_id = store.record_quote(
            "BTC", market_data.ticker, first_quote, market_data.close_time
        )
        lab.on_quote(run_id, market_data, first_quote, first_id)
        assert lab.settle_ticker(
            run_id, market_data.ticker, "up", now + timedelta(minutes=15)
        ) == 1

        second_id = store.record_quote(
            "BTC", market_data.ticker, second_quote, market_data.close_time
        )
        lab.on_quote(run_id, market_data, second_quote, second_id)
        assert lab.settle_ticker(
            run_id, market_data.ticker, "up", now + timedelta(minutes=15)
        ) == 0

        candidate = store.connection.execute(
            "SELECT * FROM lab_candidates WHERE run_id=?", (run_id,)
        ).fetchone()
        open_positions = store.connection.execute(
            """SELECT COUNT(*) FROM lab_positions WHERE run_id=?
            AND status='open'""", (run_id,),
        ).fetchone()[0]

    assert candidate["closed_trades"] == 1
    assert candidate["settlement_count"] == 1
    assert open_positions == 1


def test_replay_reproduces_authoritative_settlement(tmp_path) -> None:
    now = datetime.now(UTC)
    market_data = market(now)
    q = QuotePoint(
        observed_at=now, up_bid="0.19", up_ask="0.20",
        down_bid="0.79", down_ask="0.80",
    )
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        lab = StrategyLab(store)
        run_id = lab.start(["BTC"], 1, 2, 3600, fee_schedules("BTC"))
        store.connection.execute(
            """UPDATE lab_candidates SET side_policy='up',entry_min='0.20',
            entry_price='0.25',take_profit='0.80',stop_loss='0.15',min_seconds=60
            WHERE run_id=?""", (run_id,),
        )
        event_id = store.record_quote("BTC", market_data.ticker, q, market_data.close_time)
        lab.on_quote(run_id, market_data, q, event_id)
        lab.settle_ticker(run_id, market_data.ticker, "up", now + timedelta(minutes=15))
        source = lab.leaderboard(run_id)["candidates"][0]

        replay_id = lab.replay(run_id)
        replayed = lab.leaderboard(replay_id)["candidates"][0]

    assert replayed["net_pnl"] == source["net_pnl"]
    assert replayed["settlement_count"] == source["settlement_count"] == 1


def test_leaderboard_marks_only_with_quotes_seen_by_the_run(tmp_path) -> None:
    now = datetime.now(UTC)
    market_data = market(now)
    entry = quote(now, "0.20")
    later = quote(now + timedelta(seconds=10), "0.90")
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        lab = StrategyLab(store)
        run_id = lab.start(["BTC"], 1, 2, 3600, fee_schedules("BTC"))
        event_id = store.record_quote("BTC", market_data.ticker, entry, market_data.close_time)
        lab.on_quote(run_id, market_data, entry, event_id)
        before = lab.leaderboard(run_id)["candidates"][0]["equity"]

        store.record_quote("BTC", market_data.ticker, later, market_data.close_time)
        after = lab.leaderboard(run_id)["candidates"][0]["equity"]

    assert after == before


def test_leaderboard_reports_net_fees_outcomes_and_observed_cycles(tmp_path) -> None:
    now = datetime.now(UTC)
    market_data = market(now)
    entry = QuotePoint(
        observed_at=now, up_bid="0.19", up_ask="0.20",
        down_bid="0.79", down_ask="0.80",
    )
    take_profit = QuotePoint(
        observed_at=now + timedelta(seconds=1), up_bid="0.41", up_ask="0.42",
        down_bid="0.58", down_ask="0.59",
    )
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        lab = StrategyLab(store)
        run_id = lab.start(["BTC"], 1, 2, 3600, fee_schedules("BTC"))
        store.connection.execute(
            """UPDATE lab_candidates SET side_policy='up',entry_min='0.20',entry_price='0.25',
            take_profit='0.40',stop_loss='0.15',min_seconds=60 WHERE run_id=?""",
            (run_id,),
        )
        for item in (entry, take_profit):
            event_id = store.record_quote("BTC", market_data.ticker, item, market_data.close_time)
            lab.on_quote(run_id, market_data, item, event_id)
        row = lab.leaderboard(run_id)["candidates"][0]
    assert row["net_pnl"] == Decimal("0.72")
    assert row["total_deployed"] == Decimal("0.85")
    assert row["roi_percent"] == row["net_pnl"] / row["total_deployed"] * Decimal("100")
    assert row["total_fees"] == Decimal("0.12")
    assert row["entry_count"] == 1
    assert row["tp_count"] == 1
    assert row["sl_count"] == 0
    assert row["settlement_count"] == 0
    assert row["cycles_seen"] == 1
    assert row["low_confidence"] is True


def test_leaderboard_partitions_trials_at_eight_closed_trades(tmp_path) -> None:
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        lab = StrategyLab(store)
        run_id = lab.start(["BTC"], 2, 2, 3600, fee_schedules("BTC"))
        store.connection.execute(
            """UPDATE lab_candidates SET closed_trades=7,cash='1002.00'
            WHERE run_id=? AND candidate_id='BTC-000'""", (run_id,),
        )
        store.connection.execute(
            """UPDATE lab_candidates SET closed_trades=8,cash='1001.00'
            WHERE run_id=? AND candidate_id='BTC-001'""", (run_id,),
        )
        board = lab.leaderboard(run_id)

        assert board["minimum_closed_trades"] == 8
        assert [row["candidate_id"] for row in board["ranked_candidates"]] == ["BTC-001"]
        assert [row["candidate_id"] for row in board["insufficient_candidates"]] == ["BTC-000"]
        assert [row["candidate_id"] for row in board["candidates"]] == [
            "BTC-001", "BTC-000"
        ]
        assert board["legacy_rules"] is False

        store.connection.execute(
            "UPDATE lab_candidates SET min_seconds=181 WHERE run_id=?", (run_id,)
        )
        assert lab.leaderboard(run_id)["legacy_rules"] is True
        store.connection.execute(
            "UPDATE lab_candidates SET min_seconds=180 WHERE run_id=?", (run_id,)
        )
        store.connection.execute(
            "UPDATE lab_candidates SET entry_min=NULL WHERE run_id=?", (run_id,)
        )
        assert lab.leaderboard(run_id)["legacy_rules"] is True


def test_leaderboard_reports_negative_roi_after_stop_loss(tmp_path) -> None:
    now = datetime.now(UTC)
    market_data = market(now)
    entry = QuotePoint(
        observed_at=now, up_bid="0.19", up_ask="0.20",
        down_bid="0.79", down_ask="0.80",
    )
    stop_loss = QuotePoint(
        observed_at=now + timedelta(seconds=1), up_bid="0.14", up_ask="0.15",
        down_bid="0.85", down_ask="0.86",
    )
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        lab = StrategyLab(store)
        run_id = lab.start(["BTC"], 1, 2, 3600, fee_schedules("BTC"))
        store.connection.execute(
            """UPDATE lab_candidates SET side_policy='up',entry_min='0.20',entry_price='0.25',
            take_profit='0.40',stop_loss='0.15',min_seconds=60 WHERE run_id=?""",
            (run_id,),
        )
        for item in (entry, stop_loss):
            event_id = store.record_quote("BTC", market_data.ticker, item, market_data.close_time)
            lab.on_quote(run_id, market_data, item, event_id)
        row = lab.leaderboard(run_id)["candidates"][0]

    assert row["net_pnl"] < Decimal("0")
    assert row["roi_percent"] == row["net_pnl"] / row["total_deployed"] * Decimal("100")


def test_leaderboard_marks_open_position_at_bid_after_exit_fee(tmp_path) -> None:
    now = datetime.now(UTC)
    market_data = market(now)
    entry = QuotePoint(
        observed_at=now, up_bid="0.19", up_ask="0.20",
        down_bid="0.79", down_ask="0.80",
    )
    marked = QuotePoint(
        observed_at=now + timedelta(seconds=1), up_bid="0.30", up_ask="0.31",
        down_bid="0.69", down_ask="0.70",
    )
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        lab = StrategyLab(store)
        run_id = lab.start(["BTC"], 1, 2, 3600, fee_schedules("BTC"))
        store.connection.execute(
            """UPDATE lab_candidates SET side_policy='up',entry_min='0.20',entry_price='0.25',
            take_profit='0.80',stop_loss='0.05',min_seconds=60 WHERE run_id=?""",
            (run_id,),
        )
        for item in (entry, marked):
            event_id = store.record_quote("BTC", market_data.ticker, item, market_data.close_time)
            lab.on_quote(run_id, market_data, item, event_id)
        row = lab.leaderboard(run_id)["candidates"][0]

    assert row["net_pnl"] == Decimal("0.29")
    assert row["roi_percent"] == row["net_pnl"] / row["total_deployed"] * Decimal("100")


def test_leaderboard_returns_no_roi_for_candidate_without_entries(tmp_path) -> None:
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        lab = StrategyLab(store)
        run_id = lab.start(["BTC"], 1, 2, 3600, fee_schedules("BTC"))
        row = lab.leaderboard(run_id)["candidates"][0]

    assert row["total_deployed"] == Decimal("0")
    assert row["roi_percent"] is None


def test_leaderboard_uses_entry_cost_and_fee_when_legacy_outlay_is_missing(tmp_path) -> None:
    now = datetime.now(UTC)
    market_data = market(now)
    entry = QuotePoint(
        observed_at=now, up_bid="0.19", up_ask="0.20",
        down_bid="0.79", down_ask="0.80",
    )
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        lab = StrategyLab(store)
        run_id = lab.start(["BTC"], 1, 2, 3600, fee_schedules("BTC"))
        store.connection.execute(
            """UPDATE lab_candidates SET side_policy='up',entry_min='0.20',entry_price='0.25',
            min_seconds=60 WHERE run_id=?""",
            (run_id,),
        )
        event_id = store.record_quote("BTC", market_data.ticker, entry, market_data.close_time)
        lab.on_quote(run_id, market_data, entry, event_id)
        store.connection.execute(
            "UPDATE lab_positions SET entry_outlay=NULL WHERE run_id=?", (run_id,)
        )
        row = lab.leaderboard(run_id)["candidates"][0]

    assert row["total_deployed"] == Decimal("0.85")


def test_leaderboard_uses_fixed_bulk_queries(tmp_path) -> None:
    now = datetime.now(UTC)
    market_data = market(now)
    entry = QuotePoint(
        observed_at=now, up_bid="0.19", up_ask="0.20",
        down_bid="0.79", down_ask="0.80",
    )
    exit_quote = QuotePoint(
        observed_at=now + timedelta(seconds=1), up_bid="0.70", up_ask="0.71",
        down_bid="0.29", down_ask="0.30",
    )
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        lab = StrategyLab(store)
        run_id = lab.start(["BTC"], 100, 9, 3600, fee_schedules("BTC"))
        for item in (entry, exit_quote):
            event_id = store.record_quote(
                "BTC", market_data.ticker, item, market_data.close_time
            )
            lab.on_quote(run_id, market_data, item, event_id)

        statements: list[str] = []
        store.connection.set_trace_callback(statements.append)
        board = lab.leaderboard(run_id)
        store.connection.set_trace_callback(None)

    selects = [statement for statement in statements if statement.lstrip().upper().startswith(
        ("SELECT", "WITH")
    )]
    assert len(board["candidates"]) == 100
    assert len(selects) <= 7, "\n".join(selects)


def test_new_guarded_run_enters_on_first_valid_quote(tmp_path) -> None:
    now = datetime.now(UTC)
    market_data = market(now)
    q1 = QuotePoint(
        observed_at=now, up_bid="0.19", up_ask="0.20",
        down_bid="0.79", down_ask="0.80",
    )
    q2 = q1.model_copy(update={"observed_at": now + timedelta(seconds=1)})
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        lab = StrategyLab(store)
        guarded = lab.start(
            ["BTC"], 1, 2, 3600, fee_schedules("BTC"), EntryGuardSettings()
        )
        store.connection.execute(
            """UPDATE lab_candidates SET side_policy='up',entry_min='0.20',entry_price='0.25',
            take_profit='0.40',stop_loss='0.15',min_seconds=60 WHERE run_id=?""",
            (guarded,),
        )
        first_id = store.record_quote("BTC", market_data.ticker, q1, market_data.close_time)
        lab.on_quote(guarded, market_data, q1, first_id)
        assert store.connection.execute(
            "SELECT COUNT(*) FROM lab_positions WHERE run_id=?", (guarded,)
        ).fetchone()[0] == 1
        second_id = store.record_quote("BTC", market_data.ticker, q2, market_data.close_time)
        lab.on_quote(guarded, market_data, q2, second_id)
        assert store.connection.execute(
            "SELECT COUNT(*) FROM lab_positions WHERE run_id=?", (guarded,)
        ).fetchone()[0] == 1
