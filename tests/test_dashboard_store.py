from decimal import Decimal
from datetime import UTC, datetime, timedelta
import sqlite3

import pytest

from kaishi_bot.dashboard_models import DashboardSettings
from kaishi_bot.dashboard_store import DashboardStore
from kaishi_bot.entry_guard import GuardReason


LEGACY_LAB_SCHEMA = """
CREATE TABLE lab_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, status TEXT NOT NULL,
    started_at TEXT NOT NULL, duration_seconds INTEGER NOT NULL,
    seed INTEGER NOT NULL, assets TEXT NOT NULL,
    candidate_count INTEGER NOT NULL, quote_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE lab_candidates (
    run_id INTEGER NOT NULL, candidate_id TEXT NOT NULL, asset TEXT NOT NULL,
    entry_price TEXT NOT NULL, take_profit TEXT NOT NULL, stop_loss TEXT NOT NULL,
    min_seconds INTEGER NOT NULL, side_policy TEXT NOT NULL,
    cash TEXT NOT NULL DEFAULT '1000.00', peak_equity TEXT NOT NULL DEFAULT '1000.00',
    max_drawdown TEXT NOT NULL DEFAULT '0', realized_pnl TEXT NOT NULL DEFAULT '0',
    closed_trades INTEGER NOT NULL DEFAULT 0, wins INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(run_id,candidate_id)
);
CREATE TABLE lab_positions (
    run_id INTEGER NOT NULL, candidate_id TEXT NOT NULL,
    ticker TEXT NOT NULL, side TEXT NOT NULL, quantity TEXT NOT NULL,
    entry_price TEXT NOT NULL, entry_cost TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open', exit_price TEXT, pnl TEXT,
    PRIMARY KEY(run_id,candidate_id,ticker,side)
);
"""


def test_new_store_has_exactly_one_thousand_dollars_and_persists_settings(tmp_path) -> None:
    path = tmp_path / "dashboard.sqlite3"
    with DashboardStore(path) as store:
        assert store.cash() == Decimal("1000.00")
        changed = DashboardSettings(entry_price="0.20", stop_loss="0.10", take_profit="0.35")
        store.save_settings(changed)

    with DashboardStore(path) as reopened:
        assert reopened.cash() == Decimal("1000.00")
        assert reopened.load_settings().entry_price == Decimal("0.20")


def test_closed_paper_position_does_not_permanently_lock_reentry(tmp_path) -> None:
    now = datetime.now(UTC)
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        first = store.open_position(
            asset="BTC", ticker="BTC-1", side="up", quantity=Decimal("4"),
            entry_price=Decimal("0.20"), opened_at=now, day="2026-08-03",
        )
        assert first is not None
        assert store.close_position(first, Decimal("0.40"), now, "take_profit")
        second = store.open_position(
            asset="BTC", ticker="BTC-1", side="up", quantity=Decimal("4"),
            entry_price=Decimal("0.20"), opened_at=now, day="2026-08-03",
        )
        assert second is not None and second != first
        assert store.open_market_side("BTC-1", "up") is not None


def test_guard_cooldown_and_counters_persist(tmp_path) -> None:
    now = datetime.now(UTC)
    until = now + timedelta(seconds=10)
    path = tmp_path / "state.sqlite3"
    with DashboardStore(path) as store:
        store.set_cooldown("paper", "account", "BTC-1", "up", now, until)
        store.increment_guard_counter(
            "paper", "account", GuardReason.BELOW_FLOOR, now
        )
    with DashboardStore(path) as store:
        assert store.cooldown_until("paper", "account", "BTC-1", "up") == until
        assert store.guard_counters("paper", "account")["below_floor"] == 1


def test_live_exit_reconciliation_releases_lock_without_cooldown(tmp_path) -> None:
    now = datetime.now(UTC)
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        assert store.reserve_live_entry(
            "BTC-1", "up", "entry-1", Decimal("1"), "2026-08-03", now
        ) == Decimal("0")
        assert store.reserve_live_exit("BTC-1", "up", "exit-1", "take_profit", now)
        store.record_live_exit("exit-1", "order-exit-1")
        store.reconcile_live_positions([], now, 10)
        assert store.cooldown_until("live", "account", "BTC-1", "up") is None
        assert store.reserve_live_entry(
            "BTC-1", "up", "entry-2", Decimal("1"), "2026-08-03", now
        ) == Decimal("1")


def test_live_order_intent_preserves_human_side_and_role(tmp_path) -> None:
    now = datetime.now(UTC)
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        store.record_live_order_intent(
            "client-1", "BTC-1", "down", "entry", "entry",
            Decimal("0.62"), Decimal("0.78"), now,
        )
        store.record_live_order_result("client-1", "order-1")

        intent = store.live_order_intents()[0]
        assert intent["order_id"] == "order-1"
        assert intent["side"] == "down"
        assert intent["role"] == "entry"
        assert intent["quantity"] == "0.62"


def test_logical_live_trade_survives_restart_and_unlocks_only_after_flat(tmp_path) -> None:
    now = datetime.now(UTC)
    path = tmp_path / "state.sqlite3"
    with DashboardStore(path) as store:
        trade = store.create_live_trade(
            "trade-1", "BTC", "BTC-1", "up", Decimal("5"), now
        )
        assert trade is not None
        store.record_live_order_intent(
            "entry-1", "BTC-1", "up", "entry", "entry",
            Decimal("2"), Decimal("0.75"), now, "trade-1",
        )
        store.record_live_order_fill(
            "entry-1", Decimal("2"), Decimal("0.02"), Decimal("1.52")
        )
        assert store.lock_live_trade_exit("BTC-1", "up") == "trade-1"

    with DashboardStore(path) as store:
        assert store.active_live_trade("BTC-1")["phase"] == "exit_locked"
        store.reconcile_live_positions(
            [{"ticker": "BTC-1", "side": "up", "quantity": "1"}], now
        )
        assert store.active_live_trade("BTC-1") is not None
        store.reserve_live_exit("BTC-1", "up", "exit-1", "stop_loss", now)
        store.reconcile_live_positions([], now)
        assert store.active_live_trade("BTC-1") is None
        assert store.create_live_trade(
            "trade-same-quote", "BTC", "BTC-1", "up", Decimal("5"), now
        ) is None
        assert store.create_live_trade(
            "trade-fresh-quote", "BTC", "BTC-1", "up", Decimal("5"),
            now + timedelta(milliseconds=1),
        ) is not None


def test_live_fill_ledger_never_shrinks_when_api_pages_roll_forward(tmp_path) -> None:
    now = datetime.now(UTC)
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        store.record_live_order_intent(
            "entry-1", "BTC-1", "up", "entry", "entry",
            Decimal("2"), Decimal("0.20"), now, "trade-1",
        )
        store.record_live_order_result("entry-1", "order-1")
        first = {
            "fill_id": "fill-1", "order_id": "order-1", "ticker": "BTC-1",
            "side": "yes", "action": "buy", "count": "1",
            "yes_price": "0.20", "no_price": "0.80", "fee_cost": "0.01",
            "is_taker": True, "created_at": now.isoformat(),
        }
        second = {
            **first, "fill_id": "fill-2", "created_at": (now + timedelta(seconds=1)).isoformat(),
        }
        store.reconcile_live_account([first, second], [])
        assert store.live_order_intents()[0]["fill_count"] == "2"
        assert len(store.live_fills()) == 2

        # The API later returns only its newest page/fill. Durable fill IDs
        # preserve the earlier fill and prevent an accidental top-up.
        store.reconcile_live_account([second, second], [])
        assert store.live_order_intents()[0]["fill_count"] == "2"
        assert len(store.live_fills()) == 2


def test_live_reconcile_recovers_owned_order_id_before_external_check(tmp_path) -> None:
    now = datetime.now(UTC)
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        store.create_live_trade(
            "trade-1", "BTC", "BTC-1", "down", Decimal("1"), now
        )
        store.record_live_order_intent(
            "client-1", "BTC-1", "down", "entry", "entry",
            Decimal("1"), Decimal("0.70"), now, "trade-1",
        )
        fill = {
            "fill_id": "fill-1", "order_id": "order-1", "ticker": "BTC-1",
            "side": "no", "action": "sell", "count": "1",
            "yes_price": "0.30", "no_price": "0.70", "fee_cost": "0.01",
            "is_taker": True, "created_at": (now + timedelta(seconds=1)).isoformat(),
        }
        order = {
            "order_id": "order-1", "client_order_id": "client-1",
            "status": "executed", "remaining_count": "0",
        }

        assert store.reconcile_live_account([fill], [order]) == []
        intent = store.live_order_intents()[0]
        assert intent["order_id"] == "order-1"
        assert intent["fill_count"] == "1"


def test_external_fill_closes_trade_and_is_adopted_into_ledger(tmp_path) -> None:
    now = datetime.now(UTC)
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        store.create_live_trade(
            "trade-1", "BTC", "BTC-1", "down", Decimal("1"), now
        )
        external_fill = {
            "fill_id": "external-fill", "order_id": "external-order",
            "ticker": "BTC-1", "side": "yes", "action": "buy", "count": "1",
            "yes_price": "0.31", "no_price": "0.69", "fee_cost": "0.01",
            "is_taker": True, "created_at": (now + timedelta(seconds=2)).isoformat(),
        }
        external_order = {
            "order_id": "external-order", "client_order_id": "external-client",
            "status": "executed", "remaining_count": "0",
        }

        activity = store.reconcile_live_account(
            [external_fill], [external_order], positions=[]
        )
        assert len(activity) == 1
        assert activity[0]["kind"] == "external_exit"
        intent = store.live_order_intents()[0]
        assert intent["role"] == "exit"
        assert intent["reason"] == "external_exit"
        assert intent["fill_count"] == "1"
        assert intent["requested_price"] == "0.69"
        assert store.live_trades()[0]["phase"] == "flat"

        # The same account page is idempotent and cannot repeatedly disarm.
        assert store.reconcile_live_account(
            [external_fill], [external_order], positions=[]
        ) == []


def test_reset_requires_exact_confirmation(tmp_path) -> None:
    with DashboardStore(tmp_path / "state.sqlite3") as store:
        now = datetime.now(UTC)
        store.set_cooldown("paper", "account", "BTC-1", "up", now, now)
        store.increment_guard_counter("paper", "account", GuardReason.BELOW_FLOOR, now)
        assert store.reset_paper("wrong") is False
        assert store.reset_paper("RESET PAPER $1000") is True
        assert store.cash() == Decimal("1000.00")
        assert store.cooldown_until("paper", "account", "BTC-1", "up") is None
        assert store.guard_counters("paper", "account") == {}


def test_lab_fee_migration_is_additive_and_enforces_one_open_position(tmp_path) -> None:
    path = tmp_path / "old.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript(LEGACY_LAB_SCHEMA)
    connection.execute(
        "INSERT INTO lab_runs VALUES (1,'running','2026-08-03T00:00:00+00:00',10800,42,'[\"BTC\"]',1,0)"
    )
    connection.execute(
        """INSERT INTO lab_candidates(
        run_id,candidate_id,asset,entry_price,take_profit,stop_loss,min_seconds,side_policy,cash
        ) VALUES (1,'BTC-000','BTC','0.25','0.40','0.15',60,'both','998.40')"""
    )
    for side in ("up", "down"):
        connection.execute(
            """INSERT INTO lab_positions(
            run_id,candidate_id,ticker,side,quantity,entry_price,entry_cost
            ) VALUES (1,'BTC-000','BTC-1',?, '4','0.20','0.80')""",
            (side,),
        )
    connection.commit()
    connection.close()

    with DashboardStore(path) as store:
        run_columns = {
            row[1] for row in store.connection.execute("PRAGMA table_info(lab_runs)")
        }
        candidate_columns = {
            row[1] for row in store.connection.execute("PRAGMA table_info(lab_candidates)")
        }
        assert {
            "history_cycles_requested", "history_cycles_loaded",
            "history_events_total", "history_events_processed",
            "history_cutoff_event_id", "error_message",
        } <= run_columns
        assert "entry_min" in candidate_columns
        assert store.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='market_results'"
        ).fetchone() is not None
        assert store.connection.execute(
            """SELECT name FROM sqlite_master WHERE type='table'
            AND name='lab_candidate_eligible_cycles'"""
        ).fetchone() is not None
        run_columns = {
            row[1] for row in store.connection.execute("PRAGMA table_info(lab_runs)")
        }
        position_columns = {
            row[1] for row in store.connection.execute("PRAGMA table_info(lab_positions)")
        }
        position_indexes = {
            row[1] for row in store.connection.execute("PRAGMA index_list(lab_positions)")
        }
        assert {"fee_snapshot", "settlement_status"} <= run_columns
        assert {
            "id", "entry_fee", "exit_fee", "entry_outlay", "gross_proceeds",
            "net_proceeds", "entry_event_id", "exit_event_id", "close_reason",
        } <= position_columns
        assert "lab_positions_candidate_status" in position_indexes
        open_rows = store.connection.execute(
            "SELECT COUNT(*) FROM lab_positions WHERE status='open'"
        ).fetchone()[0]
        assert open_rows == 1
        legacy_run = store.connection.execute(
            "SELECT status FROM lab_runs WHERE id=1"
        ).fetchone()[0]
        assert legacy_run == "stopped"
        candidate = store.connection.execute(
            "SELECT cash,closed_trades,entry_count FROM lab_candidates WHERE run_id=1"
        ).fetchone()
        assert Decimal(candidate["cash"]) == Decimal("999.20")
        assert candidate["closed_trades"] == 1
        assert candidate["entry_count"] == 2
        with pytest.raises(sqlite3.IntegrityError):
            store.connection.execute(
                """INSERT INTO lab_positions(
                run_id,candidate_id,ticker,side,quantity,entry_price,entry_cost
                ) VALUES (1,'BTC-000','BTC-2','up','1','0.20','0.20')"""
            )
