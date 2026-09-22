from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from kaishi_bot.dashboard_models import DashboardSettings


class DashboardStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS dashboard_settings (
                id INTEGER PRIMARY KEY CHECK (id = 1), payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS paper_account (
                id INTEGER PRIMARY KEY CHECK (id = 1), cash TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS paper_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset TEXT NOT NULL, ticker TEXT NOT NULL, side TEXT NOT NULL,
                quantity TEXT NOT NULL, entry_price TEXT NOT NULL,
                entry_cost TEXT NOT NULL, opened_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open', exit_price TEXT,
                closed_at TEXT, close_reason TEXT, realized_pnl TEXT,
                entry_fee TEXT NOT NULL DEFAULT '0', exit_fee TEXT NOT NULL DEFAULT '0',
                entry_outlay TEXT, gross_proceeds TEXT, net_proceeds TEXT,
                liquidity_status TEXT NOT NULL DEFAULT 'liquidity_unverified',
                strategy_owner TEXT NOT NULL DEFAULT 'control_deck'
            );
            CREATE TABLE IF NOT EXISTS paper_daily_spend (
                day TEXT PRIMARY KEY, amount TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS dashboard_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, occurred_at TEXT NOT NULL,
                kind TEXT NOT NULL, message TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS quote_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, asset TEXT NOT NULL,
                ticker TEXT NOT NULL, observed_at TEXT NOT NULL,
                up_bid TEXT NOT NULL, up_ask TEXT NOT NULL,
                down_bid TEXT NOT NULL, down_ask TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS lab_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT, status TEXT NOT NULL,
                started_at TEXT NOT NULL, duration_seconds INTEGER NOT NULL,
                seed INTEGER NOT NULL, assets TEXT NOT NULL,
                candidate_count INTEGER NOT NULL, quote_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS lab_candidates (
                run_id INTEGER NOT NULL, candidate_id TEXT NOT NULL, asset TEXT NOT NULL,
                entry_price TEXT NOT NULL, take_profit TEXT NOT NULL, stop_loss TEXT NOT NULL,
                min_seconds INTEGER NOT NULL, side_policy TEXT NOT NULL,
                cash TEXT NOT NULL DEFAULT '1000.00', peak_equity TEXT NOT NULL DEFAULT '1000.00',
                max_drawdown TEXT NOT NULL DEFAULT '0', realized_pnl TEXT NOT NULL DEFAULT '0',
                closed_trades INTEGER NOT NULL DEFAULT 0, wins INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(run_id,candidate_id),
                FOREIGN KEY(run_id) REFERENCES lab_runs(id)
            );
            CREATE TABLE IF NOT EXISTS lab_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL, candidate_id TEXT NOT NULL,
                ticker TEXT NOT NULL, side TEXT NOT NULL, quantity TEXT NOT NULL,
                entry_price TEXT NOT NULL, entry_cost TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open', exit_price TEXT, pnl TEXT,
                entry_fee TEXT NOT NULL DEFAULT '0', exit_fee TEXT NOT NULL DEFAULT '0',
                entry_outlay TEXT, gross_proceeds TEXT, net_proceeds TEXT,
                entry_event_id INTEGER, exit_event_id INTEGER, close_reason TEXT
            );
            CREATE TABLE IF NOT EXISTS lab_daily_spend (
                run_id INTEGER NOT NULL, candidate_id TEXT NOT NULL, day TEXT NOT NULL,
                amount TEXT NOT NULL, PRIMARY KEY(run_id,candidate_id,day)
            );
            CREATE TABLE IF NOT EXISTS lab_seen_events (
                run_id INTEGER NOT NULL, quote_event_id INTEGER NOT NULL,
                PRIMARY KEY(run_id,quote_event_id)
            );
            CREATE TABLE IF NOT EXISTS lab_candidate_eligible_cycles (
                run_id INTEGER NOT NULL, candidate_id TEXT NOT NULL,
                ticker TEXT NOT NULL,
                PRIMARY KEY(run_id,candidate_id,ticker)
            );
            CREATE TABLE IF NOT EXISTS lab_settlement_events (
                run_id INTEGER NOT NULL, ticker TEXT NOT NULL,
                winning_side TEXT NOT NULL, event_time TEXT NOT NULL,
                PRIMARY KEY(run_id,ticker)
            );
            CREATE TABLE IF NOT EXISTS market_results (
                ticker TEXT PRIMARY KEY, winning_side TEXT NOT NULL,
                event_time TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS live_entry_locks (
                ticker TEXT NOT NULL, side TEXT NOT NULL, client_order_id TEXT NOT NULL UNIQUE,
                order_id TEXT, entry_cost TEXT NOT NULL, placed_at TEXT NOT NULL,
                PRIMARY KEY(ticker,side)
            );
            CREATE TABLE IF NOT EXISTS live_daily_spend (
                day TEXT PRIMARY KEY, amount TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS live_exit_locks (
                ticker TEXT NOT NULL, side TEXT NOT NULL, client_order_id TEXT NOT NULL UNIQUE,
                order_id TEXT, reason TEXT NOT NULL, placed_at TEXT NOT NULL,
                PRIMARY KEY(ticker,side)
            );
            CREATE TABLE IF NOT EXISTS live_order_intents (
                client_order_id TEXT PRIMARY KEY, order_id TEXT UNIQUE,
                ticker TEXT NOT NULL, side TEXT NOT NULL, role TEXT NOT NULL,
                reason TEXT NOT NULL, quantity TEXT NOT NULL,
                requested_price TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS live_fills (
                fill_id TEXT PRIMARY KEY, order_id TEXT NOT NULL,
                ticker TEXT NOT NULL, side TEXT NOT NULL, action TEXT NOT NULL,
                count TEXT NOT NULL, yes_price TEXT NOT NULL, no_price TEXT NOT NULL,
                fee_cost TEXT NOT NULL, is_taker INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS live_trades (
                trade_id TEXT PRIMARY KEY, asset TEXT NOT NULL,
                ticker TEXT NOT NULL, side TEXT NOT NULL,
                target_budget TEXT NOT NULL, phase TEXT NOT NULL,
                exit_locked INTEGER NOT NULL DEFAULT 0,
                opened_at TEXT NOT NULL, closed_at TEXT,
                last_entry_quote_at TEXT,
                UNIQUE(ticker, trade_id)
            );
            CREATE TABLE IF NOT EXISTS entry_cooldowns (
                scope TEXT NOT NULL, owner_id TEXT NOT NULL, ticker TEXT NOT NULL,
                side TEXT NOT NULL, closed_at TEXT NOT NULL, cooldown_until TEXT NOT NULL,
                PRIMARY KEY(scope,owner_id,ticker,side)
            );
            CREATE TABLE IF NOT EXISTS entry_guard_counters (
                scope TEXT NOT NULL, owner_id TEXT NOT NULL, reason TEXT NOT NULL,
                count INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL,
                PRIMARY KEY(scope,owner_id,reason)
            );
            """
        )
        quote_columns = {
            row[1] for row in self.connection.execute("PRAGMA table_info(quote_events)")
        }
        if "close_time" not in quote_columns:
            self.connection.execute("ALTER TABLE quote_events ADD COLUMN close_time TEXT")
        self._migrate_paper_schema()
        self._migrate_lab_schema()
        self._ensure_column("live_exit_locks", "last_error", "TEXT")
        self._ensure_column("live_exit_locks", "attempts", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("live_exit_locks", "order_kind", "TEXT NOT NULL DEFAULT 'legacy'")
        self._ensure_column("live_order_intents", "trade_id", "TEXT")
        self._ensure_column("live_order_intents", "fill_count", "TEXT NOT NULL DEFAULT '0'")
        self._ensure_column("live_order_intents", "fee_estimate", "TEXT NOT NULL DEFAULT '0'")
        self._ensure_column("live_order_intents", "accounted_cost", "TEXT NOT NULL DEFAULT '0'")
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO paper_account(id, cash) VALUES (1, '1000.00')"
            )
            self.connection.execute(
                "INSERT OR IGNORE INTO dashboard_settings(id, payload) VALUES (1, ?)",
                (DashboardSettings().model_dump_json(),),
            )
            # Actual prices and fills remain sourced from Kalshi. This migration
            # only preserves the user-facing meaning of older entry orders.
            self.connection.execute(
                """INSERT OR IGNORE INTO live_order_intents(
                client_order_id,order_id,ticker,side,role,reason,quantity,
                requested_price,created_at
                ) SELECT client_order_id,order_id,ticker,side,'entry','entry','0','0',placed_at
                FROM live_entry_locks WHERE order_id IS NOT NULL"""
            )

    def _ensure_column(self, table: str, name: str, declaration: str) -> None:
        columns = {
            row[1] for row in self.connection.execute(f"PRAGMA table_info({table})")
        }
        if name not in columns:
            self.connection.execute(
                f"ALTER TABLE {table} ADD COLUMN {name} {declaration}"
            )

    def _migrate_paper_schema(self) -> None:
        table_sql = self.connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='paper_positions'"
        ).fetchone()[0]
        with self.connection:
            if "UNIQUE(ticker, side)" in str(table_sql):
                self.connection.execute(
                    "ALTER TABLE paper_positions RENAME TO paper_positions_legacy"
                )
                self.connection.execute(
                    """CREATE TABLE paper_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    asset TEXT NOT NULL, ticker TEXT NOT NULL, side TEXT NOT NULL,
                    quantity TEXT NOT NULL, entry_price TEXT NOT NULL,
                    entry_cost TEXT NOT NULL, opened_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open', exit_price TEXT,
                    closed_at TEXT, close_reason TEXT, realized_pnl TEXT,
                    entry_fee TEXT NOT NULL DEFAULT '0', exit_fee TEXT NOT NULL DEFAULT '0',
                    entry_outlay TEXT, gross_proceeds TEXT, net_proceeds TEXT,
                    liquidity_status TEXT NOT NULL DEFAULT 'liquidity_unverified',
                    strategy_owner TEXT NOT NULL DEFAULT 'control_deck'
                    )"""
                )
                self.connection.execute(
                    """INSERT INTO paper_positions(
                    id,asset,ticker,side,quantity,entry_price,entry_cost,opened_at,
                    status,exit_price,closed_at,close_reason,realized_pnl,
                    entry_fee,exit_fee,entry_outlay,gross_proceeds,net_proceeds,
                    liquidity_status,strategy_owner
                    ) SELECT id,asset,ticker,side,quantity,entry_price,entry_cost,opened_at,
                    status,exit_price,closed_at,close_reason,realized_pnl,
                    '0','0',entry_cost,
                    CASE WHEN exit_price IS NULL THEN NULL ELSE CAST(quantity AS REAL) * CAST(exit_price AS REAL) END,
                    CASE WHEN exit_price IS NULL THEN NULL ELSE CAST(quantity AS REAL) * CAST(exit_price AS REAL) END,
                    'liquidity_unverified','control_deck' FROM paper_positions_legacy"""
                )
                self.connection.execute("DROP TABLE paper_positions_legacy")
            else:
                for name, declaration in (
                    ("entry_fee", "TEXT NOT NULL DEFAULT '0'"),
                    ("exit_fee", "TEXT NOT NULL DEFAULT '0'"),
                    ("entry_outlay", "TEXT"),
                    ("gross_proceeds", "TEXT"),
                    ("net_proceeds", "TEXT"),
                    ("liquidity_status", "TEXT NOT NULL DEFAULT 'liquidity_unverified'"),
                    ("strategy_owner", "TEXT NOT NULL DEFAULT 'control_deck'"),
                ):
                    self._ensure_column("paper_positions", name, declaration)
            self.connection.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS paper_one_open_position_per_market_side
                ON paper_positions(ticker,side) WHERE status='open'"""
            )

    def _migrate_lab_schema(self) -> None:
        with self.connection:
            self._ensure_column("lab_runs", "fee_snapshot", "TEXT")
            self._ensure_column("lab_runs", "guard_snapshot", "TEXT")
            self._ensure_column(
                "lab_runs", "settlement_status", "TEXT NOT NULL DEFAULT 'ready'"
            )
            for name, declaration in (
                ("backfill_engine", "TEXT NOT NULL DEFAULT 'legacy'"),
                ("history_cycles_requested", "INTEGER NOT NULL DEFAULT 0"),
                ("history_cycles_loaded", "INTEGER NOT NULL DEFAULT 0"),
                ("history_events_total", "INTEGER NOT NULL DEFAULT 0"),
                ("history_events_processed", "INTEGER NOT NULL DEFAULT 0"),
                ("history_cutoff_event_id", "INTEGER"),
                ("error_message", "TEXT"),
            ):
                self._ensure_column("lab_runs", name, declaration)
            self.connection.execute(
                """UPDATE lab_runs SET status='stopped'
                WHERE status='running' AND fee_snapshot IS NULL"""
            )
            for name, declaration in (
                ("entry_min", "TEXT"),
                ("total_fees", "TEXT NOT NULL DEFAULT '0'"),
                ("entry_count", "INTEGER NOT NULL DEFAULT 0"),
                ("tp_count", "INTEGER NOT NULL DEFAULT 0"),
                ("sl_count", "INTEGER NOT NULL DEFAULT 0"),
                ("settlement_count", "INTEGER NOT NULL DEFAULT 0"),
                ("skipped_open", "INTEGER NOT NULL DEFAULT 0"),
                ("blocked_daily_cap", "INTEGER NOT NULL DEFAULT 0"),
                ("last_closed_ticker", "TEXT"),
                ("last_closed_side", "TEXT"),
                ("last_closed_at", "TEXT"),
                ("cooldown_until", "TEXT"),
                ("guard_below_floor", "INTEGER NOT NULL DEFAULT 0"),
                ("guard_bid_in_sl_buffer", "INTEGER NOT NULL DEFAULT 0"),
                ("guard_spread_too_wide", "INTEGER NOT NULL DEFAULT 0"),
                ("guard_confirmation_pending", "INTEGER NOT NULL DEFAULT 0"),
                ("guard_reward_risk_too_low", "INTEGER NOT NULL DEFAULT 0"),
                ("guard_cooldown_active", "INTEGER NOT NULL DEFAULT 0"),
            ):
                self._ensure_column("lab_candidates", name, declaration)

            self.connection.execute(
                """INSERT OR IGNORE INTO market_results(ticker,winning_side,event_time)
                SELECT ticker,MIN(winning_side),MIN(event_time)
                FROM lab_settlement_events GROUP BY ticker"""
            )

            position_columns = {
                row[1]
                for row in self.connection.execute("PRAGMA table_info(lab_positions)")
            }
            if "id" not in position_columns:
                self.connection.execute(
                    "ALTER TABLE lab_positions RENAME TO lab_positions_legacy"
                )
                self.connection.execute(
                    """CREATE TABLE lab_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL, candidate_id TEXT NOT NULL,
                    ticker TEXT NOT NULL, side TEXT NOT NULL, quantity TEXT NOT NULL,
                    entry_price TEXT NOT NULL, entry_cost TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open', exit_price TEXT, pnl TEXT,
                    entry_fee TEXT NOT NULL DEFAULT '0', exit_fee TEXT NOT NULL DEFAULT '0',
                    entry_outlay TEXT, gross_proceeds TEXT, net_proceeds TEXT,
                    entry_event_id INTEGER, exit_event_id INTEGER, close_reason TEXT,
                    liquidity_status TEXT NOT NULL DEFAULT 'liquidity_unverified'
                    )"""
                )
                self.connection.execute(
                    """INSERT INTO lab_positions(
                    run_id,candidate_id,ticker,side,quantity,entry_price,entry_cost,
                    status,exit_price,pnl,entry_fee,exit_fee,entry_outlay
                    ) SELECT run_id,candidate_id,ticker,side,quantity,entry_price,
                    entry_cost,status,exit_price,pnl,'0','0',entry_cost
                    FROM lab_positions_legacy"""
                )
                extras = list(self.connection.execute(
                    """SELECT p.run_id,p.candidate_id,p.entry_cost
                    FROM lab_positions p WHERE p.status='open' AND p.id NOT IN (
                        SELECT MIN(id) FROM lab_positions WHERE status='open'
                        GROUP BY run_id,candidate_id
                    )"""
                ))
                restored: dict[tuple[int, str], tuple[Decimal, int]] = {}
                for extra in extras:
                    key = (int(extra["run_id"]), str(extra["candidate_id"]))
                    amount, count = restored.get(key, (Decimal("0"), 0))
                    restored[key] = (amount + Decimal(extra["entry_cost"]), count + 1)
                for (run_id, candidate_id), (amount, count) in restored.items():
                    candidate = self.connection.execute(
                        """SELECT cash,closed_trades FROM lab_candidates
                        WHERE run_id=? AND candidate_id=?""", (run_id, candidate_id),
                    ).fetchone()
                    if candidate is not None:
                        self.connection.execute(
                            """UPDATE lab_candidates SET cash=?,closed_trades=?,
                            entry_count=(SELECT COUNT(*) FROM lab_positions
                                WHERE run_id=? AND candidate_id=?)
                            WHERE run_id=? AND candidate_id=?""",
                            (str(Decimal(candidate["cash"]) + amount),
                             int(candidate["closed_trades"]) + count,
                             run_id, candidate_id, run_id, candidate_id),
                        )
                self.connection.execute("DROP TABLE lab_positions_legacy")
            else:
                for name, declaration in (
                    ("entry_fee", "TEXT NOT NULL DEFAULT '0'"),
                    ("exit_fee", "TEXT NOT NULL DEFAULT '0'"),
                    ("entry_outlay", "TEXT"),
                    ("gross_proceeds", "TEXT"),
                    ("net_proceeds", "TEXT"),
                    ("entry_event_id", "INTEGER"),
                    ("exit_event_id", "INTEGER"),
                    ("close_reason", "TEXT"),
                    ("liquidity_status", "TEXT NOT NULL DEFAULT 'liquidity_unverified'"),
                ):
                    self._ensure_column("lab_positions", name, declaration)

            self.connection.execute(
                """UPDATE lab_positions SET status='closed',
                exit_price=COALESCE(exit_price,entry_price),pnl=COALESCE(pnl,'0'),
                exit_fee=COALESCE(exit_fee,'0'),
                gross_proceeds=COALESCE(gross_proceeds,entry_cost),
                net_proceeds=COALESCE(net_proceeds,entry_cost),
                close_reason=COALESCE(close_reason,'migration_flattened')
                WHERE status='open' AND id NOT IN (
                    SELECT MIN(id) FROM lab_positions WHERE status='open'
                    GROUP BY run_id,candidate_id
                )"""
            )
            self.connection.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS lab_one_open_position_per_candidate
                ON lab_positions(run_id,candidate_id) WHERE status='open'"""
            )
            self.connection.execute(
                """CREATE INDEX IF NOT EXISTS lab_positions_candidate_status
                ON lab_positions(run_id,candidate_id,status)"""
            )
            self.connection.execute(
                """CREATE INDEX IF NOT EXISTS lab_positions_run_status_ticker
                ON lab_positions(run_id,status,ticker)"""
            )

    def __enter__(self) -> "DashboardStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def load_settings(self) -> DashboardSettings:
        row = self.connection.execute(
            "SELECT payload FROM dashboard_settings WHERE id = 1"
        ).fetchone()
        assert row is not None
        return DashboardSettings.model_validate_json(row[0])

    def save_settings(self, settings: DashboardSettings) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE dashboard_settings SET payload = ? WHERE id = 1",
                (settings.model_dump_json(),),
            )

    def cash(self) -> Decimal:
        row = self.connection.execute(
            "SELECT cash FROM paper_account WHERE id = 1"
        ).fetchone()
        assert row is not None
        return Decimal(row[0])

    def daily_spend(self, day: str) -> Decimal:
        row = self.connection.execute(
            "SELECT amount FROM paper_daily_spend WHERE day = ?", (day,)
        ).fetchone()
        return Decimal(row[0]) if row else Decimal("0")

    def has_market_side_lock(self, ticker: str, side: str) -> bool:
        return self.open_market_side(ticker, side) is not None

    def open_market_side(self, ticker: str, side: str) -> dict[str, object] | None:
        row = self.connection.execute(
            """SELECT * FROM paper_positions
            WHERE ticker=? AND side=? AND status='open' ORDER BY id LIMIT 1""",
            (ticker, side),
        ).fetchone()
        return self._position(row) if row is not None else None

    def set_cooldown(
        self, scope: str, owner_id: str, ticker: str, side: str,
        closed_at: datetime, cooldown_until: datetime,
    ) -> None:
        with self.connection:
            self.connection.execute(
                """INSERT INTO entry_cooldowns(
                scope,owner_id,ticker,side,closed_at,cooldown_until
                ) VALUES (?,?,?,?,?,?) ON CONFLICT(scope,owner_id,ticker,side)
                DO UPDATE SET closed_at=excluded.closed_at,
                cooldown_until=excluded.cooldown_until""",
                (scope, owner_id, ticker, side, closed_at.isoformat(), cooldown_until.isoformat()),
            )

    def cooldown_until(
        self, scope: str, owner_id: str, ticker: str, side: str,
    ) -> datetime | None:
        row = self.connection.execute(
            """SELECT cooldown_until FROM entry_cooldowns
            WHERE scope=? AND owner_id=? AND ticker=? AND side=?""",
            (scope, owner_id, ticker, side),
        ).fetchone()
        return datetime.fromisoformat(row[0]) if row else None

    def increment_guard_counter(
        self, scope: str, owner_id: str, reason: object, observed_at: datetime,
    ) -> None:
        value = str(getattr(reason, "value", reason))
        with self.connection:
            self.connection.execute(
                """INSERT INTO entry_guard_counters(scope,owner_id,reason,count,updated_at)
                VALUES (?,?,?,1,?) ON CONFLICT(scope,owner_id,reason)
                DO UPDATE SET count=count+1,updated_at=excluded.updated_at""",
                (scope, owner_id, value, observed_at.isoformat()),
            )

    def guard_counters(self, scope: str, owner_id: str) -> dict[str, int]:
        return {
            str(row["reason"]): int(row["count"])
            for row in self.connection.execute(
                """SELECT reason,count FROM entry_guard_counters
                WHERE scope=? AND owner_id=?""", (scope, owner_id)
            )
        }

    def open_position(
        self, *, asset: str, ticker: str, side: str, quantity: Decimal,
        entry_price: Decimal, opened_at: datetime, day: str,
        entry_fee: Decimal = Decimal("0"),
        liquidity_status: str = "liquidity_unverified",
        strategy_owner: str = "control_deck",
    ) -> int | None:
        cost = quantity * entry_price
        outlay = cost + entry_fee
        try:
            with self.connection:
                cash = self.cash()
                if cash < outlay:
                    return None
                cursor = self.connection.execute(
                    """INSERT INTO paper_positions(
                    asset,ticker,side,quantity,entry_price,entry_cost,opened_at,
                    entry_fee,entry_outlay,liquidity_status
                    ,strategy_owner) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (asset, ticker, side, str(quantity), str(entry_price), str(cost),
                     opened_at.isoformat(), str(entry_fee), str(outlay), liquidity_status,
                     strategy_owner),
                )
                self.connection.execute(
                    "UPDATE paper_account SET cash = ? WHERE id = 1",
                    (str(cash - outlay),),
                )
                current = self.daily_spend(day)
                self.connection.execute(
                    """INSERT INTO paper_daily_spend(day,amount) VALUES (?,?)
                    ON CONFLICT(day) DO UPDATE SET amount=excluded.amount""",
                    (day, str(current + outlay)),
                )
                self._event(opened_at, "entry", f"{asset} {side.upper()} ×{quantity} @ {entry_price}")
                return int(cursor.lastrowid)
        except sqlite3.IntegrityError:
            return None

    def close_position(
        self, position_id: int, exit_price: Decimal, closed_at: datetime, reason: str,
        exit_fee: Decimal = Decimal("0"),
    ) -> bool:
        with self.connection:
            row = self.connection.execute(
                "SELECT * FROM paper_positions WHERE id = ? AND status = 'open'",
                (position_id,),
            ).fetchone()
            if row is None:
                return False
            quantity = Decimal(row["quantity"])
            gross = quantity * exit_price
            proceeds = gross - exit_fee
            entry_outlay = Decimal(row["entry_outlay"] or row["entry_cost"])
            pnl = proceeds - entry_outlay
            self.connection.execute(
                """UPDATE paper_positions SET status='closed', exit_price=?,
                closed_at=?, close_reason=?, realized_pnl=?,exit_fee=?,
                gross_proceeds=?,net_proceeds=? WHERE id=?""",
                (str(exit_price), closed_at.isoformat(), reason, str(pnl), str(exit_fee),
                 str(gross), str(proceeds), position_id),
            )
            self.connection.execute(
                "UPDATE paper_account SET cash = ? WHERE id = 1",
                (str(self.cash() + proceeds),),
            )
            self._event(closed_at, reason, f"{row['asset']} {row['side'].upper()} ×{quantity} @ {exit_price}")
            return True

    def _event(self, at: datetime, kind: str, message: str) -> None:
        self.connection.execute(
            "INSERT INTO dashboard_events(occurred_at,kind,message) VALUES (?,?,?)",
            (at.isoformat(), kind, message),
        )

    @staticmethod
    def _position(row: sqlite3.Row) -> dict[str, object]:
        result = dict(row)
        for key in (
            "quantity", "entry_price", "entry_cost", "exit_price", "realized_pnl",
            "entry_fee", "exit_fee", "entry_outlay", "gross_proceeds", "net_proceeds",
        ):
            if result.get(key) is not None:
                result[key] = Decimal(str(result[key]))
        return result

    def open_positions(self) -> list[dict[str, object]]:
        rows = self.connection.execute(
            "SELECT * FROM paper_positions WHERE status='open' ORDER BY id"
        )
        return [self._position(row) for row in rows]

    def closed_positions(self) -> list[dict[str, object]]:
        rows = self.connection.execute(
            "SELECT * FROM paper_positions WHERE status='closed' ORDER BY id"
        )
        return [self._position(row) for row in rows]

    def recent_events(self, limit: int = 100) -> list[dict[str, object]]:
        rows = self.connection.execute(
            "SELECT * FROM dashboard_events ORDER BY id DESC LIMIT ?", (limit,)
        )
        return [dict(row) for row in rows]

    def record_quote(
        self, asset: str, ticker: str, quote: object,
        close_time: datetime | None = None,
    ) -> int:
        with self.connection:
            cursor = self.connection.execute(
                """INSERT INTO quote_events(
                asset,ticker,observed_at,up_bid,up_ask,down_bid,down_ask,close_time
                ) VALUES (?,?,?,?,?,?,?,?)""",
                (asset, ticker, quote.observed_at.isoformat(), str(quote.up_bid), str(quote.up_ask),
                 str(quote.down_bid), str(quote.down_ask),
                 close_time.isoformat() if close_time else None),
            )
        return int(cursor.lastrowid)

    def save_market_result(
        self, ticker: str, winning_side: str, event_time: datetime
    ) -> None:
        if winning_side not in {"up", "down"}:
            raise ValueError("winning_side must be up or down")
        with self.connection:
            self.connection.execute(
                """INSERT INTO market_results(ticker,winning_side,event_time)
                VALUES (?,?,?) ON CONFLICT(ticker) DO UPDATE SET
                winning_side=excluded.winning_side,event_time=excluded.event_time""",
                (ticker, winning_side, event_time.isoformat()),
            )

    def market_result(self, ticker: str) -> dict[str, object] | None:
        row = self.connection.execute(
            "SELECT * FROM market_results WHERE ticker=?", (ticker,)
        ).fetchone()
        return dict(row) if row is not None else None

    def reset_paper(self, confirmation: str) -> bool:
        if confirmation != "RESET PAPER $1000":
            return False
        with self.connection:
            self.connection.execute("DELETE FROM paper_positions")
            self.connection.execute("DELETE FROM paper_daily_spend")
            self.connection.execute("DELETE FROM dashboard_events")
            self.connection.execute("DELETE FROM entry_cooldowns WHERE scope='paper'")
            self.connection.execute("DELETE FROM entry_guard_counters WHERE scope='paper'")
            self.connection.execute("UPDATE paper_account SET cash='1000.00' WHERE id=1")
        return True

    def live_daily_spend(self, day: str) -> Decimal:
        row = self.connection.execute(
            "SELECT amount FROM live_daily_spend WHERE day=?", (day,)
        ).fetchone()
        return Decimal(row[0]) if row else Decimal("0")

    def reserve_live_entry(
        self, ticker: str, side: str, client_order_id: str,
        entry_cost: Decimal, day: str, placed_at: datetime,
    ) -> Decimal | None:
        previous = self.live_daily_spend(day)
        try:
            with self.connection:
                self.connection.execute(
                    """INSERT INTO live_entry_locks(
                    ticker,side,client_order_id,entry_cost,placed_at
                    ) VALUES (?,?,?,?,?)""",
                    (ticker, side, client_order_id, str(entry_cost), placed_at.isoformat()),
                )
                self.connection.execute(
                    """INSERT INTO live_daily_spend(day,amount) VALUES (?,?)
                    ON CONFLICT(day) DO UPDATE SET amount=excluded.amount""",
                    (day, str(previous + entry_cost)),
                )
        except sqlite3.IntegrityError:
            return None
        return previous

    def active_live_trade(self, ticker: str) -> dict[str, object] | None:
        row = self.connection.execute(
            """SELECT * FROM live_trades
            WHERE ticker=? AND phase!='flat' ORDER BY opened_at DESC LIMIT 1""",
            (ticker,),
        ).fetchone()
        return dict(row) if row is not None else None

    def create_live_trade(
        self, trade_id: str, asset: str, ticker: str, side: str,
        target_budget: Decimal, opened_at: datetime,
    ) -> dict[str, object] | None:
        if self.active_live_trade(ticker) is not None:
            return None
        latest = self.connection.execute(
            """SELECT closed_at FROM live_trades WHERE ticker=? AND closed_at IS NOT NULL
            ORDER BY closed_at DESC LIMIT 1""", (ticker,)
        ).fetchone()
        if latest is not None and opened_at <= datetime.fromisoformat(str(latest["closed_at"])):
            return None
        with self.connection:
            self.connection.execute(
                """INSERT INTO live_trades(
                trade_id,asset,ticker,side,target_budget,phase,opened_at,last_entry_quote_at
                ) VALUES (?,?,?,?,?,'accumulating',?,?)""",
                (trade_id, asset, ticker, side, str(target_budget),
                 opened_at.isoformat(), opened_at.isoformat()),
            )
        return self.active_live_trade(ticker)

    def lock_live_trade_exit(self, ticker: str, side: str) -> str | None:
        trade = self.active_live_trade(ticker)
        if trade is None or str(trade["side"]) != side:
            return None
        with self.connection:
            self.connection.execute(
                """UPDATE live_trades SET phase='exit_locked',exit_locked=1
                WHERE trade_id=?""", (trade["trade_id"],),
            )
        return str(trade["trade_id"])

    def close_live_trade(self, ticker: str, side: str, closed_at: datetime) -> None:
        with self.connection:
            self.connection.execute(
                """UPDATE live_trades SET phase='flat',closed_at=?
                WHERE ticker=? AND side=? AND phase='exit_locked'""",
                (closed_at.isoformat(), ticker, side),
            )

    def live_trades(self, limit: int = 1000) -> list[dict[str, object]]:
        return [dict(row) for row in self.connection.execute(
            "SELECT * FROM live_trades ORDER BY opened_at DESC LIMIT ?", (limit,)
        )]

    def live_trade_entry_totals(self, trade_id: str) -> tuple[Decimal, Decimal]:
        rows = self.connection.execute(
            """SELECT fill_count,requested_price,fee_estimate
            FROM live_order_intents WHERE trade_id=? AND role='entry'""",
            (trade_id,),
        )
        quantity = Decimal("0")
        outlay = Decimal("0")
        for row in rows:
            filled = Decimal(str(row["fill_count"]))
            quantity += filled
            outlay += filled * Decimal(str(row["requested_price"])) + Decimal(
                str(row["fee_estimate"])
            )
        return quantity, outlay

    def set_live_trade_phase(self, trade_id: str, phase: str) -> None:
        if phase not in {"accumulating", "holding", "exit_locked", "flat"}:
            raise ValueError("invalid live trade phase")
        with self.connection:
            self.connection.execute(
                "UPDATE live_trades SET phase=? WHERE trade_id=?", (phase, trade_id)
            )

    def abandon_empty_live_trade(self, trade_id: str) -> None:
        quantity, _ = self.live_trade_entry_totals(trade_id)
        if quantity > 0:
            return
        with self.connection:
            self.connection.execute(
                "DELETE FROM live_trades WHERE trade_id=? AND exit_locked=0", (trade_id,)
            )

    def record_live_order(self, client_order_id: str, order_id: str) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE live_entry_locks SET order_id=? WHERE client_order_id=?",
                (order_id, client_order_id),
            )

    def record_live_order_intent(
        self, client_order_id: str, ticker: str, side: str, role: str,
        reason: str, quantity: Decimal, requested_price: Decimal,
        created_at: datetime, trade_id: str | None = None,
    ) -> None:
        with self.connection:
            self.connection.execute(
                """INSERT INTO live_order_intents(
                client_order_id,ticker,side,role,reason,quantity,
                requested_price,created_at,trade_id
                ) VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(client_order_id) DO UPDATE SET
                ticker=excluded.ticker,side=excluded.side,role=excluded.role,
                reason=excluded.reason,quantity=excluded.quantity,
                requested_price=excluded.requested_price,trade_id=excluded.trade_id""",
                (
                    client_order_id, ticker, side, role, reason, str(quantity),
                    str(requested_price), created_at.isoformat(), trade_id,
                ),
            )

    def record_live_order_fill(
        self, client_order_id: str, fill_count: Decimal, fee_estimate: Decimal,
        accounted_cost: Decimal,
    ) -> None:
        with self.connection:
            self.connection.execute(
                """UPDATE live_order_intents SET fill_count=?,fee_estimate=?,accounted_cost=?
                WHERE client_order_id=?""",
                (str(fill_count), str(fee_estimate), str(accounted_cost), client_order_id),
            )

    def record_live_order_result(self, client_order_id: str, order_id: str) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE live_order_intents SET order_id=? WHERE client_order_id=?",
                (order_id, client_order_id),
            )

    def live_order_intents(self, limit: int = 1000) -> list[dict[str, object]]:
        return [dict(row) for row in self.connection.execute(
            """SELECT * FROM live_order_intents
            ORDER BY created_at DESC LIMIT ?""", (limit,)
        )]

    def reconcile_live_account(
        self, fills: list[dict[str, object]], orders: list[dict[str, object]],
        positions: list[dict[str, object]] | None = None,
        settlements: list[dict[str, object]] | None = None,
    ) -> list[dict[str, object]]:
        """Reconcile Kalshi activity and report fills not owned by this runtime.

        An order response and the account read model can race.  Recover our
        order ID from its durable client ID before classifying fills, otherwise
        a fast fill from this process could be mistaken for external activity.
        """
        orders_by_id = {
            str(item.get("order_id", "")): item
            for item in orders if str(item.get("order_id", ""))
        }
        with self.connection:
            for order_id, order in orders_by_id.items():
                client_order_id = str(order.get("client_order_id", ""))
                if not client_order_id:
                    continue
                self.connection.execute(
                    """UPDATE live_order_intents SET order_id=?
                    WHERE client_order_id=? AND order_id IS NULL""",
                    (order_id, client_order_id),
                )
        owned_order_ids = {
            str(row[0]) for row in self.connection.execute(
                """SELECT order_id FROM live_order_intents
                WHERE order_id IS NOT NULL AND order_id!=''"""
            )
        }
        new_unowned: list[dict[str, object]] = []
        with self.connection:
            for fill in fills:
                fill_id = str(fill.get("fill_id", ""))
                if not fill_id:
                    continue
                cursor = self.connection.execute(
                    """INSERT OR IGNORE INTO live_fills(
                    fill_id,order_id,ticker,side,action,count,yes_price,no_price,
                    fee_cost,is_taker,created_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        fill_id, str(fill.get("order_id", "")),
                        str(fill.get("ticker", "")), str(fill.get("side", "")),
                        str(fill.get("action", "")), str(fill.get("count", "0")),
                        str(fill.get("yes_price", "0")), str(fill.get("no_price", "0")),
                        str(fill.get("fee_cost", "0")), int(bool(fill.get("is_taker"))),
                        str(fill.get("created_at", "")),
                    ),
                )
                order_id = str(fill.get("order_id", ""))
                if cursor.rowcount == 1 and order_id not in owned_order_ids:
                    new_unowned.append(dict(fill))

        external_activity: list[dict[str, object]] = []
        for fill in new_unowned:
            ticker = str(fill.get("ticker", ""))
            order_id = str(fill.get("order_id", ""))
            trade = self.active_live_trade(ticker)
            if trade is None or not order_id:
                continue
            created_at = str(fill.get("created_at", ""))
            try:
                fill_time = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                opened_at = datetime.fromisoformat(str(trade["opened_at"]))
            except ValueError:
                continue
            if fill_time < opened_at:
                continue
            fill_side = str(fill.get("side", "")).lower()
            action = str(fill.get("action", "")).lower()
            direction = (
                "up" if (fill_side, action) == ("yes", "buy")
                else "down" if (fill_side, action) == ("no", "sell")
                else "unknown"
            )
            trade_side = str(trade["side"])
            relation = "external_exit" if (
                direction in {"up", "down"} and direction != trade_side
            ) else "external_fill"
            order = orders_by_id.get(order_id, {})
            external_activity.append({
                "ticker": ticker,
                "order_id": order_id,
                "client_order_id": str(order.get("client_order_id", "")),
                "trade_id": str(trade["trade_id"]),
                "trade_side": trade_side,
                "direction": direction,
                "kind": relation,
                "created_at": created_at,
            })
            if relation != "external_exit" or order_id in owned_order_ids:
                continue
            logical_price_field = "yes_price" if trade_side == "up" else "no_price"
            client_order_id = f"external:{order_id}"
            with self.connection:
                self.connection.execute(
                    """INSERT OR IGNORE INTO live_order_intents(
                    client_order_id,order_id,ticker,side,role,reason,quantity,
                    requested_price,created_at,trade_id
                    ) VALUES (?,?,?,?,?,'external_exit',?,?,?,?)""",
                    (
                        client_order_id, order_id, ticker, trade_side, "exit",
                        str(fill.get("count", "0")),
                        str(fill.get(logical_price_field, "0")),
                        created_at, str(trade["trade_id"]),
                    ),
                )
                self.connection.execute(
                    """UPDATE live_trades SET phase='exit_locked',exit_locked=1
                    WHERE trade_id=? AND phase!='flat'""",
                    (str(trade["trade_id"]),),
                )
            owned_order_ids.add(order_id)
        by_order: dict[str, list[dict[str, object]]] = {}
        for fill in fills:
            order_id = str(fill.get("order_id", ""))
            if order_id:
                by_order[order_id] = [dict(row) for row in self.connection.execute(
                    "SELECT * FROM live_fills WHERE order_id=? ORDER BY created_at,fill_id",
                    (order_id,),
                )]
        terminal_orders = {
            str(item.get("order_id", "")) for item in orders
            if str(item.get("status", "")).lower() not in {"resting", "pending", "open"}
            or Decimal(str(item.get("remaining_count", "0"))) <= 0
        }
        intents = list(self.connection.execute(
            "SELECT * FROM live_order_intents WHERE order_id IS NOT NULL"
        ))
        for intent in intents:
            order_id = str(intent["order_id"])
            matched = by_order.get(order_id, [])
            if matched:
                quantity = sum(
                    (Decimal(str(item.get("count", "0"))) for item in matched),
                    Decimal("0"),
                )
                fee = sum(
                    (Decimal(str(item.get("fee_cost", "0"))) for item in matched),
                    Decimal("0"),
                )
                premium = Decimal("0")
                side = str(intent["side"])
                for item in matched:
                    price = Decimal(str(
                        item.get("yes_price" if side == "up" else "no_price", "0")
                    ))
                    premium += Decimal(str(item.get("count", "0"))) * price
                average = premium / quantity if quantity > 0 else Decimal("0")
                with self.connection:
                    self.connection.execute(
                        """UPDATE live_order_intents SET fill_count=?,fee_estimate=?,
                        requested_price=? WHERE client_order_id=?""",
                        (str(quantity), str(fee), str(average), intent["client_order_id"]),
                    )
                if str(intent["role"]) == "entry":
                    created = datetime.fromisoformat(str(intent["created_at"]))
                    day = created.astimezone(__import__("zoneinfo").ZoneInfo(
                        "America/New_York"
                    )).date().isoformat()
                    self.account_live_entry_cost(
                        str(intent["client_order_id"]), day, premium + fee
                    )
            elif order_id in terminal_orders and str(intent["role"]) == "entry":
                created = datetime.fromisoformat(str(intent["created_at"]))
                day = created.astimezone(__import__("zoneinfo").ZoneInfo(
                    "America/New_York"
                )).date().isoformat()
                self.release_live_entry(str(intent["client_order_id"]), day)
                if intent["trade_id"]:
                    self.abandon_empty_live_trade(str(intent["trade_id"]))
        active_tickers = {
            str(item.get("ticker", "")) for item in (positions or [])
            if Decimal(str(item.get("quantity", "0"))) > 0
        }
        for activity in external_activity:
            if activity["kind"] != "external_exit":
                continue
            ticker = str(activity["ticker"])
            if ticker in active_tickers:
                continue
            with self.connection:
                self.connection.execute(
                    """UPDATE live_trades SET phase='flat',exit_locked=1,closed_at=?
                    WHERE trade_id=? AND phase='exit_locked'""",
                    (str(activity["created_at"]), str(activity["trade_id"])),
                )
        for settlement in settlements or []:
            ticker = str(settlement.get("ticker", ""))
            if not ticker or ticker in active_tickers:
                continue
            settled_at = settlement.get("settled_at")
            closed_at = (
                datetime.fromisoformat(str(settled_at).replace("Z", "+00:00"))
                if settled_at else datetime.now(UTC)
            )
            trade = self.active_live_trade(ticker)
            if trade is not None:
                with self.connection:
                    self.connection.execute(
                        """UPDATE live_trades SET phase='flat',exit_locked=1,closed_at=?
                        WHERE trade_id=?""", (closed_at.isoformat(), trade["trade_id"]),
                    )
        return external_activity

    def release_live_entry(self, client_order_id: str, day: str) -> None:
        with self.connection:
            row = self.connection.execute(
                "SELECT entry_cost FROM live_entry_locks WHERE client_order_id=?",
                (client_order_id,),
            ).fetchone()
            if row is None:
                return
            remaining = max(
                Decimal("0"), self.live_daily_spend(day) - Decimal(row["entry_cost"])
            )
            self.connection.execute(
                "DELETE FROM live_entry_locks WHERE client_order_id=?",
                (client_order_id,),
            )
            self.connection.execute(
                "UPDATE live_daily_spend SET amount=? WHERE day=?",
                (str(remaining), day),
            )

    def finalize_live_entry(
        self, client_order_id: str, day: str, actual_cost: Decimal,
    ) -> None:
        with self.connection:
            row = self.connection.execute(
                "SELECT entry_cost FROM live_entry_locks WHERE client_order_id=?",
                (client_order_id,),
            ).fetchone()
            if row is None:
                return
            reserved = Decimal(str(row["entry_cost"]))
            adjusted = max(
                Decimal("0"), self.live_daily_spend(day) - reserved + actual_cost
            )
            self.connection.execute(
                "DELETE FROM live_entry_locks WHERE client_order_id=?",
                (client_order_id,),
            )
            self.connection.execute(
                "UPDATE live_daily_spend SET amount=? WHERE day=?",
                (str(adjusted), day),
            )

    def account_live_entry_cost(
        self, client_order_id: str, day: str, actual_cost: Decimal,
    ) -> None:
        lock = self.connection.execute(
            "SELECT 1 FROM live_entry_locks WHERE client_order_id=?",
            (client_order_id,),
        ).fetchone()
        if lock is not None:
            self.finalize_live_entry(client_order_id, day, actual_cost)
        row = self.connection.execute(
            "SELECT accounted_cost FROM live_order_intents WHERE client_order_id=?",
            (client_order_id,),
        ).fetchone()
        previous = Decimal(str(row["accounted_cost"])) if row is not None else Decimal("0")
        if lock is None and previous != actual_cost:
            with self.connection:
                self.connection.execute(
                    """INSERT INTO live_daily_spend(day,amount) VALUES (?,?)
                    ON CONFLICT(day) DO UPDATE SET amount=excluded.amount""",
                    (day, str(max(Decimal("0"), self.live_daily_spend(day) - previous + actual_cost))),
                )
        with self.connection:
            self.connection.execute(
                "UPDATE live_order_intents SET accounted_cost=? WHERE client_order_id=?",
                (str(actual_cost), client_order_id),
            )

    def reserve_live_exit(
        self, ticker: str, side: str, client_order_id: str,
        reason: str, placed_at: datetime, order_kind: str = "legacy",
    ) -> bool:
        try:
            with self.connection:
                self.connection.execute(
                    """INSERT INTO live_exit_locks(
                    ticker,side,client_order_id,reason,placed_at,order_kind
                    ) VALUES (?,?,?,?,?,?)""",
                    (ticker, side, client_order_id, reason, placed_at.isoformat(), order_kind),
                )
        except sqlite3.IntegrityError:
            return False
        return True

    def record_live_exit(self, client_order_id: str, order_id: str) -> None:
        with self.connection:
            self.connection.execute(
                """UPDATE live_exit_locks SET order_id=?,last_error=NULL,
                attempts=attempts+1 WHERE client_order_id=?""",
                (order_id, client_order_id),
            )

    def mark_live_exit_filled(self, client_order_id: str, order_id: str) -> None:
        with self.connection:
            self.connection.execute(
                """UPDATE live_exit_locks SET order_id=?,last_error='fill_confirmed',
                attempts=attempts+1 WHERE client_order_id=?""",
                (order_id, client_order_id),
            )

    def live_exit_lock(self, ticker: str, side: str) -> dict[str, object] | None:
        row = self.connection.execute(
            "SELECT * FROM live_exit_locks WHERE ticker=? AND side=?",
            (ticker, side),
        ).fetchone()
        return dict(row) if row is not None else None

    def record_live_exit_error(self, client_order_id: str, error: str) -> None:
        with self.connection:
            self.connection.execute(
                """UPDATE live_exit_locks SET last_error=?,attempts=attempts+1
                WHERE client_order_id=?""",
                (error[:500], client_order_id),
            )

    def live_protection_status(self) -> list[dict[str, object]]:
        return [dict(row) for row in self.connection.execute(
            "SELECT * FROM live_exit_locks ORDER BY placed_at DESC"
        )]

    def live_fills(self, limit: int = 1000) -> list[dict[str, object]]:
        return [dict(row) for row in self.connection.execute(
            "SELECT * FROM live_fills ORDER BY created_at DESC,fill_id DESC LIMIT ?",
            (limit,),
        )]

    def release_live_exit(self, client_order_id: str) -> None:
        with self.connection:
            self.connection.execute(
                "DELETE FROM live_exit_locks WHERE client_order_id=?",
                (client_order_id,),
            )

    def reconcile_live_positions(
        self, positions: list[dict[str, object]], observed_at: datetime,
        cooldown_seconds: int = 0,
    ) -> None:
        active = {
            (str(item.get("ticker", "")), str(item.get("side", "")))
            for item in positions
            if Decimal(str(item.get("quantity", "0"))) > 0
        }
        completed = [
            row for row in self.connection.execute("SELECT * FROM live_exit_locks")
            if (str(row["ticker"]), str(row["side"])) not in active
        ]
        for row in completed:
            ticker, side = str(row["ticker"]), str(row["side"])
            with self.connection:
                self.connection.execute(
                    "DELETE FROM live_exit_locks WHERE ticker=? AND side=?",
                    (ticker, side),
                )
                self.connection.execute(
                    "DELETE FROM live_entry_locks WHERE ticker=? AND side=?",
                    (ticker, side),
                )
            self.close_live_trade(ticker, side, observed_at)
        # A manual Kalshi close or a failed exit request may leave no local
        # exit lock. Once a fresh positions snapshot confirms zero, release the
        # durable logical trade as well.
        for trade in list(self.connection.execute(
            "SELECT ticker,side FROM live_trades WHERE phase='exit_locked'"
        )):
            ticker, side = str(trade["ticker"]), str(trade["side"])
            if (ticker, side) not in active:
                self.close_live_trade(ticker, side, observed_at)
