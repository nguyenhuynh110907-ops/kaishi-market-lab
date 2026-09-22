from __future__ import annotations

import sqlite3
from decimal import Decimal
from pathlib import Path

from kaishi_bot.domain import Side


class StateStore:
    """Durable idempotency and take-profit coverage state."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS entry_locks (
                ticker TEXT NOT NULL,
                side TEXT NOT NULL,
                client_order_id TEXT NOT NULL UNIQUE,
                exchange_order_id TEXT UNIQUE,
                PRIMARY KEY (ticker, side)
            );

            CREATE TABLE IF NOT EXISTS fills (
                fill_id TEXT PRIMARY KEY,
                ticker TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS take_profit_orders (
                client_order_id TEXT PRIMARY KEY,
                exchange_order_id TEXT UNIQUE,
                ticker TEXT NOT NULL,
                side TEXT NOT NULL,
                covered_quantity TEXT NOT NULL
            );
            """
        )

    def __enter__(self) -> "StateStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.connection.close()

    def reserve_entry(
        self,
        ticker: str,
        side: Side,
        client_order_id: str,
    ) -> bool:
        try:
            with self.connection:
                self.connection.execute(
                    """
                    INSERT INTO entry_locks(ticker, side, client_order_id)
                    VALUES (?, ?, ?)
                    """,
                    (ticker, side.value, client_order_id),
                )
        except sqlite3.IntegrityError:
            return False
        return True

    def record_entry_order(
        self,
        client_order_id: str,
        exchange_order_id: str,
    ) -> None:
        with self.connection:
            cursor = self.connection.execute(
                """
                UPDATE entry_locks
                SET exchange_order_id = ?
                WHERE client_order_id = ?
                """,
                (exchange_order_id, client_order_id),
            )
        if cursor.rowcount != 1:
            raise KeyError(f"unknown entry client order ID: {client_order_id}")

    def entry_side_for_order(self, exchange_order_id: str) -> Side | None:
        row = self.connection.execute(
            "SELECT side FROM entry_locks WHERE exchange_order_id = ?",
            (exchange_order_id,),
        ).fetchone()
        return Side(row[0]) if row else None

    def locked_sides(self, ticker: str) -> frozenset[Side]:
        rows = self.connection.execute(
            "SELECT side FROM entry_locks WHERE ticker = ?",
            (ticker,),
        )
        return frozenset(Side(row[0]) for row in rows)

    def record_fill(
        self,
        fill_id: str,
        ticker: str,
        side: Side,
        quantity: Decimal,
    ) -> bool:
        try:
            with self.connection:
                self.connection.execute(
                    """
                    INSERT INTO fills(fill_id, ticker, side, quantity)
                    VALUES (?, ?, ?, ?)
                    """,
                    (fill_id, ticker, side.value, str(quantity)),
                )
        except sqlite3.IntegrityError:
            return False
        return True

    def reserve_take_profit(
        self,
        ticker: str,
        side: Side,
        client_order_id: str,
        quantity: Decimal,
    ) -> bool:
        try:
            with self.connection:
                self.connection.execute(
                    """
                    INSERT INTO take_profit_orders(
                        client_order_id, ticker, side, covered_quantity
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (client_order_id, ticker, side.value, str(quantity)),
                )
        except sqlite3.IntegrityError:
            return False
        return True

    def record_take_profit_order(
        self,
        client_order_id: str,
        exchange_order_id: str,
    ) -> None:
        with self.connection:
            cursor = self.connection.execute(
                """
                UPDATE take_profit_orders
                SET exchange_order_id = ?
                WHERE client_order_id = ?
                """,
                (exchange_order_id, client_order_id),
            )
        if cursor.rowcount != 1:
            raise KeyError(f"unknown take-profit client order ID: {client_order_id}")

    def uncovered_quantity(self, ticker: str, side: Side) -> Decimal:
        fill_rows = self.connection.execute(
            "SELECT quantity FROM fills WHERE ticker = ? AND side = ?",
            (ticker, side.value),
        )
        coverage_rows = self.connection.execute(
            """
            SELECT covered_quantity
            FROM take_profit_orders
            WHERE ticker = ? AND side = ?
            """,
            (ticker, side.value),
        )
        filled = sum((Decimal(row[0]) for row in fill_rows), Decimal())
        covered = sum((Decimal(row[0]) for row in coverage_rows), Decimal())
        return filled - covered

    def summary(self) -> dict[str, int]:
        def count(table: str) -> int:
            row = self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            assert row is not None
            return int(row[0])

        return {
            "entries": count("entry_locks"),
            "fills": count("fills"),
            "take_profits": count("take_profit_orders"),
        }
