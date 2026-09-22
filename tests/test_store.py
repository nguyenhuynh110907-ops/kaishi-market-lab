from decimal import Decimal
from pathlib import Path

from kaishi_bot.domain import Side
from kaishi_bot.store import StateStore


def test_entry_reservation_is_atomic_and_survives_restart(tmp_path: Path) -> None:
    database = tmp_path / "state.sqlite3"

    with StateStore(database) as store:
        assert store.reserve_entry("MKT", Side.DOWN, "client-1") is True
        assert store.reserve_entry("MKT", Side.DOWN, "client-2") is False

    with StateStore(database) as reopened:
        assert reopened.locked_sides("MKT") == frozenset({Side.DOWN})


def test_entry_order_id_resolves_original_side(tmp_path: Path) -> None:
    with StateStore(tmp_path / "state.sqlite3") as store:
        store.reserve_entry("MKT", Side.UP, "client-1")
        store.record_entry_order("client-1", "order-1")

        assert store.entry_side_for_order("order-1") is Side.UP
        assert store.entry_side_for_order("missing") is None


def test_repeated_fill_is_recorded_once(tmp_path: Path) -> None:
    with StateStore(tmp_path / "state.sqlite3") as store:
        assert store.record_fill("fill-1", "MKT", Side.UP, Decimal("0.40")) is True
        assert store.record_fill("fill-1", "MKT", Side.UP, Decimal("0.40")) is False

        assert store.uncovered_quantity("MKT", Side.UP) == Decimal("0.40")


def test_reserved_take_profit_covers_only_reserved_quantity(tmp_path: Path) -> None:
    database = tmp_path / "state.sqlite3"

    with StateStore(database) as store:
        store.record_fill("fill-1", "MKT", Side.UP, Decimal("0.40"))
        assert store.reserve_take_profit(
            "MKT",
            Side.UP,
            "tp-client-1",
            Decimal("0.25"),
        ) is True
        assert store.uncovered_quantity("MKT", Side.UP) == Decimal("0.15")
        store.record_take_profit_order("tp-client-1", "tp-order-1")

    with StateStore(database) as reopened:
        assert reopened.uncovered_quantity("MKT", Side.UP) == Decimal("0.15")


def test_take_profit_client_id_is_idempotent(tmp_path: Path) -> None:
    with StateStore(tmp_path / "state.sqlite3") as store:
        assert store.reserve_take_profit(
            "MKT", Side.DOWN, "tp-client-1", Decimal("0.35")
        ) is True
        assert store.reserve_take_profit(
            "MKT", Side.DOWN, "tp-client-1", Decimal("0.35")
        ) is False


def test_summary_counts_persisted_records(tmp_path: Path) -> None:
    with StateStore(tmp_path / "state.sqlite3") as store:
        store.reserve_entry("MKT", Side.UP, "client-1")
        store.record_fill("fill-1", "MKT", Side.UP, Decimal("0.40"))
        store.reserve_take_profit("MKT", Side.UP, "tp-client-1", Decimal("0.40"))

        assert store.summary() == {
            "entries": 1,
            "fills": 1,
            "take_profits": 1,
        }
