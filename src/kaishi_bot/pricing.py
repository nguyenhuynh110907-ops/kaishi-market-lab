from __future__ import annotations

from decimal import Decimal
from typing import Literal

from kaishi_bot.domain import Side, SideQuotes

BookSide = Literal["bid", "ask"]
ONE = Decimal("1")


def side_quotes(*, yes_bid: Decimal, yes_ask: Decimal) -> SideQuotes:
    """Return executable asks expressed in each user-facing side's price."""

    return SideQuotes(up_ask=yes_ask, down_ask=ONE - yes_bid)


def to_v2_entry(side: Side, side_price: Decimal) -> tuple[BookSide, Decimal]:
    """Convert a user-side buy to the unified YES book."""

    if side is Side.UP:
        return "bid", side_price
    return "ask", ONE - side_price


def to_v2_exit(side: Side, side_price: Decimal) -> tuple[BookSide, Decimal]:
    """Convert a user-side sale to the unified YES book."""

    if side is Side.UP:
        return "ask", side_price
    return "bid", ONE - side_price
