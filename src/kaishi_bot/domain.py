from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Literal


class Side(StrEnum):
    UP = "up"
    DOWN = "down"


@dataclass(frozen=True, slots=True)
class Market:
    ticker: str
    close_time: datetime


@dataclass(frozen=True, slots=True)
class SideQuotes:
    up_ask: Decimal
    down_ask: Decimal


@dataclass(frozen=True, slots=True)
class EntryIntent:
    ticker: str
    side: Side
    side_price: Decimal
    count: Decimal


BookSide = Literal["bid", "ask"]


@dataclass(frozen=True, slots=True)
class OrderRequest:
    ticker: str
    client_order_id: str
    book_side: BookSide
    count: Decimal
    yes_price: Decimal
    reduce_only: bool
    time_in_force: Literal[
        "fill_or_kill", "good_till_canceled", "immediate_or_cancel"
    ] = "good_till_canceled"


@dataclass(frozen=True, slots=True)
class OrderResult:
    order_id: str
    client_order_id: str
    fill_count: Decimal | None = None


@dataclass(frozen=True, slots=True)
class Fill:
    fill_id: str
    order_id: str
    ticker: str
    quantity: Decimal
