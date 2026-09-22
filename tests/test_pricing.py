from decimal import Decimal

import pytest

from kaishi_bot.domain import Side
from kaishi_bot.pricing import side_quotes, to_v2_entry, to_v2_exit


def test_yes_book_maps_to_up_and_down_executable_asks() -> None:
    quotes = side_quotes(
        yes_bid=Decimal("0.74"),
        yes_ask=Decimal("0.76"),
    )

    assert quotes.up_ask == Decimal("0.76")
    assert quotes.down_ask == Decimal("0.26")


@pytest.mark.parametrize(
    ("side", "price", "expected"),
    [
        (Side.UP, Decimal("0.25"), ("bid", Decimal("0.25"))),
        (Side.DOWN, Decimal("0.25"), ("ask", Decimal("0.75"))),
    ],
)
def test_entry_converts_side_price_to_yes_book(
    side: Side,
    price: Decimal,
    expected: tuple[str, Decimal],
) -> None:
    assert to_v2_entry(side, price) == expected


@pytest.mark.parametrize(
    ("side", "price", "expected"),
    [
        (Side.UP, Decimal("0.40"), ("ask", Decimal("0.40"))),
        (Side.DOWN, Decimal("0.40"), ("bid", Decimal("0.60"))),
    ],
)
def test_exit_converts_side_price_to_yes_book(
    side: Side,
    price: Decimal,
    expected: tuple[str, Decimal],
) -> None:
    assert to_v2_exit(side, price) == expected
