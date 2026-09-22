from datetime import UTC, datetime, timedelta

import httpx

from kaishi_bot.research_history_pretrain import (
    HistoricalCandle,
    HistoricalKalshiClient,
    HistoricalMarket,
    build_historical_training_rows,
)


def test_candle_is_only_available_at_its_end_and_history_is_backward_only() -> None:
    opened = datetime(2026, 8, 1, 12, tzinfo=UTC)
    market = HistoricalMarket(
        "BTC-1", "BTC", "KXBTC15M", opened, opened + timedelta(minutes=15),
        target_price=1, label_yes=1,
    )
    candles = [HistoricalCandle(
        "BTC-1", opened + timedelta(minutes=minute),
        0.5, 0.5 + minute / 100, 0.4, 0.5 + minute / 100,
        0.51, 0.51 + minute / 100, 0.41, 0.51 + minute / 100,
        0.5, 0.6, 0.4, 0.5, 0.5, 10.0 * minute, 100.0,
    ) for minute in range(1, 16)]

    rows = build_historical_training_rows([market], candles)

    assert rows[0].observed_at == opened + timedelta(minutes=9)
    assert rows[0].features["seconds_remaining"] == 360
    assert rows[0].features["mid_momentum_3m"] > 0
    assert rows[-1].observed_at == opened + timedelta(minutes=14)


def test_public_backfill_parses_bulk_one_minute_candles() -> None:
    opened = datetime(2026, 8, 1, 12, tzinfo=UTC)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/markets"):
            return httpx.Response(200, json={"markets": [{
                "ticker": "BTC-1", "open_time": opened.isoformat(),
                "close_time": (opened + timedelta(minutes=15)).isoformat(),
                "floor_strike_dollars": "65000.00", "result": "yes",
            }], "cursor": ""})
        return httpx.Response(200, json={"markets": [{
            "market_ticker": "BTC-1",
            "candlesticks": [{
                "end_period_ts": int((opened + timedelta(minutes=1)).timestamp()),
                "yes_bid": {"open_dollars": "0.50", "high_dollars": "0.60",
                            "low_dollars": "0.40", "close_dollars": "0.55"},
                "yes_ask": {"open_dollars": "0.51", "high_dollars": "0.61",
                            "low_dollars": "0.41", "close_dollars": "0.56"},
                "price": {"close_dollars": "0.55", "mean_dollars": "0.53"},
                "volume_fp": "12.00", "open_interest_fp": "100.00",
            }],
        }]})

    http = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://example.test"
    )
    client = HistoricalKalshiClient(http)
    markets = client.settled_markets(
        asset="BTC", series_ticker="KXBTC15M", start=opened,
        end=opened + timedelta(hours=1),
    )
    candles = client.candles(markets)

    assert markets[0].target_price == 65000
    assert candles[0].yes_bid_close == 0.55
    assert candles[0].end_time == opened + timedelta(minutes=1)
