from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Callable, Iterable, Sequence

from kaishi_bot.agentic_strategy.models import MarketObservation, PositionState
from kaishi_bot.fees import FeeSchedule, taker_fee
from kaishi_bot.research_models import (
    BookCheckpoint,
    ContractQuoteEvent,
    FeeMetadataVersion,
    ResearchMarket,
    RtiEvent,
)


ZERO = Decimal("0")


@dataclass(frozen=True, slots=True)
class SettlementFeatures:
    locked_sample_count: int
    locked_sample_sum: Decimal
    locked_average: Decimal | None
    remaining_sample_count: int
    required_remaining_average: Decimal | None
    missing_sample_count: int
    current_brti_minus_target: Decimal | None
    current_brti_minus_required_average: Decimal | None


def build_settlement_features(
    market: ResearchMarket, events: Iterable[RtiEvent], observation_time: datetime,
    *, current_brti: Decimal | None = None,
    quality_event: Callable[[str, dict[str, object]], None] | None = None,
) -> SettlementFeatures:
    """Build the close-window state visible at ``observation_time``.

    The window is strictly ``(close-60s, close]``. Samples are de-duplicated by
    source timestamp and only become visible at collector receipt time.
    """
    observed = observation_time.astimezone(UTC)
    start = market.settlement_window_start
    end = market.settlement_window_end
    visible: dict[tuple[str, int], RtiEvent] = {}
    for event in sorted(events, key=lambda item: item.collector_received_at):
        if event.collector_received_at > observed:
            continue
        if not start < event.source_time_utc <= end:
            continue
        key = (event.index_id, event.source_timestamp_ms)
        previous = visible.get(key)
        if previous is None:
            visible[key] = event
        elif previous.price != event.price and quality_event is not None:
            quality_event("conflicting_rti_duplicate", {
                "index_id": event.index_id,
                "source_timestamp_ms": event.source_timestamp_ms,
            })

    locked = sorted(visible.values(), key=lambda item: item.source_timestamp_ms)
    count = len(locked)
    total = sum((item.price for item in locked), ZERO)
    average = total / Decimal(count) if count else None
    remaining = 60 - count
    required = None
    if 0 < count < 60 and market.target_price is not None:
        required = (
            Decimal(60) * market.target_price - total
        ) / Decimal(remaining)

    elapsed = max(0, min(60, int((min(observed, end) - start).total_seconds())))
    missing = max(0, elapsed - count)
    current = current_brti
    if current is None:
        candidates = [
            event for event in events
            if event.collector_received_at <= observed
            and event.source_time_utc <= observed
        ]
        if candidates:
            current = max(candidates, key=lambda item: item.collector_received_at).price

    # Server averages are compared only when their declared half-open window
    # represents the exact integer-second set in our open-left/closed-right window.
    expected_start_ms = int(start.timestamp() * 1000) + 1000
    expected_end_exclusive_ms = int(end.timestamp() * 1000) + 1000
    for event in locked:
        if (
            count == 60 and event.avg_60s is not None
            and event.avg_60s_window_start_ms == expected_start_ms
            and event.avg_60s_window_end_exclusive_ms == expected_end_exclusive_ms
            and event.avg_60s != average
            and quality_event is not None
        ):
            quality_event("server_average_mismatch", {
                "local": str(average), "server": str(event.avg_60s),
                "ticker": market.ticker,
            })
            break

    return SettlementFeatures(
        locked_sample_count=count, locked_sample_sum=total,
        locked_average=average, remaining_sample_count=remaining,
        required_remaining_average=required, missing_sample_count=missing,
        current_brti_minus_target=(
            current - market.target_price
            if current is not None and market.target_price is not None else None
        ),
        current_brti_minus_required_average=(
            current - required if current is not None and required is not None else None
        ),
    )


@dataclass(frozen=True, slots=True)
class FeatureObservation:
    dataset_version: str
    feature_schema_version: str
    ticker: str
    asset: str
    observation_time: datetime
    available_at: datetime
    market_manifest_id: str
    rti_manifest_id: str
    quote_manifest_id: str
    orderbook_manifest_id: str | None
    fee_version: str
    build_commit: str
    seconds_remaining: int
    up_bid: Decimal | None
    up_ask: Decimal | None
    down_bid: Decimal | None
    down_ask: Decimal | None
    target_price: Decimal | None
    brti_price: Decimal | None
    brti_sigma_per_sqrt_second: Decimal | None
    locked_sample_count: int
    locked_sample_sum: Decimal
    elapsed_seconds: int = 0
    is_final_minute: bool = False
    second_inside_final_minute: int | None = None
    up_spread: Decimal | None = None
    down_spread: Decimal | None = None
    depth_1c: Decimal | None = None
    depth_3c: Decimal | None = None
    depth_5c: Decimal | None = None
    book_imbalance: Decimal | None = None
    quote_update_rate: Decimal | None = None
    contract_history: tuple[tuple[str, Decimal | None], ...] = ()
    brti_target_distance: Decimal | None = None
    brti_return_1s: Decimal | None = None
    brti_return_5s: Decimal | None = None
    brti_return_15s: Decimal | None = None
    brti_return_30s: Decimal | None = None
    brti_return_60s: Decimal | None = None
    realized_volatility_15s: Decimal | None = None
    realized_volatility_30s: Decimal | None = None
    realized_volatility_60s: Decimal | None = None
    distance_normalized_by_volatility: Decimal | None = None
    locked_average: Decimal | None = None
    remaining_sample_count: int = 60
    required_remaining_average: Decimal | None = None
    distance_to_required_average: Decimal | None = None
    missing_sample_count: int = 0
    quote_age_ms: int | None = None
    brti_age_ms: int | None = None
    book_age_ms: int | None = None
    fee_metadata_available: bool = False
    quote_gap: bool = False
    rti_gap: bool = False
    book_gap: bool = False
    is_stale: bool = False
    feature_complete: bool = False
    incomplete_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BuiltObservation:
    observation: MarketObservation
    dataset_version: str
    feature_schema_version: str
    manifest_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RejectedObservation:
    ticker: str
    observation_time: datetime
    reasons: tuple[str, ...]


class ResearchObservationAdapter:
    """Pure feature-row adapter. It deliberately has no API dependency."""

    REQUIRED = (
        "up_bid", "up_ask", "down_bid", "down_ask", "target_price",
        "brti_price", "brti_sigma_per_sqrt_second",
    )

    def build(
        self, feature: FeatureObservation, *, fee_schedule: FeeSchedule | None,
        position: PositionState | None = None,
    ) -> BuiltObservation | RejectedObservation:
        reasons = list(feature.incomplete_reasons)
        for name in self.REQUIRED:
            if getattr(feature, name) is None:
                reasons.append(f"missing_{name}")
        if not feature.feature_complete:
            reasons.append("feature_incomplete")
        if fee_schedule is None or not feature.fee_metadata_available:
            reasons.append("missing_fee_metadata")
        if feature.available_at > feature.observation_time:
            reasons.append("future_available_at")
        if reasons:
            return RejectedObservation(
                feature.ticker, feature.observation_time,
                tuple(sorted(set(reasons))),
            )

        assert fee_schedule is not None
        assert all(getattr(feature, name) is not None for name in self.REQUIRED)
        observation = MarketObservation(
            ticker=feature.ticker, observed_at=feature.observation_time,
            seconds_remaining=feature.seconds_remaining,
            up_bid=feature.up_bid, up_ask=feature.up_ask,
            down_bid=feature.down_bid, down_ask=feature.down_ask,
            target_price=feature.target_price, brti_price=feature.brti_price,
            brti_sigma_per_sqrt_second=feature.brti_sigma_per_sqrt_second,
            locked_sample_count=feature.locked_sample_count,
            locked_sample_sum=feature.locked_sample_sum,
            entry_fee_up=taker_fee(fee_schedule, Decimal("1"), feature.up_ask),
            entry_fee_down=taker_fee(fee_schedule, Decimal("1"), feature.down_ask),
            position=position, data_stale=feature.is_stale,
            has_gap=feature.quote_gap or feature.rti_gap or feature.book_gap,
        )
        manifests = tuple(filter(None, (
            feature.market_manifest_id, feature.rti_manifest_id,
            feature.quote_manifest_id, feature.orderbook_manifest_id,
        )))
        return BuiltObservation(
            observation, feature.dataset_version,
            feature.feature_schema_version, manifests,
        )


def _latest_visible(rows: Sequence, observed: datetime):
    visible = [row for row in rows if row.available_at <= observed]
    return max(visible, key=lambda row: row.available_at) if visible else None


def _asof(rows: Sequence, cutoff: datetime, tolerance_ms: int):
    visible = [row for row in rows if row.available_at <= cutoff]
    if not visible:
        return None
    row = max(visible, key=lambda item: item.available_at)
    age_ms = int((cutoff - row.available_at).total_seconds() * 1000)
    return row if age_ms <= tolerance_ms else None


def _return(current: Decimal | None, previous: Decimal | None) -> Decimal | None:
    if current is None or previous in {None, ZERO}:
        return None
    return (current - previous) / previous


def _realized_sigma(events: Sequence[RtiEvent], observed: datetime, seconds: int) -> Decimal | None:
    rows = sorted(
        (
            item for item in events
            if item.collector_received_at <= observed
            and observed - timedelta(seconds=seconds) <= item.source_time_utc <= observed
        ),
        key=lambda item: item.source_timestamp_ms,
    )
    unique: dict[tuple[str, int], RtiEvent] = {}
    for row in rows:
        unique.setdefault((row.index_id, row.source_timestamp_ms), row)
    prices = [row.price for row in unique.values()]
    if len(prices) < 2:
        return None
    changes = [right - left for left, right in zip(prices, prices[1:])]
    mean = sum(changes, ZERO) / Decimal(len(changes))
    variance = sum(((value - mean) ** 2 for value in changes), ZERO) / Decimal(len(changes))
    return variance.sqrt()


class PointInTimeFeatureBuilder:
    HORIZONS = (1, 3, 5, 15, 30, 60)

    def build(
        self, *, market: ResearchMarket, observation_time: datetime,
        rti_events: Sequence[RtiEvent], quote_events: Sequence[ContractQuoteEvent],
        book_checkpoints: Sequence[BookCheckpoint], fee_version: FeeMetadataVersion | None,
        dataset_version: str, market_manifest_id: str, rti_manifest_id: str,
        quote_manifest_id: str, orderbook_manifest_id: str | None,
        build_commit: str,
    ) -> FeatureObservation:
        observed = observation_time.astimezone(UTC)
        quote = _latest_visible(quote_events, observed)
        rti_visible = [
            row for row in rti_events
            if row.collector_received_at <= observed and row.source_time_utc <= observed
        ]
        rti = max(rti_visible, key=lambda row: row.collector_received_at) if rti_visible else None
        book = _latest_visible(book_checkpoints, observed)
        reasons: list[str] = []
        quote_age = int((observed - quote.available_at).total_seconds() * 1000) if quote else None
        brti_age = int((observed - rti.collector_received_at).total_seconds() * 1000) if rti else None
        book_age = int((observed - book.available_at).total_seconds() * 1000) if book else None
        if quote is None:
            reasons.append("missing_contract_quote")
        elif quote_age is not None and quote_age > 2000:
            reasons.append("stale_quote")
        if rti is None:
            reasons.append("missing_rti_sample")
        elif brti_age is not None and brti_age > 1500:
            reasons.append("stale_rti")
        fee_available = bool(
            fee_version is not None
            and fee_version.observed_at <= observed
            and fee_version.effective_from <= market.open_time
            and (fee_version.effective_to is None or fee_version.effective_to > market.open_time)
        )
        if not fee_available:
            reasons.append("missing_fee_metadata")
        if market.target_price is None:
            reasons.append("missing_target")
        if market.refreshed_at > observed:
            reasons.append("future_market_metadata")
        if orderbook_manifest_id is not None and book is None:
            reasons.append("missing_orderbook")

        settlement = build_settlement_features(
            market, rti_events, observed,
            current_brti=rti.price if rti else None,
        )
        history: list[tuple[str, Decimal | None]] = []
        if quote is not None:
            for horizon in self.HORIZONS:
                prior = _asof(
                    quote_events, observed - timedelta(seconds=horizon), 2000
                )
                for field in ("up_bid", "up_ask", "down_bid", "down_ask"):
                    current_value = getattr(quote, field)
                    prior_value = getattr(prior, field) if prior else None
                    history.extend((
                        (f"{field}_change_{horizon}s",
                         current_value - prior_value if current_value is not None and prior_value is not None else None),
                        (f"{field}_return_{horizon}s", _return(current_value, prior_value)),
                    ))

        rti_returns: dict[int, Decimal | None] = {}
        for horizon in (1, 5, 15, 30, 60):
            prior = _asof(
                [
                    _RtiAvailable(row) for row in rti_events
                    if row.collector_received_at <= observed
                ],
                observed - timedelta(seconds=horizon), 1500,
            )
            rti_returns[horizon] = _return(rti.price if rti else None, prior.price if prior else None)
        sigma = {horizon: _realized_sigma(rti_events, observed, horizon) for horizon in (15, 30, 60)}
        effective_sigma = sigma[60] or sigma[30] or sigma[15]
        if effective_sigma is None:
            reasons.append("insufficient_volatility_history")
        distance = (
            rti.price - market.target_price
            if rti is not None and market.target_price is not None else None
        )
        quote_count = sum(
            item.available_at > observed - timedelta(seconds=60)
            and item.available_at <= observed for item in quote_events
        )
        is_stale = any(reason.startswith("stale_") for reason in reasons)
        quote_gap = bool(quote and quote.gap_detected)
        rti_gap = bool(rti and rti.gap_detected)
        book_gap = bool(book and not book.is_complete)
        if quote_gap:
            reasons.append("quote_gap")
        if rti_gap:
            reasons.append("rti_gap")
        if book_gap:
            reasons.append("book_sequence_gap")
        available_inputs = [market.refreshed_at]
        if quote:
            available_inputs.append(quote.available_at)
        if rti:
            available_inputs.append(rti.collector_received_at)
        if book:
            available_inputs.append(book.available_at)
        if fee_version and fee_version.observed_at <= observed:
            available_inputs.append(fee_version.observed_at)
        seconds_remaining = max(0, int((market.close_time - observed).total_seconds()))
        elapsed = max(0, int((observed - market.open_time).total_seconds()))
        final_minute = seconds_remaining <= 60
        return FeatureObservation(
            dataset_version=dataset_version, feature_schema_version="1",
            ticker=market.ticker, asset=market.asset, observation_time=observed,
            available_at=max(available_inputs), market_manifest_id=market_manifest_id,
            rti_manifest_id=rti_manifest_id, quote_manifest_id=quote_manifest_id,
            orderbook_manifest_id=orderbook_manifest_id,
            fee_version=fee_version.fee_version if fee_version else "",
            build_commit=build_commit, seconds_remaining=seconds_remaining,
            up_bid=quote.up_bid if quote else None, up_ask=quote.up_ask if quote else None,
            down_bid=quote.down_bid if quote else None,
            down_ask=quote.down_ask if quote else None,
            target_price=market.target_price, brti_price=rti.price if rti else None,
            brti_sigma_per_sqrt_second=effective_sigma,
            locked_sample_count=settlement.locked_sample_count,
            locked_sample_sum=settlement.locked_sample_sum,
            elapsed_seconds=elapsed, is_final_minute=final_minute,
            second_inside_final_minute=(60 - seconds_remaining if final_minute else None),
            up_spread=quote.up_spread if quote else None,
            down_spread=quote.down_spread if quote else None,
            depth_1c=book.depth_1c if book else None,
            depth_3c=book.depth_3c if book else None,
            depth_5c=book.depth_5c if book else None,
            book_imbalance=book.book_imbalance if book else None,
            quote_update_rate=Decimal(quote_count) / Decimal(60),
            contract_history=tuple(history), brti_target_distance=distance,
            brti_return_1s=rti_returns[1], brti_return_5s=rti_returns[5],
            brti_return_15s=rti_returns[15], brti_return_30s=rti_returns[30],
            brti_return_60s=rti_returns[60],
            realized_volatility_15s=sigma[15], realized_volatility_30s=sigma[30],
            realized_volatility_60s=sigma[60],
            distance_normalized_by_volatility=(
                distance / effective_sigma if distance is not None and effective_sigma not in {None, ZERO} else None
            ),
            locked_average=settlement.locked_average,
            remaining_sample_count=settlement.remaining_sample_count,
            required_remaining_average=settlement.required_remaining_average,
            distance_to_required_average=(
                rti.price - settlement.required_remaining_average
                if rti and settlement.required_remaining_average is not None else None
            ), missing_sample_count=settlement.missing_sample_count,
            quote_age_ms=quote_age, brti_age_ms=brti_age, book_age_ms=book_age,
            fee_metadata_available=fee_available, quote_gap=quote_gap,
            rti_gap=rti_gap, book_gap=book_gap, is_stale=is_stale,
            feature_complete=not reasons,
            incomplete_reasons=tuple(sorted(set(reasons))),
        )


@dataclass(frozen=True, slots=True)
class _RtiAvailable:
    row: RtiEvent

    @property
    def available_at(self) -> datetime:
        return self.row.collector_received_at

    @property
    def price(self) -> Decimal:
        return self.row.price
