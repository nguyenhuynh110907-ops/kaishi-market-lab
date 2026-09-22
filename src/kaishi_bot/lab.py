from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from kaishi_bot.dashboard_models import AssetMarket, QuotePoint
from kaishi_bot.dashboard_store import DashboardStore
from kaishi_bot.fees import FeeSchedule, taker_fee, whole_contract_size
from kaishi_bot.entry_guard import EntryGuardSettings, GuardQuote, evaluate_entry


NY = ZoneInfo("America/New_York")
MAX_RUN_CANDIDATES = 25000


@dataclass(frozen=True, slots=True)
class LabCandidate:
    candidate_id: str
    asset: str
    entry_min: Decimal
    entry: Decimal
    take_profit: Decimal
    stop_loss: Decimal
    min_seconds_before_close: int
    side_policy: str


def _decimal(value: float) -> Decimal:
    return Decimal(f"{value:.4f}")


def generate_candidates(asset: str, count: int, seed: int) -> list[LabCandidate]:
    if count < 1:
        raise ValueError("candidate count must be positive")
    rng = random.Random(seed + sum((index + 1) * ord(char) for index, char in enumerate(asset)))
    entry_bins = list(range(count))
    tp_bins = list(range(count))
    sl_bins = list(range(count))
    entry_min_bins = list(range(count))
    for bins in (entry_bins, tp_bins, sl_bins):
        rng.shuffle(bins)
    rng.shuffle(entry_min_bins)
    policies = ["up", "down", "both"]
    result = []
    for index in range(count):
        entry = _decimal(0.10 + 0.75 * ((entry_bins[index] + 0.5) / count))
        tp_low = entry + Decimal("0.05")
        take_profit = tp_low + (Decimal("0.95") - tp_low) * _decimal(
            (tp_bins[index] + 0.5) / count
        )
        take_profit = take_profit.quantize(Decimal("0.0001"))
        band_width_high = min(Decimal("0.20"), entry - Decimal("0.07"))
        band_width = Decimal("0.03") + (
            band_width_high - Decimal("0.03")
        ) * _decimal((entry_min_bins[index] + 0.5) / count)
        entry_min = (entry - band_width).quantize(Decimal("0.0001"))
        sl_gap_high = min(Decimal("0.20"), entry_min - Decimal("0.02"))
        sl_gap = Decimal("0.05") + (
            sl_gap_high - Decimal("0.05")
        ) * _decimal(
            (sl_bins[index] + 0.5) / count
        )
        stop_loss = (entry_min - sl_gap).quantize(Decimal("0.0001"))
        result.append(LabCandidate(
            candidate_id=f"{asset}-{index:03d}", asset=asset,
            entry_min=entry_min, entry=entry,
            take_profit=take_profit, stop_loss=stop_loss,
            min_seconds_before_close=0,
            side_policy=policies[(index + seed) % len(policies)],
        ))
    return result


class StrategyLab:
    def __init__(self, store: DashboardStore) -> None:
        self.store = store

    def start(
        self, assets: list[str], candidate_count: int, seed: int,
        duration_seconds: int, fee_schedules: dict[str, FeeSchedule],
        guard_settings: EntryGuardSettings | None = None,
        history_cycles: int = 0, history_cutoff_event_id: int | None = None, *,
        _run_status: str = "running",
    ) -> int:
        if not 0 <= history_cycles <= 96:
            raise ValueError("history cycles must be between 0 and 96")
        if history_cycles and _run_status == "running":
            _run_status = "backfilling"
        if _run_status not in {"running", "backfilling", "replaying"}:
            raise ValueError("unsupported lab run status")
        missing = [asset for asset in assets if asset not in fee_schedules]
        if missing:
            raise ValueError(f"missing fee schedule for: {', '.join(missing)}")
        total = len(assets) * candidate_count
        if total > MAX_RUN_CANDIDATES:
            raise ValueError(f"at most {MAX_RUN_CANDIDATES} candidates per run")
        active_run = self.store.connection.execute(
            """SELECT id FROM lab_runs
            WHERE status IN ('running','backfilling') LIMIT 1"""
        ).fetchone()
        if active_run is not None and _run_status in {"running", "backfilling"}:
            raise ValueError("stop the current run before starting another")
        if duration_seconds < 60:
            raise ValueError("duration must be at least 60 seconds")
        now = datetime.now(UTC)
        fee_snapshot = json.dumps({
            asset: {
                "fee_type": fee_schedules[asset].fee_type,
                "multiplier": str(fee_schedules[asset].multiplier),
                "taker_rate": str(fee_schedules[asset].taker_rate),
                "version": fee_schedules[asset].version,
            }
            for asset in assets
        }, sort_keys=True)
        guard_snapshot = (
            guard_settings.model_dump_json() if guard_settings is not None else None
        )
        with self.store.connection:
            cursor = self.store.connection.execute(
                """INSERT INTO lab_runs(
                status,started_at,duration_seconds,seed,assets,candidate_count,
                fee_snapshot,guard_snapshot,history_cycles_requested,
                history_cutoff_event_id,backfill_engine
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (_run_status, now.isoformat(), duration_seconds, seed, json.dumps(assets),
                 candidate_count, fee_snapshot, guard_snapshot, history_cycles,
                 history_cutoff_event_id,
                 "vectorized" if history_cycles else "legacy"),
            )
            run_id = int(cursor.lastrowid)
            for asset in assets:
                for item in generate_candidates(asset, candidate_count, seed):
                    self.store.connection.execute(
                        """INSERT INTO lab_candidates(
                        run_id,candidate_id,asset,entry_min,entry_price,take_profit,
                        stop_loss,min_seconds,side_policy
                        ) VALUES (?,?,?,?,?,?,?,?,?)""",
                        (run_id, item.candidate_id, asset, str(item.entry_min),
                         str(item.entry), str(item.take_profit), str(item.stop_loss),
                         item.min_seconds_before_close, item.side_policy),
                    )
        return run_id

    def stop(self, run_id: int) -> bool:
        with self.store.connection:
            cursor = self.store.connection.execute(
                """UPDATE lab_runs SET status='stopped' WHERE id=?
                AND status IN ('running','backfilling')""", (run_id,)
            )
        return cursor.rowcount == 1

    def list_runs(self) -> list[dict[str, object]]:
        rows = self.store.connection.execute("SELECT * FROM lab_runs ORDER BY id DESC")
        return [dict(row) for row in rows]

    def on_quote(
        self, run_id: int, market: AssetMarket, quote: QuotePoint, quote_event_id: int
    ) -> None:
        run = self.store.connection.execute(
            """SELECT * FROM lab_runs WHERE id=?
            AND status IN ('running','backfilling','replaying')
            AND settlement_status='ready'""", (run_id,)
        ).fetchone()
        if run is None:
            return
        started_at = datetime.fromisoformat(str(run["started_at"]))
        if (
            run["status"] == "running"
            and (quote.observed_at - started_at).total_seconds() >= int(run["duration_seconds"])
        ):
            with self.store.connection:
                self.store.connection.execute(
                    "UPDATE lab_runs SET status='completed' WHERE id=?", (run_id,)
                )
            return
        previous_row = self.store.connection.execute(
            """SELECT q.* FROM quote_events q JOIN lab_seen_events s
            ON s.quote_event_id=q.id WHERE s.run_id=? AND q.ticker=?
            ORDER BY q.id DESC LIMIT 1""",
            (run_id, market.ticker),
        ).fetchone()
        previous_quote = (
            QuotePoint(
                observed_at=datetime.fromisoformat(previous_row["observed_at"]),
                up_bid=previous_row["up_bid"], up_ask=previous_row["up_ask"],
                down_bid=previous_row["down_bid"], down_ask=previous_row["down_ask"],
            )
            if previous_row is not None else None
        )
        guard_settings = (
            EntryGuardSettings.model_validate_json(run["guard_snapshot"])
            if run["guard_snapshot"] else None
        )
        try:
            with self.store.connection:
                self.store.connection.execute(
                    "INSERT INTO lab_seen_events(run_id,quote_event_id) VALUES (?,?)",
                    (run_id, quote_event_id),
                )
        except Exception as error:
            import sqlite3
            if isinstance(error, sqlite3.IntegrityError):
                return
            raise
        candidates = list(self.store.connection.execute(
            "SELECT * FROM lab_candidates WHERE run_id=? AND asset=?",
            (run_id, market.asset),
        ))
        for candidate in candidates:
            self._apply_candidate(
                run_id, candidate, market, quote, quote_event_id,
                guard_settings, previous_quote,
            )
        with self.store.connection:
            self.store.connection.execute(
                "UPDATE lab_runs SET quote_count=quote_count+1 WHERE id=?", (run_id,)
            )

    def history_cycle_candidates(
        self, assets: list[str], cutoff_event_id: int, at: datetime
    ) -> list[dict[str, str]]:
        if cutoff_event_id <= 0:
            return []
        placeholders = ",".join("?" for _ in assets)
        rows = list(self.store.connection.execute(
            f"""SELECT asset,ticker,MAX(close_time) AS close_time
            FROM quote_events WHERE id<=? AND asset IN ({placeholders})
            AND close_time IS NOT NULL AND close_time<=?
            GROUP BY asset,ticker ORDER BY asset,close_time DESC,ticker DESC""",
            (cutoff_event_id, *assets, at.isoformat()),
        ))
        return [{
            "asset": str(row["asset"]), "ticker": str(row["ticker"]),
            "close_time": str(row["close_time"]),
        } for row in rows]

    def history_tickers(
        self, assets: list[str], cycles: int, cutoff_event_id: int, now: datetime
    ) -> list[str]:
        if cycles <= 0:
            return []
        selected: list[str] = []
        counts = {asset: 0 for asset in assets}
        for item in self.history_cycle_candidates(assets, cutoff_event_id, now):
            asset, ticker = item["asset"], item["ticker"]
            if counts[asset] < cycles and self.store.market_result(ticker) is not None:
                selected.append(ticker)
                counts[asset] += 1
        return selected

    async def backfill(self, run_id: int, *, now: datetime | None = None) -> None:
        run = self.store.connection.execute(
            "SELECT * FROM lab_runs WHERE id=?", (run_id,)
        ).fetchone()
        if run is None:
            raise KeyError(run_id)
        if run["status"] != "backfilling":
            return
        if run["backfill_engine"] == "vectorized":
            from kaishi_bot.lab_vectorized import vectorized_backfill

            await vectorized_backfill(self, run_id, now=now)
            return
        selection_time = now or datetime.fromisoformat(str(run["started_at"]))
        try:
            tickers = self.history_tickers(
                json.loads(run["assets"]), int(run["history_cycles_requested"]),
                int(run["history_cutoff_event_id"] or 0), selection_time,
            )
            results = {
                ticker: self.store.market_result(ticker) for ticker in tickers
            }
            tickers = [ticker for ticker in tickers if results[ticker] is not None]
            rows: list[object] = []
            if tickers:
                placeholders = ",".join("?" for _ in tickers)
                rows = list(self.store.connection.execute(
                    f"""SELECT * FROM quote_events WHERE id<=?
                    AND ticker IN ({placeholders}) ORDER BY id""",
                    (int(run["history_cutoff_event_id"] or 0), *tickers),
                ))
            with self.store.connection:
                self.store.connection.execute(
                    """UPDATE lab_runs SET history_cycles_loaded=?,
                    history_events_total=? WHERE id=?""",
                    (len(tickers), len(rows), run_id),
                )
            settlements = sorted(
                (
                    datetime.fromisoformat(str(results[ticker]["event_time"])), ticker,
                    str(results[ticker]["winning_side"]),
                )
                for ticker in tickers
            )
            settlement_index = 0
            for index, row in enumerate(rows, start=1):
                status = self.store.connection.execute(
                    "SELECT status FROM lab_runs WHERE id=?", (run_id,)
                ).fetchone()[0]
                if status != "backfilling":
                    return
                observed = datetime.fromisoformat(str(row["observed_at"]))
                while (
                    settlement_index < len(settlements)
                    and settlements[settlement_index][0] <= observed
                ):
                    event_time, ticker, winning_side = settlements[settlement_index]
                    self.settle_ticker(run_id, ticker, winning_side, event_time)
                    settlement_index += 1
                close_time = datetime.fromisoformat(str(row["close_time"]))
                market = AssetMarket(
                    asset=str(row["asset"]), series="history",
                    ticker=str(row["ticker"]), title="Logged history",
                    open_time=close_time - timedelta(minutes=15), close_time=close_time,
                )
                quote = QuotePoint(
                    observed_at=observed, up_bid=row["up_bid"], up_ask=row["up_ask"],
                    down_bid=row["down_bid"], down_ask=row["down_ask"],
                )
                self.on_quote(run_id, market, quote, int(row["id"]))
                with self.store.connection:
                    self.store.connection.execute(
                        """UPDATE lab_runs SET history_events_processed=?
                        WHERE id=?""", (index, run_id),
                    )
                await asyncio.sleep(0)
            for event_time, ticker, winning_side in settlements[settlement_index:]:
                status = self.store.connection.execute(
                    "SELECT status FROM lab_runs WHERE id=?", (run_id,)
                ).fetchone()[0]
                if status != "backfilling":
                    return
                self.settle_ticker(run_id, ticker, winning_side, event_time)
            status = self.store.connection.execute(
                "SELECT status FROM lab_runs WHERE id=?", (run_id,)
            ).fetchone()[0]
            if status != "backfilling":
                return
            with self.store.connection:
                realtime_start = now or datetime.now(UTC)
                self.store.connection.execute(
                    """UPDATE lab_runs SET status='running',started_at=?
                    WHERE id=? AND status='backfilling'""",
                    (realtime_start.isoformat(), run_id),
                )
        except Exception as error:
            with self.store.connection:
                self.store.connection.execute(
                    """UPDATE lab_runs SET status='failed',error_message=?
                    WHERE id=? AND status='backfilling'""", (str(error), run_id),
                )
            raise

    def _schedule(self, run_id: int, asset: str) -> FeeSchedule:
        row = self.store.connection.execute(
            "SELECT fee_snapshot FROM lab_runs WHERE id=?", (run_id,)
        ).fetchone()
        if row is None or not row[0]:
            raise ValueError(f"run {run_id} has no verified fee schedule")
        payload = json.loads(row[0])[asset]
        return FeeSchedule(
            str(payload["fee_type"]), Decimal(str(payload["multiplier"])),
            Decimal(str(payload["taker_rate"])), str(payload["version"]),
        )

    def _apply_candidate(
        self, run_id: int, candidate: object, market: AssetMarket,
        quote: QuotePoint, quote_event_id: int,
        guard_settings: EntryGuardSettings | None = None,
        previous_quote: QuotePoint | None = None,
    ) -> None:
        cid = candidate["candidate_id"]
        tp = Decimal(candidate["take_profit"])
        sl = Decimal(candidate["stop_loss"])
        schedule = self._schedule(run_id, market.asset)
        entry_min = self._candidate_entry_min(
            candidate, guard_settings, preserve_legacy=True
        )
        with self.store.connection:
            position = self.store.connection.execute(
                """SELECT * FROM lab_positions WHERE run_id=? AND candidate_id=?
                AND status='open' ORDER BY id LIMIT 1""", (run_id, cid),
            ).fetchone()
            if position is not None:
                if position["ticker"] != market.ticker:
                    return
                bid = quote.up_bid if position["side"] == "up" else quote.down_bid
                reason = "take_profit" if bid >= tp else (
                    "stop_loss" if bid <= sl else None
                )
                if reason:
                    qty = Decimal(position["quantity"])
                    gross = qty * bid
                    exit_fee = taker_fee(schedule, qty, bid)
                    net = gross - exit_fee
                    entry_outlay = Decimal(
                        position["entry_outlay"]
                        or Decimal(position["entry_cost"]) + Decimal(position["entry_fee"])
                    )
                    pnl = net - entry_outlay
                    cash = Decimal(candidate["cash"]) + net
                    wins = int(candidate["wins"]) + int(pnl > 0)
                    closed = int(candidate["closed_trades"]) + 1
                    realized = Decimal(candidate["realized_pnl"]) + pnl
                    total_fees = Decimal(candidate["total_fees"]) + exit_fee
                    count_column = "tp_count" if reason == "take_profit" else "sl_count"
                    self.store.connection.execute(
                        """UPDATE lab_positions SET status='closed',exit_price=?,pnl=?,
                        exit_fee=?,gross_proceeds=?,net_proceeds=?,exit_event_id=?,close_reason=?
                        WHERE id=?""",
                        (str(bid), str(pnl), str(exit_fee), str(gross), str(net),
                         quote_event_id, reason, int(position["id"])),
                    )
                    self.store.connection.execute(
                        f"""UPDATE lab_candidates SET cash=?,realized_pnl=?,
                        closed_trades=?,wins=?,total_fees=?,{count_column}={count_column}+1,
                        last_closed_ticker=?,last_closed_side=?,last_closed_at=?,cooldown_until=?
                        WHERE run_id=? AND candidate_id=?""",
                        (str(cash), str(realized), closed, wins, str(total_fees),
                         market.ticker, position["side"], quote.observed_at.isoformat(),
                         (quote.observed_at + timedelta(seconds=(
                             guard_settings.reentry_cooldown_seconds
                             if guard_settings else 0
                         ))).isoformat(), run_id, cid),
                    )
                    self._update_drawdown(run_id, cid, quote, schedule)
                    return

                policy = candidate["side_policy"]
                qualifying = any(
                    policy in (side, "both")
                    and entry_min <= ask <= Decimal(candidate["entry_price"])
                    for side, ask in (("up", quote.up_ask), ("down", quote.down_ask))
                )
                if qualifying:
                    self.store.connection.execute(
                        """UPDATE lab_candidates SET skipped_open=skipped_open+1
                        WHERE run_id=? AND candidate_id=?""", (run_id, cid),
                    )
                self._update_drawdown(run_id, cid, quote, schedule)
                return

            policy = candidate["side_policy"]
            if quote.observed_at >= market.close_time:
                self._update_drawdown(run_id, cid, quote, schedule)
                return
            price_band_eligible = any(
                policy in (side, "both")
                and entry_min <= ask <= Decimal(candidate["entry_price"])
                for side, ask in (("up", quote.up_ask), ("down", quote.down_ask))
            )
            if price_band_eligible:
                self.store.connection.execute(
                    """INSERT OR IGNORE INTO lab_candidate_eligible_cycles(
                    run_id,candidate_id,ticker) VALUES (?,?,?)""",
                    (run_id, cid, market.ticker),
                )
            day = quote.observed_at.astimezone(NY).date().isoformat()
            spend_row = self.store.connection.execute(
                "SELECT amount FROM lab_daily_spend WHERE run_id=? AND candidate_id=? AND day=?",
                (run_id, cid, day),
            ).fetchone()
            spent = Decimal(spend_row[0]) if spend_row else Decimal("0")
            cash = Decimal(candidate["cash"])
            budget = min(Decimal("1.00"), Decimal("50.00") - spent, cash)
            guarded_decisions = []
            if guard_settings is not None:
                for side, ask, bid in (
                    ("up", quote.up_ask, quote.up_bid),
                    ("down", quote.down_ask, quote.down_bid),
                ):
                    if policy not in (side, "both"):
                        continue
                    previous = None
                    if previous_quote is not None:
                        previous = GuardQuote(
                            bid=previous_quote.up_bid if side == "up" else previous_quote.down_bid,
                            ask=previous_quote.up_ask if side == "up" else previous_quote.down_ask,
                            observed_at=previous_quote.observed_at,
                        )
                    cooldown_until = None
                    if (
                        candidate["last_closed_ticker"] == market.ticker
                        and candidate["last_closed_side"] == side
                        and candidate["cooldown_until"]
                    ):
                        cooldown_until = datetime.fromisoformat(candidate["cooldown_until"])
                    decision = evaluate_entry(
                        previous=previous,
                        current=GuardQuote(bid=bid, ask=ask, observed_at=quote.observed_at),
                        entry_min=entry_min,
                        entry_price=Decimal(candidate["entry_price"]),
                        stop_loss=sl, take_profit=tp, budget=budget,
                        settings=guard_settings, fee_schedule=schedule,
                        cooldown_until=cooldown_until,
                    )
                    guarded_decisions.append((ask, 0 if side == "up" else 1, side, decision))
                eligible = [
                    (ask, order, side, decision)
                    for ask, order, side, decision in guarded_decisions
                    if decision.eligible
                ]
                if not eligible and guarded_decisions:
                    reason = min(guarded_decisions)[3].reason.value
                    column = f"guard_{reason}"
                    if column in {
                        "guard_below_floor", "guard_bid_in_sl_buffer",
                        "guard_spread_too_wide", "guard_confirmation_pending",
                        "guard_reward_risk_too_low", "guard_cooldown_active",
                    }:
                        self.store.connection.execute(
                            f"UPDATE lab_candidates SET {column}={column}+1 WHERE run_id=? AND candidate_id=?",
                            (run_id, cid),
                        )
            else:
                eligible = [
                    (ask, 0 if side == "up" else 1, side, None)
                    for side, ask in (("up", quote.up_ask), ("down", quote.down_ask))
                    if policy in (side, "both")
                    and entry_min <= ask <= Decimal(candidate["entry_price"])
                ]
            if eligible:
                ask, _, side, decision = min(eligible)
                if decision is not None:
                    qty, premium, entry_fee = (
                        decision.quantity, decision.premium, decision.entry_fee
                    )
                else:
                    qty, premium, entry_fee = whole_contract_size(schedule, ask, budget)
                if qty <= 0:
                    self.store.connection.execute(
                        """UPDATE lab_candidates SET blocked_daily_cap=blocked_daily_cap+1
                        WHERE run_id=? AND candidate_id=?""", (run_id, cid),
                    )
                    self._update_drawdown(run_id, cid, quote, schedule)
                    return
                outlay = premium + entry_fee
                self.store.connection.execute(
                    """INSERT INTO lab_positions(
                    run_id,candidate_id,ticker,side,quantity,entry_price,entry_cost,
                    entry_fee,entry_outlay,entry_event_id
                    ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (run_id, cid, market.ticker, side, str(qty), str(ask),
                     str(premium), str(entry_fee), str(outlay), quote_event_id),
                )
                self.store.connection.execute(
                    """UPDATE lab_candidates SET cash=?,total_fees=?,entry_count=entry_count+1
                    WHERE run_id=? AND candidate_id=?""",
                    (str(cash - outlay), str(Decimal(candidate["total_fees"]) + entry_fee),
                     run_id, cid),
                )
                self.store.connection.execute(
                    """INSERT INTO lab_daily_spend(run_id,candidate_id,day,amount) VALUES (?,?,?,?)
                    ON CONFLICT(run_id,candidate_id,day) DO UPDATE SET amount=excluded.amount""",
                    (run_id, cid, day, str(spent + outlay)),
                )
            self._update_drawdown(run_id, cid, quote, schedule)

    def _update_drawdown(
        self, run_id: int, cid: str, quote: QuotePoint, schedule: FeeSchedule
    ) -> None:
        candidate = self.store.connection.execute(
            "SELECT * FROM lab_candidates WHERE run_id=? AND candidate_id=?", (run_id, cid)
        ).fetchone()
        positions = self.store.connection.execute(
            "SELECT * FROM lab_positions WHERE run_id=? AND candidate_id=? AND status='open'",
            (run_id, cid),
        )
        equity = Decimal(candidate["cash"])
        for position in positions:
            mark = quote.up_bid if position["side"] == "up" else quote.down_bid
            quantity = Decimal(position["quantity"])
            equity += quantity * mark - taker_fee(schedule, quantity, mark)
        peak = max(Decimal(candidate["peak_equity"]), equity)
        drawdown = (peak - equity) / Decimal("1000") * Decimal("100")
        maximum = max(Decimal(candidate["max_drawdown"]), drawdown)
        self.store.connection.execute(
            "UPDATE lab_candidates SET peak_equity=?,max_drawdown=? WHERE run_id=? AND candidate_id=?",
            (str(peak), str(maximum), run_id, cid),
        )

    def settle_ticker(
        self, run_id: int, ticker: str, winning_side: str, event_time: datetime
    ) -> int:
        settled = 0
        with self.store.connection:
            settlement = self.store.connection.execute(
                """INSERT OR IGNORE INTO lab_settlement_events(
                run_id,ticker,winning_side,event_time) VALUES (?,?,?,?)""",
                (run_id, ticker, winning_side, event_time.isoformat()),
            )
            if settlement.rowcount == 0:
                return 0
            positions = list(self.store.connection.execute(
                """SELECT * FROM lab_positions WHERE run_id=? AND ticker=?
                AND status='open' ORDER BY id""", (run_id, ticker),
            ))
            for position in positions:
                candidate = self.store.connection.execute(
                    """SELECT * FROM lab_candidates WHERE run_id=? AND candidate_id=?""",
                    (run_id, position["candidate_id"]),
                ).fetchone()
                if candidate is None:
                    continue
                payout = Decimal("1") if position["side"] == winning_side else Decimal("0")
                quantity = Decimal(position["quantity"])
                gross = quantity * payout
                entry_outlay = Decimal(
                    position["entry_outlay"]
                    or Decimal(position["entry_cost"]) + Decimal(position["entry_fee"])
                )
                pnl = gross - entry_outlay
                cash = Decimal(candidate["cash"]) + gross
                self.store.connection.execute(
                    """UPDATE lab_positions SET status='closed',exit_price=?,pnl=?,
                    exit_fee='0',gross_proceeds=?,net_proceeds=?,close_reason='settlement'
                    WHERE id=?""",
                    (str(payout), str(pnl), str(gross), str(gross), int(position["id"])),
                )
                self.store.connection.execute(
                    """UPDATE lab_candidates SET cash=?,realized_pnl=?,
                    closed_trades=closed_trades+1,wins=wins+?,
                    settlement_count=settlement_count+1
                    WHERE run_id=? AND candidate_id=?""",
                    (str(cash), str(Decimal(candidate["realized_pnl"]) + pnl),
                     int(pnl > 0), run_id, position["candidate_id"]),
                )
                settled += 1
        return settled

    def leaderboard(self, run_id: int) -> dict[str, object]:
        run = self.store.connection.execute(
            "SELECT * FROM lab_runs WHERE id=?", (run_id,)
        ).fetchone()
        if run is None:
            raise KeyError(run_id)
        run_guard = (
            EntryGuardSettings.model_validate_json(run["guard_snapshot"])
            if run["guard_snapshot"] else None
        )
        rows = list(self.store.connection.execute(
            "SELECT * FROM lab_candidates WHERE run_id=?", (run_id,)
        ))
        legacy_rules = any(
            row["entry_min"] is None or int(row["min_seconds"]) > 180
            for row in rows
        )
        fee_payload = json.loads(run["fee_snapshot"])
        schedules = {
            asset: FeeSchedule(
                payload["fee_type"], Decimal(payload["multiplier"]),
                Decimal(payload["taker_rate"]), payload["version"],
            )
            for asset, payload in fee_payload.items()
        }
        aggregate_quantum = Decimal("0.000000000001")
        position_totals = {
            str(item["candidate_id"]): (
                Decimal(str(item["deployed"] or 0)).quantize(aggregate_quantum),
                Decimal(str(item["gains"] or 0)).quantize(aggregate_quantum),
                Decimal(str(item["losses"] or 0)).quantize(aggregate_quantum),
            )
            for item in self.store.connection.execute(
                """SELECT candidate_id,
                SUM(CASE WHEN entry_outlay IS NOT NULL
                    THEN CAST(entry_outlay AS REAL)
                    ELSE CAST(entry_cost AS REAL)+CAST(entry_fee AS REAL) END) AS deployed,
                SUM(CASE WHEN status='closed' AND CAST(pnl AS REAL)>0
                    THEN CAST(pnl AS REAL) ELSE 0 END) gains,
                -SUM(CASE WHEN status='closed' AND CAST(pnl AS REAL)<0
                    THEN CAST(pnl AS REAL) ELSE 0 END) losses
                FROM lab_positions WHERE run_id=? GROUP BY candidate_id""",
                (run_id,),
            )
        }
        deployed_by_candidate = {
            candidate_id: values[0]
            for candidate_id, values in position_totals.items()
        }
        open_positions: dict[str, list[object]] = {}
        for position in self.store.connection.execute(
            "SELECT * FROM lab_positions WHERE run_id=? AND status='open'", (run_id,)
        ):
            candidate_id = str(position["candidate_id"])
            open_positions.setdefault(candidate_id, []).append(position)
        latest_marks = {
            str(mark["ticker"]): mark
            for mark in self.store.connection.execute(
                """WITH latest AS (
                    SELECT q.ticker,MAX(q.id) AS quote_event_id
                    FROM quote_events q JOIN lab_seen_events s ON s.quote_event_id=q.id
                    WHERE s.run_id=? GROUP BY q.ticker
                )
                SELECT q.ticker,q.up_bid,q.down_bid FROM latest
                JOIN quote_events q ON q.id=latest.quote_event_id""", (run_id,),
            )
        }
        cycles_by_asset = {
            str(item["asset"]): int(item["cycle_count"])
            for item in self.store.connection.execute(
                """SELECT q.asset,COUNT(DISTINCT q.ticker) AS cycle_count
                FROM quote_events q JOIN lab_seen_events s ON s.quote_event_id=q.id
                WHERE s.run_id=? GROUP BY q.asset""", (run_id,),
            )
        }
        eligible_cycles_by_candidate = {
            str(item["candidate_id"]): int(item["cycle_count"])
            for item in self.store.connection.execute(
                """SELECT candidate_id,COUNT(*) AS cycle_count
                FROM lab_candidate_eligible_cycles WHERE run_id=?
                GROUP BY candidate_id""", (run_id,),
            )
        }
        candidates = []
        for row in rows:
            cash = Decimal(row["cash"])
            liquidation = Decimal("0")
            schedule = schedules[str(row["asset"])]
            for position in open_positions.get(str(row["candidate_id"]), []):
                latest = latest_marks.get(str(position["ticker"]))
                mark = (
                    Decimal(latest["up_bid"] if position["side"] == "up" else latest["down_bid"])
                    if latest else Decimal(position["entry_price"])
                )
                quantity = Decimal(position["quantity"])
                liquidation += quantity * mark - taker_fee(schedule, quantity, mark)
            equity = cash + liquidation
            net_pnl = equity - Decimal("1000")
            total_deployed = deployed_by_candidate.get(str(row["candidate_id"]), Decimal("0"))
            roi_percent = net_pnl / total_deployed * Decimal("100") if total_deployed else None
            return_percent = net_pnl / Decimal("10")
            max_drawdown = Decimal(row["max_drawdown"])
            closed = int(row["closed_trades"])
            _, gains, losses = position_totals.get(
                str(row["candidate_id"]),
                (Decimal("0"), Decimal("0"), Decimal("0")),
            )
            cycles_seen = cycles_by_asset.get(str(row["asset"]), 0)
            candidates.append({
                "candidate_id": row["candidate_id"], "asset": row["asset"],
                "entry_min": str(self._candidate_entry_min(row, run_guard)),
                "entry": row["entry_price"], "take_profit": row["take_profit"],
                "stop_loss": row["stop_loss"], "side_policy": row["side_policy"],
                "cash": cash, "equity": equity,
                "net_pnl": net_pnl, "total_deployed": total_deployed,
                "roi_percent": roi_percent, "return_percent": return_percent,
                "max_drawdown_percent": max_drawdown,
                "score": return_percent - max_drawdown,
                "closed_trades": closed, "win_rate": Decimal(row["wins"]) / closed if closed else None,
                "profit_factor": gains / losses if losses else (None if not gains else "infinite"),
                "total_fees": Decimal(row["total_fees"]),
                "entry_count": int(row["entry_count"]),
                "tp_count": int(row["tp_count"]),
                "sl_count": int(row["sl_count"]),
                "settlement_count": int(row["settlement_count"]),
                "skipped_open": int(row["skipped_open"]),
                "blocked_daily_cap": int(row["blocked_daily_cap"]),
                "guard_counters": {
                    "below_floor": int(row["guard_below_floor"]),
                    "bid_in_sl_buffer": int(row["guard_bid_in_sl_buffer"]),
                    "spread_too_wide": int(row["guard_spread_too_wide"]),
                    "confirmation_pending": int(row["guard_confirmation_pending"]),
                    "reward_risk_too_low": int(row["guard_reward_risk_too_low"]),
                    "cooldown_active": int(row["guard_cooldown_active"]),
                },
                "cycles_seen": cycles_seen,
                "eligible_cycles": (
                    None if legacy_rules else eligible_cycles_by_candidate.get(
                        str(row["candidate_id"]), 0
                    )
                ),
                "low_confidence": closed < 8,
            })
        candidates.sort(key=lambda item: (
            -item["score"], -item["net_pnl"],
            -item["closed_trades"], item["candidate_id"],
        ))
        minimum_closed_trades = 8
        ranked_candidates = [
            item for item in candidates
            if item["closed_trades"] >= minimum_closed_trades
        ]
        insufficient_candidates = [
            item for item in candidates
            if item["closed_trades"] < minimum_closed_trades
        ]
        return {
            "run_id": run_id,
            "status": run["status"],
            "preliminary": True,
            "minimum_closed_trades": minimum_closed_trades,
            "legacy_rules": legacy_rules,
            "ranked_candidates": ranked_candidates,
            "insufficient_candidates": insufficient_candidates,
            "candidates": ranked_candidates + insufficient_candidates,
        }

    def replay(self, run_id: int) -> int:
        source = self.store.connection.execute(
            "SELECT * FROM lab_runs WHERE id=?", (run_id,)
        ).fetchone()
        if source is None:
            raise KeyError(run_id)
        assets = json.loads(source["assets"])
        replay_id = self.start(
            assets, int(source["candidate_count"]), int(source["seed"]),
            int(source["duration_seconds"]), {
                asset: FeeSchedule(
                    payload["fee_type"], Decimal(payload["multiplier"]),
                    Decimal(payload["taker_rate"]), payload["version"],
                )
                for asset, payload in json.loads(source["fee_snapshot"]).items()
            },
            EntryGuardSettings.model_validate_json(source["guard_snapshot"])
            if source["guard_snapshot"] else None,
            _run_status="replaying",
        )
        try:
            source_settings = list(self.store.connection.execute(
                """SELECT candidate_id,entry_min,entry_price,take_profit,
                stop_loss,min_seconds,side_policy FROM lab_candidates
                WHERE run_id=?""", (run_id,),
            ))
            with self.store.connection:
                self.store.connection.executemany(
                    """UPDATE lab_candidates SET entry_min=?,entry_price=?,
                    take_profit=?,stop_loss=?,min_seconds=?,side_policy=?
                    WHERE run_id=? AND candidate_id=?""",
                    ((row["entry_min"], row["entry_price"], row["take_profit"],
                      row["stop_loss"], row["min_seconds"], row["side_policy"],
                      replay_id, row["candidate_id"])
                     for row in source_settings),
                )
            rows = list(self.store.connection.execute(
                """SELECT q.* FROM quote_events q JOIN lab_seen_events s
                ON s.quote_event_id=q.id WHERE s.run_id=? ORDER BY q.id""", (run_id,),
            ))
            settlements = list(self.store.connection.execute(
                """SELECT * FROM lab_settlement_events WHERE run_id=?
                ORDER BY event_time,ticker""", (run_id,),
            ))
            if rows:
                with self.store.connection:
                    self.store.connection.execute(
                        "UPDATE lab_runs SET started_at=? WHERE id=?",
                        (rows[0]["observed_at"], replay_id),
                    )
            settlement_index = 0
            for row in rows:
                if not row["close_time"]:
                    continue
                observed = datetime.fromisoformat(row["observed_at"])
                while (
                    settlement_index < len(settlements)
                    and datetime.fromisoformat(settlements[settlement_index]["event_time"]) <= observed
                ):
                    event = settlements[settlement_index]
                    self.settle_ticker(
                        replay_id, event["ticker"], event["winning_side"],
                        datetime.fromisoformat(event["event_time"]),
                    )
                    settlement_index += 1
                close_time = datetime.fromisoformat(row["close_time"])
                market = AssetMarket(
                    asset=row["asset"], series="replay", ticker=row["ticker"],
                    title="Replay", open_time=close_time - timedelta(minutes=15),
                    close_time=close_time,
                )
                quote = QuotePoint(
                    observed_at=observed, up_bid=row["up_bid"], up_ask=row["up_ask"],
                    down_bid=row["down_bid"], down_ask=row["down_ask"],
                )
                self.on_quote(replay_id, market, quote, int(row["id"]))
            for event in settlements[settlement_index:]:
                self.settle_ticker(
                    replay_id, event["ticker"], event["winning_side"],
                    datetime.fromisoformat(event["event_time"]),
                )
            with self.store.connection:
                self.store.connection.execute(
                    "UPDATE lab_runs SET status='completed' WHERE id=?", (replay_id,)
                )
        except Exception:
            with self.store.connection:
                self.store.connection.execute(
                    "UPDATE lab_runs SET status='failed' WHERE id=?", (replay_id,)
                )
            raise
        return replay_id

    def candidate_settings(self, run_id: int, candidate_id: str) -> dict[str, object]:
        row = self.store.connection.execute(
            """SELECT * FROM lab_candidates WHERE run_id=? AND candidate_id=?""",
            (run_id, candidate_id),
        ).fetchone()
        if row is None:
            raise KeyError(candidate_id)
        run = self.store.connection.execute(
            "SELECT guard_snapshot FROM lab_runs WHERE id=?", (run_id,)
        ).fetchone()
        guard = (
            EntryGuardSettings.model_validate_json(run["guard_snapshot"])
            if run and run["guard_snapshot"]
            else EntryGuardSettings()
        )
        return {
            "stop_loss": row["stop_loss"],
            "entry_min": str(self._candidate_entry_min(row, guard)),
            "entry_price": row["entry_price"],
            "take_profit": row["take_profit"],
            "asset": row["asset"], "side_policy": row["side_policy"],
        }

    @staticmethod
    def _candidate_entry_min(
        candidate: object, guard_settings: EntryGuardSettings | None,
        *, preserve_legacy: bool = False,
    ) -> Decimal:
        stored = candidate["entry_min"]
        if stored is not None:
            return Decimal(stored)
        if preserve_legacy and guard_settings is None:
            return Decimal("0")
        guard = guard_settings or EntryGuardSettings()
        return max(
            Decimal(candidate["entry_price"]) * guard.entry_floor_ratio,
            Decimal(candidate["stop_loss"]) + guard.stop_loss_buffer,
        )
