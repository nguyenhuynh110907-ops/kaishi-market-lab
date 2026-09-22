from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

import numpy as np

from kaishi_bot.entry_guard import EntryGuardSettings
from kaishi_bot.fees import FeeSchedule, taker_fee, whole_contract_size

if TYPE_CHECKING:
    from kaishi_bot.lab import StrategyLab


NY_TZ = __import__("zoneinfo").ZoneInfo("America/New_York")


def _fee_cents_array(
    schedule: FeeSchedule, quantity: np.ndarray, price: np.ndarray,
) -> np.ndarray:
    raw = (
        float(schedule.taker_rate)
        * float(schedule.multiplier)
        * quantity
        * price
        * (1.0 - price)
    )
    return np.ceil(raw * 100.0 - 1e-10).astype(np.int64)


class VectorizedAssetEngine:
    """Exact event-order simulator with vector selection and batched persistence."""

    def __init__(
        self, lab: StrategyLab, run_id: int, asset: str,
        schedule: FeeSchedule, guard: EntryGuardSettings | None,
    ) -> None:
        self.lab = lab
        self.store = lab.store
        self.run_id = run_id
        self.asset = asset
        self.schedule = schedule
        self.guard = guard
        rows = list(self.store.connection.execute(
            "SELECT * FROM lab_candidates WHERE run_id=? AND asset=? ORDER BY candidate_id",
            (run_id, asset),
        ))
        self.n = len(rows)
        self.ids = [str(row["candidate_id"]) for row in rows]
        self.index_by_id = {
            candidate_id: index for index, candidate_id in enumerate(self.ids)
        }
        self.entry_min = np.array([
            float(lab._candidate_entry_min(row, guard, preserve_legacy=True)) for row in rows
        ])
        self.entry_max = np.array([float(row["entry_price"]) for row in rows])
        self.tp = np.array([float(row["take_profit"]) for row in rows])
        self.sl = np.array([float(row["stop_loss"]) for row in rows])
        self.min_seconds = np.array([int(row["min_seconds"]) for row in rows])
        self.policy = np.array([
            0 if row["side_policy"] == "up" else 1 if row["side_policy"] == "down" else 2
            for row in rows
        ], dtype=np.int8)

        self.cash = [Decimal(str(row["cash"])) for row in rows]
        self.realized = [Decimal(str(row["realized_pnl"])) for row in rows]
        self.total_fees = [Decimal(str(row["total_fees"])) for row in rows]
        self.peak_cents = np.array([
            int(Decimal(str(row["peak_equity"])) * 100) for row in rows
        ], dtype=np.int64)
        self.max_dd_milli = np.array([
            int(Decimal(str(row["max_drawdown"])) * 1000) for row in rows
        ], dtype=np.int64)
        self.closed = np.array([int(row["closed_trades"]) for row in rows], dtype=np.int64)
        self.wins = np.array([int(row["wins"]) for row in rows], dtype=np.int64)
        self.entry_count = np.array([int(row["entry_count"]) for row in rows], dtype=np.int64)
        self.tp_count = np.array([int(row["tp_count"]) for row in rows], dtype=np.int64)
        self.sl_count = np.array([int(row["sl_count"]) for row in rows], dtype=np.int64)
        self.settlement_count = np.array(
            [int(row["settlement_count"]) for row in rows], dtype=np.int64
        )
        self.skipped_open = np.array([int(row["skipped_open"]) for row in rows], dtype=np.int64)
        self.blocked_cap = np.array(
            [int(row["blocked_daily_cap"]) for row in rows], dtype=np.int64
        )
        self.guard_counts = {
            name: np.array([int(row[f"guard_{name}"]) for row in rows], dtype=np.int64)
            for name in (
                "below_floor", "bid_in_sl_buffer", "spread_too_wide",
                "confirmation_pending", "reward_risk_too_low", "cooldown_active",
            )
        }
        self.last_ticker = np.array([row["last_closed_ticker"] or "" for row in rows], dtype=object)
        self.last_side = np.array([
            0 if row["last_closed_side"] == "up" else 1 if row["last_closed_side"] == "down" else -1
            for row in rows
        ], dtype=np.int8)
        self.last_closed_at = [row["last_closed_at"] for row in rows]
        self.cooldown_until = np.array([
            datetime.fromisoformat(row["cooldown_until"]).timestamp()
            if row["cooldown_until"] else 0.0 for row in rows
        ])

        self.open = np.zeros(self.n, dtype=bool)
        self.open_side = np.full(self.n, -1, dtype=np.int8)
        self.open_ticker = np.full(self.n, "", dtype=object)
        self.quantity = np.zeros(self.n, dtype=np.int64)
        self.open_outlay = [Decimal("0") for _ in range(self.n)]
        self.trades: list[dict[str, object]] = []
        self.trade_for_candidate = np.full(self.n, -1, dtype=np.int64)
        self.eligible: set[tuple[str, str]] = set()
        self.spend: dict[str, list[Decimal]] = {}
        for position in self.store.connection.execute(
            """SELECT p.* FROM lab_positions p JOIN lab_candidates c
            ON c.run_id=p.run_id AND c.candidate_id=p.candidate_id
            WHERE p.run_id=? AND c.asset=? AND p.status='open'""",
            (run_id, asset),
        ):
            index = self.index_by_id[str(position["candidate_id"])]
            trade = dict(position)
            trade.update(persisted=True, dirty=False)
            self.trades.append(trade)
            self.trade_for_candidate[index] = len(self.trades) - 1
            self.open[index] = True
            self.open_side[index] = 0 if position["side"] == "up" else 1
            self.open_ticker[index] = str(position["ticker"])
            self.quantity[index] = float(position["quantity"])
            self.open_outlay[index] = Decimal(
                position["entry_outlay"]
                or Decimal(position["entry_cost"]) + Decimal(position["entry_fee"])
            )
        for spend_row in self.store.connection.execute(
            """SELECT s.candidate_id,s.day,s.amount FROM lab_daily_spend s
            JOIN lab_candidates c ON c.run_id=s.run_id AND c.candidate_id=s.candidate_id
            WHERE s.run_id=? AND c.asset=?""", (run_id, asset),
        ):
            values = self.spend.setdefault(
                str(spend_row["day"]), [Decimal("0") for _ in range(self.n)]
            )
            values[self.index_by_id[str(spend_row["candidate_id"])]] = Decimal(
                spend_row["amount"]
            )

    def _band(self, side: int, ask: float) -> np.ndarray:
        policy_ok = (self.policy == side) | (self.policy == 2)
        return policy_ok & (ask >= self.entry_min) & (ask <= self.entry_max)

    def process(self, row: object) -> None:
        ticker = str(row["ticker"])
        observed = datetime.fromisoformat(str(row["observed_at"]))
        event_id = int(row["id"])
        up_bid_decimal, up_ask_decimal = Decimal(row["up_bid"]), Decimal(row["up_ask"])
        down_bid_decimal, down_ask_decimal = Decimal(row["down_bid"]), Decimal(row["down_ask"])
        up_bid, up_ask = float(up_bid_decimal), float(up_ask_decimal)
        down_bid, down_ask = float(down_bid_decimal), float(down_ask_decimal)
        bids = np.where(self.open_side == 0, up_bid, down_bid)

        same_ticker = self.open & (self.open_ticker == ticker)
        close_tp = same_ticker & (bids >= self.tp)
        close_sl = same_ticker & ~close_tp & (bids <= self.sl)
        closed_now = close_tp | close_sl
        for index in np.flatnonzero(closed_now):
            reason = "take_profit" if close_tp[index] else "stop_loss"
            price = Decimal(str(row["up_bid"] if self.open_side[index] == 0 else row["down_bid"]))
            qty = Decimal(str(int(self.quantity[index])))
            gross = qty * price
            fee = taker_fee(self.schedule, qty, price)
            net = gross - fee
            outlay = self.open_outlay[index]
            pnl = net - outlay
            self.cash[index] += net
            self.realized[index] += pnl
            self.total_fees[index] += fee
            self.closed[index] += 1
            self.wins[index] += int(pnl > 0)
            if reason == "take_profit":
                self.tp_count[index] += 1
            else:
                self.sl_count[index] += 1
            trade = self.trades[int(self.trade_for_candidate[index])]
            trade.update(
                status="closed", exit_price=str(price), pnl=str(pnl), exit_fee=str(fee),
                gross_proceeds=str(gross), net_proceeds=str(net), exit_event_id=event_id,
                close_reason=reason,
            )
            if trade["persisted"]:
                trade["dirty"] = True
            self.last_ticker[index] = ticker
            self.last_side[index] = self.open_side[index]
            self.last_closed_at[index] = observed.isoformat()
            seconds = self.guard.reentry_cooldown_seconds if self.guard else 0
            self.cooldown_until[index] = (observed + timedelta(seconds=seconds)).timestamp()
            self.open[index] = False
            self.open_side[index] = -1
            self.open_ticker[index] = ""
            self.trade_for_candidate[index] = -1

        band_up = self._band(0, up_ask)
        band_down = self._band(1, down_ask)
        still_open = self.open & (self.open_ticker == ticker)
        self.skipped_open += (still_open & (band_up | band_down)).astype(np.int64)

        close_time = datetime.fromisoformat(str(row["close_time"]))
        idle = ~self.open & ~closed_now & (observed < close_time)
        price_band = idle & (band_up | band_down)
        for index in np.flatnonzero(price_band):
            self.eligible.add((self.ids[index], ticker))

        eligible_up = idle & band_up
        eligible_down = idle & band_down
        if self.guard is not None:
            up_spread_ok = up_ask_decimal - up_bid_decimal <= min(
                self.guard.max_spread, self.guard.max_spread_ratio * up_ask_decimal
            )
            down_spread_ok = down_ask_decimal - down_bid_decimal <= min(
                self.guard.max_spread, self.guard.max_spread_ratio * down_ask_decimal
            )
            up_cooldown = (
                (self.last_ticker == ticker) & (self.last_side == 0)
                & (observed.timestamp() < self.cooldown_until)
            )
            down_cooldown = (
                (self.last_ticker == ticker) & (self.last_side == 1)
                & (observed.timestamp() < self.cooldown_until)
            )
            allowed_up = idle & ((self.policy == 0) | (self.policy == 2))
            allowed_down = idle & ((self.policy == 1) | (self.policy == 2))
            reason_up = np.where(
                up_ask < self.entry_min, 0,
                np.where(up_ask > self.entry_max, 1,
                         np.where(not up_spread_ok, 2, np.where(up_cooldown, 3, 4))),
            )
            reason_down = np.where(
                down_ask < self.entry_min, 0,
                np.where(down_ask > self.entry_max, 1,
                         np.where(not down_spread_ok, 2, np.where(down_cooldown, 3, 4))),
            )
            eligible_up &= up_spread_ok & ~up_cooldown
            eligible_down &= down_spread_ok & ~down_cooldown
            no_eligible = idle & ~(eligible_up | eligible_down)
            choose_reason_up = allowed_up & (~allowed_down | (up_ask <= down_ask))
            chosen_reason = np.where(choose_reason_up, reason_up, reason_down)
            self.guard_counts["below_floor"] += (no_eligible & (chosen_reason == 0)).astype(np.int64)
            self.guard_counts["confirmation_pending"] += (
                no_eligible & (chosen_reason == 1)
            ).astype(np.int64)
            self.guard_counts["spread_too_wide"] += (
                no_eligible & (chosen_reason == 2)
            ).astype(np.int64)
            self.guard_counts["cooldown_active"] += (
                no_eligible & (chosen_reason == 3)
            ).astype(np.int64)

        choose_up = eligible_up & (~eligible_down | (up_ask <= down_ask))
        choose_down = eligible_down & ~choose_up
        chosen = choose_up | choose_down
        day = observed.astimezone(NY_TZ).date().isoformat()
        spent = self.spend.setdefault(day, [Decimal("0") for _ in range(self.n)])
        for index in np.flatnonzero(chosen):
            cash = self.cash[index]
            budget = min(Decimal("1.00"), Decimal("50.00") - spent[index], cash)
            ask = Decimal(str(row["up_ask"] if choose_up[index] else row["down_ask"]))
            qty, premium, fee = whole_contract_size(self.schedule, ask, budget)
            if qty <= 0:
                self.blocked_cap[index] += 1
                continue
            outlay = premium + fee
            side = 0 if choose_up[index] else 1
            trade = {
                "candidate_id": self.ids[index], "ticker": ticker,
                "side": "up" if side == 0 else "down", "quantity": str(qty),
                "entry_price": str(ask), "entry_cost": str(premium), "status": "open",
                "exit_price": None, "pnl": None, "entry_fee": str(fee), "exit_fee": "0",
                "entry_outlay": str(outlay), "gross_proceeds": None, "net_proceeds": None,
                "entry_event_id": event_id, "exit_event_id": None, "close_reason": None,
                "persisted": False, "dirty": False,
            }
            self.trades.append(trade)
            self.trade_for_candidate[index] = len(self.trades) - 1
            self.cash[index] -= outlay
            self.total_fees[index] += fee
            spent[index] += outlay
            self.entry_count[index] += 1
            self.open[index] = True
            self.open_side[index] = side
            self.open_ticker[index] = ticker
            self.quantity[index] = int(qty)
            self.open_outlay[index] = outlay

        cash_cents = np.fromiter((int(value * 100) for value in self.cash), np.int64, self.n)
        mark = np.where(
            self.open_side == 0, int(up_bid_decimal * 100), int(down_bid_decimal * 100)
        )
        mark_price = mark.astype(float) / 100.0
        liquidation_cents = np.where(
            self.open,
            self.quantity * mark - _fee_cents_array(
                self.schedule, self.quantity, mark_price
            ),
            0,
        )
        equity_cents = cash_cents + liquidation_cents
        self.peak_cents = np.maximum(self.peak_cents, equity_cents)
        self.max_dd_milli = np.maximum(
            self.max_dd_milli, self.peak_cents - equity_cents
        )

    def settle(self, ticker: str, winning_side: str, event_time: datetime) -> None:
        matching = self.open & (self.open_ticker == ticker)
        winner = 0 if winning_side == "up" else 1
        for index in np.flatnonzero(matching):
            payout = Decimal("1") if self.open_side[index] == winner else Decimal("0")
            qty = Decimal(str(int(self.quantity[index])))
            gross = qty * payout
            outlay = self.open_outlay[index]
            pnl = gross - outlay
            self.cash[index] += gross
            self.realized[index] += pnl
            self.closed[index] += 1
            self.wins[index] += int(pnl > 0)
            self.settlement_count[index] += 1
            trade = self.trades[int(self.trade_for_candidate[index])]
            trade.update(
                status="closed", exit_price=str(payout), pnl=str(pnl), exit_fee="0",
                gross_proceeds=str(gross), net_proceeds=str(gross),
                close_reason="settlement",
            )
            if trade["persisted"]:
                trade["dirty"] = True
            self.open[index] = False
            self.open_side[index] = -1
            self.open_ticker[index] = ""
            self.trade_for_candidate[index] = -1

    def checkpoint(self) -> None:
        candidate_updates = []
        for index, cid in enumerate(self.ids):
            candidate_updates.append((
                str(self.cash[index]), str(Decimal(int(self.peak_cents[index])) / 100),
                str(Decimal(int(self.max_dd_milli[index])) / 1000),
                str(self.realized[index]), int(self.closed[index]), int(self.wins[index]),
                str(self.total_fees[index]), int(self.entry_count[index]),
                int(self.tp_count[index]), int(self.sl_count[index]),
                int(self.settlement_count[index]), int(self.skipped_open[index]),
                int(self.blocked_cap[index]), self.last_ticker[index] or None,
                "up" if self.last_side[index] == 0 else "down" if self.last_side[index] == 1 else None,
                self.last_closed_at[index],
                datetime.fromtimestamp(self.cooldown_until[index], UTC).isoformat()
                if self.cooldown_until[index] else None,
                int(self.guard_counts["below_floor"][index]),
                int(self.guard_counts["bid_in_sl_buffer"][index]),
                int(self.guard_counts["spread_too_wide"][index]),
                int(self.guard_counts["confirmation_pending"][index]),
                int(self.guard_counts["reward_risk_too_low"][index]),
                int(self.guard_counts["cooldown_active"][index]),
                self.run_id, cid,
            ))
        new_trades = [trade for trade in self.trades if not trade["persisted"]]
        self.store.connection.executemany(
                """UPDATE lab_candidates SET cash=?,peak_equity=?,max_drawdown=?,
                realized_pnl=?,closed_trades=?,wins=?,total_fees=?,entry_count=?,
                tp_count=?,sl_count=?,settlement_count=?,skipped_open=?,blocked_daily_cap=?,
                last_closed_ticker=?,last_closed_side=?,last_closed_at=?,cooldown_until=?,
                guard_below_floor=?,guard_bid_in_sl_buffer=?,guard_spread_too_wide=?,
                guard_confirmation_pending=?,guard_reward_risk_too_low=?,guard_cooldown_active=?
            WHERE run_id=? AND candidate_id=?""", candidate_updates,
        )
        self.store.connection.executemany(
                """INSERT INTO lab_positions(
                run_id,candidate_id,ticker,side,quantity,entry_price,entry_cost,status,
                exit_price,pnl,entry_fee,exit_fee,entry_outlay,gross_proceeds,net_proceeds,
                entry_event_id,exit_event_id,close_reason
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                ((self.run_id, trade["candidate_id"], trade["ticker"], trade["side"],
                  trade["quantity"], trade["entry_price"], trade["entry_cost"], trade["status"],
                  trade["exit_price"], trade["pnl"], trade["entry_fee"], trade["exit_fee"],
                  trade["entry_outlay"], trade["gross_proceeds"], trade["net_proceeds"],
                  trade["entry_event_id"], trade["exit_event_id"], trade["close_reason"])
             for trade in new_trades),
        )
        self.store.connection.executemany(
                """UPDATE lab_positions SET status=?,exit_price=?,pnl=?,exit_fee=?,
                gross_proceeds=?,net_proceeds=?,exit_event_id=?,close_reason=?
                WHERE run_id=? AND candidate_id=? AND entry_event_id=?""",
                ((trade["status"], trade["exit_price"], trade["pnl"], trade["exit_fee"],
                  trade["gross_proceeds"], trade["net_proceeds"], trade["exit_event_id"],
                  trade["close_reason"], self.run_id, trade["candidate_id"],
              trade["entry_event_id"]) for trade in self.trades if trade["dirty"]),
        )
        self.store.connection.executemany(
                """INSERT OR IGNORE INTO lab_candidate_eligible_cycles(
                run_id,candidate_id,ticker) VALUES (?,?,?)""",
            ((self.run_id, cid, ticker) for cid, ticker in self.eligible),
        )
        for day, values in self.spend.items():
            self.store.connection.executemany(
                    """INSERT INTO lab_daily_spend(run_id,candidate_id,day,amount)
                    VALUES (?,?,?,?) ON CONFLICT(run_id,candidate_id,day)
                    DO UPDATE SET amount=excluded.amount""",
                ((self.run_id, self.ids[index], day, str(value))
                 for index, value in enumerate(values) if value > 0),
            )
        for trade in new_trades:
            trade["persisted"] = True
            trade["dirty"] = False
        for trade in self.trades:
            if trade["dirty"]:
                trade["dirty"] = False
        self.eligible.clear()
        self.trades = [trade for trade in self.trades if trade["status"] == "open"]
        self.trade_for_candidate.fill(-1)
        for trade_index, trade in enumerate(self.trades):
            candidate_index = self.index_by_id[str(trade["candidate_id"])]
            self.trade_for_candidate[candidate_index] = trade_index


async def vectorized_backfill(
    lab: StrategyLab, run_id: int, *, now: datetime | None = None,
) -> None:
    store = lab.store
    run = store.connection.execute("SELECT * FROM lab_runs WHERE id=?", (run_id,)).fetchone()
    if run is None:
        raise KeyError(run_id)
    selection_time = now or datetime.fromisoformat(str(run["started_at"]))
    try:
        assets = json.loads(run["assets"])
        tickers = lab.history_tickers(
            assets, int(run["history_cycles_requested"]),
            int(run["history_cutoff_event_id"] or 0), selection_time,
        )
        results = {ticker: store.market_result(ticker) for ticker in tickers}
        tickers = [ticker for ticker in tickers if results[ticker] is not None]
        all_rows: list[object] = []
        if tickers:
            placeholders = ",".join("?" for _ in tickers)
            all_rows = list(store.connection.execute(
                f"""SELECT * FROM quote_events WHERE id<=? AND ticker IN ({placeholders})
                ORDER BY observed_at,id""",
                (int(run["history_cutoff_event_id"] or 0), *tickers),
            ))
        processed_before = min(int(run["history_events_processed"]), len(all_rows))
        rows = all_rows[processed_before:]
        with store.connection:
            store.connection.execute(
                """UPDATE lab_runs SET history_cycles_loaded=?,history_events_total=?
                WHERE id=?""", (len(tickers), len(all_rows), run_id),
            )
        guard = EntryGuardSettings.model_validate_json(run["guard_snapshot"]) if run["guard_snapshot"] else None
        engines = {
            asset: VectorizedAssetEngine(lab, run_id, asset, lab._schedule(run_id, asset), guard)
            for asset in assets
        }
        persisted_settlements = {
            str(row["ticker"]) for row in store.connection.execute(
                "SELECT ticker FROM lab_settlement_events WHERE run_id=?", (run_id,)
            )
        }
        settlements = sorted(
            (datetime.fromisoformat(str(results[t]["event_time"])), t,
             str(results[t]["winning_side"])) for t in tickers
            if t not in persisted_settlements
        )
        ticker_asset = {str(row["ticker"]): str(row["asset"]) for row in all_rows}
        seen: list[tuple[int, int]] = []
        settlement_records: list[tuple[int, str, str, str]] = []
        settlement_index = 0

        def checkpoint(processed: int, assets_to_flush: set[str] | None = None) -> None:
            with store.connection:
                targets = engines.values() if assets_to_flush is None else (
                    engines[asset] for asset in assets_to_flush
                )
                for engine in targets:
                    engine.checkpoint()
                store.connection.executemany(
                    "INSERT OR IGNORE INTO lab_seen_events(run_id,quote_event_id) VALUES (?,?)",
                    seen,
                )
                store.connection.executemany(
                    """INSERT OR IGNORE INTO lab_settlement_events(
                    run_id,ticker,winning_side,event_time) VALUES (?,?,?,?)""",
                    settlement_records,
                )
                store.connection.execute(
                    """UPDATE lab_runs SET history_events_processed=?,quote_count=?
                    WHERE id=?""", (processed, processed, run_id),
                )
            seen.clear()
            settlement_records.clear()

        for index, row in enumerate(rows, start=processed_before + 1):
            observed = datetime.fromisoformat(str(row["observed_at"]))
            while settlement_index < len(settlements) and settlements[settlement_index][0] <= observed:
                event_time, ticker, winning_side = settlements[settlement_index]
                asset = ticker_asset.get(ticker)
                if asset is not None:
                    engines[asset].settle(ticker, winning_side, event_time)
                settlement_records.append((run_id, ticker, winning_side, event_time.isoformat()))
                settlement_index += 1
                checkpoint(index - 1)
            engines[str(row["asset"])].process(row)
            seen.append((run_id, int(row["id"])))
            await asyncio.sleep(0)
            status = store.connection.execute(
                "SELECT status FROM lab_runs WHERE id=?", (run_id,)
            ).fetchone()[0]
            if status != "backfilling":
                checkpoint(index)
                return

        status = store.connection.execute(
            "SELECT status FROM lab_runs WHERE id=?", (run_id,)
        ).fetchone()[0]
        checkpoint(len(all_rows))
        if status != "backfilling":
            return
        for event_time, ticker, winning_side in settlements[settlement_index:]:
            asset = ticker_asset.get(ticker)
            if asset is not None:
                engines[asset].settle(ticker, winning_side, event_time)
            settlement_records.append((run_id, ticker, winning_side, event_time.isoformat()))
        checkpoint(len(all_rows))
        status = store.connection.execute(
            "SELECT status FROM lab_runs WHERE id=?", (run_id,)
        ).fetchone()[0]
        if status != "backfilling":
            return
        realtime_start = now or datetime.now(UTC)
        with store.connection:
            store.connection.execute(
                """UPDATE lab_runs SET status='running',started_at=?
                WHERE id=? AND status='backfilling'""",
                (realtime_start.isoformat(), run_id),
            )
    except Exception as error:
        with store.connection:
            store.connection.execute(
                """UPDATE lab_runs SET status='failed',error_message=?
                WHERE id=? AND status='backfilling'""", (str(error), run_id),
            )
        raise
