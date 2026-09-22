from __future__ import annotations

from collections import deque
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

from kaishi_bot.agentic_strategy import (
    Action, AgentConfig, AgenticPolicy, Decision,
    MarketObservation,
    MonteCarloConfig,
    MonteCarloGuardConfig,
    MonteCarloGuardedPolicy,
    MonteCarloProbabilityEstimator,
    PathRiskEstimate,
    PositionState,
    XGBoostProbabilityEstimator,
)
from kaishi_bot.dashboard_models import AssetMarket, QuotePoint, RuntimeMode
from kaishi_bot.dashboard_store import DashboardStore
from kaishi_bot.domain import Side
from kaishi_bot.fees import (
    FeeSchedule, fractional_contract_size, fractional_exit_value, taker_fee,
)
from kaishi_bot.model_registry import ModelRegistry
from kaishi_bot.paper_shadow import ShadowDecisionLog
from kaishi_bot.research_capture import ResearchCaptureSupervisor
from kaishi_bot.research_store import ResearchStore


NY = ZoneInfo("America/New_York")


class PaperModelShadow:
    """XGBoost + Monte Carlo execution on Paper; never on the Live gateway."""

    MONTE_CARLO_PATHS = 4096
    MAXIMUM_STOP_BEFORE_TAKE_PROFIT = 0.35
    MINIMUM_TAKE_PROFIT_BEFORE_STOP = 0.20

    def __init__(
        self, registry: ModelRegistry, capture: ResearchCaptureSupervisor,
        dashboard_store: DashboardStore, research_store: ResearchStore,
    ) -> None:
        self.registry = registry
        self.capture = capture
        self.dashboard_store = dashboard_store
        self.research_store = research_store
        self.model_id: str | None = None
        self.asset: str | None = None
        self.policy: MonteCarloGuardedPolicy | None = None
        self.started_at: datetime | None = None
        self.last_decision_at: datetime | None = None
        self.last_error: str | None = None
        self.decision_count = 0
        self.action_count = 0
        self.paper_trade_count = 0
        self.latest_risk: PathRiskEstimate | None = None
        self.latest_risk_at: datetime | None = None
        self.recent: deque[dict[str, object]] = deque(maxlen=30)
        self._last_ticker_time: dict[str, datetime] = {}

    @property
    def active(self) -> bool:
        return self.policy is not None and self.model_id is not None

    def activate(self, model_id: str) -> dict[str, object]:
        if self.active:
            raise ValueError("một model Paper khác đang chạy")
        if self.dashboard_store.load_settings().mode is not RuntimeMode.PAPER:
            raise ValueError("Paper Shadow chỉ được bật trong mode Paper")
        item = self.registry.get(model_id)
        if not item["ready_for_paper_shadow"]:
            raise ValueError("model chưa đủ artifact để chạy Paper Shadow")
        settings = self.dashboard_store.load_settings()
        owner = f"model:{model_id}"
        conflicting = [
            position for position in self.dashboard_store.open_positions()
            if str(position.get("strategy_owner", "")).startswith("model:")
            and str(position.get("strategy_owner")) != owner
        ]
        if conflicting:
            raise ValueError(
                "còn vị thế Paper của model khác; hãy đóng vị thế đó trước"
            )
        estimator = XGBoostProbabilityEstimator.load(self.registry.directory(model_id))
        base_policy = AgenticPolicy(estimator=estimator, config=AgentConfig(
            entry_min=settings.entry_min,
            entry_max=settings.entry_price,
            first_take_profit=settings.take_profit,
            final_take_profit=settings.take_profit,
            emergency_bid=settings.stop_loss,
            max_add_count=0,
        ))
        monte_carlo = MonteCarloProbabilityEstimator(MonteCarloConfig(
            paths=self.MONTE_CARLO_PATHS,
        ))
        self.policy = MonteCarloGuardedPolicy(
            base_policy,
            monte_carlo,
            MonteCarloGuardConfig(
                stop_loss=settings.stop_loss,
                take_profit=settings.take_profit,
                maximum_stop_before_take_profit=(
                    self.MAXIMUM_STOP_BEFORE_TAKE_PROFIT
                ),
                minimum_take_profit_before_stop=(
                    self.MINIMUM_TAKE_PROFIT_BEFORE_STOP
                ),
            ),
        )
        # The model is now the only Paper entry engine. Existing positions can
        # still be manually closed, but Control Deck cannot open new ones until
        # the user explicitly enables it again after stopping the model.
        if settings.bot_enabled:
            self.dashboard_store.save_settings(
                settings.model_copy(update={"bot_enabled": False})
            )
        self.model_id = model_id
        self.asset = str(item.get("asset") or "") or None
        self.started_at = datetime.now(UTC)
        self.last_decision_at = None
        self.last_error = None
        self.decision_count = 0
        self.action_count = 0
        self.paper_trade_count = 0
        self.latest_risk = None
        self.latest_risk_at = None
        self.recent.clear()
        self._last_ticker_time.clear()
        return self.status()

    def deactivate(self) -> dict[str, object]:
        self.policy = None
        self.model_id = None
        self.asset = None
        self.started_at = None
        self._last_ticker_time.clear()
        return self.status()

    def on_quote(
        self, market: AssetMarket, quote: QuotePoint,
        fee_schedule: FeeSchedule,
    ) -> None:
        if not self.active or market.asset != self.asset:
            return
        if self.dashboard_store.load_settings().mode is not RuntimeMode.PAPER:
            self.deactivate()
            return
        prior = self._last_ticker_time.get(market.ticker)
        if prior is not None and quote.observed_at <= prior:
            return
        self._last_ticker_time[market.ticker] = quote.observed_at
        try:
            observation = self._observation(market, quote, fee_schedule)
            assert self.policy is not None and self.model_id is not None
            decision = self.policy.decide(observation)
            risk = self.policy.last_risk
            if risk is not None:
                self.latest_risk = risk
                self.latest_risk_at = quote.observed_at
            executable = (
                observation.ask(decision.side)
                if decision.side is not None and decision.action.value in {"buy", "add"}
                else observation.bid(decision.side) if decision.side is not None else None
            )
            broker_result = self._execute_decision(
                decision, market, quote, fee_schedule
            )
            broker_result.update({
                "mode": "paper_execution",
                "predicted_executable_price": str(executable) if executable is not None else None,
                "monte_carlo_risk": self._risk_payload(risk),
            })
            log = ShadowDecisionLog(
                ticker=market.ticker, observed_at=quote.observed_at,
                dataset_version=f"artifact:{self.model_id}",
                policy_version="agentic-policy-v1+mc-risk-v1", model_version=self.model_id,
                selected_action=decision.action.value,
                selected_side=decision.side.value if decision.side else None,
                masked_actions=(
                    ("buy", "add")
                    if decision.reason.startswith("monte_carlo_") else ()
                ),
                quality_state=(
                    "stale_or_gap" if observation.data_stale or observation.has_gap
                    else "complete"
                ),
                broker_result=broker_result, kill_switch_active=False,
            )
            self.research_store.save_shadow_decision(log)
            self.last_decision_at = quote.observed_at
            self.decision_count += 1
            if decision.action.value not in {"wait", "hold"}:
                self.action_count += 1
            if broker_result.get("executed"):
                self.paper_trade_count += 1
            self.recent.appendleft({
                "ticker": market.ticker,
                "observed_at": quote.observed_at.isoformat(),
                "action": decision.action.value,
                "side": decision.side.value if decision.side else None,
                "probability": str(decision.probability),
                "edge": str(decision.edge),
                "reason": decision.reason,
                "executable_price": broker_result["predicted_executable_price"],
                "quality": log.quality_state,
                "executed": bool(broker_result.get("executed")),
                "paper_result": broker_result.get("result"),
                "monte_carlo_risk": self._risk_payload(risk),
            })
            self.last_error = None
        except Exception as error:
            self.last_error = f"{type(error).__name__}: {error}"

    @staticmethod
    def _risk_payload(risk: PathRiskEstimate | None) -> dict[str, object] | None:
        if risk is None:
            return None
        return {
            "side": risk.side.value,
            "settlement_win_probability": risk.settlement_win_probability,
            "stop_before_take_profit_probability": (
                risk.stop_before_take_profit_probability
            ),
            "take_profit_before_stop_probability": (
                risk.take_profit_before_stop_probability
            ),
            "neither_barrier_probability": risk.neither_barrier_probability,
            "path_count": risk.path_count,
        }

    def _execute_decision(
        self, decision: Decision, market: AssetMarket, quote: QuotePoint,
        schedule: FeeSchedule,
    ) -> dict[str, object]:
        if decision.action in {Action.WAIT, Action.HOLD}:
            return {"executed": False, "result": decision.action.value,
                    "paper_account_mutated": False}
        assert self.model_id is not None
        owner = f"model:{self.model_id}"
        settings = self.dashboard_store.load_settings()
        if settings.mode is not RuntimeMode.PAPER:
            return {"executed": False, "result": "not_paper_mode",
                    "paper_account_mutated": False}
        positions = [
            item for item in self.dashboard_store.open_positions()
            if str(item["ticker"]) == market.ticker
            and str(item.get("strategy_owner")) == owner
        ]
        if decision.action is Action.BUY:
            if decision.side is None or positions:
                return {"executed": False, "result": "position_already_open",
                        "paper_account_mutated": False}
            # Never overlap a model position with a Control Deck position in
            # the same market, even on the opposite side.
            if any(
                str(item["ticker"]) == market.ticker
                for item in self.dashboard_store.open_positions()
            ):
                return {"executed": False, "result": "market_position_conflict",
                        "paper_account_mutated": False}
            ask = quote.up_ask if decision.side is Side.UP else quote.down_ask
            day = quote.observed_at.astimezone(NY).date().isoformat()
            remaining = settings.paper_daily_cap - self.dashboard_store.daily_spend(day)
            budget = min(
                settings.entry_amount * decision.fraction,
                remaining, self.dashboard_store.cash(), Decimal("200"),
            )
            quantity, _premium, entry_fee = fractional_contract_size(
                schedule, ask, budget
            )
            if quantity <= 0:
                return {"executed": False, "result": "insufficient_paper_budget",
                        "paper_account_mutated": False}
            position_id = self.dashboard_store.open_position(
                asset=market.asset, ticker=market.ticker,
                side=decision.side.value, quantity=quantity,
                entry_price=ask, opened_at=quote.observed_at, day=day,
                entry_fee=entry_fee, liquidity_status="realtime_quote",
                strategy_owner=owner,
            )
            return {
                "executed": position_id is not None,
                "result": "paper_entry" if position_id is not None else "entry_rejected",
                "position_id": position_id,
                "quantity": str(quantity), "price": str(ask),
                "fee": str(entry_fee), "paper_account_mutated": position_id is not None,
            }
        if decision.action is Action.EXIT_ALL:
            closed_ids: list[int] = []
            total_fee = Decimal("0")
            for position in positions:
                side = Side(str(position["side"]))
                bid = quote.up_bid if side is Side.UP else quote.down_bid
                quantity = Decimal(str(position["quantity"]))
                exit_fee, _proceeds = fractional_exit_value(schedule, quantity, bid)
                reason = (
                    "take_profit" if decision.reason == "final_take_profit"
                    else "stop_loss" if decision.reason == "emergency_stop"
                    else "model_exit"
                )
                if self.dashboard_store.close_position(
                    int(position["id"]), bid, quote.observed_at, reason, exit_fee
                ):
                    closed_ids.append(int(position["id"]))
                    total_fee += exit_fee
            return {
                "executed": bool(closed_ids), "result": "paper_exit",
                "position_ids": closed_ids, "fee": str(total_fee),
                "paper_account_mutated": bool(closed_ids),
            }
        return {"executed": False, "result": "action_disabled_for_v1",
                "paper_account_mutated": False}

    def _observation(
        self, market: AssetMarket, quote: QuotePoint, schedule: FeeSchedule,
    ) -> MarketObservation:
        rows = self.capture.recent_rti(market.asset)
        if not rows:
            raise ValueError("chưa có RTI cho asset")
        latest = max(rows, key=lambda item: item.collector_received_at)
        if market.target is None:
            raise ValueError("market chưa có target")
        prices = [row.price for row in rows[-61:]]
        changes = [right - left for left, right in zip(prices, prices[1:])]
        if changes:
            mean = sum(changes, Decimal("0")) / Decimal(len(changes))
            variance = sum(
                ((value - mean) ** 2 for value in changes), Decimal("0")
            ) / Decimal(len(changes))
            sigma = variance.sqrt()
        else:
            sigma = Decimal("0")
        window_start = market.close_time - timedelta(seconds=60)
        locked_by_source = {
            row.source_timestamp_ms: row
            for row in rows
            if window_start < row.source_time_utc <= min(quote.observed_at, market.close_time)
            and row.collector_received_at <= quote.observed_at
        }
        locked = tuple(locked_by_source.values())
        owner = f"model:{self.model_id}"
        positions = [
            item for item in self.dashboard_store.open_positions()
            if str(item["ticker"]) == market.ticker
            and str(item.get("strategy_owner")) == owner
        ]
        position = None
        if positions:
            item = positions[0]
            position = PositionState(
                side=Side(str(item["side"])),
                quantity=Decimal(str(item["quantity"])),
                average_entry_price=Decimal(str(item["entry_price"])),
                seconds_held=max(
                    0, int((quote.observed_at - datetime.fromisoformat(
                        str(item["opened_at"])
                    )).total_seconds()),
                ),
            )
        age = (quote.observed_at - latest.collector_received_at).total_seconds()
        return MarketObservation(
            ticker=market.ticker, observed_at=quote.observed_at,
            seconds_remaining=max(0, int((market.close_time - quote.observed_at).total_seconds())),
            up_bid=quote.up_bid, up_ask=quote.up_ask,
            down_bid=quote.down_bid, down_ask=quote.down_ask,
            target_price=Decimal(str(market.target)), brti_price=latest.price,
            brti_sigma_per_sqrt_second=sigma,
            locked_sample_count=len(locked),
            locked_sample_sum=sum((row.price for row in locked), Decimal("0")),
            entry_fee_up=taker_fee(schedule, Decimal("1"), quote.up_ask),
            entry_fee_down=taker_fee(schedule, Decimal("1"), quote.down_ask),
            position=position,
            data_stale=age > self.capture.config.stale_after_seconds,
            has_gap=latest.gap_detected,
        )

    def status(self) -> dict[str, object]:
        policy = self.policy
        return {
            "active": self.active,
            "mode": "paper_execution",
            "model_id": self.model_id,
            "asset": self.asset,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_decision_at": (
                self.last_decision_at.isoformat() if self.last_decision_at else None
            ),
            "decision_count": self.decision_count,
            "action_count": self.action_count,
            "paper_trade_count": self.paper_trade_count,
            "last_error": self.last_error,
            "paper_account_mutated": self.paper_trade_count > 0,
            "paper_execution_enabled": self.active,
            "live_orders_possible": False,
            "risk_engine": {
                "enabled": self.active,
                "model": "monte_carlo_first_passage",
                "paths": self.MONTE_CARLO_PATHS,
                "maximum_stop_before_take_profit": (
                    self.MAXIMUM_STOP_BEFORE_TAKE_PROFIT
                ),
                "minimum_take_profit_before_stop": (
                    self.MINIMUM_TAKE_PROFIT_BEFORE_STOP
                ),
                "evaluation_count": (
                    policy.risk_evaluation_count if policy is not None else 0
                ),
                "last_risk": self._risk_payload(
                    self.latest_risk
                ),
                "last_risk_at": (
                    self.latest_risk_at.isoformat() if self.latest_risk_at else None
                ),
            },
            "recent_decisions": list(self.recent),
        }
