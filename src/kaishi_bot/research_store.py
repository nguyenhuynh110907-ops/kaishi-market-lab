from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Iterable

from kaishi_bot.research_models import FeeMetadataVersion, ResearchMarket


def _iso(value: datetime | None) -> str | None:
    return value.astimezone(UTC).isoformat() if value is not None else None


class ResearchStore:
    """Small SQLite control plane for immutable research files.

    High-rate RTI and orderbook rows never enter this database. Only metadata,
    file manifests, stream checkpoints, and quality events are stored here.
    """

    SCHEMA_VERSION = 3

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self._lock = threading.RLock()
        self._migrate()

    def _migrate(self) -> None:
        with self._lock, self.connection:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    applied_at TEXT NOT NULL,
                    checksum TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS research_markets (
                    ticker TEXT PRIMARY KEY,
                    asset TEXT NOT NULL,
                    series_ticker TEXT NOT NULL,
                    title TEXT NOT NULL,
                    open_time TEXT NOT NULL,
                    close_time TEXT NOT NULL,
                    target_price TEXT,
                    target_source_field TEXT,
                    rules_primary TEXT,
                    rules_secondary TEXT,
                    target_window_start TEXT NOT NULL,
                    target_window_end TEXT NOT NULL,
                    settlement_window_start TEXT NOT NULL,
                    settlement_window_end TEXT NOT NULL,
                    official_result TEXT,
                    expiration_value TEXT,
                    settlement_value TEXT,
                    settlement_ts TEXT,
                    result_observed_at TEXT,
                    discovered_at TEXT NOT NULL,
                    refreshed_at TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    quality_status TEXT NOT NULL,
                    schema_version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS research_market_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    raw_payload_json TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    UNIQUE(ticker, payload_sha256)
                );
                CREATE TABLE IF NOT EXISTS research_settlements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    official_result TEXT,
                    expiration_value TEXT,
                    settlement_value TEXT,
                    settlement_ts TEXT,
                    observed_at TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    raw_payload_json TEXT NOT NULL,
                    UNIQUE(ticker, payload_sha256)
                );
                CREATE TABLE IF NOT EXISTS research_index_registry (
                    asset TEXT NOT NULL,
                    index_id TEXT NOT NULL,
                    valid_from TEXT NOT NULL,
                    valid_to TEXT,
                    discovered_at TEXT NOT NULL,
                    source_payload_sha256 TEXT,
                    PRIMARY KEY(asset, index_id, valid_from)
                );
                CREATE TABLE IF NOT EXISTS research_collector_sessions (
                    session_id TEXT PRIMARY KEY,
                    stream TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    status TEXT NOT NULL,
                    last_seq INTEGER,
                    reconnect_count INTEGER NOT NULL DEFAULT 0,
                    error_code TEXT,
                    config_sha256 TEXT NOT NULL,
                    build_commit TEXT
                );
                CREATE TABLE IF NOT EXISTS research_stream_checkpoints (
                    stream TEXT NOT NULL,
                    partition_key TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    last_source_timestamp_ms INTEGER,
                    last_seq INTEGER,
                    last_event_hash TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(stream, partition_key)
                );
                CREATE TABLE IF NOT EXISTS research_files (
                    file_id TEXT PRIMARY KEY,
                    dataset TEXT NOT NULL,
                    path TEXT NOT NULL UNIQUE,
                    sha256 TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    min_event_time TEXT NOT NULL,
                    max_event_time TEXT NOT NULL,
                    min_seq INTEGER,
                    max_seq INTEGER,
                    schema_version INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS research_quality_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    index_id TEXT,
                    stream TEXT NOT NULL,
                    event_time TEXT,
                    observed_at TEXT NOT NULL,
                    reason_code TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    missing_sample_count INTEGER NOT NULL DEFAULT 0,
                    first_missing_timestamp TEXT,
                    last_missing_timestamp TEXT,
                    expected_seq INTEGER,
                    observed_seq INTEGER,
                    stale_seconds TEXT,
                    queue_depth INTEGER,
                    session_id TEXT,
                    details_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS research_market_quality (
                    ticker TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    reason_code TEXT,
                    updated_at TEXT NOT NULL,
                    missing_target INTEGER NOT NULL DEFAULT 0,
                    missing_settlement INTEGER NOT NULL DEFAULT 0,
                    rti_gaps INTEGER NOT NULL DEFAULT 0,
                    orderbook_gaps INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS research_metadata_backfill_attempts (
                    ticker TEXT PRIMARY KEY,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    last_attempt_at TEXT NOT NULL,
                    error_code TEXT
                );
                CREATE INDEX IF NOT EXISTS research_markets_asset_close
                    ON research_markets(asset, close_time);
                CREATE INDEX IF NOT EXISTS research_quality_stream_time
                    ON research_quality_events(stream, observed_at);
                """
            )
            self.connection.execute(
                """INSERT OR IGNORE INTO schema_migrations(
                    version,name,applied_at,checksum
                ) VALUES (1,'research-control-plane-v1',?,?)""",
                (
                    datetime.now(UTC).isoformat(),
                    hashlib.sha256(b"research-control-plane-v1").hexdigest(),
                ),
            )
            self._migrate_v2()
            self._migrate_v3()

    def _migrate_v2(self) -> None:
        """Add research Phase A control-plane schema without rewriting v1."""
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS research_fee_versions (
                fee_version TEXT PRIMARY KEY,
                series_ticker TEXT NOT NULL,
                fee_type TEXT NOT NULL,
                fee_multiplier TEXT NOT NULL,
                taker_rate TEXT NOT NULL,
                maker_rate TEXT,
                effective_from TEXT NOT NULL,
                effective_to TEXT,
                observed_at TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                source_change_id TEXT,
                source_payload_hash TEXT NOT NULL,
                raw_payload_json TEXT NOT NULL,
                UNIQUE(series_ticker,effective_from,source_payload_hash)
            );
            CREATE INDEX IF NOT EXISTS research_fee_versions_effective
                ON research_fee_versions(series_ticker,effective_from,effective_to);
            CREATE TABLE IF NOT EXISTS research_quality_dimensions (
                ticker TEXT NOT NULL,
                quality_policy_version TEXT NOT NULL,
                metadata_complete INTEGER NOT NULL DEFAULT 0,
                target_complete INTEGER NOT NULL DEFAULT 0,
                official_result_complete INTEGER NOT NULL DEFAULT 0,
                rti_target_window_complete INTEGER NOT NULL DEFAULT 0,
                rti_settlement_window_complete INTEGER NOT NULL DEFAULT 0,
                contract_quotes_complete INTEGER NOT NULL DEFAULT 0,
                orderbook_complete INTEGER NOT NULL DEFAULT 0,
                fee_metadata_complete INTEGER NOT NULL DEFAULT 0,
                feature_complete INTEGER NOT NULL DEFAULT 0,
                replay_eligible INTEGER NOT NULL DEFAULT 0,
                training_eligible INTEGER NOT NULL DEFAULT 0,
                incomplete_reasons_json TEXT NOT NULL DEFAULT '[]',
                evaluated_at TEXT NOT NULL,
                evidence_manifest_ids_json TEXT NOT NULL DEFAULT '[]',
                PRIMARY KEY(ticker,quality_policy_version)
            );
            CREATE TABLE IF NOT EXISTS research_dataset_versions (
                dataset_version TEXT PRIMARY KEY,
                feature_schema_version TEXT NOT NULL,
                builder_config_json TEXT NOT NULL,
                builder_config_hash TEXT NOT NULL,
                quality_policy_version TEXT NOT NULL,
                source_manifest_root_hash TEXT NOT NULL,
                build_commit TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                status TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS research_dataset_members (
                dataset_version TEXT NOT NULL,
                ticker TEXT NOT NULL,
                close_time TEXT NOT NULL,
                split TEXT NOT NULL,
                fold_id TEXT,
                eligible INTEGER NOT NULL,
                reasons_json TEXT NOT NULL,
                source_manifest_ids_json TEXT NOT NULL,
                PRIMARY KEY(dataset_version,ticker),
                FOREIGN KEY(dataset_version) REFERENCES research_dataset_versions(dataset_version)
            );
            CREATE TABLE IF NOT EXISTS research_split_manifests (
                split_manifest_id TEXT PRIMARY KEY,
                dataset_version TEXT NOT NULL,
                train_tickers_json TEXT NOT NULL,
                validation_tickers_json TEXT NOT NULL,
                test_tickers_json TEXT NOT NULL,
                embargo_tickers_json TEXT NOT NULL,
                manifest_hash TEXT NOT NULL,
                frozen_at TEXT NOT NULL,
                UNIQUE(dataset_version,manifest_hash)
            );
            CREATE TABLE IF NOT EXISTS research_experiments (
                experiment_id TEXT PRIMARY KEY,
                dataset_version TEXT NOT NULL,
                split_manifest_id TEXT NOT NULL,
                config_json TEXT NOT NULL,
                code_commit TEXT,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                finalist_hash TEXT,
                final_test_unlocked INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS research_trials (
                trial_id TEXT PRIMARY KEY,
                parent_experiment_id TEXT NOT NULL,
                config_json TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                model_type TEXT NOT NULL,
                seed INTEGER NOT NULL,
                dataset_version TEXT NOT NULL,
                fold_id TEXT NOT NULL,
                code_commit TEXT,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                metrics_json TEXT,
                artifact_path TEXT
            );
            CREATE TABLE IF NOT EXISTS research_episode_metrics (
                episode_id TEXT PRIMARY KEY,
                trial_id TEXT NOT NULL,
                ticker TEXT NOT NULL,
                dataset_version TEXT NOT NULL,
                policy_version TEXT NOT NULL,
                seed INTEGER NOT NULL,
                starting_cash TEXT NOT NULL,
                ending_cash TEXT NOT NULL,
                net_pnl TEXT NOT NULL,
                fees TEXT NOT NULL,
                slippage TEXT NOT NULL,
                maximum_drawdown TEXT NOT NULL,
                turnover TEXT NOT NULL,
                number_of_actions INTEGER NOT NULL,
                number_of_orders INTEGER NOT NULL,
                fill_ratio TEXT NOT NULL,
                settlement_result TEXT,
                trajectory_hash TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS research_paper_shadow_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                dataset_version TEXT NOT NULL,
                policy_version TEXT NOT NULL,
                model_version TEXT NOT NULL,
                selected_action TEXT NOT NULL,
                selected_side TEXT,
                masked_actions_json TEXT NOT NULL,
                reason TEXT NOT NULL,
                quality_state TEXT NOT NULL,
                broker_result_json TEXT,
                kill_switch_active INTEGER NOT NULL
            );
            """
        )
        columns = {
            str(row[1]) for row in self.connection.execute("PRAGMA table_info(research_files)")
        }
        additions = {
            "byte_size": "INTEGER",
            "config_sha256": "TEXT",
            "partition_json": "TEXT",
            "parent_manifest_ids_json": "TEXT",
        }
        for name, sql_type in additions.items():
            if name not in columns:
                self.connection.execute(
                    f"ALTER TABLE research_files ADD COLUMN {name} {sql_type}"
                )
        self.connection.execute(
            """INSERT OR IGNORE INTO schema_migrations(
                version,name,applied_at,checksum
            ) VALUES (2,'research-phase-a-control-v2',?,?)""",
            (
                datetime.now(UTC).isoformat(),
                hashlib.sha256(b"research-phase-a-control-v2").hexdigest(),
            ),
        )

    def _migrate_v3(self) -> None:
        """Persist every train/validation run, not only tournament finalists."""
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS research_experiment_candidates (
                experiment_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                model_type TEXT NOT NULL,
                config_json TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                selected INTEGER NOT NULL,
                PRIMARY KEY(experiment_id,candidate_id)
            );
            CREATE TABLE IF NOT EXISTS research_trial_runs (
                run_id TEXT PRIMARY KEY,
                parent_experiment_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                dataset_version TEXT NOT NULL,
                fold_id TEXT NOT NULL,
                seed INTEGER NOT NULL,
                stress_profile TEXT NOT NULL,
                split TEXT NOT NULL,
                resource_fraction TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                trajectory_hash TEXT NOT NULL,
                artifact_path TEXT,
                UNIQUE(
                    parent_experiment_id,candidate_id,fold_id,seed,
                    stress_profile,split,resource_fraction
                )
            );
            CREATE INDEX IF NOT EXISTS research_trial_runs_experiment
                ON research_trial_runs(parent_experiment_id,candidate_id,split);
            """
        )
        self.connection.execute(
            """INSERT OR IGNORE INTO schema_migrations(
                version,name,applied_at,checksum
            ) VALUES (3,'research-experiment-runs-v3',?,?)""",
            (
                datetime.now(UTC).isoformat(),
                hashlib.sha256(b"research-experiment-runs-v3").hexdigest(),
            ),
        )

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    def save_market(self, market: ResearchMarket) -> bool:
        """Persist the current projection and append a changed payload version.

        Returns True only when a new raw payload version was inserted.
        """
        raw_json = json.dumps(
            market.raw_payload, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, default=str,
        )
        with self._lock, self.connection:
            existing_quality = self.connection.execute(
                "SELECT status,reason_code FROM research_market_quality WHERE ticker=?",
                (market.ticker,),
            ).fetchone()
            if market.target_price is None:
                quality, quality_reason = "incomplete", "missing_target"
            elif (
                existing_quality is not None
                and existing_quality["status"] == "incomplete"
                and existing_quality["reason_code"] != "missing_target"
            ):
                quality = "incomplete"
                quality_reason = existing_quality["reason_code"]
            else:
                quality, quality_reason = "complete", None
            cursor = self.connection.execute(
                """INSERT OR IGNORE INTO research_market_versions(
                    ticker,observed_at,payload_sha256,raw_payload_json,schema_version
                ) VALUES (?,?,?,?,?)""",
                (
                    market.ticker, _iso(market.refreshed_at), market.payload_sha256,
                    raw_json, self.SCHEMA_VERSION,
                ),
            )
            self.connection.execute(
                """INSERT INTO research_markets(
                    ticker,asset,series_ticker,title,open_time,close_time,target_price,
                    target_source_field,rules_primary,rules_secondary,target_window_start,
                    target_window_end,settlement_window_start,settlement_window_end,
                    official_result,expiration_value,settlement_value,settlement_ts,
                    result_observed_at,discovered_at,refreshed_at,payload_sha256,
                    quality_status,schema_version
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(ticker) DO UPDATE SET
                    asset=excluded.asset, series_ticker=excluded.series_ticker,
                    title=excluded.title, open_time=excluded.open_time,
                    close_time=excluded.close_time, target_price=excluded.target_price,
                    target_source_field=excluded.target_source_field,
                    rules_primary=excluded.rules_primary,
                    rules_secondary=excluded.rules_secondary,
                    official_result=excluded.official_result,
                    expiration_value=excluded.expiration_value,
                    settlement_value=excluded.settlement_value,
                    settlement_ts=excluded.settlement_ts,
                    result_observed_at=COALESCE(excluded.result_observed_at,result_observed_at),
                    refreshed_at=excluded.refreshed_at,
                    payload_sha256=excluded.payload_sha256,
                    quality_status=excluded.quality_status,
                    schema_version=excluded.schema_version""",
                (
                    market.ticker, market.asset, market.series_ticker, market.title,
                    _iso(market.open_time), _iso(market.close_time),
                    str(market.target_price) if market.target_price is not None else None,
                    market.target_source_field, market.rules_primary, market.rules_secondary,
                    _iso(market.target_window_start), _iso(market.target_window_end),
                    _iso(market.settlement_window_start), _iso(market.settlement_window_end),
                    market.official_result,
                    str(market.expiration_value) if market.expiration_value is not None else None,
                    str(market.settlement_value) if market.settlement_value is not None else None,
                    _iso(market.settlement_ts), _iso(market.result_observed_at),
                    _iso(market.discovered_at), _iso(market.refreshed_at),
                    market.payload_sha256, quality, self.SCHEMA_VERSION,
                ),
            )
            self.connection.execute(
                """INSERT INTO research_market_quality(
                    ticker,status,reason_code,updated_at,missing_target
                ) VALUES (?,?,?,?,?)
                ON CONFLICT(ticker) DO UPDATE SET
                    status=excluded.status,reason_code=excluded.reason_code,
                    updated_at=excluded.updated_at,missing_target=excluded.missing_target""",
                (
                    market.ticker, quality,
                    quality_reason,
                    _iso(market.refreshed_at), int(market.target_price is None),
                ),
            )
            if market.official_result or market.expiration_value is not None:
                self.connection.execute(
                    """INSERT OR IGNORE INTO research_settlements(
                        ticker,official_result,expiration_value,settlement_value,
                        settlement_ts,observed_at,payload_sha256,raw_payload_json
                    ) VALUES (?,?,?,?,?,?,?,?)""",
                    (
                        market.ticker, market.official_result,
                        str(market.expiration_value) if market.expiration_value is not None else None,
                        str(market.settlement_value) if market.settlement_value is not None else None,
                        _iso(market.settlement_ts), _iso(market.refreshed_at),
                        market.payload_sha256, raw_json,
                    ),
                )
            return cursor.rowcount == 1

    def register_index(self, asset: str, index_id: str, observed_at: datetime) -> None:
        with self._lock, self.connection:
            current = self.connection.execute(
                """SELECT index_id FROM research_index_registry
                   WHERE asset=? AND valid_to IS NULL ORDER BY valid_from DESC LIMIT 1""",
                (asset,),
            ).fetchone()
            if current is not None and current["index_id"] == index_id:
                return
            self.connection.execute(
                "UPDATE research_index_registry SET valid_to=? WHERE asset=? AND valid_to IS NULL",
                (_iso(observed_at), asset),
            )
            self.connection.execute(
                """INSERT INTO research_index_registry(
                    asset,index_id,valid_from,discovered_at
                ) VALUES (?,?,?,?)""",
                (asset, index_id, _iso(observed_at), _iso(observed_at)),
            )

    def save_fee_version(self, version: FeeMetadataVersion) -> bool:
        with self._lock, self.connection:
            cursor = self.connection.execute(
                """INSERT OR IGNORE INTO research_fee_versions(
                    fee_version,series_ticker,fee_type,fee_multiplier,taker_rate,
                    maker_rate,effective_from,effective_to,observed_at,source_kind,
                    source_change_id,source_payload_hash,raw_payload_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    version.fee_version, version.series_ticker, version.fee_type,
                    str(version.fee_multiplier), str(version.taker_rate),
                    str(version.maker_rate) if version.maker_rate is not None else None,
                    _iso(version.effective_from), _iso(version.effective_to),
                    _iso(version.observed_at), version.source_kind,
                    version.source_change_id, version.source_payload_hash,
                    version.raw_payload_json,
                ),
            )
            return cursor.rowcount == 1

    def fee_version_at(
        self, series_ticker: str, effective_at: datetime
    ) -> FeeMetadataVersion | None:
        with self._lock:
            row = self.connection.execute(
                """SELECT * FROM research_fee_versions
                   WHERE series_ticker=? AND effective_from<=?
                     AND (effective_to IS NULL OR effective_to>?)
                   ORDER BY effective_from DESC LIMIT 1""",
                (series_ticker, _iso(effective_at), _iso(effective_at)),
            ).fetchone()
        if row is None:
            return None
        return FeeMetadataVersion(
            fee_version=row["fee_version"], series_ticker=row["series_ticker"],
            fee_type=row["fee_type"], fee_multiplier=Decimal(row["fee_multiplier"]),
            taker_rate=Decimal(row["taker_rate"]),
            maker_rate=Decimal(row["maker_rate"]) if row["maker_rate"] else None,
            effective_from=datetime.fromisoformat(row["effective_from"]),
            effective_to=(datetime.fromisoformat(row["effective_to"])
                          if row["effective_to"] else None),
            observed_at=datetime.fromisoformat(row["observed_at"]),
            source_kind=row["source_kind"], source_change_id=row["source_change_id"],
            source_payload_hash=row["source_payload_hash"],
            raw_payload_json=row["raw_payload_json"],
        )

    def set_quality_dimensions(
        self, ticker: str, *, policy_version: str = "research-v1",
        incomplete_reasons: Iterable[str] = (), evidence_manifest_ids: Iterable[str] = (),
        **statuses: bool,
    ) -> None:
        fields = (
            "metadata_complete", "target_complete", "official_result_complete",
            "rti_target_window_complete", "rti_settlement_window_complete",
            "contract_quotes_complete", "orderbook_complete",
            "fee_metadata_complete", "feature_complete", "replay_eligible",
            "training_eligible",
        )
        unknown = set(statuses) - set(fields)
        if unknown:
            raise ValueError(f"unknown quality dimensions: {', '.join(sorted(unknown))}")
        values = [int(bool(statuses.get(field, False))) for field in fields]
        with self._lock, self.connection:
            self.connection.execute(
                f"""INSERT INTO research_quality_dimensions(
                    ticker,quality_policy_version,{','.join(fields)},
                    incomplete_reasons_json,evaluated_at,evidence_manifest_ids_json
                ) VALUES ({','.join('?' for _ in range(2 + len(fields) + 3))})
                ON CONFLICT(ticker,quality_policy_version) DO UPDATE SET
                    {','.join(f'{field}=excluded.{field}' for field in fields)},
                    incomplete_reasons_json=excluded.incomplete_reasons_json,
                    evaluated_at=excluded.evaluated_at,
                    evidence_manifest_ids_json=excluded.evidence_manifest_ids_json""",
                (
                    ticker, policy_version, *values,
                    json.dumps(sorted(set(incomplete_reasons))),
                    datetime.now(UTC).isoformat(),
                    json.dumps(sorted(set(evidence_manifest_ids))),
                ),
            )

    def mark_quality_dimension(
        self, ticker: str, dimension: str, complete: bool, *,
        reason: str | None = None, evidence_manifest_id: str | None = None,
        policy_version: str = "research-v1",
    ) -> None:
        fields = (
            "metadata_complete", "target_complete", "official_result_complete",
            "rti_target_window_complete", "rti_settlement_window_complete",
            "contract_quotes_complete", "orderbook_complete",
            "fee_metadata_complete", "feature_complete", "replay_eligible",
            "training_eligible",
        )
        if dimension not in fields:
            raise ValueError(f"unknown quality dimension: {dimension}")
        with self._lock:
            row = self.connection.execute(
                """SELECT * FROM research_quality_dimensions
                   WHERE ticker=? AND quality_policy_version=?""",
                (ticker, policy_version),
            ).fetchone()
        values = {
            field: bool(row[field]) if row is not None else False for field in fields
        }
        values[dimension] = complete
        reasons = set(json.loads(row["incomplete_reasons_json"])) if row else set()
        evidence = set(json.loads(row["evidence_manifest_ids_json"])) if row else set()
        if reason:
            (reasons.discard if complete else reasons.add)(reason)
        if evidence_manifest_id:
            evidence.add(evidence_manifest_id)
        self.set_quality_dimensions(
            ticker, policy_version=policy_version, incomplete_reasons=reasons,
            evidence_manifest_ids=evidence, **values,
        )

    def refresh_market_core_quality(self, market: ResearchMarket) -> None:
        self.mark_quality_dimension(market.ticker, "metadata_complete", True)
        self.mark_quality_dimension(
            market.ticker, "target_complete", market.target_price is not None,
            reason="missing_target",
        )
        self.mark_quality_dimension(
            market.ticker, "official_result_complete",
            market.official_result is not None or market.expiration_value is not None,
            reason="missing_official_result",
        )

    def save_shadow_decision(self, decision: object) -> None:
        """Persist an immutable Paper-shadow decision without any order side effect."""
        with self._lock, self.connection:
            self.connection.execute(
                """INSERT INTO research_paper_shadow_decisions(
                    ticker,observed_at,dataset_version,policy_version,model_version,
                    selected_action,selected_side,masked_actions_json,reason,
                    quality_state,broker_result_json,kill_switch_active
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    getattr(decision, "ticker"), _iso(getattr(decision, "observed_at")),
                    getattr(decision, "dataset_version"), getattr(decision, "policy_version"),
                    getattr(decision, "model_version"), getattr(decision, "selected_action"),
                    getattr(decision, "selected_side"),
                    json.dumps(list(getattr(decision, "masked_actions"))),
                    getattr(decision, "reason"), getattr(decision, "quality_state"),
                    json.dumps(getattr(decision, "broker_result"), default=str),
                    int(bool(getattr(decision, "kill_switch_active"))),
                ),
            )

    def save_experiment_result(
        self, result: object, candidates: Iterable[object], context: object,
        runner_config: dict[str, object],
    ) -> None:
        """Atomically persist the frozen finalists and every evaluated run."""
        experiment_id = str(getattr(result, "experiment_id"))
        dataset_version = str(getattr(context, "dataset_version"))
        selected_ids = {
            str(getattr(item, "trial_id")) for item in getattr(result, "selected")
        }
        candidate_items = tuple(candidates)
        completed_at = _iso(getattr(result, "completed_at"))
        with self._lock, self.connection:
            self.connection.execute(
                """INSERT INTO research_experiments(
                    experiment_id,dataset_version,split_manifest_id,config_json,
                    code_commit,status,started_at,completed_at,finalist_hash,
                    final_test_unlocked
                ) VALUES (?,?,?,?,?,'completed',?,?,?,0)""",
                (
                    experiment_id, dataset_version,
                    str(getattr(context, "split_manifest_id")),
                    json.dumps(runner_config, sort_keys=True, default=str),
                    getattr(context, "code_commit"),
                    _iso(getattr(result, "started_at")), completed_at,
                    str(getattr(result, "finalist_hash")),
                ),
            )
            for candidate in candidate_items:
                candidate_id = str(getattr(candidate, "candidate_id"))
                parameters = getattr(candidate, "parameters")
                self.connection.execute(
                    """INSERT INTO research_experiment_candidates(
                        experiment_id,candidate_id,model_type,config_json,
                        config_hash,selected
                    ) VALUES (?,?,?,?,?,?)""",
                    (
                        experiment_id, candidate_id,
                        str(getattr(candidate, "model_type")),
                        json.dumps(parameters, sort_keys=True, default=str),
                        str(getattr(candidate, "config_hash")),
                        int(candidate_id in selected_ids),
                    ),
                )
            for run in getattr(result, "runs"):
                identity = "|".join((
                    experiment_id, str(getattr(run, "candidate_id")),
                    str(getattr(run, "fold_id")), str(getattr(run, "seed")),
                    str(getattr(run, "stress_profile")), str(getattr(run, "split")),
                    str(getattr(run, "resource_fraction")),
                ))
                metrics = {
                    "net_return": getattr(run, "net_return"),
                    "pnl_series": getattr(run, "pnl_series"),
                    "closed_trades": getattr(run, "closed_trades"),
                    "maximum_drawdown": getattr(run, "maximum_drawdown"),
                    "sharpe": getattr(run, "sharpe"),
                    "cvar": getattr(run, "cvar"),
                }
                self.connection.execute(
                    """INSERT INTO research_trial_runs(
                        run_id,parent_experiment_id,candidate_id,dataset_version,
                        fold_id,seed,stress_profile,split,resource_fraction,status,
                        started_at,completed_at,metrics_json,trajectory_hash,artifact_path
                    ) VALUES (?,?,?,?,?,?,?,?,?,'completed',?,?,?,?,?)""",
                    (
                        hashlib.sha256(identity.encode()).hexdigest(), experiment_id,
                        str(getattr(run, "candidate_id")), dataset_version,
                        str(getattr(run, "fold_id")), int(getattr(run, "seed")),
                        str(getattr(run, "stress_profile")), str(getattr(run, "split")),
                        str(getattr(run, "resource_fraction")),
                        _iso(getattr(result, "started_at")), completed_at,
                        json.dumps(metrics, sort_keys=True),
                        str(getattr(run, "trajectory_hash")),
                        getattr(context, "artifact_root"),
                    ),
                )

    def legacy_markets_missing_metadata(self, limit: int = 5) -> list[tuple[str, str]]:
        """Return a bounded set of old quote tickers for gradual REST backfill."""
        with self._lock:
            has_quotes = self.connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='quote_events'"
            ).fetchone()
            if has_quotes is None:
                return []
            rows = self.connection.execute(
                """SELECT q.asset,q.ticker
                   FROM quote_events q
                   LEFT JOIN research_markets m ON m.ticker=q.ticker
                   LEFT JOIN research_metadata_backfill_attempts a ON a.ticker=q.ticker
                   WHERE m.ticker IS NULL AND COALESCE(a.attempts,0) < 3
                   GROUP BY q.asset,q.ticker
                   ORDER BY MAX(q.observed_at) DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        return [(str(row["asset"]), str(row["ticker"])) for row in rows]

    def record_backfill_attempt(
        self, ticker: str, status: str, error_code: str | None = None
    ) -> None:
        with self._lock, self.connection:
            self.connection.execute(
                """INSERT INTO research_metadata_backfill_attempts(
                    ticker,attempts,status,last_attempt_at,error_code
                ) VALUES (?,1,?,?,?)
                ON CONFLICT(ticker) DO UPDATE SET
                    attempts=attempts+1,status=excluded.status,
                    last_attempt_at=excluded.last_attempt_at,error_code=excluded.error_code""",
                (ticker, status, datetime.now(UTC).isoformat(), error_code),
            )

    def start_session(
        self, session_id: str, stream: str, config_sha256: str, build_commit: str | None
    ) -> None:
        with self._lock, self.connection:
            self.connection.execute(
                """INSERT INTO research_collector_sessions(
                    session_id,stream,started_at,status,config_sha256,build_commit
                ) VALUES (?,?,?,'connected',?,?)""",
                (session_id, stream, datetime.now(UTC).isoformat(), config_sha256, build_commit),
            )

    def finish_session(
        self, session_id: str, status: str, *, last_seq: int | None = None,
        error_code: str | None = None,
    ) -> None:
        with self._lock, self.connection:
            self.connection.execute(
                """UPDATE research_collector_sessions SET
                    ended_at=?,status=?,last_seq=?,error_code=? WHERE session_id=?""",
                (datetime.now(UTC).isoformat(), status, last_seq, error_code, session_id),
            )

    def checkpoint(self, stream: str, partition_key: str) -> sqlite3.Row | None:
        with self._lock:
            return self.connection.execute(
                """SELECT * FROM research_stream_checkpoints
                   WHERE stream=? AND partition_key=?""",
                (stream, partition_key),
            ).fetchone()

    def register_file_and_checkpoints(
        self, *, file_id: str, dataset: str, path: str, sha256: str,
        row_count: int, min_event_time: datetime, max_event_time: datetime,
        min_seq: int | None, max_seq: int | None, session_id: str,
        checkpoints: Iterable[tuple[str, int, int | None, str]],
        schema_version: int = 1, byte_size: int | None = None,
        config_sha256: str | None = None, partition: dict[str, object] | None = None,
        parent_manifest_ids: Iterable[str] = (),
    ) -> None:
        now = datetime.now(UTC).isoformat()
        with self._lock, self.connection:
            self.connection.execute(
                """INSERT INTO research_files(
                    file_id,dataset,path,sha256,row_count,min_event_time,max_event_time,
                    min_seq,max_seq,schema_version,created_at,status,byte_size,
                    config_sha256,partition_json,parent_manifest_ids_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,'committed',?,?,?,?)""",
                (
                    file_id, dataset, path, sha256, row_count, _iso(min_event_time),
                    _iso(max_event_time), min_seq, max_seq, schema_version, now,
                    byte_size, config_sha256,
                    json.dumps(partition or {}, sort_keys=True, separators=(",", ":")),
                    json.dumps(sorted(set(parent_manifest_ids))),
                ),
            )
            for partition_key, source_ms, seq, event_hash in checkpoints:
                self.connection.execute(
                    """INSERT INTO research_stream_checkpoints(
                        stream,partition_key,session_id,last_source_timestamp_ms,
                        last_seq,last_event_hash,updated_at
                    ) VALUES (?,?,?,?,?,?,?)
                    ON CONFLICT(stream,partition_key) DO UPDATE SET
                        session_id=excluded.session_id,
                        last_source_timestamp_ms=excluded.last_source_timestamp_ms,
                        last_seq=excluded.last_seq,last_event_hash=excluded.last_event_hash,
                        updated_at=excluded.updated_at""",
                    (dataset, partition_key, session_id, source_ms, seq, event_hash, now),
                )

    def quality_event(
        self, *, stream: str, reason_code: str, severity: str = "warning",
        observed_at: datetime | None = None, ticker: str | None = None,
        index_id: str | None = None, event_time: datetime | None = None,
        missing_sample_count: int = 0, expected_seq: int | None = None,
        observed_seq: int | None = None, stale_seconds: str | None = None,
        queue_depth: int | None = None, session_id: str | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        observed = observed_at or datetime.now(UTC)
        with self._lock, self.connection:
            self.connection.execute(
                """INSERT INTO research_quality_events(
                    ticker,index_id,stream,event_time,observed_at,reason_code,severity,
                    missing_sample_count,expected_seq,observed_seq,stale_seconds,
                    queue_depth,session_id,details_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    ticker, index_id, stream, _iso(event_time), _iso(observed),
                    reason_code, severity, missing_sample_count, expected_seq,
                    observed_seq, stale_seconds, queue_depth, session_id,
                    json.dumps(details or {}, sort_keys=True, separators=(",", ":")),
                ),
            )
            if ticker is not None and severity == "error":
                rti_increment = int(reason_code in {"rti_gap", "sequence_gap", "stale_source"})
                book_increment = int(reason_code == "book_seq_gap")
                self.connection.execute(
                    """INSERT INTO research_market_quality(
                        ticker,status,reason_code,updated_at,rti_gaps,orderbook_gaps
                    ) VALUES (?,'incomplete',?,?,?,?)
                    ON CONFLICT(ticker) DO UPDATE SET
                        status='incomplete',reason_code=excluded.reason_code,
                        updated_at=excluded.updated_at,
                        rti_gaps=rti_gaps+excluded.rti_gaps,
                        orderbook_gaps=orderbook_gaps+excluded.orderbook_gaps""",
                    (ticker, reason_code, _iso(observed), rti_increment, book_increment),
                )
                self.connection.execute(
                    "UPDATE research_markets SET quality_status='incomplete' WHERE ticker=?",
                    (ticker,),
                )

    def recover_market_quality(
        self, ticker: str, allowed_reasons: set[str], observed_at: datetime
    ) -> bool:
        """Clear only transient stale/startup failures after verified recovery.

        Historical quality events remain immutable. Window gaps and queue loss
        are intentionally not recoverable on a later healthy tick.
        """
        with self._lock, self.connection:
            row = self.connection.execute(
                "SELECT status,reason_code FROM research_market_quality WHERE ticker=?",
                (ticker,),
            ).fetchone()
            if (
                row is None or row["status"] != "incomplete"
                or row["reason_code"] not in allowed_reasons
            ):
                return False
            target = self.connection.execute(
                "SELECT target_price FROM research_markets WHERE ticker=?", (ticker,)
            ).fetchone()
            if target is None or target["target_price"] is None:
                return False
            self.connection.execute(
                """UPDATE research_market_quality SET
                    status='complete',reason_code=NULL,updated_at=? WHERE ticker=?""",
                (_iso(observed_at), ticker),
            )
            self.connection.execute(
                "UPDATE research_markets SET quality_status='complete' WHERE ticker=?",
                (ticker,),
            )
            return True

    def status(self) -> dict[str, object]:
        with self._lock:
            sessions = [dict(row) for row in self.connection.execute(
                """SELECT stream,status,started_at,ended_at,last_seq,error_code
                   FROM research_collector_sessions ORDER BY started_at DESC LIMIT 10"""
            )]
            files = self.connection.execute(
                "SELECT COUNT(*) count,COALESCE(SUM(row_count),0) rows FROM research_files"
            ).fetchone()
            markets = self.connection.execute(
                """SELECT COUNT(*) count,
                   COALESCE(SUM(CASE WHEN quality_status='incomplete' THEN 1 ELSE 0 END),0) incomplete
                   FROM research_markets"""
            ).fetchone()
            quality = [dict(row) for row in self.connection.execute(
                """SELECT stream,reason_code,severity,observed_at,index_id,ticker
                   FROM research_quality_events ORDER BY id DESC LIMIT 20"""
            )]
        return {
            "sessions": sessions,
            "files": {"count": files["count"], "rows": files["rows"]},
            "markets": {"count": markets["count"], "incomplete": markets["incomplete"]},
            "recent_quality_events": quality,
            "storage": self.storage_health(),
        }

    def storage_health(self) -> dict[str, object]:
        with self._lock:
            rows = self.connection.execute(
                """SELECT dataset,COUNT(*) file_count,COALESCE(SUM(row_count),0) rows,
                          COALESCE(SUM(byte_size),0) bytes,
                          MIN(min_event_time) first_event,MAX(max_event_time) last_event
                   FROM research_files GROUP BY dataset ORDER BY dataset"""
            ).fetchall()
        return {
            "datasets": [dict(row) for row in rows],
            "bytes_total": sum(int(row["bytes"]) for row in rows),
        }

    def coverage_report(self) -> dict[str, object]:
        """Per-market purpose-specific coverage; never infers from coarse v1 status."""
        with self._lock:
            rows = self.connection.execute(
                """SELECT m.ticker,m.asset,m.open_time,m.close_time,
                          q.metadata_complete,q.target_complete,
                          q.official_result_complete,q.rti_target_window_complete,
                          q.rti_settlement_window_complete,q.contract_quotes_complete,
                          q.orderbook_complete,q.fee_metadata_complete,
                          q.feature_complete,q.replay_eligible,q.training_eligible,
                          q.incomplete_reasons_json,q.evaluated_at
                   FROM research_markets m
                   LEFT JOIN research_quality_dimensions q
                     ON q.ticker=m.ticker AND q.quality_policy_version='research-v1'
                   ORDER BY m.close_time"""
            ).fetchall()
        markets = []
        for row in rows:
            item = dict(row)
            item["incomplete_reasons"] = json.loads(
                item.pop("incomplete_reasons_json") or '["quality_not_evaluated"]'
            )
            markets.append(item)
        return {
            "quality_policy_version": "research-v1",
            "generated_at": datetime.now(UTC).isoformat(),
            "market_count": len(markets), "markets": markets,
        }

    def latest_markets_by_asset(self, assets: tuple[str, ...]) -> dict[str, dict[str, object]]:
        """Return the newest market projection for each requested asset."""
        result: dict[str, dict[str, object]] = {}
        with self._lock:
            for asset in assets:
                row = self.connection.execute(
                    """SELECT ticker,title,open_time,close_time,target_price,
                              target_source_field,quality_status,refreshed_at
                       FROM research_markets WHERE asset=?
                       ORDER BY close_time DESC LIMIT 1""",
                    (asset,),
                ).fetchone()
                if row is not None:
                    result[asset] = dict(row)
        return result
