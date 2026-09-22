from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from kaishi_bot.config import Credentials
from kaishi_bot.dashboard_models import DashboardSettings, RuntimeMode
from kaishi_bot.dashboard_runtime import DashboardRuntime
from kaishi_bot.dashboard_store import DashboardStore
from kaishi_bot.lab import StrategyLab
from kaishi_bot.market_data import PublicKalshiMarketData
from kaishi_bot.model_registry import ModelRegistry, ModelRegistryError
from kaishi_bot.paper_model_shadow import PaperModelShadow
from kaishi_bot.production import ProductionGateway
from kaishi_bot.research_capture import (
    ResearchCaptureSupervisor,
    production_websocket_factory,
)
from kaishi_bot.research_config import ResearchCaptureConfig
from kaishi_bot.research_market_data import ResearchMarketClient
from kaishi_bot.research_store import ResearchStore
from kaishi_bot.safety import SafetyError, SafetyGate


STATIC_DIR = Path(__file__).with_name("static")


def envelope(data: object = None, error: str | None = None) -> dict[str, object]:
    return {"ok": error is None, "data": data, "error": error}


def create_app(
    db_path: Path = Path("dashboard.sqlite3"), *,
    market_data: PublicKalshiMarketData | None = None,
    start_background: bool = True,
    environ: Mapping[str, str] | None = None,
    research_config_path: Path | None = None,
) -> FastAPI:
    environment = dict(os.environ if environ is None else environ)
    store = DashboardStore(db_path)
    credentials = None
    try:
        credentials = Credentials.load(environment)
    except ValueError:
        pass
    gate = SafetyGate(credentials_available=credentials is not None)
    settings = store.load_settings()
    gate.set_mode(settings.mode)
    production = ProductionGateway.create(credentials, gate) if credentials else None
    feed = market_data or (PublicKalshiMarketData.production() if start_background else None)
    runtime = DashboardRuntime(store, feed, gate, production)
    configured_research_path = research_config_path
    if configured_research_path is None:
        configured_research_path = Path(
            environment.get("KAISHI_RESEARCH_CONFIG", "research_capture.yaml")
        )
    research_config = ResearchCaptureConfig.load(configured_research_path)
    research_store = ResearchStore(db_path)
    research_market_client = (
        ResearchMarketClient.production()
        if research_config.enabled and research_config.persist_market_metadata
        else None
    )
    research_websocket_factory = (
        production_websocket_factory(credentials)
        if research_config.enabled and credentials and (
            research_config.persist_rti
            or research_config.persist_contract_quotes
            or research_config.persist_orderbook
        )
        else None
    )
    research = ResearchCaptureSupervisor(
        research_config, research_store, research_market_client,
        research_websocket_factory,
    )
    model_registry = ModelRegistry(research_config.root / "artifacts")
    paper_model_shadow = PaperModelShadow(
        model_registry, research, store, research_store
    )
    runtime.paper_model_shadow = paper_model_shadow
    research_status_cache: dict[str, object] | None = None
    research_status_cache_until = 0.0
    research_status_lock = asyncio.Lock()
    model_list_cache: list[dict[str, object]] | None = None
    model_list_cache_until = 0.0
    model_list_lock = asyncio.Lock()
    leaderboard_cache: dict[int, dict[str, object]] = {}
    leaderboard_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if start_background:
            await research.start()
            await runtime.start()
        yield
        await runtime.close()
        await research.close()
        research_store.close()
        store.close()

    app = FastAPI(title="Kaishi Local Dashboard", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.runtime = runtime
    app.state.research = research
    app.state.model_registry = model_registry
    app.state.paper_model_shadow = paper_model_shadow
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(
            STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"}
        )

    @app.get("/health")
    async def health() -> dict[str, object]:
        return envelope({"status": "ok", "version": runtime.version})

    @app.get("/api/state")
    async def state() -> dict[str, object]:
        return envelope(runtime.latest_snapshot())

    @app.get("/api/research/status")
    async def research_status() -> dict[str, object]:
        nonlocal research_status_cache, research_status_cache_until
        now = monotonic()
        if research_status_cache is not None and now < research_status_cache_until:
            return envelope(research_status_cache)
        async with research_status_lock:
            now = monotonic()
            if (
                research_status_cache is None
                or now >= research_status_cache_until
            ):
                # SQLite lives on the mounted D: drive. Never block the WS
                # event loop while the dashboard polls status from many tabs.
                research_status_cache = await asyncio.to_thread(research.status)
                research_status_cache_until = monotonic() + 2.0
        return envelope(research_status_cache)

    @app.get("/api/research/coverage")
    async def research_coverage() -> dict[str, object]:
        return envelope(await asyncio.to_thread(research_store.coverage_report))

    @app.get("/api/research/models")
    async def research_models() -> dict[str, object]:
        nonlocal model_list_cache, model_list_cache_until
        now = monotonic()
        if model_list_cache is None or now >= model_list_cache_until:
            async with model_list_lock:
                now = monotonic()
                if model_list_cache is None or now >= model_list_cache_until:
                    model_list_cache = await asyncio.to_thread(model_registry.list)
                    model_list_cache_until = monotonic() + 5.0
        return envelope({
            "models": model_list_cache,
            "paper_shadow": paper_model_shadow.status(),
            "artifact_root": "data/research/artifacts",
        })

    @app.get("/api/research/models/{model_id}")
    async def research_model(model_id: str) -> dict[str, object]:
        try:
            return envelope(model_registry.get(model_id))
        except ModelRegistryError as error:
            raise HTTPException(404, str(error)) from error

    @app.post("/api/research/models/{model_id}/paper-shadow/start")
    async def start_model_shadow(model_id: str) -> dict[str, object]:
        try:
            result = paper_model_shadow.activate(model_id)
        except (ModelRegistryError, RuntimeError, ValueError) as error:
            raise HTTPException(409, str(error)) from error
        await runtime.publish()
        return envelope(result)

    @app.post("/api/research/paper-shadow/stop")
    async def stop_model_shadow() -> dict[str, object]:
        result = paper_model_shadow.deactivate()
        await runtime.publish()
        return envelope(result)

    @app.get("/api/events")
    async def events(request: Request) -> StreamingResponse:
        queue = runtime.subscribe()
        delta_stream = request.query_params.get("stream") == "delta-v1"
        async def stream():
            try:
                yield f"data: {json.dumps(runtime.latest_snapshot())}\n\n"
                while not await request.is_disconnected():
                    data = await queue.get()
                    if not delta_stream:
                        data = json.dumps(runtime.latest_snapshot())
                    yield f"data: {data}\n\n"
            finally:
                runtime.unsubscribe(queue)
        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.put("/api/settings")
    async def update_settings(payload: dict[str, object]) -> dict[str, object]:
        current = store.load_settings().model_dump()
        current.update(payload)
        current["mode"] = store.load_settings().mode
        current["bot_enabled"] = store.load_settings().bot_enabled
        try:
            changed = DashboardSettings.model_validate(current)
        except ValidationError as error:
            raise HTTPException(422, str(error)) from error
        store.save_settings(changed)
        await runtime.publish()
        return envelope(changed.model_dump(mode="json"))

    @app.put("/api/mode")
    async def update_mode(payload: dict[str, object]) -> dict[str, object]:
        try:
            mode = RuntimeMode(str(payload.get("mode", "")))
        except ValueError as error:
            raise HTTPException(422, "mode must be paper, read_only, or live") from error
        settings = store.load_settings().model_copy(update={"mode": mode})
        store.save_settings(settings)
        gate.set_mode(mode)
        if mode is not RuntimeMode.PAPER:
            paper_model_shadow.deactivate()
        if mode is RuntimeMode.LIVE:
            runtime.stop_active_labs()
        await runtime.publish()
        return envelope({"mode": mode.value, "armed": False})

    @app.put("/api/bot")
    async def update_bot(payload: dict[str, object]) -> dict[str, object]:
        enabled = payload.get("enabled")
        if not isinstance(enabled, bool):
            raise HTTPException(422, "enabled must be boolean")
        settings = store.load_settings().model_copy(update={"bot_enabled": enabled})
        store.save_settings(settings)
        await runtime.publish()
        return envelope({"bot_enabled": enabled})

    @app.put("/api/settings/log-quotes")
    async def update_quote_logging(payload: dict[str, object]) -> dict[str, object]:
        enabled = payload.get("enabled")
        if not isinstance(enabled, bool):
            raise HTTPException(422, "enabled must be boolean")
        settings = store.load_settings().model_copy(update={"log_quotes": enabled})
        store.save_settings(settings)
        await runtime.publish()
        return envelope({"log_quotes": enabled})

    @app.post("/api/live/challenge")
    async def live_challenge() -> dict[str, object]:
        try:
            return envelope({"phrase": gate.create_challenge(), "expires_in": 60})
        except SafetyError as error:
            raise HTTPException(409, str(error)) from error

    @app.post("/api/live/arm")
    async def live_arm(payload: dict[str, object]) -> dict[str, object]:
        try:
            gate.arm(str(payload.get("phrase", "")))
        except SafetyError as error:
            raise HTTPException(409, str(error)) from error
        runtime.stop_active_labs()
        return envelope({"armed": True})

    @app.post("/api/live/disarm")
    async def live_disarm() -> dict[str, object]:
        gate.disarm()
        return envelope({"armed": False})

    @app.post("/api/paper/reset")
    async def reset_paper(payload: dict[str, object]) -> dict[str, object]:
        if not store.reset_paper(str(payload.get("confirmation", ""))):
            raise HTTPException(409, "type RESET PAPER $1000 to confirm")
        await runtime.publish()
        return envelope({"cash": "1000.00"})

    @app.post("/api/positions/{position_id}/close")
    async def close_paper_position(position_id: int) -> dict[str, object]:
        if store.load_settings().mode is not RuntimeMode.PAPER:
            raise HTTPException(409, "manual close is available in Paper mode only")
        position = next(
            (item for item in store.open_positions() if int(item["id"]) == position_id),
            None,
        )
        if position is None:
            raise HTTPException(404, "open position not found")
        quote = runtime.latest_quotes.get(str(position["ticker"]))
        if quote is None:
            raise HTTPException(409, "no current executable quote for this position")
        closed = runtime.paper.close_position(position_id, quote, datetime.now(UTC))
        if not closed:
            raise HTTPException(409, "position could not be closed")
        await runtime.publish()
        return envelope({"closed": True})

    @app.post("/api/lab/runs")
    async def start_lab(payload: dict[str, object]) -> dict[str, object]:
        settings = store.load_settings()
        if settings.mode is RuntimeMode.LIVE:
            raise HTTPException(
                409, "Strategy Lab đã tắt trong Live; chuyển sang Paper hoặc Chỉ xem"
            )
        raw_assets = payload.get("assets", list(settings.assets))
        if not isinstance(raw_assets, list) or not raw_assets:
            raise HTTPException(422, "assets must be a non-empty list")
        assets = [str(asset) for asset in raw_assets]
        if len(assets) != len(set(assets)):
            raise HTTPException(422, "assets must be unique")
        unknown = [asset for asset in assets if asset not in settings.assets]
        if unknown:
            raise HTTPException(422, f"unsupported assets: {', '.join(unknown)}")
        try:
            count = int(payload.get("candidate_count", 5000))
            seed = int(payload.get("seed", 42))
            duration_seconds = int(payload.get("duration_seconds", 10800))
            history_cycles = int(payload.get("history_cycles", 12))
        except (TypeError, ValueError) as error:
            raise HTTPException(
                422, "candidate_count, seed, duration, and history cycles must be integers"
            ) from error
        if not 0 <= history_cycles <= 96:
            raise HTTPException(422, "history cycles must be between 0 and 96")
        if runtime.market_data is None or not hasattr(runtime.market_data, "fee_schedule"):
            raise HTTPException(409, "verified Kalshi fee metadata is unavailable")
        try:
            fee_schedules = {
                asset: await runtime.market_data.fee_schedule(
                    settings.assets[asset].series, refresh=True
                )
                for asset in assets
            }
        except (httpx.HTTPError, KeyError, RuntimeError, ValueError) as error:
            raise HTTPException(409, f"verified Kalshi fee metadata is unavailable: {error}") from error
        try:
            cutoff_row = store.connection.execute(
                "SELECT MAX(id) FROM quote_events"
            ).fetchone()
            cutoff_event_id = int(cutoff_row[0] or 0)
            run_id = runtime.lab.start(
                assets, count, seed, duration_seconds, fee_schedules,
                settings.entry_guard, history_cycles, cutoff_event_id,
            )
        except (TypeError, ValueError) as error:
            raise HTTPException(422, str(error)) from error
        initial_status = "backfilling" if history_cycles else "running"
        if history_cycles:
            runtime.start_lab_backfill(run_id)
        return envelope({
            "run_id": run_id, "candidate_total": len(assets) * count,
            "history_cycles": history_cycles, "initial_status": initial_status,
        })

    @app.get("/api/lab/runs")
    async def list_lab_runs() -> dict[str, object]:
        return envelope(runtime.lab.list_runs())

    @app.post("/api/lab/runs/{run_id}/stop")
    async def stop_lab(run_id: int) -> dict[str, object]:
        return envelope({"stopped": runtime.lab.stop(run_id)})

    @app.get("/api/lab/runs/{run_id}/leaderboard")
    async def lab_board(run_id: int) -> dict[str, object]:
        def read_leaderboard() -> dict[str, object]:
            # A sqlite connection must not be used concurrently by the event
            # loop and a worker thread. Give the heavy leaderboard query its
            # own short-lived read connection.
            with DashboardStore(store.path) as read_store:
                return StrategyLab(read_store).leaderboard(run_id)

        try:
            cached = leaderboard_cache.get(run_id)
            if cached is not None:
                return envelope(cached)
            async with leaderboard_lock:
                cached = leaderboard_cache.get(run_id)
                if cached is not None:
                    return envelope(cached)
                board = await asyncio.to_thread(read_leaderboard)
                # The browser renders only the leading rows. Avoid encoding
                # and transferring tens of thousands of candidate objects.
                ranked = list(board.get("ranked_candidates", []))
                insufficient = list(board.get("insufficient_candidates", []))
                board["ranked_candidate_count"] = len(ranked)
                board["insufficient_candidate_count"] = len(insufficient)
                board["ranked_candidates"] = ranked[:100]
                board["insufficient_candidates"] = insufficient[:100]
                board["candidates"] = ranked[:100] + insufficient[:100]
                if board.get("status") not in {"running", "backfilling", "replaying"}:
                    leaderboard_cache[run_id] = board
                return envelope(board)
        except KeyError as error:
            raise HTTPException(404, "lab run not found") from error

    @app.post("/api/lab/runs/{run_id}/replay")
    async def replay_lab(run_id: int) -> dict[str, object]:
        try:
            return envelope({"run_id": runtime.lab.replay(run_id)})
        except (KeyError, ValueError) as error:
            raise HTTPException(409, str(error)) from error

    @app.post("/api/lab/runs/{run_id}/candidates/{candidate_id}/copy-to-paper")
    async def copy_candidate(
        run_id: int, candidate_id: str, payload: dict[str, object]
    ) -> dict[str, object]:
        try:
            candidate = runtime.lab.candidate_settings(run_id, candidate_id)
        except KeyError as error:
            raise HTTPException(404, "candidate not found") from error
        current = store.load_settings()
        asset = str(candidate.pop("asset"))
        side_policy = str(candidate.pop("side_policy"))
        assets = dict(current.assets)
        assets[asset] = assets[asset].model_copy(update={
            "trade_up": side_policy in {"up", "both"},
            "trade_down": side_policy in {"down", "both"},
        })
        changed_payload = current.model_dump()
        changed_payload.update({
            **candidate, "mode": RuntimeMode.PAPER, "bot_enabled": False,
            "assets": assets,
        })
        changed = DashboardSettings.model_validate(changed_payload)
        apply = payload.get("apply") is True
        if apply:
            store.save_settings(changed)
            gate.set_mode(RuntimeMode.PAPER)
            await runtime.publish()
        return envelope({"applied": apply, "settings": changed.model_dump(mode="json")})

    return app
