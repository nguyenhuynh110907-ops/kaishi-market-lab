from fastapi.testclient import TestClient
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
import subprocess

import httpx
from kaishi_bot.web import create_app
from kaishi_bot.dashboard_models import QuotePoint
from kaishi_bot.fees import FeeSchedule


class FeeFeed:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    async def fee_schedule(self, series: str, *, refresh: bool = False) -> FeeSchedule:
        if self.fail:
            raise RuntimeError("fee metadata unavailable")
        return FeeSchedule("quadratic", Decimal("1"), Decimal("0.07"), "test")

    async def close(self) -> None:
        pass


def test_dashboard_state_starts_in_paper_with_one_thousand_dollars(tmp_path) -> None:
    app = create_app(tmp_path / "dashboard.sqlite3", start_background=False)
    with TestClient(app) as client:
        response = client.get("/api/state")
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["mode"] == "paper"
        assert data["paper"]["cash"] == "1000.00"
        assert set(data["assets"]) == {"BTC", "ETH", "SOL", "XRP", "DOGE"}


def test_settings_mode_bot_and_lab_controls_persist(tmp_path) -> None:
    app = create_app(
        tmp_path / "dashboard.sqlite3", start_background=False, market_data=FeeFeed()
    )
    with TestClient(app) as client:
        assert client.put("/api/settings", json={"entry_price": "0.20", "stop_loss": "0.10", "take_profit": "0.35"}).status_code == 200
        assert client.put("/api/bot", json={"enabled": True}).json()["data"]["bot_enabled"] is True
        assert client.put("/api/mode", json={"mode": "read_only"}).json()["data"]["mode"] == "read_only"
        started = client.post(
            "/api/lab/runs", json={"duration_seconds": 10800, "seed": 7}
        ).json()
        assert started["ok"] is True
        assert started["data"]["candidate_total"] == 25000
        assert started["data"]["history_cycles"] == 12
        assert started["data"]["initial_status"] == "backfilling"
        run_id = started["data"]["run_id"]
        run = next(item for item in app.state.runtime.lab.list_runs() if item["id"] == run_id)
        assert run["candidate_count"] == 5000
        preview = client.post(
            f"/api/lab/runs/{run_id}/candidates/BTC-000/copy-to-paper",
            json={"apply": False},
        ).json()["data"]
        assert preview["settings"]["mode"] == "paper"
        assert preview["applied"] is False
        assert client.post(f"/api/lab/runs/{run_id}/stop").json()["data"]["stopped"] is True
        candidate_rows = app.state.runtime.store.connection.execute(
            "SELECT COUNT(*) FROM lab_candidates WHERE run_id=?", (run_id,)
        ).fetchone()[0]
        assert candidate_rows == 25000


def test_quote_logging_toggle_persists_immediately(tmp_path) -> None:
    app = create_app(tmp_path / "dashboard.sqlite3", start_background=False)
    with TestClient(app) as client:
        response = client.put("/api/settings/log-quotes", json={"enabled": False})
        assert response.status_code == 200
        assert response.json()["data"]["log_quotes"] is False
        assert client.get("/api/state").json()["data"]["settings"]["log_quotes"] is False

        invalid = client.put("/api/settings/log-quotes", json={"enabled": "false"})
        assert invalid.status_code == 422


def test_research_status_is_independent_and_disabled_by_default(tmp_path) -> None:
    app = create_app(
        tmp_path / "dashboard.sqlite3", start_background=False,
        research_config_path=tmp_path / "missing-research.yaml",
    )
    with TestClient(app) as client:
        before = client.get("/api/research/status").json()["data"]
        client.put("/api/settings/log-quotes", json={"enabled": False})
        after = client.get("/api/research/status").json()["data"]
    assert before["enabled"] is False
    assert after["enabled"] is False
    assert after["rti_connected"] is False


def test_lab_history_cycles_can_be_disabled_and_are_range_checked(tmp_path) -> None:
    app = create_app(
        tmp_path / "dashboard.sqlite3", start_background=False, market_data=FeeFeed()
    )
    with TestClient(app) as client:
        invalid = client.post("/api/lab/runs", json={"history_cycles": 97})
        assert invalid.status_code == 422
        started = client.post(
            "/api/lab/runs",
            json={"assets": ["BTC"], "candidate_count": 1, "history_cycles": 0},
        )
        assert started.status_code == 200
        assert started.json()["data"]["initial_status"] == "running"


def test_lab_start_fails_closed_when_fee_metadata_is_unavailable(tmp_path) -> None:
    app = create_app(
        tmp_path / "dashboard.sqlite3", start_background=False,
        market_data=FeeFeed(fail=True),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/lab/runs", json={"assets": ["BTC"], "candidate_count": 1}
        )
        assert response.status_code == 409
        assert app.state.runtime.lab.list_runs() == []


def test_lab_start_rejects_duplicate_or_unknown_assets(tmp_path) -> None:
    app = create_app(
        tmp_path / "dashboard.sqlite3", start_background=False, market_data=FeeFeed()
    )
    with TestClient(app) as client:
        duplicate = client.post("/api/lab/runs", json={"assets": ["BTC", "BTC"]})
        unknown = client.post("/api/lab/runs", json={"assets": ["NOT_A_COIN"]})
    assert duplicate.status_code == 422
    assert unknown.status_code == 422


def test_lab_start_rejects_non_numeric_candidate_count(tmp_path) -> None:
    app = create_app(
        tmp_path / "dashboard.sqlite3", start_background=False, market_data=FeeFeed()
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/lab/runs", json={"assets": ["BTC"], "candidate_count": "many"}
        )
    assert response.status_code == 422


def test_lab_start_maps_fee_http_failure_to_closed_response(tmp_path) -> None:
    class HttpFailureFeed(FeeFeed):
        async def fee_schedule(self, series: str, *, refresh: bool = False) -> FeeSchedule:
            raise httpx.ConnectError("offline")

    app = create_app(
        tmp_path / "dashboard.sqlite3", start_background=False,
        market_data=HttpFailureFeed(),
    )
    with TestClient(app) as client:
        response = client.post("/api/lab/runs", json={"assets": ["BTC"]})
    assert response.status_code == 409


def test_live_without_credentials_fails_closed_and_secrets_are_absent(tmp_path) -> None:
    app = create_app(tmp_path / "dashboard.sqlite3", start_background=False, environ={})
    with TestClient(app) as client:
        client.put("/api/mode", json={"mode": "live"})
        response = client.post("/api/live/challenge")
        assert response.status_code == 409
        assert "private" not in client.get("/api/state").text.lower()


def test_static_dashboard_is_local_only(tmp_path) -> None:
    app = create_app(tmp_path / "dashboard.sqlite3", start_background=False)
    with TestClient(app) as client:
        response = client.get("/")
        html = response.text
        assert response.headers["cache-control"] == "no-store"
        assert "Strategy Lab" in html
        assert "https://" not in html
        assert "/static/app.css" in html
        assert "?v=live-ledger-6" in html
        assert "/static/lab_activity.js" in html
        assert "/static/paper_sessions.js" in html
        assert "/static/live_sessions.js" in html
        assert "/static/time_window.js" in html
        assert 'id="paper-sessions"' in html
        assert "DỮ LIỆU THEO PHIÊN 15 PHÚT" in html
        assert 'id="lab-status"' in html
        assert 'role="status"' in html
        assert 'id="stop-lab"' in html
        assert 'id="lab-history-cycles"' in html
        assert 'value="12"' in html
        assert 'id="price-band-control"' in html
        assert 'id="time-window-control"' in html
        assert 'name="entry_start_seconds"' in html
        assert 'name="entry_end_seconds"' in html
        for price_handle in ("stop_loss", "entry_min", "entry_price", "take_profit"):
            assert f'data-price-field="{price_handle}"' in html
        assert "25.000 chiến lược" in html
        assert "5.000 mỗi coin" in html
        assert 'name="paper_daily_cap"' in html
        assert 'class="paper-only"' in html
        assert 'id="cash-label"' in html
        assert 'id="positions-title"' in html
        assert 'id="activity-title"' in html
        assert 'id="live-status"' in html
        assert 'id="live-status-detail"' in html
        assert 'name="paper_daily_cap" type="number" min="0.01" max="10000"' in html
        assert 'name="daily_cap" form="settings-form" type="number" min="0.01" max="10000"' in html
        assert 'name="entry_amount" type="number" min="0.01" max="200"' in html
        assert 'name="entry_min"' in html
        assert '<details class="guard-advanced"' not in html
        assert 'id="guard-summary"' in html
        assert 'form="settings-form"' in html
        app_js = client.get("/static/app.js").text
        assert "Đã dùng" in app_js
        assert "daily_remaining" in app_js
        assert "Tiền khả dụng Kalshi" in app_js
        assert "VỊ THẾ THẬT TRÊN KALSHI" in app_js
        assert "GIAO DỊCH LIVE — ENTRY → TP/SL CÙNG HƯỚNG" in app_js
        assert "LiveSessions.assetFromTicker" in app_js
        assert "state.live_daily_used" in app_js
        assert "LIVE ĐÃ ARM" in app_js
        assert "LIVE ARMED" in app_js
        assert "price_transport" in app_js
        assert "Bỏ qua / Hết limit" in app_js
        assert "candidate_count:5000" in app_js
        assert "history_cycles" in app_js
        assert "ĐANG NẠP LỊCH SỬ" in app_js
        assert "Entry thấp–cao / TP / SL" in app_js
        assert "Bảng xếp hạng đủ mẫu" in app_js
        assert "Chưa đủ mẫu" in app_js
        assert "Đủ điều kiện / Đã thấy" in app_js
        assert "RUN CŨ" in app_js
        assert "ranked_candidates" in app_js
        assert "insufficient_candidates" in app_js
        assert "['running','backfilling'].includes(x.status)" in app_js
        assert "result.stopped?'Đã dừng Strategy Lab':'Run đã dừng trước đó'" in app_js
        assert "ROI" in app_js
        assert "ROI = P&L ròng sau phí / tổng vốn đã triển khai" in app_js
        assert "roi-positive" in app_js
        assert "roi-negative" in app_js
        assert "Chỉ mua ${start}–${end}, giá" in app_js
        assert "PaperSessions.groupPositions" in app_js
        assert "TimeWindow.bind" in app_js


def test_time_window_slider_clamps_handles_and_formats_clock() -> None:
    project = Path(__file__).parents[1]
    script = r"""
const assert = require('node:assert/strict');
const window = require('./src/kaishi_bot/static/time_window.js');
assert.deepEqual(window.normalize({start:300,end:600}), {start:300,end:600});
assert.deepEqual(window.normalize({start:610,end:600}, 'start'), {start:585,end:600});
assert.deepEqual(window.normalize({start:300,end:290}, 'end'), {start:300,end:315});
assert.equal(window.clock(0), '00:00');
assert.equal(window.clock(315), '05:15');
assert.equal(window.clock(900), '15:00');
"""
    result = subprocess.run(
        ["node", "-e", script], cwd=project, capture_output=True, text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_paper_sessions_group_positions_and_include_losses() -> None:
    project = Path(__file__).parents[1]
    script = r"""
const assert = require('node:assert/strict');
const sessions = require('./src/kaishi_bot/static/paper_sessions.js');
const grouped = sessions.groupPositions({
  positions: [{
    id: 3, asset: 'ETH', side: 'down', quantity: '5', entry_price: '0.20',
    opened_at: '2026-08-06T19:14:59+00:00', unrealized_pnl: '0.25', status: 'open'
  }],
  closed_positions: [
    {id: 1, asset: 'BTC', side: 'up', quantity: '10', entry_price: '0.20',
     opened_at: '2026-08-06T19:01:00+00:00', closed_at: '2026-08-06T19:03:00+00:00',
     exit_price: '0.40', realized_pnl: '1.90', close_reason: 'take_profit', status: 'closed'},
    {id: 2, asset: 'SOL', side: 'up', quantity: '10', entry_price: '0.30',
     opened_at: '2026-08-06T19:14:00+00:00', closed_at: '2026-08-06T19:16:00+00:00',
     exit_price: '0.19', realized_pnl: '-1.20', close_reason: 'stop_loss', status: 'closed'},
    {id: 4, asset: 'XRP', side: 'up', quantity: '4', entry_price: '0.25',
     opened_at: '2026-08-06T19:16:00+00:00', closed_at: '2026-08-06T19:18:00+00:00',
     exit_price: '0.20', realized_pnl: '-0.20', close_reason: 'manual', status: 'closed'}
  ]
});
assert.equal(grouped.length, 2);
assert.equal(grouped[0].label, '15:15–15:30');
assert.equal(grouped[0].netPnl, -0.20);
assert.equal(grouped[1].label, '15:00–15:15');
assert.equal(grouped[1].trades.length, 3);
assert.equal(grouped[1].closedCount, 2);
assert.equal(grouped[1].openCount, 1);
assert.equal(grouped[1].takeProfitCount, 1);
assert.equal(grouped[1].stopLossCount, 1);
assert.equal(grouped[1].netPnl, 0.95);
assert.equal(grouped[1].trades.find(x => x.id === 2).exitTime, '15:16:00');
"""
    result = subprocess.run(
        ["node", "-e", script], cwd=project, capture_output=True, text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_four_point_price_band_clamps_handles_in_order() -> None:
    project = Path(__file__).parents[1]
    script = r"""
const assert = require('node:assert/strict');
const band = require('./src/kaishi_bot/static/price_band.js');
assert.deepEqual(
  band.normalize({stop_loss:19,entry_min:27,entry_price:29,take_profit:45}, 'entry_min'),
  {stop_loss:19,entry_min:27,entry_price:29,take_profit:45}
);
assert.equal(
  band.normalize({stop_loss:19,entry_min:30,entry_price:29,take_profit:45}, 'entry_min').entry_min,
  28
);
assert.equal(
  band.normalize({stop_loss:19,entry_min:27,entry_price:46,take_profit:45}, 'entry_price').entry_price,
  44
);
"""
    result = subprocess.run(
        ["node", "-e", script], cwd=project, capture_output=True, text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_price_band_legend_uses_non_overlapping_columns() -> None:
    project = Path(__file__).parents[1]
    css = (project / "src/kaishi_bot/static/app.css").read_text()
    assert ".price-band-labels{display:grid;grid-template-columns:repeat(4,minmax(0,1fr))" in css
    assert ".price-band-labels span{position:static" in css


def test_roi_cells_use_the_exact_definition_tooltip() -> None:
    project = Path(__file__).parents[1]
    script = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const app = fs.readFileSync('./src/kaishi_bot/static/app.js', 'utf8');
const start = app.indexOf('function roiCell');
const end = app.indexOf('function draw');
const roiCell = Function('money', `${app.slice(start, end)}; return roiCell`)(
  value => `$${Number(value).toFixed(2)}`
);
const definition = 'ROI = P&L ròng sau phí / tổng vốn đã triển khai';
for (const candidate of [
  {roi_percent: null, total_deployed: '0'},
  {roi_percent: '12.34', total_deployed: '50'},
  {roi_percent: '-12.34', total_deployed: '50'},
  {roi_percent: '0', total_deployed: '50'},
]) {
  const title = roiCell(candidate).match(/title="([^"]+)"/)[1];
  assert.equal(title, definition);
}
"""
    result = subprocess.run(
        ["node", "-e", script], cwd=project, capture_output=True, text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_manual_close_uses_current_executable_bid_in_paper_mode(tmp_path) -> None:
    app = create_app(tmp_path / "dashboard.sqlite3", start_background=False)
    runtime = app.state.runtime
    now = datetime.now(UTC)
    position_id = runtime.store.open_position(
        asset="BTC", ticker="BTC-1", side="up", quantity=Decimal("10"),
        entry_price=Decimal("0.20"), opened_at=now, day="2026-08-03",
    )
    runtime.latest_quotes["BTC-1"] = QuotePoint(
        observed_at=now, up_bid="0.30", up_ask="0.31", down_bid="0.69", down_ask="0.70"
    )
    with TestClient(app) as client:
        response = client.post(f"/api/positions/{position_id}/close")
        assert response.status_code == 200
        assert response.json()["data"]["closed"] is True


def test_realtime_render_does_not_overwrite_unsaved_control_deck_values() -> None:
    project = Path(__file__).parents[1]
    script = r"""
const assert = require('node:assert/strict');
const formState = require('./src/kaishi_bot/static/settings_form.js');
const names = ['entry_min','entry_price','take_profit','stop_loss','entry_amount','paper_daily_cap','daily_cap','entry_start_seconds','entry_end_seconds'];
const form = {dataset:{}, elements:Object.fromEntries(names.map(name=>[name,{value:''}]))};
let changeHandler = null;
const assetContainer = {
  dataset:{},
  addEventListener(type, handler) { assert.equal(type, 'change'); changeHandler = handler; }
};
let assetRenders = 0;
const saved = {
  entry_min:'0.17', entry_price:'0.25', take_profit:'0.40', stop_loss:'0.15',
  entry_amount:'10.00', paper_daily_cap:'1000.00', daily_cap:'1000.00',
  entry_start_seconds:0, entry_end_seconds:900,
  entry_guard:{entry_floor_ratio:'0.60',stop_loss_buffer:'0.02',max_spread:'0.01',max_spread_ratio:'0.15',confirmation_ticks:2,minimum_reward_risk:'1.50',reentry_cooldown_seconds:10}
};
assert.equal(formState.sync(form, saved, ()=>assetRenders++), true);
assert.equal(form.elements.entry_min.value, '0.17');
assert.equal(form.elements.entry_price.value, '0.25');
assert.equal(form.elements.paper_daily_cap.value, '1000.00');
formState.markDirty(form);
form.elements.entry_price.value = '0.30';
form.elements.paper_daily_cap.value = '850.00';
assert.equal(formState.sync(form, saved, ()=>assetRenders++), false);
assert.equal(form.elements.entry_price.value, '0.30');
assert.equal(form.elements.paper_daily_cap.value, '850.00');
assert.equal(assetRenders, 1);
formState.markSaved(form);
assert.equal(formState.sync(form, {...saved, entry_price:'0.30', paper_daily_cap:'850.00'}, ()=>assetRenders++), true);
assert.equal(form.elements.entry_price.value, '0.30');
assert.equal(form.elements.paper_daily_cap.value, '850.00');
formState.bindDirtyContainer(form, assetContainer);
assert.equal(typeof changeHandler, 'function');
changeHandler({target:{matches: selector => selector === 'input'}});
assert.equal(form.dataset.settingsDirty, 'true');
formState.bindDirtyContainer(form, assetContainer);
"""
    result = subprocess.run(
        ["node", "-e", script], cwd=project, capture_output=True, text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_lab_activity_requires_recent_quote_progress() -> None:
    project = Path(__file__).parents[1]
    script = r"""
const assert = require('node:assert/strict');
const activity = require('./src/kaishi_bot/static/lab_activity.js');
const tracker = activity.createTracker(5000);
const run = (status, quote_count) => ({id:3, status, quote_count});
assert.equal(tracker.observe(run('running', 10), 1000).kind, 'waiting');
const active = tracker.observe(run('running', 11), 2000);
assert.equal(active.kind, 'active');
assert.equal(active.label, 'ĐANG CHẠY');
assert.equal(active.ageSeconds, 0);
assert.equal(tracker.observe(run('running', 11), 8001).kind, 'waiting');
const completed = tracker.observe(run('completed', 11), 8002);
assert.equal(completed.kind, 'finished');
assert.equal(completed.label, 'HOÀN TẤT');
const nextRun = tracker.observe({id:4,status:'running',quote_count:99}, 9000);
assert.equal(nextRun.kind, 'waiting');
assert.equal(nextRun.lastActivityAt, null);
"""
    result = subprocess.run(
        ["node", "-e", script], cwd=project, capture_output=True, text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
