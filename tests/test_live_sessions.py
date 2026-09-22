from pathlib import Path
import subprocess


def test_live_fills_are_paired_by_bot_intent_and_real_kalshi_prices() -> None:
    project = Path(__file__).parents[1]
    script = r"""
const assert = require('node:assert/strict');
const sessions = require('./src/kaishi_bot/static/live_sessions.js');
const grouped = sessions.groupActivity([
  {fill_id:'f1',order_id:'entry-up',ticker:'KXBTC15M-26AUG061915-15',
   side:'yes',action:'buy',count:'0.63',yes_price:'0.7700',no_price:'0.2300',
   fee_cost:'0.007900',is_taker:true,created_at:'2026-08-06T23:12:33.845578Z'},
  {fill_id:'f2',order_id:'exit-up',ticker:'KXBTC15M-26AUG061915-15',
   side:'no',action:'sell',count:'0.63',yes_price:'0.2200',no_price:'0.7800',
   fee_cost:'0.007600',is_taker:true,created_at:'2026-08-06T23:13:04.883899Z'},
  {fill_id:'f3',order_id:'entry-down',ticker:'KXBTC15M-26AUG061915-15',
   side:'no',action:'sell',count:'0.62',yes_price:'0.2200',no_price:'0.7800',
   fee_cost:'0.007500',is_taker:true,created_at:'2026-08-06T23:13:04.992971Z'},
  {fill_id:'f4',order_id:'exit-down',ticker:'KXBTC15M-26AUG061915-15',
   side:'yes',action:'buy',count:'0.62',yes_price:'0.0450',no_price:'0.9550',
   fee_cost:'0.001900',is_taker:true,created_at:'2026-08-06T23:13:48.597146Z'}
], [], [
  {order_id:'entry-up',ticker:'KXBTC15M-26AUG061915-15',side:'up',role:'entry',reason:'entry'},
  {order_id:'exit-up',ticker:'KXBTC15M-26AUG061915-15',side:'up',role:'exit',reason:'stop_loss'},
  {order_id:'entry-down',ticker:'KXBTC15M-26AUG061915-15',side:'down',role:'entry',reason:'entry'},
  {order_id:'exit-down',ticker:'KXBTC15M-26AUG061915-15',side:'down',role:'exit',reason:'take_profit'}
], {stop_loss:'0.49',take_profit:'0.94'});
assert.equal(sessions.assetFromTicker('KXBTC15M-26AUG061915-15'), 'BTC');
assert.equal(grouped.length, 1);
assert.equal(grouped[0].trades.length, 2);
assert.equal(grouped[0].closedCount, 2);
assert.equal(grouped[0].tpCount, 1);
assert.equal(grouped[0].slCount, 1);
assert.equal(grouped[0].trades[0].side, 'up');
assert.equal(grouped[0].trades[0].entry_price, 0.77);
assert.equal(grouped[0].trades[0].exit_price, 0.22);
assert.equal(grouped[0].trades[0].close_reason, 'stop_loss');
assert.ok(Math.abs(grouped[0].trades[0].pnl - (-0.3620)) < 0.000001);
assert.equal(grouped[0].trades[1].side, 'down');
assert.equal(grouped[0].trades[1].entry_price, 0.78);
assert.equal(grouped[0].trades[1].exit_price, 0.955);
assert.equal(grouped[0].trades[1].close_reason, 'take_profit');
assert.ok(Math.abs(grouped[0].trades[1].pnl - 0.0991) < 0.000001);
assert.equal(grouped[0].isOpen, false);
assert.ok(Math.abs(grouped[0].netPnl - (-0.2629)) < 0.000001);
"""
    result = subprocess.run(
        ["node", "-e", script], cwd=project, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr


def test_partial_entry_and_exit_fills_stay_in_one_logical_trade() -> None:
    project = Path(__file__).parents[1]
    script = r"""
const assert = require('node:assert/strict');
const sessions = require('./src/kaishi_bot/static/live_sessions.js');
const fills = [
  {fill_id:'e1',order_id:'entry-1',ticker:'KXETH15M-X',count:'2',yes_price:'0.78',no_price:'0.22',fee_cost:'0.02',created_at:'2026-08-07T13:01:00Z'},
  {fill_id:'e2',order_id:'entry-2',ticker:'KXETH15M-X',count:'3',yes_price:'0.80',no_price:'0.20',fee_cost:'0.03',created_at:'2026-08-07T13:02:00Z'},
  {fill_id:'x1',order_id:'exit-1',ticker:'KXETH15M-X',count:'2',yes_price:'0.50',no_price:'0.50',fee_cost:'0.02',created_at:'2026-08-07T13:05:00Z'},
  {fill_id:'x2',order_id:'exit-2',ticker:'KXETH15M-X',count:'3',yes_price:'0.95',no_price:'0.05',fee_cost:'0.01',created_at:'2026-08-07T13:08:00Z'},
  {fill_id:'x2',order_id:'exit-2',ticker:'KXETH15M-X',count:'3',yes_price:'0.95',no_price:'0.05',fee_cost:'0.01',created_at:'2026-08-07T13:08:00Z'}
];
const intents = [
  {order_id:'entry-1',trade_id:'t1',side:'up',role:'entry',reason:'entry'},
  {order_id:'entry-2',trade_id:'t1',side:'up',role:'entry',reason:'entry'},
  {order_id:'exit-1',trade_id:'t1',side:'up',role:'exit',reason:'stop_loss'},
  {order_id:'exit-2',trade_id:'t1',side:'up',role:'exit',reason:'take_profit'}
];
const trades = [{trade_id:'t1',ticker:'KXETH15M-X',side:'up',phase:'flat',exit_locked:1,target_budget:'5'}];
const grouped = sessions.groupActivity(fills, [], intents, {}, trades);
assert.equal(grouped[0].trades.length, 1);
const trade = grouped[0].trades[0];
assert.equal(trade.quantity, 5);
assert.equal(trade.exited_quantity, 5);
assert.equal(trade.remaining_quantity, 0);
assert.equal(trade.close_reason, 'mixed');
assert.equal(trade.entry_fills.length, 2);
assert.equal(trade.exit_fills.length, 2);
assert.equal(grouped[0].unmatched.length, 0);
assert.ok(Math.abs(trade.pnl - (-0.19)) < 0.000001);
"""
    result = subprocess.run(
        ["node", "-e", script], cwd=project, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr


def test_legacy_opposite_entry_closes_old_side_before_opening_residual() -> None:
    project = Path(__file__).parents[1]
    script = r"""
const assert = require('node:assert/strict');
const sessions = require('./src/kaishi_bot/static/live_sessions.js');
const ticker = 'KXETH15M-26AUG071500-00';
const fills = [
  {fill_id:'down-entry-fill',order_id:'down-entry',ticker,count:'5.82',yes_price:'0.31',no_price:'0.69',fee_cost:'0.0872',created_at:'2026-08-07T18:55:51Z'},
  {fill_id:'up-entry-fill',order_id:'up-entry',ticker,count:'6.09',yes_price:'0.81',no_price:'0.19',fee_cost:'0.0657',created_at:'2026-08-07T18:57:53Z'},
  {fill_id:'up-exit-fill',order_id:'up-exit',ticker,count:'0.27',yes_price:'0.96',no_price:'0.04',fee_cost:'0.0008',created_at:'2026-08-07T18:58:38Z'}
];
const intents = [
  {order_id:'down-entry',ticker,side:'down',role:'entry',reason:'entry',created_at:'2026-08-07T18:55:51Z'},
  {order_id:'failed-sl',ticker,side:'down',role:'exit',reason:'stop_loss',created_at:'2026-08-07T18:57:52Z'},
  {order_id:'up-entry',ticker,side:'up',role:'entry',reason:'entry',created_at:'2026-08-07T18:57:53Z'},
  {order_id:'up-exit',ticker,side:'up',role:'exit',reason:'take_profit',created_at:'2026-08-07T18:58:38Z'}
];
const grouped = sessions.groupActivity(fills, [], intents, {});
assert.equal(grouped.length, 1);
assert.equal(grouped[0].trades.length, 2);
assert.equal(grouped[0].closedCount, 2);
assert.equal(grouped[0].openCount, 0);
assert.equal(grouped[0].tpCount, 1);
assert.equal(grouped[0].slCount, 1);
const down = grouped[0].trades.find(x => x.side === 'down');
const up = grouped[0].trades.find(x => x.side === 'up');
assert.equal(down.quantity, 5.82);
assert.equal(down.exited_quantity, 5.82);
assert.equal(down.close_reason, 'stop_loss');
assert.ok(Math.abs(up.quantity - 0.27) < 0.000001);
assert.ok(Math.abs(up.exited_quantity - 0.27) < 0.000001);
assert.equal(up.close_reason, 'take_profit');
assert.ok(Math.abs(grouped[0].netPnl - (-3.0232)) < 0.000001);
"""
    result = subprocess.run(
        ["node", "-e", script], cwd=project, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
