(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.LiveSessions = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  const SLOT_MS = 15 * 60 * 1000;
  const TIME_ZONE = 'America/New_York';

  function assetFromTicker(ticker) {
    const match = String(ticker || '').match(/^KX(BTC|ETH|SOL|XRP|DOGE)15M/i);
    return match ? match[1].toUpperCase() : '—';
  }
  function timeLabel(date, includeSeconds = false) {
    return new Intl.DateTimeFormat('en-GB', {
      timeZone: TIME_ZONE, hour: '2-digit', minute: '2-digit',
      ...(includeSeconds ? {second: '2-digit'} : {}), hour12: false,
    }).format(date);
  }
  function dateLabel(date) {
    return new Intl.DateTimeFormat('vi-VN', {
      timeZone: TIME_ZONE, day: '2-digit', month: '2-digit', year: 'numeric',
    }).format(date);
  }
  function logicalPrice(fill, side) {
    return Number(side === 'down' ? fill.no_price : fill.yes_price);
  }
  function emptySession(ticker, timestamp) {
    const startMs = Math.floor(timestamp / SLOT_MS) * SLOT_MS;
    const start = new Date(startMs), end = new Date(startMs + SLOT_MS);
    return {
      key: ticker || start.toISOString(), ticker, startMs,
      label: `${timeLabel(start)}–${timeLabel(end)}`, date: dateLabel(start),
      trades: [], unmatched: [], settlements: [], netPnl: 0,
      closedCount: 0, openCount: 0, tpCount: 0, slCount: 0, isOpen: false,
    };
  }
  function sum(items, pick) { return items.reduce((total, item) => total + pick(item), 0); }
  function cleanNumber(value) { return Number(Number(value).toFixed(8)); }

  function groupActivity(fills, settlements, intents = [], settings = {}, tradeRows = []) {
    const intentByOrder = new Map((intents || []).filter(x => x.order_id).map(x => [String(x.order_id), x]));
    const tradeMeta = new Map((tradeRows || []).map(x => [String(x.trade_id), x]));
    const seen = new Set(), uniqueFills = [];
    (fills || []).forEach(fill => {
      const fillId = String(fill.fill_id || '');
      if (fillId && seen.has(fillId)) return;
      if (fillId) seen.add(fillId);
      uniqueFills.push(fill);
    });
    uniqueFills.sort((a, b) => new Date(a.created_at) - new Date(b.created_at));

    // Before trade_id existed, an opposite "entry" could net/close the old
    // Kalshi position first. Reconstruct that cash flow instead of showing two
    // independent open trades. New trade_id fills never need this fallback.
    const exitHistory = new Map();
    (intents || []).filter(x => x.role === 'exit').forEach(intent => {
      const key = `${intent.ticker}|${intent.side}`;
      if (!exitHistory.has(key)) exitHistory.set(key, []);
      exitHistory.get(key).push(intent);
    });
    exitHistory.forEach(rows => rows.sort((a, b) => new Date(a.created_at) - new Date(b.created_at)));
    function precedingExitReason(ticker, side, at) {
      const rows = exitHistory.get(`${ticker}|${side}`) || [];
      for (let index = rows.length - 1; index >= 0; index -= 1) {
        if (new Date(rows[index].created_at) <= at) return rows[index].reason || 'exit';
      }
      return 'netting';
    }
    function fillPart(fill, count, fee, suffix) {
      return {...fill, fill_id: `${fill.fill_id || fill.order_id}:${suffix}`, count: String(count), fee_cost: String(fee)};
    }
    const activeLegacy = new Map(), prepared = [], unmatchedByTicker = new Map();
    function unmatched(fill) {
      if (!unmatchedByTicker.has(fill.ticker)) unmatchedByTicker.set(fill.ticker, []);
      unmatchedByTicker.get(fill.ticker).push(fill);
    }
    uniqueFills.forEach(fill => {
      const intent = intentByOrder.get(String(fill.order_id));
      if (!intent) { unmatched(fill); return; }
      if (intent.trade_id) {
        prepared.push({fill, intent, tradeId: String(intent.trade_id)});
        return;
      }
      const ticker = String(fill.ticker), at = new Date(fill.created_at);
      const totalQty = Number(fill.count || 0), totalFee = Number(fill.fee_cost || 0);
      let quantity = totalQty, feeUsed = 0, current = activeLegacy.get(ticker) || null;
      if (intent.role === 'entry') {
        if (current && current.side !== intent.side && current.entry - current.exit > 0.000001) {
          const closeQty = Math.min(quantity, current.entry - current.exit);
          const closeFee = totalQty > 0 ? totalFee * closeQty / totalQty : 0;
          prepared.push({
            fill: fillPart(fill, closeQty, closeFee, 'net-close'),
            intent: {...intent, role: 'exit', side: current.side,
              reason: precedingExitReason(ticker, current.side, at)},
            tradeId: current.id,
          });
          current.exit += closeQty; quantity -= closeQty; feeUsed += closeFee;
          if (current.entry - current.exit <= 0.000001) {
            activeLegacy.delete(ticker); current = null;
          }
        }
        if (quantity > 0.000001) {
          current = activeLegacy.get(ticker) || null;
          if (!current || current.side !== intent.side) {
            current = {
              id: `legacy:${ticker}:${fill.fill_id || fill.order_id}`,
              side: intent.side, entry: 0, exit: 0,
            };
            activeLegacy.set(ticker, current);
          }
          prepared.push({
            fill: fillPart(fill, quantity, Math.max(0, totalFee - feeUsed), 'entry'),
            intent, tradeId: current.id,
          });
          current.entry += quantity;
        }
        return;
      }
      if (!current || current.side !== intent.side) { unmatched(fill); return; }
      const closeQty = Math.min(quantity, Math.max(0, current.entry - current.exit));
      if (closeQty <= 0.000001) { unmatched(fill); return; }
      const closeFee = totalQty > 0 ? totalFee * closeQty / totalQty : 0;
      prepared.push({fill: fillPart(fill, closeQty, closeFee, 'exit'), intent, tradeId: current.id});
      current.exit += closeQty;
      if (current.entry - current.exit <= 0.000001) activeLegacy.delete(ticker);
      if (quantity - closeQty > 0.000001) {
        unmatched(fillPart(fill, quantity - closeQty, totalFee - closeFee, 'excess'));
      }
    });

    const groupedFills = new Map();
    prepared.forEach(part => {
      if (!groupedFills.has(part.tradeId)) groupedFills.set(part.tradeId, []);
      groupedFills.get(part.tradeId).push({fill: part.fill, intent: part.intent});
    });

    const sessions = new Map();
    groupedFills.forEach((parts, tradeId) => {
      parts.sort((a, b) => new Date(a.fill.created_at) - new Date(b.fill.created_at));
      const meta = tradeMeta.get(tradeId) || {};
      const ticker = String(meta.ticker || parts[0].fill.ticker || '');
      const side = String(meta.side || parts[0].intent.side || '').toLowerCase();
      const entries = parts.filter(x => x.intent.role === 'entry');
      const exits = parts.filter(x => x.intent.role === 'exit');
      if (!entries.length) {
        if (!unmatchedByTicker.has(ticker)) unmatchedByTicker.set(ticker, []);
        unmatchedByTicker.get(ticker).push(...parts.map(x => x.fill));
        return;
      }
      const entryQty = cleanNumber(sum(entries, x => Number(x.fill.count || 0)));
      const exitQty = cleanNumber(sum(exits, x => Number(x.fill.count || 0)));
      const entryPremium = sum(entries, x => Number(x.fill.count || 0) * logicalPrice(x.fill, side));
      const exitGross = sum(exits, x => Number(x.fill.count || 0) * logicalPrice(x.fill, side));
      const entryFee = sum(entries, x => Number(x.fill.fee_cost || 0));
      const exitFee = sum(exits, x => Number(x.fill.fee_cost || 0));
      const entryOutlay = entryPremium + entryFee;
      const exitedRatio = entryQty > 0 ? Math.min(1, exitQty / entryQty) : 0;
      const allocatedBasis = entryOutlay * exitedRatio;
      const netProceeds = exitGross - exitFee;
      const remaining = cleanNumber(Math.max(0, entryQty - exitQty));
      const isOpen = String(meta.phase || '') !== 'flat' && remaining > 0.000001;
      const reasons = new Set(exits.map(x => String(x.intent.reason || 'exit')));
      const first = new Date(entries[0].fill.created_at);
      const last = exits.length ? new Date(exits[exits.length - 1].fill.created_at) : null;
      const trade = {
        trade_id: tradeId, ticker, asset: assetFromTicker(ticker), side,
        quantity: entryQty, entry_price: entryQty ? entryPremium / entryQty : 0,
        entry_fee: entryFee, entry_outlay: entryOutlay,
        exited_quantity: exitQty, remaining_quantity: remaining,
        exit_price: exitQty ? exitGross / exitQty : 0,
        exit_fee: exitFee, net_proceeds: netProceeds,
        pnl: netProceeds - allocatedBasis,
        entryTime: timeLabel(first, true), entry_at: entries[0].fill.created_at,
        exitTime: last ? timeLabel(last, true) : null,
        close_reason: reasons.size === 1 ? [...reasons][0] : reasons.size > 1 ? 'mixed' : null,
        isOpen, phase: String(meta.phase || (isOpen ? 'holding' : 'flat')),
        exit_locked: Boolean(Number(meta.exit_locked || 0)),
        target_budget: Number(meta.target_budget || entryOutlay),
        entry_fills: entries.map(x => ({...x.fill, price: logicalPrice(x.fill, side), time: timeLabel(new Date(x.fill.created_at), true)})),
        exit_fills: exits.map(x => ({...x.fill, price: logicalPrice(x.fill, side), reason: x.intent.reason, time: timeLabel(new Date(x.fill.created_at), true)})),
      };
      if (!sessions.has(ticker)) sessions.set(ticker, emptySession(ticker, first.getTime()));
      sessions.get(ticker).trades.push(trade);
    });

    unmatchedByTicker.forEach((raw, ticker) => {
      const valid = raw.filter(x => Number.isFinite(new Date(x.created_at).getTime()));
      if (!valid.length) return;
      if (!sessions.has(ticker)) sessions.set(ticker, emptySession(ticker, new Date(valid[0].created_at).getTime()));
      sessions.get(ticker).unmatched.push(...valid.map(fill => ({...fill, asset: assetFromTicker(ticker), time: timeLabel(new Date(fill.created_at), true)})));
    });
    (settlements || []).forEach(item => {
      const at = new Date(item.settled_at), key = String(item.ticker || '');
      if (!Number.isFinite(at.getTime())) return;
      if (!sessions.has(key)) sessions.set(key, emptySession(key, at.getTime() - SLOT_MS));
      sessions.get(key).settlements.push({
        ...item, asset: assetFromTicker(key), time: timeLabel(at, true),
        event: 'SETTLEMENT',
      });
    });
    sessions.forEach(session => {
      session.closedCount = session.trades.filter(x => !x.isOpen).length;
      session.openCount = session.trades.filter(x => x.isOpen).length;
      session.tpCount = session.trades.filter(x => x.exit_fills.some(f => f.reason === 'take_profit')).length;
      session.slCount = session.trades.filter(x => x.exit_fills.some(f => f.reason === 'stop_loss')).length;
      session.netPnl = sum(session.trades, x => Number(x.pnl || 0));
      session.isOpen = session.openCount > 0;
    });
    return [...sessions.values()].sort((a, b) => b.startMs - a.startMs);
  }
  return {assetFromTicker, groupActivity};
});
