(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.PaperSessions = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  const SLOT_MS = 15 * 60 * 1000;
  const TIME_ZONE = 'America/New_York';

  function safeNumber(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number : 0;
  }

  function timeLabel(date, includeSeconds = false) {
    return new Intl.DateTimeFormat('en-GB', {
      timeZone: TIME_ZONE,
      hour: '2-digit', minute: '2-digit',
      ...(includeSeconds ? {second: '2-digit'} : {}),
      hour12: false,
    }).format(date);
  }

  function dateLabel(date) {
    return new Intl.DateTimeFormat('vi-VN', {
      timeZone: TIME_ZONE, day: '2-digit', month: '2-digit', year: 'numeric',
    }).format(date);
  }

  function groupPositions(paper) {
    const all = [
      ...((paper && paper.closed_positions) || []),
      ...((paper && paper.positions) || []),
    ];
    const grouped = new Map();
    all.forEach(position => {
      const openedAt = new Date(position.opened_at);
      if (!Number.isFinite(openedAt.getTime())) return;
      const startMs = Math.floor(openedAt.getTime() / SLOT_MS) * SLOT_MS;
      const start = new Date(startMs);
      const end = new Date(startMs + SLOT_MS);
      const key = start.toISOString();
      if (!grouped.has(key)) {
        grouped.set(key, {
          key, startMs,
          label: `${timeLabel(start)}–${timeLabel(end)}`,
          date: dateLabel(start), trades: [], netPnl: 0,
          realizedPnl: 0, unrealizedPnl: 0,
          closedCount: 0, openCount: 0,
          takeProfitCount: 0, stopLossCount: 0,
        });
      }
      const session = grouped.get(key);
      const isOpen = position.status === 'open' || !position.closed_at;
      const pnl = safeNumber(isOpen ? position.unrealized_pnl : position.realized_pnl);
      session.trades.push({
        ...position, id: position.id, isOpen, pnl,
        entryTime: timeLabel(openedAt, true),
        exitTime: position.closed_at ? timeLabel(new Date(position.closed_at), true) : null,
      });
      session.netPnl += pnl;
      if (isOpen) {
        session.openCount += 1;
        session.unrealizedPnl += pnl;
      } else {
        session.closedCount += 1;
        session.realizedPnl += pnl;
        if (position.close_reason === 'take_profit') session.takeProfitCount += 1;
        if (position.close_reason === 'stop_loss') session.stopLossCount += 1;
      }
    });
    return [...grouped.values()]
      .map(session => ({
        ...session,
        trades: session.trades.sort((a, b) =>
          new Date(a.opened_at).getTime() - new Date(b.opened_at).getTime()),
      }))
      .sort((a, b) => b.startMs - a.startMs);
  }

  return {groupPositions};
});
