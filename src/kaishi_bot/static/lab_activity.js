(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.LabRunActivity = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  function createTracker(staleAfterMs = 5000) {
    let runId = null;
    let quoteCount = null;
    let lastActivityAt = null;

    function observe(run, nowMs = Date.now()) {
      const nextCount = Number(run.quote_count || 0);
      if (run.id !== runId) {
        runId = run.id;
        quoteCount = nextCount;
        lastActivityAt = null;
      } else if (nextCount > quoteCount) {
        quoteCount = nextCount;
        lastActivityAt = nowMs;
      }

      if (run.status !== 'running') {
        const labels = {stopped: 'ĐÃ DỪNG', completed: 'HOÀN TẤT'};
        return {
          kind: 'finished',
          label: labels[run.status] || 'KHÔNG HỢP LỆ',
          lastActivityAt,
          ageSeconds: null,
        };
      }

      const ageMs = lastActivityAt === null ? null : nowMs - lastActivityAt;
      const active = ageMs !== null && ageMs <= staleAfterMs;
      return {
        kind: active ? 'active' : 'waiting',
        label: active ? 'ĐANG CHẠY' : 'ĐANG CHỜ DỮ LIỆU / SETTLEMENT',
        lastActivityAt,
        ageSeconds: ageMs === null ? null : Math.max(0, Math.floor(ageMs / 1000)),
      };
    }

    return {observe};
  }

  return {createTracker};
});
