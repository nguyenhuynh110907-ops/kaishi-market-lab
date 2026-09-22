(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.TimeWindow = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  const STEP = 15;
  const LIMIT = 900;
  const clamp = value => Math.max(0, Math.min(LIMIT, Math.round(Number(value) / STEP) * STEP));

  function normalize(input, changed) {
    let start = clamp(input.start);
    let end = clamp(input.end);
    if (changed === 'start') start = Math.min(start, end - STEP);
    if (changed === 'end') end = Math.max(end, start + STEP);
    if (start >= end) {
      end = Math.min(LIMIT, start + STEP);
      if (start >= end) start = end - STEP;
    }
    return {start, end};
  }

  function clock(seconds) {
    const value = clamp(seconds);
    return `${String(Math.floor(value / 60)).padStart(2, '0')}:${String(value % 60).padStart(2, '0')}`;
  }

  function readForm(form) {
    return {
      start: Number(form.elements.entry_start_seconds.value),
      end: Number(form.elements.entry_end_seconds.value),
    };
  }

  function paint(root, values) {
    root.style.setProperty('--entry-start', `${values.start / 9}%`);
    root.style.setProperty('--entry-end', `${values.end / 9}%`);
    root.querySelector('[data-time-field="start"]').value = values.start;
    root.querySelector('[data-time-field="end"]').value = values.end;
    root.querySelector('[data-time-output="start"]').textContent = clock(values.start);
    root.querySelector('[data-time-output="end"]').textContent = clock(values.end);
  }

  function syncFromForm(root, form) {
    paint(root, normalize(readForm(form)));
  }

  function bind(root, form) {
    if (root.dataset.bound === 'true') return;
    root.dataset.bound = 'true';
    root.querySelectorAll('[data-time-field]').forEach(slider => {
      slider.addEventListener('input', () => {
        const raw = readForm(form);
        raw[slider.dataset.timeField] = Number(slider.value);
        const values = normalize(raw, slider.dataset.timeField);
        form.elements.entry_start_seconds.value = values.start;
        form.elements.entry_end_seconds.value = values.end;
        paint(root, values);
      });
    });
  }

  return {normalize, clock, bind, syncFromForm};
});
