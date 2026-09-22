(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.PriceBand = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  const order = ['stop_loss', 'entry_min', 'entry_price', 'take_profit'];
  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

  function normalize(input, changed) {
    const values = Object.fromEntries(order.map(key => [
      key, clamp(Math.round(Number(input[key]) || 1), 1, 99)
    ]));
    if (changed === 'stop_loss') {
      values.stop_loss = clamp(values.stop_loss, 1, values.entry_min - 1);
    } else if (changed === 'entry_min') {
      values.entry_min = clamp(values.entry_min, values.stop_loss + 1, values.entry_price - 1);
    } else if (changed === 'entry_price') {
      values.entry_price = clamp(values.entry_price, values.entry_min + 1, values.take_profit - 1);
    } else if (changed === 'take_profit') {
      values.take_profit = clamp(values.take_profit, values.entry_price + 1, 99);
    }
    return values;
  }

  function readForm(form) {
    return Object.fromEntries(order.map(key => [key, Number(form.elements[key].value) * 100]));
  }

  function paint(root, values) {
    order.forEach(key => {
      const slider = root.querySelector(`[data-price-field="${key}"]`);
      const output = root.querySelector(`[data-price-output="${key}"]`);
      slider.value = values[key];
      if (output) output.textContent = `${values[key]}¢`;
      root.style.setProperty(`--${key.replace('_', '-')}`, `${values[key]}%`);
    });
  }

  function syncFromForm(root, form) {
    paint(root, normalize(readForm(form)));
  }

  function bind(root, form) {
    if (root.dataset.bound === 'true') return;
    root.dataset.bound = 'true';
    root.querySelectorAll('[data-price-field]').forEach(slider => {
      slider.addEventListener('input', () => {
        const changed = slider.dataset.priceField;
        const raw = readForm(form);
        raw[changed] = Number(slider.value);
        const values = normalize(raw, changed);
        order.forEach(key => {
          form.elements[key].value = (values[key] / 100).toFixed(2);
        });
        paint(root, values);
      });
    });
  }

  return {normalize, bind, syncFromForm};
});
