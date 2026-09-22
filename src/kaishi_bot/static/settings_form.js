(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.SettingsFormState = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  const numericFields = [
    'entry_min', 'entry_price', 'take_profit', 'stop_loss', 'entry_amount',
    'paper_daily_cap', 'daily_cap', 'entry_start_seconds', 'entry_end_seconds'
  ];

  function markDirty(form) {
    form.dataset.settingsDirty = 'true';
  }

  function markSaved(form) {
    delete form.dataset.settingsDirty;
  }

  function bindDirtyContainer(form, container) {
    if (container.dataset.settingsDirtyBound === 'true') return;
    container.dataset.settingsDirtyBound = 'true';
    container.addEventListener('change', event => {
      if (event.target && event.target.matches('input')) markDirty(form);
    });
  }

  function sync(form, settings, renderAssets) {
    if (form.dataset.settingsDirty === 'true') return false;
    numericFields.forEach(name => {
      form.elements[name].value = settings[name];
    });
    renderAssets();
    return true;
  }

  return {markDirty, markSaved, bindDirtyContainer, sync};
});
