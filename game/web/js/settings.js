// ---- settings (engine parameters go to both the worker and the page's copy)
const SETTINGS = {
  crush: { el: 's-crush', def: 0.5, fmt: v => `${(+v).toFixed(2)} m`, apply: v => setParam(2, v) },
  realtime: { el: 's-realtime', def: 1, fmt: v => (+v ? 'on' : 'off'), apply: () => {} },
  adaptive: { el: 's-adaptive', def: 1, fmt: v => (+v ? 'on' : 'off'), apply: v => { if (gpu && !+v) gpu.substeps = setting.substeps || 12; } },
  restLoads: { el: 's-restLoads', def: 1, fmt: v => (+v ? 'on' : 'off'), apply: () => {} },
  // (the engineer's view: load / strength as colour, and the displacements drawn larger; key V)
  view: { el: 's-view', def: 0, fmt: v => (+v ? 'on' : 'off'), apply: () => applyView() },
  viewScale: { el: 's-viewScale', def: 1, fmt: v => (VIEW_SCALE[+v] ? `× ${VIEW_SCALE[+v]}` : 'not drawn'), apply: () => { if (+setting.view) applyView(); } },
  // (the standing structure stepped as a dynamic model, docs/DYNAMIC.md)
  dynamic: { el: 's-dynamic', def: window.__static ? 0 : 1, fmt: v => (+v ? 'on' : 'off'), apply: v => { setParam(7, +v); applyView(); } },
  // (chunks: loose bricks break into convex chunks, and the engine splits bricks on a fracture surface into
  // convex cells, at most 1,500 standing bricks)
  chunks: { el: 's-chunks', def: 1, fmt: v => (+v ? 'on' : 'off'), apply: v => setParam(3, +v && !window.__noFracCells ? 1500 : 0) },
  particles: { el: 's-particles', def: 1, fmt: v => (+v ? 'on' : 'off'), apply: v => setParam(5, +v) },   // (on: crushed material over the budgets falls as particles, not dust)
  crumbs: { el: 's-crumbs', def: 384, fmt: v => `${v}`, apply: v => setParam(0, v) },
  rubble: { el: 's-rubble', def: 64, fmt: v => `${v}`, apply: v => setParam(1, v) },
  substeps: { el: 's-substeps', def: 12, fmt: v => `${v}`, apply: v => { if (gpu) gpu.substeps = v; } },
  colors: { el: 's-colors', def: 16, fmt: v => `${v}`, apply: v => { if (gpu) { gpu.maxColors = v; gpu.colorLimit = Math.min(gpu.colorLimit, v); } } },
  // the most debris kept at once (falling pieces and rubble particles): capDebris
  debris: { el: 's-debris', def: 10000, fmt: v => (+v >= 20000 ? 'no limit' : `${v}`), apply: v => { debrisMax = +v >= 20000 ? Infinity : +v; } },
  far: { el: 's-far', def: 100, fmt: v => (+v > 500 ? 'off' : `${v} m`), apply: v => { farDist = +v > 500 ? Infinity : +v; } },
  // graphics (the Graphics window): shadows, the sun's shadow map, the most device pixels to a CSS pixel
  gShadows: { el: 's-gShadows', def: 1, fmt: v => (+v ? 'on' : 'off'), apply: v => { if (renderer.shadowMap.enabled !== !!+v) { renderer.shadowMap.enabled = !!+v; scene.traverse(o => { if (o.material) o.material.needsUpdate = true; }); } } },
  gMap: { el: 's-gMap', def: 1, fmt: v => `${SHADOW_MAPS[+v]} px`, apply: v => { const m = SHADOW_MAPS[+v]; if (sun.shadow.mapSize.x !== m) { sun.shadow.mapSize.set(m, m); if (sun.shadow.map) { sun.shadow.map.dispose(); sun.shadow.map = null; } } } },
  gPx: { el: 's-gPx', def: 2, fmt: v => `× ${v}`, apply: v => { renderer.setPixelRatio(Math.min(+v, window.devicePixelRatio || 1)); resize(); } },
  bspeed: { el: 's-bspeed', def: 60, fmt: v => `${v} m/s`, apply: () => {} },
  bsize: { el: 's-bsize', def: 0.5, fmt: v => `${(+v).toFixed(2)} m`, apply: () => {} },
};
const SHADOW_MAPS = [1024, 2048, 4096];
const setting = {};
const diag = { physics: true, readback: true, draw: true, events: true, timing: true, indirect: false };
function initDiag() {
  for (const k of Object.keys(diag)) {
    const el = $('d-' + k);
    el.addEventListener('change', () => {
      diag[k] = el.checked;
      if (k === 'draw') dMesh.visible = el.checked;
      if (k === 'events' && gpu) gpu.evEnergy = el.checked ? 20000 : 0;
      if (k === 'indirect' && gpu) gpu.indirect = el.checked;
      if (k === 'timing' && gpu) { if (!el.checked) { gpu.tsSaved = gpu.ts; gpu.ts = false; } else gpu.ts = gpu.tsSaved ?? gpu.ts; }
    });
  }
}
// long tasks the browser reports (anything on the main thread over 50 ms that is not our timed code)
const longTasks = { n: 0, ms: 0, last: '' };
try { new PerformanceObserver(l => { for (const e of l.getEntries()) { longTasks.n++; longTasks.ms += e.duration; longTasks.last = `${e.name} ${e.duration.toFixed(0)} ms`; } }).observe({ entryTypes: ['longtask'] }); } catch (e) {}
function setParam(id, v) { if (E) E.set_param(id, +v); if (worker) worker.postMessage({ op: 'param', id, v: +v }); }
function initSettings() {
  let saved = {}; try { saved = JSON.parse(localStorage.getItem('sf-settings') || '{}'); } catch (e) {}
  const ks = $('s-kind'); ks.value = String(KIND);
  const cs = $('s-city'); cs.value = String(CITY); setting.city = CITY;
  cs.addEventListener('change', () => { setting.city = +cs.value; try { localStorage.setItem('sf-settings', JSON.stringify(setting)); } catch (e) {} if (KIND === 5) location.reload(); });
  ks.addEventListener('change', () => { setting.kind = +ks.value; try { localStorage.setItem('sf-settings', JSON.stringify(setting)); } catch (e) {} location.reload(); });
  setting.kind = KIND;
  for (const [k, d] of Object.entries(SETTINGS)) {
    // (the view starts off: it is a look, not a setting)
    const el = $(d.el); el.value = k === 'view' ? 0 : (saved[k] ?? d.def); setting[k] = +el.value; $('v-' + k).textContent = d.fmt(el.value); d.apply(setting[k]);
    el.addEventListener('input', () => { setting[k] = +el.value; $('v-' + k).textContent = d.fmt(el.value); d.apply(setting[k]); try { localStorage.setItem('sf-settings', JSON.stringify(setting)); } catch (e) {} });
  }
}
// Two presets, each in its own window. Physics: how much is simulated. Graphics: how it is drawn. Each sets the
// sliders it owns (its window's advanced section shows them); a slider moved by hand makes its preset "custom".
const PRESETS = {
  low: { debris: 1500, crumbs: 64, rubble: 16, substeps: 6, colors: 10 },
  medium: { debris: 4000, crumbs: 192, rubble: 32, substeps: 8, colors: 12 },
  high: { debris: 10000, crumbs: 384, rubble: 64, substeps: 12, colors: 16 },
  ultra: { debris: 20000, crumbs: 1024, rubble: 192, substeps: 12, colors: 24 },
};
const PHYS_KEYS = ['debris', 'crumbs', 'rubble', 'substeps', 'colors'];
const GFX = {
  low: { gShadows: 0, gMap: 0, gPx: 1 },
  medium: { gShadows: 1, gMap: 0, gPx: 1.5 },
  high: { gShadows: 1, gMap: 1, gPx: 2 },
  ultra: { gShadows: 1, gMap: 2, gPx: 3 },
};
const GFX_KEYS = ['gShadows', 'gMap', 'gPx'];
function saveSettings() { try { localStorage.setItem('sf-settings', JSON.stringify(setting)); } catch (e) {} }
function applySet(table, keys, name, which) {
  const pr = table[name]; if (!pr) return;
  for (const k of keys) { const el = $(SETTINGS[k].el); el.value = pr[k]; el.dispatchEvent(new Event('input')); }
  setting[which] = name; saveSettings();
}
const applyPreset = (name) => applySet(PRESETS, PHYS_KEYS, name, 'phys');
const applyGfx = (name) => applySet(GFX, GFX_KEYS, name, 'gfx');
function initPreset() {
  let saved = {}; try { saved = JSON.parse(localStorage.getItem('sf-settings') || '{}'); } catch (e) {}
  // (settings saved before 14 Nov had one preset, "quality", for both)
  const ph = $('s-phys'), gf = $('s-gfx');
  const p0 = saved.phys !== undefined ? saved.phys : (saved.preset !== undefined ? saved.preset : 'high');
  const g0 = saved.gfx !== undefined ? saved.gfx : (saved.preset || 'high');
  ph.value = PRESETS[p0] ? p0 : ''; setting.phys = ph.value;
  gf.value = GFX[g0] ? g0 : ''; setting.gfx = gf.value;
  // (graphics saved as a preset before its parameters were sliders: applied once more, so they agree)
  if (gf.value && saved.gPx === undefined) applyGfx(gf.value);
  ph.addEventListener('change', () => { if (ph.value) applyPreset(ph.value); else { setting.phys = ''; saveSettings(); } });
  gf.addEventListener('change', () => { if (gf.value) applyGfx(gf.value); else { setting.gfx = ''; saveSettings(); } });
  // a slider moved by hand makes it a custom setting
  for (const k of PHYS_KEYS) $(SETTINGS[k].el).addEventListener('input', (e) => { if (e.isTrusted) { ph.value = ''; setting.phys = ''; saveSettings(); } });
  for (const k of GFX_KEYS) $(SETTINGS[k].el).addEventListener('input', (e) => { if (e.isTrusted) { gf.value = ''; setting.gfx = ''; saveSettings(); } });
}
// ---- the menu (Esc) and its two windows. One panel shows at a time; the simulation is paused while one is open.
const PANELS = ['menu', 'win-gfx', 'win-phys'];
let menuPaused = false;
const panelOpen = () => PANELS.find(id => !$(id).hidden) || null;
function showPanel(id) {
  for (const k of PANELS) $(k).hidden = k !== id;
  if (id) {
    if (document.exitPointerLock && document.pointerLockElement) document.exitPointerLock();
    if (!physicsPaused && !LAB.on) { physicsPaused = true; menuPaused = true; }
  } else if (menuPaused) { menuPaused = false; physicsPaused = false; lastActive = clock(); }
}
// (kept for the page checks and old callers: the menu on, off or toggled)
function toggleSettings(on) { showPanel((on === undefined ? !panelOpen() : on) ? 'menu' : null); }
// Esc: a window goes back to the menu, the menu closes, and from play the menu opens.
function escapeKey() { const p = panelOpen(); showPanel(p === 'menu' ? null : 'menu'); }
function resume() { showPanel(null); const r = cv.requestPointerLock && cv.requestPointerLock(); if (r && r.catch) r.catch(() => {}); }
// Reset the level: the page starts again on the same level and goes straight in (no start card). Everything the
// level holds is in the engine's worker, the GPU world and the page's copies; starting over is the one way that
// cannot leave any of it behind.
function resetLevel() { try { sessionStorage.setItem('sf-reset', '1'); } catch (e) {} location.reload(); }
function initMenu() {
  $('m-resume').addEventListener('click', resume);
  $('m-reset').addEventListener('click', resetLevel);
  $('m-gfx').addEventListener('click', () => showPanel('win-gfx'));
  $('m-phys').addEventListener('click', () => showPanel('win-phys'));
  document.querySelectorAll('.settings .back').forEach(b => b.addEventListener('click', () => showPanel('menu')));
}
