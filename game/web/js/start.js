// ---- start
(async () => {
  try {
    const bytes = Uint8Array.from(atob(WASM_B64), c => c.charCodeAt(0));
    E = (await WebAssembly.instantiate(bytes, {})).instance.exports;
    E.game_new_kind(STOREYS, KIND);
    $('title').textContent = ['A 12-storey concrete-frame tower at real scale', 'A 5-storey masonry building at real scale', 'A 6-storey mass-timber building at real scale', 'A 5-storey building of brick, steel, concrete and timber at real scale', 'A braced steel frame with a pitched timber roof and a masonry arcade, at real scale', `A city of ${STOREYS} × ${STOREYS} lots, a building of every kind, at real scale`][KIND];
    if (KIND === 5) setupCity();
    for (let k = 0; k < E.chunk_count(); k++) chunkMesh(k);
    if (!LABSTATE) await startWorker(bytes);   // (the lab runs the engine inline, its jobs delayed by whole frames)
    initSettings(); initDiag(); initPreset();
    // (in the city: in the street east of the first lot, facing it)
    if (KIND === 5) { const L0 = lots()[0]; E.player_set(L0.x + 20, 0.0, L0.z - 16); } else E.player_set(0, 0.0, -16);
  } catch (e) { $('warn').textContent = 'Could not start the engine: ' + e.message; return; }
  try {
    if (navigator.gpu) {
      const a = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (a) {
        try { const i = a.info || (a.requestAdapterInfo ? await a.requestAdapterInfo() : null); if (i) window.__gpuInfo = { vendor: i.vendor, architecture: i.architecture, device: i.device, description: i.description }; } catch (e) {}
        dev = await a.requestDevice({ requiredFeatures: a.features.has('timestamp-query') ? ['timestamp-query'] : [], requiredLimits: { maxStorageBufferBindingSize: a.limits.maxStorageBufferBindingSize, maxBufferSize: a.limits.maxBufferSize } });
        E.bricks_dump();
        const f = new Float32Array(E.memory.buffer, E.out_ptr(), E.out_len()).slice();
        const n = f[0], shapes = [], idx = [];
        for (let k = 0; k < n; k++) { const o = 1 + k * 8; idx.push(f[o]); shapes.push(boxShape(f, o + 1)); }
        gpu = await GpuWorld.create(dev, [{ fixed: true, shapes }], { colorCheck: !!window.__colCheck, substeps: 12, sleep: true, maxBodies: 30000, maxShapes: n + 120000, maxContacts: 160000, maxPairs: 640000, eventSpeed: 2.0, eventEnergy: 20000, maxParticles: MAXP, particleMaxRadius: PART_R + 0.05 });
        if (window.__colR) { const [x, y] = window.__colR.split(',').map(Number); gpu.colRoundsS = x; gpu.colRounds = y; }   // (debugging: the colour rounds)
        gpu.substeps = setting.substeps || 12; gpu.maxColors = setting.colors || 16; gpu.colorLimit = Math.min(gpu.colorLimit, gpu.maxColors);
        idx.forEach((b, k) => slots.set(b, k + 1));
        // (loose room contents, kind 4: the first structure step sends them off as bodies resting where they stand)
        if (E.loose_pending && E.loose_pending()) jobs.structure = true;
      }
    }
    if (!gpu) $('warn').textContent = 'WebGPU is not available in this browser: holes still work, but loose pieces are not simulated.';
  } catch (e) { gpu = null; $('warn').textContent = 'GPU physics unavailable: ' + e.message; }
  $('go').disabled = false; $('go').textContent = 'Enter';
  hud();
  // a benchmark or lab run asked for before the reload starts by itself
  if (LABSTATE && gpu) labRun().catch(e => { $('bench').hidden = false; $('bench-out').textContent = 'Lab failed: ' + e.message + '\n' + e.stack; window.LAB_ERROR = e.message; });
  else if (benchPending()) { $('start').hidden = true; benchStart(); }
})();
$('go').addEventListener('click', () => {
  $('start').hidden = true; physicsPaused = false;
  const r = cv.requestPointerLock && cv.requestPointerLock(); if (r && r.catch) r.catch(() => {});
});
requestAnimationFrame(frame);
