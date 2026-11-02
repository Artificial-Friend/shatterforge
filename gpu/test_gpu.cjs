// Headless correctness checks of the WebGPU solver (SwiftShader in CI; any WebGPU in a real browser).
// Usage: node gpu/test_gpu.cjs [scene...]   scenes: box stack column pile
const { chromium } = require(process.env.PW || 'playwright');
const fs = require('fs'), path = require('path');
const root = path.resolve(__dirname, '..');
const wasm = fs.readFileSync(path.join(root, 'engine/target/wasm32-unknown-unknown/release/sf_engine.wasm')).toString('base64');
const scenes = process.argv.slice(2).length ? process.argv.slice(2) : ['box', 'stack', 'column', 'pile'];
(async () => {
  const b = await chromium.launch({ args: ['--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--enable-features=Vulkan'] });
  const p = await b.newPage();
  p.on('console', m => console.log('  [page]', m.text()));
  p.on('pageerror', e => console.log('  [pageerror]', e.message));
  fs.writeFileSync('/tmp/sf_blank.html', '<html><body></body></html>');
  await p.goto('file:///tmp/sf_blank.html');
  await p.addScriptTag({ path: path.join(__dirname, 'gpu_world.js') });
  const res = await p.evaluate(async ({ wasm, scenes }) => {
    const bytes = Uint8Array.from(atob(wasm), c => c.charCodeAt(0));
    const E = (await WebAssembly.instantiate(bytes, {})).instance.exports;
    const a = await navigator.gpu.requestAdapter();
    const dev = await a.requestDevice({ requiredFeatures: a.features.has('timestamp-query') ? ['timestamp-query'] : [] });
    dev.addEventListener('uncapturederror', e => console.log('GPU error: ' + e.error.message));
    const grab = () => { const n = E.body_count(); return new Float32Array(E.memory.buffer, E.pose(), n * 10).slice(); };
    const out = {};
    const run = async (poses, frames, sub) => {
      const w = await GpuWorld.create(dev, poses, 1900, { substeps: sub });
      const t0 = performance.now();
      for (let f = 0; f < frames; f++) { w.step(1 / 60); if (f % 20 === 19) await dev.queue.onSubmittedWorkDone(); }
      await dev.queue.onSubmittedWorkDone();
      const ms = (performance.now() - t0) / frames;
      const r = await w.readPose(); r.stats = w.stats; r.ms = ms; r.gpu = w.gpuMs; w.destroy(); return r;
    };
    const speed = (v, i) => Math.hypot(v[i * 3], v[i * 3 + 1], v[i * 3 + 2]);
    if (scenes.includes('box')) {
      // one box dropped tilted
      const q = [0.32, 0.1, 0.07]; const s = Math.sin(0.35), qq = [q[0], q[1], q[2]].map(x => x / Math.hypot(...q) * s);
      const r = await run(new Float32Array([0, 1, 0, qq[0], qq[1], qq[2], Math.cos(0.35), 0.25, 0.125, 0.5]), 300, 12);
      out.box = { y: r.pose[1], speed: speed(r.vel, 0), stats: r.stats };
    }
    if (scenes.includes('stack')) {
      E.stack_setup(20, 12, 0);
      const p0 = grab(); const top0 = p0[19 * 10 + 1];
      const r = await run(p0, 600, 12);
      let mv = 0; for (let i = 0; i < 20; i++) mv = Math.max(mv, speed(r.vel, i));
      const t = r.pose.subarray(190, 193);
      out.stack = { drift_mm: Math.hypot(t[0], t[1] - top0, t[2]) * 1e3, dy_mm: (t[1] - top0) * 1e3, maxv_mms: mv * 1e3, stats: r.stats, ms: r.ms };
    }
    if (scenes.includes('column')) {
      const n = 10, f = new Float32Array(n * 10);
      for (let k = 0; k < n; k++) f.set([0.06 * k, 0.25 + 0.5 * k, 0, 0, 0, 0, 1, 0.25, 0.25, 0.25], k * 10);
      const r = await run(f, 600, 12);
      let maxy = 0, miny = 9, mv = 0; for (let k = 0; k < n; k++) { maxy = Math.max(maxy, r.pose[k * 10 + 1]); miny = Math.min(miny, r.pose[k * 10 + 1]); mv = Math.max(mv, speed(r.vel, k)); }
      out.column = { maxy, miny, maxv: mv };
    }
    if (scenes.includes('pile')) {
      E.pile_setup(2000, 4);
      const r = await run(grab(), 120, 4);
      let miny = 9, maxy = -9, mv = 0, nan = 0;
      for (let i = 0; i < 2000; i++) { const y = r.pose[i * 10 + 1]; if (!isFinite(y)) nan++; miny = Math.min(miny, y); maxy = Math.max(maxy, y); mv = Math.max(mv, speed(r.vel, i)); }
      // CPU reference after the same 120 frames
      E.pile_setup(2000, 4); for (let f = 0; f < 120; f++) E.step(1 / 60);
      const c = grab(); let cmin = 9, cmax = -9; for (let i = 0; i < 2000; i++) { cmin = Math.min(cmin, c[i * 10 + 1]); cmax = Math.max(cmax, c[i * 10 + 1]); }
      out.pile = { gpu: { miny, maxy, maxv: mv, nan, stats: r.stats, ms: r.ms, stages: r.gpu }, cpu: { miny: cmin, maxy: cmax, manifolds: E.manifolds() } };
    }
    return out;
  }, { wasm, scenes });
  console.log(JSON.stringify(res, null, 1));
  await b.close();
})();
