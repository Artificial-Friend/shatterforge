// Headless checks of the WebGPU engine (SwiftShader here; any WebGPU in a real browser).
// Usage: node gpu/test_gpu.cjs [scene...]
//   parity : box, stack, column, pile      (same scenes as the CPU reference engine, compared to it)
//   physics: bounce, incline, wall, bullet, beam, sleep
const { chromium } = require(process.env.PW || 'playwright');
const fs = require('fs'), path = require('path');
const root = path.resolve(__dirname, '..');
const wasm = fs.readFileSync(path.join(root, 'engine/target/wasm32-unknown-unknown/release/sf_engine.wasm')).toString('base64');
const ALL = ['box', 'stack', 'column', 'bounce', 'incline', 'wall', 'bullet', 'beam', 'plate', 'sleep', 'pile', 'events'];  // 'bouncedbg' traces a bounce
const scenes = process.argv.slice(2).length ? process.argv.slice(2) : ALL;
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
    const settle = () => new Promise(r => setTimeout(r, 40));
    const out = {};
    const speed = (v, i) => Math.hypot(v[i * 3], v[i * 3 + 1], v[i * 3 + 2]);
    const qa = (ax, ang) => { const s = Math.sin(ang / 2), l = Math.hypot(...ax); return [ax[0] / l * s, ax[1] / l * s, ax[2] / l * s, Math.cos(ang / 2)]; };
    const box = (p, h, mat, q = [0, 0, 0, 1], extra = {}) => ({ shapes: [{ p, q, h, mat }], ...extra });
    // run frames, sampling a callback every `every` frames on read-back state
    const run = async (list, frames, opt = {}, every = 0, cb = null) => {
      const w = await GpuWorld.create(dev, list, { substeps: 12, sleep: false, ...opt });
      const t0 = performance.now();
      for (let f = 1; f <= frames; f++) {
        w.step(1 / 60);
        if (every && f % every === 0) { await dev.queue.onSubmittedWorkDone(); await settle(); cb(await w.readPose(), f, w); }
        else if (f % 30 === 0) await dev.queue.onSubmittedWorkDone();
      }
      await dev.queue.onSubmittedWorkDone();
      const ms = (performance.now() - t0) / frames;
      await settle(); w.step(1 / 60); await dev.queue.onSubmittedWorkDone(); await settle();
      const r = await w.readPose(); r.stats = w.stats; r.ms = ms; r.gpu = w.gpuMs; r.world = w;
      return r;
    };
    const done = (r) => { r.world.destroy(); delete r.world; };
    const check = (name, ok, detail) => { out[name] = { ok, ...detail }; };

    if (scenes.includes('box')) {
      const r = await run([box([0, 1, 0], [0.25, 0.125, 0.5], MAT.reference, qa([0.32, 0.1, 0.07], 0.7))], 300);
      check('box', Math.abs(r.pose[1] - 0.125) < 0.005 && speed(r.vel, 0) < 0.01, { y: r.pose[1], speed: speed(r.vel, 0) }); done(r);
    }
    if (scenes.includes('stack')) {
      E.stack_setup(20, 12, 0);
      const p0 = grab(); const top0 = p0[191];
      const r = await run(GpuWorld.boxes(p0), 600);
      let mv = 0; for (let i = 0; i < 20; i++) mv = Math.max(mv, speed(r.vel, i));
      const drift = Math.hypot(r.pose[190], r.pose[191] - top0, r.pose[192]) * 1e3;
      check('stack', drift < 10 && mv < 0.01, { drift_mm: drift, maxv_mms: mv * 1e3, cpu: 'drift 6.93 mm, 1.7 mm/s' }); done(r);
    }
    if (scenes.includes('column')) {
      const list = []; for (let k = 0; k < 10; k++) list.push(box([0.06 * k, 0.25 + 0.5 * k, 0], [0.25, 0.25, 0.25], MAT.reference));
      const r = await run(list, 600);
      let maxy = 0, miny = 9; for (let k = 0; k < 10; k++) { maxy = Math.max(maxy, r.pose[k * 10 + 1]); miny = Math.min(miny, r.pose[k * 10 + 1]); }
      check('column', maxy < 0.6 && miny > 0.24, { maxy, miny, note: 'topples; with no two cubes left stacked' }); done(r);
    }
    if (scenes.includes('bouncedbg')) {
      const tr = [];
      const r = await run([box([0, 2.25, 0], [0.25, 0.25, 0.25], MAT['test-bouncy'])], 50, {}, 1, (s, f, w) => { if (f >= 34 && f <= 46) tr.push(`f${f} y ${(s.pose[1] - 0.25).toFixed(4)} vy ${s.vel[1].toFixed(3)} awake ${w.stats.manifolds}`); });
      done(r); check('bouncedbg', true, { tr });
    }
    if (scenes.includes('bounce')) {
      // drop from 2 m (bottom face 2 m above ground): rebound height should be e^2 x 2 m
      const res = {};
      for (const [name, e] of [['test-bouncy', 0.5], ['timber', 0.3], ['masonry', 0.05]]) {
        let peak = 0, landed = false, prevVy = -1;
        const r = await run([box([0, 2.25, 0], [0.25, 0.25, 0.25], MAT[name])], 150, {}, 1, (s, f) => {
          const y = s.pose[1] - 0.25, vy = s.vel[1];
          if (!landed && prevVy < 0 && vy > 0) landed = true;
          if (landed) peak = Math.max(peak, y);
          prevVy = vy;
        });
        done(r);
        const expect = e * e * 2;
        res[name] = { peak_m: +peak.toFixed(4), expected_m: +expect.toFixed(4) };
      }
      const okB = Math.abs(res['test-bouncy'].peak_m - 0.5) < 0.08 && Math.abs(res.timber.peak_m - 0.18) < 0.05 && res.masonry.peak_m < 0.03;
      check('bounce', okB, res);
    }
    if (scenes.includes('events')) {
      // a 1 m concrete cube dropped 2 m: one hard impact (v = 6.3 m/s, 1/2 J v ~ 50 kJ) reported once
      const evs = [];
      const r = await run([box([0, 2.5, 0], [0.5, 0.5, 0.5], MAT.concrete)], 60, { eventEnergy: 1000, eventSpeed: 1 }, 1, (s, f, w) => { for (const e of w.takeEvents()) evs.push({ f, ...e }); });
      done(r);
      const e = evs[0] || {};
      check('events', evs.length >= 1 && e.approach > 5 && e.approach < 7 && e.impulse > 5000, { count: evs.length, first: evs.length ? { f: e.f, approach: e.approach, impulse: e.impulse, n: e.n, p: e.p, a: e.a, b: e.b } : null });
    }
    if (scenes.includes('incline')) {
      // a box on a fixed slab tilted by theta; friction 0.6 holds below atan(0.6) = 31 deg
      const res = {};
      for (const deg of [25, 40]) {
        const th = deg * Math.PI / 180, q = qa([0, 0, 1], th);
        const n = [-Math.sin(th), Math.cos(th), 0], t = [Math.cos(th), Math.sin(th), 0];
        const slab = { shapes: [{ p: [0, 3, 0], q, h: [4, 0.25, 1], mat: MAT.reference }], fixed: true };
        const c = [0 + n[0] * 0.5, 3 + n[1] * 0.5, 0];  // box centre resting on the slab top
        let s0 = null, sAt1 = null;
        const r = await run([slab, box(c, [0.25, 0.25, 0.25], MAT.reference, q)], 120, {}, 30, (s, f) => {
          const along = (s.pose[10] - c[0]) * t[0] + (s.pose[11] - c[1]) * t[1];
          if (f === 30) s0 = along; if (f === 90) sAt1 = along;
        });
        done(r);
        // from 0.5 s to 1.5 s, sliding (downhill = negative t) covers v0 t + a t^2 / 2 with a = g (sin - mu cos)
        const acc = Math.max(0, 9.81 * (Math.sin(th) - 0.6 * Math.cos(th)));
        const v05 = acc * 0.5, expect = -(v05 * 1 + 0.5 * acc * 1);
        res[deg + 'deg'] = { moved_m: +(sAt1 - s0).toFixed(4), expected_m: +expect.toFixed(4) };
      }
      check('incline', Math.abs(res['25deg'].moved_m) < 0.01 && Math.abs(res['40deg'].moved_m - res['40deg'].expected_m) < 0.15 * Math.abs(res['40deg'].expected_m), res);
    }
    if (scenes.includes('wall')) {
      // a 2 x 2.5 m wall section of 0.5 x 0.25 x 0.25 m bricks (running bond) as ONE body, tipped 12 deg (past its balance point): it
      // topples as a piece and ends lying on its face
      const shapes = [];
      for (let row = 0; row < 10; row++) for (let k = 0; k < 4; k++) {
        const x = -0.75 + k * 0.5 + (row % 2 ? 0.25 : 0);
        if (row % 2 && k === 3) { shapes.push({ p: [-1.0 + 0.125, 0.125 + row * 0.25, 0], q: [0, 0, 0, 1], h: [0.125, 0.125, 0.125], mat: MAT.masonry }); shapes.push({ p: [0.875 + 0.0, 0.125 + row * 0.25, 0], q: [0, 0, 0, 1], h: [0.125, 0.125, 0.125], mat: MAT.masonry }); continue; }
        shapes.push({ p: [x, 0.125 + row * 0.25, 0], q: [0, 0, 0, 1], h: [0.25, 0.125, 0.125], mat: MAT.masonry });
      }
      // tip it about the bottom edge: rotate all shapes 5 deg about the x axis through (0, 0, -0.125)
      const tq = qa([1, 0, 0], -12 * Math.PI / 180), R = (v) => { const [x, y, z, w] = tq; const u = [x, y, z]; const c = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; const t = c(u, v).map(e => 2 * e); const c2 = c(u, t); return [v[0] + w * t[0] + c2[0], v[1] + w * t[1] + c2[1], v[2] + w * t[2] + c2[2]]; };
      for (const s of shapes) { const r0 = R([s.p[0], s.p[1], s.p[2] + 0.125]); s.p = [r0[0], r0[1] + 0.001, r0[2] - 0.125]; s.q = tq; }
      const r = await run([{ shapes }], 240);
      const pts = []; for (let k = 0; k < shapes.length; k++) pts.push([r.pose[k * 10], r.pose[k * 10 + 1], r.pose[k * 10 + 2]]);
      const maxY = Math.max(...pts.map(p => p[1])), minY = Math.min(...pts.map(p => p[1]));
      // lying flat on its face: all brick centres at half the wall thickness
      check('wall', maxY < 0.16 && minY > 0.10 && speed(r.vel, 0) < 0.05, { shapes: shapes.length, body_count: r.vel.length / 3, centre_y_range: [+minY.toFixed(3), +maxY.toFixed(3)], speed: speed(r.vel, 0) });
      done(r);
    }
    if (scenes.includes('bullet')) {
      // a brick at 80 m/s at a fixed 0.125 m thick wall 3 m away must not pass through (speculative contacts)
      const wall = { shapes: [{ p: [3, 1, 0], q: [0, 0, 0, 1], h: [0.0625, 1, 1], mat: MAT.concrete }], fixed: true };
      let maxX = -9;
      const r = await run([wall, box([0, 1, 0], [0.125, 0.0625, 0.0625], MAT.steel, [0, 0, 0, 1], { v: [80, 0, 0] })], 40, {}, 1, (s) => { maxX = Math.max(maxX, s.pose[10]); });
      check('bullet', maxX < 3 - 0.0625, { max_x: +maxX.toFixed(4), wall_face_x: 3 - 0.0625, final_vx: r.vel[3] }); done(r);
    }
    if (scenes.includes('beam')) {
      // an 8 m steel beam (a coarse-grid shape) dropped across three small blocks: it must land on them
      const list = [box([-3, 0.25, 0], [0.25, 0.25, 0.25], MAT.reference), box([0, 0.25, 0], [0.25, 0.25, 0.25], MAT.reference), box([3, 0.25, 0], [0.25, 0.25, 0.25], MAT.reference),
                    box([0, 1.5, 0], [4, 0.15, 0.15], MAT.steel)];
      const r = await run(list, 180);
      check('beam', r.pose[31] > 0.6 && r.pose[31] < 0.7, { beam_y: r.pose[31], expected: 0.65 }); done(r);
    }
    if (scenes.includes('plate')) {
      // one body of 12 x 12 bricks (a slab piece) dropped flat on 144 loose blocks: 144 contacts on one body,
      // far more than the colours, so most are solved as overflow with mass splitting
      const shapes = [];
      for (let i = 0; i < 12; i++) for (let k = 0; k < 12; k++) shapes.push({ p: [i * 0.5 - 2.75, 1.0, k * 0.5 - 2.75], q: [0, 0, 0, 1], h: [0.25, 0.125, 0.25], mat: MAT.concrete });
      const list = [{ shapes }];
      for (let i = 0; i < 12; i++) for (let k = 0; k < 12; k++) list.push(box([i * 0.5 - 2.75, 0.25, k * 0.5 - 2.75], [0.2, 0.25, 0.2], MAT.masonry));
      const r = await run(list, 240);
      let ys = []; for (let k = 0; k < 144; k++) ys.push(r.pose[k * 10 + 1]);
      const lo = Math.min(...ys), hi = Math.max(...ys);
      // resting on the blocks: block top 0.5 m + half the plate 0.125
      check('plate', lo > 0.6 && hi < 0.65 && speed(r.vel, 0) < 0.02, { plate_y: [+lo.toFixed(4), +hi.toFixed(4)], expected: 0.625, speed: speed(r.vel, 0), stats: r.stats });
      done(r);
    }
    if (scenes.includes('sleep')) {
      E.pile_setup(300, 12);
      const w = await GpuWorld.create(dev, GpuWorld.boxes(grab()), { substeps: 12, sleep: true });
      const awake = [];
      for (let f = 1; f <= 600; f++) { w.step(1 / 60); if (f % 60 === 0) { await dev.queue.onSubmittedWorkDone(); await settle(); awake.push(w.stats.awake); } }
      w.destroy();
      check('sleep', awake[awake.length - 1] < 300, { awake_per_s: awake });
    }
    if (scenes.includes('pile')) {
      E.pile_setup(2000, 4);
      const r = await run(GpuWorld.boxes(grab()), 120, { substeps: 4 });
      let miny = 9, maxy = -9, nan = 0;
      for (let i = 0; i < 2000; i++) { const y = r.pose[i * 10 + 1]; if (!isFinite(y)) nan++; miny = Math.min(miny, y); maxy = Math.max(maxy, y); }
      E.pile_setup(2000, 4); for (let f = 0; f < 120; f++) E.step(1 / 60);
      const c = grab(); let cmin = 9, cmax = -9; for (let i = 0; i < 2000; i++) { cmin = Math.min(cmin, c[i * 10 + 1]); cmax = Math.max(cmax, c[i * 10 + 1]); }
      check('pile', nan === 0 && Math.abs(miny - cmin) < 0.02 && Math.abs(maxy - cmax) < 0.1, { gpu: { miny, maxy, stats: r.stats, ms: r.ms, stages: r.gpu }, cpu: { miny: cmin, maxy: cmax, manifolds: E.manifolds() } });
      done(r);
    }
    return out;
  }, { wasm, scenes });
  let fail = 0;
  for (const [k, v] of Object.entries(res)) { console.log((v.ok ? 'PASS ' : 'FAIL ') + k + ' ' + JSON.stringify(v)); if (!v.ok) fail++; }
  await b.close();
  process.exit(fail ? 1 : 0);
})();
