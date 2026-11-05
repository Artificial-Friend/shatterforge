// Headless checks of the WebGPU engine (SwiftShader here; any WebGPU in a real browser).
// Usage: node gpu/test_gpu.cjs [scene...]
//   parity : box, stack, column, pile      (same scenes as the CPU reference engine, compared to it)
//   physics: bounce, incline, wall, bullet, beam, sleep
const { chromium } = require(process.env.PW || 'playwright');
const fs = require('fs'), path = require('path');
const root = path.resolve(__dirname, '..');
const wasm = fs.readFileSync(path.join(root, 'engine/target/wasm32-unknown-unknown/release/sf_engine.wasm')).toString('base64');
const ALL = ['box', 'stack', 'column', 'bounce', 'incline', 'wall', 'bullet', 'beam', 'plate', 'sleep', 'pile', 'events', 'paircount', 'reuse', 'inherit', 'particles', 'rubble', 'sandwich', 'compound', 'crush', 'stumps', 'wakeup', 'hulls', 'hullpairs', 'hullthrow'];  // 'bouncedbg' traces a bounce
const scenes = process.argv.slice(2).length ? process.argv.slice(2) : ALL;
(async () => {
  const b = await chromium.launch({ args: ['--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--enable-features=Vulkan'] });
  const p = await b.newPage();
  p.on('console', m => console.log('  [page]', m.text()));
  p.on('pageerror', e => console.log('  [pageerror]', e.message));
  fs.writeFileSync('/tmp/sf_blank.html', '<html><body></body></html>');
  await p.goto('file:///tmp/sf_blank.html');
  await p.addScriptTag({ path: path.join(__dirname, 'gpu_world.js') });
  const res = await p.evaluate(async ({ wasm, scenes, SUB, SAND_PUSH, SAND_MAT, SAND_SUB, SAND_COL, SAND_FALL, COMP_N, COMP_FLOOR, RELAX, HNOSLEEP, HBOX, HNOCRUSH, HNORED, HONLY, HPDYN, HRANDOM, HBIG, KN }) => {
    Object.assign(globalThis, KN || {}); if (globalThis.DBG) { const c0 = GpuWorld.create; GpuWorld.create = async (...a) => { const w = await c0(...a); w.dbgBits = +globalThis.DBG; return w; }; }
    if (SAND_PUSH) globalThis.SAND_PUSH = SAND_PUSH; if (SAND_MAT) globalThis.SAND_MAT = SAND_MAT; if (SAND_SUB) globalThis.SAND_SUB = SAND_SUB; if (SAND_COL) globalThis.SAND_COL = SAND_COL; if (SAND_FALL) globalThis.SAND_FALL = 1; if (COMP_N) globalThis.COMP_N = COMP_N; if (COMP_FLOOR) globalThis.COMP_FLOOR = 1; if (RELAX) globalThis.RELAX = RELAX; if (HNOSLEEP) globalThis.HNOSLEEP = 1; if (HBOX) globalThis.HBOX = 1; if (HNOCRUSH) globalThis.HNOCRUSH = 1; if (HNORED) globalThis.HNORED = 1; if (HONLY) globalThis.HONLY = HONLY; if (HPDYN) globalThis.HPDYN = 1; if (HRANDOM) globalThis.HRANDOM = 1; if (HBIG) globalThis.HBIG = 1;
    // SF_SUB=n forces n sub-steps everywhere (to see what fewer cost in accuracy)
    if (SUB) { const c0 = GpuWorld.create; GpuWorld.create = (d, l, o = {}) => c0.call(GpuWorld, d, l, { ...o, substeps: SUB }); }
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
      check('plate', lo > 0.6 && hi < 0.65 && speed(r.vel, 0) < 0.02, { plate_y: [+lo.toFixed(4), +hi.toFixed(4)], expected: 0.625, speed: speed(r.vel, 0), stages: r.gpu, stats: r.stats });
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
    if (scenes.includes('inherit')) {
      // a tumbling falling body is broken from a read-back 10 steps old: its two halves must appear where the body
      // is now, moving as it moves (not where it was)
      const w = await GpuWorld.create(dev, [{ shapes: [{ p: [-0.5, 20, 0], q: [0, 0, 0, 1], h: [0.5, 0.25, 0.25], mat: 0 }, { p: [0.5, 20, 0], q: [0, 0, 0, 1], h: [0.5, 0.25, 0.25], mat: 0 }], v: [3, 0, 1], w: [0, 0, 2] }], { substeps: 12, sleep: false, maxBodies: 8, maxShapes: 16, maxParticles: 16 });
      for (let f = 0; f < 10; f++) w.step(1 / 60);
      await dev.queue.onSubmittedWorkDone(); await settle();
      const r0 = await w.readPose();
      const p0 = Array.from(r0.body.subarray(0, 3)), q0 = Array.from(r0.body.subarray(3, 7));
      const halves = [0, 1].map(k => ({ shapes: [{ p: Array.from(r0.pose.subarray(k * 10, k * 10 + 3)), q: Array.from(r0.pose.subarray(k * 10 + 3, k * 10 + 7)), h: [0.5, 0.25, 0.25], mat: 0 }], v: [0, 0, 0], w: [0, 0, 0] }));
      for (let f = 0; f < 10; f++) w.step(1 / 60);
      await dev.queue.onSubmittedWorkDone(); await settle();
      const rNow = await w.readPose();
      const truth = [0, 1].map(k => Array.from(rNow.pose.subarray(k * 10, k * 10 + 3)));
      const vTruth = Array.from(rNow.vel.subarray(0, 3));
      const kids = w.addBodies(halves);
      // the second half: no motion away from a contact pushing down (n = -y): its downward velocity goes
      // and a crumb of the first half as a particle, made at the old pose too
      // (1 m above the body in its frame then, clear of its pieces)
      const qm = (a, b) => [a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1], a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0], a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3], a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]];
      const qr_ = (q, v) => { const r = qm(qm(q, [...v, 0]), [-q[0], -q[1], -q[2], q[3]]); return r.slice(0, 3); };
      const up = qr_(q0, [0, 1, 0]), xp = [p0[0] + up[0], p0[1] + up[1], p0[2] + up[2]];
      const [ps] = w.addParticles([{ p: xp, v: [0, 0, 0], r: 0.1, mat: 0, wait: true }]);
      const qNow = Array.from(rNow.body.subarray(3, 7)), pNow = Array.from(rNow.body.subarray(0, 3));
      const upNow = qr_(qNow, [0, 1, 0]), pTruth = [pNow[0] + upNow[0], pNow[1] + upNow[1], pNow[2] + upNow[2]];
      const wq = Array.from(rNow.ang.subarray(0, 3)), pv = [vTruth[0] + wq[1] * upNow[2] - wq[2] * upNow[1], vTruth[1] + wq[2] * upNow[0] - wq[0] * upNow[2], vTruth[2] + wq[0] * upNow[1] - wq[1] * upNow[0]];
      w.breakBody(1, kids, p0, q0, [[0, 0, 0], { noRebound: [0, -1, 0] }], [ps], [[0, 0, 0]]);
      w.step(1 / 60); await dev.queue.onSubmittedWorkDone(); await settle();
      const r1 = await w.readPose(2);
      // one more step happened: compare with the truth moved by that step (about v dt)
      let err = 0; for (let a = 0; a < 3; a++) err = Math.max(err, Math.abs(r1.pose[a] - truth[0][a] - vTruth[a] / 60));
      const vk = Array.from(r1.vel.subarray((kids[0] - 1) * 3, (kids[0] - 1) * 3 + 3));
      // the half's velocity is the parent's at its place: v + w x r (and one more step of gravity)
      const wp = Array.from(rNow.ang.subarray(0, 3)), pc = Array.from(rNow.body.subarray(0, 3)), rr = [0, 1, 2].map(a => truth[0][a] - pc[a]);
      const vExp = [vTruth[0] + wp[1] * rr[2] - wp[2] * rr[1], vTruth[1] + wp[2] * rr[0] - wp[0] * rr[2] - 9.81 / 60, vTruth[2] + wp[0] * rr[1] - wp[1] * rr[0]];
      const verr = Math.max(...[0, 1, 2].map(a => Math.abs(vk[a] - vExp[a])));
      const v2 = Array.from(r1.vel.subarray((kids[1] - 1) * 3, (kids[1] - 1) * 3 + 3));
      const rp = await w.readParticles(); let perr = 0; for (let a = 0; a < 3; a++) perr = Math.max(perr, Math.abs(rp.data[ps * 12 + a] - pTruth[a] - pv[a] / 60));
      w.destroy();
      // (one step of gravity after the placing: about -0.16 m/s)
      check('inherit', err < 0.05 && verr < 0.2 && v2[1] > -0.3 && Math.abs(v2[0]) > 1 && perr < 0.05, { particleError_m: perr, maxError_m: err, velocityError: verr, vKid: vk, vExpected: vExp, vNoRebound: v2, kids });
    }
    if (scenes.includes('reuse')) {
      // bodies retired and new ones made in their slots while everything is in contact (as when pieces break)
      E.pile_setup(400, 12);
      const w = await GpuWorld.create(dev, GpuWorld.boxes(grab()), { substeps: 12, sleep: true, maxBodies: 2000, maxShapes: 4000 });
      let rnd = 7; const rand = () => { rnd = (rnd * 1103515245 + 12345) & 0x7fffffff; return rnd / 0x7fffffff; };
      let firstBad = -1, kills = 0;
      for (let f = 1; f <= 240; f++) {
        w.step(1 / 60);
        if (f % 6 === 0) {
          await dev.queue.onSubmittedWorkDone(); await settle();
          const r = await w.readPose();
          if (firstBad < 0) for (let k = 0; k < r.body.length; k++) if (!Number.isFinite(r.body[k])) { firstBad = f; break; }
          // retire 10 live bodies and put 10 two-box bodies where they were (moving), as a break would
          const live = []; for (let i = 1; i <= w.n; i++) if (w.bodies[i - 1] && !w.bodies[i - 1].dead) live.push(i);
          const list = [];
          for (let q = 0; q < 10 && live.length; q++) {
            const i = live.splice(Math.floor(rand() * live.length), 1)[0];
            const o = (i - 1) * 7, p = [r.body[o], r.body[o + 1], r.body[o + 2]];
            if (!p.every(Number.isFinite)) continue;
            w.killBody(i); kills++;
            list.push({ shapes: [{ p: [p[0] - 0.13, p[1], p[2]], q: [0, 0, 0, 1], h: [0.12, 0.12, 0.12], mat: 0 }, { p: [p[0] + 0.13, p[1], p[2]], q: [0, 0, 0, 1], h: [0.12, 0.12, 0.12], mat: 0 }], v: [0, -3, 0], w: [0, 0, 0] });
          }
          if (list.length) w.addBodies(list);
        }
      }
      await dev.queue.onSubmittedWorkDone(); await settle();
      const r = await w.readPose(); let bad = 0; for (let k = 0; k < r.body.length; k++) if (!Number.isFinite(r.body[k])) bad++;
      w.destroy();
      check('reuse', bad === 0 && firstBad < 0, { kills, bad, firstBad, bodies: w.n });
    }
    if (scenes.includes('paircount')) {
      // the broad phase must find the same pairs whatever its search order: count them on the first frames
      E.pile_setup(2000, 4);
      const w = await GpuWorld.create(dev, GpuWorld.boxes(grab()), { substeps: 4, sleep: false });
      const counts = [];
      for (let f = 1; f <= 3; f++) { w.step(1 / 60); await dev.queue.onSubmittedWorkDone(); await settle(); await settle(); counts.push(w.stats.candidates); }
      w.destroy();
      check('paircount', true, { candidates: counts });
    }
    if (scenes.includes('sandwich')) {
      // a heavy block (a building's upper part, ~1,900 t) lands at 5.7 m/s on light rubble pieces (~5 t each)
      // lying on the ground: it must not rebound (concrete restitution 0.1: at most 0.6 m/s) nor throw them up
      const pieces = [];
      // (SAND_FALL: the pieces fall with the block, just under it, as pieces cut from its base do)
      const fall = !!globalThis.SAND_FALL, y0 = fall ? 0.65 : 0.6, v0 = fall ? -5.7 : 0;
      for (let i = 0; i < 6; i++) for (let k = 0; k < 4; k++) pieces.push({ shapes: [{ p: [-7.5 + i * 3, y0, -6 + k * 4], q: [0, 0, 0, 1], h: [1.0, 0.6, 1.0], mat: globalThis.SAND_MAT != null ? +globalThis.SAND_MAT : MAT.concrete }], v: [0, v0, 0], w: [0, 0, 0] });
      const SM = globalThis.SAND_MAT != null ? +globalThis.SAND_MAT : MAT.concrete;
      const block = { shapes: [{ p: [0, y0 + 0.6 + 0.001 + 4, 0], q: [0, 0, 0, 1], h: [10, 4, 10], mat: SM }], v: [0, -5.7, 0], w: [0, 0, 0] };
      // (the block weighs 2400 kg/m^3 x 3200 m^3 = 7,700 t; scaled to the case seen, 1,900 t, by its density)
      const w = await GpuWorld.create(dev, [...pieces, block], { substeps: +(globalThis.SAND_SUB || 12), sleep: false });
      if (globalThis.SAND_PUSH) w.maxPush = +globalThis.SAND_PUSH;
      w.crushing = false;   // (this tests the solver's mass ratio limit; crushing is the crush test)
      if (globalThis.SAND_COL) { w.maxColors = +globalThis.SAND_COL; w.colorLimit = +globalThis.SAND_COL; }
      const statsAt = [];
      const bi = w.n;   // the block is the last body
      let vmax = -1e9, pmax = -1e9, yBlock = [], hit = -1;
      for (let f = 1; f <= 90; f++) {
        w.step(1 / 60);
        await dev.queue.onSubmittedWorkDone(); await settle();
        const r = await w.readPose();
        const vy = r.vel[(bi - 1) * 3 + 1];
        if (hit < 0 && vy > -5) hit = f;
        if (hit > 0) vmax = Math.max(vmax, vy);
        for (let b = 1; b < bi; b++) pmax = Math.max(pmax, r.vel[(b - 1) * 3 + 1]);
        if (f % 5 === 0) yBlock.push(+r.body[(bi - 1) * 7 + 1].toFixed(3));
        if (f === 3) statsAt.push({ ...w.stats, colorLimit: w.colorLimit });
      }
      w.destroy();
      check('sandwich', vmax < 0.6 && pmax < 2, { hitAt: hit, blockMaxUp_ms: +vmax.toFixed(2), piecesMaxUp_ms: +pmax.toFixed(2), blockY: yBlock, stats: statsAt });
    }
    if (scenes.includes('hullthrow')) {
      // chunks thrown at the ground at 4-14 m/s with spin up to 15 rad/s: none passes into it (corner > 5 cm deep,
      // checked every 2nd frame)
      let rnd = 21; const rand = () => { rnd = (rnd * 1103515245 + 12345) & 0x7fffffff; return rnd / 0x7fffffff; };
      const H = GpuWorld.hulls, nh = H.hulls.length, list = [], ids = [];
      for (let k = 0; k < 200; k++) { const id0 = 1 + Math.floor(rand() * (nh - 1)), id = globalThis.HBOX ? 0 : id0; ids.push(id);
        list.push({ shapes: [{ p: [(k % 20) * 2 - 20, 1.5 + rand(), Math.floor(k / 20) * 2 - 10], q: qa([rand() - 0.5, rand() - 0.5, rand() - 0.5], rand() * 6), h: H.hulls[id0].box.h.map(v => v * 0.25 + 0.04), hull: id, mat: MAT.concrete }],
          v: [(rand() - 0.5) * 6, -4 - rand() * 10, (rand() - 0.5) * 6], w: [0, 1, 2].map(() => (rand() - 0.5) * 30) }); }
      const only = globalThis.HONLY != null ? +globalThis.HONLY : -1;
      const w = await GpuWorld.create(dev, only >= 0 ? [list[only]] : list, { substeps: 12, sleep: false }); if (only >= 0) { ids.splice(0, ids.length, ids[only]); }
      const corner = (pose, k, id) => { const o = k * 10, q = [pose[o + 3], pose[o + 4], pose[o + 5], pose[o + 6]], h = [pose[o + 7], pose[o + 8], pose[o + 9]];
        let m = 1e9; for (const v of H.hulls[id].V) { const x = [v[0] * h[0], v[1] * h[1], v[2] * h[2]];
          const y = 2 * (q[0] * q[1] + q[3] * q[2]) * x[0] + (1 - 2 * (q[0] * q[0] + q[2] * q[2])) * x[1] + 2 * (q[1] * q[2] - q[3] * q[0]) * x[2]; m = Math.min(m, pose[o + 1] + y); } return m; };
      let worst = 0, first = null; const tr = [];
      for (let f = 1; f <= 90; f++) { w.step(1 / 60);
        if (f % 2 === 0 || only >= 0) { await dev.queue.onSubmittedWorkDone(); await settle(); const r = await w.readPose();
          if (only >= 0 && f <= 16) tr.push([f, +corner(r.pose, 0, ids[0]).toFixed(3), +r.vel[1].toFixed(2), w.stats.manifolds]);
          for (let k = 0; k < ids.length; k++) { const c = corner(r.pose, k, ids[k]); if (c < worst) { worst = c; if (c < -0.05 && !first) first = { f, k, id: ids[k], corner: +c.toFixed(3), v: Array.from(r.vel.subarray(k * 3, k * 3 + 3)).map(x => +x.toFixed(2)), wv: Array.from(r.ang.subarray(k * 3, k * 3 + 3)).map(x => +x.toFixed(2)) }; } } } }
      w.destroy();
      check('hullthrow', worst > -0.05, { deepest_m: +worst.toFixed(3), first, trace: tr.length ? tr : undefined });
    }
    if (scenes.includes('hullpairs')) {
      // chunk on chunk: 200 random pairs, a fixed chunk and one dropped on it from 5 cm above; after 2 s no corner
      // of the dropped one is more than 3 cm inside the fixed one (it rests on it or slid off)
      let rnd = 9; const rand = () => { rnd = (rnd * 1103515245 + 12345) & 0x7fffffff; return rnd / 0x7fffffff; };
      const H = GpuWorld.hulls, nh = H.hulls.length;
      const q2m_ = (q) => { const [x, y, z, w] = q; return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]; };
      const verts = (id, p, q, h) => { const R = q2m_(q); return H.hulls[id].V.map(v => { const l = [v[0] * h[0], v[1] * h[1], v[2] * h[2]]; return [0, 1, 2].map(i => p[i] + R[i][0] * l[0] + R[i][1] * l[1] + R[i][2] * l[2]); }); };
      const planes = (id, p, q, h) => { const R = q2m_(q); return H.hulls[id].faces.map(f => { const nl = [f.n[0] / h[0], f.n[1] / h[1], f.n[2] / h[2]]; let n = [0, 1, 2].map(i => R[i][0] * nl[0] + R[i][1] * nl[1] + R[i][2] * nl[2]); const L = Math.hypot(...n); n = n.map(x => x / L); return { n, d: f.d / L + n[0] * p[0] + n[1] * p[1] + n[2] * p[2] }; }); };
      const list = [], meta = [];
      for (let k = 0; k < 200; k++) {
        const x = (k % 20) * 3 - 30, z = Math.floor(k / 20) * 3 - 15;
        const ia = Math.floor(rand() * nh), ib = Math.floor(rand() * nh);
        const qa_ = qa([rand() - 0.5, rand() - 0.5, rand() - 0.5], rand() * 6), qb_ = qa([rand() - 0.5, rand() - 0.5, rand() - 0.5], rand() * 6);
        const ha = H.hulls[ia].box.h.map(v => v * 0.4 + 0.1), hb = H.hulls[ib].box.h.map(v => v * 0.25 + 0.05);
        const pa = [x, globalThis.HPDYN ? 0.02 - Math.min(...verts(ia, [0, 0, 0], qa_, ha).map(v => v[1])) : 1.5, z];
        // B just above A's top
        const topA = Math.max(...verts(ia, pa, qa_, ha).map(v => v[1])), vb0 = verts(ib, [0, 0, 0], qb_, hb), lowB = Math.min(...vb0.map(v => v[1]));
        const pb = [x + (rand() - 0.5) * 0.2, topA - lowB + 0.05, z + (rand() - 0.5) * 0.2];
        list.push(globalThis.HPDYN ? { shapes: [{ p: pa, q: qa_, h: ha, hull: ia, mat: MAT.concrete }], v: [0, 0, 0], w: [0, 0, 0] } : { shapes: [{ p: pa, q: qa_, h: ha, hull: ia, mat: MAT.concrete }], fixed: true });
        list.push({ shapes: [{ p: pb, q: qb_, h: hb, hull: ib, mat: MAT.concrete }], v: [0, 0, 0], w: [0, 0, 0] });
        meta.push({ ia, ib, pa, qa: qa_, ha, hb });
      }
      const w = await GpuWorld.create(dev, list, { substeps: 12, sleep: false });
      for (let f = 0; f < 120; f++) w.step(1 / 60);
      await dev.queue.onSubmittedWorkDone(); await settle();
      const r = await w.readPose();
      let worst = 0, bad = [], vmax = 0;
      for (let k = 0; k < 200; k++) {
        const m = meta[k], o = (2 * k + 1) * 10, pb = [r.pose[o], r.pose[o + 1], r.pose[o + 2]], qb = [r.pose[o + 3], r.pose[o + 4], r.pose[o + 5], r.pose[o + 6]];
        const oa = 2 * k * 10, paN = [r.pose[oa], r.pose[oa + 1], r.pose[oa + 2]], qaN = [r.pose[oa + 3], r.pose[oa + 4], r.pose[oa + 5], r.pose[oa + 6]];
        const PA = planes(m.ia, paN, qaN, m.ha);
        const lowA = Math.min(...verts(m.ia, paN, qaN, m.ha).map(v => v[1])); if (lowA < -0.03 && bad.length < 8) bad.push({ k, aUnderGround: +lowA.toFixed(3), ia: m.ia, ib: m.ib });
        let depth = 0;
        for (const v of verts(m.ib, pb, qb, m.hb)) { const d = Math.min(...PA.map(pl => pl.d - (pl.n[0] * v[0] + pl.n[1] * v[1] + pl.n[2] * v[2]))); depth = Math.max(depth, d); }
        worst = Math.max(worst, depth); if (depth > 0.03 && bad.length < 8) bad.push({ k, ia: m.ia, ib: m.ib, depth: +depth.toFixed(3), yB: +pb[1].toFixed(2) });
        vmax = Math.max(vmax, speed(r.vel, 2 * k + 1));
      }
      w.destroy();
      check('hullpairs', worst < 0.03 && vmax < 3 && !bad.length, { deepest_m: +worst.toFixed(3), fastest_ms: +vmax.toFixed(2), bad });
    }
    if (scenes.includes('hulls')) {
      // convex chunks: (a) every cell of the library, scaled to a 0.5 m brick, dropped tumbling on the ground:
      // all come to rest on a face, none under the ground (centre of mass at the box centre: a wedge toppled about
      // a point outside it; the true centroid: rest speed 0.03 -> 0.0004 m/s); (b) the 6 chunks of a brick placed
      // together on the ground stay as they are (face contacts between chunks hold); (c) a heap of 300 chunks
      // dropped from a grid (HRANDOM=1: at random, deeply overlapping) settles and sleeps, nothing under the ground
      // at any time, no NaN, no step losing pairs. (With 8 pairs per shape the list overflowed in such a heap and
      // lost ground pairs: chunks fell through, -7.8 m; 24 per shape: -0.9 mm.)
      let rnd = 3; const rand = () => { rnd = (rnd * 1103515245 + 12345) & 0x7fffffff; return rnd / 0x7fffffff; };
      const H = GpuWorld.hulls, ids = H.hulls.map((_, i) => i).slice(1);
      const minY = (r, w, b) => { let m = 1e9; for (const sl of w.bodyShapes.get(b) || []) { const sh = w.shapes[sl - 1]; m = Math.min(m, sh ? 0 : 0); } return m; };
      // (a)
      const la = ids.map((id, k) => ({ shapes: [{ p: [(k % 8) * 1.5 - 5, 1 + rand(), Math.floor(k / 8) * 1.5 - 5], q: qa([rand() - 0.5, rand() - 0.5, rand() - 0.5], rand() * 6), h: H.hulls[id].box.h.map(x => x * 0.25 + 0.05), hull: id, mat: MAT.concrete }], v: [0, 0, 0], w: [rand() * 4 - 2, rand() * 4 - 2, rand() * 4 - 2] }));
      let w = await GpuWorld.create(dev, la, { substeps: 12, sleep: false });
      for (let f = 0; f < 240; f++) w.step(1 / 60);
      await dev.queue.onSubmittedWorkDone(); await settle();
      let r = await w.readPose();
      let vmax = 0, ylow = 1e9, nanA = 0;
      for (let b = 1; b <= ids.length; b++) { const v = speed(r.vel, b - 1); if (!Number.isFinite(v)) nanA++; else vmax = Math.max(vmax, v); }
      // the lowest corner of each chunk (a shape's centre is its cell's box centre, which may lie outside a wedge)
      const corner = (pose, k, id) => { const o = k * 10, q = [pose[o + 3], pose[o + 4], pose[o + 5], pose[o + 6]], h = [pose[o + 7], pose[o + 8], pose[o + 9]];
        let m = 1e9; for (const v of H.hulls[id].V) { const x = [v[0] * h[0], v[1] * h[1], v[2] * h[2]];
          const y = 2 * (q[0] * q[1] + q[3] * q[2]) * x[0] + (1 - 2 * (q[0] * q[0] + q[2] * q[2])) * x[1] + 2 * (q[1] * q[2] - q[3] * q[0]) * x[2]; m = Math.min(m, pose[o + 1] + y); } return m; };
      for (let k = 0; k < r.pose.length / 10; k++) ylow = Math.min(ylow, corner(r.pose, k, ids[k]));
      w.destroy();
      // (b) the 6 cells of a pattern, one body each, exactly tiling a 1 x 0.5 x 1 m brick on the ground
      const pat = H.patterns.findIndex(p => p.length === 6);
      const lb = GpuWorld.hullShapes([0, 0.25, 0], [0, 0, 0, 1], [0.5, 0.25, 0.5], pat, MAT.concrete).map(sh => ({ shapes: [sh], v: [0, 0, 0], w: [0, 0, 0] }));
      w = await GpuWorld.create(dev, lb, { substeps: 12, sleep: false });
      const r0 = await w.readPose();
      for (let f = 0; f < 180; f++) w.step(1 / 60);
      await dev.queue.onSubmittedWorkDone(); await settle();
      const r1 = await w.readPose();
      let drift = 0; for (let b = 0; b < lb.length; b++) for (let a = 0; a < 3; a++) drift = Math.max(drift, Math.abs(r1.body[b * 7 + a] - r0.body[b * 7 + a]));
      w.destroy();
      // (c) a heap
      const lc = [];
      for (let k = 0; k < 300; k++) { const pat2 = Math.floor(rand() * H.patterns.length), cells = H.patterns[pat2], id = cells[Math.floor(rand() * cells.length)];
        // (on a jittered grid, as debris made from bricks: never overlapping at the start)
        const gx = k % 5, gz = Math.floor(k / 5) % 5, gy = Math.floor(k / 25);
        lc.push({ shapes: [{ p: globalThis.HRANDOM ? [(rand() - 0.5) * 3, 0.5 + rand() * 6, (rand() - 0.5) * 3] : [(gx - 2) * 0.75 + (rand() - 0.5) * 0.1, 0.5 + gy * 0.75, (gz - 2) * 0.75 + (rand() - 0.5) * 0.1], q: qa([rand() - 0.5, rand() - 0.5, rand() - 0.5], rand() * 6), h: H.hulls[id].box.h.map(x => x * 0.25 + 0.04), hull: globalThis.HBOX ? 0 : id, mat: k % 2 }], v: [0, -2, 0], w: [0, 0, 0] }); }
      if (globalThis.HONLY) { const K = +globalThis.HONLY; const one = [lc[K]]; const w1 = await GpuWorld.create(dev, one, { substeps: 12, sleep: false });
        const tr = []; for (let f = 1; f <= 60; f++) { w1.step(1 / 60); await dev.queue.onSubmittedWorkDone(); await settle(); const rr = await w1.readPose(); tr.push([f, +rr.pose[1].toFixed(3), +rr.vel[1].toFixed(2), +corner(rr.pose, 0, lc[K].shapes[0].hull).toFixed(3)]); }
        w1.destroy(); return { hulls: { ok: false, single: K, spec: lc[K], trace: tr } }; }
      w = await GpuWorld.create(dev, lc, { substeps: 12, sleep: !globalThis.HNOSLEEP, ...(globalThis.HBIG ? { maxContacts: 20000, maxPairs: 40000 } : {}) });
      if (globalThis.HNOCRUSH) w.crushing = false; if (globalThis.HNORED) w.reduce = false;
      const awake = [];
      let firstUnder = null;
      for (let f = 1; f <= 480; f++) {
        w.step(1 / 60);
        if (f % 60 === 0) { await dev.queue.onSubmittedWorkDone(); await settle(); awake.push(w.stats.awake); }
        if (f % 4 === 0) globalThis.__drop = Math.max(globalThis.__drop || 0, w.stats.dropped || 0);
        if (!firstUnder && f % 4 === 0) {
          await dev.queue.onSubmittedWorkDone(); await settle();
          const rr = await w.readPose();
          for (let k = 0; k < rr.pose.length / 10; k++) { const c = corner(rr.pose, k, lc[k].shapes[0].hull || 0);
          if (c < -0.1) { firstUnder = { f, k, hull: lc[k].shapes[0].hull, corner: +c.toFixed(3), centre: Array.from(rr.pose.subarray(k * 10, k * 10 + 3)).map(x => +x.toFixed(2)), h: lc[k].shapes[0].h.map(x => +x.toFixed(2)), v: Array.from(rr.vel.subarray(k * 3, k * 3 + 3)).map(x => +x.toFixed(2)) }; break; } }
        }
      }
      const rc = await w.readPose();
      let nanC = 0, lowC = 1e9, topC = 0;
      for (let k = 0; k < rc.pose.length / 10; k++) { const y = rc.pose[k * 10 + 1]; if (!Number.isFinite(y)) nanC++; else { lowC = Math.min(lowC, corner(rc.pose, k, lc[k].shapes[0].hull || 0)); topC = Math.max(topC, y); } }
      const st = w.stats, lostSteps = w.lost || 0; w.destroy();
      check('hulls', nanA === 0 && vmax < 0.05 && ylow > -0.02 && drift < 0.01 && nanC === 0 && lowC > -0.02 && !firstUnder && !lostSteps && awake[awake.length - 1] < 15,
        { a_restSpeedMax: +vmax.toFixed(4), a_lowestCorner_m: +ylow.toFixed(4), b_pattern: pat, b_drift_m: +drift.toFixed(4), c_awakePerSecond: awake, c_lowestCorner_m: +lowC.toFixed(4), c_firstUnder: firstUnder, c_droppedMax: globalThis.__drop || 0, c_top: +topC.toFixed(2), c_stats: st });
    }
    if (scenes.includes('crush')) {
      // the sandwich again, with crushing: the light pieces under the heavy block are crushed (listed, then
      // colliding with the static world only), the block comes down on the ground, nothing is thrown up; a light
      // piece merely hit from the side by a heavy one is not crushed
      const pieces = [];
      // (rubble-sized pieces, 0.5 m: the stress under the block passes concrete's 30 MPa; the 2 m blocks of the
      // sandwich test would hold it, 17 MPa)
      for (let i = 0; i < 6; i++) for (let k = 0; k < 4; k++) pieces.push({ shapes: [{ p: [-7.5 + i * 3, 0.25, -6 + k * 4], q: [0, 0, 0, 1], h: [0.25, 0.25, 0.25], mat: MAT.concrete }], v: [0, 0, 0], w: [0, 0, 0] });
      const side = { shapes: [{ p: [30, 0.25, 0], q: [0, 0, 0, 1], h: [0.25, 0.25, 0.25], mat: MAT.concrete }], v: [0, 0, 0], w: [0, 0, 0] };
      const ram = { shapes: [{ p: [35.3, 3.0, 0], q: [0, 0, 0, 1], h: [5, 3, 5], mat: MAT.concrete }], v: [-6, 0, 0], w: [0, 0, 0] };
      const block = { shapes: [{ p: [0, 0.5 + 0.001 + 4, 0], q: [0, 0, 0, 1], h: [10, 4, 10], mat: MAT.concrete }], v: [0, -5.7, 0], w: [0, 0, 0] };
      const w = await GpuWorld.create(dev, [...pieces, side, ram, block], { substeps: 12, sleep: false });
      const bi = w.n, sideI = 25;
      let crushed = new Set(), pmax = -1e9;
      for (let f = 1; f <= 60; f++) {
        w.step(1 / 60); await dev.queue.onSubmittedWorkDone(); await settle();
        for (const c of w.takeCrushed()) crushed.add(c.body);
        const r = await w.readPose();
        for (let b = 1; b < 25; b++) pmax = Math.max(pmax, r.vel[(b - 1) * 3 + 1]);
        if (f === 60) { var yB = r.body[(bi - 1) * 7 + 1], vB = r.vel[(bi - 1) * 3 + 1]; }
      }
      w.destroy();
      check('crush', crushed.size >= 20 && !crushed.has(sideI) && yB < 4.2 && pmax < 1, { crushed: crushed.size, sideCrushed: crushed.has(sideI), blockY: +yB.toFixed(2), blockVy: +vB.toFixed(2), piecesMaxUp_ms: +pmax.toFixed(2) });
    }
    if (scenes.includes('compound')) {
      // a big compound body (the upper building: 20 x 20 x 2 bricks of 0.5 m, one body, 400 contacts with the
      // floor) lands at 5.9 m/s on a static floor of bricks: concrete (e 0.1) gives 0.59 m/s of rebound, < 0.8 passes
      const N = +(globalThis.COMP_N || 20), sh = [], fl = [];
      for (let i = 0; i < N; i++) for (let k = 0; k < N; k++) for (let j = 0; j < 2; j++) sh.push({ p: [-N / 2 * 0.5 + i * 0.5 + 0.25, 0.75 + j * 0.5, -N / 2 * 0.5 + k * 0.5 + 0.25], q: [0, 0, 0, 1], h: [0.25, 0.25, 0.25], mat: globalThis.SAND_MAT != null ? +globalThis.SAND_MAT : MAT.concrete });
      if (globalThis.COMP_FLOOR) fl.push({ p: [0, 0.25, 0], q: [0, 0, 0, 1], h: [N / 4 + 1, 0.25, N / 4 + 1], mat: MAT.concrete });
      else for (let i = 0; i < N / 2 + 2; i++) for (let k = 0; k < N / 2 + 2; k++) fl.push({ p: [-(N / 2 + 2) / 2 + i + 0.5, 0.25, -(N / 2 + 2) / 2 + k + 0.5], q: [0, 0, 0, 1], h: [0.5, 0.25, 0.5], mat: globalThis.SAND_MAT != null ? +globalThis.SAND_MAT : MAT.concrete });
      const w = await GpuWorld.create(dev, [{ fixed: true, shapes: fl }, { shapes: sh, v: [0, -5.9, 0], w: [0, 0, 0] }], { substeps: 12, sleep: false, maxShapes: 2000 });
      if (globalThis.SAND_COL) { w.maxColors = +globalThis.SAND_COL; w.colorLimit = +globalThis.SAND_COL; }
      if (globalThis.RELAX) w.relaxRounds = +globalThis.RELAX;
      const bi = w.n; let vmax = -1e9, hit = -1; const vs = [];
      for (let f = 1; f <= 60; f++) {
        w.step(1 / 60);
        await dev.queue.onSubmittedWorkDone(); await settle();
        const r = await w.readPose(); const vy = r.vel[(bi - 1) * 3 + 1];
        if (hit < 0 && vy > -5.5) hit = f;
        if (hit > 0) vmax = Math.max(vmax, vy);
        if (f <= 12) vs.push(+vy.toFixed(2));
      }
      const st = { ...w.stats, colorLimit: w.colorLimit };
      w.destroy();
      check('compound', vmax < 0.8, { hitAt: hit, maxUp_ms: +vmax.toFixed(2), vy: vs, stats: st });
    }
    if (scenes.includes('stumps')) {
      // the upper building landing on the stumps of its cut columns with rubble on it (lab collapse, frame 50: a
      // 1,554-brick body at 6.3 m/s on 5 static stumps, touching dozens of pieces: it went from -6.34 to +1.11
      // m/s, then up again from 0.44 to 0.64 m/s). Here: a slab of 16 x 16 bricks with 6 columns on 5 stumps, two
      // layers of 225 pieces riding on it. Concrete (e 0.1) allows 0.63 m/s of rebound; after that nothing may
      // push the slab up, and the pieces may not be thrown (they land on it at 7.6 m/s).
      // Before the fixes (the mass-ratio limit applied before mass splitting and both ways, the slab's supports
      // left to the parallel pass): slab up at 2.1-8.7 m/s, pieces at up to 15 m/s, varying run to run. After:
      // slab 0.11-0.17 m/s, pieces 1.5 m/s, in three runs.
      const sh = [], fl = [], B = 0.25;
      for (let i = 0; i < 16; i++) for (let k = 0; k < 16; k++) sh.push({ p: [-4 + i * 0.5 + B, 4.25, -4 + k * 0.5 + B], q: [0, 0, 0, 1], h: [B, B, B], mat: MAT.concrete });
      const cols = [[-3.75, -3.75], [-3.75, 3.75], [3.75, 3.75], [3.75, -3.75], [-3.75, 0.25], [3.75, 0.25]];
      for (const [x, z] of cols) for (let j = 0; j < 6; j++) sh.push({ p: [x, 3.75 - j * 0.5, z], q: [0, 0, 0, 1], h: [B, B, B], mat: MAT.concrete });
      cols.forEach(([x, z], c) => { if (c !== 3) fl.push({ p: [x, 0.25, z], q: [0, 0, 0, 1], h: [B, B, B], mat: MAT.concrete }); });
      // starts 5 cm above the stumps, falling at 6.3 m/s
      for (const q of sh) q.p[1] += 0.55;
      // with loose pieces riding on the slab (in the collapse the body touched dozens of pieces, so its contacts
      // went past the colours to the parallel pass)
      const NP = +(globalThis.STUMP_PIECES ?? 450), pcs = [];
      for (let k = 0; k < NP; k++) pcs.push({ shapes: [{ p: [-3.5 + (k % 15) * 0.5, 5.3 + Math.floor(k / 225) * 0.5, -3.5 + Math.floor(k / 15) % 15 * 0.5], q: [0, 0, 0, 1], h: [0.2, 0.2, 0.2], mat: MAT.concrete }], v: [0, -6.3, 0], w: [0, 0, 0] });
      const w = await GpuWorld.create(dev, [{ fixed: true, shapes: fl }, { shapes: sh, v: [0, -6.3, 0], w: [0, 0, 0] }, ...pcs], { substeps: 12, sleep: false, maxShapes: 1000 });
      if (globalThis.STUMP_OP) w.ovfPush = +globalThis.STUMP_OP; if (globalThis.SAND_PUSH) w.maxPush = +globalThis.SAND_PUSH; if (globalThis.RELAX) w.relaxRounds = +globalThis.RELAX; if (globalThis.SAND_COL) { w.maxColors = +globalThis.SAND_COL; w.colorLimit = +globalThis.SAND_COL; }
      const bi = 2; let vmax = -1e9, hit = -1, later = -1e9; const vs = [];
      for (let f = 1; f <= 40; f++) {
        w.step(1 / 60);
        await dev.queue.onSubmittedWorkDone(); await settle();
        const r = await w.readPose(); const vy = r.vel[(bi - 1) * 3 + 1];
        if (hit < 0 && vy > -5) hit = f;
        if (hit > 0 && f <= hit + 1) vmax = Math.max(vmax, vy);
        if (hit > 0 && f > hit + 1) later = Math.max(later, vy);
        if (f <= 16) vs.push(+vy.toFixed(2));
        if (NP && hit > 0 && f <= hit + 4) { let pu = -1e9, pd = 1e9; for (let b = 3; b <= w.n; b++) { const v = r.vel[(b - 1) * 3 + 1]; pu = Math.max(pu, v); pd = Math.min(pd, v); } (globalThis.pcv = globalThis.pcv || []).push([+pu.toFixed(2), +pd.toFixed(2)]); }
        if (f === hit + 2 && w.gpuMs) var stg = Object.fromEntries(Object.entries(w.gpuMs).map(([k, v]) => [k, +v.toFixed(1)]));
      }
      const ovf = w.colCounts[w.colorLimit];
      w.destroy();
      const pmax = Math.max(...(globalThis.pcv || [[0]]).map(x => x[0]));
      check('stumps', vmax < 0.7 && later < 0.7 && pmax < 2.5, { hitAt: hit, reboundUp_ms: +vmax.toFixed(2), laterUp_ms: +later.toFixed(2), vy: vs, overflow: ovf, stages: stg, piecesUpDown: globalThis.pcv });
    }
    if (scenes.includes('particles')) {
      // granular crumbs poured on a fixed table: they pile up on it and around it on the ground, never inside it
      // or under the ground, come to rest and sleep (then the world costs nothing); a box dropped on the pile
      // wakes it and pushes the particles out of its way
      let rnd = 5; const rand = () => { rnd = (rnd * 1103515245 + 12345) & 0x7fffffff; return rnd / 0x7fffffff; };
      const w = await GpuWorld.create(dev, [{ shapes: [{ p: [0, 1, 0], q: [0, 0, 0, 1], h: [1, 0.25, 1], mat: MAT.concrete }], fixed: true }],
        { substeps: 12, sleep: true, maxBodies: 8, maxShapes: 16, maxParticles: 4096 });
      const N = 2000, R = 0.08, list = [];
      for (let i = 0; i < N; i++) list.push({ p: [(rand() - 0.5) * 1.6, 2 + rand() * 6, (rand() - 0.5) * 1.6], v: [0, -1, 0], r: R * (0.7 + 0.6 * rand()), mat: MAT.concrete });
      w.addParticles(list);
      const rows = []; let bad = { below: 0, inside: 0 };
      const look = async (f, boxAt) => {
        await dev.queue.onSubmittedWorkDone(); await settle();
        const r = await w.readParticles(), d = r.data;
        let onTable = 0, top = 0, inBox = 0, active = 0, crushed = 0;
        for (let i = 0; i < r.n; i++) {
          const x = d[i * 12], y = d[i * 12 + 1], z = d[i * 12 + 2], st = d[i * 12 + 10], rr = d[i * 12 + 11];
          if (!st) continue; if (st === 3) { crushed++; continue; } if (st === 1) { active++; if (active <= 2 && f > 200) rows.push({ probe: i, x: +x.toFixed(3), y: +y.toFixed(3), z: +z.toFixed(3), v: [d[i * 12 + 3], d[i * 12 + 4], d[i * 12 + 5]].map(a => +a.toFixed(3)), r: +rr.toFixed(3) }); }
          if (y < rr - 0.02) bad.below++;
          if (Math.abs(x) < 1 - rr && Math.abs(z) < 1 - rr && y > 0.75 && y < 1.25 + rr - 0.03) { bad.inside++; if (bad.inside <= 3) rows.push({ inside: i, t: f / 60, x: +x.toFixed(3), y: +y.toFixed(3), z: +z.toFixed(3), r: +rr.toFixed(3), st }); }
          if (Math.abs(x) < 1 && Math.abs(z) < 1 && y > 1.25) onTable++;
          top = Math.max(top, y);
          if (boxAt && Math.abs(x - boxAt[0]) < 0.3 - rr * 0.5 && Math.abs(y - boxAt[1]) < 0.3 - rr * 0.5 && Math.abs(z - boxAt[2]) < 0.3 - rr * 0.5) inBox++;
        }
        rows.push({ t: +(f / 60).toFixed(1), active, crushed, onTable, top: +top.toFixed(2), inBox, statsActive: w.stats.pActive, skipped: w.skipped || 0, ms: w.gpuMs ? +w.gpuMs.particles.toFixed(2) : null });
      };
      for (let f = 1; f <= 420; f++) { w.step(1 / 60); if (f % 60 === 0) await look(f); }
      const asleep = rows[rows.length - 1].active === 0;
      const skippedBefore = w.skipped || 0;
      for (let f = 0; f < 30; f++) w.step(1 / 60);
      await dev.queue.onSubmittedWorkDone(); await settle();
      const skips = (w.skipped || 0) - skippedBefore;
      // a 0.6 m box dropped on the pile
      const [bi] = w.addBodies([{ shapes: [{ p: [0.3, 3, 0.2], q: [0, 0, 0, 1], h: [0.3, 0.3, 0.3], mat: MAT.concrete }], v: [0, -3, 0], w: [0, 0, 0] }]);
      let woke = 0;
      for (let f = 1; f <= 120; f++) { w.step(1 / 60); if (f === 30) { await dev.queue.onSubmittedWorkDone(); await settle(); woke = w.stats.pActive; } }
      const pr = await w.readPose(); const bp = Array.from(pr.body.subarray((bi - 1) * 7, (bi - 1) * 7 + 3));
      await look(570, bp);
      w.destroy();
      const last = rows[rows.length - 1];
      const perSecond = rows.filter(r => r.t !== undefined && r.onTable !== undefined);   // (probe rows are mixed in)
      check('particles', bad.below === 0 && bad.inside === 0 && perSecond[6].onTable > 300 && asleep && skips > 20 && woke > 0 && last.inBox < 5,
        { bad, asleepAt7s: asleep, skippedSteps: skips, wokeByBox: woke, box: bp.map(v => +v.toFixed(2)), rows });
    }
    if (scenes.includes('wakeup')) {
      // small pieces asleep on the ground, a heavy slab (307 t) set down on them, then the pieces woken from
      // outside: their contacts with the slab were made while they slept (held as static) and carry the slab's
      // weight share into the woken pieces. A suspect for the lab burst (collapse frame 138: a 2-brick piece under
      // the upper part, woken, went from rest to 2,800 m/s); it does not reproduce here (0.16 m/s). Nothing may
      // move faster than 2 m/s.
      const B = 0.25, list = [];
      for (const [x, z] of [[-2, -2], [2, -2], [-2, 2], [2, 2], [0, 0]]) list.push({ shapes: [{ p: [x - B, B, z], q: [0, 0, 0, 1], h: [B, B, B], mat: MAT.concrete }, { p: [x + B, B, z], q: [0, 0, 0, 1], h: [B, B, B], mat: MAT.concrete }], v: [0, 0, 0], w: [0, 0, 0] });
      const w = await GpuWorld.create(dev, list, { substeps: 12, sleep: true, maxShapes: 1200, maxBodies: 16 });
      for (let f = 0; f < 150; f++) w.step(1 / 60);
      await dev.queue.onSubmittedWorkDone(); await settle();
      let r = await w.readPose(); const asleepBefore = Array.from(r.sleeping.slice(0, 5)).reduce((a, b) => a + b, 0);
      const slab = []; for (let i = 0; i < 16; i++) for (let k = 0; k < 16; k++) for (let j = 0; j < 4; j++) slab.push({ p: [-3.75 + i * 0.5, 0.5 + B + 0.005 + j * 0.5, -3.75 + k * 0.5], q: [0, 0, 0, 1], h: [B, B, B], mat: MAT.concrete });
      // set down gently (it does not wake them), then 40 frames later one piece is woken from outside
      for (const q of slab) q.p[1] -= 0.003;
      w.addBodies([{ shapes: slab, v: [0, 0, 0], w: [0, 0, 0] }]);
      let vmax = 0, at = -1; const vs = []; let asleepAt40 = -1;
      for (let f = 1; f <= 90; f++) {
        if (f === 40) {
          await dev.queue.onSubmittedWorkDone(); await settle(); const r0 = await w.readPose();
          asleepAt40 = Array.from(r0.sleeping.slice(0, 5)).reduce((a, b) => a + b, 0);
          for (const b of [1, 2, 3, 4, 5]) w.setVelocity(b, [0, 0.01, 0], [0, 0, 0], [...r0.body.slice((b - 1) * 7, (b - 1) * 7 + 3)], [...r0.body.slice((b - 1) * 7 + 3, (b - 1) * 7 + 7)]);
        }
        w.step(1 / 60);
        await dev.queue.onSubmittedWorkDone(); await settle();
        r = await w.readPose();
        for (let i = 0; i < w.n; i++) { const v = speed(r.vel, i); if (v > vmax) { vmax = v; at = f; } }
        if (f <= 12) vs.push(+Math.max(...[0, 1, 2, 3, 4].map(i => speed(r.vel, i))).toFixed(2));
      }
      w.destroy();
      check('wakeup', asleepBefore === 5 && vmax < 2, { asleepBefore, asleepAt40, fastest_ms: +vmax.toFixed(2), at, piecesMax: vs });
    }
    if (scenes.includes('stress')) {
      // Collapse-like rubble at rest must stay at rest: compound pieces of 1-12 bricks piled between stumps, a
      // heavy slab (288 bricks) landing on them, then 900 frames. Lab collapse (run 3 of 3 full runs): at frame 138
      // a 2-brick piece at rest went to 2,800 m/s and 20,000 rad/s in one frame and threw the pile apart (575 MJ).
      // No body may pass 15 m/s after the landing settles (frame 150 on).
      let rnd = +(globalThis.STRESS_SEED || 7); const rand = () => { rnd = (rnd * 1103515245 + 12345) & 0x7fffffff; return rnd / 0x7fffffff; };
      const B = 0.25, fl = [], list = [];
      for (let i = -3; i <= 3; i += 2) for (let k = -3; k <= 3; k += 2) fl.push({ p: [i * 1.25, 0.5, k * 1.25], q: [0, 0, 0, 1], h: [0.25, 0.5, 0.25], mat: MAT.concrete });
      for (let n = 0; n < +(globalThis.STRESS_N || 120); n++) {
        // (on a grid, none overlapping: placed at random, overlapping pieces were thrown out at 18 m/s on the first
        // frames, which is not what this test is about)
        const nb = 1 + Math.floor(rand() * 12), sh = [], ax = 2 * Math.floor(rand() * 2);
        for (let j = 0; j < nb; j++) { const p = [0, 0, 0]; p[ax] = (j % 6) * 0.5; p[1] = Math.floor(j / 6) * 0.5; sh.push({ p, q: [0, 0, 0, 1], h: [B, B, B], mat: MAT.concrete }); }
        const c = [-4.5 + (n % 3) * 3.25, 1.5 + Math.floor(n / 9) * 1.1, -4.5 + (Math.floor(n / 3) % 3) * 3.25];
        for (const q of sh) for (let a = 0; a < 3; a++) q.p[a] += c[a];
        list.push({ shapes: sh, v: [0, -3, 0], w: [(rand() - 0.5) * 2, (rand() - 0.5) * 2, (rand() - 0.5) * 2] });
      }
      // chunks too (single convex cells, as the page makes from loose bricks)
      const top = 1.5 + Math.ceil(+(globalThis.STRESS_N || 120) / 9) * 1.1 + 0.5;
      for (let n = 0; n < (globalThis.STRESS_NOCHUNK ? 0 : 60); n++) { const c = [-4.5 + (n % 6) * 1.8, top + Math.floor(n / 36) * 0.8, -4.5 + (Math.floor(n / 6) % 6) * 1.8]; for (const s of GpuWorld.hullShapes(c, [0, 0, 0, 1], [B, B, B], n % 16, MAT.concrete).slice(0, 1)) list.push({ shapes: [s], v: [0, -3, 0], w: [0, 0, 0] }); }
      const slab = []; for (let i = 0; i < 12; i++) for (let k = 0; k < 12; k++) for (let j = 0; j < 2; j++) slab.push({ p: [-2.75 + i * 0.5, top + 2.5 + j * 0.5, -2.75 + k * 0.5], q: [0, 0, 0, 1], h: [B, B, B], mat: MAT.concrete });
      list.push({ shapes: slab, v: [0, -5, 0], w: [0, 0, 0] });
      const w = await GpuWorld.create(dev, [{ fixed: true, shapes: fl }, ...list], { substeps: 12, sleep: true, maxShapes: 4000, maxBodies: list.length + 64 });
      if (globalThis.SAND_COL) { w.maxColors = +globalThis.SAND_COL; w.colorLimit = +globalThis.SAND_COL; } if (globalThis.RELAX) w.relaxRounds = +globalThis.RELAX; if (globalThis.STRESS_NORED) w.reduce = false;
      let vmax = 0, wmax = 0, at = -1, nan = 0; const rows = []; const nOrig = w.n;   // (the dropped pieces fall fast; not counted)
      const F = +(globalThis.STRESS_F || 900);
      // the pile asleep by frame 450 is woken by pieces dropped on it (the lab burst came one frame after a body woke)
      const drop = []; for (let n = 0; n < 12; n++) drop.push({ shapes: [{ p: [(rand() - 0.5) * 6, top + 6 + n * 0.6, (rand() - 0.5) * 6], q: [0, 0, 0, 1], h: [B, B, B], mat: MAT.concrete }], v: [0, -4, 0], w: [0, 0, 0] });
      for (let f = 1; f <= F; f++) {
        if (f === 450) { w.addBodies(drop); }
        w.step(1 / 60);
        if (f % 5 === 0) {
          await dev.queue.onSubmittedWorkDone(); await settle();
          const r = await w.readPose();
          let fv = 0, fw = 0, fb = 0;
          // phantom velocity: bodies reported faster than 2 m/s that moved under 2 cm since the last read (5 frames)
          if (globalThis.prevPose && f >= 150) for (let i = 1; i < nOrig; i++) { const v = speed(r.vel, i); if (v > 2 && Math.hypot(r.body[i * 7] - prevPose[i * 7], r.body[i * 7 + 1] - prevPose[i * 7 + 1], r.body[i * 7 + 2] - prevPose[i * 7 + 2]) < 0.02) { globalThis.phantom = (globalThis.phantom || 0) + 1; } }
          globalThis.prevPose = r.body.slice();
          for (let i = 1; i < nOrig; i++) { const v = speed(r.vel, i); const ww = r.ang ? Math.hypot(r.ang[i * 3], r.ang[i * 3 + 1], r.ang[i * 3 + 2]) : 0; if (!Number.isFinite(v)) { nan++; continue; } if (v > fv) { fv = v; fb = i; } fw = Math.max(fw, ww); }
          if (globalThis.fastPend) { const q = globalThis.fastPend; q.after = [0, 1, 2].map(k => +r.body[q.body * 7 + k].toFixed(3)); q.moved = +Math.hypot(...q.after.map((x, k) => x - q.p[k])).toFixed(3); q.expected = +(q.v * 5 / 60).toFixed(3); globalThis.fastPend = null; }
          if (f >= 150 && fv > vmax) { vmax = fv; at = f; globalThis.fastInfo = globalThis.fastPend = { body: fb, f, v: +fv.toFixed(2), p: [0, 1, 2].map(k => +r.body[fb * 7 + k].toFixed(3)), vel: [0, 1, 2].map(k => +r.vel[fb * 3 + k].toFixed(2)), w: [0, 1, 2].map(k => +r.ang[fb * 3 + k].toFixed(1)), asleep: r.sleeping[fb] }; }
          if (f >= 150) wmax = Math.max(wmax, fw);
          if (f % 60 === 0) rows.push({ f, fastest: +fv.toFixed(2), body: fb, spin: +fw.toFixed(1) });
          const wb = +(globalThis.STRESS_WATCH || 0); if (wb && f % 10 === 0) (globalThis.watch = globalThis.watch || []).push([f, ...[0, 1, 2].map(k => +r.body[wb * 7 + k].toFixed(2)), ...[0, 1, 2].map(k => +r.vel[wb * 3 + k].toFixed(2)), ...[0, 1, 2].map(k => +r.ang[wb * 3 + k].toFixed(1)), r.sleeping[wb]]);
        }
      }
      w.destroy();
      check('stress', vmax < 15 && nan === 0, { fastest_ms: +vmax.toFixed(2), at, spin_max: +wmax.toFixed(1), nan, rows, phantom: globalThis.phantom || 0, fastest: globalThis.fastInfo, watch: globalThis.watch });
    }
    if (scenes.includes('rubble')) {
      // a heap of mixed rubble falls and must come to rest: awake bodies per second, and of those how many are
      // still because they themselves are at rest (kept awake by others in their island) or moving
      let rnd = 11; const rand = () => { rnd = (rnd * 1103515245 + 12345) & 0x7fffffff; return rnd / 0x7fffffff; };
      const N = +(globalThis.RUBBLE_N || 1500), list = [];
      for (let i = 0; i < N; i++) {
        const h = [0.05 + 0.25 * rand() ** 2, 0.05 + 0.15 * rand() ** 2, 0.05 + 0.25 * rand() ** 2];
        list.push({ shapes: [{ p: [(rand() - 0.5) * 6, 0.5 + rand() * 12, (rand() - 0.5) * 6], q: qa([rand() - 0.5, rand() - 0.5, rand() - 0.5], rand() * 6), h, mat: i % 3 }], v: [(rand() - 0.5) * 2, -2, (rand() - 0.5) * 2], w: [0, 0, 0] });
      }
      const w = await GpuWorld.create(dev, list, { substeps: +(globalThis.RUBBLE_SUB || 12), sleep: true });
      const rows = [];
      for (let f = 1; f <= 600; f++) {
        w.step(1 / 60);
        if (f % 60 === 0) {
          await dev.queue.onSubmittedWorkDone(); await settle();
          const r = await w.readPose();
          let awake = 0, still = 0, slow = 0, fast = 0;
          for (let i = 0; i < N; i++) { if (r.sleeping[i]) continue; awake++; const v = speed(r.vel, i); if (r.timer[i] >= 0.5) still++; else if (v < 0.5) slow++; else fast++; }
          rows.push({ t: f / 60, awake, stillInIsland: still, slowCreep: slow, fast, stages: w.gpuMs ? Object.fromEntries(Object.entries(w.gpuMs).map(([k, v]) => [k, +v.toFixed(1)])) : null });
        }
      }
      w.destroy();
      const last = rows[rows.length - 1];
      check('rubble', last.awake < N * 0.05, { rows });
    }
    if (scenes.includes('pile')) {
      E.pile_setup(2000, 4);
      const r = await run(GpuWorld.boxes(grab()), 120, { substeps: 4 });
      let miny = 9, maxy = -9, nan = 0;
      for (let i = 0; i < 2000; i++) { const y = r.pose[i * 10 + 1]; if (!isFinite(y)) nan++; miny = Math.min(miny, y); maxy = Math.max(maxy, y); }
      E.pile_setup(2000, 4); for (let f = 0; f < 120; f++) E.step(1 / 60);
      const c = grab(); let cmin = 9, cmax = -9; for (let i = 0; i < 2000; i++) { cmin = Math.min(cmin, c[i * 10 + 1]); cmax = Math.max(cmax, c[i * 10 + 1]); }
      // narrow-phase cost against the broad phase (a ratio, so it holds on any GPU): box pairs must not pay for the
      // convex chunk test or for threads sized by the pair capacity. Software GPU, 2000 boxes: before the fix narrow
      // 260 ms against broad 49 ms (5.3x); after, 63 against 53 (1.2x); before chunks came in, 57 against 49
      const nr = r.gpu ? r.gpu.narrow / r.gpu.broad : 0;
      check('pile', nan === 0 && Math.abs(miny - cmin) < 0.02 && Math.abs(maxy - cmax) < 0.1 && nr < 2.5, { narrowOverBroad: +nr.toFixed(2), gpu: { miny, maxy, stats: r.stats, ms: r.ms, stages: r.gpu }, cpu: { miny: cmin, maxy: cmax, manifolds: E.manifolds() } });
      done(r);
    }
    return out;
  }, { wasm, scenes, SUB: +(process.env.SF_SUB || 0), SAND_PUSH: process.env.SAND_PUSH || null, SAND_MAT: process.env.SAND_MAT || null, SAND_SUB: process.env.SAND_SUB || null, SAND_COL: process.env.SAND_COL || null, SAND_FALL: process.env.SAND_FALL || null, COMP_N: process.env.COMP_N || null, COMP_FLOOR: process.env.COMP_FLOOR || null, RELAX: process.env.RELAX || null, HNOSLEEP: process.env.HNOSLEEP || null, HBOX: process.env.HBOX || null, HNOCRUSH: process.env.HNOCRUSH || null, HNORED: process.env.HNORED || null, HONLY: process.env.HONLY || null, HPDYN: process.env.HPDYN || null, HRANDOM: process.env.HRANDOM || null, HBIG: process.env.HBIG || null, KN: Object.fromEntries(['STUMP_PIECES', 'STUMP_TILT', 'STUMP_OP', 'DBG', 'STRESS_SEED', 'STRESS_N', 'STRESS_F', 'STRESS_WATCH', 'STRESS_NORED', 'STRESS_NOCHUNK'].filter(k => process.env[k]).map(k => [k, process.env[k]])) });
  let fail = 0;
  for (const [k, v] of Object.entries(res)) { console.log((v.ok ? 'PASS ' : 'FAIL ') + k + ' ' + JSON.stringify(v)); if (!v.ok) fail++; }
  // the results for docs/STATUS.md (tools/status.py): full runs only (a run of a few scenes is not the state)
  if (process.argv.length <= 2) try {
    const out = process.env.SF_GPU_OUT || require('path').join(__dirname, '..', 'lab-results', 'gpu.json');
    require('fs').mkdirSync(require('path').dirname(out), { recursive: true });
    require('fs').writeFileSync(out, JSON.stringify({ date: new Date().toISOString(), scenes: process.argv.slice(2), results: res }));
  } catch (e) {}
  await b.close();
  process.exit(fail ? 1 : 0);
})();
