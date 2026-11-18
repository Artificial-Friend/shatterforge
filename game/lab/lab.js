// ---- lab: scripted scenarios on a virtual clock, checked as data (not screenshots)
// Every lab frame runs the page's own logic (engine jobs, physics steps, read-backs, the instance lists handed to
// the renderer) at a simulated 60 fps, however long the GPU takes: a software GPU and a fast one go through the
// same frames. Read-backs arrive LAB.readLag frames after they were issued and engine jobs LAB.jobLag frames
// after they were sent, as on a real machine. What would be drawn is checked every frame; the physics at every
// read-back. The result is a report of checks (pass/fail with limits), events and a time series.
function labTruthStatic() { let v = 0; for (let k = 0; k < E.chunk_count(); k++) v += E.chunk_volume(k); return v; }
// the standing volume above the ground storey (m^3; a building with a core keeps its roof height over the core
// while everything around it comes down)
function labUpper() { let v = 0; for (let k = 1; k < E.chunk_count(); k++) v += E.chunk_volume(k); return v; }
function labBoxVol(h) { return 8 * h[0] * h[1] * h[2]; }
function labInstall() {
  // the volume of everything the GPU holds, kept as it changes (the ball is not part of the building)
  const L = LAB;
  L.bodyVol = new Map(); L.partVol = new Map(); L.gpuVol = 0; L.balls = new Set();
  // (what the page still asks of the GPU in the last second of a run: any of these makes a step that cannot be
  // skipped, settled or not)
  if (!gpu.__labCalls) { gpu.__labCalls = true; for (const k of ['addBodies', 'setStaticShape', 'killBody', 'setSnap', 'breakBody', 'setVelocity', 'wakeAll', 'addParticles', 'killParticle', 'killShape']) { const f0 = gpu[k].bind(gpu); gpu[k] = (...a) => { const L = LAB; if (L.on && L.spec && L.f > L.spec.frames - 60) { L.lateCalls = L.lateCalls || {}; L.lateCalls[k] = (L.lateCalls[k] || 0) + 1; } return f0(...a); }; } }
  // (a piece weighs on the GPU what it weighs in the engine: the worst ratio, and what it was)
  L.massWorst = 1; L.massChecked = 0;
  window.__labMass = (body, piece) => {
    const gb = gpu.bodies[body - 1]; if (!gb || !(piece >= 0) || !E.piece_mass) return;
    const m = E.piece_mass(piece); if (!(m > 1) || !(gb.mass > 0)) return;
    L.massChecked++;
    const r = gb.mass / m; if (Math.abs(Math.log(r)) > Math.abs(Math.log(L.massWorst))) { L.massWorst = r; L.massWorstOf = { gpu_kg: Math.round(gb.mass), engine_kg: Math.round(m), frame: L.f }; }
  };
  const addB = gpu.addBodies.bind(gpu), brk = gpu.breakBody.bind(gpu), addP = gpu.addParticles.bind(gpu), killP = gpu.killParticle.bind(gpu);
  gpu.addBodies = (list) => {
    for (const bd of list) { for (const x of (bd.v || [0, 0, 0])) mixIn(x); for (const x of (bd.w || [0, 0, 0])) mixIn(x); for (const sh of bd.shapes) for (const x of sh.p) mixIn(x); }
    const idx = addB(list);
    idx.forEach((b, k) => {
      if (b <= 0) return;
      const gb = gpu.bodies[b - 1];
      if (L.ballNext) { L.balls.add(b); if (gb) gb.labBall = true; for (const k of gpu.bodyShapes.get(b) || []) { const sh = gpu.shapes[k - 1]; if (sh) sh.labBall = true; } return; }
      const v = list[k].shapes.reduce((s, sh) => s + labBoxVol(sh.h) * (sh.hull ? GpuWorld.hulls.hulls[sh.hull].vol : 1), 0); L.bodyVol.set(b, v); L.gpuVol += v; if (L.jobVol) L.jobVol.bodiesIn += v;
      // (where parts come loose: frame, the job that made them, height of the body's top and bottom, volume)
      if (L.made && L.made.length < 4000) { const ys = list[k].shapes.map(sh => sh.p[1]); L.made.push([L.f, L.jobOp || '', +Math.min(...ys).toFixed(1), +Math.max(...ys).toFixed(1), +v.toFixed(2), list[k].shapes.length]); }
      if (gb) { gb.labVol = v; gb.labRho = gb.mass > 0 && v > 0 ? gb.mass / v : 2400; gb.labH = list[k].shapes.length === 1 ? list[k].shapes[0].h.map(x => +x.toFixed(2)) : `${list[k].shapes.length} boxes`; gb.labF = L.f; }
    });
    L.ballNext = false; return idx;
  };
  const setV = gpu.setVelocity.bind(gpu);
  gpu.setVelocity = (b, v, ...rest) => { const gb = gpu.bodies[b - 1]; if (gb) gb.labSetV = { f: L.f, v: v.map(x => +x.toFixed(2)) }; return setV(b, v, ...rest); };
  const takeE = gpu.takeEvents.bind(gpu);
  gpu.takeEvents = () => {
    const evs = takeE(), tr = L.traceWin;
    // (the last 3 frames of impact events, for the flight recorder)
    L.recentImpacts = (L.recentImpacts || []).filter(e => e.f >= L.f - 3).concat(evs.slice(0, 200).map(e => ({ f: L.f, a: e.a, b: e.b, J: Math.round(e.impulse || 0), approach: +(e.approach || 0).toFixed(2), n: e.n && e.n.map(x => +x.toFixed(2)), p: e.p && e.p.map(x => +x.toFixed(2)) })));
    if (tr && L.f >= tr[0] && L.f <= tr[1]) for (const e of evs) {
      const ga = gpu.bodies[e.a - 1], gb = gpu.bodies[e.b - 1];
      const inWin = (g) => g && g.labF >= tr[0] - 15;
      if ((inWin(ga) || inWin(gb)) && L.traceEv.length < 3000) L.traceEv.push({ f: L.f, a: e.a, b: e.b, ma: e.a === 1 ? 'static' : (ga && ga.labH), mb: e.b === 1 ? 'static' : (gb && gb.labH), J: Math.round(e.impulse || 0), approach: +(e.approach || 0).toFixed(2), n: e.n && e.n.map(x => +x.toFixed(2)), p: e.p && e.p.map(x => +x.toFixed(2)) });
    }
    return evs;
  };
  gpu.breakBody = (parent, ...rest) => {
    const v = L.bodyVol.get(parent) || 0; L.gpuVol -= v; L.bodyVol.delete(parent); if (L.jobVol) L.jobVol.bodiesOut += v;
    // (a ball removed frees its slot: the next body there is not a ball. Left marked, a piece of the facade given
    // that slot was left out of what is drawn, and the lab saw a 0.2 m3 hole for 214 frames, scenario far)
    const wasBall = L.balls.has(parent); L.balls.delete(parent);
    if (window.__farKill === 'budget') { L.lostBy.budget = (L.lostBy.budget || 0) + v; L.budgetBodies = (L.budgetBodies || 0) + 1; L.budgetBodyVol = Math.max(L.budgetBodyVol || 0, v); }
    else if (window.__farKill) { L.lostBy.far = (L.lostBy.far || 0) + v; L.farBodies = (L.farBodies || 0) + 1; const gb = gpu.bodies[parent - 1]; labEvent('removed_far', L.f, { body: parent, vol: +v.toFixed(3), ball: wasBall, size: gb && gb.labH, made_at: gb && gb.labF, shapes: (gpu.bodyShapes.get(parent) || []).length }); }   // (removed past the clean-up distance)
    // (a break must not make volume: the kids, bodies and particles, against the parent; logged when over by 5%)
    { const kv = (rest[0] || []).reduce((s, k) => s + (L.bodyVol.get(k) || 0), 0) + (L.pendingParts || []).reduce((s, i) => s + (i >= 0 && pInfo[i] ? labBoxVol(pInfo[i].h) : 0), 0);
      if (kv > 1.05 * v + 0.5 && (L.overBreaks = L.overBreaks || []).length < 20) { const gb = gpu.bodies[parent - 1]; L.overBreaks.push({ f: L.f, op: L.jobOp, parent, parent_m3: +v.toFixed(2), kids_m3: +kv.toFixed(2), kids: (rest[0] || []).length, made_at: gb && gb.labF, what: gb && gb.labH, spare: !!(gb && gb.spare), shapes_gpu: (gpu.bodyShapes.get(parent) || []).length }); } }
    // the pieces handed over must be the ones just made: an older body among them would be moved to the parent's
    // pose and given the pieces' velocity
    (rest[0] || []).forEach((kid, k) => { const gb = kid > 0 && gpu.bodies[kid - 1]; if (gb && gb.labF !== undefined && gb.labF < L.f) { L.badKids = (L.badKids || 0) + 1; labEvent('break_gives_old_body', L.f, { parent, kid, k, made_at: gb.labF, size: gb.labH }); } });
    // Parts cut free keep a velocity of their own (the page's crush-down: they fall on, slowed by the energy the
    // section that gave way took), above the parent's, which the GPU had stopped rigidly. That energy comes from
    // the page, not the solver: it is counted as given, and the energy check allows for it while it spreads to
    // the bodies they hit (in the collapse at frame 54 the upper part, cut free at -5 m/s, hit the pieces under
    // it: 0.6-0.9 MJ "gained" by bodies that were there before).
    const pw = L.prevBodies && L.prevBodies.get(parent), vp = pw ? pw.v : [0, 0, 0];
    (rest[0] || []).forEach((kid, k) => {
      const d = (rest[3] || [])[k], gb = kid > 0 && gpu.bodies[kid - 1];
      if (!gb || !d || Array.isArray(d) || !d.v) return;
      const m = (gb.labVol || 0) * (gb.labRho || 2400), e = 0.5 * m * (d.v[0] ** 2 + d.v[1] ** 2 + d.v[2] ** 2 - vp[0] ** 2 - vp[1] ** 2 - vp[2] ** 2);
      if (e > 0) { L.given = (L.given || 0) + e; L.givenTotal = (L.givenTotal || 0) + e; }
    });
    return brk(parent, ...rest);
  };
  // a piece broken on the GPU: the spare takes its part's volume from the piece; a spare given up loses it
  const snp = gpu.snapped.bind(gpu), kill = gpu.killBody.bind(gpu);
  gpu.snapped = (i, j, imp) => {
    const r = snp(i, j, imp);
    (L.ops = L.ops || []).push({ f: L.f, op: 'snapped', body: i, spare: j, books_i: +(L.bodyVol.get(i) || 0).toFixed(3), had_i: L.bodyVol.has(i), spare_shapes: (gpu.bodyShapes.get(j) || []).length });
    const moved = (gpu.bodyShapes.get(j) || []).reduce((t, k) => { const sh = gpu.shapes[k - 1]; return t + (sh ? labBoxVol(sh.h) * (sh.hull ? GpuWorld.hulls.hulls[sh.hull].vol : 1) : 0); }, 0);
    if (L.bodyVol.has(i) && moved > 0) {
      L.bodyVol.set(i, L.bodyVol.get(i) - moved); L.bodyVol.set(j, moved); const gb = gpu.bodies[j - 1]; if (gb) { gb.labVol = moved; gb.labRho = (gpu.bodies[i - 1] || {}).labRho || 2400; gb.labF = L.f; gb.labH = 'spare'; }
      // (the piece left is another body from here: its mass is less and its centre elsewhere. Compared with the whole
      // at the last read-back, its centre's jump read as a gain: 7.9 MJ for the 816 m3 upper part at frame 58 of the
      // lab collapse, 21 MJ at frame 223, at the speed it had)
      const gi = gpu.bodies[i - 1]; if (gi) { gi.labVol = L.bodyVol.get(i); gi.labCut = L.f; }
    }
    return r;
  };
  const shapesVol = (b) => (gpu.bodyShapes.get(b) || []).reduce((t, k) => { const sh = gpu.shapes[k - 1]; return t + (sh && !sh.dead ? labBoxVol(sh.h) * (sh.hull ? GpuWorld.hulls.hulls[sh.hull].vol : 1) : 0); }, 0);
  const op = (o) => { (L.ops = L.ops || []).push({ f: L.f, ...o }); if (L.ops.length > 40) L.ops.shift(); };
  gpu.killBody = (b) => { const v = L.bodyVol.get(b) || 0; op({ op: 'kill', body: b, books: +v.toFixed(3), shapes_m3: +shapesVol(b).toFixed(3), spare: !!(gpu.bodies[b - 1] || {}).spare }); L.gpuVol -= v; L.bodyVol.delete(b); return kill(b); };
  // (what goes in, hashed: two runs that differ in what they give the GPU differ here first)
  L.inHash = 2166136261 >>> 0; const mixIn = (x) => { L.inHash = Math.imul(L.inHash ^ (Math.round(x * 1e4) >>> 0), 16777619) >>> 0; };
  gpu.addParticles = (list) => { for (const e of list) { for (const x of e.p) mixIn(x); for (const x of e.v) mixIn(x); } const s = addP(list); L.pendingParts = (L.pendingParts || []).concat(s); return s; };
  gpu.killParticle = (i) => { const v = L.partVol.get(i) || 0; L.gpuVol -= v; L.partVol.delete(i); if (window.__farKill === 'budget') { L.lostBy.budget = (L.lostBy.budget || 0) + v; L.budgetParts = (L.budgetParts || 0) + 1; } else if (window.__farKill) { L.lostBy.far = (L.lostBy.far || 0) + v; L.farParts = (L.farParts || 0) + 1; labEvent('removed_far', L.f, { particle: i, vol: +v.toFixed(3) }); } else L.lostParticles = (L.lostParticles || 0) + v; return killP(i); };
}
// (particle volumes are known once the page has their brick sizes: after addPieces)
function labParticlesMade() {
  const L = LAB; if (!L.on || !L.pendingParts) return;
  for (const i of L.pendingParts) if (i >= 0 && pInfo[i]) { const v = labBoxVol(pInfo[i].h); L.partVol.set(i, v); L.gpuVol += v; }
  L.pendingParts = null;
}
function labTruth() { LAB.T = LAB.staticT + LAB.gpuVol; }
// (a read-back shows the bodies as they were when it was issued: a slot reused since holds another body, so its
// checks use the body records of that moment)
function labGate(p) {
  if (!LAB.on) return p;
  const at = gpu.bodies.slice();
  return new Promise(res => LAB.gates.push({ p: p.then(v => { v[0].bodiesAt = at; return v; }), due: LAB.f + LAB.readLag, res }));
}
function labEvent(kind, f, detail) { const e = LAB.events; (e[kind] = e[kind] || { n: 0, first: [] }).n++; if (e[kind].first.length < 12) e[kind].first.push({ f, ...detail }); }

// what would be drawn this frame (filled by drawDebris)
function labDrawBegin() { const d = LAB.draw; d.n = 0; d.vol = 0; }
function labDrawn(key, x, y, z, bound, vol) {
  const d = LAB.draw;
  if (d.n >= d.keys.length) { const g = (a, C) => { const b = new C(a.length * 2); b.set(a); return b; }; d.keys = g(d.keys, Float64Array); d.pos = g(d.pos, Float32Array); d.bound = g(d.bound, Float32Array); d.vols = g(d.vols, Float32Array); }
  d.vols[d.n] = vol; d.keys[d.n] = key; d.pos[d.n * 3] = x; d.pos[d.n * 3 + 1] = y; d.pos[d.n * 3 + 2] = z; d.bound[d.n] = bound; d.n++; d.vol += vol;
}
// (for a hole: the bodies and particles that exist but have nothing drawn, biggest first)
function labUndrawn() {
  const L = LAB, d = L.draw, drawnBodies = new Set(), drawnParts = new Set();
  for (let k = 0; k < d.n; k++) { const key = d.keys[k]; if (key < 0) drawnParts.add((-key - 1) % 1048576); else { const sh = gpu.shapes[(key % 1048576) - 1]; if (sh) drawnBodies.add(sh.body); } }
  const out = [];
  for (const [b, v] of L.bodyVol) if (!drawnBodies.has(b)) { const gb = gpu.bodies[b - 1]; out.push({ body: b, vol: +v.toFixed(3), made: gb && gb.labF, size: gb && gb.labH }); }
  for (const [i, v] of L.partVol) if (!drawnParts.has(i)) out.push({ particle: i, vol: +v.toFixed(3), state: snapCur && snapCur.parts && snapCur.parts.n > i ? snapCur.parts.data[i * 12 + 10] : null });
  out.sort((a, b) => b.vol - a.vol);
  return { count: out.length, vol: +out.reduce((a, x) => a + x.vol, 0).toFixed(3), first: out.slice(0, 6) };
}
// (for a hole or a duplicate: body by body, what is drawn of it against what the books hold for it; and the
// particles as a whole)
function labDrawDiff() {
  const L = LAB, d = L.draw, by = new Map(); let pv = 0, pn = 0;
  for (let k = 0; k < d.n; k++) { const key = d.keys[k]; if (key < 0) { pn++; const inf = pInfo[(-key - 1) % 1048576]; pv += inf ? labBoxVol(inf.h) : 0; } else { const sh = gpu.shapes[(key % 1048576) - 1], b = sh ? sh.body : 0; by.set(b, (by.get(b) || 0) + (d.vols ? d.vols[k] : 0)); } }
  const out = [];
  for (const [b, v] of by) { const t = L.bodyVol.get(b) || 0; if (Math.abs(v - t) > 0.005) { const gb = gpu.bodies[b - 1]; out.push({ body: b, drawn: +v.toFixed(3), books: +t.toFixed(3), made: gb && gb.labF, size: gb && gb.labH, cut: gb && gb.labCut, ball: !!(gb && gb.labBall) }); } }
  out.sort((a, b) => Math.abs(b.drawn - b.books) - Math.abs(a.drawn - a.books));
  let pt = 0; for (const v of L.partVol.values()) pt += v;
  return { bodies: out.slice(0, 8), bodies_off: out.length, particles_drawn: pn, particles_drawn_m3: +pv.toFixed(3), particles_books: L.partVol.size, particles_books_m3: +pt.toFixed(3) };
}
// every frame: what is drawn against what exists
function labFrameChecks() {
  const L = LAB, d = L.draw, f = L.f;
  // a) volume: drawn (building mesh + debris + particles) within what existed over the last few frames (the
  //    read-back delay); below = something that exists is not drawn (a hole, flicker), above = drawn twice
  const D = L.staticDrawn + d.vol;
  const parts = () => ({ static_drawn: +L.staticDrawn.toFixed(2), static_now: +L.staticT.toFixed(2), gpu_drawn: +d.vol.toFixed(2), gpu_now: +L.gpuVol.toFixed(2), reads_pending: L.gates.length, chunks_pending: pendingChunks.length, placing: gpu.placing() });
  L.Thist.push(L.T); if (L.Thist.length > L.readLag + L.jobLag + 4) L.Thist.shift();
  // (the last frames' sums, for a hole's or a duplicate's event: which part of the picture changed)
  { let bv = 0, pv = 0, bn = 0, pn = 0; for (let k = 0; k < d.n; k++) { if (d.keys[k] < 0) { pv += d.vols[k]; pn++; } else { bv += d.vols[k]; bn++; } }
    (L.sums = L.sums || []).push({ f, static_drawn: +L.staticDrawn.toFixed(3), bodies_drawn: +bv.toFixed(3), shapes: bn, parts_drawn: +pv.toFixed(3), parts: pn, static_now: +L.staticT.toFixed(3), gpu_now: +L.gpuVol.toFixed(3), read: snapCur ? snapCur.frame : -1 }); if (L.sums.length > 6) L.sums.shift(); }
  const lo = Math.min(...L.Thist), hi = Math.max(...L.Thist), tol = L.volTol;
  if (D < lo - tol) { L.holeFrames++; L.holeMax = Math.max(L.holeMax, lo - D); if (!L.inHole) labEvent('hole', f, { missing_m3: +(lo - D).toFixed(3), T_now: +L.T.toFixed(3), T_window: L.Thist.map(x => +x.toFixed(3)), ...parts(), undrawn: labUndrawn(), diff: labDrawDiff(), last_frames: L.sums.slice(), ops: (L.ops || []).filter(o => o.f >= f - 6) }); L.inHole = true; }
  else if (L.inHole) { L.inHole = false; L.holesRecovered++; }
  if (D > hi + tol) { L.dupFrames++; L.dupMax = Math.max(L.dupMax, D - hi); labEvent('duplicate', f, { extra_m3: +(D - hi).toFixed(3), T_now: +L.T.toFixed(3), T_window: L.Thist.map(x => +x.toFixed(3)), ...parts(), diff: labDrawDiff(), last_frames: L.sums.slice(), ops: (L.ops || []).filter(o => o.f >= f - 6) }); }
  L.D = D;
  // b) instances: each drawn piece moves no further between frames than its speed allows, and a piece does not
  //    disappear and come back (flicker)
  const prev = L.prevDraw, cur = new Map();
  for (let k = 0; k < d.n; k++) {
    const key = d.keys[k]; cur.set(key, k);
    if (prev) {
      const j = prev.map.get(key);
      if (j !== undefined) {
        const dx = d.pos[k * 3] - prev.pos[j * 3], dy = d.pos[k * 3 + 1] - prev.pos[j * 3 + 1], dz = d.pos[k * 3 + 2] - prev.pos[j * 3 + 2];
        const m = Math.hypot(dx, dy, dz); let lim = 0.25 + 2 * Math.max(d.bound[k], prev.bound[j]) / 60;
        // (a particle is carried by a body that lands on it: it goes at the body's speed while pushed, and its own
        // velocity, read before and after, is that of rest. The speed that counts is then the fastest shape's
        // within 3 m (a brick is at most 2 m), over the 3 steps a read-back can span. Lab collapse, frame 52: the
        // upper storeys land at 8.5 m/s on crumbs lying since the cut, which go down 0.25 to 0.28 m to the
        // ground, with speed 0 before and after: 4 "jumps" against a limit of 0.25 m.)
        if (m > lim && key < 0) {
          let vn = 0; for (let q = 0; q < d.n; q++) { if (d.keys[q] < 0 || d.bound[q] <= vn) continue; if (Math.hypot(d.pos[q * 3] - d.pos[k * 3], d.pos[q * 3 + 1] - d.pos[k * 3 + 1], d.pos[q * 3 + 2] - d.pos[k * 3 + 2]) < 3) vn = d.bound[q]; }
          lim = Math.max(lim, 0.25 + 3 * vn / 60);
        }
        // (a body whose velocity the page has just set, a ball going on through a wall it has paid for, moves at that
        // velocity before a read-back shows it: the speed that counts is the one it was given, over the 3 steps a
        // read-back can span. Lab qballtemple, 15 Nov: the ball, read at 4.8 m/s where it had met the wall before
        // the engine's answer, went on at 43.8 m/s and was counted four times, once a shape.)
        if (m > lim && key >= 0) { const sh0 = gpu.shapes[(key % 1048576) - 1], g0 = sh0 ? gpu.bodies[sh0.body - 1] : null; if (g0 && g0.labSetV && f - g0.labSetV.f <= 4) lim = Math.max(lim, 0.25 + 3 * Math.hypot(...g0.labSetV.v) / 60); }
        if (m > lim) { L.teleports++; const sh = key >= 0 ? gpu.shapes[(key % 1048576) - 1] : null, gb = sh ? gpu.bodies[sh.body - 1] : null; labEvent('teleport', f, { key, track: (L.drawHist || []).map(h => { const q = h.map.get(key); return q === undefined ? [h.f, h.read] : [h.f, h.read, +h.pos[q * 3].toFixed(3), +h.pos[q * 3 + 1].toFixed(3), +h.pos[q * 3 + 2].toFixed(3), +h.bound[q].toFixed(2)]; }), moved_m: +m.toFixed(2), limit_m: +lim.toFixed(2), speed_was: +prev.bound[j].toFixed(2), speed_now: +d.bound[k].toFixed(2), part: key < 0 ? (() => { const pi = (-key - 1) % 1048576, inf = pInfo[pi]; let best = null, bd = 1e9; for (let q = 0; q < d.n; q++) { if (d.keys[q] < 0) continue; const dd = Math.hypot(d.pos[q * 3] - d.pos[k * 3], d.pos[q * 3 + 1] - d.pos[k * 3 + 1], d.pos[q * 3 + 2] - d.pos[k * 3 + 2]); if (dd < bd) { bd = dd; best = q; } } const sh = best !== null ? gpu.shapes[(d.keys[best] % 1048576) - 1] : null, gb = sh ? gpu.bodies[sh.body - 1] : null; const pd = snapCur && snapCur.parts ? snapCur.parts.data : null; return inf ? { made: inf.labF, why: inf.labWhy, v0: inf.labV, h: inf.h, state: pd ? pd[pi * 12 + 10] : null, v: pd ? [pd[pi * 12 + 3], pd[pi * 12 + 4], pd[pi * 12 + 5]].map(x => +x.toFixed(2)) : null, nearest_shape_m: +bd.toFixed(2), nearest_speed: best !== null ? +d.bound[best].toFixed(2) : null, nearest_body: sh ? sh.body : null, nearest_size: gb && gb.labH, nearest_made: gb && gb.labF, nearest_at: best !== null ? [d.pos[best * 3], d.pos[best * 3 + 1], d.pos[best * 3 + 2]].map(x => +x.toFixed(2)) : null } : null; })() : undefined, body: sh ? sh.body : null, made: gb && gb.labF, size: gb && gb.labH, from: [prev.pos[j * 3], prev.pos[j * 3 + 1], prev.pos[j * 3 + 2]].map(v => +v.toFixed(2)), to: [d.pos[k * 3], d.pos[k * 3 + 1], d.pos[k * 3 + 2]].map(v => +v.toFixed(2)) }); }
      } else {
        const gone = L.goneAt.get(key);
        if (gone !== undefined && f - gone <= 30) { L.flickers++; labEvent('flicker', f, { key, absent_frames: f - gone }); }
      }
    }
  }
  if (prev) for (const [key] of prev.map) if (!cur.has(key)) L.goneAt.set(key, f);
  L.prevDraw = { map: cur, pos: d.pos.slice(0, d.n * 3), bound: d.bound.slice(0, d.n) };
  // (the last 16 frames as drawn, for a jump's event: where the piece was before)
  (L.drawHist = L.drawHist || []).push({ f, read: snapCur ? snapCur.frame : -1, ...L.prevDraw }); if (L.drawHist.length > 16) L.drawHist.shift();
}
// every read-back: the physics
const LAB_RHO = [2400, 1800, 2500, 2500, 500, 7800, 2400, 2400];
// (for the rebound check) How much faster than its centre falls a piece can meet what it lands on: its own spin
// times its reach, and the upward speed, at the piece's place, of the fastest heavier body that reaches it.
function labReach(b) {
  let r2 = 0;
  for (const k of gpu.bodyShapes.get(b) || []) { const sh = gpu.shapes[k - 1]; if (!sh || sh.dead || !sh.lp) continue; r2 = Math.max(r2, (Math.abs(sh.lp[0]) + sh.h[0]) ** 2 + (Math.abs(sh.lp[1]) + sh.h[1]) ** 2 + (Math.abs(sh.lp[2]) + sh.h[2]) ** 2); }
  return Math.sqrt(r2);
}
function labMeet(b, was, cur, prev, now) {
  const R = labReach(b), own = Math.hypot(...was.w) * R;
  let up = 0;
  for (const set of [prev, now]) for (const [o, x] of set || []) {
    if (o === b || !(x.m >= was.m) || !x.p) continue;
    const d = [cur.p[0] - x.p[0], cur.p[1] - x.p[1], cur.p[2] - x.p[2]], dist = Math.hypot(...d);
    if (dist > R + (x.reach === undefined ? (x.reach = labReach(o)) : x.reach)) continue;
    up = Math.max(up, x.v[1] + x.w[2] * d[0] - x.w[0] * d[2]);   // (v + w x d, its y)
  }
  return own + up;
}

function labOnRead(r, pr) {
  // spares parked under the ground must sleep there (woken, they fell for the rest of the run: 8 November)
  if (r.sleeping && r.body) { let n = 0; const nb = Math.min(r.sleeping.length, r.body.length / 7); for (let b = 1; b <= nb; b++) if (!r.sleeping[b - 1] && r.body[(b - 1) * 7 + 1] < -500) n++; LAB.parkedAwake = Math.max(LAB.parkedAwake || 0, n); }
  // (the GPU world's input count over the last second: any change there makes every step count as changed)
  if (LAB.spec && LAB.f >= LAB.spec.frames - 60 && !LAB.lateInputs) LAB.lateInputs = { from: gpu.inputs, at: LAB.f };
  if (LAB.spec && LAB.lateInputs && LAB.f >= LAB.spec.frames - 12) { LAB.lateInputs.to = gpu.inputs; LAB.lateInputs.skipped = gpu.skipped || 0; LAB.lateInputs.wakeReq = !!gpu.wakeReq; LAB.lateInputs.inherit = (gpu.pendingInherit || []).length; LAB.lateInputs.stats = { awake: gpu.stats.awake, pActive: gpu.stats.pActive }; LAB.lateInputs.statsInputs = gpu.statsInputs; LAB.lateInputs.blasts = (gpu.pBlasts || []).length; LAB.lateInputs.sleep = gpu.sleep; LAB.lateInputs.frame = gpu.frame; LAB.lateInputs.realFrame = gpu.realFrame; }
  // (what is still awake near the end, for a run that does not settle: bodies, and where the awake particles are)
  if (LAB.f >= LAB.spec.frames - 12 && r.sleeping && !LAB.awakeEnd) {
    const L = LAB, list = [], nb = Math.min(gpu.bodies.length, r.body.length / 7);
    for (let b = 1; b <= nb; b++) { const gb = gpu.bodies[b - 1]; if (r.sleeping[b - 1]) continue; const o = (b - 1) * 7, vo = (b - 1) * 3;
      if (!gb || gb.dead || r.body[o + 1] < -500) { (L.awakeOdd = L.awakeOdd || []).push({ b, dead: !gb || !!gb.dead, spare: !!(gb && gb.spare), fixed: !!(gb && gb.fixed), y: +r.body[o + 1].toFixed(1) }); continue; }
      list.push({ b, made: gb.labF, size: gb.labH, vol: +(gb.labVol || 0).toFixed(2), p: [r.body[o], r.body[o + 1], r.body[o + 2]].map(x => +x.toFixed(2)), v: [r.vel[vo], r.vel[vo + 1], r.vel[vo + 2]].map(x => +x.toFixed(3)), w: r.ang ? [r.ang[vo], r.ang[vo + 1], r.ang[vo + 2]].map(x => +x.toFixed(3)) : null }); }
    const cells = new Map();
    if (pr) for (const i of pr.list) { if (pr.data[i * 12 + 10] !== 1) continue; const k = [0, 1, 2].map(c => Math.round(pr.data[i * 12 + c] / 2) * 2).join(','); const e = cells.get(k) || { n: 0, v: 0 }; e.n++; e.v = Math.max(e.v, Math.hypot(pr.data[i * 12 + 3], pr.data[i * 12 + 4], pr.data[i * 12 + 5])); cells.set(k, e); }
    const plist = []; if (pr) for (const i of pr.list) { if (pr.data[i * 12 + 10] !== 1 || plist.length >= 12) continue; const o = i * 12, inf = pInfo[i]; plist.push({ i, p: [pr.data[o], pr.data[o + 1], pr.data[o + 2]].map(x => +x.toFixed(3)), v: [pr.data[o + 3], pr.data[o + 4], pr.data[o + 5]].map(x => +x.toFixed(3)), r: +pr.data[o + 11].toFixed(3), from: inf && inf.src, made: inf && inf.f }); }
    // (and their whole records, from the GPU: position r, velocity state, rotation, sub-step start + rest timer,
    // spin + material, corrected position + touching, timer anchor + crushing)
    for (const e of plist.slice(0, 6)) gpu.read(gpu.prtBuf, 112, e.i * 112).then(a => { e.rec = Array.from(new Float32Array(a.buffer || a)).map(x => +x.toFixed(4)); }).catch(() => {});
    L.awakeEnd = { late_calls: L.lateCalls || {}, particles: plist, odd: (L.awakeOdd || []).slice(0, 20), odd_n: (L.awakeOdd || []).length, bodies: list.slice(0, 20), particle_cells: [...cells.entries()].sort((a, b) => b[1].n - a[1].n).slice(0, 12).map(([k, e]) => ({ at: k, n: e.n, vmax: +e.v.toFixed(2) })) };
  }
  // the clean-up distance: how far past it anything still in the world is from where it was first seen
  if (farDist < Infinity) {
    const L = LAB; let worst = 0;
    // (less what its speed carries it over the read-backs' lag, 3 frames: it goes at the next read-back)
    const lag = 3 / 60;
    for (const [b, f] of farO) { const gb = gpu.bodies[b - 1]; if (!gb || gb.dead || !r.body || (b - 1) * 7 + 2 >= r.body.length) continue; const o = (b - 1) * 7, vo = (b - 1) * 3; worst = Math.max(worst, Math.hypot(r.body[o] - f.p[0], r.body[o + 1] - f.p[1], r.body[o + 2] - f.p[2]) - farDist - lag * Math.hypot(r.vel[vo], r.vel[vo + 1], r.vel[vo + 2])); }
    if (pr) for (const i of pr.list) { const inf = pInfo[i]; if (inf && inf.o) worst = Math.max(worst, Math.hypot(pr.data[i * 12] - inf.o[0], pr.data[i * 12 + 1] - inf.o[1], pr.data[i * 12 + 2] - inf.o[2]) - farDist - lag * Math.hypot(pr.data[i * 12 + 3], pr.data[i * 12 + 4], pr.data[i * 12 + 5])); }
    L.farOver = Math.max(L.farOver || 0, worst);
  }
  const L = LAB, f = L.f, nb = r.body.length / 7;
  const quiet = f > L.quietFrom;   // nothing pushed from outside lately (blast, ball, cut)
  let bad = 0, below = 0, gain = 0, ekPrev = 0, slack = 0;
  const now = new Map();
  const at = r.bodiesAt || gpu.bodies;
  for (let b = 2; b <= nb; b++) {
    const gb = at[b - 1]; if (!gb || gb.dead) continue;
    const o = (b - 1) * 7, y = r.body[o + 1], vo = (b - 1) * 3;
    const vx = r.vel[vo], vy = r.vel[vo + 1], vz = r.vel[vo + 2];
    if (![r.body[o], y, r.body[o + 2], vx, vy, vz].every(Number.isFinite)) { bad++; continue; }
    // (a spare body parked for a break on the GPU: not there yet; once placed, it counts as new)
    if (gb.spare !== undefined && y < -500) continue;
    if (y < -1) { below++; labEvent('below_ground_body', f, { body: b, y: +y.toFixed(2), size: gb.labH, made_at: gb.labF }); }
    // (a ball's track: frame, where it is, how fast)
    if (gb.labBall) { const sp0 = Math.hypot(vx, vy, vz); if (!gb.labDir && sp0 > 1) gb.labDir = [vx / sp0, vy / sp0, vz / sp0]; const bd = gb.labDir || [0, 0, 0];
      // (and its speed along the way it was sent: negative is back towards where it came from)
      (L.ballTrack = L.ballTrack || []).push([f, +r.body[o].toFixed(2), +y.toFixed(2), +r.body[o + 2].toFixed(2), +sp0.toFixed(2), +(vx * bd[0] + vy * bd[1] + vz * bd[2]).toFixed(2), b]); }
    // (its mass: its volume at the density it was made with on the GPU; the ball's own. Until 15 Nov every volume
    // as concrete, and the ball left out: a 383 kg timber wardrobe pushed by the ball at 37 m/s showed as 6.2 MJ
    // gained from nothing, of the ball's 0.9 MJ)
    const m = gb.labBall ? gb.mass : (gb.labVol || 0) * (gb.labRho || 2400), ek = 0.5 * m * (vx * vx + vy * vy + vz * vz), e = ek + m * 9.81 * y;
    now.set(b, { gb, vy, e, ek, v: [vx, vy, vz], y, vol: gb.labVol || 0, cut: gb.labCut, m, p: [r.body[o], y, r.body[o + 2]], w: r.ang ? [r.ang[vo], r.ang[vo + 1], r.ang[vo + 2]] : [0, 0, 0] });
    // the same body as at the last read-back (a reused slot holds another one)
    const was = L.prevBodies && L.prevBodies.get(b);
    // (nor a body cut on the GPU since, or a few frames before: the poses read back lag the step)
    if (!was || was.gb !== gb || was.cut !== gb.labCut || (gb.labCut !== undefined && f - gb.labCut <= 3)) continue;
    gain += e - was.e; ekPrev += was.ek;
    // (what a body may gain without it being energy from nothing: its weight over 2 mm. The solver holds a
    // resting body's place to millimetres, not to nothing, and for the upper part of a collapsed frame, 1,673 t,
    // 1.2 mm of height is 20 kJ: it showed as "20,193 J gained from 1 J" at frame 579 of the lab's collapse, the
    // part lying still on its heap. For a 50 kg piece the same 2 mm is 1 J, and a launch is still a launch.)
    if (e > was.e) slack += Math.min(e - was.e, m * 9.81 * 0.002);
    if (e - was.e > 1000) (L.gainers = L.gainers || []).push({ body: b, gain_J: Math.round(e - was.e), size: gb.labH, made_at: gb.labF, y: [+was.y.toFixed(2), +y.toFixed(2)], vy: [+was.vy.toFixed(2), +vy.toFixed(2)], vol: +(gb.labVol || 0).toFixed(2), vol_was: +(was.vol || 0).toFixed(2), asleep: r.sleeping ? r.sleeping[b - 1] : null });
    if (quiet && !gb.labBall) {
      // A rebound: up afterwards at over half the speed it met what it hit at. That speed is not its centre's
      // fall alone (until 18 Nov it was): a piece that tumbles meets things with its edge, faster by its spin
      // times its reach, and what it lands on may be coming up to meet it. Lab collapse, frame 69: a 2.7 t
      // fragment falling at 2.27 m/s and turning at 1.7 rad/s met a piece on the ground with a corner at 4.97 m/s
      // and left it at 1.44 m/s up, turning faster, with less energy than it came with: counted as a rebound of
      // 63%. Lab complex, frame 49: chairs falling at 3.3 m/s onto the edge of a 110 t floor that was rocking
      // on its stumps, that edge coming up at 1.5 m/s; they left at 2.0 m/s, 0.4 faster than the floor: 60%.
      // (labMeet: its own spin x its reach, and the upward speed, where it is, of the fastest heavier body
      // that reaches it, at this read-back or the last)
      const meet = was.vy < -2 && vy > 0.5 * -was.vy ? -was.vy + labMeet(b, was, now.get(b), L.prevBodies, now) : Infinity;
      if (was.vy < -2 && vy > 0.5 * meet) { L.bounces++; L.bounceMax = Math.max(L.bounceMax, vy / -was.vy); labEvent('bounce', f, { body: b, vy_before: +was.vy.toFixed(2), vy_after: +vy.toFixed(2), met_at: +Math.min(meet, 999).toFixed(2), y: +y.toFixed(2), size: gb.labH, made_at: gb.labF, pushed_by_page: gb.labSetV || null }); }
      if (vy > 5 && vy > was.vy + 2) { L.launches++; L.launchMax = Math.max(L.launchMax, vy); labEvent('launch', f, { body: b, vy_before: +was.vy.toFixed(2), vy: +vy.toFixed(2), y: +y.toFixed(2), size: gb.labH, made_at: gb.labF, pushed_by_page: gb.labSetV || null }); }
    }
  }
  if (pr) for (let i = 0; i < pr.n; i++) { const st = pr.data[i * 12 + 10]; if (st === 1 || st === 2) { if (!Number.isFinite(pr.data[i * 12 + 1])) bad++; else if (pr.data[i * 12 + 1] < -0.5) below++; } }
  L.bad = Math.max(L.bad, bad); L.below = Math.max(L.below, below);
  if (bad) labEvent('nan', f, { count: bad });
  if (restOnFloors > (L.restMax || 0)) { L.restMax = restOnFloors; L.restSplit = [Math.round(restBodies / 9810), Math.round(restParts / 9810)]; } L.cracked = crackedJoints;
  // the energy (kinetic + height) of the bodies present at both read-backs can only fall, unless pushed
  // (energy the page gave parts it cut free is allowed for, used up as it shows and fading over ~30 read-backs)
  const given = L.given || 0; L.given = Math.max(0, given - Math.max(0, gain)) * 0.97;
  if (quiet && L.prevBodies && gain > 0.05 * ekPrev + 2000 + given + slack) { L.energyGains++; L.energyGainMax = Math.max(L.energyGainMax, ekPrev > 0 ? gain / ekPrev : 0); labEvent('energy_gain', f, { gain_J: Math.round(gain), kinetic_before_J: Math.round(ekPrev), most: (L.gainers || []).sort((a, b) => b.gain_J - a.gain_J).slice(0, 3) }); }
  L.gainers = [];
  // Crumbs thrown (18 Nov). Rubble has no restitution, and the engine lets nothing leave a squeeze faster than
  // what squeezes it plus the 2 m/s at which touching things part (docs/GPU.md). A crumb going UP at over 10 m/s
  // (a fall of 5 m) AND over 2.5 m/s faster than any point of any body that reaches it, at this read-back or the
  // one before, has been thrown by the solver. (The first limit was 10 m/s alone, which is less than the engine's
  // own rule allows under a part landing at 8.9 m/s, as collapse's upper part does: docs/LAB.md, 18 November.)
  { const pq = snapCur && snapCur.parts;
    if (quiet && pq && pq.data && pq.list) for (const i of pq.list) { const o = i * 12; if (pq.data[o + 10] !== 1 || !(pq.data[o + 4] > 10)) continue;
      if ((L.pThrown = L.pThrown || new Set()).has(i)) continue;
      const cp = [pq.data[o], pq.data[o + 1], pq.data[o + 2]]; let near = 0;
      for (const set of [L.prevBodies, now]) for (const [ob, x] of set || []) { if (!x.p) continue;
        const R = x.reach === undefined ? (x.reach = labReach(ob)) : x.reach;
        if (Math.hypot(cp[0] - x.p[0], cp[1] - x.p[1], cp[2] - x.p[2]) > R + pq.data[o + 11] + 0.5) continue;
        near = Math.max(near, Math.hypot(...x.v) + Math.hypot(...x.w) * R); }
      if (!(pq.data[o + 4] > near + 2.5)) continue;
      L.pThrown.add(i);
      const inf = pInfo[i]; labEvent('crumb_thrown', f, { i, p: cp.map(x => +x.toFixed(2)), v: [pq.data[o + 3], pq.data[o + 4], pq.data[o + 5]].map(x => +x.toFixed(2)), r: +pq.data[o + 11].toFixed(3), pushed_at: +near.toFixed(2), from: inf && inf.src, made: inf && inf.f }); } }
  L.prevBodies = now;
  // flight recorder: every body at the last 4 read-backs; the first 2 bursts (a launch or a gain over 1 MJ while
  // quiet) keep it, with the impact events of those frames, so a rare burst can be read from any full run
  const snap = { f, bodies: [] };
  for (const [b, x] of now) { const o = (b - 1) * 7, wo = (b - 1) * 3; snap.bodies.push({ b, made: x.gb.labF, size: x.gb.labH, p: [r.body[o], r.body[o + 1], r.body[o + 2]].map(z => +z.toFixed(3)), v: x.v.map(z => +z.toFixed(2)), w: r.ang ? [r.ang[wo], r.ang[wo + 1], r.ang[wo + 2]].map(z => +z.toFixed(2)) : null, asleep: r.sleeping ? r.sleeping[b - 1] : null }); }
  L.ring = (L.ring || []).concat([snap]).slice(-4);
  const burst = quiet && ((L.launches || 0) > (L.launchesSeen || 0) || gain > 1e6 + 0.05 * ekPrev + given);
  L.launchesSeen = L.launches;
  if (burst && (L.bursts = L.bursts || []).length < 2) L.bursts.push({ f, gain_J: Math.round(gain), ring: L.ring.slice(), impacts: (L.recentImpacts || []).slice() });
  // optional trace: every body made in the window, and the biggest one, at each read-back within it
  const tr = L.traceWin;
  if (tr && f >= tr[0] && f <= tr[1]) {
    let big = 0, bigV = 0;
    for (let b = 2; b <= nb; b++) { const gb = at[b - 1]; if (gb && !gb.dead && (gb.labVol || 0) > bigV) { bigV = gb.labVol; big = b; } }
    for (let b = 2; b <= nb; b++) {
      const gb = at[b - 1]; if (!gb || gb.dead) continue;
      if (!(b === big || (gb.labF >= tr[0] - 15 && gb.labF <= tr[1]))) continue;
      if (L.trace.length > 4000) break;
      const o = (b - 1) * 7, vo = (b - 1) * 3, wo = vo;
      L.trace.push({ f, rf: r.frame, b, made: gb.labF, vol: +(gb.labVol || 0).toFixed(3), size: gb.labH, p: [r.body[o], r.body[o + 1], r.body[o + 2]].map(x => +x.toFixed(3)), v: [r.vel[vo], r.vel[vo + 1], r.vel[vo + 2]].map(x => +x.toFixed(2)), w: r.ang ? [r.ang[wo], r.ang[wo + 1], r.ang[wo + 2]].map(x => +x.toFixed(2)) : null, asleep: r.sleeping ? r.sleeping[b - 1] : null });
    }
  }
}
async function labAct(act, a) {
  const L = LAB; L.injectedAt = L.f;
  // the energy and bounce checks wait for what is pushed from outside to play out
  L.quietFrom = Math.max(L.quietFrom, L.f + (act === 'ball' || act === 'qball' ? 120 : 30));
  if (act === 'cut') {
    // the benchmark's cut: the ground storey's columns and walls blown out, before the physics runs
    physicsPaused = true;
    // (a: in the city, the lots to cut, by index; all of them when absent)
    const at = lots().filter((_, k) => !a || a.includes(k));
    for (const P of at) for (let x = -10; x <= 10.1; x += 4) for (let z = -10; z <= 10.1; z += 4) jobs.blast.push({ op: 'blast', x: P.x + x, y: 1.8, z: P.z + z, r: 2.5, ctx: { d: [0, 0, 1], t0: clock() } });
    pump();
    const T0 = L.T;
    while (L.jobq.length) { const j = L.jobq.shift(); finish(runInline(j.job)); labParticlesMade(); }
    L.staticT = labTruthStatic(); labTruth();
    L.lostBy.tool = (L.lostBy.tool || 0) + Math.max(0, T0 - L.T);
    physicsPaused = false; lastActive = clock();
  } else if (act === 'ball') {
    // (position, yaw, pitch; then its speed, 120 m/s if not given, and its radius, 0.4 m)
    const [x, y, z, yw, pt] = a, sp = a[5] || 120, rad = a[6] || 0.4, d = [Math.sin(yw) * Math.cos(pt), Math.sin(pt), Math.cos(yw) * Math.cos(pt)];
    labBall([x, y, z], rad, [d[0] * sp, d[1] * sp, d[2] * sp]);
  } else if (act === 'qball') {
    // (a ball in a building of the Quarter's own frame: building, from x y z, along dx dy dz, speed, radius)
    const [name, x, y, z, dx, dy, dz, sp, rad] = a, dl = Math.hypot(dx, dy, dz), d = quarterDir(name, dx / dl, dy / dl, dz / dl);
    labBall(quarterAt(name, x, y, z), rad || 0.25, d.map(q => q * (sp || 60)));
  } else if (act === 'qblast') {
    // (a shot in a building's own frame: building, from x y z, along dx dy dz, radius)
    const [name, x, y, z, dx, dy, dz, r] = a, o = quarterAt(name, x, y, z), d = quarterDir(name, dx, dy, dz);
    setRadius(r); doBlast(new THREE.Vector3(...o), new THREE.Vector3(...d).normalize());
  } else if (act === 'qcut') {
    // (a building's ground storey blown out over x0..x1, z0..z1 of its own frame, as `cut` does a lot's)
    const [name, x0, x1, z0, z1] = a;
    physicsPaused = true;
    for (let x = x0; x <= x1 + 0.1; x += 4) for (let z = z0; z <= z1 + 0.1; z += 4) { const q = quarterAt(name, x, 1.8, z); jobs.blast.push({ op: 'blast', x: q[0], y: q[1], z: q[2], r: 2.5, ctx: { d: [0, 0, 1], t0: clock() } }); }
    pump();
    const T0 = L.T;
    while (L.jobq.length) { const j = L.jobq.shift(); finish(runInline(j.job)); labParticlesMade(); }
    L.staticT = labTruthStatic(); labTruth();
    L.lostBy.tool = (L.lostBy.tool || 0) + Math.max(0, T0 - L.T);
    physicsPaused = false; lastActive = clock();
  } else if (act === 'blast') {
    const [ox, oy, oz, dx, dy, dz, r] = a;
    setRadius(r); doBlast(new THREE.Vector3(ox, oy, oz), new THREE.Vector3(dx, dy, dz).normalize());
  }
}
// the lab's ball: the game's (throwBall), marked so its track is kept and it is not counted as the building's
function labBall(c, rad, v) {
  const L = LAB;
  L.ballNext = true;
  const b = throwBall(c, rad, v, { oldMass: !!window.__oldBall });
  const gb = b ? gpu.bodies[b - 1] : null;
  L.ballMass = gb && gb.mass; L.ballR = rad;
}
function labReport(wallS) {
  const L = LAB, S = L.spec, se = L.series, last = se[se.length - 1] || {};
  const med = (a) => { const s = a.filter(Number.isFinite).sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : 0; };
  const p95 = (a) => { const s = a.filter(Number.isFinite).sort((x, y) => x - y); return s.length ? s[Math.min(s.length - 1, Math.floor(s.length * 0.95))] : 0; };
  // (the roof before the fall: the highest in the first half second, once the read-backs show the cut-off part)
  const roof0 = Math.max(0, ...se.filter(x => x.f <= 30).map(x => x.roof));
  const checks = [
    ['drawn volume below what exists (frames)', L.holeFrames, 0, 'holes, flicker or missing parts'],
    ['drawn volume above what exists (frames)', L.dupFrames, 0, 'a part drawn twice (building and debris)'],
    ['pieces that disappear and come back', L.flickers, 0, 'flicker of single pieces'],
    ['pieces moving further than their speed allows', L.teleports, 0, 'jumps'],
    ['read-back check: vanished / jumped', flick.vanished + flick.jumps, 0, ''],
    ['invalid poses (NaN)', L.bad, 0, ''],
    ['steps that lost contacts or pairs (capacity)', gpu.lost || 0, 0, 'a lost ground pair lets a piece fall through'],
    ['pieces below the ground', L.below, 0, ''],
    ['body slots freed twice', gpu.doubleFree || 0, 0, 'a slot given to two bodies at once'],
    // a big falling part broken on landing crumbles where it is crushed, not all over: the share of its volume that
    // leaves as crumbs and particles (7 November: a brick building landing on its ground storey burst into loose
    // bricks up to its roof, the playground's "particle waterfall")
    ['crumbled share of a big part when it breaks (worst)', +Math.max(0, ...(L.bigBreaks || []).map(b => b[2])).toFixed(2), 0.25, 'parts bursting into crumbs'],
    ['rebounds over half the impact speed (while quiet)', L.bounces, 0, 'unnatural bouncing'],
    ['pieces thrown up faster than 5 m/s (while quiet)', L.launches, 0, 'launches'],
    ['crumbs thrown up faster than 10 m/s (while quiet)', (L.pThrown || { size: 0 }).size, 0, 'launches of rubble: up at over 10 m/s and over 2.5 m/s faster than any body that reaches them'],
    ['energy gains over 5% (while quiet)', L.energyGains, 0, 'energy made from nothing'],
    // what the tools did not take out must stay in the world: standing, falling, as rubble or particles. Dust (fines
    // removed) is real but small; a whole floor gone as dust is a floor that vanished (7 November: 57% in collapse)
    // (what the clean-up distance removed is gone on purpose, like what the tools took)
    ['volume lost, not to the tools (% of the building)', +(100 * (L.V0 - L.T - (L.lostBy.tool || 0) - (L.lostBy.far || 0) - (L.lostBy.budget || 0)) / L.V0).toFixed(1), 2, 'parts vanishing'],
    // (and none made: a part doubled, 7 November: the mixed building's falling part doubled where the GPU had broken it)
    ['volume made (% of the building)', +Math.max(0, -100 * (L.V0 - L.T - (L.lostBy.tool || 0) - (L.lostBy.far || 0) - (L.lostBy.budget || 0)) / L.V0).toFixed(1), 0.5, 'parts doubled'],
  ].map(([name, value, limit, what]) => ({ name, value, limit, pass: value <= limit, what }));
  if (S.expect && S.expect.gradual) {
    // a building eaten away fails progressively: joints crack (and wear through) before the upper storeys come
    // down, over seconds; not all at once at one more cut, after standing unchanged
    const up0 = (se[0] || {}).upper || 0, fall = se.find(x => x.upper < 0.9 * up0), first = se.find(x => x.failed > 0);
    const before = fall ? (se.filter(x => x.f < fall.f).pop() || { cracked: 0, worn: 0 }) : last;
    const warn = first ? ((fall ? fall.f : last.f) - first.f) / 60 : 0;
    checks.push({ name: 'joints cracked before the upper storeys fall', value: before.cracked || 0, limit: '>= 50', pass: (before.cracked || 0) >= 50, what: 'cracks before the collapse' });
    checks.push({ name: 'seconds from the first failure to the fall', value: +warn.toFixed(1), limit: '>= 2', pass: warn >= 2, what: 'a gradual failure, not a trigger' });
    const lost = up0 > 0 ? 1 - (last.upper || 0) / up0 : 0;
    checks.push({ name: 'upper storeys come down (share of their volume)', value: +lost.toFixed(2), limit: `>= ${S.expect.upperLoss}`, pass: lost >= S.expect.upperLoss, what: 'eaten away, it does not stand for ever' });
    L.gradual = { fall_f: fall ? fall.f : null, first_failure_f: first ? first.f : null, worn_before: before.worn || 0, worn_total: last.worn || 0 };
  }
  // (with the default distance, 100 m, nothing in these scenarios flies that far: what goes was measured from the
  // wrong place. 8 November: spare bodies seen parked 1 km under the ground, 1,018 m3 of the mixed building went)
  checks.push({ name: 'spare bodies awake while parked (most at once)', value: L.parkedAwake || 0, limit: 0, pass: !(L.parkedAwake > 0), what: 'the world never at rest' });
  if (S.far == null) checks.push({ name: 'removed past the clean-up distance (% of the building)', value: +(100 * (L.lostBy.far || 0) / L.V0).toFixed(1), limit: 0.5, pass: 100 * (L.lostBy.far || 0) / L.V0 <= 0.5, what: 'parts removed that did not fly that far' });
  if (S.expect && S.expect.debrisCap) {
    // the debris budget: never more pieces and particles at once than it allows (and one read-back's worth that
    // had not been counted yet), and the oldest have gone to keep it
    const cap = S.expect.debrisCap, peak = L.debrisPeak || 0;
    checks.push({ name: 'debris kept at once, at most', value: peak, limit: cap, pass: peak <= cap, what: 'pieces and particles together, after each read-back' });
    checks.push({ name: 'debris removed over the budget', value: debrisCapped, limit: '> 0', pass: debrisCapped > 0, what: 'the scenario makes more than the budget' });
    checks.push({ name: 'debris removed: the smallest first, the oldest of those', value: L.debrisOrder === false ? 'no' : 'yes', limit: 'yes', pass: L.debrisOrder !== false, what: 'nothing is left that is of a smaller class than what was removed, or of its class and older' });
    // (what the budget costs the picture: the heaviest piece it took, and how much of the building; limits in
    // `expect`, from the scenario's first runs, docs/LAB.md 14 November)
    const heavy = Math.round(L.debrisHeaviest || 0), share = +(100 * (L.lostBy.budget || 0) / L.V0).toFixed(1);
    if (S.expect.debrisHeaviest != null) checks.push({ name: 'heaviest piece the budget removed (kg)', value: heavy, limit: S.expect.debrisHeaviest, pass: heavy <= S.expect.debrisHeaviest, what: 'big pieces stay while crumbs and small pieces can go' });
    if (S.expect.debrisShare != null) checks.push({ name: 'removed by the budget (% of the building)', value: share, limit: S.expect.debrisShare, pass: share <= S.expect.debrisShare, what: 'the budget takes crumbs, not the building' });
    labEvent('budget', L.f, { removed: debrisCapped, particles: L.budgetParts || 0, bodies: L.budgetBodies || 0, heaviest_kg: heavy, largest_m3: +(L.budgetBodyVol || 0).toFixed(3), share_pct: share });
  }
  if (S.expect && S.expect.ballStops) {
    // a ball sent into a building is stopped by it: it ends at rest inside, above the ground storey, and what it
    // lost in its first second of breaking through is most of its speed
    const E = S.expect.ballStops, tr = L.ballTrack || [], end = tr[tr.length - 1] || [0, 0, 0, 0, 99];
    const v0 = tr.length ? tr[0][4] : 0;
    const hit = tr.findIndex(t => t[4] < 0.95 * v0), after = hit >= 0 ? tr.find(t => t[0] >= tr[hit][0] + 30) : null;
    const levels = []; for (const t of tr) { if (t[4] < 1 && (!levels.length || Math.abs(levels[levels.length - 1] - t[2]) > 0.5)) levels.push(t[2]); }
    checks.push({ name: 'ball: speed at the end (m/s)', value: end[4], limit: 1, pass: end[4] <= 1, what: 'the building stops it' });
    checks.push({ name: 'ball: height where it stops (m)', value: end[2], limit: `${E.above} to ${E.top}`, pass: end[2] >= E.above && end[2] <= E.top, what: 'it stops inside the building, not on the ground under it' });
    if (E.box) checks.push({ name: 'ball: where it stops, x and z (m)', value: `${end[1]}, ${end[3]}`, limit: `x ${E.box[0]} to ${E.box[1]}, z ${E.box[2]} to ${E.box[3]}`, pass: end[1] >= E.box[0] && end[1] <= E.box[1] && end[3] >= E.box[2] && end[3] <= E.box[3], what: 'inside the building\'s plan' });
    checks.push({ name: 'ball: speed half a second after it first hits, of its speed before', value: after && v0 ? +(after[4] / v0).toFixed(2) : 'no hit', limit: '<= 0.5', pass: !!after && after[4] <= 0.5 * v0, what: 'the first of the building takes most of it' });
    L.ballSummary = { v0, first_hit_frame: hit >= 0 ? tr[hit][0] : null, end, rests_at_m: levels, track: tr.filter((t, i) => i % 6 === 0) };
  }
  // Debris weighs what the structure weighed (15 Nov). Before: a steel member that fell weighed the solid box it is
  // drawn as, 6.25 times what the engine weighs it at standing (a rolled section, structure::STEEL_AREA). The limit
  // 2%: a piece's corners are kept in single precision and its cells' volumes are the hull library's.
  if (L.massChecked > 0) checks.push({ name: 'a piece\'s mass on the GPU over the engine\'s (worst)', value: +L.massWorst.toFixed(3), limit: '0.98 to 1.02', pass: Math.abs(L.massWorst - 1) <= 0.02, what: `debris weighs what it weighed standing (${L.massChecked} pieces${L.massWorstOf ? `; worst ${L.massWorstOf.gpu_kg} kg for ${L.massWorstOf.engine_kg} kg` : ''})` });
  if (L.ballTrack && L.ballTrack.length && !L.ballSummary) { const tr = L.ballTrack; L.ballSummary = { v0: tr[0][4], end: tr[tr.length - 1], track: tr.filter((t, i) => i % 6 === 0) }; }
  if (L.ballSummary) {
    const tr = L.ballTrack, back = Math.min(...tr.map(t => t[5] ?? 0)), sphere = 7850 * 4 / 3 * Math.PI * (L.ballR || 0) ** 3;
    if (L.smashed) L.ballSummary.smashed = L.smashed;
    Object.assign(L.ballSummary, { mass_kg: Math.round(L.ballMass || 0), sphere_kg: Math.round(sphere), fastest_back_m_s: +(-back).toFixed(2), punches: { ...punchStat } });
    // a steel ball is a steel sphere (15 Nov: its four overlapping boxes weighed 3.2 spheres)
    checks.push({ name: 'ball: its mass, of a steel sphere of its size', value: +((L.ballMass || 0) / sphere).toFixed(3), limit: '0.99 to 1.01', pass: Math.abs((L.ballMass || 0) / sphere - 1) <= 0.01, what: 'the ball weighs what a ball weighs' });
    // it does not come back off what it breaks (15 Nov, `anticipate`; lab tallball before: back at 12 m/s off a
    // 0.25 m slab it went through four frames later). The limit: a tenth of the speed it was sent at.
    // and it is not stopped dead by what it had not paid for (18 Nov: lab qballglass passed the check above with the
    // ball at 2 m/s of 58 in one step, on the rim of a hole made where it had been going before two chairs put it
    // 0.11 m off; a wall, 20,839 N s). Counted: read-backs in which a ball faster than 10 m/s lost over 70% of its
    // speed with no thing arrived at in the page's log about then. (A wardrobe of 383 kg takes 51%.)
    if (S.expect && S.expect.ballOn) {
      // (the log's frames are the GPU's, 4 on from the lab's; a stop is seated up to a second after it is arrived at)
      const arr = (punchStat.log || []).filter(l => l.ev === 'arrives'), near = (f) => arr.some(l => l.f >= f - 8 && l.f <= f + 12) || arr.some(l => l.stop && l.f >= f - 70 && l.f <= f + 12);
      let dead = 0, first = null;
      // (each ball by itself: its track is the rows with its body, the 7th number)
      for (const id of new Set(tr.map(t => t[6]))) { const one = tr.filter(t => t[6] === id);
        for (let k = 1; k < one.length; k++) if (one[k - 1][4] >= 10 && one[k][4] < 0.3 * one[k - 1][4] && !near(one[k][0])) { dead++; first = first || [one[k][0], one[k - 1][4], one[k][4]]; } }
      L.ballSummary.stopped_dead = first;
      checks.push({ name: 'ball: stopped dead by what it had not paid for (read-backs)', value: dead, limit: 0, pass: dead === 0, what: `it loses its speed to what it breaks, as the page's log has it${first ? `; first at frame ${first[0]}: ${first[1]} to ${first[2]} m/s` : ''}` });
    }
    if (S.expect && S.expect.ballOn) checks.push({ name: 'ball: fastest it came back the way it was sent (m/s)', value: +(-back).toFixed(2), limit: +(0.1 * tr[0][4]).toFixed(1), pass: -back <= 0.1 * tr[0][4], what: 'it goes on through what it breaks, or stops in it' });
  }
  if (S.expect && S.expect.far) {
    // past the clean-up distance things go at the next read-back (what their speed carries them over the lag is
    // allowed for, see labOnRead)
    checks.push({ name: 'removed past the clean-up distance (bodies + particles)', value: (L.farBodies || 0) + (L.farParts || 0), limit: '>= 1', pass: (L.farBodies || 0) + (L.farParts || 0) >= 1, what: 'the clean-up distance works' });
    checks.push({ name: 'farthest past the clean-up distance, still in the world (m)', value: +(L.farOver || 0).toFixed(2), limit: 1, pass: (L.farOver || 0) <= 1, what: 'nothing stays far out' });
  }
  // (and no further than the scenario says: the building comes down by its cut storey, it does not turn to crumbs.
  // Lab timber, 14 Nov: the pinch test crushed the whole upper building, one body, 0.3 s into its fall; the roof
  // fell 13.5 m of 15, which the lower limit alone took for a pass.)
  if (S.expect && S.expect.roofDropMax) checks.push({ name: 'roof fall, at most (m)', value: +(roof0 - (last.roof || 0)).toFixed(2), limit: `<= ${S.expect.roofDropMax}`, pass: roof0 - (last.roof || 0) <= S.expect.roofDropMax, what: 'the upper storeys come down as a building, not as dust' });
  if (S.expect && S.expect.roofDrop) checks.push({ name: 'roof fall (m)', value: +(roof0 - (last.roof || 0)).toFixed(2), limit: `>= ${S.expect.roofDrop}`, pass: roof0 - (last.roof || 0) >= S.expect.roofDrop, what: 'the cut storey brings the building down' });
  // (a building left alone stands: nothing of it fails, its roof stays where it is)
  if (S.expect && S.expect.stand) checks.push({ name: 'joints failed with nothing done to it', value: failedJoints, limit: 0, pass: failedJoints === 0, what: 'a building left alone stands' });
  // Settled: everything asleep at the end. Or: every body was asleep, and what is awake at the end was woken by
  // the standing structure losing joints in the last second (and the bodies had been asleep within 2 s before
  // that): the remains wear under the heap and give way, minutes after a collapse in a real one too, and that is
  // the structure's doing (docs/STRUCTURE.md, wear), not a heap that will not rest. Lab collapse, 14 Nov: every
  // body asleep from 7.8 s; at 8.6, 9.0 and 9.9 s joints wear through under the heap (2, 1 and 2) and wake it.
  if (S.expect && S.expect.settle) {
    const woken = L.settledAt == null && L.structFailAt !== undefined && L.f - L.structFailAt <= 60 && L.bodiesQuietAt !== undefined && L.bodiesQuietAt >= L.structFailAt - 120;
    checks.push({ name: 'settled (s)', value: L.settledAt != null ? +(L.settledAt / 60).toFixed(1) : woken ? `${(L.bodiesQuietAt / 60).toFixed(1)}, woken at ${(L.structFailAt / 60).toFixed(1)} by the structure` : 'no', limit: `by ${(S.frames / 60).toFixed(0)}`, pass: L.settledAt != null || woken, what: 'everything asleep before the end with no engine job changing anything since, or woken in its last second by the structure giving way under it' });
  }
  const gpuMs = se.map(x => x.gpu), jsMs = L.jsMs;
  return {
    scenario: L.name, shortened: S.frames !== LAB_SCENARIOS[L.name].frames, title: S.title, kind: KIND_NAMES[S.kind], storeys: S.storeys, frames: S.frames,
    readLag: L.readLag, jobLag: L.jobLag, settings: { ...setting }, ua: navigator.userAgent, adapter: L.adapter, date: new Date().toISOString(),
    pass: checks.every(c => c.pass), checks, colConflicts: L.colConflicts || 0, colConflictAt: L.colConflictAt || null, digest: L.digest, dumpRaw: L.dumpRaw ? { ...L.dumpRaw, colRows: L.colRows, allRows: L.allRows } : (L.colRows ? { colRows: L.colRows, allRows: L.allRows } : null),
    physics: { ball: L.ballSummary, roof_m: [+roof0.toFixed(2), +(last.roof || 0).toFixed(2)], dust_of_building: +(1 - (L.T) / L.V0).toFixed(3), particles_by: { ...partsBy }, big_breaks: L.bigBreaks, made: L.made, gains: L.gains, spill_why: { crumbs: E.spill_why(0), fragments: E.spill_why(1), crushed_storey: E.spill_why(2), static: E.spill_why(3), fragment_bricks_by_mat: [0, 1, 2, 3, 4, 5, 6, 7].map(k => E.spill_mat(k)), single_brick_fragments: E.spill_mat(8) }, over_breaks: L.overBreaks, dbg_breaks: L.dbgBreaks, lost_m3: Object.fromEntries(Object.entries({ ...L.lostBy, particles_crushed: L.lostParticles || 0, building: L.V0 }).map(([k, v]) => [k, +v.toFixed(1)])), particles_made: partsMade, particles_crushed: partsCrushed, bodies_crushed: crushedBodies, removed_far: { bodies: L.farBodies || 0, particles: L.farParts || 0, m3: +((L.lostBy.far || 0)).toFixed(2) }, breaks, failed_joints: failedJoints, settled_s: L.settledAt == null ? null : +(L.settledAt / 60).toFixed(2), bounce_max: +L.bounceMax.toFixed(2), launch_max: +L.launchMax.toFixed(2), hole_max_m3: +L.holeMax.toFixed(3), dup_max_m3: +L.dupMax.toFixed(3), energy_given_by_cuts_J: Math.round(L.givenTotal || 0), rubble_on_floors_max_t: +((L.restMax || 0) / 9810).toFixed(1), rubble_sent_bodies_particles_t: L.restSplit || null, joints_cracked: L.cracked || 0, gpu_breaks: gpuSnaps, gradual: L.gradual || undefined, awake_at_end: L.awakeEnd || undefined, late_calls: L.lateCalls || {}, late_inputs: L.lateInputs || null, skip_state: { inputs: gpu.inputs, statsInputs: gpu.statsInputs, skipped: gpu.skipped || 0, frame: gpu.frame, realFrame: gpu.realFrame, awake: gpu.stats.awake, pActive: gpu.stats.pActive, blasts: (gpu.pBlasts || []).length, wakeReq: !!gpu.wakeReq, inherit: (gpu.pendingInherit || []).length, sleep: gpu.sleep } },
    perf: {
      wall_s: +wallS.toFixed(1), steps: stepsDone, steps_skipped: gpu.skipped || 0,
      gpu_ms: { median: +med(gpuMs).toFixed(2), p95: +p95(gpuMs).toFixed(2), max: +Math.max(0, ...gpuMs).toFixed(2) },
      gpu_stages_ms_at_peak: L.peakStages,
      js_ms: { median: +med(jsMs).toFixed(2), p95: +p95(jsMs).toFixed(2), max: +Math.max(0, ...jsMs).toFixed(2) },
      engine_ms: { jobs: engineMsLog.length, median: +med(engineMsLog.slice()).toFixed(1), p95: +p95(engineMsLog.slice()).toFixed(1), max: +Math.max(0, ...engineMsLog).toFixed(1) },
      peak: { bodies: Math.max(0, ...se.map(x => x.bodies)), awake: Math.max(0, ...se.map(x => x.awake)), contacts: Math.max(0, ...se.map(x => x.contacts)), particles: Math.max(0, ...se.map(x => x.parts)), particles_moving: Math.max(0, ...se.map(x => x.pAwake)), drawn: Math.max(0, ...se.map(x => x.drawn)) },
      work: { contact_substeps: L.work.cs, particle_substeps: L.work.ps },
    },
    partDump: L.partDump || undefined, motionLog: window.__motionLog || undefined, events: L.events, timeline: L.timeline, trace: L.trace.length ? L.trace : undefined, btrace: L.btrace || undefined, parked: globalThis.__parked || undefined, slotLog: globalThis.__slotLog || undefined, bursts: L.bursts || undefined, traceEvents: L.traceEv.length ? L.traceEv : undefined,
    series: se,
  };
}
function labSummary(rep) {
  const lines = [`${rep.pass ? 'PASS' : 'FAIL'} ${rep.scenario}: ${rep.title}`];
  for (const c of rep.checks) lines.push(`  ${c.pass ? 'ok  ' : 'FAIL'} ${c.name}: ${c.value} (limit ${c.limit})${!c.pass && rep.events ? '' : ''}`);
  const p = rep.physics, q = rep.perf;
  lines.push(`  physics: roof ${p.roof_m[0]} -> ${p.roof_m[1]} m · dust ${(p.dust_of_building * 100).toFixed(1)}% (m³ ${JSON.stringify(p.lost_m3)}) · particles from ${JSON.stringify(p.particles_by)} (spill bricks ${JSON.stringify(p.spill_why)}) · breaks ${p.breaks} · crushed ${p.bodies_crushed} · particles ${p.particles_made} (${p.particles_crushed} crushed) · settled ${p.settled_s ?? 'no'} s · energy given to parts cut free ${((p.energy_given_by_cuts_J || 0) / 1e6).toFixed(2)} MJ · rubble on floors up to ${p.rubble_on_floors_max_t} t · ${p.joints_cracked} joints cracked · ${p.gpu_breaks} broken on the GPU`);
  lines.push(`  perf: ${q.steps} steps (${q.steps_skipped} skipped) in ${q.wall_s} s · GPU ms median ${q.gpu_ms.median} / p95 ${q.gpu_ms.p95} / max ${q.gpu_ms.max} · JS ms median ${q.js_ms.median} / p95 ${q.js_ms.p95} · engine jobs ${q.engine_ms.jobs}, ms median ${q.engine_ms.median} / p95 ${q.engine_ms.p95} / max ${q.engine_ms.max} · peak awake ${q.peak.awake}, contacts ${q.peak.contacts}, particles ${q.peak.particles}`);
  if (q.gpu_stages_ms_at_peak) lines.push(`  GPU stages at the slowest step (ms): ${Object.entries(q.gpu_stages_ms_at_peak).map(([k, v]) => `${k} ${v}`).join(' · ')}`);
  for (const [k, e] of Object.entries(rep.events)) lines.push(`  events ${k}: ${e.n} · first ${JSON.stringify(e.first.slice(0, 3))}`);
  return lines.join('\n');
}
// A frame's state as one number, to compare two runs of a scenario (node game/lab/lab.cjs --twice): the poses read
// back (what is drawn), the engine's bricks and joints, the bodies and breaks. Two runs of the same build agree
// frame for frame or the lab is not deterministic.
function labDigest() {
  const H = () => { let h = 2166136261 >>> 0; return { mix(x) { h = Math.imul(h ^ (x >>> 0), 16777619) >>> 0; }, get v() { return h; } }; };
  const pose = H(), parts = H(), job = H();
  if (snapCur && snapCur.pose) { const u = new Uint32Array(snapCur.pose.buffer, snapCur.pose.byteOffset, snapCur.pose.length); for (let k = 0; k < u.length; k++) pose.mix(u[k]); }
  const pr = snapCur && snapCur.parts;
  if (pr && pr.data) { const u = new Uint32Array(pr.data.buffer, pr.data.byteOffset, pr.data.length); for (let k = 0; k < u.length; k++) parts.mix(u[k]); }
  for (const j of jobs.blast) job.mix(j.op.length);
  // (each part apart, so a difference says where it starts)
  return [pose.v, parts.v, gpu.n, breaks, dustBricks, E.bricks_alive(), E.joints_live(), gpu.stats.manifolds || 0, partsMade, gpu.stats.awake || 0, L_events(), job.v, gpu.inputs, gpu.frame, gpu.skipped || 0, gpu.colorLimit, gpu.substeps, gpu.ns, gpu.np || 0, JSON.stringify(gpu.stats).length, (gpu.pendingInherit || []).length, LAB.inHash || 0];
}
function L_events() { return (gpu.events && gpu.events.length) || 0; }
async function labRun() {
  const L = LAB, st = LABSTATE, name = st.queue[0], S = { ...LAB_SCENARIOS[name] };
  if (st.frames) S.frames = st.frames;   // (a shortened run, for quick checks of the lab itself)
  farDist = S.far ?? 100;   // (the clean-up distance: the page's default, or the scenario's)
  Object.assign(L, {
    on: true, name, spec: S, t: 0, f: 0, readLag: st.readLag ?? 2, jobLag: st.jobLag ?? 4, render: !!st.render, gates: [], jobq: [],
    draw: { n: 0, vol: 0, keys: new Float64Array(4096), pos: new Float32Array(4096 * 3), bound: new Float32Array(4096), vols: new Float32Array(4096) },
    Thist: [], holeFrames: 0, holeMax: 0, inHole: false, holesRecovered: 0, dupFrames: 0, dupMax: 0, flickers: 0, teleports: 0, goneAt: new Map(), prevDraw: null,
    bad: 0, below: 0, bounces: 0, bounceMax: 0, launches: 0, launchMax: 0, energyGains: 0, energyGainMax: 0, prevBodies: null, quietFrom: -1,
    injectedAt: -1000, settledAt: null, digest: [], timeline: [], trace: [], traceEv: [], traceWin: st.trace || null, events: {}, series: [], jsMs: [], work: (engineMsLog.length = 0, { cs: 0, ps: 0 }), peakStages: null, peakGpu: 0, volTol: 0.05,
  });
  // the defaults, whatever this browser has saved (the report lists them)
  for (const [k, d] of Object.entries(SETTINGS)) { setting[k] = d.def; d.apply(d.def); }
  setting.phys = 'high (lab defaults)'; setting.gfx = 'high (lab defaults)';
  // fixed sub-steps: adapting them to this machine's speed would make the runs differ between machines
  // (SF_PRESET: a quality preset over the defaults, as a player would pick it: a benchmark on 'low' kept bodies
  // awake far longer than the lab's 'high')
  if (window.__preset && PRESETS[window.__preset]) { const pr = PRESETS[window.__preset]; for (const k of PHYS_KEYS) { setting[k] = pr[k]; SETTINGS[k].apply(pr[k]); } setting.phys = window.__preset; }
  // (the debris budget: none in the lab, so that a scenario is the same whatever it leaves lying, unless the
  // scenario sets one, `debris`, or a preset does)
  if (!(window.__preset && PRESETS[window.__preset])) { setting.debris = LAB.spec.debris || 20000; SETTINGS.debris.apply(setting.debris); }
  setting.adaptive = 0; gpu.substeps = setting.substeps;
  simRandom = seededRandom(window.__seed || 20261104);   // (SF_SEED: another run of the same scenario, to tell a trend from one run's luck)
  // (lab.cjs --twice-view: the view on from the start, without the structure step that switching it by hand asks
  // for, which would shift every job after it by one and is not what is being compared)
  if (window.__viewOn) { setting.view = 1; viewShown = true; E.set_param(8, viewParam()); }
  if (window.__pinch && gpu) gpu.pinching = true;
  if (window.__noPinch && gpu) gpu.pinching = false;
  if (window.__noSup && gpu) gpu.support = false;   // (SF_NOSUP: particles hold no body up, to see what stands on rubble)
  if (window.__noPunch) setting.punch = 0;   // (SF_NOPUNCH: the ball as before 15 Nov, hitting rigidly; SF_OLDBALL: and at its old mass)
  if (window.__noBound && gpu) { gpu.velocityBound = false; gpu.pinching = false; }   // (as before 14 Nov: neither the bound nor the pinch test)
  if (window.__oldMasonry) E.set_param(6, 1);   // (SF_OLD_MASONRY: masonry in falling parts parts at once, for comparison)   // (SF_PINCH: crushing of pinched pieces, off by default)
  fly = true; E.player_set(-20, 8, -30); yaw = Math.PI + 0.62; pitch = -0.12;
  $('start').hidden = true; physicsPaused = false;
  $('bench').hidden = false; $('bench-title').textContent = `Lab: ${name} (${st.done.length + 1} of ${st.done.length + st.queue.length})`;
  labInstall();
  L.staticT = labTruthStatic(); labTruth(); L.V0 = L.T; L.lostBy = {}; L.farBodies = 0; L.farParts = 0; L.farOver = 0; L.awakeEnd = null; L.parkedAwake = 0; L.awakeOdd = []; L.lateCalls = {}; L.lateInputs = null; L.made = []; L.bigBreaks = []; L.Thist = [L.T]; L.staticDrawn = Object.values(chunkVol).reduce((a, b) => a + b, 0);
  try { const info = dev.adapterInfo || (dev.adapter && dev.adapter.info); L.adapter = info ? `${info.vendor} ${info.architecture} ${info.description || ''}`.trim() : ''; } catch (e) {}
  const tw = performance.now();
  let lastShow = 0;
  for (L.f = 0; L.f < S.frames; L.f++) {
    L.t += 1000 / 60;
    for (const [fr, act, args] of S.script) if (fr === L.f) await labAct(act, args);
    // engine jobs that are due (the worker's time)
    while (L.jobq.length && L.jobq[0].due <= L.f) {
      const j = L.jobq.shift(), nb0 = gpu.n, b0 = breaks, T0 = L.T, S0 = L.staticT, G0 = L.gpuVol, pm0 = partsMade, r = runInline(j.job);
      L.jobOp = j.job.op; L.jobVol = { bodiesIn: 0, bodiesOut: 0 }; finish(r); L.jobOp = ''; labParticlesMade(); L.staticT = labTruthStatic(); labTruth();
      // (volume each kind of job took out of the world: a blast is the player's tool; the rest is lost)
      { const lost = T0 - L.T; if (lost < -1 && (L.gains = L.gains || []).length < 20) L.gains.push({ f: L.f, op: j.job.op, gain: +(-lost).toFixed(2), static: +(L.staticT - S0).toFixed(2), gpu: +(L.gpuVol - G0).toFixed(2), bodies_added: +L.jobVol.bodiesIn.toFixed(2), bodies_removed: +L.jobVol.bodiesOut.toFixed(2), particles: partsMade - pm0, dup_bricks: E.dup_bricks() });
        if (Math.abs(lost) > 1e-6) { const k = (j.job.op === 'blast' || j.job.op === 'pblast') ? 'tool' : j.job.op; L.lostBy[k] = (L.lostBy[k] || 0) + lost; } }   // (signed: a job can add what another takes away)
      // what the engine did, frame by frame (to explain an event)
      if (j.job.op === 'structure' && r.n > 0) L.structFailAt = L.f;   // (the last frame the standing structure lost joints)
      // (a job that changed the world unsettles it; one that found nothing to do does not: see quietNow)
      if (r.n > 0 || gpu.n !== nb0 || breaks !== b0 || (r.out && r.out.length && r.out[0] > 0) || r.moving) L.settledAt = null;
      if (L.timeline.length < 400) L.timeline.push({ f: L.f, op: j.job.op, n: r.n, ms: +(r.ms || 0).toFixed(1), dyn: j.job.op === 'structure' && r.dynIts !== undefined ? { its: r.dynIts, moving: !!r.moving, left: r.dynLeft, cut: r.dynCut, capped: r.dynCapped, mm_s: +(1000 * (r.dynSpeed || 0)).toFixed(1) } : undefined, sent: j.due - L.jobLag, new_bodies: gpu.n - nb0, breaks: breaks - b0, static_updates: r.out && r.out.length ? r.out[0] : 0 });
    }
    // read-backs that are due
    while (L.gates.length && L.gates[0].due <= L.f) { const g = L.gates.shift(); g.res(await g.p); await new Promise(r => setTimeout(r, 0)); }
    labParticlesMade(); labTruth();
    const tj = performance.now();
    frameBody(L.t, 1 / 60);
    L.jsMs.push(performance.now() - tj);
    labParticlesMade(); labTruth();
    labFrameChecks();
    L.digest.push(labDigest());
    // (debugging a difference between two runs: the particles' records at one frame, SF_PARTDUMP)
    if (window.__partDumpAt && !window.__motionLog) window.__motionLog = [];
    if (window.__partDumpAt === L.f && snapCur && snapCur.parts && snapCur.parts.data) L.partDump = Array.from(snapCur.parts.data.subarray(0, snapCur.parts.n * 12));
    // let the GPU finish this frame's work and its stats come back (the next frame sees them)
    // (and wait for all of them: what the next frame sees must not depend on how fast the browser maps buffers)
    if (dev) { await dev.queue.onSubmittedWorkDone(); await gpu.flushStats(); await new Promise(r => setTimeout(r, 0)); }
    // (SF_SHOTS: a picture at these frames, from outside the building, for a person or Claude to look at)
    if (window.__shots && window.__shots.includes(L.f)) {
      const p0 = cam.position.clone(), r0 = cam.rotation.clone(), v = window.__shotView || [-26, 14, -30];
      if (typeof drawDebris === 'function') drawDebris(performance.now());
      cam.position.set(v[0], v[1], v[2]); cam.lookAt(0, 6, 0); renderer.render(scene, cam);
      window.__shotReady = L.f; while (window.__shotReady) await new Promise(r => setTimeout(r, 100));
      cam.position.copy(p0); cam.rotation.copy(r0);
    }
    // (debugging a bounce: chosen bodies, every frame of the trace window, straight from the GPU: pose, velocity,
    // mass, and every contact they have, with its normal, points, impulses and approach speeds)
    if (window.__traceBodies && dev && L.traceWin && L.f >= L.traceWin[0] && L.f <= L.traceWin[1]) {
      const want = new Set(String(window.__traceBodies).split(',').map(Number).filter(x => x > 0));
      // ('ball': the heavy bodies, whatever their numbers)
      if (String(window.__traceBodies).includes('ball')) gpu.bodies.forEach((gb, k) => { if (gb && !gb.dead && gb.heavy) want.add(k + 1); });
      // ('big': the bodies over 10 t)
      if (String(window.__traceBodies).includes('big')) gpu.bodies.forEach((gb, k) => { if (gb && !gb.dead && gb.mass > 1e4) want.add(k + 1); });
      const nm = Math.min(gpu.stats.manifolds || 0, gpu.maxMan), cur = (gpu.frame - 1) & 1;
      const [man, bod, sol] = await Promise.all([gpu.read(gpu.manBuf, 2 * gpu.maxMan * 17 * 16), gpu.read(gpu.bodBuf, gpu.n1 * 80), gpu.read(gpu.solBuf, gpu.n1 * 96)]);
      const r3 = (a, o) => [a[o], a[o + 1], a[o + 2]].map(x => +x.toFixed(3));
      // ('awake': every movable body awake now, when they are few: what keeps a scene from settling)
      if (String(window.__traceBodies).includes('awake')) { const aw = []; for (let b = 1; b < gpu.n1; b++) if (bod[b * 20 + 3] > 0 && sol[b * 24 + 3] > 0 && bod[b * 20 + 1] > -500) aw.push(b); if (aw.length <= 12) aw.forEach(b => want.add(b)); }
      const bodyRow = (b) => ({ b, p: r3(bod, b * 20), m: bod[b * 20 + 3] > 0 ? Math.round(1 / bod[b * 20 + 3]) : 0, v: r3(sol, b * 24), w: r3(sol, b * 24 + 4), asleep: bod[b * 20 + 12] > 0.5 ? 1 : 0, rest_s: +bod[b * 20 + 11].toFixed(3), from_anchor_mm: +(1000 * Math.hypot(bod[b * 20] - bod[b * 20 + 13], bod[b * 20 + 1] - bod[b * 20 + 14], bod[b * 20 + 2] - bod[b * 20 + 15])).toFixed(2), turned_mrad: +(2000 * Math.hypot(...[0, 1, 2].map(k => { const q = [bod[b * 20 + 4], bod[b * 20 + 5], bod[b * 20 + 6], bod[b * 20 + 7]], r = [-bod[b * 20 + 16], -bod[b * 20 + 17], -bod[b * 20 + 18], bod[b * 20 + 19]]; const c = [q[1] * r[2] - q[2] * r[1], q[2] * r[0] - q[0] * r[2], q[0] * r[1] - q[1] * r[0]]; return q[3] * r[k] + r[3] * q[k] + c[k]; }))).toFixed(2), shapes: (gpu.bodyShapes.get(b) || []).length });
      const row = { f: L.f, gf: gpu.frame, bodies: [...want].filter(b => b > 0 && b < gpu.n1).map(bodyRow), contacts: [] };
      for (let k = 0; k < nm; k++) {
        const o = (cur * gpu.maxMan + k) * 68, a = man[o], b = man[o + 1];
        if (!want.has(a) && !want.has(b)) continue;
        const cnt = man[o + 2] & 7, pts = [];
        for (let q = 0; q < cnt; q++) pts.push({ ra: r3(man, o + (6 + 2 * q) * 4), jn: +man[o + 14 * 4 + q].toFixed(0), rv: +man[o + 15 * 4 + q].toFixed(2) });
        row.contacts.push({ a, b, ma: a > 0 ? (bod[a * 20 + 3] > 0 ? Math.round(1 / bod[a * 20 + 3]) : 0) : 0, mb: bod[b * 20 + 3] > 0 ? Math.round(1 / bod[b * 20 + 3]) : 0, n: r3(man, o + 4), cnt, pts, e: +man[o + 64].toFixed(2), J: Math.round(man[o + 65]), ap: +man[o + 66].toFixed(2) });
      }
      // (and the static shapes within a metre of the first of them)
      if (row.bodies[0]) { const c = row.bodies[0].p, o1 = gpu.bodies[0].p; const wp = (sh) => [0, 1, 2].map(k => o1[k] + sh.lp[k]);
        row.near = gpu.shapes.map((sh, k) => ({ sh, k })).filter(({ sh }) => { if (!sh || sh.dead || sh.body !== 1 || !sh.lp) return false; const w = wp(sh); return Math.abs(w[0] - c[0]) < 0.6 + sh.h[0] && Math.abs(w[1] - c[1]) < 0.6 + sh.h[1] && Math.abs(w[2] - c[2]) < 0.6 + sh.h[2]; }).slice(0, 80).map(({ sh, k }) => ({ k, p: wp(sh).map(x => +x.toFixed(3)), h: sh.h.map(x => +x.toFixed(3)), hull: sh.hull || 0 })); }
      (L.btrace = L.btrace || []).push(row);
    }
    // (debugging determinism: the GPU's own body, solver and shape records, every frame)
    if (window.__detDump && dev) {
      const hb = async (buf, bytes) => { const u = new Uint32Array(await gpu.read(buf, bytes)); let h = 2166136261 >>> 0; for (let k = 0; k < u.length; k++) h = Math.imul(h ^ u[k], 16777619) >>> 0; return h; };
      L.dump = [await hb(gpu.bodBuf, gpu.n1 * 80), await hb(gpu.solBuf, gpu.n1 * 96), await hb(gpu.shpBuf, Math.min(gpu.ns1 * 128, 64 << 20))];
      {
        // the colours as a set of (shapes, colour), whatever the slots
        const nm = Math.min(gpu.stats.manifolds || 0, gpu.maxMan), cur = (gpu.frame - 1) & 1;
        const mc = new Uint32Array((await gpu.read(gpu.intBuf, gpu.maxMan * 4, gpu.L.mcolor * 4)).buffer);
        const cold = await gpu.read(gpu.coldBuf, 2 * gpu.maxMan * 6 * 16);
        const keys = [];
        for (let k = 0; k < nm; k++) { const o = ((cur * gpu.maxMan + k) * 6) * 4; keys.push(`${cold[o + 19]},${cold[o + 23]}:${mc[k]}`); }
        if (window.__detAt === L.f || window.__detRows) {
          const man = await gpu.read(gpu.manBuf, 2 * gpu.maxMan * 17 * 16), sol = await gpu.read(gpu.solBuf, gpu.n1 * 96);
          const rows = [];
          for (let k = 0; k < nm; k++) { const o = ((cur * gpu.maxMan + k) * 6) * 4, b = (cur * gpu.maxMan + k) * 68; rows.push([cold[o + 19], cold[o + 23], mc[k], man[b], man[b + 1], man[b + 2], sol[man[b] * 24 + 3] > 0 ? 1 : 0, sol[man[b + 1] * 24 + 3] > 0 ? 1 : 0]); }
          rows.sort((x, y) => x[0] - y[0] || x[1] - y[1]);
          L.colRows = rows; if (window.__detRows) (L.allRows = L.allRows || {})[L.f] = rows;
        }
        keys.sort(); let h = 2166136261 >>> 0; for (const c of keys.join('|')) h = Math.imul(h ^ c.charCodeAt(0), 16777619) >>> 0;
        L.dump.push(h, nm);
      }
      const last = L.digest[L.digest.length - 1]; if (last) last.push(...L.dump);
      if (gpu.colorCheck) { const c = new Uint32Array((await gpu.read(gpu.intBuf, 4, gpu.L.cchk * 4)).buffer)[0]; if (c) L.colConflicts = (L.colConflicts || 0) + c; if (c && !L.colConflictAt) L.colConflictAt = L.f; }
      if (window.__detAt === L.f) L.dumpRaw = { sol: Array.from(new Float32Array(await gpu.read(gpu.solBuf, gpu.n1 * 96))), shp: Array.from(new Float32Array(await gpu.read(gpu.shpBuf, gpu.ns1 * 128))), bod: Array.from(new Float32Array(await gpu.read(gpu.bodBuf, gpu.n1 * 80))) };
    }
    const g = gpu.gpuMs ? Object.values(gpu.gpuMs).reduce((a, b) => a + b, 0) : 0;
    if (g > L.peakGpu) { L.peakGpu = g; L.peakStages = Object.fromEntries(Object.entries(gpu.gpuMs).map(([k, v]) => [k, +v.toFixed(2)])); }
    L.work.cs += gpu.stats.manifolds * gpu.substeps; L.work.ps += (gpu.stats.pActive || 0) * (gpu.psub || 4);
    // Settled: nothing awake, and no engine job since that changed anything. (Until 18 Nov: and no job in flight.
    // While joints wear, the structure is checked every quarter second for as long as they do, each check a job
    // that finds nothing; whether one was in flight at a frame decided the check. Lab qbarn, 18 Nov, 900 frames:
    // nothing awake from frame 300, 3 joints wearing, a check every 20 frames with no failure and nothing sent,
    // "not settled".)
    const quietNow = (gpu.stats.awake || 0) === 0 && !(gpu.stats.pActive > 0) && L.f > L.quietFrom;
    if (quietNow && L.settledAt == null) L.settledAt = L.f; else if (!quietNow) L.settledAt = null;
    if ((gpu.stats.awake || 0) === 0) L.bodiesQuietAt = L.f;   // (the last frame every body was asleep)
    if (L.f % 10 === 0 || L.f === S.frames - 1) {
      let roof = 0;
      // (over the building the scenario is about: roofAt, a building of the Quarter and half the square to look in)
      const ra = S.roofAt ? [...quarterAt(S.roofAt[0], S.roofAt[1] || 0, 0, S.roofAt[2] || 0), S.roofAt[3] || 9] : [0, 0, 0, 9];
      for (let x = -ra[3]; x <= ra[3]; x += ra[3] / 3) for (let z = -ra[3]; z <= ra[3]; z += ra[3] / 3) { const ts = E.raycast(ra[0] + x + 0.3, 200, ra[2] + z + 0.3, 0, -1, 0, 300); if (ts >= 0) roof = Math.max(roof, 200 - ts); }
      if (snapCur) for (let k = 0; k < snapCur.pose.length / 10; k++) { const y = snapCur.pose[k * 10 + 1] + snapCur.pose[k * 10 + 8]; if (Number.isFinite(y) && Math.abs(snapCur.pose[k * 10] - ra[0]) < ra[3] + 3 && Math.abs(snapCur.pose[k * 10 + 2] - ra[2]) < ra[3] + 3) roof = Math.max(roof, y); }
      L.roofMin = Math.min(L.roofMin ?? 1e9, L.f > 10 ? roof : 1e9);
      L.series.push({ f: L.f, D: +L.D.toFixed(2), T: +L.T.toFixed(2), drawn: L.draw.n, bodies: gpu.n, awake: gpu.stats.awake, contacts: gpu.stats.manifolds, parts: gpu.stats.pLive || 0, pAwake: gpu.stats.pActive || 0, sub: gpu.substeps, gpu: +g.toFixed(2), js: +L.jsMs[L.jsMs.length - 1].toFixed(2), roof: +roof.toFixed(2), upper: +labUpper().toFixed(1), breaks, dust: dustBricks, failed: failedJoints, cracked: crackedJoints, worn: wornJoints, wearing: structWearing });
    }
    if (performance.now() - lastShow > 1000) { lastShow = performance.now(); $('bench-out').textContent = `Lab: ${name} · frame ${L.f} of ${S.frames} · ${((performance.now() - tw) / 1000).toFixed(0)} s\nholes ${L.holeFrames} · duplicates ${L.dupFrames} · flickers ${L.flickers} · jumps ${L.teleports} · bounces ${L.bounces} · launches ${L.launches}`; L.status = { f: L.f, frames: S.frames }; }
  }
  const rep = labReport((performance.now() - tw) / 1000);
  L.on = false; simRandom = () => Math.random();
  // the next scenario, after a reload (each starts from a fresh building)
  st.done.push(name); st.queue.shift(); st.results.push(rep);
  try { localStorage.setItem('sf-lab', JSON.stringify(st.queue.length ? st : null)); } catch (e) {}
  if (!st.queue.length) { try { localStorage.setItem('sf-lab-last', JSON.stringify(st.results)); } catch (e) {} }
  window.LAB_RESULT = rep; window.LAB_DONE = !st.queue.length; window.LAB_ALL = st.results;
  labShow(st.results, !st.queue.length);
  if (st.queue.length && !st.external) location.reload();
}
function labShow(results, done) {
  $('bench').hidden = false;
  $('bench-title').textContent = done ? `Lab: done (${results.filter(r => r.pass).length} of ${results.length} passed)` : 'Lab: running';
  $('bench-out').textContent = results.map(labSummary).join('\n\n') + (done ? '\n\n(Copy gives the full JSON with events and time series.)' : '');
  LAB.copyText = done ? JSON.stringify(results.map(r => ({ ...r, series: r.series.filter((_, k) => k % 3 === 0) }))) : null;
}
function labStart(queue, opts = {}) {
  const st = { queue, done: [], results: [], readLag: opts.readLag ?? 2, jobLag: opts.jobLag ?? 4, render: opts.render ?? true, external: !!opts.external };
  try { localStorage.setItem('sf-lab', JSON.stringify(st)); } catch (e) {}
  location.reload();
}
