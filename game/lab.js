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
  const addB = gpu.addBodies.bind(gpu), brk = gpu.breakBody.bind(gpu), addP = gpu.addParticles.bind(gpu), killP = gpu.killParticle.bind(gpu);
  gpu.addBodies = (list) => {
    const idx = addB(list);
    idx.forEach((b, k) => {
      if (b <= 0) return;
      const gb = gpu.bodies[b - 1];
      if (L.ballNext) { L.balls.add(b); if (gb) gb.labBall = true; for (const k of gpu.bodyShapes.get(b) || []) { const sh = gpu.shapes[k - 1]; if (sh) sh.labBall = true; } return; }
      const v = list[k].shapes.reduce((s, sh) => s + labBoxVol(sh.h) * (sh.hull ? GpuWorld.hulls.hulls[sh.hull].vol : 1), 0); L.bodyVol.set(b, v); L.gpuVol += v; if (L.jobVol) L.jobVol.bodiesIn += v;
      // (where parts come loose: frame, the job that made them, height of the body's top and bottom, volume)
      if (L.made && L.made.length < 4000) { const ys = list[k].shapes.map(sh => sh.p[1]); L.made.push([L.f, L.jobOp || '', +Math.min(...ys).toFixed(1), +Math.max(...ys).toFixed(1), +v.toFixed(2), list[k].shapes.length]); }
      if (gb) { gb.labVol = v; gb.labH = list[k].shapes.length === 1 ? list[k].shapes[0].h.map(x => +x.toFixed(2)) : `${list[k].shapes.length} boxes`; gb.labF = L.f; }
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
    if (window.__farKill) { L.lostBy.far = (L.lostBy.far || 0) + v; L.farBodies = (L.farBodies || 0) + 1; const gb = gpu.bodies[parent - 1]; labEvent('removed_far', L.f, { body: parent, vol: +v.toFixed(3), ball: wasBall, size: gb && gb.labH, made_at: gb && gb.labF, shapes: (gpu.bodyShapes.get(parent) || []).length }); }   // (removed past the clean-up distance)
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
      const m = (gb.labVol || 0) * 2400, e = 0.5 * m * (d.v[0] ** 2 + d.v[1] ** 2 + d.v[2] ** 2 - vp[0] ** 2 - vp[1] ** 2 - vp[2] ** 2);
      if (e > 0) { L.given = (L.given || 0) + e; L.givenTotal = (L.givenTotal || 0) + e; }
    });
    return brk(parent, ...rest);
  };
  // a piece broken on the GPU: the spare takes its part's volume from the piece; a spare given up loses it
  const snp = gpu.snapped.bind(gpu), kill = gpu.killBody.bind(gpu);
  gpu.snapped = (i, j, imp) => {
    const r = snp(i, j, imp);
    const moved = (gpu.bodyShapes.get(j) || []).reduce((t, k) => { const sh = gpu.shapes[k - 1]; return t + (sh ? labBoxVol(sh.h) * (sh.hull ? GpuWorld.hulls.hulls[sh.hull].vol : 1) : 0); }, 0);
    if (L.bodyVol.has(i) && moved > 0) {
      L.bodyVol.set(i, L.bodyVol.get(i) - moved); L.bodyVol.set(j, moved); const gb = gpu.bodies[j - 1]; if (gb) { gb.labVol = moved; gb.labF = L.f; gb.labH = 'spare'; }
      // (the piece left is another body from here: its mass is less and its centre elsewhere. Compared with the whole
      // at the last read-back, its centre's jump read as a gain: 7.9 MJ for the 816 m3 upper part at frame 58 of the
      // lab collapse, 21 MJ at frame 223, at the speed it had)
      const gi = gpu.bodies[i - 1]; if (gi) { gi.labVol = L.bodyVol.get(i); gi.labCut = L.f; }
    }
    return r;
  };
  gpu.killBody = (b) => { const v = L.bodyVol.get(b) || 0; L.gpuVol -= v; L.bodyVol.delete(b); return kill(b); };
  gpu.addParticles = (list) => { const s = addP(list); L.pendingParts = (L.pendingParts || []).concat(s); return s; };
  gpu.killParticle = (i) => { const v = L.partVol.get(i) || 0; L.gpuVol -= v; L.partVol.delete(i); if (window.__farKill) { L.lostBy.far = (L.lostBy.far || 0) + v; L.farParts = (L.farParts || 0) + 1; labEvent('removed_far', L.f, { particle: i, vol: +v.toFixed(3) }); } else L.lostParticles = (L.lostParticles || 0) + v; return killP(i); };
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
  if (d.n >= d.keys.length) { const g = (a, C) => { const b = new C(a.length * 2); b.set(a); return b; }; d.keys = g(d.keys, Float64Array); d.pos = g(d.pos, Float32Array); d.bound = g(d.bound, Float32Array); }
  d.keys[d.n] = key; d.pos[d.n * 3] = x; d.pos[d.n * 3 + 1] = y; d.pos[d.n * 3 + 2] = z; d.bound[d.n] = bound; d.n++; d.vol += vol;
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
// every frame: what is drawn against what exists
function labFrameChecks() {
  const L = LAB, d = L.draw, f = L.f;
  // a) volume: drawn (building mesh + debris + particles) within what existed over the last few frames (the
  //    read-back delay); below = something that exists is not drawn (a hole, flicker), above = drawn twice
  const D = L.staticDrawn + d.vol;
  const parts = () => ({ static_drawn: +L.staticDrawn.toFixed(2), static_now: +L.staticT.toFixed(2), gpu_drawn: +d.vol.toFixed(2), gpu_now: +L.gpuVol.toFixed(2), reads_pending: L.gates.length, chunks_pending: pendingChunks.length, placing: gpu.placing() });
  L.Thist.push(L.T); if (L.Thist.length > L.readLag + L.jobLag + 4) L.Thist.shift();
  const lo = Math.min(...L.Thist), hi = Math.max(...L.Thist), tol = L.volTol;
  if (D < lo - tol) { L.holeFrames++; L.holeMax = Math.max(L.holeMax, lo - D); if (!L.inHole) labEvent('hole', f, { missing_m3: +(lo - D).toFixed(3), ...parts(), undrawn: labUndrawn() }); L.inHole = true; }
  else if (L.inHole) { L.inHole = false; L.holesRecovered++; }
  if (D > hi + tol) { L.dupFrames++; L.dupMax = Math.max(L.dupMax, D - hi); labEvent('duplicate', f, { extra_m3: +(D - hi).toFixed(3), ...parts() }); }
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
        const m = Math.hypot(dx, dy, dz), lim = 0.25 + 2 * Math.max(d.bound[k], prev.bound[j]) / 60;
        if (m > lim) { L.teleports++; const sh = key >= 0 ? gpu.shapes[(key % 1048576) - 1] : null, gb = sh ? gpu.bodies[sh.body - 1] : null; labEvent('teleport', f, { key, moved_m: +m.toFixed(2), limit_m: +lim.toFixed(2), body: sh ? sh.body : null, made: gb && gb.labF, size: gb && gb.labH, from: [prev.pos[j * 3], prev.pos[j * 3 + 1], prev.pos[j * 3 + 2]].map(v => +v.toFixed(2)), to: [d.pos[k * 3], d.pos[k * 3 + 1], d.pos[k * 3 + 2]].map(v => +v.toFixed(2)) }); }
      } else {
        const gone = L.goneAt.get(key);
        if (gone !== undefined && f - gone <= 30) { L.flickers++; labEvent('flicker', f, { key, absent_frames: f - gone }); }
      }
    }
  }
  if (prev) for (const [key] of prev.map) if (!cur.has(key)) L.goneAt.set(key, f);
  L.prevDraw = { map: cur, pos: d.pos.slice(0, d.n * 3), bound: d.bound.slice(0, d.n) };
}
// every read-back: the physics
const LAB_RHO = [2400, 1800, 2500, 2500, 500, 7800, 2400, 2400];
function labOnRead(r, pr) {
  // spares parked under the ground must sleep there (woken, they fell for the rest of the run: 8 November)
  if (r.sleeping && r.body) { let n = 0; const nb = Math.min(r.sleeping.length, r.body.length / 7); for (let b = 1; b <= nb; b++) if (!r.sleeping[b - 1] && r.body[(b - 1) * 7 + 1] < -500) n++; LAB.parkedAwake = Math.max(LAB.parkedAwake || 0, n); }
  // (what is still awake near the end, for a run that does not settle: bodies, and where the awake particles are)
  if (LAB.f >= LAB.spec.frames - 3 && r.sleeping && !LAB.awakeEnd) {
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
    L.awakeEnd = { particles: plist, odd: (L.awakeOdd || []).slice(0, 20), odd_n: (L.awakeOdd || []).length, bodies: list.slice(0, 20), particle_cells: [...cells.entries()].sort((a, b) => b[1].n - a[1].n).slice(0, 12).map(([k, e]) => ({ at: k, n: e.n, vmax: +e.v.toFixed(2) })) };
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
  let bad = 0, below = 0, gain = 0, ekPrev = 0;
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
    if (gb.labBall) continue;
    const m = (gb.labVol || 0) * 2400, ek = 0.5 * m * (vx * vx + vy * vy + vz * vz), e = ek + m * 9.81 * y;
    now.set(b, { gb, vy, e, ek, v: [vx, vy, vz], y, vol: gb.labVol || 0, cut: gb.labCut });
    // the same body as at the last read-back (a reused slot holds another one)
    const was = L.prevBodies && L.prevBodies.get(b);
    // (nor a body cut on the GPU since, or a few frames before: the poses read back lag the step)
    if (!was || was.gb !== gb || was.cut !== gb.labCut || (gb.labCut !== undefined && f - gb.labCut <= 3)) continue;
    gain += e - was.e; ekPrev += was.ek;
    if (e - was.e > 1000) (L.gainers = L.gainers || []).push({ body: b, gain_J: Math.round(e - was.e), size: gb.labH, made_at: gb.labF, y: [+was.y.toFixed(2), +y.toFixed(2)], vy: [+was.vy.toFixed(2), +vy.toFixed(2)], vol: +(gb.labVol || 0).toFixed(2), vol_was: +(was.vol || 0).toFixed(2), asleep: r.sleeping ? r.sleeping[b - 1] : null });
    if (quiet) {
      if (was.vy < -2 && vy > 0.5 * -was.vy) { L.bounces++; L.bounceMax = Math.max(L.bounceMax, vy / -was.vy); labEvent('bounce', f, { body: b, vy_before: +was.vy.toFixed(2), vy_after: +vy.toFixed(2), y: +y.toFixed(2), size: gb.labH, made_at: gb.labF, pushed_by_page: gb.labSetV || null }); }
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
  if (quiet && L.prevBodies && gain > 0.05 * ekPrev + 2000 + given) { L.energyGains++; L.energyGainMax = Math.max(L.energyGainMax, ekPrev > 0 ? gain / ekPrev : 0); labEvent('energy_gain', f, { gain_J: Math.round(gain), kinetic_before_J: Math.round(ekPrev), most: (L.gainers || []).sort((a, b) => b.gain_J - a.gain_J).slice(0, 3) }); }
  L.gainers = [];
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
  L.quietFrom = Math.max(L.quietFrom, L.f + (act === 'ball' ? 120 : 30));
  if (act === 'cut') {
    // the benchmark's cut: the ground storey's columns and walls blown out, before the physics runs
    physicsPaused = true;
    for (let x = -10; x <= 10.1; x += 4) for (let z = -10; z <= 10.1; z += 4) jobs.blast.push({ op: 'blast', x, y: 1.8, z, r: 2.5, ctx: { d: [0, 0, 1], t0: clock() } });
    pump();
    const T0 = L.T;
    while (L.jobq.length) { const j = L.jobq.shift(); finish(runInline(j.job)); labParticlesMade(); }
    L.staticT = labTruthStatic(); labTruth();
    L.lostBy.tool = (L.lostBy.tool || 0) + Math.max(0, T0 - L.T);
    physicsPaused = false; lastActive = clock();
  } else if (act === 'ball') {
    const [x, y, z, yw, pt] = a, d = [Math.sin(yw) * Math.cos(pt), Math.sin(pt), Math.cos(yw) * Math.cos(pt)];
    L.ballNext = true; gpu.addBodies([{ shapes: ballShapes([x, y, z], 0.4), v: [d[0] * 120, d[1] * 120, d[2] * 120], w: [0, 0, 0] }]); lastActive = clock();
  } else if (act === 'blast') {
    const [ox, oy, oz, dx, dy, dz, r] = a;
    setRadius(r); doBlast(new THREE.Vector3(ox, oy, oz), new THREE.Vector3(dx, dy, dz).normalize());
  }
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
    ['energy gains over 5% (while quiet)', L.energyGains, 0, 'energy made from nothing'],
    // what the tools did not take out must stay in the world: standing, falling, as rubble or particles. Dust (fines
    // removed) is real but small; a whole floor gone as dust is a floor that vanished (7 November: 57% in collapse)
    // (what the clean-up distance removed is gone on purpose, like what the tools took)
    ['volume lost, not to the tools (% of the building)', +(100 * (L.V0 - L.T - (L.lostBy.tool || 0) - (L.lostBy.far || 0)) / L.V0).toFixed(1), 2, 'parts vanishing'],
    // (and none made: a part doubled, 7 November: the mixed building's falling part doubled where the GPU had broken it)
    ['volume made (% of the building)', +Math.max(0, -100 * (L.V0 - L.T - (L.lostBy.tool || 0) - (L.lostBy.far || 0)) / L.V0).toFixed(1), 0.5, 'parts doubled'],
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
  if (S.expect && S.expect.far) {
    // past the clean-up distance things go at the next read-back (what their speed carries them over the lag is
    // allowed for, see labOnRead)
    checks.push({ name: 'removed past the clean-up distance (bodies + particles)', value: (L.farBodies || 0) + (L.farParts || 0), limit: '>= 1', pass: (L.farBodies || 0) + (L.farParts || 0) >= 1, what: 'the clean-up distance works' });
    checks.push({ name: 'farthest past the clean-up distance, still in the world (m)', value: +(L.farOver || 0).toFixed(2), limit: 1, pass: (L.farOver || 0) <= 1, what: 'nothing stays far out' });
  }
  if (S.expect && S.expect.roofDrop) checks.push({ name: 'roof fall (m)', value: +(roof0 - (last.roof || 0)).toFixed(2), limit: `>= ${S.expect.roofDrop}`, pass: roof0 - (last.roof || 0) >= S.expect.roofDrop, what: 'the cut storey brings the building down' });
  if (S.expect && S.expect.settle) checks.push({ name: 'settled (s)', value: L.settledAt == null ? 'no' : +(L.settledAt / 60).toFixed(1), limit: `by ${(S.frames / 60).toFixed(0)}`, pass: L.settledAt != null, what: 'everything asleep before the end' });
  const gpuMs = se.map(x => x.gpu), jsMs = L.jsMs;
  return {
    scenario: L.name, shortened: S.frames !== LAB_SCENARIOS[L.name].frames, title: S.title, kind: ['concrete frame', 'masonry', 'mass timber', 'mixed'][S.kind], storeys: S.storeys, frames: S.frames,
    readLag: L.readLag, jobLag: L.jobLag, settings: { ...setting }, ua: navigator.userAgent, adapter: L.adapter, date: new Date().toISOString(),
    pass: checks.every(c => c.pass), checks, colConflicts: L.colConflicts || 0, colConflictAt: L.colConflictAt || null, digest: L.digest, dumpRaw: L.dumpRaw ? { ...L.dumpRaw, colRows: L.colRows, allRows: L.allRows } : (L.colRows ? { colRows: L.colRows, allRows: L.allRows } : null),
    physics: { roof_m: [+roof0.toFixed(2), +(last.roof || 0).toFixed(2)], dust_of_building: +(1 - (L.T) / L.V0).toFixed(3), particles_by: { ...partsBy }, big_breaks: L.bigBreaks, made: L.made, gains: L.gains, spill_why: { crumbs: E.spill_why(0), fragments: E.spill_why(1), crushed_storey: E.spill_why(2), static: E.spill_why(3), fragment_bricks_by_mat: [0, 1, 2, 3, 4, 5, 6, 7].map(k => E.spill_mat(k)), single_brick_fragments: E.spill_mat(8) }, over_breaks: L.overBreaks, dbg_breaks: L.dbgBreaks, lost_m3: Object.fromEntries(Object.entries({ ...L.lostBy, particles_crushed: L.lostParticles || 0, building: L.V0 }).map(([k, v]) => [k, +v.toFixed(1)])), particles_made: partsMade, particles_crushed: partsCrushed, bodies_crushed: crushedBodies, removed_far: { bodies: L.farBodies || 0, particles: L.farParts || 0, m3: +((L.lostBy.far || 0)).toFixed(2) }, breaks, failed_joints: failedJoints, settled_s: L.settledAt == null ? null : +(L.settledAt / 60).toFixed(2), bounce_max: +L.bounceMax.toFixed(2), launch_max: +L.launchMax.toFixed(2), hole_max_m3: +L.holeMax.toFixed(3), dup_max_m3: +L.dupMax.toFixed(3), energy_given_by_cuts_J: Math.round(L.givenTotal || 0), rubble_on_floors_max_t: +((L.restMax || 0) / 9810).toFixed(1), rubble_sent_bodies_particles_t: L.restSplit || null, joints_cracked: L.cracked || 0, gpu_breaks: gpuSnaps, gradual: L.gradual || undefined, awake_at_end: L.awakeEnd || undefined },
    perf: {
      wall_s: +wallS.toFixed(1), steps: stepsDone, steps_skipped: gpu.skipped || 0,
      gpu_ms: { median: +med(gpuMs).toFixed(2), p95: +p95(gpuMs).toFixed(2), max: +Math.max(0, ...gpuMs).toFixed(2) },
      gpu_stages_ms_at_peak: L.peakStages,
      js_ms: { median: +med(jsMs).toFixed(2), p95: +p95(jsMs).toFixed(2), max: +Math.max(0, ...jsMs).toFixed(2) },
      engine_ms: { jobs: engineMsLog.length, median: +med(engineMsLog.slice()).toFixed(1), p95: +p95(engineMsLog.slice()).toFixed(1), max: +Math.max(0, ...engineMsLog).toFixed(1) },
      peak: { bodies: Math.max(0, ...se.map(x => x.bodies)), awake: Math.max(0, ...se.map(x => x.awake)), contacts: Math.max(0, ...se.map(x => x.contacts)), particles: Math.max(0, ...se.map(x => x.parts)), particles_moving: Math.max(0, ...se.map(x => x.pAwake)), drawn: Math.max(0, ...se.map(x => x.drawn)) },
      work: { contact_substeps: L.work.cs, particle_substeps: L.work.ps },
    },
    events: L.events, timeline: L.timeline, trace: L.trace.length ? L.trace : undefined, btrace: L.btrace || undefined, parked: globalThis.__parked || undefined, slotLog: globalThis.__slotLog || undefined, bursts: L.bursts || undefined, traceEvents: L.traceEv.length ? L.traceEv : undefined,
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
// A frame's state as one number, to compare two runs of a scenario (node game/lab.cjs --twice): the poses read
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
  return [pose.v, parts.v, gpu.n, breaks, dustBricks, E.bricks_alive(), E.joints_live(), gpu.stats.manifolds || 0, partsMade, gpu.stats.awake || 0, L_events(), job.v, gpu.inputs, gpu.frame, gpu.skipped || 0, gpu.colorLimit, gpu.substeps, gpu.ns, gpu.np || 0, JSON.stringify(gpu.stats).length, (gpu.pendingInherit || []).length];
}
function L_events() { return (gpu.events && gpu.events.length) || 0; }
async function labRun() {
  const L = LAB, st = LABSTATE, name = st.queue[0], S = { ...LAB_SCENARIOS[name] };
  if (st.frames) S.frames = st.frames;   // (a shortened run, for quick checks of the lab itself)
  farDist = S.far ?? 100;   // (the clean-up distance: the page's default, or the scenario's)
  Object.assign(L, {
    on: true, name, spec: S, t: 0, f: 0, readLag: st.readLag ?? 2, jobLag: st.jobLag ?? 4, render: !!st.render, gates: [], jobq: [],
    draw: { n: 0, vol: 0, keys: new Float64Array(4096), pos: new Float32Array(4096 * 3), bound: new Float32Array(4096) },
    Thist: [], holeFrames: 0, holeMax: 0, inHole: false, holesRecovered: 0, dupFrames: 0, dupMax: 0, flickers: 0, teleports: 0, goneAt: new Map(), prevDraw: null,
    bad: 0, below: 0, bounces: 0, bounceMax: 0, launches: 0, launchMax: 0, energyGains: 0, energyGainMax: 0, prevBodies: null, quietFrom: -1,
    injectedAt: -1000, settledAt: null, digest: [], timeline: [], trace: [], traceEv: [], traceWin: st.trace || null, events: {}, series: [], jsMs: [], work: (engineMsLog.length = 0, { cs: 0, ps: 0 }), peakStages: null, peakGpu: 0, volTol: 0.05,
  });
  // the defaults, whatever this browser has saved (the report lists them)
  for (const [k, d] of Object.entries(SETTINGS)) { setting[k] = d.def; d.apply(d.def); }
  setting.preset = 'high (lab defaults)';
  // fixed sub-steps: adapting them to this machine's speed would make the runs differ between machines
  // (SF_PRESET: a quality preset over the defaults, as a player would pick it: a benchmark on 'low' kept bodies
  // awake far longer than the lab's 'high')
  if (window.__preset && PRESETS[window.__preset]) { const pr = PRESETS[window.__preset]; for (const k of ['crumbs', 'rubble', 'substeps', 'colors']) { setting[k] = pr[k]; SETTINGS[k].apply(pr[k]); } setting.preset = window.__preset; }
  setting.adaptive = 0; gpu.substeps = setting.substeps;
  Math.random = seededRandom(window.__seed || 20261104);   // (SF_SEED: another run of the same scenario, to tell a trend from one run's luck)
  if (window.__pinch && gpu) gpu.pinching = true;
  if (window.__oldMasonry) E.set_param(6, 1);   // (SF_OLD_MASONRY: masonry in falling parts parts at once, for comparison)   // (SF_PINCH: crushing of pinched pieces, off by default)
  fly = true; E.player_set(-20, 8, -30); yaw = Math.PI + 0.62; pitch = -0.12;
  $('start').hidden = true; physicsPaused = false;
  $('bench').hidden = false; $('bench-title').textContent = `Lab: ${name} (${st.done.length + 1} of ${st.done.length + st.queue.length})`;
  labInstall();
  L.staticT = labTruthStatic(); labTruth(); L.V0 = L.T; L.lostBy = {}; L.farBodies = 0; L.farParts = 0; L.farOver = 0; L.awakeEnd = null; L.parkedAwake = 0; L.awakeOdd = []; L.made = []; L.bigBreaks = []; L.Thist = [L.T]; L.staticDrawn = Object.values(chunkVol).reduce((a, b) => a + b, 0);
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
      if (L.timeline.length < 400) L.timeline.push({ f: L.f, op: j.job.op, n: r.n, sent: j.due - L.jobLag, new_bodies: gpu.n - nb0, breaks: breaks - b0, static_updates: r.out && r.out.length ? r.out[0] : 0 });
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
      const want = new Set(String(window.__traceBodies).split(',').map(Number));
      const nm = Math.min(gpu.stats.manifolds || 0, gpu.maxMan), cur = (gpu.frame - 1) & 1;
      const [man, bod, sol] = await Promise.all([gpu.read(gpu.manBuf, 2 * gpu.maxMan * 17 * 16), gpu.read(gpu.bodBuf, gpu.n1 * 80), gpu.read(gpu.solBuf, gpu.n1 * 96)]);
      const r3 = (a, o) => [a[o], a[o + 1], a[o + 2]].map(x => +x.toFixed(3));
      const bodyRow = (b) => ({ b, p: r3(bod, b * 20), m: bod[b * 20 + 3] > 0 ? Math.round(1 / bod[b * 20 + 3]) : 0, v: r3(sol, b * 24), w: r3(sol, b * 24 + 4), asleep: bod[b * 20 + 12] > 0.5 ? 1 : 0, shapes: (gpu.bodyShapes.get(b) || []).length });
      const row = { f: L.f, gf: gpu.frame, bodies: [...want].filter(b => b > 0 && b < gpu.n1).map(bodyRow), contacts: [] };
      for (let k = 0; k < nm; k++) {
        const o = (cur * gpu.maxMan + k) * 68, a = man[o], b = man[o + 1];
        if (!want.has(a) && !want.has(b)) continue;
        const cnt = man[o + 2] & 7, pts = [];
        for (let q = 0; q < cnt; q++) pts.push({ ra: r3(man, o + (6 + 2 * q) * 4), jn: +man[o + 14 * 4 + q].toFixed(0), rv: +man[o + 15 * 4 + q].toFixed(2) });
        row.contacts.push({ a, b, ma: a > 0 ? (bod[a * 20 + 3] > 0 ? Math.round(1 / bod[a * 20 + 3]) : 0) : 0, mb: bod[b * 20 + 3] > 0 ? Math.round(1 / bod[b * 20 + 3]) : 0, n: r3(man, o + 4), cnt, pts, e: +man[o + 64].toFixed(2), J: Math.round(man[o + 65]), ap: +man[o + 66].toFixed(2) });
      }
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
    const quietNow = (gpu.stats.awake || 0) === 0 && !(gpu.stats.pActive > 0) && !busy && !L.jobq.length && L.f > L.quietFrom;
    if (quietNow && L.settledAt == null) L.settledAt = L.f; else if (!quietNow) L.settledAt = null;
    if (L.f % 10 === 0 || L.f === S.frames - 1) {
      let roof = 0;
      for (let x = -9; x <= 9; x += 3) for (let z = -9; z <= 9; z += 3) { const ts = E.raycast(x + 0.3, 200, z + 0.3, 0, -1, 0, 300); if (ts >= 0) roof = Math.max(roof, 200 - ts); }
      if (snapCur) for (let k = 0; k < snapCur.pose.length / 10; k++) { const y = snapCur.pose[k * 10 + 1] + snapCur.pose[k * 10 + 8]; if (Number.isFinite(y) && Math.abs(snapCur.pose[k * 10]) < 12 && Math.abs(snapCur.pose[k * 10 + 2]) < 12) roof = Math.max(roof, y); }
      L.roofMin = Math.min(L.roofMin ?? 1e9, L.f > 10 ? roof : 1e9);
      L.series.push({ f: L.f, D: +L.D.toFixed(2), T: +L.T.toFixed(2), drawn: L.draw.n, bodies: gpu.n, awake: gpu.stats.awake, contacts: gpu.stats.manifolds, parts: gpu.stats.pLive || 0, pAwake: gpu.stats.pActive || 0, sub: gpu.substeps, gpu: +g.toFixed(2), js: +L.jsMs[L.jsMs.length - 1].toFixed(2), roof: +roof.toFixed(2), upper: +labUpper().toFixed(1), breaks, dust: dustBricks, failed: failedJoints, cracked: crackedJoints, worn: wornJoints, wearing: structWearing });
    }
    if (performance.now() - lastShow > 1000) { lastShow = performance.now(); $('bench-out').textContent = `Lab: ${name} · frame ${L.f} of ${S.frames} · ${((performance.now() - tw) / 1000).toFixed(0)} s\nholes ${L.holeFrames} · duplicates ${L.dupFrames} · flickers ${L.flickers} · jumps ${L.teleports} · bounces ${L.bounces} · launches ${L.launches}`; L.status = { f: L.f, frames: S.frames }; }
  }
  const rep = labReport((performance.now() - tw) / 1000);
  L.on = false; Math.random = mathRandom;
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
