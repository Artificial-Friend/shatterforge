// ---- lab: scripted scenarios on a virtual clock, checked as data (not screenshots)
// Every lab frame runs the page's own logic (engine jobs, physics steps, read-backs, the instance lists handed to
// the renderer) at a simulated 60 fps, however long the GPU takes: a software GPU and a fast one go through the
// same frames. Read-backs arrive LAB.readLag frames after they were issued and engine jobs LAB.jobLag frames
// after they were sent, as on a real machine. What would be drawn is checked every frame; the physics at every
// read-back. The result is a report of checks (pass/fail with limits), events and a time series.
function labTruthStatic() { let v = 0; for (let k = 0; k < E.chunk_count(); k++) v += E.chunk_volume(k); return v; }
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
      if (L.ballNext) { L.balls.add(b); if (gb) gb.labBall = true; return; }
      const v = list[k].shapes.reduce((s, sh) => s + labBoxVol(sh.h) * (sh.hull ? GpuWorld.hulls.hulls[sh.hull].vol : 1), 0); L.bodyVol.set(b, v); L.gpuVol += v;
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
    const v = L.bodyVol.get(parent) || 0; L.gpuVol -= v; L.bodyVol.delete(parent);
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
    if (L.bodyVol.has(i) && moved > 0) { L.bodyVol.set(i, L.bodyVol.get(i) - moved); L.bodyVol.set(j, moved); const gb = gpu.bodies[j - 1]; if (gb) { gb.labVol = moved; gb.labF = L.f; gb.labH = 'spare'; } }
    return r;
  };
  gpu.killBody = (b) => { const v = L.bodyVol.get(b) || 0; L.gpuVol -= v; L.bodyVol.delete(b); return kill(b); };
  gpu.addParticles = (list) => { const s = addP(list); L.pendingParts = (L.pendingParts || []).concat(s); return s; };
  gpu.killParticle = (i) => { const v = L.partVol.get(i) || 0; L.gpuVol -= v; L.partVol.delete(i); return killP(i); };
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
    now.set(b, { gb, vy, e, ek, v: [vx, vy, vz] });
    // the same body as at the last read-back (a reused slot holds another one)
    const was = L.prevBodies && L.prevBodies.get(b);
    if (!was || was.gb !== gb) continue;
    gain += e - was.e; ekPrev += was.ek;
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
  if (quiet && L.prevBodies && gain > 0.05 * ekPrev + 2000 + given) { L.energyGains++; L.energyGainMax = Math.max(L.energyGainMax, ekPrev > 0 ? gain / ekPrev : 0); labEvent('energy_gain', f, { gain_J: Math.round(gain), kinetic_before_J: Math.round(ekPrev) }); }
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
    while (L.jobq.length) { const j = L.jobq.shift(); finish(runInline(j.job)); labParticlesMade(); }
    L.staticT = labTruthStatic(); labTruth();
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
    ['rebounds over half the impact speed (while quiet)', L.bounces, 0, 'unnatural bouncing'],
    ['pieces thrown up faster than 5 m/s (while quiet)', L.launches, 0, 'launches'],
    ['energy gains over 5% (while quiet)', L.energyGains, 0, 'energy made from nothing'],
  ].map(([name, value, limit, what]) => ({ name, value, limit, pass: value <= limit, what }));
  if (S.expect && S.expect.roofDrop) checks.push({ name: 'roof fall (m)', value: +(roof0 - (last.roof || 0)).toFixed(2), limit: `>= ${S.expect.roofDrop}`, pass: roof0 - (last.roof || 0) >= S.expect.roofDrop, what: 'the cut storey brings the building down' });
  if (S.expect && S.expect.settle) checks.push({ name: 'settled (s)', value: L.settledAt == null ? 'no' : +(L.settledAt / 60).toFixed(1), limit: `by ${(S.frames / 60).toFixed(0)}`, pass: L.settledAt != null, what: 'everything asleep before the end' });
  const gpuMs = se.map(x => x.gpu), jsMs = L.jsMs;
  return {
    scenario: L.name, shortened: S.frames !== LAB_SCENARIOS[L.name].frames, title: S.title, kind: ['concrete frame', 'masonry', 'mass timber', 'mixed'][S.kind], storeys: S.storeys, frames: S.frames,
    readLag: L.readLag, jobLag: L.jobLag, settings: { ...setting }, ua: navigator.userAgent, adapter: L.adapter, date: new Date().toISOString(),
    pass: checks.every(c => c.pass), checks,
    physics: { roof_m: [+roof0.toFixed(2), +(last.roof || 0).toFixed(2)], dust_of_building: +(1 - (L.T) / L.V0).toFixed(3), particles_made: partsMade, particles_crushed: partsCrushed, bodies_crushed: crushedBodies, breaks, failed_joints: failedJoints, settled_s: L.settledAt == null ? null : +(L.settledAt / 60).toFixed(2), bounce_max: +L.bounceMax.toFixed(2), launch_max: +L.launchMax.toFixed(2), hole_max_m3: +L.holeMax.toFixed(3), dup_max_m3: +L.dupMax.toFixed(3), energy_given_by_cuts_J: Math.round(L.givenTotal || 0), rubble_on_floors_max_t: +((L.restMax || 0) / 9810).toFixed(1), rubble_sent_bodies_particles_t: L.restSplit || null, joints_cracked: L.cracked || 0, gpu_breaks: gpuSnaps },
    perf: {
      wall_s: +wallS.toFixed(1), steps: stepsDone, steps_skipped: gpu.skipped || 0,
      gpu_ms: { median: +med(gpuMs).toFixed(2), p95: +p95(gpuMs).toFixed(2), max: +Math.max(0, ...gpuMs).toFixed(2) },
      gpu_stages_ms_at_peak: L.peakStages,
      js_ms: { median: +med(jsMs).toFixed(2), p95: +p95(jsMs).toFixed(2), max: +Math.max(0, ...jsMs).toFixed(2) },
      engine_ms: { jobs: engineMsLog.length, median: +med(engineMsLog.slice()).toFixed(1), p95: +p95(engineMsLog.slice()).toFixed(1), max: +Math.max(0, ...engineMsLog).toFixed(1) },
      peak: { bodies: Math.max(0, ...se.map(x => x.bodies)), awake: Math.max(0, ...se.map(x => x.awake)), contacts: Math.max(0, ...se.map(x => x.contacts)), particles: Math.max(0, ...se.map(x => x.parts)), particles_moving: Math.max(0, ...se.map(x => x.pAwake)), drawn: Math.max(0, ...se.map(x => x.drawn)) },
      work: { contact_substeps: L.work.cs, particle_substeps: L.work.ps },
    },
    events: L.events, timeline: L.timeline, trace: L.trace.length ? L.trace : undefined, bursts: L.bursts || undefined, traceEvents: L.traceEv.length ? L.traceEv : undefined,
    series: se,
  };
}
function labSummary(rep) {
  const lines = [`${rep.pass ? 'PASS' : 'FAIL'} ${rep.scenario}: ${rep.title}`];
  for (const c of rep.checks) lines.push(`  ${c.pass ? 'ok  ' : 'FAIL'} ${c.name}: ${c.value} (limit ${c.limit})${!c.pass && rep.events ? '' : ''}`);
  const p = rep.physics, q = rep.perf;
  lines.push(`  physics: roof ${p.roof_m[0]} -> ${p.roof_m[1]} m · dust ${(p.dust_of_building * 100).toFixed(1)}% · breaks ${p.breaks} · crushed ${p.bodies_crushed} · particles ${p.particles_made} (${p.particles_crushed} crushed) · settled ${p.settled_s ?? 'no'} s · energy given to parts cut free ${((p.energy_given_by_cuts_J || 0) / 1e6).toFixed(2)} MJ · rubble on floors up to ${p.rubble_on_floors_max_t} t · ${p.joints_cracked} joints cracked · ${p.gpu_breaks} broken on the GPU`);
  lines.push(`  perf: ${q.steps} steps (${q.steps_skipped} skipped) in ${q.wall_s} s · GPU ms median ${q.gpu_ms.median} / p95 ${q.gpu_ms.p95} / max ${q.gpu_ms.max} · JS ms median ${q.js_ms.median} / p95 ${q.js_ms.p95} · engine jobs ${q.engine_ms.jobs}, ms median ${q.engine_ms.median} / p95 ${q.engine_ms.p95} / max ${q.engine_ms.max} · peak awake ${q.peak.awake}, contacts ${q.peak.contacts}, particles ${q.peak.particles}`);
  if (q.gpu_stages_ms_at_peak) lines.push(`  GPU stages at the slowest step (ms): ${Object.entries(q.gpu_stages_ms_at_peak).map(([k, v]) => `${k} ${v}`).join(' · ')}`);
  for (const [k, e] of Object.entries(rep.events)) lines.push(`  events ${k}: ${e.n} · first ${JSON.stringify(e.first.slice(0, 3))}`);
  return lines.join('\n');
}
async function labRun() {
  const L = LAB, st = LABSTATE, name = st.queue[0], S = { ...LAB_SCENARIOS[name] };
  if (st.frames) S.frames = st.frames;   // (a shortened run, for quick checks of the lab itself)
  Object.assign(L, {
    on: true, name, spec: S, t: 0, f: 0, readLag: st.readLag ?? 2, jobLag: st.jobLag ?? 4, render: !!st.render, gates: [], jobq: [],
    draw: { n: 0, vol: 0, keys: new Float64Array(4096), pos: new Float32Array(4096 * 3), bound: new Float32Array(4096) },
    Thist: [], holeFrames: 0, holeMax: 0, inHole: false, holesRecovered: 0, dupFrames: 0, dupMax: 0, flickers: 0, teleports: 0, goneAt: new Map(), prevDraw: null,
    bad: 0, below: 0, bounces: 0, bounceMax: 0, launches: 0, launchMax: 0, energyGains: 0, energyGainMax: 0, prevBodies: null, quietFrom: -1,
    injectedAt: -1000, settledAt: null, timeline: [], trace: [], traceEv: [], traceWin: st.trace || null, events: {}, series: [], jsMs: [], work: (engineMsLog.length = 0, { cs: 0, ps: 0 }), peakStages: null, peakGpu: 0, volTol: 0.05,
  });
  // the defaults, whatever this browser has saved (the report lists them)
  for (const [k, d] of Object.entries(SETTINGS)) { setting[k] = d.def; d.apply(d.def); }
  setting.preset = 'high (lab defaults)';
  // fixed sub-steps: adapting them to this machine's speed would make the runs differ between machines
  setting.adaptive = 0; gpu.substeps = setting.substeps;
  Math.random = seededRandom(20261104);
  fly = true; E.player_set(-20, 8, -30); yaw = Math.PI + 0.62; pitch = -0.12;
  $('start').hidden = true; physicsPaused = false;
  $('bench').hidden = false; $('bench-title').textContent = `Lab: ${name} (${st.done.length + 1} of ${st.done.length + st.queue.length})`;
  labInstall();
  L.staticT = labTruthStatic(); labTruth(); L.V0 = L.T; L.Thist = [L.T]; L.staticDrawn = Object.values(chunkVol).reduce((a, b) => a + b, 0);
  try { const info = dev.adapterInfo || (dev.adapter && dev.adapter.info); L.adapter = info ? `${info.vendor} ${info.architecture} ${info.description || ''}`.trim() : ''; } catch (e) {}
  const tw = performance.now();
  let lastShow = 0;
  for (L.f = 0; L.f < S.frames; L.f++) {
    L.t += 1000 / 60;
    for (const [fr, act, args] of S.script) if (fr === L.f) await labAct(act, args);
    // engine jobs that are due (the worker's time)
    while (L.jobq.length && L.jobq[0].due <= L.f) {
      const j = L.jobq.shift(), nb0 = gpu.n, b0 = breaks, r = runInline(j.job);
      finish(r); labParticlesMade(); L.staticT = labTruthStatic(); labTruth();
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
    // let the GPU finish this frame's work and its stats come back (the next frame sees them)
    if (dev) { await dev.queue.onSubmittedWorkDone(); await new Promise(r => setTimeout(r, 0)); await new Promise(r => setTimeout(r, 0)); }
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
      L.series.push({ f: L.f, D: +L.D.toFixed(2), T: +L.T.toFixed(2), drawn: L.draw.n, bodies: gpu.n, awake: gpu.stats.awake, contacts: gpu.stats.manifolds, parts: gpu.stats.pLive || 0, pAwake: gpu.stats.pActive || 0, sub: gpu.substeps, gpu: +g.toFixed(2), js: +L.jsMs[L.jsMs.length - 1].toFixed(2), roof: +roof.toFixed(2), breaks, dust: dustBricks });
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
