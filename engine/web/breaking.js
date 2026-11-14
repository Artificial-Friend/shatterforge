// ---- breaking on the GPU in the step of a hit (GpuWorld.setSnap). The engine sends with each new falling piece its
// break plan: the impulse in one step that breaks it, and the bricks of the part that comes away. The GPU breaks it
// in the step of the hit; the page then tells the engine (a 'snap' job, first in line), which cuts the plan's joints
// and answers with the two pieces: the one kept by the body, and the one the spare body took.
const SNAP_ON = !window.__noSnap;
let jobPlans = new Map(), gpuSnaps = 0;
const engineBroken = new Map();   // body -> when the engine broke it (see takeGpuSnaps)
const snapPending = new Map();   // piece -> { body, spare, infB } until the engine's answer
function readPlans(r) {
  jobPlans = new Map(); const f = r.plans; if (!f) return;
  for (let o = 0; o + 2 < f.length;) { const n = f[o + 2]; jobPlans.set(f[o], { imp: f[o + 1], pos: Array.from(f.subarray(o + 3, o + 3 + n)) }); o += 3 + n; }
}
function armPlan(idx, pid) {
  const pl = jobPlans.get(pid); if (!pl || !SNAP_ON || !gpu || !gpu.setSnap) return;
  const own = gpu.bodyShapes && gpu.bodyShapes.get(idx); if (!own) return;
  const other = pl.pos.map(k => own[k]).filter(Boolean);
  if (other.length === pl.pos.length) gpu.setSnap(idx, pl.imp, other);
}
function takeGpuSnaps() {
  if (!gpu.takeSnaps) return;
  for (const s of gpu.takeSnaps()) {
    const inf = bodyInfo.get(s.body);
    // the engine broke the whole piece in the frame the GPU broke it by its plan (its answer came first): the
    // engine's kids hold the spare's bricks too, so the spare goes. Left, it was a second copy of that part (the
    // lab's mixed scenario: 21 slabs, 186 m3 of 1,542, doubled at frame 72 for good)
    if (!inf) { const eb = engineBroken.get(s.body); if (eb !== undefined && clock() - eb < 2000) { const sp = gpu.bodies[s.spare - 1]; if (sp && !sp.dead) gpu.killBody(s.spare); } continue; }
    // a part's frame: the body's at its creation moved by the part's centre and rotation (the same tower-to-world
    // map, see frameOf)
    const fr = (part) => { const d = mv(q2m(inf.q0), part.p); return { ...inf, p0: [inf.p0[0] + d[0], inf.p0[1] + d[1], inf.p0[2] + d[2]], q0: qmulJ(inf.q0, part.q) }; };
    bodyInfo.set(s.body, fr(s.A));
    snapPending.set(inf.piece, { body: s.body, spare: s.spare, infB: fr(s.B), genA: gpu.genOf ? gpu.genOf(s.body) : 0, genB: gpu.genOf ? gpu.genOf(s.spare) : 0 });
    // (all of a frame's breaks in one job, first in line; the impacts after it still get their turn)
    const j = jobs.blast[0] && jobs.blast[0].op === 'snap' ? jobs.blast[0] : (jobs.blast.unshift({ op: 'snap', pieces: [] }), jobs.blast[0]);
    j.pieces.push(inf.piece);
    gpuSnaps++;
  }
  pump();
}
function onSnap(job, r) {
  const f = r.out; let o = 0;
  for (const pc of job.pieces) {
    const pend = snapPending.get(pc); snapPending.delete(pc);
    const n = f[o++];
    // (the engine broke the piece first: its parts already include the spare's bricks)
    if (!(n >= 2)) { if (pend && (!gpu.genOf || gpu.genOf(pend.spare) === pend.genB)) gpu.killBody(pend.spare); continue; }
    const idA = f[o], nA = f[o + 1]; o += 2 + 7 * nA;
    const idB = f[o], nB = f[o + 1]; o += 2 + 7 * nB;
    if (!pend) continue;
    // (either part may have been broken or crushed while the engine worked, and its slot given to another body:
    // the answer is then not about that slot any more. A stale entry broke a spare made later in slot 98.)
    const live = (b, g) => { const gb = gpu.bodies[b - 1]; return gb && !gb.dead && (!gpu.genOf || gpu.genOf(b) === g); };
    const inf = bodyInfo.get(pend.body);
    if (inf && live(pend.body, pend.genA)) { inf.piece = idA; inf.n = nA; }
    if (live(pend.spare, pend.genB)) bodyInfo.set(pend.spare, { ...pend.infB, piece: idB, n: nB });
    // (the parts get no plan of their own: the engine's answer to the hit refines the break from here)
  }
}
function readStatic(f) {
  let o = 0;
  const nu = f[o++], updates = [];
  for (let k = 0; k < nu; k++, o += 9) updates.push(f.subarray(o, o + 9));
  const sb = readBodies(f, o);
  return { updates, bodies: sb.bodies, o: sb.o };
}

// The ray against the debris as drawn (oriented boxes): the nearest hit, as { t, slot }
// The debris near the player, from the latest read-back, to the engine's walker as oriented boxes: it stands on a
// broken floor, a falling slab, a heap of rubble, and walks into them (it saw only the standing bricks, and fell
// through every floor that broke loose). Particles are left out (crumbs: it walks over them).
function playerDebris() {
  if (!E.player_debris_clear) return;
  E.player_debris_clear();
  const r = snapCur; if (!r || !gpu) return;
  const px = E.player_get(0), py = E.player_get(1) + 0.9, pz = E.player_get(2);
  const n = r.pose.length / 10;
  for (let k = 0; k < n; k++) {
    const b = k * 10, x = r.pose[b], y = r.pose[b + 1], z = r.pose[b + 2], h0 = r.pose[b + 7], h1 = r.pose[b + 8], h2 = r.pose[b + 9];
    const reach = 1.5 + Math.hypot(h0, h1, h2);
    if (!(Math.abs(x - px) < reach && Math.abs(y - py) < reach + 1 && Math.abs(z - pz) < reach)) continue;
    if (!Number.isFinite(x + y + z + r.pose[b + 6])) continue;
    E.player_debris_box(x, y, z, r.pose[b + 3], r.pose[b + 4], r.pose[b + 5], r.pose[b + 6], h0, h1, h2);
  }
}
function rayDebris(o, d, maxT) {
  const r = snapCur; if (!r) return null;
  let best = null;
  const n = r.pose.length / 10;
  for (let k = 0; k < n; k++) {
    const b = k * 10, p = [r.pose[b], r.pose[b + 1], r.pose[b + 2]], h = [r.pose[b + 7], r.pose[b + 8], r.pose[b + 9]];
    if (!(Number.isFinite(p[0]) && Number.isFinite(p[1]) && Number.isFinite(p[2]) && Number.isFinite(r.pose[b + 6]))) continue;
    // quick reject: the ray against the bounding sphere
    const ox = o[0] - p[0], oy = o[1] - p[1], oz = o[2] - p[2], rad = Math.hypot(h[0], h[1], h[2]);
    const bq = ox * d[0] + oy * d[1] + oz * d[2], cq = ox * ox + oy * oy + oz * oz - rad * rad;
    if (cq > 0 && bq > 0) continue;
    if (bq * bq - cq < 0) continue;
    // the slab test in the box's own frame
    const Rt = mt(q2m([r.pose[b + 3], r.pose[b + 4], r.pose[b + 5], r.pose[b + 6]]));
    const lo = mv(Rt, [ox, oy, oz]), ld = mv(Rt, d);
    let t0 = 0, t1 = best ? best.t : maxT, ok = true;
    for (let a = 0; a < 3 && ok; a++) {
      if (Math.abs(ld[a]) < 1e-9) { if (Math.abs(lo[a]) > h[a]) ok = false; continue; }
      let ta = (-h[a] - lo[a]) / ld[a], tb = (h[a] - lo[a]) / ld[a];
      if (ta > tb) [ta, tb] = [tb, ta];
      t0 = Math.max(t0, ta); t1 = Math.min(t1, tb); if (t0 > t1) ok = false;
    }
    if (ok && Number.isFinite(t0)) best = { t: t0, slot: r.slots[k] };
  }
  return best;
}
// A blast throws the debris near it outward (strongest at the centre, nothing beyond 2.5 radii)
function blastWave(c, r) {
  if (gpu && gpu.blastParticles) gpu.blastParticles(c, 2.5 * r, 6 * r);
  if (!gpu || !lastBodies) return;
  const reach = 2.5 * r;
  for (let i = 2; i <= gpu.n; i++) {
    const o = (i - 1) * 7; if (o + 6 >= lastBodies.length) break;
    const b = gpu.bodies[i - 1]; if (!b || b.dead) continue;
    const p = [lastBodies[o], lastBodies[o + 1], lastBodies[o + 2]];
    const dx = p[0] - c[0], dy = p[1] - c[1], dz = p[2] - c[2], dist = Math.hypot(dx, dy, dz);
    if (dist > reach || dist < 1e-6) continue;
    const s = 6 * (1 - dist / reach) * r;   // m/s
    const v0 = velOf(i), v = [v0[0] + dx / dist * s, v0[1] + dy / dist * s + 0.3 * s, v0[2] + dz / dist * s];
    gpu.setVelocity(i, v, [0, 1, 2].map(() => (Math.random() - 0.5) * 2), p, [lastBodies[o + 3], lastBodies[o + 4], lastBodies[o + 5], lastBodies[o + 6]]);
  }
  lastActive = clock();
}
function doBlast(o0, d0) {
  const d = d0 || new THREE.Vector3(); if (!d0) cam.getWorldDirection(d);
  const o = o0 || cam.position;
  const ts = E.raycast(o.x, o.y, o.z, d.x, d.y, d.z, 200);
  const hit = gpu ? rayDebris([o.x, o.y, o.z], [d.x, d.y, d.z], ts >= 0 ? ts : 200) : null;
  if (hit) {
    // debris is nearer than the building: a piece breaks, loose rubble is thrown
    const p = o.clone().addScaledVector(d, hit.t + 0.05), c = [p.x, p.y, p.z];
    const sh = gpu.shapes[hit.slot - 1], idx = sh ? sh.body : 0;
    if (bodyInfo.has(idx)) {
      const xf = frameOf(idx), Rt = mt(xf.R);
      const sp = mv(Rt, [c[0] - xf.T[0], c[1] - xf.T[1], c[2] - xf.T[2]]).map((v, k) => v + xf.C[k]);
      const v = velOf(idx), w = angOf(idx);
      const ctx = new Map([[idx, { J: [0, 0, 0], vPre: v, vPost: v, w0: w }]]);
      jobs.blast.push({ op: 'pblast', piece: bodyInfo.get(idx).piece, x: sp[0], y: sp[1], z: sp[2], r: radius, blastC: c, ctx });
    }
    blastWave(c, radius);
    lastBlast = bodyInfo.has(idx) ? `blast on a falling piece` : `blast in the rubble`;
    pump();
    return;
  }
  if (ts < 0) return;
  const p = o.clone().addScaledVector(d, ts + 0.05);
  blastWave([p.x, p.y, p.z], radius);
  jobs.blast.push({ op: 'blast', x: p.x, y: p.y, z: p.z, r: radius, ctx: { d: [d.x, d.y, d.z], t0: performance.now() } });
  pump();
}
function onBlast(job, r) {
  const { updates, bodies } = readStatic(r.out);
  remeshDirty();
  const d = job.ctx.d;
  if (gpu) {
    applyStatic(updates);
    // the hit knocks the crumbled material out along the shot (2-4 m/s, a little spread); pieces that only
    // lost their support start at rest
    addPieces(bodies, null, (c, b) => {
      if (b.id >= 0) return { v: [0, 0, 0], w: [0, 0, 0] };
      const s = 2 + 2 * Math.random();
      return { v: [d[0] * s + (Math.random() - 0.5), d[1] * s + (Math.random() - 0.5), d[2] * s + (Math.random() - 0.5)], w: [0, 1, 2].map(() => (Math.random() - 0.5) * 6) };
    });
  }
  const pieces = bodies.filter(b => b.id >= 0).length;
  lastActive = clock();
  jobs.structure = true;
  lastBlast = `${bodies.length - pieces} crumbs, ${pieces} pieces · ${r.ms.toFixed(1)} ms in the engine, ${(performance.now() - job.ctx.t0).toFixed(0)} ms to the screen`;
}
// the standing structure under its own weight: overloaded joints fail and cracks open, round after round
function onStructure(r) {
  structRounds++;
  if (r.wearing > 0) wearNext = clock() + WEAR_EVERY;
  structWearing = r.wearing || 0; wornJoints = r.worn || 0;
  // (a structure still moving is stepped again)
  if (r.moving) jobs.structure = true;
  if (r.n === 0 && !r.detached) return;
  // (another round only after failures: pieces sent off with none, loose contents, need none)
  if (r.n > 0) jobs.structure = true;
  failedJoints += r.n;
  const { updates, bodies } = readStatic(r.out);
  remeshDirty();
  if (gpu) { applyStatic(updates); addPieces(bodies, null, null, 150); }
  lastActive = clock();
  lastBlast = `structure: ${r.n} failures this round (${r.ms.toFixed(0)} ms in the engine)`;
}

// Hard impacts, both sides of every contact: the impact energy crushes material on both bodies, the standing
// structure takes the contact force as a load, a falling piece cracks under its inertia, and when stopping it
// takes more than its weakest section near the contacts can carry, that section gives way and what lies
// beyond keeps falling (a crush-down, storey by storey). Events gather while the engine is busy.
const PER_BODY = 32, STATIC_HITS = 96;
// Pieces crushed on the GPU (pressed by a body many times heavier, see the crush kernel) become rubble
// particles where they are: each brick split into cubes of at most 0.4 m (what does not fit is dust).
let crushedBodies = 0;
const qmul = (a, b) => [a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1], a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0], a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3], a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]];
const qrot = (q, v) => qmul(qmul(q, [...v, 0]), [-q[0], -q[1], -q[2], q[3]]).slice(0, 3);
const vadd = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
function crushBodies() {
  if (!gpu.takeCrushed) return;
  for (const c of gpu.takeCrushed()) {
    const idx = c.body, gb = gpu.bodies[idx - 1];
    if (!gb || gb.dead) continue;
    const list = [], info = [];
    for (const slot of gpu.bodyShapes.get(idx) || []) {
      const sh = gpu.shapes[slot - 1]; if (!sh || sh.dead) continue;
      const qs = qmul(c.q, sh.lq || [0, 0, 0, 1]), cen = vadd(c.p, qrot(c.q, sh.lp));
      const n = sh.h.map(h => Math.max(1, Math.ceil(2 * h / 0.4))), hs0 = sh.h.map((h, k) => h / n[k]);
      // (a chunk fills only part of its box: its rubble cubes are shrunk to its volume)
      const shrink = sh.hull ? Math.cbrt(GpuWorld.hulls.hulls[sh.hull].vol) : 1, hs = hs0.map(x => x * shrink);
      for (let i = 0; i < n[0]; i++) for (let j = 0; j < n[1]; j++) for (let k = 0; k < n[2]; k++) {
        const lp = [(2 * i + 1 - n[0]) * hs0[0], (2 * j + 1 - n[1]) * hs0[1], (2 * k + 1 - n[2]) * hs0[2]];
        list.push({ p: vadd(cen, qrot(qs, lp)), v: [0, 0, 0], q: qs, r: Math.cbrt(6 * hs[0] * hs[1] * hs[2] / Math.PI), mat: sh.mat, wait: true });
        info.push({ h: hs, col: sh.col ?? sh.mat });
      }
    }
    const room = gpu.roomParticles(), keep = list.slice(0, room);
    const parts = gpu.addParticles(keep); partsBy.crushed = (partsBy.crushed || 0) + keep.length;
    parts.forEach((slot, k) => { if (slot < 0) return; const g = pInfo[slot] ? pInfo[slot].gen + 1 : 1; pInfo[slot] = { h: info[k].h, col: info[k].col, gen: g, dead: false }; partsMade++; });
    gpu.breakBody(idx, [], c.p, c.q, [], parts, parts.map(() => [0, 0, 0]));
    bodyInfo.delete(idx); crushedBodies++;
  }
}
function impacts() {
  takeGpuSnaps();
  crushBodies();
  const evs = gpu.takeEvents();
  if (!evs.length) return;
  const now = clock();
  const matOf = (slot) => { const sh = slot > 0 ? gpu.shapes[slot - 1] : null; return sh ? sh.mat : -1; };
  const sides = new Map();
  for (const e of evs) for (const [idx, sgn, oslot] of [[e.a, -1, e.sb], [e.b, 1, e.sa]]) {
    if (idx !== 1 && !bodyInfo.has(idx)) continue;           // crumbs and the ground take no damage
    if (idx !== 1 && bodyInfo.get(idx).cool > now) continue;
    const om = matOf(oslot);
    if (!sides.has(idx)) sides.set(idx, []);
    sides.get(idx).push({ e, sgn, other: om < 5 ? om : -1 });
  }
  if (!sides.size) return;
  const job = jobs.impacts || (jobs.impacts = { op: 'impacts', dt: 1 / 60, recs: [], ctx: new Map() });
  job.dt = Math.max(job.dt, lastStepDt);   // (forces from impulses: over the step they came from)
  const recs = [];
  for (const [idx, list] of sides) {
    let c = job.ctx.get(idx);
    if (!c) {
      if (idx !== 1) {
        const vo = (idx - 1) * 3;
        // the velocity before the hit: events arrive a read-back or two late, so take the fastest of the last
        // few read-backs (the hit only slows the body)
        let pre = null, best = -1;
        const inf = bodyInfo.get(idx);
        for (const { v, frame } of velHist) if (vo + 2 < v.length && frame > inf.f0) { const sp = Math.hypot(v[vo], v[vo + 1], v[vo + 2]); if (sp > best) { best = sp; pre = [v[vo], v[vo + 1], v[vo + 2]]; } }
        // a body made a few steps ago (a piece cut free) has little history: its starting velocity counts too
        if (inf.v0 && gpu.frame - inf.f0 <= 8) { const sp = Math.hypot(...inf.v0); if (sp > best) { best = sp; pre = inf.v0.slice(); } }
        c = { J: [0, 0, 0], n: 0, vPre: pre, vPost: velOf(idx), w0: angOf(idx) };
      } else c = { n: 0 };
      job.ctx.set(idx, c);
    }
    const cap = (idx === 1 ? STATIC_HITS : PER_BODY) - c.n;
    if (cap <= 0) continue;
    // rubble pelting a big piece barely changes its motion and does not break it (re-making a body of
    // thousands of bricks for a few crumbs every frame would be waste, too)
    if (idx !== 1 && c.vPre && lastVel && bodyInfo.get(idx).n > 100) {
      const vn = velOf(idx), dv = Math.hypot(c.vPre[0] - vn[0], c.vPre[1] - vn[1], c.vPre[2] - vn[2]);
      // ...unless one hit carries enough energy into one spot to crush there (a ball, a falling block): 100 kJ
      let eMax = 0; for (const { e } of list) eMax = Math.max(eMax, 0.5 * e.impulse * e.approach);
      if (dv < 0.5 && eMax < 1e5) { if (!c.n) job.ctx.delete(idx); continue; }
    }
    list.sort((x, y) => y.e.impulse * y.e.approach - x.e.impulse * x.e.approach);
    const keep = list.slice(0, cap);
    c.n += keep.length;
    let xf = null, Rt = null;
    if (idx !== 1) { xf = frameOf(idx); Rt = mt(xf.R); }
    // only the strongest contacts are sent, and the GPU reports only those above the event thresholds: for a
    // piece, scale their impulses up to its real change of momentum, M |v_before - v_now|
    let scale = 1;
    if (idx !== 1 && c.vPre && lastVel) {
      const vo = (idx - 1) * 3, M = gpu.bodies[idx - 1].mass || 0;
      const vn = velOf(idx), dv = Math.hypot(c.vPre[0] - vn[0], c.vPre[1] - vn[1], c.vPre[2] - vn[2]);
      let jk = 0; for (const { e } of keep) jk += e.impulse;
      if (jk > 0) scale = Math.min(50, Math.max(1, M * dv / jk));
      if (window.__dbg) console.warn(`impact on body ${idx} (piece ${bodyInfo.get(idx).piece}, ${(M / 1000).toFixed(0)} t): ${list.length} events, ${keep.length} kept, approach up to ${Math.max(...keep.map(k => k.e.approach)).toFixed(1)} m/s, dv ${dv.toFixed(2)} m/s, impulse x${scale.toFixed(1)}`);
    }
    for (const { e, sgn, other } of keep) {
      const dw = [sgn * e.n[0], sgn * e.n[1], sgn * e.n[2]];
      const imp = e.impulse * scale;
      if (c.J) for (let k = 0; k < 3; k++) c.J[k] += dw[k] * imp;
      let p = e.p, d = dw;
      if (xf) { p = mv(Rt, [e.p[0] - xf.T[0], e.p[1] - xf.T[1], e.p[2] - xf.T[2]]).map((v, k) => v + xf.C[k]); d = mv(Rt, dw); }
      recs.push(idx === 1 ? -1 : bodyInfo.get(idx).piece, p[0], p[1], p[2], d[0], d[1], d[2], imp, e.approach, other, 0);
    }
  }
  job.recs = job.recs.concat(recs);
  pump();
}
function onImpacts(job, r) {
  impactMs = r.ms;
  const f = r.out;
  // 1. the standing structure: crushed bricks, failures, what came loose
  const st = readStatic(f);
  let o = st.o;
  if (st.updates.length || st.bodies.length) {
    remeshDirty();
    applyStatic(st.updates);
    // crushed material sprays out a little; what came loose starts where the structure had it and as it moved
    // there (motionAt: it started as built and at rest, and a part that had sagged 20 cm jumped back up as it let go)
    addPieces(st.bodies, null, (c, b) => b.id >= 0 ? motionAt(c) : { v: [(Math.random() - 0.5) * 3, Math.random() * 2, (Math.random() - 0.5) * 3], w: [0, 1, 2].map(() => (Math.random() - 0.5) * 8) });
    jobs.structure = true;
  }
  // 2. broken pieces
  const pieceBody = new Map(); for (const [idx, inf] of bodyInfo) pieceBody.set(inf.piece, idx);
  const nbr = f[o++];
  for (let q = 0; q < nbr; q++) {
    const parent = f[o++], nk = f[o++], cap = f[o++];
    const kids = [];
    for (let k = 0; k < nk; k++) {
      const id = f[o++], hit = f[o++] > 0.5, n = f[o++], boxes = [];
      for (let b = 0; b < n; b++, o += 7) boxes.push(f.subarray(o, o + 7));
      kids.push({ id, hit, boxes });
    }
    // (lab: how much of a big falling part leaves as crumbs and particles when it is broken)
    if (LAB.on) { let all = 0, fine = 0; for (const kd of kids) for (const bx of kd.boxes) { const v = bx[3] * bx[4] * bx[5] * TICK ** 3; all += v; if (kd.id < 0) fine += v; } if (all >= 50) (LAB.bigBreaks = LAB.bigBreaks || []).push([LAB.f, +all.toFixed(0), +(fine / all).toFixed(3)]); }
    const idx = pieceBody.get(parent);
    if (idx === undefined) continue;
    if (LAB.on && (LAB.dbgBreaks = LAB.dbgBreaks || []).length < 40) { const bi = bodyInfo.get(idx); let nb = 0, vb = 0; for (const kd of kids) for (const bx of kd.boxes) { nb++; vb += bx[3] * bx[4] * bx[5] * TICK ** 3; } LAB.dbgBreaks.push({ f: LAB.f, piece: parent, body: idx, n_when_made: bi && bi.n, kid_boxes: nb, kid_box_m3: +vb.toFixed(1), kids: kids.length, shapes_gpu: (gpu.bodyShapes.get(idx) || []).length, pend: snapPending.get(parent) ? { spare: snapPending.get(parent).spare, shapes: (gpu.bodyShapes.get(snapPending.get(parent).spare) || []).length } : null, plan: gpu.planOf && gpu.planOf.get(idx) ? { spare: gpu.planOf.get(idx).spare, done: gpu.planOf.get(idx).done } : null }); }
    const c = job.ctx.get(idx) || { J: [0, 0, 0], vPost: [0, 0, 0], w0: [0, 0, 0] };
    const xf = frameOf(idx);
    const M = gpu.bodies[idx - 1].mass || 1;
    // (else: the contacts pushed the body by sum(J d) / M; only the strongest contacts are counted, so this
    // underestimates a hit on many contacts)
    const vPre = c.vPre || [0, 1, 2].map(k => c.vPost[k] - c.J[k] / M);
    const Jl = Math.hypot(...c.J) || 1, D = c.J.map(v => v / Jl);
    // parts cut free keep falling, slowed by the energy it took to crush the section that gave way
    let mFree = 0;
    for (const kd of kids) if (!kd.hit) for (const bx of kd.boxes) mFree += shapeVol(boxShape(bx, 0)) * ((MATERIALS[bx[6] & 15] || MATERIALS[0]).density);
    // (cap is the energy crushing that section took: the speed into the contacts drops to sqrt(v^2 - 2E/m))
    const along = vPre[0] * D[0] + vPre[1] * D[1] + vPre[2] * D[2];   // < 0: moving into the contacts
    const speed = Math.max(-along, 0);
    const after = mFree > 0 ? Math.sqrt(Math.max(0, speed * speed - 2 * cap / mFree)) : speed;
    const vFree = [0, 1, 2].map(k => vPre[k] + D[k] * (speed - after));
    if (window.__dbg) console.warn(`break piece ${parent} (body ${idx}, ${M.toFixed(0)} kg): crumbs ${kids.filter(k => k.id < 0).length}, parts ${kids.filter(k => k.id >= 0 && k.boxes.length > 20).map(k => (k.hit ? 'H' : 'F') + k.boxes.length).join(' ')} · vPre ${vPre.map(v => v.toFixed(2))} (${c.vPre ? 'history' : 'impulse'}) vPost ${c.vPost.map(v => v.toFixed(2))} · D ${D.map(v => v.toFixed(2))} · E ${(cap / 1e6).toFixed(2)} MJ mFree ${mFree.toFixed(0)} · speed ${speed.toFixed(2)} -> ${after.toFixed(2)}`);
    // the hit parts move on as the body does now
    const vNow = velOf(idx), w0 = c.w0;
    bodyInfo.delete(idx);
    // the pieces are made where the parent was in the read-back (xf); the GPU carries them along by the parent's
    // motion since then at the step they appear, so each one only needs its velocity beyond the parent's own
    const extra = [];
    const made = addPieces(kids, xf, (cc, b) => {
      const rr = [cc[0] - xf.p[0], cc[1] - xf.p[1], cc[2] - xf.p[2]];
      const rigid = [vNow[0] + w0[1] * rr[2] - w0[2] * rr[1], vNow[1] + w0[2] * rr[0] - w0[0] * rr[2], vNow[2] + w0[0] * rr[1] - w0[1] * rr[0]];
      // parts at the contacts move on with the parent as it is now (stopped by the hit); parts cut free keep
      // the velocity they had before it, of their own
      const dv = [0, 0, 0];
      if (b.id < 0) for (let k = 0; k < 3; k++) dv[k] += (Math.random() - 0.5) * 2;   // crushed crumbs scatter
      if (b.id < 0 && job.blastC) { const dd = [cc[0] - job.blastC[0], cc[1] - job.blastC[1], cc[2] - job.blastC[2]], l = Math.hypot(...dd) || 1; for (let k = 0; k < 3; k++) dv[k] += dd[k] / l * (2 + 2 * Math.random()); }
      if (!b.hit) { const v = [vFree[0] + w0[1] * rr[2] - w0[2] * rr[1], vFree[1] + w0[2] * rr[0] - w0[0] * rr[2], vFree[2] + w0[0] * rr[1] - w0[1] * rr[0]]; extra.push({ v }); return { v, w: w0.slice() }; }
      // a part at the contacts keeps the parent's motion now, minus any rebound away from the contacts (D)
      if (b.id >= 0) { extra.push({ noRebound: D }); return { v: rigid, w: w0.slice() }; }
      extra.push(dv);
      return { v: [rigid[0] + dv[0], rigid[1] + dv[1], rigid[2] + dv[2]], w: w0.slice() };
    }, 0);
    // the velocities were made for the bodies, then for the particles
    gpu.breakBody(idx, made.idxs, xf.p, xf.q, extra.slice(0, made.idxs.length), made.parts, extra.slice(made.idxs.length));
    // the GPU had broken this body by its plan, and the engine broke the whole piece before its answer to that:
    // the kids hold the spare's bricks too, so the spare goes (left, it was a second copy of that part: the lab's
    // mixed scenario, 186 m3 of 1,542 doubled at frame 72)
    { const pend = snapPending.get(parent); if (pend) { snapPending.delete(parent); if (!gpu.genOf || gpu.genOf(pend.spare) === pend.genB) gpu.killBody(pend.spare); } }
    engineBroken.set(idx, clock());
    breaks++;
  }
  if (window.__dbg && nbr) console.warn(`last cut: contacts at ${f[o].toFixed(0)} ticks, plane ${f[o + 1]}, capacity ${f[o + 2].toFixed(0)} MN, ${f[o + 3].toFixed(1)} MJ/m, needed ${f[o + 4].toFixed(0)} MN, squeezed ${f[o + 5]}`);
  lastActive = clock();
}

