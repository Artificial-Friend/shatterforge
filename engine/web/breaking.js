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
function rayDebris(o, d, maxT, skipBody) {
  const r = snapCur; if (!r) return null;
  let best = null;
  const n = r.pose.length / 10;
  for (let k = 0; k < n; k++) {
    const b = k * 10, p = [r.pose[b], r.pose[b + 1], r.pose[b + 2]], h = [r.pose[b + 7], r.pose[b + 8], r.pose[b + 9]];
    if (!(Number.isFinite(p[0]) && Number.isFinite(p[1]) && Number.isFinite(p[2]) && Number.isFinite(r.pose[b + 6]))) continue;
    if (skipBody) { const sh = gpu.shapes[r.slots[k] - 1]; if (!sh || sh.body === skipBody) continue; }
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
    gpu.setVelocity(i, v, [0, 1, 2].map(() => (simRandom() - 0.5) * 2), p, [lastBodies[o + 3], lastBodies[o + 4], lastBodies[o + 5], lastBodies[o + 6]]);
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
      const s = 2 + 2 * simRandom();
      return { v: [d[0] * s + (simRandom() - 0.5), d[1] * s + (simRandom() - 0.5), d[2] * s + (simRandom() - 0.5)], w: [0, 1, 2].map(() => (simRandom() - 0.5) * 6) };
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

// (J/m3 to crush: engine structure::crush_energy; kg/m3 solid. Used by `ahead`.)
const CRUSH_E = [1.0e6, 0.4e6, 20e6, 0.05e6, 0.2e6], SOLID = [2400, 1900, 7850, 2500, 500];
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
function crushBodies(more) {
  if (!gpu.takeCrushed) return;
  for (const c of (more || gpu.takeCrushed())) {
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
    parts.forEach((slot, k) => { if (slot < 0) return; const g = pInfo[slot] ? pInfo[slot].gen + 1 : 1; pInfo[slot] = { h: info[k].h, col: info[k].col, gen: g, dead: false }; if (LAB.on) { pInfo[slot].labF = LAB.f; pInfo[slot].labWhy = 'crushed'; pInfo[slot].labV = keep[k].v.map(x => +x.toFixed(2)); } partsMade++; });
    gpu.breakBody(idx, [], c.p, c.q, [], parts, parts.map(() => [0, 0, 0]));
    bodyInfo.delete(idx); crushedBodies++;
  }
}
function impacts() {
  takeGpuSnaps();
  crushBodies();
  anticipate();
  const evs0 = gpu.takeEvents();
  const now = clock();
  // (a heavy body whose hit was seen coming has paid for it there: `anticipate`)
  const paid = (i) => { const gb = i > 1 ? gpu.bodies[i - 1] : null; return !!(gb && gb.heavy && gb.heavy.quiet > now); };
  // (there: on the standing structure. What it does to furniture and debris in its way is an impact like any other;
  // left out with the rest, a wardrobe it hit at 56 m/s stayed whole, and it came back off it at 19 m/s: lab
  // qballglass, 18 Nov)
  const evs = evs0.filter(e => !(paid(e.a) && e.b <= 1) && !(paid(e.b) && e.a <= 1));
  if (!evs.length) return;
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
// A weight at rest on loose pieces (list: { idx: the piece's body, p: where it bears, f: N, slot: the shape there }):
// a piece with nothing under it at the contact carries the weight as a slab carries a column, in shear round the
// loaded patch, or the weight punches through it.
// - The patch: the weight crushes its way in until the patch carries it, A = f / crushing strength (a 5 MN ball on
//   30 MPa concrete: 0.17 m2, radius a = 0.23 m).
// - The shear: on the surface round the patch through the piece's thickness t, u t with u = pi (2 a + t) (the
//   critical perimeter at t / 2 from the patch), or 2 w t for a member narrower than that; at the material's shear
//   strength (PUNCH_TAU; concrete 0.33 sqrt(f_c) = 1.8 MPa for 30 MPa, the two-way shear of ACI 318).
//   A 0.25 m slab: 1.8 MPa x pi (0.46 + 0.25) m x 0.25 m = 1.0 MN. Under a 5 MN ball it goes; a 1 m block holds.
// - The bending: see PUNCH_M. The same slab folds at 1.1 MN whatever the patch, so the ball's 5 MN goes through
//   a hole it has already widened to where the shear holds.
// - Punched, the hole grows until the shear round it carries the weight: radius rho with pi (2 rho + t) t tau = f
//   (the 0.25 m slab under 5 MN: 1.64 m), no wider than the ball. The piece loses that (the engine's blast on a
//   loose piece, with nothing thrown) and is split by the hole as any piece is. A piece that carries the weight
//   passes it on to the structure under the contact. Once a second, with the rest loads (worker.js restLoads).
// Before (24 Nov): a loose piece carried whatever lay on it; lab qroofball, the 514 t ball at rest on the fragments
// of the floor it had broken, lying across their own hole: 7.13 m down in 15 s.
const PUNCH_TAU = [1.8e6, 0.3e6, 145e6 * 0.16, 0.5e6, 2.0e6];   // Pa: concrete, masonry, steel (a rolled section, 16% of its box), glass, timber
// (N m per m of width, for a thickness of 1 m; times t^2. Reinforced concrete: 0.5% of steel at 500 MPa, its
// lever arm 0.9 of a depth 0.8 t: 1.44 MN; a 0.25 m slab 90 kN m / m, and its fan 1.1 MN. The others: their
// bending strength x t^2 / 6 (masonry 0.3 MPa, glass 30, timber 30); steel plate at yield, x t^2 / 4, 16% of its box)
const PUNCH_M = [1.44e6, 0.3e6 / 6, 250e6 * 0.16 / 4, 30e6 / 6, 30e6 / 6];
let weightPunches = 0;
function bearOnPieces(list) {
  // (one patch a piece: the sample points under a ball are one contact, and their shares one force, at the first)
  const by = new Map(), punched = new Set();
  for (const c of list) { const e = by.get(c.idx); if (e) e.f += c.f; else by.set(c.idx, { ...c }); }
  for (const [idx, c] of by) {
    const inf = bodyInfo.get(idx), sh = gpu.shapes[c.slot - 1]; if (!inf || !sh || sh.body !== idx || snapPending.has(inf.piece)) continue;
    const m = sh.mat & 7, crushPa = ((MATERIALS[m] || MATERIALS[0]).crush || 30) * 1e6, a = Math.sqrt(c.f / crushPa / Math.PI);
    // (its thickness: the shape's under the contact; its width: the piece's own, over all its shapes, in its frame.
    // By the one shape, a slab's fragment of 0.43 m bricks was a member 0.43 m wide, cut 0.3 m at a time: lab
    // qroofball, the ball at rest on the same fragments for 9 s)
    const t = 2 * Math.min(...sh.h); let w = 0;
    { const lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9]; for (const k of gpu.bodyShapes.get(idx) || []) { const s = gpu.shapes[k - 1]; if (!s || s.dead || !s.lp) continue; const e = Math.hypot(...s.h); for (let q = 0; q < 3; q++) { lo[q] = Math.min(lo[q], s.lp[q] - e); hi[q] = Math.max(hi[q], s.lp[q] + e); } }
      w = [0, 1, 2].map(q => hi[q] - lo[q]).sort((x, y) => x - y)[1]; if (!(w > 0)) w = 2 * [...sh.h].sort((x, y) => x - y)[1]; }
    const tau = (PUNCH_TAU[m] ?? PUNCH_TAU[0]) * ((sh.col & 8) ? 0.2 : 1), slab = Math.PI * (2 * a + t) <= 2 * w;
    const cap = tau * t * (slab ? Math.PI * (2 * a + t) : 2 * w);
    if (window.__dbg) console.warn(`weight on body ${idx}: ${(c.f / 1e6).toFixed(2)} MN on a ${t.toFixed(2)} m thick, ${w.toFixed(2)} m wide shape of material ${m}: carries ${(cap / 1e6).toFixed(2)} MN`);
    // and in bending: a weight on a slab folds it along lines out from the contact (the yield-line fan) at
    // 4 pi m, m the moment a metre's width of it carries (PUNCH_M x t^2)
    const bend = 4 * Math.PI * (PUNCH_M[m] ?? PUNCH_M[0]) * t * t * ((sh.col & 8) ? 0.2 : 1);
    if (window.__dbg) console.warn(`  in bending ${(bend / 1e6).toFixed(2)} MN`);
    if (!(c.f > cap) && !(slab && c.f > bend)) continue;
    // (how wide it goes: the hole grows until the shear round it carries the weight, pi (2 rho + t) t tau = f; a
    // member is cut across)
    // (folded, it lets the whole weight through: a hole a twentieth wider than the ball)
    const rho = !slab ? w / 2 + 0.1 : c.f > bend ? (c.rBall ? 1.05 * c.rBall : 1e9) : Math.max(a + t / 2, (c.f / (tau * t * Math.PI) - t) / 2);
    const xf = frameOf(idx), Rt = mt(xf.R), sp = mv(Rt, [c.p[0] - xf.T[0], c.p[1] - 0.05 - xf.T[1], c.p[2] - xf.T[2]]).map((v, k) => v + xf.C[k]);
    const v = velOf(idx), wv = angOf(idx);
    jobs.blast.push({ op: 'pblast', piece: inf.piece, x: sp[0], y: sp[1], z: sp[2], r: Math.min(rho, c.rBall ? 1.05 * c.rBall : rho), ctx: new Map([[idx, { J: [0, 0, 0], vPre: v, vPost: v, w0: wv }]]) });
    punched.add(idx); weightPunches++;
  }
  return punched;
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
    addPieces(st.bodies, null, (c, b) => b.id >= 0 ? motionAt(c) : { v: [(simRandom() - 0.5) * 3, simRandom() * 2, (simRandom() - 0.5) * 3], w: [0, 1, 2].map(() => (simRandom() - 0.5) * 8) });
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
    if (LAB.on) { let all = 0, fine = 0, mas = 0, fineMas = 0; for (const kd of kids) for (const bx of kd.boxes) { const v = bx[3] * bx[4] * bx[5] * TICK ** 3, m = (bx[6] & 7) === 1; all += v; if (m) mas += v; if (kd.id < 0) { fine += v; if (m) fineMas += v; } } if (all >= 50) (LAB.bigBreaks = LAB.bigBreaks || []).push([LAB.f, +all.toFixed(0), +(fine / all).toFixed(3), +(mas / all).toFixed(3), +((fine - fineMas) / Math.max(all - mas, 1e-9)).toFixed(3)]); }   // (frame, m3, share in crumbs, share that is masonry, share in crumbs of what is not masonry)
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
    for (const kd of kids) if (!kd.hit) for (const bx of kd.boxes) mFree += shapeVol(boxShape(bx, 0)) * codeDensity(bx[6]);
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
      if (b.id < 0) for (let k = 0; k < 3; k++) dv[k] += (simRandom() - 0.5) * 2;   // crushed crumbs scatter
      if (b.id < 0 && job.blastC) { const dd = [cc[0] - job.blastC[0], cc[1] - job.blastC[1], cc[2] - job.blastC[2]], l = Math.hypot(...dd) || 1; for (let k = 0; k < 3; k++) dv[k] += dd[k] / l * (2 + 2 * simRandom()); }
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


// ---- a hit seen coming (15 Nov)
// The GPU stops a body rigidly in the step it lands; the engine breaks what it landed on some frames later (the
// contact comes back with a read-back, the job takes its turn). For most debris that is right enough. For a steel
// ball at 60 m/s it is not: it had bounced off the wall it then broke (lab `tallball`, 14 Nov: down onto a
// 0.25 m roof slab at 47 m/s, back up at 12 m/s and 7.3 m high before it came down again through the hole).
// So a heavy fast body (`markHeavy`: the game's ball) looks ahead along its path, PUNCH_AHEAD seconds of flight:
// - what is in its way (`Sim::sweep`), and what it breaks of that and what each thing costs it (`Sim::punch_probe`:
//   a hole along the path through one thing after another, each paid for from its kinetic energy), are asked of
//   the page's own copy of the engine, in the frame, and the things are kept as `stages`;
// - the holes are made by the engine in its turn (`Sim::punch`, the same question), some frames later: before the
//   body arrives if nothing disturbs it;
// - as it reaches each thing its speed drops by what that thing cost, and it goes through without touching the
//   fixed world (`GpuWorld.setShot` 2) from three steps before its front is there until its back has left; or it
//   is stopped in the crater if nothing is left;
// - gone another way (two radii off the line it asked along, or 25 degrees), the things not yet reached are
//   forgotten and asked about again from where it is.
// (Until 15 Nov the answer was waited for, 4 to 8 frames, and the body had to fit the hole made where it was going
// when it asked, 0.09 of its radius wider than its corners reach. Lab qballglass: two chairs put the ball 0.12 m
// off; it met the rim of its own hole and came back at 15 m/s; asked again, the answer came after the next
// wall, 2.75 m on, 3 steps at 54 m/s.)
const PUNCH_MIN = 10;        // m/s: slower than this, a body hits as everything else does
const PUNCH_AHEAD = 0.25;    // s of flight looked ahead
const PUNCH_STRAIGHT = 0.14; // s of flight the engine may take as straight beyond the first thing: it falls 0.1 m in that
const PUNCH_SEAT = 1;        // m/s: a body that spent everything settles into its crater at this speed
let punchStat = { sent: 0, through: 0, stopped: 0, none: 0, off: 0, log: [] };
const punchLog = (ev, o) => { if (punchStat.log.length < 80) punchStat.log.push({ f: gpu.frame, ev, ...o }); };
// body: a GPU body just made (1-based); r: its radius (m); p, v: where it is and how it moves now
function markHeavy(body, r, p, v) {
  const gb = gpu.bodies[body - 1]; if (!gb) return;
  // stages: the things ahead that are paid for, nearest first; plan: where it comes out of the last of them, and
  // with what; pass: the thing it is going through; seat: the crater it is stopping in; set: the velocity last
  // given to it and the step it was given at (the read-backs of the next steps do not show it yet)
  gb.heavy = { r, f0: gpu.frame, quiet: 0, again: 0, stages: [], plan: null, set: null, pass: null, seat: null };
  if (p && v) tryPunch(body, gb, p, v);
}
// what the body's velocity is: the read-back's, or what it was last set to while the read-backs are older than that
function heavyVel(i, h) {
  if (h.set && lastReadFrame <= h.set.frame) return h.set.v.slice();
  const vo = (i - 1) * 3; return [lastVel[vo], lastVel[vo + 1], lastVel[vo + 2]];
}
function tryPunch(i, gb, p, v) {
  const h = gb.heavy;
  if (!E.punch || !E.punch_probe || !E.sweep || !+setting.punch) return;
  // (one thing after another, each asked from where it comes out of the last, as far as it looks)
  for (let round = 0; round < 6; round++) if (!askPunch(i, gb, p, v)) break;
}
function askPunch(i, gb, p, v) {
  const h = gb.heavy;
  const m = gb.mass, G = 9.81, spNow = Math.hypot(v[0], v[1], v[2]);
  if (!(spNow >= PUNCH_MIN)) return false;
  // From where, and how fast: where it is, as it moves; or, with things ahead already paid for, from where it comes
  // out of the last of them, at the speed it will have left
  let ps = p, vs = v;
  if (h.plan) { const sl = Math.sqrt(2 * Math.max(0, h.plan.e) / m); ps = h.plan.exit; vs = h.plan.d.map(x => x * sl); }
  const sp = Math.hypot(vs[0], vs[1], vs[2]);
  if (!(sp >= PUNCH_MIN)) return false;
  // (no further ahead than it flies in PUNCH_AHEAD from where it is now)
  const reach = spNow * PUNCH_AHEAD + h.r - Math.hypot(ps[0] - p[0], ps[1] - p[1], ps[2] - p[2]);
  if (!(reach > 0.5)) return false;
  const d = vs.map(x => x / sp);
  // the first thing in its way, or the floor it comes down on (Sim::sweep: what its core meets, or its side
  // catches; a floor it rolls on is not). (Five rays, the centre line and four at 0.4 of the radius, missed a door's
  // jamb that the ball caught with its side: lab qballglass, 15 Nov.)
  const look = (o, dd, far) => E.sweep(o[0], o[1], o[2], dd[0], dd[1], dd[2], h.r, far);
  const t = look(ps, d, reach);
  if (t < 0) return false;
  // Where and how it arrives: it falls on the way (from 60 m/s at 12 m, 0.19 m: a hole made on the straight line
  // was 0.19 m too high, and the ball met its lower rim: lab qballtemple, 15 Nov). The line is looked along once
  // more towards where the fall takes it, and the hole is made there, along the way it is then going.
  const fall = (tt) => { const tau = tt / sp; return [ps[0] + vs[0] * tau, ps[1] + vs[1] * tau - 0.5 * G * tau * tau, ps[2] + vs[2] * tau]; };
  let da = d, ta = t, hit = fall(t);
  {
    const q = hit, cl = Math.hypot(q[0] - ps[0], q[1] - ps[1], q[2] - ps[2]) || 1, cd = [0, 1, 2].map(k => (q[k] - ps[k]) / cl);
    const t2 = look(ps, cd, reach + 1);
    if (t2 >= 0) { ta = t2; hit = fall(t2); }
    const tau = ta / sp, va = [vs[0], vs[1] - G * tau, vs[2]], vl = Math.hypot(va[0], va[1], va[2]);
    da = va.map(x => x / vl);
  }
  // (gain: what its weight adds along the way, J/m; e: what it will have where it meets the first thing)
  const gain = -m * G * da[1];
  const e = Math.max(0, 0.5 * m * sp * sp + m * G * (ps[1] - hit[1]));
  // (p0: where it would have come from, straight along the way it arrives, so that its distance along that way to
  // the first thing is ta)
  const p0 = [0, 1, 2].map(k => hit[k] - da[k] * ta);
  const job = { op: 'punch', x: hit[0], y: hit[1], z: hit[2], dx: da[0], dy: da[1], dz: da[2], r: h.r, energy: e, mass: m, reach: Math.max(0.5, Math.min(reach - ta, sp * PUNCH_STRAIGHT)), gain,
    ctx: { d: da, t0: performance.now(), body: i, gb, p0, t: ta, e, sp, gain, at: clock(), done: false } };
  punchStat.sent++; punchLog('sent', { t: +ta.toFixed(2), sp: +sp.toFixed(1), p: p.map(x => +x.toFixed(2)), hit: hit.map(x => +x.toFixed(2)), planned: !!h.plan });
  // asked now, of the page's copy; done by the engine in its turn
  E.punch_probe(job.x, job.y, job.z, job.dx, job.dy, job.dz, job.r, job.energy, job.mass, job.reach, job.gain);
  const got = planStages(job.ctx, Array.from(new Float32Array(E.memory.buffer, E.punch_ptr(), E.punch_len())));
  if (got) { jobs.blast.push(job); pump(); }
  return got;
}
// The engine's answer (Sim::punch_out) as the things ahead of the body. Returns whether there is anything.
function planStages(c, po) {
  const gb = c.gb, h = gb.heavy, n = po[0] | 0;
  let e = c.e, got = 0, from = 0, out = 0;
  for (let k = 0; k < n; k++) {
    const [begins, spent, ends, through] = [po[1 + 4 * k], po[2 + 4 * k], po[3 + 4 * k], po[4 + 4 * k]];
    e = Math.max(0, e + c.gain * (begins - from)); from = begins;
    if (!(spent > 0)) continue;
    const before = Math.sqrt(2 * e / gb.mass);
    e = Math.max(0, e - spent); got++; out = ends;
    // (s: how far along the way from where it was when it asked its front meets this thing; due: when, at the latest)
    h.stages.push({ d: c.d, p0: c.p0, s: c.t + begins, depth: ends - begins, spent, before, ctx: c, stop: !(through > 0) || Math.sqrt(2 * e / gb.mass) < PUNCH_SEAT, due: c.at + ((c.t + begins) / Math.max(c.sp, 1)) * 1000 + 300 });
    punchLog('answer', { k, begins: +(c.t + begins).toFixed(2), spent_kJ: Math.round(spent / 1e3), left_kJ: Math.round(e / 1e3), through });
  }
  // (nothing it could break: it hits as any body does, and looks again in a quarter of a second)
  // (with things ahead already, it asks on from their far side every step: as far as it then looks there may be
  // nothing yet. Left for a quarter of a second, the next partition of the glass house, 2.75 m on, was asked about
  // when the ball was at it, and held it for a step: lab qballglass, 18 Nov)
  if (!got) { punchStat.none++; if (!h.plan) { punchLog('nothing', {}); h.again = clock() + 250; } if (!h.stages.length) { h.quiet = 0; h.plan = null; } return false; }
  // (exit: where it comes out of the last of them, its own radius past what was crushed)
  h.plan = { d: c.d, e, exit: [0, 1, 2].map(k => c.p0[k] + c.d[k] * (c.t + out + h.r)) };
  h.quiet = Infinity;
  return true;
}
// The engine has made the hole (its answer was taken when it was asked: askPunch)
function onPunch(job, r) {
  if (r.n > 0) onBlast(job, r);
  job.ctx.done = true;
}
// The bodies a thrown one (body i, at p, moving at v) reaches in the next three steps and can crush: lighter than
// it, and crushing all of them (CRUSH_E) taking less than the two would lose if they met and went on together,
// 1/2 x mM/(m+M) x v2. They become rubble where they are, now; returns the energy that took (J).
// (Smashed after the contact, from its events, the wardrobe had already stopped the ball against the wall behind
// it for two steps: back at 19 m/s. Lab qballglass, 18 Nov.)
function ahead(i, gb, p, v, dt) {
  const h = gb.heavy, sp = Math.hypot(v[0], v[1], v[2]); if (!(sp >= PUNCH_MIN)) return 0;
  const d = v.map(x => x / sp), reach = sp * dt * 3 + h.r, list = []; let spent = 0;
  for (let o = 2; o <= gpu.n; o++) {
    const go = gpu.bodies[o - 1], k = (o - 1) * 7;
    if (o === i || !go || go.dead || go.heavy || !(go.mass > 0) || go.mass > gb.mass || k + 6 >= lastBodies.length) continue;
    const c = [lastBodies[k], lastBodies[k + 1], lastBodies[k + 2]];
    if (Math.hypot(c[0] - p[0], c[1] - p[1], c[2] - p[2]) > reach + 6) continue;
    const q = [lastBodies[k + 3], lastBodies[k + 4], lastBodies[k + 5], lastBodies[k + 6]], qi = [-q[0], -q[1], -q[2], q[3]];
    let hit = false, m = 0;
    for (const slot of gpu.bodyShapes.get(o) || []) {
      const sh = gpu.shapes[slot - 1]; if (!sh || sh.dead) continue;
      m = sh.mat;
      // the path against the shape's box grown by the body's radius, in the shape's frame
      const qs = sh.lq ? [-sh.lq[0], -sh.lq[1], -sh.lq[2], sh.lq[3]] : [0, 0, 0, 1];
      const lo = qrot(qs, qrot(qi, [p[0] - c[0], p[1] - c[1], p[2] - c[2]]).map((x, a) => x - sh.lp[a])), ld = qrot(qs, qrot(qi, d));
      let t0 = 0, t1 = reach;
      for (let a = 0; a < 3 && t0 <= t1; a++) {
        const e = sh.h[a] + h.r;
        if (Math.abs(ld[a]) < 1e-9) { if (Math.abs(lo[a]) > e) t0 = 1e9; continue; }
        const ta = (-e - lo[a]) / ld[a], tb = (e - lo[a]) / ld[a];
        t0 = Math.max(t0, Math.min(ta, tb)); t1 = Math.min(t1, Math.max(ta, tb));
      }
      if (t0 <= t1) { hit = true; break; }
    }
    if (!hit) continue;
    const need = (CRUSH_E[m] ?? 1e6) * go.mass / (SOLID[m] ?? 2400), can = 0.5 * go.mass * gb.mass / (go.mass + gb.mass) * sp * sp;
    if (need > can || need > 0.5 * gb.mass * sp * sp - spent) continue;
    spent += need;
    list.push({ body: o, p: c, q });
    if (LAB.on) (LAB.smashed = LAB.smashed || []).push({ f: LAB.f, body: o, kg: Math.round(go.mass), need_kJ: Math.round(need / 1e3), ball_m_s: +sp.toFixed(1) });
  }
  if (list.length) crushBodies(list);
  return spent;
}
function passOff(i, h) { if (h.pass) { h.pass = null; if (gpu.setShot) gpu.setShot(i, 1); } }
function anticipate() {
  if (!gpu || !lastBodies || !lastVel) return;
  const dt = lastStepDt || 1 / 60, now = clock();
  for (let i = 2; i <= gpu.n; i++) {
    const gb = gpu.bodies[i - 1], h = gb && gb.heavy;
    if (!h || gb.dead || lastReadFrame <= h.f0 || (i - 1) * 7 + 6 >= lastBodies.length) continue;
    const o = (i - 1) * 7;
    const v = heavyVel(i, h), lag = Math.max(0, gpu.frame - lastReadFrame) * dt;
    // (where it is now: the read-back is a step or two old)
    const p = [lastBodies[o] + v[0] * lag, lastBodies[o + 1] + v[1] * lag, lastBodies[o + 2] + v[2] * lag];
    const q = [lastBodies[o + 3], lastBodies[o + 4], lastBodies[o + 5], lastBodies[o + 6]];
    if (h.seat) {
      // The step into the crater is done. It rests there once the crater is there: until the engine has made it
      // (its turn may come after the body does), the body is held where it is, still passing.
      if (!h.seat.ctx.done && now < h.seat.until) { gpu.setVelocity(i, [0, 0, 0], [0, 0, 0], p, q); h.set = { v: [0, 0, 0], frame: gpu.frame }; continue; }
      passOff(i, h);
      const sv = h.seat.d.map(x => x * PUNCH_SEAT);
      gpu.setVelocity(i, sv, [0, 0, 0], p, q);
      h.set = { v: sv, frame: gpu.frame }; h.seat = null; h.quiet = now + 150; h.again = now + 250; continue;
    }
    // through the thing it was passing: it touches the fixed world again
    if (h.pass) {
      const g = h.pass, s = (p[0] - g.p0[0]) * g.d[0] + (p[1] - g.p0[1]) * g.d[1] + (p[2] - g.p0[2]) * g.d[2];
      if (s >= g.end || now >= g.until) passOff(i, h);
    }
    // Knocked off the line it asked along, before it reached the next thing. A little (chairs, crumbs: 0.1 m) changes
    // nothing: the thing is paid for and its hole asked for, and the body goes through it without touching it
    // wherever its hole is (h.pass). (Forgotten and asked again, the thing was not seen where its hole already
    // was, and the body met the rim: lab qballglass, 18 Nov, 58 to 2 m/s on the core's west panel.) Gone another
    // way, two radii off or 25 degrees, the things ahead are forgotten, with the holes not yet asked of the
    // engine, and it looks again below, from where it is.
    if (h.stages.length) {
      const st = h.stages[0], e = [p[0] - st.p0[0], p[1] - st.p0[1], p[2] - st.p0[2]];
      const s = e[0] * st.d[0] + e[1] * st.d[1] + e[2] * st.d[2], off = Math.hypot(e[0] - s * st.d[0], e[1] - s * st.d[1], e[2] - s * st.d[2]);
      const sp = Math.hypot(v[0], v[1], v[2]), along = v[0] * st.d[0] + v[1] * st.d[1] + v[2] * st.d[2];
      // (the path was taken as straight from the thing back to here: it is above that line by what it has yet to fall)
      const tau = Math.max(0, (st.s - s) / Math.max(along, 1)), sagging = 0.5 * 9.81 * tau * tau;
      if (s < st.s - h.r - sp * dt && sp >= PUNCH_MIN && (off > 2 * h.r + sagging || along < 0.9 * sp - 9.81 * tau)) {
        punchStat.off++; punchLog('off the line', { s: +s.toFixed(2), at: +st.s.toFixed(2), off: +off.toFixed(2), sp: +sp.toFixed(1), along: +along.toFixed(1) });
        h.stages.length = 0; h.plan = null; h.again = 0; passOff(i, h);
        jobs.blast = jobs.blast.filter(j => !(j.op === 'punch' && j.ctx.body === i && j.ctx.gb === gb));
      }
    }
    // the things it reaches in the next step: its front is at each within that step
    let cur = v.slice(), changed = false;
    // furniture and debris in its way in the next three steps, that it has the energy to crush: rubble before it
    // is there, paid for as the standing things are (ahead)
    {
      const spent = ahead(i, gb, p, v, dt);
      if (spent > 0) { const sp = Math.hypot(v[0], v[1], v[2]), left = Math.sqrt(Math.max(0, sp * sp - 2 * spent / gb.mass)); cur = v.map(x => x * left / sp); changed = true; }
    }
    while (h.stages.length) {
      const st = h.stages[0];
      const s = (p[0] - st.p0[0]) * st.d[0] + (p[1] - st.p0[1]) * st.d[1] + (p[2] - st.p0[2]) * st.d[2];
      const along = cur[0] * st.d[0] + cur[1] * st.d[1] + cur[2] * st.d[2], sp = Math.hypot(cur[0], cur[1], cur[2]);
      // (near: within three steps of it. From then it does not touch the fixed world, until its back has left the thing)
      // (for 0.3 s from the first time, and no longer: held up before the thing, by a wardrobe pushed ahead of it,
      // it stayed so for a second and fell through the floor it was on: lab qballglass, 18 Nov)
      if (s + Math.max(along, 0.3 * st.before) * dt * 3 >= st.s - h.r && now < (st.passAt = st.passAt || now) + 300) {
        const end = st.s + st.depth + 2 * h.r, same = h.pass && h.pass.p0 === st.p0;
        if (!same || end > h.pass.end) {
          // (s0, s1: the thing's own depth along the way, and 0.3 m each side; with the one before it on the same way)
          // (still in the thing before, asked about along another line: its near side, as far along this one)
          let s0 = st.s - 0.3;
          if (h.pass) { const g = h.pass, q = [0, 1, 2].map(k => g.p0[k] + g.d[k] * g.s0 - st.p0[k]); s0 = Math.min(s0, same ? g.s0 : q[0] * st.d[0] + q[1] * st.d[1] + q[2] * st.d[2]); }
          h.pass = { p0: st.p0, d: st.d, end, until: st.passAt + 300, s0, s1: st.s + st.depth + 0.3 };
          if (gpu.setShot) gpu.setShot(i, 2, h.pass);
        }
      }
      if (!(s + Math.max(along, 0) * dt >= st.s - h.r)) {
        // (never reached, long overdue: it went elsewhere or stopped short; the hole stays, the plan goes)
        if (now >= st.due + 1000) { h.stages.length = 0; h.plan = null; passOff(i, h); punchLog('dropped', { s: +s.toFixed(2), at: +st.s.toFixed(2) }); }
        break;
      }
      h.stages.shift(); changed = true;
      // its speed after this thing: what it has now (with what it gained falling since), less what the thing cost
      const base = sp, left = Math.sqrt(Math.max(0, base * base - 2 * st.spent / gb.mass));
      punchLog('arrives', { s: +s.toFixed(2), at: +st.s.toFixed(2), along: +along.toFixed(1), to: +left.toFixed(1), stop: st.stop });
      if (st.stop || left < PUNCH_SEAT) {
        // stopped in what it crushed: one step takes it to the bottom of the crater, then it is seated
        punchStat.stopped++;
        const go = Math.max(0, st.s - h.r + st.depth - s), step = Math.max(PUNCH_SEAT, Math.min(base, go / dt));
        cur = st.d.map(x => x * step); h.seat = { d: st.d, ctx: st.ctx, until: now + 1000 }; h.stages.length = 0; h.plan = null;
        break;
      }
      punchStat.through++;
      const dir = !(sp > 1e-6) ? st.d : cur.map(x => x / sp);
      cur = dir.map(x => x * left);
    }
    if (changed) {
      // (without spin: turning, its corners sweep the hole's rim)
      gpu.setVelocity(i, cur, [0, 0, 0], p, q);
      h.set = { v: cur, frame: gpu.frame };
      if (!h.stages.length && !h.seat) { h.plan = null; h.quiet = now + 50; }
      lastActive = now;
    }
    if (!h.stages.length && !h.seat && h.quiet === Infinity) h.quiet = now + 50;
    // (not from inside what it is passing, with nothing planned beyond: its hole may not be there yet, and would be
    // asked about again)
    if (!h.seat && now >= h.again && !(h.pass && !h.plan)) tryPunch(i, gb, p, cur);
  }
}
