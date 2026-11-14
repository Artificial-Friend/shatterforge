// ---- the static tower, falling pieces and their breaking
// A piece's bricks stay in the engine in the tower's own frame; the GPU body moves them. For each body we keep
// the map from the tower frame to the world at the body's creation (C, qs, P) and the body's pose then (p0, q0):
//   world(x) = p(t) + R(q(t)) R(q0)^T (P + R(qs) (x - C) - p0)
const bodyInfo = new Map();   // GPU body (1-based) -> { piece, C, qs, P, p0, q0, cool }
let lastBodies = null, lastVel = null, lastAng = null, velHist = [], lastReadFrame = -1;
// A body's state from the latest read-back, when that read-back saw it: a body made since (a piece just cut
// free, or one in a reused slot) is not in it, and reading it there gave NaN or another body's state
function seen(idx) { const inf = bodyInfo.get(idx); return lastVel && (idx - 1) * 3 + 2 < lastVel.length && (!inf || lastReadFrame > inf.f0); }
function velOf(idx) { if (seen(idx)) { const o = (idx - 1) * 3; return [lastVel[o], lastVel[o + 1], lastVel[o + 2]]; } const inf = bodyInfo.get(idx); return inf && inf.v0 ? inf.v0.slice() : [0, 0, 0]; }
function angOf(idx) { if (seen(idx) && lastAng) { const o = (idx - 1) * 3; return [lastAng[o], lastAng[o + 1], lastAng[o + 2]]; } return [0, 0, 0]; }
const mm = (A, B) => A.map((r, i) => [0, 1, 2].map(j => r[0] * B[0][j] + r[1] * B[1][j] + r[2] * B[2][j]));
const mt = (A) => [0, 1, 2].map(i => [A[0][i], A[1][i], A[2][i]]);
function bodyPose(idx) {
  const b = gpu.bodies[idx - 1];
  if (lastBodies && (idx - 1) * 7 + 7 <= lastBodies.length && seen(idx)) { const o = (idx - 1) * 7; return { p: [lastBodies[o], lastBodies[o + 1], lastBodies[o + 2]], q: [lastBodies[o + 3], lastBodies[o + 4], lastBodies[o + 5], lastBodies[o + 6]] }; }
  return { p: b.p, q: b.q };
}
function frameOf(idx) {
  const inf = bodyInfo.get(idx), { p, q } = bodyPose(idx);
  const Rt = mm(q2m(q), mt(q2m(inf.q0)));
  const R = mm(Rt, q2m(inf.qs));
  const d = [inf.P[0] - inf.p0[0], inf.P[1] - inf.p0[1], inf.P[2] - inf.p0[2]];
  const rd = mv(Rt, d);
  return { R, T: [p[0] + rd[0], p[1] + rd[1], p[2] + rd[2]], C: inf.C, p, q };
}
function readBodies(f, o) {
  const nb = f[o++], bodies = [];
  for (let k = 0; k < nb; k++) {
    const id = f[o++], n = f[o++], boxes = [];
    for (let q = 0; q < n; q++, o += 7) boxes.push(f.subarray(o, o + 7));
    bodies.push({ id, boxes });
  }
  return { bodies, o };
}
function boxShape(f, o) {
  const [x, y, z, sx, sy, sz, code] = [f[o], f[o + 1], f[o + 2], f[o + 3], f[o + 4], f[o + 5], f[o + 6]];
  // the engine sends material + 16 x paint (the material drives physics, both the colour) + 512 x hull: a convex
  // cell of the hull library (the engine's and the GPU's are the same, engine/gpu/test_hulls.cjs), mapped into the box
  const p = [(x + sx / 2) * TICK, (y + sy / 2) * TICK, (z + sz / 2) * TICK], h = [sx * TICK / 2, sy * TICK / 2, sz * TICK / 2];
  const hull = Math.floor(code / 512), s = { p, q: [0, 0, 0, 1], h, mat: code & 15, col: code & 255, hull: 0 };
  // a free piece (docs/PHASE2.md): 256 + its index in the engine; its shape is its own hull, turned
  if (hull >= 256) { const ps = pieceShapeOf(hull - 256, code); if (ps) return ps; piecesAsBoxes++; return s; }
  if (hull && GpuWorld.hulls && hull < GpuWorld.hulls.lib) {
    const bx = GpuWorld.hulls.hulls[hull].box;
    s.p = p.map((c, k) => c + bx.c[k] * h[k]); s.h = h.map((x, k) => x * bx.h[k]); s.hull = hull;
  }
  return s;
}
// Free piece k of the engine as a GPU shape (GpuWorld.pieceShape: its hull registered by its corners), or null
// beyond the GPU's hull limits or with the runtime hull slots full (then it is its box: counted in piecesAsBoxes)
let piecesAsBoxes = 0;
function pieceShapeOf(k, code) {
  if (!E || !E.piece_shape || !GpuWorld.pieceShape) return null;
  const n = E.piece_shape(k); if (!n) return null;
  const f = new Float32Array(E.memory.buffer, E.piece_ptr(), n).slice();
  const centre = [f[0], f[1], f[2]], q = [f[3], f[4], f[5], f[6]], nv = f[7];
  const corners = []; let o = 8;
  for (let i = 0; i < nv; i++, o += 3) corners.push([f[o], f[o + 1], f[o + 2]]);
  const nf = f[o++], loops = [];
  for (let i = 0; i < nf; i++) { const c = f[o++]; loops.push(Array.from(f.subarray(o, o + c))); o += c; }
  return GpuWorld.pieceShape(corners, loops, centre, q, code & 15, code & 255);
}
// a shape's volume (a cell: its share of its own box)
const shapeVol = (s) => 8 * s.h[0] * s.h[1] * s.h[2] * (s.hull ? GpuWorld.hulls.hulls[s.hull].vol : 1);
function applyStatic(updates) {
  let woke = false; const cells = new Map();
  for (const u of updates) {
    const [idx, alive] = [u[0], u[1]];
    const s = boxShape(u, 2);
    // (a brick split into convex cells: each cell is a brick of its own, a hull shape)
    if (alive) slots.set(idx, gpu.setStaticShape(slots.get(idx) ?? null, 1, s.p, s.h, s.mat, s.hull, s.q));
    else if (slots.has(idx)) {
      gpu.killShape(slots.get(idx)); slots.delete(idx);
      // what rested on it must fall: woken round the brick (30 cm), not the whole world (which never came to rest
      // while the building lost bricks: benchmark, NVIDIA Ampere, mixed, 153 bodies awake at 42 s)
      // (bricks gathered by 4 m cell, one box each: a break removes hundreds at once, and over 64 boxes in a step
      // wake everything)
      const r = Math.hypot(s.h[0], s.h[1], s.h[2]) + 0.3, key = s.p.map(c => Math.floor(c / 4)).join(',');
      const b = cells.get(key) || [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
      for (let k = 0; k < 3; k++) { b[k] = Math.min(b[k], s.p[k] - r); b[k + 3] = Math.max(b[k + 3], s.p[k] + r); }
      cells.set(key, b); woke = true;
    }
  }
  if (woke) { if (gpu.wakeNear) for (const b of cells.values()) gpu.wakeNear(b.slice(0, 3), b.slice(3)); else gpu.wakeAll(); }
}
// Add bodies whose boxes are in the tower frame. xf: tower frame -> world { R, T, C } (identity when absent);
// vel(c): the velocity a body starts with at world point c, and its spin.
// Crumbs (loose single bricks, small enough) become particles when that is on: granular debris that piles and
// slides at a fraction of a rigid body's cost (they do not push bodies). Returns the GPU bodies for the rest
// (idxs, aligned with the returned `list`) and the particle slots (parts).
const PART_R = 0.25;   // largest equal-volume radius (m) made a particle
const pInfo = [];      // particle slot -> { h, col, gen, dead }
let partsMade = 0, partsCrushed = 0;
// Clean-up distance: rubble particles and pieces carried farther than this from where they were first seen (where
// they broke off, a read-back or two later) are removed: they no longer take part in the collapse, and cost steps
// and drawing (a blast or a fall can throw crumbs a long way).
// ---- The debris budget (setting "Debris kept at once", the physics preset): falling pieces and rubble particles
// together are at most `debrisMax`; past it the ones that have been there longest go, oldest first. A piece's age
// runs from the read-back that first shows it. Not counted: the building, parked spares, balls, and what stood
// loose in the building from the start (furniture) until something has moved it half a metre.
let debrisMax = Infinity, debrisSeq = 0, debrisCapped = 0, debrisLive = 0;
const debrisBorn = new Map();   // body slot -> { gen, seq, p: where it was first seen, orig: there before anything was done }
// What goes first when there is more debris than the budget: the smallest, and among those the oldest. Three
// classes: crumbs (particles), pieces lighter than `DEBRIS_SMALL` kg, the rest. Strictly oldest first (tried, lab
// `debriscap`, 14 Nov) took 89.7% of the collapsing building's volume out of the world: the upper storeys fall as a
// few big bodies, made before the 13,000 crumbs of the cut and of their landing, and went first.
const DEBRIS_SMALL = 300;   // kg: a 0.5 m cube of concrete
function capDebris(r, pr) {
  const live = [];
  if (r.body) {
    const nb = Math.min(gpu.bodies.length, r.body.length / 7), fresh = lastBlast === '' && balls === 0;
    for (let b = 1; b <= nb; b++) {
      const gb = gpu.bodies[b - 1]; if (!gb || gb.dead || gb.im === 0 || gb.ball || gb.labBall) { debrisBorn.delete(b); continue; }
      const o = (b - 1) * 7, p = [r.body[o], r.body[o + 1], r.body[o + 2]], gen = gpu.genOf(b);
      if (p[1] < -500) { debrisBorn.delete(b); continue; }
      let f = debrisBorn.get(b);
      if (!f || f.gen !== gen) { f = { gen, seq: ++debrisSeq, p, orig: fresh }; debrisBorn.set(b, f); }
      if (f.orig) { if (Math.hypot(p[0] - f.p[0], p[1] - f.p[1], p[2] - f.p[2]) < 0.5) continue; f.orig = false; f.seq = ++debrisSeq; }
      live.push([f.seq, b, o, 1 / gb.im < DEBRIS_SMALL ? 1 : 2]);
    }
  }
  if (pr && pr.list) for (const i of pr.list) { const inf = pInfo[i]; if (!inf || inf.dead) continue; if (inf.born === undefined) inf.born = ++debrisSeq; live.push([inf.born, -1 - i, 0, 0]); }
  debrisLive = live.length;
  let over = live.length - debrisMax;
  if (!(over > 0)) { if (LAB.on && debrisMax < Infinity) LAB.debrisPeak = Math.max(LAB.debrisPeak || 0, live.length); return; }
  live.sort((a, b) => a[3] - b[3] || a[0] - b[0]);
  const newest = [-1, -1, -1], taken = new Set();
  window.__farKill = 'budget';   // (the lab's books: taken out on purpose, by the budget)
  try {
    for (let k = 0; k < live.length && over > 0; k++) {
      const [, id, o, cls] = live[k];
      if (id < 0) { const i = -1 - id; pInfo[i].dead = true; gpu.killParticle(i); }
      else {
        // (not a part the engine is still splitting, nor one with a split on the GPU waiting for it)
        if (engineBroken.has(id) || snapPending.has(id)) continue;
        if (LAB.on) LAB.debrisHeaviest = Math.max(LAB.debrisHeaviest || 0, 1 / gpu.bodies[id - 1].im);
        gpu.breakBody(id, [], [r.body[o], r.body[o + 1], r.body[o + 2]], [r.body[o + 3], r.body[o + 4], r.body[o + 5], r.body[o + 6]], [], [], []);
        bodyInfo.delete(id); debrisBorn.delete(id); farO.delete(id);
      }
      over--; debrisCapped++; newest[cls] = live[k][0]; taken.add(k);
    }
  } finally { window.__farKill = false; }
  // (for the lab: what is left after the cut, and whether anything is left that should have gone before something
  // that went: smaller, or as small and older; other than parts the engine is splitting)
  if (LAB.on) {
    const L = LAB; L.debrisPeak = Math.max(L.debrisPeak || 0, live.length - taken.size);
    const top = newest[2] >= 0 ? 2 : newest[1] >= 0 ? 1 : 0;
    for (let k = 0; k < live.length; k++) {
      if (taken.has(k) || (live[k][1] > 0 && (engineBroken.has(live[k][1]) || snapPending.has(live[k][1])))) continue;
      if (live[k][3] < top || (live[k][3] === top && live[k][0] < newest[top])) L.debrisOrder = false;
    }
  }
}
let farDist = 100, partsFar = 0, bodiesFar = 0;
const farO = new Map();   // body slot -> { gen, p: where it was first seen }
function cleanFar(r) {
  if (LAB.on && LAB.spec && LAB.spec.far != null) farDist = LAB.spec.far;   // (a lab scenario's own distance, over the setting)
  if (!(farDist < Infinity) || !r.body) return;
  const nb = Math.min(gpu.bodies.length, r.body.length / 7);
  for (let b = 1; b <= nb; b++) {
    const gb = gpu.bodies[b - 1]; if (!gb || gb.dead) { farO.delete(b); continue; }
    const o = (b - 1) * 7, p = [r.body[o], r.body[o + 1], r.body[o + 2]], gen = gpu.genOf(b);
    // (a spare body waits parked 1 km under the ground until a break on the GPU hands it a part: it is first
    // seen where that part is. Seen parked, it was 1 km from there, and the mixed building's parts went: 1,018 m3)
    if (p[1] < -500) { farO.delete(b); continue; }
    const f = farO.get(b);
    if (!f || f.gen !== gen) { farO.set(b, { gen, p }); continue; }
    if (Math.hypot(p[0] - f.p[0], p[1] - f.p[1], p[2] - f.p[2]) <= farDist) continue;
    // (not a part the engine is still splitting, nor one with a split on the GPU waiting for it)
    if (engineBroken.has(b) || snapPending.has(b)) continue;
    window.__farKill = true;
    try { gpu.breakBody(b, [], p, [r.body[o + 3], r.body[o + 4], r.body[o + 5], r.body[o + 6]], [], [], []); } finally { window.__farKill = false; }
    bodyInfo.delete(b); farO.delete(b); bodiesFar++;
  }
}
function asParticle(b) {
  if (!setting.particles || !gpu.capP || b.id >= 0 || b.boxes.length !== 1) return false;
  const s = boxShape(b.boxes[0], 0);
  // (a cell spalled off a fracture surface always: as chunk bodies, slivers of them rebounded and thousands ran the
  // GPU world out of shapes)
  if (s.hull) return true;
  return Math.cbrt(6 * shapeVol(s) / 8 / Math.PI) <= PART_R && Math.max(...s.h) <= 0.35;
}
function addPieces(list, xf, vel, cool) {
  if (!gpu || !list.length) return { idxs: [], list: [], parts: [], kinds: [] };
  const pl = [], bl = [];
  let proom = gpu.roomParticles ? gpu.roomParticles() : 0;
  for (const b of list) {
    // spill (id -2): crushed material beyond the engine's budgets, as rubble particles of at most 0.4 m whatever
    // the setting (it was removed as dust: whole floors vanished, 47% of the building in the lab collapse). Only
    // when the particles are full is it lost (counted).
    if (b.id === -2) {
      const f = b.boxes[0], code = f[6];
      if (code >= 512 || !gpu.capP) { if (proom > 0 && gpu.capP) { proom--; pl.push(b); } else spillLost += f[3] * f[4] * f[5]; continue; }
      const n = [3, 4, 5].map(k => Math.max(1, Math.ceil(f[k] * TICK / 0.4)));
      for (let i = 0; i < n[0]; i++) for (let j = 0; j < n[1]; j++) for (let k = 0; k < n[2]; k++) {
        const sx = f[3] / n[0], sy = f[4] / n[1], sz = f[5] / n[2];
        if (proom <= 0) { spillLost += sx * sy * sz; continue; }
        proom--; pl.push({ id: -2, hit: b.hit, boxes: [Float32Array.of(f[0] + i * sx, f[1] + j * sy, f[2] + k * sz, sx, sy, sz, code)] });
      }
      continue;
    }
    if (proom > 0 && asParticle(b)) { proom--; pl.push(b); } else bl.push(b);
  }
  list = bl;
  // capacity: bodies and shapes (slots of retired bodies are not reused)
  let room = gpu.roomBodies(), sroom = gpu.roomShapes(), k = 0;
  const nShapes = (b) => b.chunk ? 1 : b.boxes.length;
  while (k < list.length && room > 0 && sroom >= nShapes(list[k])) { sroom -= nShapes(list[k]); room--; k++; }
  if (k < list.length) { dropped += list.length - k; list = list.slice(0, k); }
  const R = xf ? xf.R : [[1, 0, 0], [0, 1, 0], [0, 0, 1]], qs = xf ? m2q(R) : [0, 0, 0, 1], C = xf ? xf.C : [0, 0, 0], T = xf ? xf.T : [0, 0, 0];
  // Loose bricks too big to be particles break into the chunks of a random fracture pattern (convex cells that
  // tile the brick), each its own body flying with the brick's velocity. Not for pieces cut from a moving one
  // (xf): those are placed on the GPU one for one.
  if (!xf && setting.chunks && GpuWorld.hulls) {
    const out = [];
    for (const b of list) {
      if (b.chunk || b.id >= 0 || b.boxes.length !== 1) { out.push(b); continue; }
      const s0 = boxShape(b.boxes[0], 0);
      if (Math.max(...s0.h) < 0.15 || s0.hull) { out.push(b); continue; }
      const pat = Math.floor(Math.random() * GpuWorld.hulls.patterns.length);
      for (const sh of GpuWorld.hullShapes(s0.p, [0, 0, 0, 1], s0.h, pat, s0.mat, s0.col)) out.push({ id: -1, boxes: b.boxes, chunk: sh, of: b });
    }
    list = out;
  }
  const bodies = list.map(b => {
    if (b.chunk) { const v = vel ? vel(b.chunk.p, b.of) : { v: [0, 0, 0], w: [0, 0, 0] }; return { shapes: [{ ...b.chunk }], v: v.v, w: v.w }; }
    let m = 0, c = [0, 0, 0];
    const shapes = b.boxes.map((bx) => {
      const s = boxShape(bx, 0);
      const w = mv(R, [s.p[0] - C[0], s.p[1] - C[1], s.p[2] - C[2]]);
      // (a free piece's shape is turned in the structure: its rotation follows the piece's)
      s.p = [T[0] + w[0], T[1] + w[1], T[2] + w[2]]; s.q = s.hull >= GpuWorld.hulls.lib ? qmul(qs, s.q) : qs;
      // (a cell's mass is at its centroid, not its box's centre: near enough for the velocity it starts with)
      const vol = shapeVol(s); m += vol; for (let k = 0; k < 3; k++) c[k] += vol * s.p[k];
      return s;
    });
    c = c.map(x => x / m);
    const v = vel ? vel(c, b) : { v: [0, 0, 0], w: [0, 0, 0] };
    // (v.u, v.r, where the structure had this part, are not used: the bodies and the ground they meet are where
    // the building was built, and a part put down by its sag started inside what is under it. Lab collapse, 13 Nov:
    // 101 energy gains while quiet with the shift, for 15. So a sagged part still jumps back by its sag as it lets go.)
    return { shapes, v: v.v, w: v.w };
  });
  // particles: equal-volume spheres, drawn as their bricks
  const pent = pl.map(b => {
    const s = boxShape(b.boxes[0], 0);
    const w = mv(R, [s.p[0] - C[0], s.p[1] - C[1], s.p[2] - C[2]]);
    const c = [T[0] + w[0], T[1] + w[1], T[2] + w[2]];
    const v = vel ? vel(c, b) : { v: [0, 0, 0], w: [0, 0, 0] };
    return { e: { p: c, v: v.v, w: v.w, q: s.hull >= GpuWorld.hulls.lib ? qmul(qs, s.q) : qs, r: Math.cbrt(6 * shapeVol(s) / 8 / Math.PI), mat: s.mat, wait: !!xf }, s };
  });
  const parts = gpu.addParticles ? gpu.addParticles(pent.map(x => x.e)) : [];
  for (const b of pl) { const k = b.id === -2 ? 'spill' : b.id < 0 ? 'crumb' : 'piece'; partsBy[k] = (partsBy[k] || 0) + 1; }   // (where particles come from)
  // (h: the box of the crumb's volume; a cell keeps its hull id, to be drawn as itself)
  parts.forEach((slot, k) => { if (slot < 0) return; const s = pent[k].s, e = s.hull ? Math.cbrt(GpuWorld.hulls.hulls[s.hull].vol) : 1; const g = pInfo[slot] ? pInfo[slot].gen + 1 : 1; pInfo[slot] = { h: s.h.map(x => x * e), hull: s.hull, col: s.col, gen: g, dead: false }; partsMade++; });
  const idxs = gpu.addBodies(bodies);
  list.forEach((b, k) => {
    if (b.id < 0) return;
    const idx = idxs[k], gb = idx ? gpu.bodies[idx - 1] : null;
    if (!gb) return;
    bodyInfo.set(idx, { piece: b.id, n: b.boxes.length, v0: bodies[k].v, f0: gpu.frame, C, qs, P: T, p0: gb.p.slice(), q0: gb.q.slice(), cool: clock() + (cool || 0) });
    armPlan(idx, b.id);
  });
  // (vel was called for the bodies kept, in order, then for the particles)
  return { idxs, list, parts };
}
function remeshDirty() {
  const dirty = []; for (let k = E.dirty_lo(); k <= E.dirty_hi(); k++) dirty.push(k);
  if (gpu) { for (const k of dirty) pendingChunks.push({ k, need: readSeq + 1 }); } else dirty.forEach(chunkMesh);
}
let lastBlast = '', structPending = false, structRounds = 0, failedJoints = 0, breaks = 0, structRatio = 0;
// the dynamic structure (docs/DYNAMIC.md): still moving after its last step (another is due), its fastest cell
// (m/s), its largest displacement (m), the solver's iterations, and the steps it has moved in
let structMoving = false, structSpeed = 0, structSag = 0, structIts = 0, dynSteps = 0;
let dustBricks = 0, impactMs = 0, dropped = 0, workerMs = 0, spillLost = 0; const partsBy = {};   // (spillLost: tick^3 of spill with no particle room)
// each engine job's time in the engine (ms), for the lab
const engineMsLog = [];
// the structure's steps since the page started: how many, the solver's iterations, steps that stopped at its cap,
// and their times in the engine (the benchmark's report, docs/DYNPLAN.md stage 0)
const structStat = { steps: 0, its: 0, capped: 0, maxMs: 0, ms: [] };

