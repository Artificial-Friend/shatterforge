// ---- tower meshes, one per storey band
const towerMat = new THREE.MeshLambertMaterial({ vertexColors: true });
const chunks = [], chunkVol = {};
function chunkMesh(k) {
  chunkVol[k] = E.chunk_volume ? E.chunk_volume(k) : 0;
  if (LAB.on) LAB.staticDrawn = Object.values(chunkVol).reduce((a, b) => a + b, 0);
  const vl = E.chunk_vlen(k), il = E.chunk_ilen(k);
  const v = new Float32Array(E.memory.buffer, E.chunk_vptr(k), vl).slice();
  const ix = new Uint32Array(E.memory.buffer, E.chunk_iptr(k), il).slice();
  const g = new THREE.BufferGeometry();
  const ib = new THREE.InterleavedBuffer(v, 9);
  g.setAttribute('position', new THREE.InterleavedBufferAttribute(ib, 3, 0));
  g.setAttribute('normal', new THREE.InterleavedBufferAttribute(ib, 3, 3));
  g.setAttribute('color', new THREE.InterleavedBufferAttribute(ib, 3, 6));
  g.setIndex(new THREE.BufferAttribute(ix, 1));
  if (chunks[k]) { scene.remove(chunks[k]); chunks[k].geometry.dispose(); }
  const m = new THREE.Mesh(g, towerMat); m.castShadow = true; m.receiveShadow = true;
  chunks[k] = m; scene.add(m);
}

// ---- debris (GPU bodies), drawn as instanced boxes
// every debris shape the GPU can hold is drawn: with fewer instances than shapes, which ones were left out
// changed from frame to frame (flicker, half the building gone)
const MAXP = 65536;   // particles the GPU holds (crumbs)
const MAXD = 140000 + MAXP;
const dMesh = new THREE.InstancedMesh(new THREE.BoxGeometry(2, 2, 2), new THREE.MeshLambertMaterial(), MAXD);
dMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage); dMesh.count = 0;
// debris moves every frame: never cull it by a bounding sphere computed once (it was computed while empty, so
// the camera culled all debris while the shadow pass still drew it)
dMesh.frustumCulled = false; dMesh.castShadow = true; dMesh.receiveShadow = true;
// the per-instance colour buffer, sized for every instance (setColorAt would size it by the current count, 0)
dMesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(MAXD * 3).fill(1), 3); dMesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
scene.add(dMesh);
// convex chunks: one instanced mesh per cell of the library (made when first needed), drawn from the same
// read-backs; a chunk's instance is its cell in [-1,1]^3, scaled by the shape's h
const hullMeshes = new Map(), HULL_CAP = 16384;
// (a runtime hull is one free piece's shape, shared only by identical members: 256 instances, not the library's
// 16,384, whose buffers take 1.2 MB a mesh)
const hullCap = (id) => id >= GpuWorld.hulls.lib ? 256 : HULL_CAP;
function hullMesh(id) {
  let m = hullMeshes.get(id); if (m) return m;
  const H = GpuWorld.hulls.hulls[id], pos = [];
  for (const f of H.faces) for (let t = 1; t + 1 < f.v.length; t++) for (const i of [f.v[0], f.v[t], f.v[t + 1]]) pos.push(...H.V[i]);
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.computeVertexNormals();
  m = new THREE.InstancedMesh(g, dMesh.material, hullCap(id));
  m.instanceMatrix.setUsage(THREE.DynamicDrawUsage); m.count = 0; m.frustumCulled = false; m.castShadow = true; m.receiveShadow = true;
  m.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(hullCap(id) * 3).fill(1), 3); m.instanceColor.setUsage(THREE.DynamicDrawUsage);
  scene.add(m); hullMeshes.set(id, m); return m;
}
// where each read-back shape is drawn: box (-1) or a chunk mesh, and its instance there
function drawTargets(r) {
  const n = r.pose.length / 10, tgt = new Int32Array(n), idx = new Int32Array(n), cnt = new Map();
  let nb = 0;
  for (let k = 0; k < n; k++) {
    const sh = gpu.shapes[r.slots[k] - 1], h = sh && sh.hull ? sh.hull : 0;
    if (!h) { tgt[k] = -1; idx[k] = nb++; continue; }
    const c = cnt.get(h) || 0; if (c >= hullCap(h)) { tgt[k] = -2; continue; }
    tgt[k] = h; idx[k] = c; cnt.set(h, c + 1);
  }
  r.tgt = tgt; r.tidx = idx; r.nBox = nb; r.hullCount = cnt;
}
// the same colours as the tower mesh (engine/src/mesh.rs colour): by material and paint, a little variation
const GLASS = 3, TIMBER = 4;
const PAINTCOL = [[0.62, 0.61, 0.58], [0.56, 0.56, 0.54], [0.79, 0.77, 0.71], [0.67, 0.65, 0.61], [0.58, 0.57, 0.55]];
const colTmp = new THREE.Color();
const VIEW_FREE = [0.56, 0.47, 0.70];   // (violet: not blue, green, yellow or red, the view's scale of stress)
function debrisColour(code, k) {
  const m = code & 15, paint = code >> 4;
  const c = m === GLASS ? [0.36, 0.50, 0.58] : m === 1 ? [0.60, 0.36, 0.28] : m === TIMBER ? (paint === 8 ? [0.62, 0.46, 0.30] : [0.86, 0.84, 0.79]) : m === 2 ? [0.36, 0.40, 0.44] : (PAINTCOL[paint] || [0.7, 0.7, 0.7]);
  const h = ((Math.imul(k, 2654435761) >>> 24) & 255) / 255, s = 0.96 + 0.06 * h;
  // In the engineer's view what has left the structure has one colour, off the scale of stress (VIEW_FREE): falling
  // pieces, crumbs and rubble. In their materials' colours the view seemed to switch off wherever a building came
  // down (reported 13 Nov: a fallen building drawn grey beside its blue columns).
  if (+setting.view) { const t = 0.90 + 0.16 * h; return colTmp.setRGB(VIEW_FREE[0] * t, VIEW_FREE[1] * t, VIEW_FREE[2] * t); }
  // the tower mesh is sRGB-like vertex colour: keep the same space
  return colTmp.setRGB(c[0] * s, c[1] * s, c[2] * s);
}
const M4 = new THREE.Matrix4(), QQ = new THREE.Quaternion(), PP = new THREE.Vector3(), SS = new THREE.Vector3();
let particlesDrawn = 0;
let reading = 0, debrisShapes = 0, pendingChunks = [], readGen = 0;
// Debris poses come back from the GPU a frame or more late, and not every frame: they are drawn blended
// between the last two read-backs (matched shape by shape), so motion stays smooth.
const flick = { vanished: 0, jumps: 0 };
let snapPrev = null, snapCur = null, stepsDone = 0, readsDone = 0, readSeq = 0, readLatest = 0, stepsAtRead = 0;
// main-thread time per frame by stage (smoothed), shown in the HUD
const perf = {};
function timed(name, fn) { const t = performance.now(); const r = fn(); const d = performance.now() - t; perf[name] = perf[name] == null ? d : perf[name] * 0.9 + d * 0.1; return r; }
function perfAdd(name, d) { perf[name] = perf[name] == null ? d : perf[name] * 0.9 + d * 0.1; }
// Several read-backs are kept in flight (each resolves a frame or more after it was issued), so a fresh one
// arrives every frame; an older one arriving after a newer is dropped.
const MAX_READS = 3;
let lastReadReal = -1, drawnFinal = null;
function pullDebris() {
  if (!gpu || !diag.readback || reading >= MAX_READS) return;
  // pieces of a break wait for the next step to be placed: a read-back before it would show them where the
  // parent was, then jump
  if (gpu.placing()) return;
  // at rest (every step since the last read-back skipped, nothing changed): the last one still holds, and the
  // GPU and the page do nothing for it; settled rubble is scenery
  // (and the steps since count as read: the page steps at most 3 past the newest read-back, and with none issued at
  // rest it stopped stepping, so whatever came next, a ball, a blast's pieces, never stepped, never read, never
  // showed: "clicking does nothing" on the complex building, whose furniture falls asleep while the page still steps;
  // and the benchmark stood still at 39.6 s once its rubble had settled. Reproduced headless, play/rest.cjs.)
  if (!LAB.on && gpu.realFrame === lastReadReal && snapCur) { stepsAtRead = stepsDone; return; }
  lastReadReal = gpu.realFrame;
  reading++;
  const seq = ++readSeq, issuedAt = stepsDone;
  const tr = performance.now();
  // particle generations now: a slot freed and reused before this read-back arrives holds another particle,
  // and the data shows the old one (matching them by today's generation drew a crumb flying across the scene)
  const pgen = pInfo.map(x => (x ? x.gen : 0));
  labGate(Promise.all([gpu.readPose(debrisFrom), gpu.readParticles ? gpu.readParticles() : null])).then(([r, pr]) => {
    r.parts = pr;
    perfAdd('read wait', performance.now() - tr);
    reading--;
    if (seq < readLatest) return;
    readLatest = seq; readsDone++; stepsAtRead = Math.max(stepsAtRead, issuedAt);
    const t0 = performance.now();
    // the building's mesh changes in the frame the debris that replaces it first shows: with the first
    // read-back issued after that debris was made
    if (pendingChunks.length) {
      const due = new Set(), keep = [];
      for (const e of pendingChunks) { if (e.need <= seq) due.add(e.k); else keep.push(e); }
      pendingChunks = keep; for (const k of due) chunkMesh(k);
    }
    r.t = clock();
    // shapes are matched between read-backs by slot and generation (a reused slot holds another body)
    r.index = new Map(); for (let k = 0; k < r.slots.length; k++) r.index.set(r.gens[k] * 1048576 + r.slots[k], k);
    snapPrev = snapCur; snapCur = r;
    // flicker check: between two read-backs, a shape must not vanish unless its body was broken or removed, nor
    // move much further than its speed allows
    if (snapPrev && snapPrev.frame < r.frame) {
      const P = snapPrev, dt = (r.frame - P.frame) / 60;
      for (let k = 0; k < P.slots.length; k++) {
        const key = P.gens[k] * 1048576 + P.slots[k], j = r.index.get(key);
        if (j === undefined) {
          const sh = gpu.shapes[P.slots[k] - 1];
          if (sh && !sh.dead && (gpu.shapeGen ? gpu.shapeGen[P.slots[k] - 1] || 0 : 0) === P.gens[k]) flick.vanished++;
          continue;
        }
        const dx = r.pose[j * 10] - P.pose[k * 10], dy = r.pose[j * 10 + 1] - P.pose[k * 10 + 1], dz = r.pose[j * 10 + 2] - P.pose[k * 10 + 2];
        const b = P.slots[k] && gpu.shapes[P.slots[k] - 1] ? gpu.shapes[P.slots[k] - 1].body : 0, vo = (b - 1) * 3;
        const sp = r.vel && vo + 2 < r.vel.length ? Math.hypot(r.vel[vo], r.vel[vo + 1], r.vel[vo + 2]) : 0;
        if (Math.hypot(dx, dy, dz) > 1 + 3 * (sp + 10) * dt) flick.jumps++;
      }
    }
    // particles: crushed ones go (as dust); the draw list is the live ones with their colours
    if (pr) {
      const d = pr.data, list = [];
      for (let i = 0; i < pr.n; i++) {
        const st = d[i * 12 + 10], inf = pInfo[i];
        if (!inf || inf.dead || st === 0 || st === 4 || inf.gen !== pgen[i]) continue;
        if (st === 3) { inf.dead = true; gpu.killParticle(i); partsCrushed++; continue; }
        if (farDist < Infinity) {
          const x = d[i * 12], y = d[i * 12 + 1], z = d[i * 12 + 2];
          if (!inf.o) inf.o = [x, y, z];
          else if (Math.hypot(x - inf.o[0], y - inf.o[1], z - inf.o[2]) > farDist) { inf.dead = true; window.__farKill = true; try { gpu.killParticle(i); } finally { window.__farKill = false; } partsFar++; continue; }
        }
        list.push(i);
      }
      pr.list = list; pr.gens = list.map(i => pInfo[i].gen);
      pr.index = new Map(); list.forEach((i, k) => pr.index.set(pInfo[i].gen * 1048576 + i, k));
    }
    cleanFar(r);
    capDebris(r, pr);
    if (LAB.on) labOnRead(r, pr);
    velHist.unshift({ v: r.vel, frame: r.frame }); if (velHist.length > 4) velHist.pop(); lastReadFrame = r.frame; lastBodies = r.body; lastVel = r.vel; lastAng = r.ang;
    drawTargets(r);
    const n = Math.min(MAXD, r.pose.length / 10), nb = Math.min(MAXD, r.nBox);
    for (let k = 0; k < n; k++) { const t = r.tgt[k]; if (t === -2) continue; (t < 0 ? dMesh : hullMesh(t)).setColorAt(r.tidx[k], debrisColour(r.cols[k], r.slots[k])); }
    const np = pr ? Math.min(MAXD - nb, pr.list.length) : 0;
    for (let k = 0; k < np; k++) dMesh.setColorAt(nb + k, debrisColour(pInfo[pr.list[k]].col, 7919 * pr.list[k] + 1));
    if (dMesh.instanceColor) { dMesh.instanceColor.updateRange.count = (nb + np) * 3; dMesh.instanceColor.needsUpdate = true; }
    for (const [h, c] of r.hullCount) { const m = hullMesh(h); m.instanceColor.updateRange.count = c * 3; m.instanceColor.needsUpdate = true; }
    perfAdd('read', performance.now() - t0 + (r.jsMs || 0));
  }, () => { reading = Math.max(0, reading - 1); });
}
const QA = new THREE.Quaternion(), QB = new THREE.Quaternion();
function drawDebris(now) {
  const r = snapCur; if (!r) return;
  const p = snapPrev;
  // blend from the previous read-back to the latest over the interval between them (one interval behind)
  const span = p ? Math.max(r.t - p.t, 1) : 1;
  const a = p ? Math.min(1, (now - r.t) / span) : 1;
  // (drawn once at the end of its blend, a read-back stays drawn: nothing to rebuild until the next one)
  if (!LAB.on && a >= 1 && drawnFinal === r) return;
  if (a >= 1) drawnFinal = r;
  const n = Math.min(MAXD, r.pose.length / 10);
  const lab = LAB.on; if (lab) labDrawBegin();
  for (let k = 0; k < n; k++) {
    const o = k * 10;
    const j = p && a < 1 ? p.index.get(r.gens[k] * 1048576 + r.slots[k]) : undefined;
    if (j !== undefined) {
      const q = j * 10, b = 1 - a;
      PP.set(p.pose[q] * b + r.pose[o] * a, p.pose[q + 1] * b + r.pose[o + 1] * a, p.pose[q + 2] * b + r.pose[o + 2] * a);
      QA.set(p.pose[q + 3], p.pose[q + 4], p.pose[q + 5], p.pose[q + 6]); QB.set(r.pose[o + 3], r.pose[o + 4], r.pose[o + 5], r.pose[o + 6]);
      QQ.copy(QA).slerp(QB, a);
    } else {
      PP.set(r.pose[o], r.pose[o + 1], r.pose[o + 2]); QQ.set(r.pose[o + 3], r.pose[o + 4], r.pose[o + 5], r.pose[o + 6]);
    }
    SS.set(r.pose[o + 7], r.pose[o + 8], r.pose[o + 9]);
    M4.compose(PP, QQ, SS);
    const tg = r.tgt ? r.tgt[k] : -1;
    if (tg === -1) dMesh.setMatrixAt(r.tidx ? r.tidx[k] : k, M4); else if (tg > 0) hullMesh(tg).setMatrixAt(r.tidx[k], M4);
    if (lab) {
      const sh = gpu.shapes[r.slots[k] - 1], b = sh ? sh.body : 0, vo = (b - 1) * 3;
      const sp = r.vel && vo >= 0 && vo + 2 < r.vel.length ? Math.hypot(r.vel[vo], r.vel[vo + 1], r.vel[vo + 2]) : 0;
      const hv = tg > 0 ? GpuWorld.hulls.hulls[tg].vol : 1;
      labDrawn(r.gens[k] * 1048576 + r.slots[k], PP.x, PP.y, PP.z, sp, (sh && sh.labBall) || tg === -2 ? 0 : 8 * SS.x * SS.y * SS.z * hv);
    }
  }
  // particles, blended the same way (matched by slot and generation)
  let np = 0, np2 = 0;
  const pHull = new Map();
  const nb = r.tgt ? Math.min(MAXD, r.nBox) : n;
  const pr = r.parts, pp = p && p.parts;
  if (pr && pr.list) {
    // (all of them: drawn as cells they do not take the box mesh's instances; those drawn as boxes stop at MAXD)
    np = pr.list.length;
    const d = pr.data, pd = pp && pp.data;
    for (let k = 0; k < np; k++) {
      const i = pr.list[k], o = i * 12, inf = pInfo[i];
      const j = pp && pp.index && a < 1 ? pp.index.get(pr.gens[k] * 1048576 + i) : undefined;
      if (j !== undefined) {
        const q = pp.list[j] * 12, b = 1 - a;
        PP.set(pd[q] * b + d[o] * a, pd[q + 1] * b + d[o + 1] * a, pd[q + 2] * b + d[o + 2] * a);
        QA.set(pd[q + 6], pd[q + 7], pd[q + 8], pd[q + 9]); QB.set(d[o + 6], d[o + 7], d[o + 8], d[o + 9]);
        QQ.copy(QA).slerp(QB, a);
      } else { PP.set(d[o], d[o + 1], d[o + 2]); QQ.set(d[o + 6], d[o + 7], d[o + 8], d[o + 9]); }
      SS.set(inf.h[0], inf.h[1], inf.h[2]);
      const vol = 8 * SS.x * SS.y * SS.z;
      // a crumb is drawn as a Voronoi cell (one of the library's, by its slot), scaled to the crumb's volume; as a
      // cube it read as a pile of boxes. (Its physics is a sphere either way.)
      const hid = inf.hull ? inf.hull : setting.chunks && GpuWorld.hulls ? 1 + (Math.imul(i + 1, 2654435761) >>> 0) % (GpuWorld.hulls.hulls.length - 1) : 0;
      if (hid) {
        const sc = Math.cbrt(1 / GpuWorld.hulls.hulls[hid].vol); SS.multiplyScalar(sc);
        M4.compose(PP, QQ, SS);
        const c = (r.hullCount ? r.hullCount.get(hid) || 0 : 0) + (pHull.get(hid) || 0);
        if (c < hullCap(hid)) { const m = hullMesh(hid); m.setMatrixAt(c, M4); m.setColorAt(c, debrisColour(inf.col || 0, i)); pHull.set(hid, (pHull.get(hid) || 0) + 1); }
      } else if (nb + np2 < MAXD) { M4.compose(PP, QQ, SS); dMesh.setMatrixAt(nb + np2, M4); np2++; }
      if (lab) labDrawn(-(pr.gens[k] * 1048576 + i) - 1, PP.x, PP.y, PP.z, Math.hypot(d[o + 3], d[o + 4], d[o + 5]), vol);
    }
  }
  dMesh.count = nb + np2; debrisShapes = n; particlesDrawn = np;
  dMesh.instanceMatrix.updateRange.count = (nb + np2) * 16; dMesh.instanceMatrix.needsUpdate = true;
  for (const [h, m] of hullMeshes) { const c = Math.min(hullCap(h), (r.hullCount ? r.hullCount.get(h) || 0 : 0) + (pHull.get(h) || 0)); m.count = c; if (c) { m.instanceMatrix.updateRange.count = c * 16; m.instanceMatrix.needsUpdate = true; if (pHull.get(h)) { m.instanceColor.updateRange.count = c * 3; m.instanceColor.needsUpdate = true; } } }
}

