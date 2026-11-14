// ---- the simulation engine runs in a worker
// The worker holds the authoritative brick store and does everything that changes it: blasts, the
// structural check, impacts. The page keeps a copy for walking, aiming and meshes, replaced from the worker
// after each change. One job runs at a time; blasts go first, then impacts, then structural rounds.
const WORKER_SRC = `
let E = null;
onmessage = async (m) => {
  const d = m.data;
  if (d.op === 'init') {
    E = (await WebAssembly.instantiate(d.wasm, {})).instance.exports;
    E.game_new_kind(d.storeys, d.kind); E.game_mesh(0);
    postMessage({ op: 'ready' }); return;
  }
  if (d.op === 'param') { E.set_param(d.id, d.v); return; }
  const t0 = performance.now();
  let n = 0;
  if (d.op === 'blast') n = E.blast(d.x, d.y, d.z, d.r);
  else if (d.op === 'structure') { E.set_param(4, d.dt || 0); if (d.rest) E.rest_begin(); if (d.rest) for (let k = 0; k < d.rest.length; k += 4) E.rest_load(d.rest[k], d.rest[k + 1], d.rest[k + 2], d.rest[k + 3]); n = E.structure_step(d.max); }
  else if (d.op === 'pblast') n = E.piece_blast(d.piece, d.x, d.y, d.z, d.r);
  else if (d.op === 'impacts') { const p = E.impacts_in(d.recs.length / 11); new Float32Array(E.memory.buffer, p, d.recs.length).set(d.recs); n = E.impacts(d.dt); }
  else if (d.op === 'snap') { const o = []; for (const pc of d.pieces) { n += E.piece_snap(pc); o.push(...new Float32Array(E.memory.buffer, E.out_ptr(), E.out_len())); } d.outs = o; }
  const out = d.op === 'snap' ? Float32Array.from(d.outs) : new Float32Array(E.memory.buffer, E.out_ptr(), E.out_len()).slice();
  const plans = new Float32Array(E.memory.buffer, E.plan_ptr(), E.plan_len()).slice();
  let snap = null;
  // (a structure step that failed nothing may still send pieces off: loose room contents, at the first one)
  const detached = d.op === 'structure' && E.last_detached ? E.last_detached() : 0;
  if (n > 0 || detached > 0) { const len = E.snap_save(); snap = new Uint8Array(E.memory.buffer, E.snap_ptr(), len).slice(); }
  const r = { op: d.op, n, detached, out, snap, lo: E.dirty_lo(), hi: E.dirty_hi(), ratio: E.structure_ratio(), wearing: E.structure_wearing(), worn: E.worn_total(), dust: E.dust_bricks(), cracked: E.cracked_total(), restOn: E.rest_applied(), plans, ms: performance.now() - t0, moving: E.structure_dyn ? E.structure_dyn(0) > 0 : false, dynSpeed: E.structure_dyn ? E.structure_dyn(1) : 0, dynIts: E.structure_dyn ? E.structure_dyn(2) : 0, sag: E.structure_dyn ? E.structure_dyn(4) : 0, dynLeft: E.structure_dyn ? E.structure_dyn(5) : 0, dynCut: E.structure_dyn ? E.structure_dyn(6) : 0, dynCapped: E.structure_dyn ? E.structure_dyn(7) : 0, view: (d.op === 'structure' && E.view_len && E.view_len()) ? new Float32Array(E.memory.buffer, E.view_ptr(), E.view_len()).slice() : null };
  postMessage(r, snap ? [out.buffer, snap.buffer, plans.buffer] : [out.buffer, plans.buffer]);
};`;
let worker = null, busy = null;
const jobs = { blast: [], impacts: null, structure: false };
function startWorker(bytes) {
  return new Promise((res) => {
    try {
      const w = new Worker(URL.createObjectURL(new Blob([WORKER_SRC], { type: 'text/javascript' })));
      w.onmessage = (m) => { if (m.data.op === 'ready') { worker = w; w.onmessage = (m) => finish(m.data); res(true); } };
      w.onerror = () => res(false);
      w.postMessage({ op: 'init', wasm: bytes, storeys: STOREYS, kind: KIND });
      setTimeout(() => res(false), 15000);
    } catch (e) { res(false); }
  });
}
// run a job inline (no worker): the page's own engine is then the authoritative one
function runInline(job) {
  const t0 = performance.now();
  let n = 0;
  if (job.op === 'blast') n = E.blast(job.x, job.y, job.z, job.r);
  else if (job.op === 'structure') { E.set_param(4, job.dt || 0); if (job.rest) E.rest_begin(); if (job.rest) for (let k = 0; k < job.rest.length; k += 4) E.rest_load(job.rest[k], job.rest[k + 1], job.rest[k + 2], job.rest[k + 3]); n = E.structure_step(job.max); }
  else if (job.op === 'pblast') n = E.piece_blast(job.piece, job.x, job.y, job.z, job.r);
  else if (job.op === 'snap') { const o = []; for (const pc of job.pieces) { n += E.piece_snap(pc); o.push(...new Float32Array(E.memory.buffer, E.out_ptr(), E.out_len())); } job.outs = o; }
  else { const p = E.impacts_in(job.recs.length / 11); new Float32Array(E.memory.buffer, p, job.recs.length).set(job.recs); n = E.impacts(job.dt); }
  return { op: job.op, n, detached: job.op === 'structure' && E.last_detached ? E.last_detached() : 0, out: job.op === 'snap' ? Float32Array.from(job.outs) : new Float32Array(E.memory.buffer, E.out_ptr(), E.out_len()).slice(), snap: null, lo: E.dirty_lo(), hi: E.dirty_hi(), ratio: E.structure_ratio(), wearing: E.structure_wearing(), worn: E.worn_total(), dust: E.dust_bricks(), cracked: E.cracked_total(), restOn: E.rest_applied(), moving: E.structure_dyn ? E.structure_dyn(0) > 0 : false, dynSpeed: E.structure_dyn ? E.structure_dyn(1) : 0, dynIts: E.structure_dyn ? E.structure_dyn(2) : 0, sag: E.structure_dyn ? E.structure_dyn(4) : 0, dynLeft: E.structure_dyn ? E.structure_dyn(5) : 0, dynCut: E.structure_dyn ? E.structure_dyn(6) : 0, dynCapped: E.structure_dyn ? E.structure_dyn(7) : 0, view: (job.op === 'structure' && E.view_len && E.view_len()) ? new Float32Array(E.memory.buffer, E.view_ptr(), E.view_len()).slice() : null, plans: new Float32Array(E.memory.buffer, E.plan_ptr(), E.plan_len()).slice(), ms: performance.now() - t0, inline: true };
}
// Rubble lying on the building weighs on it: once a second the weight of the debris at rest (bodies slower than
// 0.5 m/s and particles asleep) goes to the engine with a structure check, binned in 0.5 m cells; the engine
// puts each on the standing brick under it, and a floor heaped too high gives way.
const REST_EVERY = 1000;   // ms (clock() is in ms, real or the lab's virtual one)
let restNext = 0, restSent = 0, restOnFloors = 0, crackedJoints = 0, restParts = 0, restBodies = 0;
const DENS = [2400, 1900, 7850, 2500, 500];
function restLoads() {
  const bins = new Map();
  const add = (x, y, z, f) => {
    if (!(y > 1) || !Number.isFinite(x + y + z + f)) return;
    const key = `${Math.floor(x * 2)},${Math.floor(y)},${Math.floor(z * 2)}`, e = bins.get(key);
    if (e) { e[0] += x * f; e[1] = Math.max(e[1], y); e[2] += z * f; e[3] += f; } else bins.set(key, [x * f, y, z * f, f]);
  };
  if (gpu && lastBodies && lastVel) for (let i = 2; i <= gpu.n; i++) {
    const b = gpu.bodies[i - 1], o = (i - 1) * 7, vo = (i - 1) * 3;
    if (!b || b.dead || !b.mass || o + 2 >= lastBodies.length) continue;
    if (Math.hypot(lastVel[vo], lastVel[vo + 1], lastVel[vo + 2]) > 0.5) continue;
    add(lastBodies[o], lastBodies[o + 1], lastBodies[o + 2], b.mass * 9.81);
  }
  const pr = snapCur && snapCur.parts;
  let partW = 0;
  if (pr && pr.list) for (const i of pr.list) {
    const d = pr.data, inf = pInfo[i];
    if (d[i * 12 + 10] !== 2 || !inf) continue;
    const m = 8 * inf.h[0] * inf.h[1] * inf.h[2] * (DENS[(inf.col || 0) & 15] || 2000);
    if (d[i * 12 + 1] > 1) partW += m * 9.81;
    add(d[i * 12], d[i * 12 + 1], d[i * 12 + 2], m * 9.81);
  }
  const out = [];
  let total = 0;
  for (const e of bins.values()) { out.push(e[0] / e[3], e[1], e[2] / e[3], e[3]); total += e[3]; }
  restSent = total;
  restParts = partW; restBodies = total - partW;
  return out;
}
// A structure check carries the time since the last one: joints loaded near their strength wear for that long
// (engine: wear_joints). While some are wearing, checks keep coming every WEAR_EVERY ms, so a building left near
// its limit cracks and gives way over seconds rather than standing unchanged.
const WEAR_EVERY = 250;
let structT = null, wearNext = 0, structWearing = 0, wornJoints = 0;
function structJob(extra) { const t = clock(), dt = structT == null ? 0 : Math.min(1, (t - structT) / 1000); structT = t; return { op: 'structure', max: 48, dt, ...extra }; }
function pump() {
  if (busy) return;
  let job = null;
  if (jobs.blast.length) job = jobs.blast.shift();
  else if (jobs.impacts) { job = jobs.impacts; jobs.impacts = null; }
  else if (gpu && setting.restLoads && clock() >= restNext && (!physicsPaused || window.__forceJobs)) { restNext = clock() + REST_EVERY; jobs.structure = false; job = structJob({ rest: restLoads() }); }
  else if ((jobs.structure || (wearNext && clock() >= wearNext)) && gpu && (!physicsPaused || window.__forceJobs)) { jobs.structure = false; wearNext = 0; job = structJob({}); }
  else if (jobs.structure && !gpu) { jobs.structure = false; job = structJob({}); }
  if (!job) return;
  busy = job;
  if (LAB.on) { LAB.jobq.push({ job, due: LAB.f + LAB.jobLag }); return; }
  if (worker) { const { ctx, ...msg } = job; worker.postMessage(msg); }
  else finish(runInline(job));
}
function finish(r) {
  const tf = performance.now();
  finishInner(r);
  perfAdd('apply', performance.now() - tf);
  pump();
}
function finishInner(r) {
  const job = busy; busy = null;
  workerMs = r.ms; dustBricks = r.dust; structRatio = r.ratio;
  if (r.op === 'structure') { structMoving = !!r.moving; structSpeed = r.dynSpeed || 0; structSag = r.sag || 0; structIts = r.dynIts || 0; if (structMoving) dynSteps++; }
  engineMsLog.push(r.ms); if (engineMsLog.length > 5000) engineMsLog.shift();
  if (r.op === 'structure') { structStat.steps++; structStat.its += r.dynIts || 0; if (r.dynCapped) structStat.capped++; if (r.ms > structStat.maxMs) structStat.maxMs = r.ms; structStat.ms.push(r.ms); if (structStat.ms.length > 5000) structStat.ms.shift(); }
  if (r.cracked !== undefined) crackedJoints = r.cracked;
  if (r.op === 'structure' && job && job.rest) restOnFloors = r.restOn || 0;
  // the page's copy of the store
  if (r.snap) { const p = E.snap_in(r.snap.length); new Uint8Array(E.memory.buffer, p, r.snap.length).set(r.snap); E.snap_load(r.lo, r.hi); }
  readPlans(r);
  // (a hit on the dynamic structure is a load of its next step: it must have one)
  if (r.op === 'impacts' && +setting.dynamic) jobs.structure = true;
  if (r.op === 'blast') onBlast(job, r);
  else if (r.op === 'snap') onSnap(job, r);
  else if (r.op === 'pblast') onImpacts(job, r);
  else if (r.op === 'structure') onStructure(r);
  else onImpacts(job, r);
  if (r.view && viewParam()) showView(r.view);
}
