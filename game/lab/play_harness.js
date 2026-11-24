// The page as a player has it, on a virtual clock (game/lab/play_check.cjs loads this before the page's scripts).
// Not the lab: the lab runs the engine inline, with no debris limit, its own frame loop and one act per run. Here
// the page's own loop runs, with the worker, the saved settings and the budgets, and only time is replaced:
// - a frame is 1/hz s whatever the wall time (performance.now and requestAnimationFrame);
// - the GPU's work and its buffer maps are waited for before the next frame: a GPU that answers within the frame,
//   which the software GPU (a second a step) never is, and a player's GPU always;
// - the worker's answers reach the page jobLag frames after they were asked for, debris read-backs readLag frames.
// What happens when is then the same on every run, and a slow machine changes only how long the check takes.
(() => {
  const C = window.__HCFG || {};
  const H = window.__H = { f: 0, raf: [], sent: [], held: [], reads: [], maps: 0, script: C.script || [], samples: [], done: false, stopAt: C.stopAt || 600, every: C.every || 10, log: [], gate: false };
  const FR = 1000 / (C.hz || 60); let vt = 5000;
  const sleep = (ms) => new Promise(r => setTimeout(r, ms));
  performance.now = () => vt;
  window.requestAnimationFrame = (cb) => { H.raf.push(cb); return 0; };
  const jobLag = () => (C.jobLag ?? 2), readLag = () => (C.readLag ?? 2);
  if (C.noGpu) Object.defineProperty(navigator, 'gpu', { get: () => undefined });
  // (failCompile: the shader modules come back with an error, as from a compiler that refuses them)
  if (C.failCompile) { const c0 = GPUDevice.prototype.createShaderModule; let n = 0; GPUDevice.prototype.createShaderModule = function (d) { const m = c0.call(this, d); if (d.code.length > 5000) m.getCompilationInfo = async () => ({ messages: [{ type: 'error', lineNum: 1929, linePos: 1, message: 'a refusal made up by the check' }] }); return m; }; }
  if (C.noWorker) window.Worker = function () { throw new Error('no worker'); };
  else {
    const RW = window.Worker;
    window.Worker = class {
      constructor(u) { this.w = new RW(u); this._on = null;
        this.w.onmessage = (m) => { if (m.data.op === 'ready') { this._on && this._on(m); return; } const at = H.sent.shift(); H.held.push({ due: at + jobLag(), m, self: this }); };
        this.w.onerror = (e) => { H.log.push('worker error ' + e.message); this.onerror && this.onerror(e); }; }
      set onmessage(f) { this._on = f; } get onmessage() { return this._on; }
      postMessage(msg, tr) { if (msg.op !== 'init' && msg.op !== 'param') H.sent.push(H.f); this.w.postMessage(msg, tr || []); }
    };
  }
  // every buffer map counted: a frame waits for the GPU's answers (a fast GPU answers within the frame)
  const map0 = GPUBuffer.prototype.mapAsync;
  GPUBuffer.prototype.mapAsync = function (...a) { H.maps++; const p = map0.apply(this, a); const dn = () => { H.maps--; }; p.then(dn, dn); return p; };
  H.install = () => {   // once the page has its GPU world
    if (gpu) for (const k of ['readPose', 'readParticles']) { if (!gpu[k]) continue; const f0 = gpu[k].bind(gpu);
      gpu[k] = (...a) => { const rec = { due: H.f + readLag(), done: false, v: null, res: null }; f0(...a).then(v => { rec.v = v; rec.done = true; }); H.reads.push(rec); return new Promise(r => { rec.res = r; }); }; }
    if (gpu && C.fast) Object.defineProperty(gpu, 'gpuMs', { get: () => ({ all: 3 }), set: () => {} });
    H.render0 = renderer.render.bind(renderer); if (C.noRender) renderer.render = () => {};
    H.shots = {}; H.shot = () => { drawDebris(performance.now() + 1000); H.render0(scene, cam); H.shots[H.f - H.f0] = cv.toDataURL('image/jpeg', 0.85); };
    H.f0 = H.f; H.gate = true;
  };
  async function tick() {
    if (H.gate) {
      const next = H.f + 1;
      // a lost device answers nothing more: the harness stops waiting for it (the page's own frames go on, as on a real screen)
      if (!H.lostSeen && dev && !H.lostHook) { H.lostHook = 1; dev.lost.then(() => { H.lostSeen = 1; }); }
      if (!H.lostSeen) await Promise.race([dev.queue.onSubmittedWorkDone().catch(() => {}), sleep(5000)]);
      let guard = 0;
      while (!H.lostSeen && (H.maps > 0 || H.sent.some(at => at + jobLag() <= next) || H.reads.some(r => r.due <= next && !r.done)) && guard++ < 60000) await sleep(1);
      if (guard >= 60000) H.log.push('frame ' + next + ' waited 60 s: maps ' + H.maps + ' sent ' + H.sent.length + ' reads ' + H.reads.filter(r => !r.done).length);
    }
    H.f++; vt += FR;
    if (H.gate) {
      const due = H.held.filter(h => h.due <= H.f); H.held = H.held.filter(h => h.due > H.f);
      for (const h of due) h.self._on && h.self._on(h.m);
      const rd = H.reads.filter(r => r.due <= H.f && r.done); H.reads = H.reads.filter(r => !(r.due <= H.f && r.done));
      for (const r of rd) r.res(r.v);
      await null; await null; await sleep(0);
      for (const s of H.script) if (s.f === H.f - H.f0) { try { (0, eval)(s.code); } catch (e) { H.log.push('script ' + s.f + ': ' + e.message); } }
    }
    const cbs = H.raf; H.raf = [];
    for (const cb of cbs) { try { cb(vt); } catch (e) { H.log.push('frame ' + H.f + ': ' + e.message + ' ' + String(e.stack).split('\n')[1]); } }
    if (H.gate && (C.shots || []).includes(H.f - H.f0)) { try { H.shot(); } catch (e) { H.log.push('shot: ' + e.message); } }
    // (every frame: is less drawn than exists, and for how many frames running)
    if (H.gate && H.sample && !C.noGpu) { try { const x = H.sample(), d = (x.statDrawn + x.drawnB + x.drawnP) - (x.stat + x.bod + x.par); H.def = H.def || { run: 0, longest: 0, worst: 0, frames: 0, at: 0 }; if (d < -0.05) { H.def.run++; H.def.frames++; H.def.worst = Math.min(H.def.worst, d); if (H.def.run > H.def.longest) { H.def.longest = H.def.run; H.def.at = x.f; } } else H.def.run = 0; } catch (e) {} }
    if (H.gate && H.sample && (H.f - H.f0) % H.every === 0) { try { H.samples.push(H.sample()); } catch (e) { H.log.push('sample: ' + e.message); } }
    if (H.gate && H.f - H.f0 >= H.stopAt) { H.done = true; return; }
    setTimeout(tick, H.gate ? 0 : 16);
  }
  setTimeout(tick, 0);
  H.sample = () => {
    const vol = (h, hull) => 8 * h[0] * h[1] * h[2] * (hull ? GpuWorld.hulls.hulls[hull].vol : 1);
    let stat = 0; for (let k = 0; k < E.chunk_count(); k++) stat += E.chunk_volume(k);
    let statDrawn = 0; for (const k in chunkVol) statDrawn += chunkVol[k];
    let bod = 0, nb = 0, ballsLive = 0, ballP = null;
    if (gpu) gpu.bodies.forEach((b, i) => { if (!b || b.dead || i === 0) return; if (b.ball) { ballsLive++; if (lastBodies && i * 7 + 2 < lastBodies.length) ballP = [lastBodies[i * 7], lastBodies[i * 7 + 1], lastBodies[i * 7 + 2]].map(x => +x.toFixed(1)); return; } nb++; for (const k of gpu.bodyShapes.get(i + 1) || []) { const sh = gpu.shapes[k - 1]; if (sh && !sh.dead) bod += vol(sh.h, sh.hull); } });
    let par = 0, np = 0; pInfo.forEach(x => { if (x && !x.dead) { par += vol(x.h, 0); np++; } });
    let dB = 0, dP = 0, dn = 0; const r = snapCur;
    if (r && gpu) { const n = r.pose.length / 10; for (let k = 0; k < n; k++) { const sh = gpu.shapes[r.slots[k] - 1]; if (!sh || sh.dead || (gpu.bodies[sh.body - 1] || {}).ball) continue; const o = k * 10; if (r.pose[o + 1] < -500) continue; dB += vol([r.pose[o + 7], r.pose[o + 8], r.pose[o + 9]], sh.hull); dn++; }
      if (r.parts && r.parts.list) for (const i of r.parts.list) { const x = pInfo[i]; if (x && !x.dead) dP += vol(x.h, 0); } }
    return { f: H.f - H.f0, bricks: E.bricks_alive(), stat: +stat.toFixed(2), statDrawn: +statDrawn.toFixed(2), bod: +bod.toFixed(2), nb, par: +par.toFixed(2), np, drawnB: +dB.toFixed(2), drawnP: +dP.toFixed(2), dn, awake: gpu ? gpu.stats.awake : -1, pAct: gpu ? gpu.stats.pActive : -1, steps: stepsDone, atRead: stepsAtRead, read: readSeq, skipped: gpu ? gpu.skipped || 0 : 0, capped: debrisCapped, pFar: partsFar, bFar: bodiesFar, pCrushed: partsCrushed, dust: dustBricks, lost: spillLost, pendCh: pendingChunks.length, busy: busy ? busy.op : '', balls: ballsLive, ballP, sub: gpu ? gpu.substeps : 0, dLive: debrisLive, issues: [...health.issues.keys()].join(','), punch: [punchStat.sent, punchStat.through, punchStat.stopped], last: (lastBlast || '').slice(0, 60) };
  };
})();
