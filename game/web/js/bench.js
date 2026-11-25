// ---- benchmark: a scripted, seeded destruction with a report
// (after the collapse, 12 s: then up to 30 s more for it to settle, and 2 s once all is at rest, to measure what a
// settled scene costs: a game keeps its rubble on screen long after it fell)
const BENCH = { on: false, phase: '', t0: 0, f0: 0, rows: [], frameMs: [], balls: [[180, -4, 9, -30, 0.18, 0.05], [300, 6, 6, -30, -0.15, 0.02], [420, 0, 3, -30, 0, 0.0]], next: 0, end: 720, settleEnd: 720 + 1800, restFrames: 120, gpuMax: 0, settledAt: null, restMs: [], restGpu: [] };
function seededRandom(seed) { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const benchAsk = (which) => { try { localStorage.setItem('sf-bench', which); } catch (e) {} location.reload(); };
$('bench-go').addEventListener('click', () => benchAsk('1'));
$('bench-go2').addEventListener('click', () => benchAsk('ball'));
$('bench-close').addEventListener('click', () => { $('bench').hidden = true; });
$('lab-quick').addEventListener('click', () => labStart(LAB_SUITES.quick));
$('lab-full').addEventListener('click', () => labStart(LAB_SUITES.full));
$('bench-copy').addEventListener('click', () => { const t = LAB.copyText || $('bench-out').textContent; navigator.clipboard.writeText(t).catch(() => { const r = document.createRange(); r.selectNodeContents($('bench-out')); const sel = getSelection(); sel.removeAllRanges(); sel.addRange(r); }); });
// (which one was asked for before the reload: '1' the collapse, 'ball' the ball on the roof; '' none)
function benchPending() { try { const w = localStorage.getItem('sf-bench'); if (w === '1' || w === 'ball') { localStorage.removeItem('sf-bench'); return w; } } catch (e) {} return ''; }
// The second benchmark (24 Nov): a steel ball 5 m across, 514 t, set down at rest on the roof of the office tower.
// Nothing is cut and nothing thrown: the tower fails under the weight. The roof gives way (the weight of what lies
// at rest goes to the structure once a second, worker.js restLoads; natively a 263 t ball already fails this roof,
// docs/LAB.md), the ball comes down a storey, and the next floor has its weight and its fall to carry.
const BENCH_BALL_R = 2.5;
function benchBallY() { const b = BENCH.ballBody; return b && lastBodies && (b - 1) * 7 + 1 < lastBodies.length ? lastBodies[(b - 1) * 7 + 1] : (BENCH.ballAt ? BENCH.ballAt[1] : 0); }
// What one compute dispatch costs on this device (empty work, 4,000 of them in one pass, wall time over the
// queue): the solver's sub-steps are many small dispatches, so this bounds how fast a step can be here.
async function dispatchCost() {
  try {
    const mod = dev.createShaderModule({ code: '@group(0) @binding(0) var<storage, read_write> o: array<u32>; @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3u) { if (g.x == 0xffffffffu) { o[0] = 1u; } }' });
    const pl = dev.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'main' } });
    const buf = dev.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
    const bg = dev.createBindGroup({ layout: pl.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: buf } }] });
    const run = async (n) => { const e = dev.createCommandEncoder(); const p = e.beginComputePass(); p.setPipeline(pl); p.setBindGroup(0, bg); for (let k = 0; k < n; k++) p.dispatchWorkgroups(1); p.end(); const t = performance.now(); dev.queue.submit([e.finish()]); await dev.queue.onSubmittedWorkDone(); return performance.now() - t; };
    await run(100);
    // (the fastest of 3 each: other work on the machine only ever adds; the difference cancels the submit's cost)
    let a = Infinity, b = Infinity; for (let k = 0; k < 3; k++) { a = Math.min(a, await run(100)); b = Math.min(b, await run(4100)); }
    buf.destroy();
    return +Math.max(0, 1000 * (b - a) / 4000).toFixed(2);   // µs per dispatch
  } catch (e) { return null; }
}
async function benchStart(which) {
  if (!gpu) { $('bench').hidden = false; $('bench-title').textContent = 'Benchmark: cannot run'; $('bench-out').textContent = 'GPU physics is off: ' + (health.why || 'no GPU world') + '.\nThe benchmark is the GPU physics; there is nothing to measure without it.'; return; }
  BENCH.kind = which === 'ball' ? 'ball' : 'collapse';
  BENCH.dispatchUs = await dispatchCost();
  simRandom = seededRandom(20261103);
  BENCH.on = true; BENCH.phase = 'cutting the ground storey'; flick.vanished = 0; flick.jumps = 0;
  $('bench').hidden = false; $('bench-title').textContent = 'Benchmark: running';
  physicsPaused = true;
  // a fixed view from outside, looking at the tower
  // (the Quarter: from over the south-west, looking across the market hall to the mill and the office tower)
  fly = true; if (BENCH.kind === 'ball' && KIND === 6) { const q = quarterAt('office', 0, 0, 0); E.player_set(q[0] - 34, 30, q[2] - 30); yaw = Math.PI + Math.atan2(34, 30); pitch = -0.2; }
  else if (KIND === 6) { E.player_set(-22, 30, -44); yaw = Math.PI + 0.75; pitch = -0.3; } else { E.player_set(-26, 10, -38); yaw = Math.PI + 0.62; pitch = -0.12; }
  const t = performance.now();
  // (the city: every lot's ground storey)
  // (the Quarter: the ground storeys of the office tower and the mill, and the balls at the market hall)
  const cuts = KIND === 6 ? ['office', 'mill'].map(n => { const q = quarterAt(n, 0, 0, 0); return { x: q[0], z: q[2] }; }) : lots();
  if (BENCH.kind === 'ball') {
    BENCH.phase = 'the ball is set down'; BENCH.balls = []; BENCH.end = 1500;
    const q = KIND === 6 ? quarterAt('office', 0, 0, 0) : [0, 0, 0], pl = placeBall(q[0], q[2], BENCH_BALL_R);
    BENCH.ballBody = pl.body; BENCH.ballAt = pl.c; BENCH.ballTop = pl.top; BENCH.ballXZ = [q[0], q[2]];
  } else for (const L of cuts) for (let x = -10; x <= 10.1; x += 4) for (let z = -10; z <= 10.1; z += 4) jobs.blast.push({ op: 'blast', x: L.x + x, y: 1.8, z: L.z + z, r: 2.5, ctx: { d: [0, 0, 1], t0: t } });
  pump();
  await new Promise(res => { const c = () => (!busy && !jobs.blast.length) ? res() : setTimeout(c, 30); c(); });
  BENCH.cutMs = performance.now() - t;
  BENCH.phase = 'collapse';
  BENCH.t0 = performance.now(); BENCH.f0 = stepsDone; BENCH.lastRow = stepsDone; BENCH.wall0 = BENCH.t0;
  lastActive = clock();
  physicsPaused = false;
}
function benchTick(now, dt) {
  if (!BENCH.on || (BENCH.phase !== 'collapse' && BENCH.phase !== 'settling' && BENCH.phase !== 'at rest')) return;
  const f = stepsDone - BENCH.f0;
  const g = gpu.gpuMs ? Object.values(gpu.gpuMs).reduce((a, b) => a + b, 0) : 0;
  if (BENCH.phase !== 'collapse') {
    // all at rest: no body or particle awake (or a step skipped)
    const rest = (gpu.stats.awake || 0) === 0 && (gpu.stats.pActive || 0) === 0;
    if (BENCH.phase === 'settling' && rest) { BENCH.phase = 'at rest'; BENCH.settledAt = f / 60; BENCH.restF = f; }
    if (BENCH.phase === 'at rest') { BENCH.restMs.push(dt * 1000); BENCH.restGpu.push(g); }
    if ((BENCH.phase === 'at rest' && f - BENCH.restF >= BENCH.restFrames) || f >= BENCH.settleEnd) { BENCH.on = false; BENCH.phase = 'done'; simRandom = () => Math.random(); benchShow(true); }
    else if (f % 60 === 0) { $('bench-title').textContent = `Benchmark: ${BENCH.phase} (${(f / 60).toFixed(0)} s)`; }
    // (a row every 2 s while it settles, so a report copied before the end shows the way there)
    if (f - BENCH.lastRow >= 120) { BENCH.lastRow = f; BENCH.rows.push({ sim: f / 60, wall: (now - BENCH.wall0) / 1000, fps, steps: rate.steps, gpu: g, gpuStages: gpu.gpuMs ? { ...gpu.gpuMs } : null, ballY: +benchBallY().toFixed(2), bodies: gpu.n, awake: gpu.stats.awake, contacts: gpu.stats.manifolds, breaks, dust: dustBricks, impacts: impactMs, structure: failedJoints, roof: BENCH.rows.length ? BENCH.rows[BENCH.rows.length - 1].roof : 0, bad: 0, vanished: flick.vanished, jumps: flick.jumps, parts: gpu.stats.pLive || 0, pAwake: gpu.stats.pActive || 0, skipped: gpu.skipped || 0, sub: gpu.substeps }); benchShow(); }
    return;
  }
  BENCH.frameMs.push(dt * 1000);
  BENCH.gpuMax = Math.max(BENCH.gpuMax, g);
  // the steel balls, at fixed simulated times
  while (BENCH.next < BENCH.balls.length && f >= BENCH.balls[BENCH.next][0]) {
    const [, x, y, z, yw, pt] = BENCH.balls[BENCH.next++];
    const d = [Math.sin(yw) * Math.cos(pt), Math.sin(pt), Math.cos(yw) * Math.cos(pt)];
    throwBall([x, y, z], 0.4, [d[0] * 120, d[1] * 120, d[2] * 120]);
  }
  if (f - BENCH.lastRow >= 60 || f >= BENCH.end) {
    BENCH.lastRow = f;
    let roof = 0, bad = 0;
    if (snapCur) for (let k = 0; k < snapCur.pose.length / 10; k++) { const y = snapCur.pose[k * 10 + 1] + snapCur.pose[k * 10 + 8]; if (Number.isFinite(y)) roof = Math.max(roof, y); else bad++; }
    // and the standing building: a ray down through the middle
    // (the ball benchmark: the tower's own roof, beside the ball's way down, not the debris anywhere)
    if (BENCH.kind === 'ball') { roof = 0; const tb = E.raycast(BENCH.ballXZ[0] + 7, 200, BENCH.ballXZ[1] + 7, 0, -1, 0, 300); if (tb >= 0) roof = 200 - tb; }
    else { const ts = E.raycast(0.3, 200, 0.3, 0, -1, 0, 300); if (ts >= 0) roof = Math.max(roof, 200 - ts); }
    BENCH.rows.push({ sim: f / 60, wall: (now - BENCH.wall0) / 1000, fps, steps: rate.steps, gpu: g, gpuStages: gpu.gpuMs ? { ...gpu.gpuMs } : null, ballY: +benchBallY().toFixed(2), bodies: gpu.n, awake: gpu.stats.awake, contacts: gpu.stats.manifolds, breaks, dust: dustBricks, impacts: impactMs, structure: failedJoints, roof, bad, vanished: flick.vanished, jumps: flick.jumps, parts: gpu.stats.pLive || 0, pAwake: gpu.stats.pActive || 0, skipped: gpu.skipped || 0, sub: gpu.substeps });
    benchShow();
  }
  if (f >= BENCH.end) { BENCH.phase = 'settling'; benchShow(); }
}
function benchShow(done) {
  const r = BENCH.rows, last = r[r.length - 1] || {};
  const fm = BENCH.frameMs.slice().sort((a, b) => a - b), pct = (q) => fm.length ? fm[Math.min(fm.length - 1, Math.floor(q * fm.length))] : 0;
  const lines = [];
  lines.push(`Shatterforge benchmark (${BENCH.kind === 'ball' ? 'the ball on the roof' : 'the collapse'}) · ${KIND_NAMES[KIND]}, ${STOREYS} storeys · ${new Date().toISOString().slice(0, 19)}Z`);
  lines.push(`${navigator.userAgent}`);
  if (BENCH.dispatchUs != null) lines.push(`one compute dispatch: ${BENCH.dispatchUs} µs`);
  if (window.__gpuInfo) lines.push(`GPU: ${[window.__gpuInfo.vendor, window.__gpuInfo.architecture, window.__gpuInfo.device, window.__gpuInfo.description].filter(Boolean).join(' · ') || 'not told'}`);
  lines.push(`settings: physics ${setting.phys || 'custom'}, graphics ${setting.gfx || 'custom'} · debris kept ${debrisMax < Infinity ? debrisMax : 'no limit'} (${debrisCapped} removed over it) · crush depth ${setting.crush} m · crumbs ${setting.crumbs} (${setting.particles ? 'particles' : 'rigid'}) · rubble ${setting.rubble} · sub-steps ${setting.substeps} · colours ${setting.colors}`);
  if (BENCH.kind === 'ball') {
    const y0 = BENCH.ballAt ? BENCH.ballAt[1] : 0, y = last.ballY ?? y0, m = ballMass(BENCH_BALL_R);
    lines.push(`the ball on the roof: steel, ${2 * BENCH_BALL_R} m across, ${(m / 1000).toFixed(0)} t, set down at rest on the office tower's roof (${(BENCH.ballTop || 0).toFixed(2)} m)`);
    lines.push(`its centre: ${y0.toFixed(2)} m at the start, ${y.toFixed(2)} m now: ${(y0 - y).toFixed(1)} m down, ${Math.floor((y0 - y + 0.3) / 3.5)} storeys of 3.5 m`);
    lines.push(`its height each second: ${r.filter(x => Math.abs(x.sim - Math.round(x.sim)) < 0.01).map(x => (x.ballY ?? 0).toFixed(1)).join(' ')}`);
  } else lines.push(`cut: 36 blasts in ${(BENCH.cutMs / 1000).toFixed(1)} s`);
  lines.push('');
  lines.push('  sim s  wall s   fps  steps/s  GPU ms   broad  bodies*  rubble*  bodies  awake  contacts  particles  p.awake  sub  breaks   dust  roof m  bad poses  vanished  jumps');
  lines.push('  (GPU ms per step; broad = finding pairs, bodies* = the bodies\' sub-steps, rubble* = the particles)');
  for (const x of r) lines.push(`${x.sim.toFixed(1).padStart(7)} ${x.wall.toFixed(1).padStart(7)} ${x.fps.toFixed(0).padStart(5)} ${x.steps.toFixed(0).padStart(8)} ${x.gpu.toFixed(1).padStart(7)} ${(x.gpuStages ? (x.gpuStages.broad || 0) + (x.gpuStages.shapes || 0) + (x.gpuStages.grid || 0) : 0).toFixed(2).padStart(7)} ${(x.gpuStages ? x.gpuStages['sub-steps'] || 0 : 0).toFixed(2).padStart(8)} ${(x.gpuStages ? x.gpuStages.particles || 0 : 0).toFixed(2).padStart(8)} ${String(x.bodies).padStart(7)} ${String(x.awake).padStart(6)} ${String(x.contacts).padStart(9)} ${String(x.parts).padStart(10)} ${String(x.pAwake).padStart(8)} ${String(x.sub).padStart(4)} ${String(x.breaks).padStart(7)} ${String(x.dust).padStart(6)} ${x.roof.toFixed(1).padStart(7)} ${String(x.bad).padStart(10)} ${String(x.vanished).padStart(9)} ${String(x.jumps).padStart(6)}`);
  if (done) {
    const simT = last.sim || 0, wallT = last.wall || 1;
    lines.push('');
    lines.push(`real-time factor ${(simT / wallT).toFixed(2)} (simulated ${simT.toFixed(1)} s in ${wallT.toFixed(1)} s)`);
    lines.push(`frame time: median ${pct(0.5).toFixed(1)} ms · 95th ${pct(0.95).toFixed(1)} ms · worst ${pct(1).toFixed(1)} ms`);
    lines.push(`GPU step: worst ${BENCH.gpuMax.toFixed(1)} ms${last.gpuStages ? ' · last ' + Object.entries(last.gpuStages).map(([k, v]) => `${k} ${v.toFixed(1)}`).join(' · ') : ''}`);
    { const m = structStat.ms.slice().sort((a, b) => a - b), q = (p) => m.length ? m[Math.min(m.length - 1, Math.floor(p * m.length))] : 0, e = engineMsLog.slice().sort((a, b) => a - b), qe = (p) => e.length ? e[Math.min(e.length - 1, Math.floor(p * e.length))] : 0;
      lines.push(`structure: ${+setting.dynamic ? 'dynamic' : 'solved at rest'} · ${structStat.steps} steps, ${structStat.its} iterations, ${structStat.capped} at the cap · ms in the engine median ${q(0.5).toFixed(1)} / 95th ${q(0.95).toFixed(1)} / worst ${structStat.maxMs.toFixed(1)} · all engine jobs ${e.length}: median ${qe(0.5).toFixed(1)} / 95th ${qe(0.95).toFixed(1)} / worst ${(e.length ? e[e.length - 1] : 0).toFixed(1)}`); }
    lines.push(`invalid inputs refused: ${gpu.rejected || 0} · bad poses at the end: ${last.bad || 0} · flicker check: ${flick.vanished} shapes vanished unexplained, ${flick.jumps} jumped`);
    lines.push(`particles: ${partsMade} made, ${partsCrushed} crushed to dust, ${partsFar} and ${bodiesFar} pieces removed past ${farDist < Infinity ? farDist + ' m' : 'no limit'}, peak ${Math.max(0, ...r.map(x => x.parts))} · steps skipped at rest: ${last.skipped || 0}`);
    lines.push(`peak bodies ${Math.max(...r.map(x => x.bodies))} · peak contacts ${Math.max(...r.map(x => x.contacts))} · breaks ${last.breaks} · bricks to dust ${last.dust} · roof ${r.length ? r[0].roof.toFixed(1) : '?'} -> ${(last.roof || 0).toFixed(1)} m`);
    const rm = BENCH.restMs.slice().sort((a, b) => a - b), rg = BENCH.restGpu.slice().sort((a, b) => a - b), med = (a) => a.length ? a[a.length >> 1] : 0;
    lines.push(BENCH.settledAt != null ? `settled (nothing awake) at ${BENCH.settledAt.toFixed(1)} s · at rest: frame median ${med(rm).toFixed(1)} ms, GPU ${med(rg).toFixed(2)} ms` : `not settled by ${(BENCH.settleEnd / 60).toFixed(0)} s: ${gpu.stats.awake || 0} bodies and ${gpu.stats.pActive || 0} particles still awake`);
    // the same as data, for bench/ (paste it back)
    BENCH.json = { date: new Date().toISOString(), ua: navigator.userAgent, gpu: window.__gpuInfo || null, benchmark: BENCH.kind, ball: BENCH.kind === 'ball' ? { radius_m: BENCH_BALL_R, mass_kg: Math.round(ballMass(BENCH_BALL_R)), set_at_m: BENCH.ballAt, roof_m: BENCH.ballTop, end_m: last.ballY } : undefined, building: { kind: KIND_NAMES[KIND], storeys: STOREYS }, settings: { ...setting }, cut_s: +(BENCH.cutMs / 1000).toFixed(2), realtime: +(simT / wallT).toFixed(3), frame_ms: { median: +pct(0.5).toFixed(2), p95: +pct(0.95).toFixed(2), worst: +pct(1).toFixed(2) }, gpu_worst_ms: +BENCH.gpuMax.toFixed(2), dispatch_us: BENCH.dispatchUs, settled_s: BENCH.settledAt, rest: { frame_ms: +med(rm).toFixed(2), gpu_ms: +med(rg).toFixed(3) }, rows: r };
    LAB.copyText = lines.join('\n') + '\n\n' + JSON.stringify(BENCH.json);
    $('bench-title').textContent = 'Benchmark: done';
  }
  $('bench-out').textContent = lines.join('\n');
}

