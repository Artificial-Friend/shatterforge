// ---- loop
let last = performance.now(), acc = 0, fps = 60, frames = 0, fpsT = last;
function frame(now) {
  requestAnimationFrame(frame);
  if (LAB.on) { last = now; return; }   // the lab drives the frames itself
  const fj = performance.now();
  perfAdd('interval', now - last);
  const dt = Math.min(0.05, (now - last) / 1000); last = now;
  try { frameBody(now, dt); } finally { perfAdd('frame js', performance.now() - fj); }
}
function frameBody(now, dt) {
  if (E) {
    const f = new THREE.Vector3(-Math.sin(yaw), 0, -Math.cos(yaw)), r = new THREE.Vector3(Math.cos(yaw), 0, -Math.sin(yaw));
    const run = keys.has('ShiftLeft') || keys.has('ShiftRight');
    const sp = fly ? (run ? 24 : 8) : (run ? 4.5 : 1.6);
    const w = new THREE.Vector3();
    if (keys.has('KeyW')) w.add(f); if (keys.has('KeyS')) w.sub(f); if (keys.has('KeyD')) w.add(r); if (keys.has('KeyA')) w.sub(r);
    if (w.lengthSq() > 0) w.normalize().multiplyScalar(sp);
    if (fly) { if (keys.has('KeyE')) w.y += sp; if (keys.has('KeyQ')) w.y -= sp; }
    playerDebris();
    E.player_step(dt, w.x, w.y, w.z, keys.has('Space') ? 1 : 0, fly ? 1 : 0);
    cam.position.set(E.player_get(0), E.player_get(1) + 1.65, E.player_get(2));
    // (the city is wider than the sun's shadow: the shadow follows the player)
    if (KIND === 5) { const px = E.player_get(0), pz = E.player_get(2); sun.target.position.set(px, 0, pz); sun.position.set(px - 30, 60, pz - 22); }
    cam.rotation.set(pitch, yaw, 0);
    pump();
    // physics runs while anything is awake (and briefly after a blast, before the first stats arrive)
    if (gpu && diag.physics && !physicsPaused && (gpu.stats.awake > 0 || gpu.stats.pActive > 0 || now - lastActive < 2000)) {
      acc += dt; physFrames++;
      // never queue more GPU work than it finishes: a backlog makes every read-back late
      const ts = performance.now();
      // pace the steps by the GPU's own measured step time (waiting on the browser's completion callbacks
      // stalled everything: they arrive late while nothing new is submitted)
      const stepMs = gpu.gpuMs ? Object.values(gpu.gpuMs).reduce((a, b) => a + b, 0) : 4;
      const maxSteps = stepMs * 2 < 12 ? 2 : 1;
      // and never run ahead of the read-backs: at most a few steps queued past the newest one that came back,
      // or work piles up in the GPU queue and everything (debris, breaking) shows seconds late
      // a GPU that cannot do 60 steps a second (a laptop's, in a collapse) steps 1/60 s per frame anyway, and
      // everything falls in slow motion (gravity itself is 9.81: GPU tests box, bounce). Behind, with the
      // sub-steps already down to the adaptive floor, a step covers 1/30 s with twice the sub-steps: the same
      // sub-step length, so the same accuracy (GPU tests with DT2 and 16 sub-steps match 8 at 1/60: stack drift
      // 15.5 mm both), and half the per-step work (broad phase, pairs, colouring: about a fifth of a step)
      let n = 0; while (acc >= 1 / 60 && n < maxSteps && (!diag.readback || stepsDone - stepsAtRead < 3 || (n === 0 && gpu.placing()))) {
        const h = !LAB.on && setting.realtime && acc >= 2 / 60 && n === maxSteps - 1 && gpu.substeps <= Math.min(8, setting.substeps || 12) ? 2 / 60 : 1 / 60;
        if (h > 1 / 60) { const s0 = gpu.substeps; gpu.substeps = 2 * s0; gpu.step(h); gpu.substeps = s0; } else gpu.step(h);
        acc -= h; n++; stepsDone++; simTime += h; lastStepDt = h;
      }
      perfAdd('step', performance.now() - ts);
      if (acc > 0.1) acc = 0;
      pullDebris();
      timed('impacts', impacts);
    }
  }
  benchTick(now, dt);
  if (diag.draw) timed('draw', () => drawDebris(now));
  if (!LAB.on || LAB.render) timed('render', () => renderer.render(scene, cam));
  frames++; frames0++; if (now - fpsT > 500) { fps = frames * 1000 / (now - fpsT); frames = 0; fpsT = now; hud(); }
}
const rate = { steps: 0, reads: 0, t: performance.now(), s0: 0, r0: 0 };
// Adaptive sub-steps: while physics falls behind real time (a heavy collapse), fewer sub-steps per step, down
// to 8 (a 20-box tower still stands there; at 6 it does not); back up to the setting when the GPU has room
let physFrames = 0, frames0 = 0, simTime = 0, lastStepDt = 1 / 60;
function physicsBusy() { return gpu && ((gpu.stats.awake || 0) > 0 || (gpu.stats.pActive || 0) > 0); }
function adaptSubsteps(busy) {
  if (!gpu || !setting.adaptive || !busy) return;
  const target = setting.substeps || 12, lo = Math.min(8, target);
  const stepMs = gpu.gpuMs ? Object.values(gpu.gpuMs).reduce((a, b) => a + b, 0) : 0;
  if (rate.steps < 54 && gpu.substeps > lo) gpu.substeps--;
  else if (rate.steps >= 58 && gpu.substeps < target && (!stepMs || stepMs * (gpu.substeps + 1) / gpu.substeps < 11)) gpu.substeps++;
}
function hud() {
  if (!E) return;
  const tn = performance.now(), dt = (tn - rate.t) / 1000;
  if (dt > 0.4) {
    rate.steps = (stepsDone - rate.s0) / dt; rate.reads = (readsDone - rate.r0) / dt; rate.sim = (simTime - (rate.sim0 || 0)) / dt; rate.sim0 = simTime; rate.t = tn; rate.s0 = stepsDone; rate.r0 = readsDone;
    adaptSubsteps(physFrames >= frames0 * 0.9 && physFrames > 0); physFrames = 0; frames0 = 0;
  }
  const y = E.player_get(1), storey = Math.max(0, Math.floor((y + 0.01) / 3.75));
  const g = gpu && gpu.gpuMs ? Object.values(gpu.gpuMs).reduce((a, b) => a + b, 0) : null;
  $('hud').innerHTML = [
    `<b>${fps.toFixed(0)} fps</b> · ${fly ? 'flying' : 'walking'} · storey <b>${storey === 0 ? 'G' : storey}</b>`,
    `<span class="k">bricks</span> ${E.bricks_alive().toLocaleString()} · <span class="k">joints</span> ${E.joints_live().toLocaleString()} · ${E.bytes_per_brick().toFixed(0)} B/brick`,
    gpu ? `<span class="k">debris</span> ${(gpu.n - 1).toLocaleString()} bodies, ${debrisShapes.toLocaleString()} pieces · ${gpu.stats.awake} awake · ${(gpu.stats.pLive || 0).toLocaleString()} particles (${gpu.stats.pActive || 0} moving)${snapCur && snapCur.pose.length / 10 > MAXD ? ` · <b>${(snapCur.pose.length / 10 - MAXD).toLocaleString()} not drawn</b>` : ''}` : `<span class="k">debris physics</span> off (no WebGPU)`,
    gpu ? `<span class="k">contacts</span> ${gpu.stats.manifolds.toLocaleString()} · <span class="k">GPU</span> ${g == null ? '—' : g.toFixed(2) + ' ms'} · ${gpu.substeps} sub-steps${gpu.skipped ? ` · ${gpu.skipped.toLocaleString()} steps skipped at rest` : ''}` : '',
    gpu ? `<span class="k">physics</span> ${rate.steps.toFixed(0)} steps/s${rate.sim && physicsBusy() ? ` (${rate.sim < 0.95 ? '<b>' : ''}time ×${rate.sim.toFixed(2)}${rate.sim < 0.95 ? '</b>' : ''})` : ''} · ${rate.reads.toFixed(0)} read-backs/s · <span class="k">flicker check</span> ${flick.vanished} vanished, ${flick.jumps} jumps · <span class="k">long tasks</span> ${longTasks.n} (${longTasks.ms.toFixed(0)} ms total${longTasks.last ? ', last ' + longTasks.last : ''})` : '',
    `<span class="k">main thread ms</span> ${Object.entries(perf).map(([k, v]) => `${k} ${v.toFixed(1)}`).join(' · ')}${gpu && gpu.gpuMs ? ` · <span class="k">GPU</span> ${Object.entries(gpu.gpuMs).map(([k, v]) => `${k} ${v.toFixed(1)}`).join(' · ')}` : ''}`,
    `<span class="k">structure</span> ${(structRatio * 100).toFixed(0)}% of strength at the worst joint · ${+setting.view ? `engineer's view, displacements × ${VIEW_SCALE[setting.viewScale] || 0} · ` : ''}${+setting.dynamic ? `sag ${(structSag * 1000).toFixed(1)} mm${structMoving ? `, moving ${(structSpeed * 1000).toFixed(0)} mm/s` : ''}` : 'solved at rest'} · ${failedJoints} failed · ${crackedJoints} cracked and holding · rubble on its floors ${(restOnFloors / 9810).toFixed(0)} t · ${breaks} pieces broke on impact · ${crushedBodies} crushed · ${dustBricks} bricks to dust${spillLost ? ` · ${(spillLost * TICK ** 3).toFixed(1)} m³ of rubble lost (particles full)` : ''} · impacts ${impactMs.toFixed(0)} ms · engine ${worker ? 'in a worker' : 'inline'}${dropped ? ` · ${dropped} bodies over capacity` : ''}${gpu && gpu.rejected ? ` · ${gpu.rejected} invalid inputs refused` : ''}`,
    lastBlast ? `<span class="k">last</span> ${lastBlast}` : `<span class="k">click to ${radius === 'ball' ? 'fire the steel ball' : `blast · radius ${radius} m`}</span>`,
  ].filter(Boolean).join('<br>');
}

