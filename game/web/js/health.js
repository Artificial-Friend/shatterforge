// ---- what is wrong, said where the player looks
// Until 24 Nov the page had one place for trouble: a line on the start card, gone with the card. A browser whose
// shader compiler refused the GPU world (Firefox: engine/gpu/test_wgsl.cjs), one without WebGPU, a GPU device lost
// in play, a world with no room left: each left a game where holes still open, nothing that comes off is there,
// and the ball does not fly, with nothing on screen to say why. Now each is a line in a banner that stays, and
// key I (or the menu) gives the whole state as text to copy.
const health = { issues: new Map(), why: '', gpuErrors: 0, gpuError: '', lost: '', selfTest: null, noRoom: 0, stalls: 0, createMs: 0, stallSince: 0, lastReads: -1, lastSteps: -1 };
const healthEsc = (s) => String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
function healthShow() {
  const el = $('alert'); if (!el) return;
  el.hidden = !health.issues.size;
  el.innerHTML = [...health.issues.values()].map(t => `<div>${healthEsc(t)}</div>`).join('') + '<div class="k">I · the full state, to copy</div>';
}
function healthSay(key, text) { if (health.issues.get(key) === text) return; health.issues.set(key, text); healthShow(); }
function healthClear(key) { if (health.issues.delete(key)) healthShow(); }
const NO_GPU = 'Holes still open, but nothing that breaks off is simulated and the steel ball cannot be thrown.';
function healthNoGpu(why) { health.why = why; healthSay('gpu', `GPU physics is off: ${String(why).slice(0, 300)}. ${NO_GPU}`); }
// the device's own complaints: validation errors nobody caught, and the device going away (a driver reset, the
// GPU process restarted, the adapter taken by another page)
function healthWatch(d) {
  d.addEventListener('uncapturederror', (e) => {
    health.gpuErrors++; if (!health.gpuError) health.gpuError = String(e.error && e.error.message || e.error).slice(0, 600);
    healthSay('gpuerr', `The GPU reported ${health.gpuErrors} error${health.gpuErrors > 1 ? 's' : ''}; the first: ${health.gpuError.slice(0, 160)}`);
  });
  // read-backs asked of a lost device are refused, each one an unhandled rejection: they are the loss already
  // said above, counted in the state, and not page errors (play_check.cjs lost: 2 page errors before this)
  addEventListener('unhandledrejection', (e) => {
    if (/device (was|is) lost|device was destroyed/i.test(String(e.reason && e.reason.message || e.reason))) { health.lostReads = (health.lostReads || 0) + 1; e.preventDefault(); }
  });
  d.lost.then((info) => {
    health.lost = `${info.reason || 'unknown'}: ${info.message || ''}`;
    healthSay('lost', `The GPU device was lost (${health.lost.slice(0, 120)}): physics has stopped. Reset the level (R R) or reload the page.`);
  });
}
// The GPU world checked on this device before the game starts, in the world itself: a 0.4 m concrete box and a
// steel ball (the game's: three hulls) dropped 0.5 m onto a fixed slab 150 m up, 50 steps. Both must have fallen and
// lie on it. It passes on the software GPU (page checks); a device where the kernels compile and do not simulate
// fails it here, by numbers, instead of in play, by nothing happening.
async function gpuSelfTest() {
  const t0 = performance.now(), Y = 150, top = Y + 0.1;
  try {
    const made = gpu.addBodies([
      { fixed: true, shapes: [{ p: [0, Y, 0], q: [0, 0, 0, 1], h: [2, 0.1, 2], mat: 0 }], v: [0, 0, 0], w: [0, 0, 0] },
      { shapes: [{ p: [-0.8, top + 0.7, 0], q: [0, 0, 0, 1], h: [0.2, 0.2, 0.2], mat: 0 }], v: [0, 0, 0], w: [0, 0, 0] },
      { shapes: ballShapes([0.8, top + 0.75, 0], 0.25), mass: ballMass(0.25), v: [0, 0, 0], w: [0, 0, 0] }]);
    const [slab, box, ball] = made || [];
    if (!(slab > 0 && box > 0 && ball > 0)) throw new Error('the world took none of the three test bodies');
    const run = (async () => {
      for (let k = 0; k < 50; k++) { gpu.step(1 / 60); if (k % 10 === 9) await dev.queue.onSubmittedWorkDone(); }
      await dev.queue.onSubmittedWorkDone();
      return gpu.readPose(debrisFrom);
    })();
    const r = await Promise.race([run, new Promise((_, rej) => setTimeout(() => rej(new Error('no answer from the GPU in 90 s')), 90000))]);
    const y = (b) => r.body[(b - 1) * 7 + 1], sp = (b) => Math.hypot(r.vel[(b - 1) * 3], r.vel[(b - 1) * 3 + 1], r.vel[(b - 1) * 3 + 2]);
    const res = { box_y: +(y(box) - top).toFixed(3), ball_y: +(y(ball) - top).toFixed(3), box_speed: +sp(box).toFixed(3), ball_speed: +sp(ball).toFixed(3), ms: Math.round(performance.now() - t0) };
    // (the box rests with its centre 0.2 m over the slab; the ball on a flat, 0.25 m, or an edge, up to 0.29 m)
    res.ok = Math.abs(res.box_y - 0.2) < 0.03 && res.ball_y > 0.22 && res.ball_y < 0.32 && res.box_speed < 0.3 && res.ball_speed < 0.3;
    health.selfTest = res;
    for (const b of [ball, box, slab]) gpu.killBody(b);
    if (!res.ok) healthSay('selftest', `GPU physics failed its check on this device: a box dropped 0.5 m is ${res.box_y} m over the slab (0.2 expected) at ${res.box_speed} m/s, the ball ${res.ball_y} m (0.25 to 0.29) at ${res.ball_speed} m/s. What breaks off may not behave.`);
  } catch (e) {
    health.selfTest = { ok: false, error: e.message, ms: Math.round(performance.now() - t0) };
    healthSay('selftest', `GPU physics failed its check on this device: ${String(e.message).slice(0, 200)}. What breaks off may not be simulated.`);
  }
}
// Steps going out and nothing coming back: twice a second (from the HUD). The page steps at most 3 past the newest
// read-back, so a GPU that stops answering stops the world without a word.
function healthTick() {
  // debris removed because more exists than the setting keeps: said for 6 s each time it happens (a limit of 250
  // keeps 250 crumbs of a break that makes thousands: "what I break disappears", and nothing on screen said why)
  if (!LAB.on) {
    if (debrisCapped > (health.capSeen || 0)) { health.capSeen = debrisCapped; health.capUntil = performance.now() + 6000; healthSay('budget', `${debrisCapped} pieces of debris removed so far: more than the ${debrisMax} kept at once. Settings, Physics, "Debris kept at once".`); }
    else if (health.issues.has('budget') && performance.now() > health.capUntil) healthClear('budget');
  }
  if (!gpu || physicsPaused || LAB.on) { health.stallSince = 0; return; }
  const now = performance.now(), wants = (gpu.stats.awake || 0) > 0 || (gpu.stats.pActive || 0) > 0 || clock() - lastActive < 2000;
  if (!wants || readsDone !== health.lastReads) { health.lastReads = readsDone; health.stallSince = now; healthClear('stall'); return; }
  if (!health.stallSince) health.stallSince = now;
  if (now - health.stallSince > 3000) { if (!health.issues.has('stall')) health.stalls++; healthSay('stall', `Physics is not answering: no read-back from the GPU for ${((now - health.stallSince) / 1000).toFixed(0)} s (${stepsDone} steps sent, ${readsDone} read back). What breaks off will not show until it does.`); }
}
function diagnosis() {
  const lim = dev ? dev.limits : null, L = (k) => (lim ? lim[k] : null);
  const live = gpu ? gpu.bodies.reduce((n, b) => n + (b && !b.dead ? 1 : 0), 0) : 0;
  return {
    date: new Date().toISOString(), ua: navigator.userAgent, secure: !!window.isSecureContext, webgpu: !!navigator.gpu, gpu: window.__gpuInfo || null,
    limits: lim ? { maxStorageBufferBindingSize: L('maxStorageBufferBindingSize'), maxBufferSize: L('maxBufferSize'), maxStorageBuffersPerShaderStage: L('maxStorageBuffersPerShaderStage'), maxComputeWorkgroupsPerDimension: L('maxComputeWorkgroupsPerDimension'), maxComputeInvocationsPerWorkgroup: L('maxComputeInvocationsPerWorkgroup') } : null,
    world: gpu ? { made_ms: health.createMs, bodies: gpu.n, live, room_bodies: gpu.roomBodies(), room_shapes: gpu.roomShapes(), frame: gpu.frame, skipped: gpu.skipped || 0, substeps: gpu.substeps, stats: gpu.stats, step_ms: gpu.gpuMs || null, dispatch: gpu.dispatchCal || null, lost_steps: gpu.lost || 0, rejected: gpu.rejected || 0 } : null,
    off_because: health.why || null, self_test: health.selfTest, issues: [...health.issues.values()], gpu_errors: health.gpuErrors, first_gpu_error: health.gpuError || null, device_lost: health.lost || null, reads_refused_after_loss: health.lostReads || 0, stalls: health.stalls, no_room: health.noRoom,
    engine: { in_worker: !!worker, bricks: E ? E.bricks_alive() : null, busy: busy ? busy.op : null, queued: jobs.blast.length, last_ms: workerMs },
    page: { level: KIND_NAMES[KIND], started, paused: physicsPaused, steps: stepsDone, steps_at_read: stepsAtRead, reads: readsDone, reading, fps: +fps.toFixed(1), steps_per_s: +(rate.steps || 0).toFixed(1), reads_per_s: +(rate.reads || 0).toFixed(1), drawn_shapes: debrisShapes, chunks_waiting: pendingChunks.length },
    debris: { kept_at_most: debrisMax < Infinity ? debrisMax : 'no limit', live: debrisLive, removed_over_budget: debrisCapped, particles_made: partsMade, particles_crushed: partsCrushed, particles_removed_far: partsFar, bodies_removed_far: bodiesFar, clean_up_m: farDist < Infinity ? farDist : 'off', dust_bricks: dustBricks, spill_lost: spillLost, vanished: flick.vanished, jumps: flick.jumps },
    balls: { thrown: balls, look_ahead: { sent: punchStat.sent, through: punchStat.through, stopped: punchStat.stopped, none: punchStat.none, off: punchStat.off } },
    settings: { ...setting }, last: lastBlast,
  };
}
function showDiagnosis() {
  const d = diagnosis(), text = 'Shatterforge state\n' + JSON.stringify(d, null, 1);
  if (document.exitPointerLock && document.pointerLockElement) document.exitPointerLock();
  $('bench-title').textContent = 'State of the page (copy it into a report)'; $('bench-out').textContent = text; LAB.copyText = text; $('bench').hidden = false;
  return d;
}
