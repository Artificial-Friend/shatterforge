// ---- input
cv.addEventListener('click', () => {
  if (panelOpen()) { resume(); return; }
  if (!locked) { const r = cv.requestPointerLock && cv.requestPointerLock(); if (r && r.catch) r.catch(() => {}); return; }
  if (radius === 'ball') fireBall(); else doBlast();
});
// (Esc while looking around never reaches the page as a key: the browser takes it to free the pointer. So the
// pointer coming free while the game is on, with no panel asked for, is Esc: the menu opens.)
let started = false, menuAt = -1e9, resetArm = -1e9;
document.addEventListener('pointerlockchange', () => {
  const was = locked; locked = document.pointerLockElement === cv;
  if (was && !locked && started && !panelOpen() && !LAB.on && $('bench').hidden) { showPanel('menu'); menuAt = performance.now(); }
});
let drag = null;
cv.addEventListener('mousedown', (e) => { if (!locked) drag = [e.clientX, e.clientY]; });
addEventListener('mouseup', () => { drag = null; });
addEventListener('mousemove', (e) => {
  let dx = 0, dy = 0;
  if (locked) { dx = e.movementX; dy = e.movementY; }
  else if (drag) { dx = e.clientX - drag[0]; dy = e.clientY - drag[1]; drag = [e.clientX, e.clientY]; }
  yaw -= dx * 0.0022; pitch = Math.max(-1.5, Math.min(1.5, pitch - dy * 0.0022));
});
addEventListener('keydown', (e) => {
  keys.add(e.code);
  if (e.code === 'KeyF') fly = !fly;
  if (e.code === 'KeyV') { const el = $('s-view'); el.value = +setting.view ? 0 : 1; el.dispatchEvent(new Event('input')); }
  if (e.code === 'Digit1') setRadius(0.4); if (e.code === 'Digit2') setRadius(0.7); if (e.code === 'Digit3') setRadius(1.2); if (e.code === 'Digit4') setRadius('ball');
  if (e.code === 'KeyB' && !locked) { if (radius === 'ball') fireBall(); else doBlast(); }
  if (e.code === 'KeyK') { try { localStorage.setItem('sf-bench', '1'); } catch (err) {} location.reload(); }
  // (a browser that also sends the key that freed the pointer: the menu it just opened stays open)
  if (e.code === 'Escape' && started && !e.repeat && performance.now() - menuAt > 300) escapeKey();
  // (R twice within 1.5 s: it sits between E and F, and one slip must not cost the level)
  if (e.code === 'KeyR' && started && !e.repeat && !panelOpen() && !LAB.on) {
    const t = performance.now(), el = $('t-reset');
    if (t - resetArm < 1500) resetLevel();
    else { resetArm = t; el.textContent = 'R again · reset'; el.classList.add('on'); setTimeout(() => { el.textContent = 'R R · reset'; el.classList.remove('on'); }, 1500); }
  }
});
addEventListener('keyup', (e) => keys.delete(e.code));
function setRadius(r) { radius = r; document.querySelectorAll('#tool span[data-r]').forEach(s => s.classList.toggle('on', s.dataset.r === String(r))); }

