// ---- input
cv.addEventListener('click', () => {
  if (!locked) { const r = cv.requestPointerLock && cv.requestPointerLock(); if (r && r.catch) r.catch(() => {}); return; }
  if (radius === 'ball') fireBall(); else doBlast();
});
document.addEventListener('pointerlockchange', () => { locked = document.pointerLockElement === cv; });
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
  if (e.code === 'KeyO' || (e.code === 'Escape' && !$('settings').hidden)) toggleSettings(e.code === 'KeyO' ? undefined : false);
});
addEventListener('keyup', (e) => keys.delete(e.code));
function setRadius(r) { radius = r; document.querySelectorAll('#tool span').forEach(s => s.classList.toggle('on', s.dataset.r === String(r))); }

