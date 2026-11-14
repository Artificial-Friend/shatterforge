// ---- the steel ball: a heavy round body fired from the camera; its hits crush and load the walls
function ballShapes(c, r) {
  // three crossed boxes and a core: round enough to roll and hit like a ball
  const a = r, b = r * 0.62;
  return [[a, b, b], [b, a, b], [b, b, a], [r * 0.82, r * 0.82, r * 0.82]].map(h => ({ p: c.slice(), q: [0, 0, 0, 1], h, mat: 2, col: 2 }));
}
function fireBall() {
  if (!gpu) return;
  const d = new THREE.Vector3(); cam.getWorldDirection(d);
  const r = setting.bsize / 2, v = setting.bspeed;
  const c = [cam.position.x + d.x * (r + 0.6), cam.position.y + d.y * (r + 0.6), cam.position.z + d.z * (r + 0.6)];
  if (gpu.roomBodies() < 1 || gpu.roomShapes() < 4) return;
  const made = gpu.addBodies([{ shapes: ballShapes(c, r), v: [d.x * v, d.y * v, d.z * v], w: [0, 0, 0] }]);
  for (const b of made || []) { const gb = gpu.bodies[b - 1]; if (gb) gb.ball = true; }   // (a ball is not debris: capDebris)
  lastActive = clock();
  balls++;
  lastBlast = `steel ball ${(2 * r).toFixed(2)} m at ${v} m/s`;
}
let balls = 0;
setRadius(0.7);

