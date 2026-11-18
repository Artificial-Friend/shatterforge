// ---- the steel ball: a heavy round body fired from the camera; its hits crush and load the walls
function ballShapes(c, r) {
  // Three eight-sided discs at right angles to each other (r to their flats, 0.8 r thick): round in every plane it
  // can roll in, its corners within 1.16 r. (Until 15 Nov four crossed boxes: thrown level onto the ground at 40 m/s
  // it caught a corner and went up at 10 m/s, 5 m high, where a ball skids: lab qballtemple.) A convex piece has at
  // most 16 corners, and an eight-sided disc has 16.
  const k = 1 / Math.cos(Math.PI / 8), L = 0.4 * r, corners = [], top = [], bottom = [], sides = [];
  for (let i = 0; i < 8; i++) { const a = (2 * i + 1) * Math.PI / 8; corners.push([r * k * Math.cos(a), r * k * Math.sin(a), L], [r * k * Math.cos(a), r * k * Math.sin(a), -L]); }
  for (let i = 0; i < 8; i++) { top.push(2 * i); bottom.unshift(2 * i + 1); const j = (i + 1) % 8; sides.push([2 * i, 2 * i + 1, 2 * j + 1, 2 * j]); }
  const h = Math.SQRT1_2, out = [];
  // (the disc's axis along z, then turned onto x and onto y)
  for (const q of [[0, 0, 0, 1], [0, h, 0, h], [-h, 0, 0, h]]) { const sh = GpuWorld.pieceShape(corners, [top, bottom, ...sides], c, q, 2, 2); if (sh) out.push(sh); }
  if (out.length === 3) return out;
  // (no room for its hull: four crossed boxes, as before)
  const a = r, b = r * 0.5;
  return [[a, b, b], [b, a, b], [b, b, a], [r * 0.7, r * 0.7, r * 0.7]].map(h => ({ p: c.slice(), q: [0, 0, 0, 1], h, mat: 2, col: 2 }));
}
// a steel sphere's mass. The ball's shapes overlap, and the GPU world adds their masses: until 15 Nov a 0.5 m ball
// (then four crossed boxes) weighed 1,668 kg (13.6 r^3 of steel) where a sphere weighs 514 kg (4.19 r^3): 3.2 times
// too heavy.
const ballMass = (r) => 7850 * 4 / 3 * Math.PI * r * r * r;
// A ball of radius r thrown from c at velocity v: the body, its true mass, and its look ahead (markHeavy). Returns
// the body (1-based), or 0 with no room.
function throwBall(c, r, v, opt) {
  if (!gpu || gpu.roomBodies() < 1 || gpu.roomShapes() < 4) return 0;
  const made = gpu.addBodies([{ shapes: ballShapes(c, r), mass: opt && opt.oldMass ? 0 : ballMass(r), v: v.slice(), w: [0, 0, 0] }]);
  const b = (made || [])[0] || 0, gb = b ? gpu.bodies[b - 1] : null;
  if (gb) { gb.ball = true; if (gpu.setShot) gpu.setShot(b); markHeavy(b, r, c, v); }   // (a ball is not debris: capDebris; and a body thrown: GpuWorld.setShot)
  lastActive = clock();
  return b;
}
function fireBall() {
  if (!gpu) return;
  const d = new THREE.Vector3(); cam.getWorldDirection(d);
  const r = setting.bsize / 2, v = setting.bspeed;
  const c = [cam.position.x + d.x * (r + 0.6), cam.position.y + d.y * (r + 0.6), cam.position.z + d.z * (r + 0.6)];
  if (!throwBall(c, r, [d.x * v, d.y * v, d.z * v])) return;
  balls++;
  lastBlast = `steel ball ${(2 * r).toFixed(2)} m at ${v} m/s`;
}
let balls = 0;
setRadius(0.7);

