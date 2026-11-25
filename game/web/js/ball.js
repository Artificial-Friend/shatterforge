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
  if (!gpu) return 0;   // (said on screen since the page started: health.js)
  // (no room: said, not swallowed. Until 24 Nov the click did nothing and nothing said why.)
  if (gpu.roomBodies() < 1 || gpu.roomShapes() < 4) { health.noRoom++; healthSay('room', `No room in the GPU world for the ball: ${gpu.roomBodies()} bodies and ${gpu.roomShapes()} shapes free. Reset the level (R R).`); return 0; }
  healthClear('room');
  const made = gpu.addBodies([{ shapes: ballShapes(c, r), mass: opt && opt.oldMass ? 0 : ballMass(r), v: v.slice(), w: [0, 0, 0] }]);
  const b = (made || [])[0] || 0, gb = b ? gpu.bodies[b - 1] : null;
  // (set down at rest, opt.rest: the benchmark's weight on a roof. Not thrown: no look-ahead, no shot)
  if (gb) { gb.ball = true; gb.ballR = r; if (!(opt && opt.rest)) { if (gpu.setShot) gpu.setShot(b); markHeavy(b, r, c, v); } }   // (a ball is not debris: capDebris; and a body thrown: GpuWorld.setShot)
  worldTouched = true; lastActive = clock();
  return b;
}
// Where a ball at rest bears, and on what. Under points of its footprint: the standing structure's top (the
// engine's own copy) and the debris' top (the last read-back). The points where its surface comes down to either
// (within 15 cm of the nearest) share its weight:
// - on the structure, or on a loose piece that lies on the structure there (within 0.6 m under the contact): the
//   weight goes to the structure at that point (`onS`: [x, y, z] on the structure);
// - on a loose piece with nothing under it there, a slab's fragment across the hole it came out of: the piece has
//   the weight to carry itself (`onP`: { idx: its body, p: the contact, slot: its shape there }; breaking.js
//   bearOnPieces: it carries it in shear round the contact, or is punched through).
// Until 24 Nov a ball's weight went to the brick under its centre, as any piece's does, and a loose piece carried
// whatever lay on it. Lab qroofball, first run: the 514 t ball came down one storey onto the roof slab, sank 0.6 m
// into a dent 1,100 failed joints wide and lay on its rim for 20 s, its weight booked to the storey below, which
// cracked and held (4.05 m down). With the weight at the rim: through the roof, and then at rest on the fragments
// of the next floor lying across their own hole (7.13 m down).
function ballBearing(body, p, r) {
  const pts = [], onS = [], onP = [];
  // (out to its edge: the ball is three eight-sided discs, and a disc's lower edge is at r under the centre to
  // 0.41 r out, then rises at 45 degrees to the centre's height at 1.41 r - d. Sampled to 0.9 r as a sphere, the
  // ball lying in a hole 0.95 r wide touched nothing: lab qroofball, at rest on a rim for 8 s with no bearing found)
  const under = (d) => r * Math.min(1, 1.414 - d);
  for (const d of [0, 0.3, 0.55, 0.75, 0.9, 1.0]) for (let k = 0; k < (d ? 12 : 1); k++) {
    const a = k * Math.PI / 6, x = p[0] + d * r * Math.cos(a), z = p[2] + d * r * Math.sin(a);
    const ts = E.raycast(x, p[1], z, 0, -1, 0, r + 1), st = ts >= 0 ? p[1] - ts : -1e9;
    const hit = rayDebris([x, p[1], z], [0, -1, 0], r + 1, body), dt = hit ? p[1] - hit.t : -1e9;
    if (Math.max(st, dt) < -1e8) continue;
    pts.push({ x, z, gap: (p[1] - under(d)) - Math.max(st, dt), st, dt, slot: hit ? hit.slot : 0 });
  }
  if (!pts.length) return { onS, onP };
  // (the nearest thing under it, up to 1.5 m off this profile: a ball at rest lies on something, and the discs'
  // corners reach 0.16 r further than the profile, 0.4 m for the 5 m ball. With a limit of 0.3 m the ball wedged
  // by its corners in the hole it had made bore on nothing (0.36 m off); with 0.6 m it went down seven storeys and
  // stopped the same way on the eighth, 0.85 m off: lab qroofball, 26.4 m down)
  const g0 = Math.min(...pts.map(q => q.gap)); if (g0 > 1.5) return { onS, onP, why: `nothing within 1.5 m: nearest ${g0.toFixed(2)} m` };
  pts.sort((x, y) => x.gap - y.gap);   // (nearest first: the contact is there)
  for (const q of pts) {
    if (q.gap > g0 + 0.15) continue;
    if (q.dt <= q.st + 0.05 || q.st > q.dt - 0.6) { if (q.st > -1e8) onS.push([q.x, q.st, q.z]); continue; }
    const sh = gpu.shapes[q.slot - 1], idx = sh ? sh.body : 0;
    if (bodyInfo.has(idx)) onP.push({ idx, p: [q.x, q.dt, q.z], slot: q.slot });
  }
  return { onS, onP, why: onS.length + onP.length ? '' : `on ${pts.filter(q => q.gap <= g0 + 0.15).length} points of debris that is no piece (crumbs), the structure ${(pts[0].dt - pts[0].st).toFixed(1)} m under it` };
}
// A steel ball set down at rest with its lowest point 2 cm over whatever is under (x, z): on a roof, as the
// benchmark `ball` and the lab's `qroofball` do. Returns the body and where its centre is.
function placeBall(x, z, r) {
  const ts = E.raycast(x, 400, z, 0, -1, 0, 500), top = ts >= 0 ? 400 - ts : 0, c = [x, top + r + 0.02, z];
  return { body: throwBall(c, r, [0, 0, 0], { rest: true }), c, top };
}
// Heavy balls at rest punch what cannot carry them four times a second (breaking.js bearOnPieces): a floor's
// fragments go as they would, at once, not a second later with the next rest loads. Balls over 10 t only: the
// game's ball, 514 kg, bears 5 kN, which everything carries.
let ballWeightNext = 0;
function ballWeights() {
  if (window.__dbg && !(window.__bw = (window.__bw || 0) + 1 & 63)) console.warn(`ballWeights: gpu ${!!gpu} bodies ${!!lastBodies} vel ${!!lastVel} clock ${clock().toFixed(0)} next ${ballWeightNext.toFixed(0)}`);
  if (!gpu || !lastBodies || !lastVel || clock() < ballWeightNext) return;
  ballWeightNext = clock() + 250;
  for (let i = 2; i <= gpu.n; i++) {
    const b = gpu.bodies[i - 1], o = (i - 1) * 7, vo = (i - 1) * 3;
    if (!b || b.dead || !b.ball || !b.ballR || !(b.mass > 10000) || o + 2 >= lastBodies.length) continue;
    if (Math.hypot(lastVel[vo], lastVel[vo + 1], lastVel[vo + 2]) > 0.5) continue;
    const pos = [lastBodies[o], lastBodies[o + 1], lastBodies[o + 2]]; if (!(pos[1] - b.ballR > 1)) continue;
    const bear = ballBearing(i, pos, b.ballR), n = bear.onS.length + bear.onP.length;
    if (window.__dbg) console.warn(`ballWeights: ball ${i} y ${pos[1].toFixed(2)} onS ${bear.onS.length} onP ${bear.onP.length} ${bear.why || ''}`);
    if (bear.onP.length) { bearOnPieces(bear.onP.map(q => ({ ...q, f: b.mass * 9.81 / n, rBall: b.ballR }))); pump(); }
  }
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

