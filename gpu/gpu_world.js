// Shatterforge GPU rigid bodies on WebGPU. The reference model is engine/src/phys.rs (boxes, speculative
// contacts, soft step with anchored patch friction); this adds compound bodies, materials and restitution.
//
// A body is rigid and made of one or more box shapes (a wall chunk is one body of many bricks). Collision
// works on shapes, the solver on bodies. One step, all on the GPU:
//   broad     body prep (inertia), shape poses + AABBs, multi-level hashed grid (level per shape size),
//             candidate shape pairs (a small shape queries its own and coarser levels)
//   narrow    one thread per shape pair: box-box SAT, clipping, 4-point reduction, warm-start match, masses
//   colour    greedy graph colouring of contacts with atomic per-body masks; per-body contact lists
//   sub-steps integrate + warm start (per body), solve per colour, integrate positions, relax per colour;
//             then restitution per colour and the per-contact impulse totals for damage
//   sleep     islands by GPU union-find; an island at rest for 0.5 s sleeps; moving bodies wake what they hit
'use strict';

const MATERIALS = [
  // density kg/m^3, friction, restitution
  { name: 'concrete', density: 2400, mu: 0.6, e: 0.1 },
  { name: 'masonry', density: 1900, mu: 0.6, e: 0.05 },
  { name: 'steel', density: 7850, mu: 0.45, e: 0.25 },
  { name: 'glass', density: 2500, mu: 0.5, e: 0.2 },
  { name: 'timber', density: 500, mu: 0.5, e: 0.3 },
  { name: 'test-bouncy', density: 1100, mu: 0.6, e: 0.5 },
  { name: 'ground', density: 0, mu: 0.6, e: 0.0 },
  { name: 'reference', density: 1900, mu: 0.6, e: 0.0 },   // the CPU reference model: one friction, no bounce
];
const MAT = Object.fromEntries(MATERIALS.map((m, i) => [m.name, i]));
const GROUND_MAT = MAT.ground;
const CNEAR = 8192;   // (slots of the coarse awake map, see near_mark in the shader)

const WGSL = /* wgsl */`
struct Params {
  gravity: vec3f, h: f32,
  inv_h: f32, bias_rate: f32, mass_scale: f32, imp_scale: f32,
  max_push: f32, rest_threshold: f32, spec: f32, dt: f32,
  cell: f32, n1: u32, grid_mask: u32, max_man: u32,
  tbl_mask: u32, color_limit: u32, wmax: f32, cur_base: u32,
  old_base: u32, o_gcount: u32, o_gstart: u32, o_gfill: u32,
  o_sorted: u32, o_cell: u32, o_mask: u32, o_ccount: u32,
  o_cstart: u32, o_cfill: u32, o_order: u32, o_mcolor: u32,
  o_tnew: u32, o_told: u32, anchor: u32, max_pairs: u32,
  o_pairs: u32, o_bcount: u32, o_bstart: u32, o_bfill: u32,
  o_blist: u32, o_par: u32, o_mint: u32, allow_sleep: u32,
  sleep_time: f32, sleep_lin: f32, sleep_ang: f32, ns1: u32,
  levels: u32, ground_mat: u32, sleep_dist: f32, sleep_rot: f32,
  o_ocount: u32, o_acc: u32, o_events: u32, max_events: u32,
  ev_speed: f32, ev_energy: f32, o_cawake: u32, o_cnear: u32,
  n_inherit: u32, o_inherit: u32, wake_speed: f32, pad7: u32,
  mat: array<vec4f, 8>,   // friction, restitution, -, -
  np1: u32, pcell: f32, pg_mask: u32, p_wake: u32,
  p_h: f32, n_blast: u32, p_cap: u32, pad9: u32,
  blast: array<vec4f, 6>,   // particle blasts: (centre, reach), (speed, -, -, -)
  o_ptab: u32, ptab_mask: u32, o_big: u32, o_mslot: u32,   // contact reduction (see red_count)
  o_crushed: u32, o_crush: u32, crush_ratio: f32, crush_dv: f32,   // crushing (see crush)
  o_hull: u32, o_hl: u32, dbg: u32, ovf_push: f32,   // the convex chunk library, in the shape buffer
  o_plan: u32, o_spart: u32, o_snap: u32, o_claim: u32,   // breaking on the GPU (see snap_split); colour claims
  o_mtry: u32, o_claim2: u32, o_sorted2: u32, o_evhist: u32,
  o_cchk: u32, o_pinch: u32, step: u32, o_level: u32,
  o_level2: u32, o_psup: u32, psup_n: u32, o_released: u32,   // (see color_claim, sort_cells); pinching (see pinch)
}
struct ColorU { c: u32 }

// body record (5 vec4): centre of mass + inverse mass, rotation (principal frame), local inverse inertia +
// sleep timer, (asleep, pose when the sleep timer restarted: position), (that rotation). Body 0 is the ground.
const BS = 5u;
const F_P = 0u; const F_Q = 1u; const F_IL = 2u; const F_FL = 3u; const F_RQ = 4u;
// shape record (8 vec4): (body, material, -, -), local position, local rotation, half extents,
// world position + grid level, world rotation, AABB min, AABB max. Shape 0 is the ground.
const SH = 8u;
const H_ID = 0u; const H_LP = 1u; const H_LQ = 2u; const H_H = 3u; const H_WP = 4u; const H_WQ = 5u; const H_LO = 6u; const H_HI = 7u;
// solver record (6 vec4): velocity + effective inverse mass (0 when static or asleep), angular velocity,
// displacement, rotation over the step, inverse inertia (xx, yy, zz, xy), (xz, yz, -, -)
const SS = 6u;
const S_V = 0u; const S_W = 1u; const S_DP = 2u; const S_TH = 3u; const S_I0 = 4u; const S_I1 = 5u;
// contact record (17 vec4): header (body a, body b, count, stick) as floats; normal + mu; friction anchor
// arms fa + tm1, fb + tm2; anchor offset e0 + twm; (jt1, jt2, jtw, rad); 4 x [(ra, adj), (rb, nm)]; jn x 4;
// normal approach speeds at the start of the step x 4; (restitution, impulse total this step, approach speed, -)
const MS = 17u;
const M_N = 1u; const M_FA = 2u; const M_FB = 3u; const M_E0 = 4u; const M_JT = 5u; const M_PT = 6u; const M_JN = 14u;
const M_RV = 15u; const M_OUT = 16u;
// cold record (6 vec4): 4 x local anchor on a; friction anchor on a + shape a; on b + shape b
const CS = 6u;
const C_MAN = 0u; const C_OVF = 1u; const C_COLORS = 2u; const C_PAIRS = 3u; const C_DROP = 4u; const C_NPAIR = 5u; const C_AWAKE = 6u; const C_EVENTS = 7u; const C_LEVEL = 8u;   // shapes per grid level: 8..11

@group(0) @binding(0) var<storage, read_write> bod: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> ints: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> man: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> cold: array<vec4f>;
@group(0) @binding(4) var<uniform> P: Params;
@group(0) @binding(5) var<storage, read_write> sol: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> shp: array<vec4f>;
@group(1) @binding(0) var<uniform> CU: ColorU;

fn ld(i: u32) -> u32 { return atomicLoad(&ints[i]); }
fn st(i: u32, v: u32) { atomicStore(&ints[i], v); }
fn B(i: u32, f: u32) -> vec4f { return bod[i * BS + f]; }
fn setB(i: u32, f: u32, v: vec4f) { bod[i * BS + f] = v; }
fn S(i: u32, f: u32) -> vec4f { return sol[i * SS + f]; }
fn setS(i: u32, f: u32, v: vec4f) { sol[i * SS + f] = v; }
fn Hs(i: u32, f: u32) -> vec4f { return shp[i * SH + f]; }
fn setHs(i: u32, f: u32, v: vec4f) { shp[i * SH + f] = v; }
fn sbody(s: u32) -> u32 { return u32(Hs(s, H_ID).x); }
fn imul(i0: vec4f, i1: vec4f, v: vec3f) -> vec3f {
  return vec3f(i0.x * v.x + i0.w * v.y + i1.x * v.z, i0.w * v.x + i0.y * v.y + i1.y * v.z, i1.x * v.x + i1.y * v.y + i0.z * v.z);
}
fn qmat(q: vec4f) -> mat3x3f {
  let x = q.x; let y = q.y; let z = q.z; let w = q.w;
  return mat3x3f(vec3f(1.0 - 2.0 * (y*y + z*z), 2.0 * (x*y + z*w), 2.0 * (x*z - y*w)),
                 vec3f(2.0 * (x*y - z*w), 1.0 - 2.0 * (x*x + z*z), 2.0 * (y*z + x*w)),
                 vec3f(2.0 * (x*z + y*w), 2.0 * (y*z - x*w), 1.0 - 2.0 * (x*x + y*y)));
}
fn qmul(a: vec4f, b: vec4f) -> vec4f {
  return vec4f(a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y, a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
               a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w, a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z);
}
fn qrot(q: vec4f, v: vec3f) -> vec3f { let u = q.xyz; let t = 2.0 * cross(u, v); return v + q.w * t + cross(u, t); }
fn lcell(L: u32) -> f32 { return P.cell * f32(1u << (2u * L)); }   // level L cell: 4^L x base
fn cellc(p: vec3f, L: u32) -> vec3i { return vec3i(floor(p / lcell(L))); }
fn chash(c: vec3i, L: u32) -> u32 { return ((u32(c.x) * 73856093u) ^ (u32(c.y) * 19349663u) ^ (u32(c.z) * 83492791u) ^ (L * 2654435761u)) & P.grid_mask; }
fn phash(a: u32, b: u32) -> u32 { return ((a * 2654435761u) ^ ((b + 0x9e3779b9u) * 2246822519u)) & P.tbl_mask; }
fn asleep(i: u32) -> bool { return B(i, F_FL).x > 0.5; }
fn dyn_awake(i: u32) -> bool { return S(i, S_V).w > 0.0; }
fn tangents(n: vec3f) -> mat2x3f {
  var t1: vec3f;
  if (abs(n.x) > 0.57) { t1 = vec3f(n.y, -n.x, 0.0); } else { t1 = vec3f(0.0, n.z, -n.y); }
  t1 = normalize(t1);
  return mat2x3f(t1, cross(n, t1));
}

// ---------------------------------------------------------------- broad phase
@compute @workgroup_size(64) fn prep_bodies(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + 1u; if (i >= P.n1) { return; }
  let p = B(i, F_P); let il = B(i, F_IL).xyz;
  if (asleep(i) || p.w == 0.0) {
    setS(i, S_V, vec4f(0.0)); setS(i, S_W, vec4f(0.0)); setS(i, S_I0, vec4f(0.0)); setS(i, S_I1, vec4f(0.0));
  } else {
    let R = qmat(B(i, F_Q));
    let a = R[0] * il.x; let b = R[1] * il.y; let c = R[2] * il.z;
    let c0 = a * R[0].x + b * R[1].x + c * R[2].x;
    let c1 = a * R[0].y + b * R[1].y + c * R[2].y;
    let c2 = a * R[0].z + b * R[1].z + c * R[2].z;
    setS(i, S_V, vec4f(S(i, S_V).xyz, p.w));
    setS(i, S_I0, vec4f(c0.x, c1.y, c2.z, c1.x)); setS(i, S_I1, vec4f(c2.x, c2.y, 0.0, 0.0));
    atomicAdd(&ints[C_AWAKE], 1u);
  }
  setS(i, S_DP, vec4f(0.0)); setS(i, S_TH, vec4f(0.0));
  st(P.o_mask + i, 0u);
  for (var c = 0u; c < P.color_limit; c++) { st(P.o_claim + i * 32u + c, 0u); st(P.o_claim2 + i * 32u + c, 0u); }
  st(P.o_par + i, i);
  st(P.o_mint + i, 0x7f7fffffu);
}
fn dead(s: u32) -> bool { return Hs(s, H_ID).z > 0.5; }
// Where anything awake is, coarsely: 4 m cells (hashed into CNEAR slots) under each awake shape's box. A resting
// or static shape whose box touches none of them can pair with nothing awake, and skips its search of the grid
// (27 cells at each level: in a standing building of 20,000 boxes with 8 boxes falling, that search was most of
// the broad phase: GPU test quietcity). A box over too many cells marks everything (the last slot).
const CNEAR = 8192u; const CNEAR_CELL = 4.0;
fn H_LO_v(s: u32) -> vec3f { return Hs(s, H_LO).xyz; }
fn H_HI_v(s: u32) -> vec3f { return Hs(s, H_HI).xyz; }
fn near_hash(c: vec3i) -> u32 { return ((u32(c.x) * 73856093u) ^ (u32(c.y) * 19349663u) ^ (u32(c.z) * 83492791u)) % CNEAR; }
fn near_mark(lo: vec3f, hi: vec3f) {
  let a = vec3i(floor(lo / CNEAR_CELL)); let b = vec3i(floor(hi / CNEAR_CELL)); let n = b - a + vec3i(1);
  if (n.x * n.y * n.z > 64) { atomicOr(&ints[P.o_cnear + CNEAR], 1u); return; }
  for (var z = a.z; z <= b.z; z++) { for (var y = a.y; y <= b.y; y++) { for (var x = a.x; x <= b.x; x++) { atomicOr(&ints[P.o_cnear + near_hash(vec3i(x, y, z))], 1u); } } }
}
fn near_any(lo: vec3f, hi: vec3f) -> bool {
  if (ld(P.o_cnear + CNEAR) != 0u) { return true; }
  let a = vec3i(floor(lo / CNEAR_CELL)); let b = vec3i(floor(hi / CNEAR_CELL)); let n = b - a + vec3i(1);
  if (n.x * n.y * n.z > 64) { return true; }
  for (var z = a.z; z <= b.z; z++) { for (var y = a.y; y <= b.y; y++) { for (var x = a.x; x <= b.x; x++) { if (ld(P.o_cnear + near_hash(vec3i(x, y, z))) != 0u) { return true; } } } }
  return false;
}
@compute @workgroup_size(64) fn prep_shapes(@builtin(global_invocation_id) g: vec3u) {
  let s = g.x + 1u; if (s >= P.ns1 || dead(s)) { return; }
  let b = sbody(s);
  let bq = B(b, F_Q);
  let lp = Hs(s, H_LP).xyz; let h = Hs(s, H_H).xyz;
  let wp = B(b, F_P).xyz + qrot(bq, lp);
  let wq = normalize(qmul(bq, Hs(s, H_LQ)));
  let R = qmat(wq);
  let e = abs(R[0]) * h.x + abs(R[1]) * h.y + abs(R[2]) * h.z;
  let v = S(b, S_V).xyz; let w = S(b, S_W).xyz;
  let m = (length(v) + length(w) * (length(lp) + length(h))) * P.dt + 0.5 * P.spec;
  let ext = 2.0 * (max(e.x, max(e.y, e.z)) + m);
  var L = 0u;
  loop { if (L + 1u >= P.levels || lcell(L) >= ext) { break; } L++; }
  setHs(s, H_WP, vec4f(wp, f32(L))); setHs(s, H_WQ, wq);
  atomicAdd(&ints[C_LEVEL + L], 1u);
  setHs(s, H_LO, vec4f(wp - e - vec3f(m), 0.0)); setHs(s, H_HI, vec4f(wp + e + vec3f(m), 0.0));
  let hc = chash(cellc(wp, L), L);
  st(P.o_cell + s, hc);
  atomicAdd(&ints[P.o_gcount + hc], 1u);
  // cells holding a moving shape: resting and static shapes only look into these
  if (dyn_awake(b)) { atomicOr(&ints[P.o_cawake + hc], 1u); near_mark(H_LO_v(s), H_HI_v(s)); }
  // (and 2: one that is going somewhere, fast enough to wake what it touches (see pair_wake_emit) or moved over
  // 1 cm in the last quarter second (its rest timer, see finish): only these wake resting particles. Any awake
  // body woke them, one only chattering in place too, and a settled pile never slept: lab mixed at 10 s, 36,879 of
  // 36,882 particles awake, most under 0.1 m/s. On speed alone, a block sinking slowly into a sleeping heap went
  // through it: GPU test heapcreep, slab awake)
  if (dyn_awake(b) && (length(v) + length(w) * (length(lp) + length(h)) > P.wake_speed || B(b, F_IL).w < 0.5 * P.sleep_time)) { atomicOr(&ints[P.o_cawake + hc], 2u); }
}

var<workgroup> wsum: array<u32, 256>;
fn wg_scan(t: u32, s: u32) -> u32 {
  wsum[t] = s;
  workgroupBarrier();
  for (var off = 1u; off < 256u; off *= 2u) {
    var v = 0u; if (t >= off) { v = wsum[t - off]; }
    workgroupBarrier(); wsum[t] += v; workgroupBarrier();
  }
  var run = 0u; if (t > 0u) { run = wsum[t - 1u]; }
  return run;
}
@compute @workgroup_size(256) fn scan(@builtin(local_invocation_index) t: u32) {
  let G = P.grid_mask + 1u; let per = G / 256u; let base = t * per;
  var s = 0u;
  for (var k = 0u; k < per; k++) { s += ld(P.o_gcount + base + k); }
  var run = wg_scan(t, s);
  for (var k = 0u; k < per; k++) { st(P.o_gstart + base + k, run); run += ld(P.o_gcount + base + k); }
}
@compute @workgroup_size(64) fn scatter(@builtin(global_invocation_id) g: vec3u) {
  let s = g.x + 1u; if (s >= P.ns1 || dead(s)) { return; }
  let hc = ld(P.o_cell + s);
  st(P.o_sorted2 + ld(P.o_gstart + hc) + atomicAdd(&ints[P.o_gfill + hc], 1u), s);
}

// Determinism: the grid's cells are filled by atomics, in whatever order the threads get there; each shape then
// takes its place in its cell by shape order (its rank among the cell's shapes), so everything that walks a cell
// (pairs, particle candidates, capped at 24) is the same every run (GPU test determinism). (Sorting each cell
// with a thread per cell of the hash grid cost 0.5 M threads a step in the page.)
@compute @workgroup_size(64) fn sort_cells(@builtin(global_invocation_id) g: vec3u) {
  let s = g.x + 1u; if (s >= P.ns1 || dead(s)) { return; }
  let hc = ld(P.o_cell + s);
  let s0 = ld(P.o_gstart + hc); let n = ld(P.o_gcount + hc);
  var r = 0u;
  for (var k = 0u; k < n; k++) { if (ld(P.o_sorted2 + s0 + k) < s) { r++; } }
  st(P.o_sorted + s0 + r, s);
}
// A contact's key: its two shapes, mixed (the same every run, unlike its slot, which narrow hands out by atomics).
// Choices between contacts (colours, the contacts kept by reduction, the order of a body's sums) go by it.
fn mkey(m: u32) -> u32 {
  let sa = u32(cold[m * CS + 4u].w); let sb = u32(cold[m * CS + 5u].w);
  var h = (sa * 0x9E3779B1u) ^ ((sb + 0x7F4A7C15u) * 0x85EBCA77u);
  h ^= h >> 15u; h *= 0x2C1B3C6Du; h ^= h >> 12u;
  return h | 1u;
}
fn push_pair(a: u32, b: u32) {
  let k = atomicAdd(&ints[C_NPAIR], 1u);
  if (k < P.max_pairs) { st(P.o_pairs + 2u * k, a); st(P.o_pairs + 2u * k + 1u, b); }
}
// Only shapes of awake bodies search (static and sleeping shapes are found by them). A shape searches every
// non-empty level: its own and coarser levels in the 27 cells around its centre, finer levels over its AABB.
// Each shape sits in the cell of its centre, so checking that cell visits it once. Two awake shapes report
// their pair from the lower index.
// Every shape searches its own grid level and the coarser ones, a 3 x 3 x 3 neighbourhood at each (a shape's
// level fits its size, so anything it overlaps on those levels sits in a neighbouring cell). A big shape never
// scans finer levels: the small shapes there find it. A moving shape looks at everything; a resting or static
// one only into cells that hold a moving shape, and only at moving shapes. Each pair is found once: on one
// level by the lower index, across levels by the finer shape. The moving shape goes first in the pair.
@compute @workgroup_size(64) fn pairs(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + 1u; if (i >= P.ns1 || dead(i)) { return; }
  let bi = sbody(i);
  let ai = dyn_awake(bi);
  let lo = Hs(i, H_LO).xyz; let hi = Hs(i, H_HI).xyz;
  let wp = Hs(i, H_WP); let Li = u32(wp.w);
  if (ai && lo.y < 0.0) { push_pair(i, 0u); }
  if (!ai && !near_any(lo, hi)) { return; }
  for (var L = Li; L < P.levels; L++) {
    if (ld(C_LEVEL + L) == 0u) { continue; }
    let c = cellc(wp.xyz, L);
    for (var cz = c.z - 1; cz <= c.z + 1; cz++) { for (var cy = c.y - 1; cy <= c.y + 1; cy++) { for (var cx = c.x - 1; cx <= c.x + 1; cx++) {
      let cell = vec3i(cx, cy, cz);
      let hc = chash(cell, L);
      if (!ai && ld(P.o_cawake + hc) == 0u) { continue; }
      let s0 = ld(P.o_gstart + hc); let s1 = s0 + ld(P.o_gcount + hc);
      for (var s = s0; s < s1; s++) {
        let j = ld(P.o_sorted + s);
        let wj = Hs(j, H_WP);
        if (u32(wj.w) != L || any(cellc(wj.xyz, L) != cell)) { continue; }
        if (L == Li && j <= i) { continue; }
        let bj = sbody(j);
        if (bj == bi) { continue; }
        let aj = dyn_awake(bj);
        if (!ai && !aj) { continue; }
        let lj = Hs(j, H_LO).xyz; let hj = Hs(j, H_HI).xyz;
        if (any(lo > hj) || any(lj > hi)) { continue; }
        if (ai) { push_pair(i, j); } else { push_pair(j, i); }
      }
    } } }
  }
}

// ---------------------------------------------------------------- narrow phase
struct Contact { n: vec3f, cnt: u32, pts: array<vec4f, 8> }  // xyz point, w separation

fn ground_contact(p: vec3f, q: vec4f, h: vec3f, margin: f32) -> Contact {
  var c: Contact; c.cnt = 0u; c.n = vec3f(0.0, -1.0, 0.0);
  let R = qmat(q);
  for (var k = 0u; k < 8u; k++) {
    let s = vec3f(select(-1.0, 1.0, (k & 1u) != 0u), select(-1.0, 1.0, (k & 2u) != 0u), select(-1.0, 1.0, (k & 4u) != 0u));
    let x = p + R * (s * h);
    if (x.y < margin) { c.pts[c.cnt] = vec4f(x.x, 0.5 * x.y, x.z, x.y); c.cnt++; }
  }
  return c;
}

fn box_box(pa: vec3f, ra: mat3x3f, ha: vec3f, pb: vec3f, rb: mat3x3f, hb: vec3f, margin: f32) -> Contact {
  var c: Contact; c.cnt = 0u;
  let t = pb - pa;
  var ac: array<vec3f, 3>;
  for (var i = 0u; i < 3u; i++) { ac[i] = abs(vec3f(dot(ra[i], rb[0]), dot(ra[i], rb[1]), dot(ra[i], rb[2]))) + vec3f(1e-6); }
  var sa = -1e30; var ia = 0u;
  for (var i = 0u; i < 3u; i++) {
    let s = abs(dot(t, ra[i])) - (ha[i] + dot(hb, ac[i]));
    if (s > margin) { return c; }
    if (s > sa) { sa = s; ia = i; }
  }
  var sb = -1e30; var jb = 0u;
  for (var j = 0u; j < 3u; j++) {
    let s = abs(dot(t, rb[j])) - (hb[j] + ha.x * ac[0][j] + ha.y * ac[1][j] + ha.z * ac[2][j]);
    if (s > margin) { return c; }
    if (s > sb) { sb = s; jb = j; }
  }
  var se = -1e30; var ei = 0u; var ej = 0u; var el = vec3f(0.0);
  for (var i = 0u; i < 3u; i++) { for (var j = 0u; j < 3u; j++) {
    let l0 = cross(ra[i], rb[j]); let len2 = dot(l0, l0);
    if (len2 < 1e-8) { continue; }
    let l = l0 * inverseSqrt(len2);
    let r = ha.x * abs(dot(ra[0], l)) + ha.y * abs(dot(ra[1], l)) + ha.z * abs(dot(ra[2], l))
          + hb.x * abs(dot(rb[0], l)) + hb.y * abs(dot(rb[1], l)) + hb.z * abs(dot(rb[2], l));
    let s = abs(dot(t, l)) - r;
    if (s > margin) { return c; }
    if (s > se) { se = s; ei = i; ej = j; el = l; }
  } }
  let use_b = sb > sa + 0.005;
  let face_best = select(sa, sb, use_b);
  if (se > face_best + 0.01) {
    var n = el; if (dot(t, el) < 0.0) { n = -el; }
    var ca = pa; var cb = pb;
    for (var k = 0u; k < 3u; k++) {
      if (k != ei) { ca += ra[k] * (select(-1.0, 1.0, dot(ra[k], n) > 0.0) * ha[k]); }
      if (k != ej) { cb += rb[k] * (select(1.0, -1.0, dot(rb[k], n) > 0.0) * hb[k]); }
    }
    let d1 = ra[ei]; let d2 = rb[ej]; let r = ca - cb;
    let bb = dot(d1, d2); let cc = dot(d1, r); let f = dot(d2, r);
    let den = 1.0 - bb * bb;
    var s = 0.0; if (den > 1e-6) { s = (bb * f - cc) / den; }
    s = clamp(s, -ha[ei], ha[ei]);
    let tt = clamp(bb * s + f, -hb[ej], hb[ej]);
    s = clamp(bb * tt - cc, -ha[ei], ha[ei]);
    let p1 = ca + d1 * s; let p2 = cb + d2 * tt;
    c.n = n; c.cnt = 1u; c.pts[0] = vec4f(0.5 * (p1 + p2), dot(p2 - p1, n));
    return c;
  }
  // face contact: reference R, incident I
  var pr = pa; var rr = ra; var hr = ha; var pi = pb; var ri = rb; var hi = hb; var k = ia;
  if (use_b) { pr = pb; rr = rb; hr = hb; pi = pa; ri = ra; hi = ha; k = jb; }
  var n = rr[k]; if (dot(pi - pr, n) < 0.0) { n = -n; }
  var j = 0u; var bd = -1.0;
  for (var q = 0u; q < 3u; q++) { let d = abs(dot(ri[q], n)); if (d > bd) { bd = d; j = q; } }
  let sg = select(1.0, -1.0, dot(ri[j], n) > 0.0);
  let fc = pi + ri[j] * (sg * hi[j]);
  let u = (j + 1u) % 3u; let v = (j + 2u) % 3u;
  let eu = ri[u] * hi[u]; let ev = ri[v] * hi[v];
  var poly: array<vec3f, 8>; var tmp: array<vec3f, 8>;
  poly[0] = fc + eu + ev; poly[1] = fc - eu + ev; poly[2] = fc - eu - ev; poly[3] = fc + eu - ev;
  var pn = 4u;
  let u2 = (k + 1u) % 3u; let v2 = (k + 2u) % 3u;
  for (var side = 0u; side < 4u; side++) {
    let ax = select(u2, v2, side >= 2u);
    let sgn = select(1.0, -1.0, (side & 1u) != 0u);
    let dir = rr[ax] * sgn;
    let off = dot(dir, pr) + hr[ax];
    var tn = 0u;
    for (var q = 0u; q < pn; q++) {
      let p0 = poly[q]; let p1 = poly[(q + 1u) % pn];
      let d0 = dot(p0, dir) - off; let d1 = dot(p1, dir) - off;
      if (d0 <= 0.0 && tn < 8u) { tmp[tn] = p0; tn++; }
      if ((d0 <= 0.0) != (d1 <= 0.0) && tn < 8u) { tmp[tn] = p0 + (p1 - p0) * (d0 / (d0 - d1)); tn++; }
    }
    pn = tn;
    for (var q = 0u; q < pn; q++) { poly[q] = tmp[q]; }
    if (pn == 0u) { return c; }
  }
  let rc = pr + n * hr[k];
  for (var q = 0u; q < pn; q++) {
    let p = poly[q]; let sep = dot(p - rc, n);
    if (sep <= margin) { c.pts[c.cnt] = vec4f(p - n * (0.5 * sep), sep); c.cnt++; }
  }
  c.n = select(n, -n, use_b);
  return c;
}


// ---- convex chunks (see hullLibrary): a shape with hull t > 0 is that hull in its box, stretched by h
const HSTR = 80u;
fn hhead(t: u32) -> vec4f { return shp[P.o_hull + t * HSTR]; }
fn hv(t: u32, k: u32) -> vec3f { return shp[P.o_hull + t * HSTR + 1u + k].xyz; }
fn hplane(t: u32, k: u32) -> vec4f { return shp[P.o_hull + t * HSTR + 17u + k]; }
fn hfidx(t: u32, f: u32, i: u32) -> u32 { let v = shp[P.o_hull + t * HSTR + 29u + 2u * f + i / 4u]; return u32(v[i % 4u]); }
fn hfcnt(t: u32, f: u32) -> u32 { let v = shp[P.o_hull + t * HSTR + 77u + f / 4u]; return u32(v[f % 4u]); }
fn hseg(t: u32, k: u32) -> vec2u { let v = shp[P.o_hull + t * HSTR + 53u + k / 2u]; return select(vec2u(u32(v.x), u32(v.y)), vec2u(u32(v.z), u32(v.w)), (k & 1u) != 0u); }
fn hdir(t: u32, k: u32) -> vec3f { return shp[P.o_hull + t * HSTR + 65u + k].xyz; }
struct HShape { p: vec3f, R: mat3x3f, h: vec3f, t: u32 }
fn hw(s: HShape, k: u32) -> vec3f { return s.p + s.R * (s.h * hv(s.t, k)); }
// the face's outward normal and plane offset in the world
fn hface(s: HShape, f: u32) -> vec4f { let pl = hplane(s.t, f); let nw = s.R * (pl.xyz / s.h); let l = length(nw); let n = nw / l; return vec4f(n, pl.w / l + dot(n, s.p)); }
fn hivl(s: HShape, ax: vec3f) -> vec2f {
  let al = (transpose(s.R) * ax) * s.h; let nv = u32(hhead(s.t).x);
  var lo = 1e30; var hi = -1e30;
  for (var k = 0u; k < nv; k++) { let d = dot(hv(s.t, k), al); lo = min(lo, d); hi = max(hi, d); }
  let c = dot(s.p, ax); return vec2f(c + lo, c + hi);
}
fn hull_ground(s: HShape, margin: f32) -> Contact {
  var c: Contact; c.cnt = 0u; c.n = vec3f(0.0, -1.0, 0.0);
  let nv = u32(hhead(s.t).x);
  // the 8 lowest corners within the margin (a chunk has up to 16: taking the first 8 found can miss the lowest)
  for (var k = 0u; k < nv; k++) {
    let x = hw(s, k); if (x.y >= margin) { continue; }
    let pt = vec4f(x.x, 0.5 * x.y, x.z, x.y);
    if (c.cnt < 8u) { c.pts[c.cnt] = pt; c.cnt++; continue; }
    var hi = 0u; for (var q = 1u; q < 8u; q++) { if (c.pts[q].w > c.pts[hi].w) { hi = q; } }
    if (pt.w < c.pts[hi].w) { c.pts[hi] = pt; }
  }
  return c;
}
// Separating axes: the faces of both, then the cross products of their edge directions; the contact from the
// best one: a reference face clipped against the incident face (up to 8 points, reduced to 4 later), or the
// closest points of two edges. The normal points from A to B; w is the separation (negative: overlap).
fn hull_hull(A: HShape, B: HShape, margin: f32) -> Contact {
  var c: Contact; c.cnt = 0u;
  let ha = hhead(A.t); let hb = hhead(B.t);
  let nfa = u32(ha.y); let nfb = u32(hb.y); let nva = u32(ha.x); let nvb = u32(hb.x);
  // both hulls' corners and edge directions in the world, read once (every axis projects all corners: read
  // from the shape buffer each time that was ~5,000 reads per pair)
  var VA: array<vec3f, 16>; var VB: array<vec3f, 16>; var DA: array<vec3f, 12>; var DB: array<vec3f, 12>;
  for (var k = 0u; k < nva; k++) { VA[k] = hw(A, k); }
  for (var k = 0u; k < nvb; k++) { VB[k] = hw(B, k); }
  let nda = u32(ha.w); let ndb = u32(hb.w);
  for (var k = 0u; k < nda; k++) { DA[k] = normalize(A.R * (A.h * hdir(A.t, k))); }
  for (var k = 0u; k < ndb; k++) { DB[k] = normalize(B.R * (B.h * hdir(B.t, k))); }
  var sa = -1e30; var fa = 0u;
  for (var f = 0u; f < nfa; f++) {
    let pl = hface(A, f); var lo = 1e30; for (var k = 0u; k < nvb; k++) { lo = min(lo, dot(VB[k], pl.xyz)); }
    let s = lo - pl.w; if (s > margin) { return c; } if (s > sa) { sa = s; fa = f; }
  }
  var sb = -1e30; var fb = 0u;
  for (var f = 0u; f < nfb; f++) {
    let pl = hface(B, f); var lo = 1e30; for (var k = 0u; k < nva; k++) { lo = min(lo, dot(VA[k], pl.xyz)); }
    let s = lo - pl.w; if (s > margin) { return c; } if (s > sb) { sb = s; fb = f; }
  }
  var se = -1e30; var el = vec3f(0.0); var ei = 0u; var ej = 0u;
  for (var i = 0u; i < nda; i++) {
    for (var j = 0u; j < ndb; j++) {
      let l0 = cross(DA[i], DB[j]); let l2 = dot(l0, l0); if (l2 < 1e-6) { continue; }
      // (its sign: the side where B is clear of A the most; the box centres can lie outside a wedge)
      var l = l0 * inverseSqrt(l2);
      var alo = 1e30; var ahi = -1e30; var blo = 1e30; var bhi = -1e30;
      for (var k = 0u; k < nva; k++) { let d = dot(VA[k], l); alo = min(alo, d); ahi = max(ahi, d); }
      for (var k = 0u; k < nvb; k++) { let d = dot(VB[k], l); blo = min(blo, d); bhi = max(bhi, d); }
      let s1 = blo - ahi; let s2 = alo - bhi;
      var s = s1; if (s2 > s1) { s = s2; l = -l; }
      if (s > margin) { return c; }
      if (s > se) { se = s; el = l; ei = i; ej = j; }
    }
  }
  let use_b = sb > sa + 0.005;
  let fbest = select(sa, sb, use_b);
  if (se > fbest + 0.01) {
    // edge against edge: the support segments along the axis
    let da = normalize(A.R * (A.h * hdir(A.t, ei))); let db = normalize(B.R * (B.h * hdir(B.t, ej)));
    var a0 = A.p; var a1 = A.p; var best = -1e30;
    for (var k = 0u; k < u32(ha.z); k++) { let sg = hseg(A.t, k); let p0 = hw(A, sg.x); let p1 = hw(A, sg.y);
      if (abs(dot(normalize(p1 - p0), da)) < 0.999) { continue; } let m = dot(p0 + p1, el); if (m > best) { best = m; a0 = p0; a1 = p1; } }
    var b0 = B.p; var b1 = B.p; best = 1e30;
    for (var k = 0u; k < u32(hb.z); k++) { let sg = hseg(B.t, k); let p0 = hw(B, sg.x); let p1 = hw(B, sg.y);
      if (abs(dot(normalize(p1 - p0), db)) < 0.999) { continue; } let m = dot(p0 + p1, el); if (m < best) { best = m; b0 = p0; b1 = p1; } }
    let d1 = a1 - a0; let d2 = b1 - b0; let r = a0 - b0;
    let aa = dot(d1, d1); let ee = dot(d2, d2); let ff = dot(d2, r); let cc = dot(d1, r); let bb = dot(d1, d2);
    let den = aa * ee - bb * bb;
    var s1 = 0.0; if (den > 1e-9) { s1 = clamp((bb * ff - cc * ee) / den, 0.0, 1.0); }
    var t1 = (bb * s1 + ff) / max(ee, 1e-9);
    if (t1 < 0.0) { t1 = 0.0; s1 = clamp(-cc / max(aa, 1e-9), 0.0, 1.0); } else if (t1 > 1.0) { t1 = 1.0; s1 = clamp((bb - cc) / max(aa, 1e-9), 0.0, 1.0); }
    let p1 = a0 + d1 * s1; let p2 = b0 + d2 * t1;
    c.n = el; c.cnt = 1u; c.pts[0] = vec4f(0.5 * (p1 + p2), dot(p2 - p1, el));
    return c;
  }
  // a face: reference R (the shape whose face it is), incident I (the other's face most against it)
  var Rs = A; var Is = B; var fr = fa; if (use_b) { Rs = B; Is = A; fr = fb; }
  let rpl = hface(Rs, fr); let n = rpl.xyz;
  let nfi = select(nfb, nfa, use_b);
  var fi = 0u; var bd = 1e30;
  for (var f = 0u; f < nfi; f++) { let d = dot(hface(Is, f).xyz, n); if (d < bd) { bd = d; fi = f; } }
  var poly: array<vec3f, 16>; var tmp: array<vec3f, 16>;
  var pn = hfcnt(Is.t, fi);
  for (var q = 0u; q < pn; q++) { poly[q] = hw(Is, hfidx(Is.t, fi, q)); }
  let rn = hfcnt(Rs.t, fr);
  var rc = vec3f(0.0); for (var q = 0u; q < rn; q++) { rc += hw(Rs, hfidx(Rs.t, fr, q)); } rc /= f32(rn);
  for (var e = 0u; e < rn; e++) {
    let v0 = hw(Rs, hfidx(Rs.t, fr, e)); let v1 = hw(Rs, hfidx(Rs.t, fr, (e + 1u) % rn));
    var sn = normalize(cross(v1 - v0, n)); if (dot(sn, rc - v0) > 0.0) { sn = -sn; }
    var tn = 0u;
    for (var q = 0u; q < pn; q++) {
      let p0 = poly[q]; let p1 = poly[(q + 1u) % pn];
      let d0 = dot(p0 - v0, sn); let d1 = dot(p1 - v0, sn);
      if (d0 <= 0.0 && tn < 16u) { tmp[tn] = p0; tn++; }
      if ((d0 <= 0.0) != (d1 <= 0.0) && tn < 16u) { tmp[tn] = p0 + (p1 - p0) * (d0 / (d0 - d1)); tn++; }
    }
    pn = tn; for (var q = 0u; q < pn; q++) { poly[q] = tmp[q]; }
    if (pn == 0u) { return c; }
  }
  for (var q = 0u; q < pn; q++) {
    let p = poly[q]; let sep = dot(p, n) - rpl.w;
    if (sep > margin) { continue; }
    let pt = vec4f(p - n * (0.5 * sep), sep);
    if (c.cnt < 8u) { c.pts[c.cnt] = pt; c.cnt++; continue; }
    // (more than 8 after clipping: keep the deepest)
    var hi = 0u; for (var r = 1u; r < 8u; r++) { if (c.pts[r].w > c.pts[hi].w) { hi = r; } }
    if (pt.w < c.pts[hi].w) { c.pts[hi] = pt; }
  }
  c.n = select(n, -n, use_b);
  return c;
}

fn reduce4(c: ptr<function, Contact>) {
  let cnt = (*c).cnt; if (cnt <= 4u) { return; }
  let n = (*c).n;
  var i0 = 0u; for (var k = 1u; k < cnt; k++) { if ((*c).pts[k].w < (*c).pts[i0].w) { i0 = k; } }
  let p0 = (*c).pts[i0].xyz;
  var i1 = 0u; var best = -1.0;
  for (var k = 0u; k < cnt; k++) { let d = (*c).pts[k].xyz - p0; if (dot(d, d) > best) { best = dot(d, d); i1 = k; } }
  let e = (*c).pts[i1].xyz - p0;
  var i2 = 0u; var b2 = -1e30; var i3 = 0u; var b3 = -1e30;
  for (var k = 0u; k < cnt; k++) {
    let a = dot(cross((*c).pts[k].xyz - p0, e), n);
    if (a > b2) { b2 = a; i2 = k; }
    if (-a > b3) { b3 = -a; i3 = k; }
  }
  var sel = array<u32, 4>(i0, i1, i2, i3);
  var out: array<vec4f, 4>; var oc = 0u;
  for (var k = 0u; k < 4u; k++) {
    var dup = false; for (var q = 0u; q < k; q++) { if (sel[q] == sel[k]) { dup = true; } }
    if (!dup) { out[oc] = (*c).pts[sel[k]]; oc++; }
  }
  for (var k = 0u; k < oc; k++) { (*c).pts[k] = out[k]; }
  (*c).cnt = oc;
}


fn find_old(sa: u32, sb: u32) -> u32 {
  var s = phash(sa, sb);
  for (var probe = 0u; probe < 64u; probe++) {
    let v = ld(P.o_told + s);
    if (v == 0u) { return 0xffffffffu; }
    let m = v - 1u;
    if (u32(cold[m * CS + 4u].w) == sa && u32(cold[m * CS + 5u].w) == sb) { return m; }
    s = (s + 1u) & P.tbl_mask;
  }
  return 0xffffffffu;
}
fn insert_new(sa: u32, sb: u32, idx: u32) {
  var s = phash(sa, sb);
  for (var probe = 0u; probe < 64u; probe++) {
    let r = atomicCompareExchangeWeak(&ints[P.o_tnew + s], 0u, idx + 1u);
    if (r.exchanged) { return; }
    if (r.old_value == 0u) { continue; }
    s = (s + 1u) & P.tbl_mask;
  }
}

fn crushed(i: u32) -> bool { return i != 0u && ld(P.o_crushed + i) != 0u; }
fn moving(i: u32) -> bool { return i != 0u && B(i, F_P).w > 0.0; }
// (a contact against a particle that holds a body up, see p_support: b is the static world, sb this key, and the
// material the particle's)
const SUP_KEY = 0xffffffu;
var<private> sup_mat: u32 = 0xffffffffu; var<private> sup_cap: f32 = 0.0;   // (a support contact's material, and its points' push limit, N s a sub-step)
fn emit(sa: u32, sb: u32, a: u32, b: u32, c0: Contact) {
  // a crushed body (see crush) touches only the static world until the page turns it into rubble
  if ((crushed(a) && moving(b)) || (crushed(b) && moving(a))) { return; }
  var c = c0;
  reduce4(&c);
  let slot = atomicAdd(&ints[C_MAN], 1u);
  if (slot >= P.max_man) { atomicAdd(&ints[C_DROP], 1u); return; }
  let mi = P.cur_base + slot;
  let base = mi * MS; let cb = mi * CS;
  let n = c.n;
  let tt = tangents(n); let t1 = tt[0]; let t2 = tt[1];
  let pa = B(a, F_P).xyz; let Ra = qmat(B(a, F_Q));
  var pb = vec3f(0.0); var Rb = mat3x3f(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0));
  if (b != 0u) { pb = B(b, F_P).xyz; Rb = qmat(B(b, F_Q)); }
  let va = S(a, S_V); let wa = S(a, S_W).xyz; let vb = S(b, S_V); let wb = S(b, S_W).xyz;
  let ima = va.w; let imb = vb.w;
  let A0 = S(a, S_I0); let A1 = S(a, S_I1); let B0 = S(b, S_I0); let B1 = S(b, S_I1);
  let ma = P.mat[u32(Hs(sa, H_ID).y)];
  var mb = P.mat[P.ground_mat]; if (sb == SUP_KEY) { mb = P.mat[sup_mat]; } else if (sb != 0u) { mb = P.mat[u32(Hs(sb, H_ID).y)]; }
  let mu = sqrt(ma.x * mb.x); var rest = max(ma.y, mb.y);
  // (a heap of rubble does not bounce: no restitution against particles)
  if (sb == SUP_KEY) { rest = -1.0; }   // (none; and -1 marks the contact for solve_man, see p_support)
  var cen = vec3f(0.0);
  for (var k = 0u; k < c.cnt; k++) { cen += c.pts[k].xyz; }
  cen /= f32(c.cnt);
  var rad = 0.0; for (var k = 0u; k < c.cnt; k++) { rad += length(c.pts[k].xyz - cen); }
  rad /= f32(c.cnt);
  var old = find_old(sa, sb);
  // (nor a contact of a body that something crushed in the last step was pressing: its impulse was that load,
  // now gone, and warm-started it threw the body. GPU test chain: a 223 kg piece pressed into the one under it
  // by a block that a 259 t part landed on, the block crushed, went up at 20.6 m/s)
  if (old != 0xffffffffu && (ld(P.o_released + a) == P.step || ld(P.o_released + b) == P.step)) { old = 0xffffffffu; }
  // (a contact against particles starts afresh each step: its particles change, and an impulse warm-started at
  // other points lifted the body)
  if (sb == SUP_KEY) { old = 0xffffffffu; }
  var jt1 = 0.0; var jt2 = 0.0; var jtw = 0.0;
  var fa = cen - pa; var fb = cen - pb; var e0 = vec3f(0.0);
  var laf = transpose(Ra) * fa; var lbf = select(transpose(Rb) * fb, cen, b == 0u);
  var ojn = vec4f(0.0); var ocnt = 0u;
  // Which side was fixed (static, or asleep and held as static) when the contact was made. A moving body that
  // hits a sleeping one is stopped as if by a wall, and the contact keeps that whole impact impulse; the
  // sleeper wakes, and at the next step that impulse was warm-started into it at its real (small) mass: single
  // bricks at rest were thrown at 20-35 m/s, once at 2,800 m/s (lab collapse, real GPU and software GPU,
  // always one step after waking; GPU test wakeup). Such a contact starts afresh.
  let fixedNow = select(0.0, 1.0, ima == 0.0) + select(0.0, 2.0, imb == 0.0);
  if (old != 0xffffffffu && man[old * MS + M_OUT].w != fixedNow && (P.dbg & 16u) == 0u) { old = 0xffffffffu; }
  if (old != 0xffffffffu) {
    let ob = old * MS; let oc = old * CS;
    let ohd = man[ob]; ocnt = u32(ohd.z) & 7u; ojn = man[ob + M_JN];
    let olaf = cold[oc + 4u].xyz; let olbf = cold[oc + 5u].xyz;
    if (ohd.w > 0.5) {
      let wa_ = pa + Ra * olaf;
      var wb_ = olbf; if (b != 0u) { wb_ = pb + Rb * olbf; }
      let lim = max(0.05, rad);
      if (dot(wa_ - cen, wa_ - cen) < lim * lim && dot(wb_ - wa_, wb_ - wa_) < 0.03 * 0.03) {
        laf = olaf; lbf = olbf; fa = wa_ - pa; fb = wb_ - pb; e0 = wb_ - wa_;
      }
    }
    let on = man[ob + M_N].xyz; let ot = tangents(on); let ojt = man[ob + M_JT];
    let f = ot[0] * ojt.x + ot[1] * ojt.y;
    jt1 = dot(f, t1); jt2 = dot(f, t2); jtw = ojt.z * dot(on, n);
  }
  var jn = vec4f(0.0); var rv = vec4f(0.0); var approach = 0.0;
  for (var k = 0u; k < 4u; k++) {
    if (k >= c.cnt) { man[base + M_PT + 2u * k] = vec4f(0.0); man[base + M_PT + 2u * k + 1u] = vec4f(0.0); continue; }
    let p = c.pts[k].xyz; let s = c.pts[k].w;
    let ra = p - pa; let rb = p - pb;
    let raxn = cross(ra, n); let rbxn = cross(rb, n);
    let kk = ima + imb + dot(raxn, imul(A0, A1, raxn)) + dot(rbxn, imul(B0, B1, rbxn));
    let la = transpose(Ra) * ra;
    if (old != 0xffffffffu) {
      var best = 0.03 * 0.03;
      for (var q = 0u; q < ocnt; q++) { let d = cold[old * CS + q].xyz - la; if (dot(d, d) < best) { best = dot(d, d); jn[k] = ojn[q]; } }
    }
    rv[k] = dot(vb.xyz + cross(wb, rb) - va.xyz - cross(wa, ra), n);
    approach = max(approach, -rv[k]);
    man[base + M_PT + 2u * k] = vec4f(ra, s);
    man[base + M_PT + 2u * k + 1u] = vec4f(rb, select(0.0, 1.0 / kk, kk > 0.0));
    cold[cb + k] = vec4f(la, 0.0);
  }
  let ca1 = cross(fa, t1); let ca2 = cross(fa, t2); let cb1 = cross(fb, t1); let cb2 = cross(fb, t2);
  let k1 = ima + imb + dot(ca1, imul(A0, A1, ca1)) + dot(cb1, imul(B0, B1, cb1));
  let k2 = ima + imb + dot(ca2, imul(A0, A1, ca2)) + dot(cb2, imul(B0, B1, cb2));
  let kt = dot(n, imul(A0, A1, n)) + dot(n, imul(B0, B1, n));
  man[base] = vec4f(f32(a), f32(b), f32(c.cnt), f32(P.anchor));
  man[base + M_N] = vec4f(n, mu);
  man[base + M_FA] = vec4f(fa, select(0.0, 1.0 / k1, k1 > 0.0));
  man[base + M_FB] = vec4f(fb, select(0.0, 1.0 / k2, k2 > 0.0));
  man[base + M_E0] = vec4f(e0, select(0.0, 1.0 / kt, kt > 0.0));
  man[base + M_JT] = vec4f(jt1, jt2, jtw, rad);
  man[base + M_JN] = jn;
  if (sb == SUP_KEY) { rv = vec4f(sup_cap); }
  man[base + M_RV] = rv;
  man[base + M_OUT] = vec4f(rest, 0.0, approach, fixedNow);
  cold[cb + 4u] = vec4f(laf, f32(sa)); cold[cb + 5u] = vec4f(lbf, f32(sb));
  insert_new(sa, sb, mi);
}

fn wake(i: u32) {
  // (never a spare parked 1 km under the ground, waiting for a break on the GPU to hand it a part: woken by a
  // blast's wake-all, spares fell from there for the rest of the run, and the world never came to rest; lab
  // collapse, 200 frames: 38 of 234 awake bodies)
  if (B(i, F_P).y < -500.0) { return; }
  if (i != 0u && asleep(i)) { setB(i, F_FL, vec4f(0.0, B(i, F_P).xyz)); setB(i, F_RQ, B(i, F_Q)); let il = B(i, F_IL); setB(i, F_IL, vec4f(il.xyz, 0.0)); }
}

// New bodies cut from a body that moves on the GPU: JS made them where the parent was in an older read-back
// (Pr, Qr); at the step they appear they are carried along by the parent's motion since then, and take its
// current velocity (plus a given extra dv). Entries: [kid, parent, Pr xyz, Qr xyzw, dv xyz]; kid 0 retires the
// parent after all its pieces have been placed.
fn inh(k: u32, o: u32) -> f32 { return bitcast<f32>(ld(P.o_inherit + k * 12u + o)); }
@compute @workgroup_size(64) fn inherit(@builtin(global_invocation_id) g: vec3u) {
  let k = g.x; if (k >= P.n_inherit) { return; }
  let kid = ld(P.o_inherit + k * 12u); let praw = ld(P.o_inherit + k * 12u + 1u);
  // the high bit: dv is the piece's own velocity (a part cut free keeps falling while the parent was stopped);
  // the next: dv is the direction the contacts pushed, and the piece keeps none of a rebound along it (a crushing
  // impact absorbs the energy, it does not bounce)
  let absolute = (praw & 0x80000000u) != 0u; let noRebound = (praw & 0x40000000u) != 0u; let par = praw & 0x3fffffffu;
  if (kid == 0u || par == 0u || (kid & 0x80000000u) != 0u) { return; }   // (high bit: a particle, see p_place)
  let pr = vec3f(inh(k, 2u), inh(k, 3u), inh(k, 4u));
  let qr = vec4f(inh(k, 5u), inh(k, 6u), inh(k, 7u), inh(k, 8u));
  let dv = vec3f(inh(k, 9u), inh(k, 10u), inh(k, 11u));
  let pn = B(par, F_P).xyz; let qn = B(par, F_Q);
  let vn = S(par, S_V).xyz; let wn = S(par, S_W).xyz;
  let rd = normalize(qmul(qn, vec4f(-qr.xyz, qr.w)));
  let pk = B(kid, F_P); let qk = B(kid, F_Q);
  let p = pn + qrot(rd, pk.xyz - pr);
  let q = normalize(qmul(rd, qk));
  setB(kid, F_P, vec4f(p, pk.w)); setB(kid, F_Q, q);
  setB(kid, F_FL, vec4f(0.0, p)); setB(kid, F_RQ, q);
  if (absolute) { setS(kid, S_V, vec4f(dv, S(kid, S_V).w)); return; }
  var v = vn + cross(wn, p - pn);
  if (noRebound) { let n = dv; let a = dot(v, n); if (a > 0.0) { v -= a * n; } }
  else { v += qrot(rd, dv); }
  setS(kid, S_V, vec4f(v, S(kid, S_V).w)); setS(kid, S_W, vec4f(wn, 0.0));
}
@compute @workgroup_size(64) fn retire(@builtin(global_invocation_id) g: vec3u) {
  let k = g.x; if (k >= P.n_inherit) { return; }
  let kid = ld(P.o_inherit + k * 12u); let par = ld(P.o_inherit + k * 12u + 1u) & 0x3fffffffu;
  if (kid != 0u || par == 0u) { return; }
  setB(par, F_P, vec4f(0.0, -1000.0, 0.0, 0.0)); setB(par, F_FL, vec4f(1.0, 0.0, -1000.0, 0.0));
  setS(par, S_V, vec4f(0.0)); setS(par, S_W, vec4f(0.0));
  st(P.o_plan + par * SN + 2u, 0u);
}

// wake every sleeping body (the static structure under them changed)
@compute @workgroup_size(64) fn wake_all(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + 1u; if (i >= P.n1) { return; }
  if (B(i, F_P).w == 0.0) { return; }
  wake(i);
}

// Two kernels: boxes here, and every pair with a convex chunk in narrow_hull. The chunk test keeps both hulls'
// corners, directions and up to 16 clipped points per thread; in one kernel with the boxes that made every
// thread pay for them (the box-only pile went from 59 to 260 ms of narrow phase on the software GPU when chunks
// came in). Pairs with a chunk are listed here and tested there, over exactly that list.
fn pair_wake_emit(si: u32, sj: u32, bi: u32, bj: u32, vi: vec3f, ri: f32, rj: f32, c: Contact) {
  if (c.cnt == 0u) { return; }
  // a moving body wakes a sleeping one it touches (it joins the solver next step) when it moves enough to
  // matter: a piece just woken (at rest) or creeping does not wake its neighbours, so a pile wakes only as far
  // as the disturbance really goes
  let spi = length(vi) + length(S(bi, S_W).xyz) * ri; let spj = length(S(bj, S_V).xyz) + length(S(bj, S_W).xyz) * rj;
  if (dyn_awake(bi) && spi > P.wake_speed) { wake(bj); }
  if (dyn_awake(bj) && spj > P.wake_speed) { wake(bi); }
  emit(si, sj, bi, bj, c);
}
// the ground margin covers how far a corner can come down in a step: the fall and the spin (a chunk spinning
// at 10 rad/s swung its corners 8-15 cm into the ground before the contact was there)
fn ground_margin(si: u32, bi: u32) -> f32 {
  return P.spec + (max(-S(bi, S_V).y, 0.0) + length(S(bi, S_W).xyz) * (length(Hs(si, H_LP).xyz) + length(Hs(si, H_H).xyz))) * P.dt;
}
fn pair_margin(si: u32, sj: u32, bi: u32, bj: u32) -> vec3f {   // (margin, ri, rj)
  let ri = length(Hs(si, H_LP).xyz) + length(Hs(si, H_H).xyz); let rj = length(Hs(sj, H_LP).xyz) + length(Hs(sj, H_H).xyz);
  let rel = length(S(bj, S_V).xyz - S(bi, S_V).xyz) + length(S(bi, S_W).xyz) * ri + length(S(bj, S_W).xyz) * rj;
  return vec3f(P.spec + rel * P.dt, ri, rj);
}
// (a grid-stride loop: with direct dispatches it is sized by the recent pair count, not by the capacity; 24
// pairs per shape made the capacity 3x larger and the box pile's narrow phase 57 -> 131 ms on the software
// GPU, all threads with nothing to do)
@compute @workgroup_size(64) fn narrow(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let n = min(ld(C_NPAIR), P.max_pairs);
  for (var k = g.x; k < n; k += nw.x * 64u) { narrow_box(k); }
}
fn narrow_box(k: u32) {
  let si = ld(P.o_pairs + 2u * k); let sj = ld(P.o_pairs + 2u * k + 1u);
  let hull = Hs(si, H_ID).w != 0.0 || (sj != 0u && Hs(sj, H_ID).w != 0.0);
  if (hull) { let h = atomicAdd(&ints[P.o_hl], 1u); st(P.o_hl + 1u + h, k); return; }
  let bi = sbody(si);
  if (sj == 0u) {
    let gc = ground_contact(Hs(si, H_WP).xyz, Hs(si, H_WQ), Hs(si, H_H).xyz, ground_margin(si, bi));
    if (gc.cnt > 0u) { emit(si, 0u, bi, 0u, gc); }
    return;
  }
  atomicAdd(&ints[C_PAIRS], 1u);
  let bj = sbody(sj);
  let m = pair_margin(si, sj, bi, bj);
  let c = box_box(Hs(si, H_WP).xyz, qmat(Hs(si, H_WQ)), Hs(si, H_H).xyz, Hs(sj, H_WP).xyz, qmat(Hs(sj, H_WQ)), Hs(sj, H_H).xyz, m.x);
  pair_wake_emit(si, sj, bi, bj, S(bi, S_V).xyz, m.y, m.z, c);
}
@compute @workgroup_size(64) fn narrow_hull(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= min(ld(P.o_hl), P.max_pairs)) { return; }
  let k = ld(P.o_hl + 1u + g.x);
  let si = ld(P.o_pairs + 2u * k); let sj = ld(P.o_pairs + 2u * k + 1u);
  let bi = sbody(si);
  let ti = u32(Hs(si, H_ID).w);
  if (sj == 0u) {
    var gc = hull_ground(HShape(Hs(si, H_WP).xyz, qmat(Hs(si, H_WQ)), Hs(si, H_H).xyz, ti), ground_margin(si, bi));
    // a chunk's corners on the ground: up to 8, as two contacts of 4 (one patch of 4 left the others free to
    // swing into the ground); the second is keyed as against ground "shape" 0x7fffffff
    if (gc.cnt > 4u) {
      var c2: Contact; c2.n = gc.n; c2.cnt = gc.cnt - 4u;
      for (var q = 0u; q < c2.cnt; q++) { c2.pts[q] = gc.pts[4u + q]; }
      gc.cnt = 4u;
      emit(si, 0x7fffffffu, bi, 0u, c2);
    }
    if (gc.cnt > 0u) { emit(si, 0u, bi, 0u, gc); }
    return;
  }
  atomicAdd(&ints[C_PAIRS], 1u);
  let bj = sbody(sj);
  let m = pair_margin(si, sj, bi, bj);
  let tj = u32(Hs(sj, H_ID).w);
  let c = hull_hull(HShape(Hs(si, H_WP).xyz, qmat(Hs(si, H_WQ)), Hs(si, H_H).xyz, ti), HShape(Hs(sj, H_WP).xyz, qmat(Hs(sj, H_WQ)), Hs(sj, H_H).xyz, tj), m.x);
  pair_wake_emit(si, sj, bi, bj, S(bi, S_V).xyz, m.y, m.z, c);
}

// ---------------------------------------------------------------- contact reduction
// Two bodies touching through many shape pairs (the upper part of a building on a floor: hundreds of brick
// contacts) keep only representative ones: per body pair, the contacts furthest out in 26 directions (the
// corners, edges and faces of a cube) and the deepest. That supports the body at its extremes with a small set the solver can handle
// one by one; with all of them, most went to the parallel overflow pass, which did not take back the push-out
// (a large body landing on a brick floor rebounded at 1.1 m/s where concrete gives 0.59). A dropped contact
// that starts to sink becomes the deepest and is kept the next step.
const RED_K = 32u;   // pairs with more contacts than this are reduced; slots: 26 directions + the deepest
const RED_D = 26u;
const C_NBIG = 14u; const RED_NBIG = 4096u;
fn f2o(x: f32) -> u32 { let b = bitcast<u32>(x); return select(b | 0x80000000u, ~b, (b & 0x80000000u) != 0u); }
// the 26 directions to the neighbours of a cube cell (corners, edges, faces)
fn red_dir(k: u32) -> vec3f {
  let j = select(k, k + 1u, k >= 13u);   // skip the centre (13)
  return vec3f(f32(j % 3u) - 1.0, f32((j / 3u) % 3u) - 1.0, f32(j / 9u) - 1.0);
}
// (the pair's slot in the table, or none)
fn red_slot(mi: u32) -> u32 { return ld(P.o_mslot + mi); }
@compute @workgroup_size(64) fn red_count(@builtin(global_invocation_id) g: vec3u) {
  let mi = g.x; if (mi >= min(ld(C_MAN), P.max_man)) { return; }
  let hd = man[(P.cur_base + mi) * MS]; let a = u32(hd.x); let b = u32(hd.y);
  st(P.o_mslot + mi, 0xffffffffu);
  if (u32(hd.z) == 0u) { return; }
  let key = (min(a, b) | (max(a, b) << 16u)) + 1u;
  // (mixed: key * constant keeps only the low bits of the lower body in the low bits of the slot, and every pair
  // with the static building (body 1) started from one slot. Chains past the 64 probes left contacts uncounted,
  // which ones by thread order, and the building's pairs were reduced in one run and not in the next: lab
  // collapse, frames 91-102)
  var h = key; h ^= h >> 16u; h *= 0x7feb352du; h ^= h >> 15u; h *= 0x846ca68bu; h ^= h >> 16u;
  h &= P.ptab_mask;
  for (var t = 0u; t < 64u; t++) {
    let o = P.o_ptab + 3u * h;
    let r = atomicCompareExchangeWeak(&ints[o], 0u, key);
    if (r.exchanged || r.old_value == key) { atomicAdd(&ints[o + 1u], 1u); st(P.o_mslot + mi, h); return; }
    if (r.old_value != 0u) { h = (h + 1u) & P.ptab_mask; }
  }
}
// Each pair to reduce gets its place once, by a thread per table slot. (Handed out by its contacts as they came, a
// pair with hundreds of them took hundreds of places: past the 4,096 the ones that came last went unreduced, and
// which those were changed run to run; lab collapse, frame 91.)
@compute @workgroup_size(64) fn red_big(@builtin(global_invocation_id) g: vec3u) {
  let h = g.x; if (h > P.ptab_mask) { return; }
  let o = P.o_ptab + 3u * h;
  if (ld(o) == 0u || ld(o + 1u) <= RED_K) { return; }
  st(o + 2u, atomicAdd(&ints[C_NBIG], 1u) + 1u);
}
fn man_centre(mi: u32) -> vec4f {
  let base = (P.cur_base + mi) * MS; let hd = man[base]; let cnt = u32(hd.z);
  let pa = B(u32(hd.x), F_P).xyz;
  var c = vec3f(0.0); var smin = 1e9;
  for (var k = 0u; k < cnt; k++) { let q = man[base + M_PT + 2u * k]; c += q.xyz; smin = min(smin, q.w); }
  return vec4f(pa + c / f32(max(cnt, 1u)), smin);
}
@compute @workgroup_size(64) fn red_score(@builtin(global_invocation_id) g: vec3u) {
  let mi = g.x; if (mi >= min(ld(C_MAN), P.max_man)) { return; }
  let h = red_slot(mi); if (h == 0xffffffffu) { return; }
  let o = P.o_ptab + 3u * h;
  if (ld(o + 1u) <= RED_K) { return; }
  // this pair's place among the reduced ones (red_big)
  let big = ld(o + 2u);
  if (big == 0u || big > RED_NBIG) { return; }
  let c = man_centre(mi); let bo = P.o_big + (big - 1u) * 64u;
  for (var k = 0u; k < RED_D; k++) { atomicMax(&ints[bo + k], f2o(dot(c.xyz, red_dir(k)))); }
  atomicMax(&ints[bo + RED_D], f2o(-c.w));
}
@compute @workgroup_size(64) fn red_pick(@builtin(global_invocation_id) g: vec3u) {
  let mi = g.x; if (mi >= min(ld(C_MAN), P.max_man)) { return; }
  let h = red_slot(mi); if (h == 0xffffffffu) { return; }
  let big = ld(P.o_ptab + 3u * h + 2u); if (big == 0u || big > RED_NBIG) { return; }
  let c = man_centre(mi); let bo = P.o_big + (big - 1u) * 64u;
  // (ties go to the higher key: by slot, they went to whichever contact narrow happened to place later)
  let key = mkey(P.cur_base + mi);
  for (var k = 0u; k < RED_D; k++) { if (f2o(dot(c.xyz, red_dir(k))) == ld(bo + k)) { atomicMax(&ints[bo + 32u + k], key); } }
  if (f2o(-c.w) == ld(bo + RED_D)) { atomicMax(&ints[bo + 32u + RED_D], key); }
}
@compute @workgroup_size(64) fn red_apply(@builtin(global_invocation_id) g: vec3u) {
  let mi = g.x; if (mi >= min(ld(C_MAN), P.max_man)) { return; }
  let h = red_slot(mi); if (h == 0xffffffffu) { return; }
  let big = ld(P.o_ptab + 3u * h + 2u); if (big == 0u || big > RED_NBIG) { return; }
  let bo = P.o_big + (big - 1u) * 64u;
  let key = mkey(P.cur_base + mi);
  for (var k = 32u; k <= 32u + RED_D; k++) { if (ld(bo + k) == key) { return; } }
  // not kept: marked (+8 on the point count) so the colouring and the solver skip it, while its points and
  // impulses stay for the warm start of the next step (where it may be kept again)
  let base = (P.cur_base + mi) * MS; let hd = man[base];
  man[base] = vec4f(hd.x, hd.y, hd.z + 8.0, hd.w);
}

// ---------------------------------------------------------------- colouring
fn mhdr(m: u32) -> vec2u { let h = man[m * MS]; return vec2u(u32(h.x), u32(h.y)); }
// Contacts with the static world (or a resting body) are coloured first, in their own pass: they hold the
// bodies up, and a body's supports left to the parallel overflow pass (a slab under hundreds of pieces) did not
// converge and pushed it up (GPU test stumps).
// Determinism: colours are claimed in rounds, not first come first served. Each round, a contact not yet coloured
// picks a colour free at both its bodies (one of the lowest few, by its key and the round) and claims it at each
// with its priority; it takes the colour where it is the strongest claim at both. Contacts with the static world
// or a resting body (supports) claim first: they hold the bodies up, and a body's supports left to the parallel
// overflow pass (a slab under hundreds of pieces) did not converge and pushed it up (GPU test stumps). What is
// not coloured after the last round, or finds no colour free, goes to the overflow pass. A claim holds the round
// in its top bits, so the last round's claims need no clearing.
// (supports alone first, then everything: with supports only first in each claim, the other contacts at a body
// took colours its supports then could not. The rounds are set in JS, colRoundsS and colRounds)
const COL_ROUNDS = 12u; const UNC = 0xfffffffeu;
fn col_pri(mi: u32) -> u32 {
  let ab = mhdr(P.cur_base + mi);
  let support = !dyn_awake(ab.x) || !dyn_awake(ab.y);
  // 26 bits: support first, then the key
  return select(0x2000000u, 0u, support) | (mkey(P.cur_base + mi) >> 7u);
}
@compute @workgroup_size(64) fn color_init(@builtin(global_invocation_id) g: vec3u) {
  let mi = g.x; if (mi >= min(ld(C_MAN), P.max_man)) { return; }
  let reduced = u32(man[(P.cur_base + mi) * MS].z) >= 8u;
  st(P.o_mcolor + mi, select(UNC, 0xffffffffu, reduced));
  st(P.o_mtry + mi, 0u);
  if (mi == 0u) { st(P.o_mtry + P.max_man, 0u); }
}
// the round, for everyone (claims are tagged with it: a claim left from an earlier round never blocks a new one)
@compute @workgroup_size(1) fn color_round() { st(P.o_mtry + P.max_man, ld(P.o_mtry + P.max_man) + 1u); }
@compute @workgroup_size(64) fn color_claim_s(@builtin(global_invocation_id) g: vec3u) { claim_one(g.x, true); }
@compute @workgroup_size(64) fn color_claim(@builtin(global_invocation_id) g: vec3u) { claim_one(g.x, false); }
fn claim_one(mi: u32, support_only: bool) {
  if (mi >= min(ld(C_MAN), P.max_man)) { return; }
  if (ld(P.o_mcolor + mi) != UNC) { return; }
  let ab = mhdr(P.cur_base + mi); let a = ab.x; let b = ab.y;
  let da = dyn_awake(a); let db = dyn_awake(b);
  if (support_only && da && db) { return; }
  let round = ld(P.o_mtry + mi) & 0xffffu;
  let gr = ld(P.o_mtry + P.max_man);
  var used = 0u;
  if (da) { used |= ld(P.o_mask + a); } if (db) { used |= ld(P.o_mask + b); }
  let all = select((1u << P.color_limit) - 1u, 0xffffffffu, P.color_limit >= 32u);
  var free = all & ~used;
  if (free == 0u || round >= COL_ROUNDS) { st(P.o_mcolor + mi, P.color_limit); return; }
  // one of the lowest four free colours, by key and round
  let pick = ((mkey(P.cur_base + mi) >> 3u) + round * 0x9E37u) % min(countOneBits(free), 4u);
  for (var k = 0u; k < pick; k++) { free &= free - 1u; }
  let c = firstTrailingBit(free);
  let v = ((gr + 1u) << 27u) | (0x3ffffffu - col_pri(mi));
  if (da) { atomicMax(&ints[P.o_claim + a * 32u + c], v); }
  if (db) { atomicMax(&ints[P.o_claim + b * 32u + c], v); }
  st(P.o_mtry + mi, (round + 1u) | (c << 16u) | 0x80000000u);
}
// Two contacts at a body with the same 26-bit priority would both take the colour there and be solved at once
// on the same body (in a pile of 600 boxes, about a one-in-ten chance over 90 frames). The ones that tie claim
// again with 27 more bits of a second hash.
fn mkey2(m: u32) -> u32 {
  let sa = u32(cold[m * CS + 4u].w); let sb = u32(cold[m * CS + 5u].w);
  var h = (sb * 0x2545F491u) ^ ((sa + 0x68E31DA4u) * 0xB5297A4Du);
  h ^= h >> 13u; h *= 0x1B873593u; h ^= h >> 16u;
  return h;
}
fn claimed(mi: u32, stage2: bool) -> bool {
  let t = ld(P.o_mtry + mi); let c = (t >> 16u) & 0x7fffu;
  let ab = mhdr(P.cur_base + mi); let a = ab.x; let b = ab.y;
  let da = dyn_awake(a); let db = dyn_awake(b);
  let tag = (ld(P.o_mtry + P.max_man) + 1u) << 27u;
  let v = tag | (0x3ffffffu - col_pri(mi));
  if (da && ld(P.o_claim + a * 32u + c) != v) { return false; }
  if (db && ld(P.o_claim + b * 32u + c) != v) { return false; }
  let v2 = tag | (mkey2(P.cur_base + mi) >> 5u);
  if (!stage2) {
    if (da) { atomicMax(&ints[P.o_claim2 + a * 32u + c], v2); }
    if (db) { atomicMax(&ints[P.o_claim2 + b * 32u + c], v2); }
    return true;
  }
  if (da && ld(P.o_claim2 + a * 32u + c) != v2) { return false; }
  if (db && ld(P.o_claim2 + b * 32u + c) != v2) { return false; }
  return true;
}
@compute @workgroup_size(64) fn color_tie(@builtin(global_invocation_id) g: vec3u) {
  let mi = g.x; if (mi >= min(ld(C_MAN), P.max_man)) { return; }
  if (ld(P.o_mcolor + mi) != UNC || (ld(P.o_mtry + mi) & 0x80000000u) == 0u) { return; }
  _ = claimed(mi, false);
}
@compute @workgroup_size(64) fn color_resolve(@builtin(global_invocation_id) g: vec3u) {
  let mi = g.x; if (mi >= min(ld(C_MAN), P.max_man)) { return; }
  if (ld(P.o_mcolor + mi) != UNC) { return; }
  let t = ld(P.o_mtry + mi); if ((t & 0x80000000u) == 0u) { return; }   // (it made no claim this round)
  let won = claimed(mi, true);
  st(P.o_mtry + mi, t & 0x7fffffffu);
  if (!won) { return; }
  let c = (t >> 16u) & 0x7fffu;
  let ab = mhdr(P.cur_base + mi); let a = ab.x; let b = ab.y;
  let da = dyn_awake(a); let db = dyn_awake(b);
  st(P.o_mcolor + mi, c);
  // (the winner at a body and colour is the only one: no other contact there takes this colour this round)
  if (da) { atomicOr(&ints[P.o_mask + a], 1u << c); }
  if (db) { atomicOr(&ints[P.o_mask + b], 1u << c); }
}
@compute @workgroup_size(64) fn color_count(@builtin(global_invocation_id) g: vec3u) {
  let mi = g.x; if (mi >= min(ld(C_MAN), P.max_man)) { return; }
  var col = ld(P.o_mcolor + mi);
  if (col == 0xffffffffu) { return; }
  if (col == UNC) { col = P.color_limit; st(P.o_mcolor + mi, col); }
  let ab = mhdr(P.cur_base + mi); let a = ab.x; let b = ab.y;
  let da = dyn_awake(a); let db = dyn_awake(b);
  atomicAdd(&ints[P.o_ccount + col], 1u);
  if (col == P.color_limit) {
    atomicAdd(&ints[C_OVF], 1u);
    if (da) { atomicAdd(&ints[P.o_ocount + a], 1u); }
    if (db) { atomicAdd(&ints[P.o_ocount + b], 1u); }
  } else { atomicMax(&ints[C_COLORS], col + 1u); }
  if (da) { atomicAdd(&ints[P.o_bcount + a], 1u); }
  if (db) { atomicAdd(&ints[P.o_bcount + b], 1u); }
}
// The overflow contacts in key order: their restitution goes through them one by one (restitute_ovf), and in the
// order color_scatter's atomics left them it differed run to run (GPU test determinism, scene stumps). A bitonic
// sort in one workgroup; past the end counts as the largest, so every compare puts the smaller first.
var<workgroup> wn: u32;
fn okey_less(x: u32, y: u32) -> bool { let kx = mkey(x); let ky = mkey(y); return kx < ky || (kx == ky && x < y); }
@compute @workgroup_size(256) fn ovf_sort(@builtin(local_invocation_index) t: u32) {
  if (t == 0u) { let r = ovf_range(); wn = r.y - r.x; }
  let n = workgroupUniformLoad(&wn);
  let o = P.o_order + ld(P.o_cstart + P.color_limit);
  var k = 2u;
  loop {
    if (k >= 2u * n) { break; }
    var j = k >> 1u;
    loop {
      if (j == 0u) { break; }
      for (var i = t; i < n; i += 256u) {
        let l = select(i ^ j, i ^ (k - 1u), j == (k >> 1u));
        if (l > i && l < n) {
          let x = ld(o + i); let y = ld(o + l);
          if (okey_less(y, x)) { st(o + i, y); st(o + l, x); }
        }
      }
      storageBarrier(); workgroupBarrier();
      j >>= 1u;
    }
    k <<= 1u;
  }
}
// (debugging: two contacts of one colour on the same moving body, counted at o_cchk[0]; bit 32 of dbg)
@compute @workgroup_size(64) fn color_check(@builtin(global_invocation_id) g: vec3u) {
  let mi = g.x; if (mi >= min(ld(C_MAN), P.max_man)) { return; }
  let col = ld(P.o_mcolor + mi); if (col >= P.color_limit) { return; }
  let ab = mhdr(P.cur_base + mi);
  for (var k = 0u; k < 2u; k++) {
    let i = select(ab.y, ab.x, k == 0u);
    if (dyn_awake(i) && atomicAdd(&ints[P.o_cchk + 1u + i * 32u + col], 1u) > 0u) { atomicAdd(&ints[P.o_cchk], 1u); }
  }
}
// a body's contacts in key order (filled by atomics): its warm start sums them in that order every run
@compute @workgroup_size(64) fn blist_sort(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + 1u; if (i >= P.n1) { return; }
  let s0 = ld(P.o_bstart + i); let n = ld(P.o_bcount + i);
  for (var a = 1u; a < n; a++) {
    let e = ld(P.o_blist + s0 + a); let ke = mkey(e >> 1u); var b = a;
    loop {
      if (b == 0u) { break; }
      let f = ld(P.o_blist + s0 + b - 1u); let kf = mkey(f >> 1u);
      if (kf < ke || (kf == ke && f <= e)) { break; }
      st(P.o_blist + s0 + b, f); b--;
    }
    st(P.o_blist + s0 + b, e);
  }
}
@compute @workgroup_size(64) fn color_scatter(@builtin(global_invocation_id) g: vec3u) {
  let mi = g.x; if (mi >= min(ld(C_MAN), P.max_man)) { return; }
  let c = ld(P.o_mcolor + mi); if (c == 0xffffffffu) { return; }
  let m = P.cur_base + mi;
  st(P.o_order + ld(P.o_cstart + c) + atomicAdd(&ints[P.o_cfill + c], 1u), m);
  let ab = mhdr(m);
  if (dyn_awake(ab.x)) { st(P.o_blist + ld(P.o_bstart + ab.x) + atomicAdd(&ints[P.o_bfill + ab.x], 1u), m * 2u); }
  if (dyn_awake(ab.y)) { st(P.o_blist + ld(P.o_bstart + ab.y) + atomicAdd(&ints[P.o_bfill + ab.y], 1u), m * 2u + 1u); }
}
@compute @workgroup_size(256) fn adj_scan(@builtin(local_invocation_index) t: u32) {
  let per = (P.n1 + 255u) / 256u; let base = t * per;
  var s = 0u;
  for (var k = 0u; k < per; k++) { if (base + k < P.n1) { s += ld(P.o_bcount + base + k); } }
  var run = wg_scan(t, s);
  for (var k = 0u; k < per; k++) { if (base + k < P.n1) { st(P.o_bstart + base + k, run); run += ld(P.o_bcount + base + k); } }
}

// ---------------------------------------------------------------- solver
// gravity, spin clamp, then the warm start gathered per body (each thread owns its body: no colouring)
@compute @workgroup_size(64) fn integrate_warm(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + 1u; if (i >= P.n1) { return; }
  let v4 = S(i, S_V); if (v4.w == 0.0) { return; }
  var v = v4.xyz + P.gravity * P.h; var w = S(i, S_W).xyz;
  let wl = length(w); if (wl > P.wmax) { w *= P.wmax / wl; }
  let I0 = S(i, S_I0); let I1 = S(i, S_I1);
  var dv = vec3f(0.0); var dl = vec3f(0.0);
  let s0 = ld(P.o_bstart + i); let s1 = s0 + ld(P.o_bcount + i);
  for (var k = s0; k < s1; k++) {
    let e = ld(P.o_blist + k); let m = e >> 1u; let side_b = (e & 1u) != 0u;
    let base = m * MS;
    let cnt = u32(man[base].z) & 7u;
    let n = man[base + M_N].xyz; let tt = tangents(n);
    let jn = man[base + M_JN]; let jt = man[base + M_JT];
    let off = select(0u, 1u, side_b);
    var pn = 0.0; var rj = vec3f(0.0);
    for (var q = 0u; q < cnt; q++) { pn += jn[q]; rj += man[base + M_PT + 2u * q + off].xyz * jn[q]; }
    let arm = man[base + select(M_FA, M_FB, side_b)].xyz;
    let ft = tt[0] * jt.x + tt[1] * jt.y;
    let p = n * pn + ft;
    let l = cross(rj, n) + cross(arm, ft) + n * jt.z;   // angular impulse on side b; side a gets minus
    if (side_b) { dv += p; dl += l; } else { dv -= p; dl -= l; }
  }
  v += dv * v4.w; w += imul(I0, I1, dl);
  setS(i, S_V, vec4f(v, v4.w)); setS(i, S_W, vec4f(w, 0.0));
}
@compute @workgroup_size(64) fn integrate_pos(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + 1u; if (i >= P.n1) { return; }
  let v4 = S(i, S_V); if (v4.w == 0.0) { return; }
  let w = S(i, S_W).xyz;
  setS(i, S_DP, vec4f(S(i, S_DP).xyz + v4.xyz * P.h, 0.0));
  setS(i, S_TH, vec4f(S(i, S_TH).xyz + w * P.h, 0.0));
  let q = B(i, F_Q);
  setB(i, F_Q, normalize(q + 0.5 * P.h * qmul(vec4f(w, 0.0), q)));
}

// split = an overflow contact solved in parallel with others on the same bodies (Jacobi with mass
// splitting): each body acts as if it had 1/k of its mass for its k overflow contacts, and the velocity
// changes are summed atomically and averaged afterwards (ovf_apply).
// Mass ratio limit: across one contact the heavier body responds as if at most P_RATIO times heavier. A block of
// thousands of tonnes on light rubble (1:600) does not converge in a few iterations: the rubble is squeezed and
// the overlap comes back as a rebound of the whole block. (Momentum is not kept exactly there; the energy goes
// down, as in crushing.) Returns the factors for the inverse masses of a and b.
const P_RATIO = 20.0;
fn ratio_scale(ia: f32, ib: f32) -> vec2f {
  if (ia <= 0.0 || ib <= 0.0) { return vec2f(1.0); }
  if (ia * P_RATIO < ib) { return vec2f(ib / (P_RATIO * ia), 1.0); }
  if (ib * P_RATIO < ia) { return vec2f(1.0, ia / (P_RATIO * ib)); }
  return vec2f(1.0);
}
// The limit may only take energy from the heavy body, never give it: its response at the limited mass (sc
// times its real one) is kept where that slowed it down; where it sped it up, the heavy body takes the change
// at its real mass. Applied both ways, the limit made momentum: a slab landing on stumps with pieces on it
// rose at 8 m/s and threw the pieces at 15 (GPU test stumps); in the lab collapse the upper part of the
// building rose 0.5 m after landing.
fn limited(v0: vec3f, w0: vec3f, v: ptr<function, vec3f>, w: ptr<function, vec3f>, sc: f32) {
  if (sc == 1.0 || (P.dbg & 2u) != 0u) { return; }
  *v = v0 + slowing(v0, *v - v0, sc); *w = w0 + slowing(w0, *w - w0, sc);
}
// the part of a change d to x that brings |x| down to its least is kept; the rest is divided by sc (all or
// nothing lost the slowing too when a change overshot zero: a block landing on rubble sank and rebounded at
// 1.9 m/s, GPU test sandwich)
fn slowing(x: vec3f, d: vec3f, sc: f32) -> vec3f {
  let dd = dot(d, d); if (dd == 0.0) { return d; }
  let t = clamp(-dot(x, d) / dd, 0.0, 1.0);
  return d * (t + (1.0 - t) / sc);
}
// Shock propagation in the relax pass. A light piece on the ground under a heavy body is a sandwich the
// Gauss-Seidel pass does not settle: each sub-step's push out of the ground (the soft contact's bias) goes up
// through the light piece into the heavy body, and the relax pass, which should take it back, gives back only
// about 1/21 of it per iteration at 20:1 (GPU test pinchdbg: a 960 kg slab landing at 7.7 m/s on a 47 kg piece
// rebounded at 3.6 m/s; 8 relax iterations, 1.4). Each body's level is its number of contacts above the static
// world (the static world and sleeping bodies 0, a body on them 1, ...), found from the contacts whose normal
// is mostly vertical. In the relax pass, the lower body of such a contact does not respond: the upper one alone
// takes the impulse, as it would once the pass had converged with the lower one held by what is under it. The
// biased solve and restitution are unchanged.
// (stored as 255 - level, raised with atomicMax; each pass reads one copy and writes the other, so the result
// does not depend on the order threads run in: read and written in one place, a chain could climb several
// levels in one pass or one, as the threads fell, and runs differed. An even number of passes ends in o_level.)
fn height(i: u32) -> u32 { if (!dyn_awake(i)) { return 255u; } return ld(P.o_level + i); }
fn height_in(o: u32, i: u32) -> u32 { if (!dyn_awake(i)) { return 255u; } return ld(o + i); }
@compute @workgroup_size(64) fn level_pass(@builtin(global_invocation_id) g: vec3u) {
  let mi = g.x; if (mi >= min(ld(C_MAN), P.max_man)) { return; }
  let base = (P.cur_base + mi) * MS; let hd = man[base];
  if ((u32(hd.z) & 7u) == 0u) { return; }
  let rd = select(P.o_level, P.o_level2, (CU.c & 1u) == 1u); let wr = select(P.o_level2, P.o_level, (CU.c & 1u) == 1u);
  let a = u32(hd.x); let b = u32(hd.y); let ny = man[base + M_N].y;
  let ha = height_in(rd, a); let hb = height_in(rd, b);
  atomicMax(&ints[wr + a], ld(rd + a)); atomicMax(&ints[wr + b], ld(rd + b));
  // (the normal points from a to b: n.y > 0, b rests on a)
  if (ny > 0.7 && ha > 1u && dyn_awake(b)) { atomicMax(&ints[wr + b], ha - 1u); }
  if (ny < -0.7 && hb > 1u && dyn_awake(a)) { atomicMax(&ints[wr + a], hb - 1u); }
}
fn solve_man(m: u32, use_bias: bool, split: bool) {
  let base = m * MS;
  let hd = man[base]; let a = u32(hd.x); let b = u32(hd.y); let cnt = u32(hd.z);
  let va4 = S(a, S_V); let vb4 = S(b, S_V);
  var va = va4.xyz; var wa = S(a, S_W).xyz; var vb = vb4.xyz; var wb = S(b, S_W).xyz;
  let va0 = va; let wa0 = wa; let vb0 = vb; let wb0 = wb;
  var ka = 1.0; var kb = 1.0;
  if (split) { ka = f32(max(ld(P.o_ocount + a), 1u)); kb = f32(max(ld(P.o_ocount + b), 1u)); }
  // (the ratio is limited between the masses as split: limited first, a body split over k contacts acted as
  // 20/k times the light body, lighter than it; a slab landing with 100 pieces on it was thrown up at 8 m/s,
  // GPU test stumps)
  let sc = select(ratio_scale(va4.w * ka, vb4.w * kb), ratio_scale(va4.w, vb4.w), (P.dbg & 1u) != 0u);
  // (shock propagation, relax only: where the pass takes back a push, the lower body of a vertical contact is
  // held, see level_pass)
  var holdA = false; var holdB = false;
  if (!use_bias && (P.dbg & 16u) == 0u) {   // (bit 16: off, GpuWorld.shock = false)
    let ny = man[base + M_N].y; let la = height(a); let lb = height(b);
    holdA = ny > 0.7 && la > lb; holdB = ny < -0.7 && lb > la;
  }
  let rec = split || sc.x != 1.0 || sc.y != 1.0;   // effective masses recomputed
  let ima = va4.w * ka * sc.x; let imb = vb4.w * kb * sc.y;
  let A0 = S(a, S_I0) * ka * sc.x; let A1 = S(a, S_I1) * ka * sc.x; let B0 = S(b, S_I0) * kb * sc.y; let B1 = S(b, S_I1) * kb * sc.y;
  let tha = S(a, S_TH).xyz; let thb = S(b, S_TH).xyz;
  let dpab = S(b, S_DP).xyz - S(a, S_DP).xyz;
  let n4 = man[base + M_N]; let n = n4.xyz;
  let dn = dot(dpab, n);
  var jn = man[base + M_JN];
  // (a contact against particles: they bear at most RUBBLE_BEAR over their cross-section, see p_support)
  let sup = man[base + M_OUT].x < 0.0; let jcap = man[base + M_RV];
  var tot = 0.0;
  for (var k = 0u; k < cnt; k++) {
    let pa = man[base + M_PT + 2u * k]; let pb = man[base + M_PT + 2u * k + 1u];
    let raxn = cross(pa.xyz, n); let rbxn = cross(pb.xyz, n);
    let s = pa.w + dn + dot(thb, rbxn) - dot(tha, raxn);
    var bias = 0.0; var ms = 1.0; var is_ = 0.0;
    if (s > 0.0) { bias = s * P.inv_h; }
    else if (use_bias) { bias = max(P.bias_rate * s, -P.max_push * select(1.0, P.ovf_push, split)); ms = P.mass_scale; is_ = P.imp_scale; }
    let vn = dot(vb - va, n) + dot(wb, rbxn) - dot(wa, raxn);
    let j0 = jn[k];
    // (separating, in the relax pass: the push is taken back from the upper body alone)
    let fa = select(1.0, 0.0, holdA && vn > 0.0); let fb = select(1.0, 0.0, holdB && vn > 0.0);
    var nm = pb.w;
    if (rec || fa != 1.0 || fb != 1.0) { let kk = fa * (ima + dot(raxn, imul(A0, A1, raxn))) + fb * (imb + dot(rbxn, imul(B0, B1, rbxn))); nm = select(0.0, 1.0 / kk, kk > 0.0); }
    var nj = max(j0 - nm * ms * (vn + bias) - is_ * j0, 0.0);
    if (sup) { nj = min(nj, jcap[k]); }
    let imp = nj - j0;
    jn[k] = nj; tot += nj;
    va -= n * (imp * ima * fa); wa -= imul(A0, A1, raxn) * (imp * fa);
    vb += n * (imp * imb * fb); wb += imul(B0, B1, rbxn) * (imp * fb);
  }
  man[base + M_JN] = jn;
  // patch friction at the anchor, soft-anchored to where the patch first stuck
  let fa4 = man[base + M_FA]; let fb4 = man[base + M_FB]; let e04 = man[base + M_E0]; let jt = man[base + M_JT];
  let tt = tangents(n); let t1 = tt[0]; let t2 = tt[1];
  let ca1 = cross(fa4.xyz, t1); let ca2 = cross(fa4.xyz, t2); let cb1 = cross(fb4.xyz, t1); let cb2 = cross(fb4.xyz, t2);
  let lim = n4.w * tot;
  let dv = vb - va;
  let v1 = dot(dv, t1) + dot(wb, cb1) - dot(wa, ca1);
  let v2 = dot(dv, t2) + dot(wb, cb2) - dot(wa, ca2);
  var b1 = 0.0; var b2 = 0.0; var ms = 1.0; var is_ = 0.0;
  var stick = hd.w;
  if (use_bias && stick > 0.5) {
    let e1 = dot(e04.xyz, t1) + dot(dpab, t1) + dot(thb, cb1) - dot(tha, ca1);
    let e2 = dot(e04.xyz, t2) + dot(dpab, t2) + dot(thb, cb2) - dot(tha, ca2);
    b1 = clamp(P.bias_rate * e1, -P.max_push, P.max_push); b2 = clamp(P.bias_rate * e2, -P.max_push, P.max_push);
    ms = P.mass_scale; is_ = P.imp_scale;
  }
  let o1 = jt.x; let o2 = jt.y;
  var tm1 = fa4.w; var tm2 = fb4.w; var twm = e04.w;
  if (rec) {
    let k1 = ima + imb + dot(ca1, imul(A0, A1, ca1)) + dot(cb1, imul(B0, B1, cb1));
    let k2 = ima + imb + dot(ca2, imul(A0, A1, ca2)) + dot(cb2, imul(B0, B1, cb2));
    let kt = dot(n, imul(A0, A1, n)) + dot(n, imul(B0, B1, n));
    tm1 = select(0.0, 1.0 / k1, k1 > 0.0); tm2 = select(0.0, 1.0 / k2, k2 > 0.0); twm = select(0.0, 1.0 / kt, kt > 0.0);
  }
  var j1 = o1 - tm1 * ms * (v1 + b1) - is_ * o1;
  var j2 = o2 - tm2 * ms * (v2 + b2) - is_ * o2;
  let l2 = j1 * j1 + j2 * j2;
  if (l2 > lim * lim) { let kk = lim * inverseSqrt(max(l2, 1e-30)); j1 *= kk; j2 *= kk; stick = 0.0; }
  let d1 = j1 - o1; let d2 = j2 - o2;
  let pt = t1 * d1 + t2 * d2;
  va -= pt * ima; wa -= imul(A0, A1, ca1 * d1 + ca2 * d2);
  vb += pt * imb; wb += imul(B0, B1, cb1 * d1 + cb2 * d2);
  // twist
  let wl = lim * jt.w;
  let jtw = clamp(jt.z - twm * dot(wb - wa, n), -wl, wl);
  let dtw = jtw - jt.z;
  wa -= imul(A0, A1, n) * dtw; wb += imul(B0, B1, n) * dtw;
  man[base + M_JT] = vec4f(j1, j2, jtw, jt.w);
  if (stick != hd.w) { man[base] = vec4f(hd.xyz, stick); }
  // the relax pass ends each sub-step: its normal impulses are what the sub-step applied
  if (!use_bias) { let o = man[base + M_OUT]; man[base + M_OUT] = vec4f(o.x, o.y + tot, o.zw); }
  limited(va0, wa0, &va, &wa, sc.x); limited(vb0, wb0, &vb, &wb, sc.y);
  if (split) {
    // (each change divided by the count here, not after summing: a large body has thousands of overflow
    // contacts, and their changes against a share of its mass (several m/s each) can overflow the 32-bit sum)
    if (ima > 0.0) { acc_add(a, (va - va0) / ka, (wa - wa0) / ka); }
    if (imb > 0.0) { acc_add(b, (vb - vb0) / kb, (wb - wb0) / kb); }
    return;
  }
  if (ima > 0.0) { setS(a, S_V, vec4f(va, va4.w)); setS(a, S_W, vec4f(wa, 0.0)); }
  if (imb > 0.0) { setS(b, S_V, vec4f(vb, vb4.w)); setS(b, S_W, vec4f(wb, 0.0)); }
}

// fixed-point accumulation of velocity changes (there are no float atomics): 2^18 per m/s
const ACC = 262144.0;
fn acc_add(i: u32, dv: vec3f, dw: vec3f) {
  let o = P.o_acc + i * 6u;
  for (var k = 0u; k < 3u; k++) {
    atomicAdd(&ints[o + k], bitcast<u32>(i32(round(dv[k] * ACC))));
    atomicAdd(&ints[o + 3u + k], bitcast<u32>(i32(round(dw[k] * ACC))));
  }
}
@compute @workgroup_size(64) fn ovf_apply(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + 1u; if (i >= P.n1) { return; }
  let c = ld(P.o_ocount + i); if (c == 0u) { return; }
  let o = P.o_acc + i * 6u;
  var dv = vec3f(0.0); var dw = vec3f(0.0);
  for (var k = 0u; k < 3u; k++) {
    dv[k] = f32(bitcast<i32>(atomicExchange(&ints[o + k], 0u))) / ACC;
    dw[k] = f32(bitcast<i32>(atomicExchange(&ints[o + 3u + k], 0u))) / ACC;
  }
  let v4 = S(i, S_V); if (v4.w == 0.0) { return; }
  // (the changes were already divided by the count)
  setS(i, S_V, vec4f(v4.xyz + dv, v4.w)); setS(i, S_W, vec4f(S(i, S_W).xyz + dw, 0.0));
}

// restitution after the sub-steps (Box2D v3 style): points that approached faster than the threshold get
// their normal velocity reflected by the pair's restitution
fn restitute_man(m: u32) {
  if ((P.dbg & 8u) != 0u) { return; }
  let base = m * MS;
  // only contacts that pushed during the step (the last sub-step may already be separating)
  let out = man[base + M_OUT]; let rest = out.x; if (rest <= 0.0 || out.y == 0.0) { return; }
  let hd = man[base]; let a = u32(hd.x); let b = u32(hd.y); let cnt = u32(hd.z);
  let va4 = S(a, S_V); let vb4 = S(b, S_V);
  var va = va4.xyz; var wa = S(a, S_W).xyz; var vb = vb4.xyz; var wb = S(b, S_W).xyz;
  let sc = ratio_scale(va4.w, vb4.w); let rec = sc.x != 1.0 || sc.y != 1.0;
  let ima = va4.w * sc.x; let imb = vb4.w * sc.y;
  let A0 = S(a, S_I0) * sc.x; let A1 = S(a, S_I1) * sc.x; let B0 = S(b, S_I0) * sc.y; let B1 = S(b, S_I1) * sc.y;
  let n = man[base + M_N].xyz; let rv = man[base + M_RV];
  var jn = man[base + M_JN];
  // the points of a patch are coupled (one point's bounce changes the others' velocity): iterate
  for (var it = 0u; it < 8u; it++) { for (var k = 0u; k < cnt; k++) {
    if (rv[k] > -P.rest_threshold) { continue; }
    let pa = man[base + M_PT + 2u * k]; let pb = man[base + M_PT + 2u * k + 1u];
    let raxn = cross(pa.xyz, n); let rbxn = cross(pb.xyz, n);
    let vn = dot(vb - va, n) + dot(wb, rbxn) - dot(wa, raxn);
    var nm = pb.w;
    if (rec) { let kk = ima + imb + dot(raxn, imul(A0, A1, raxn)) + dot(rbxn, imul(B0, B1, rbxn)); nm = select(0.0, 1.0 / kk, kk > 0.0); }
    let nj = max(jn[k] - nm * (vn + rest * rv[k]), 0.0);
    let imp = nj - jn[k]; jn[k] = nj;
    va -= n * (imp * ima); wa -= imul(A0, A1, raxn) * imp;
    vb += n * (imp * imb); wb += imul(B0, B1, rbxn) * imp;
  } }
  man[base + M_JN] = jn;
  limited(va4.xyz, S(a, S_W).xyz, &va, &wa, sc.x); limited(vb4.xyz, S(b, S_W).xyz, &vb, &wb, sc.y);
  if (ima > 0.0) { setS(a, S_V, vec4f(va, va4.w)); setS(a, S_W, vec4f(wa, 0.0)); }
  if (imb > 0.0) { setS(b, S_V, vec4f(vb, vb4.w)); setS(b, S_W, vec4f(wb, 0.0)); }
}

fn color_item(g: u32) -> u32 {
  let c = CU.c;
  if (g >= ld(P.o_ccount + c)) { return 0xffffffffu; }
  return ld(P.o_order + ld(P.o_cstart + c) + g);
}
// grid-stride loops: the dispatch is sized from recent counts, and a colour that grew since is still solved
// in full (each thread takes every stride-th contact; contacts of one colour never share a body)
@compute @workgroup_size(64) fn solve(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  var k = g.x; loop { let m = color_item(k); if (m == 0xffffffffu) { break; } solve_man(m, true, false); k += nw.x * 64u; }
}
@compute @workgroup_size(64) fn relax(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  var k = g.x; loop { let m = color_item(k); if (m == 0xffffffffu) { break; } solve_man(m, false, false); k += nw.x * 64u; }
}
@compute @workgroup_size(64) fn restitute(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  var k = g.x; loop { let m = color_item(k); if (m == 0xffffffffu) { break; } restitute_man(m); k += nw.x * 64u; }
}
// overflow contacts (no free colour): one thread, in sequence
fn ovf_range() -> vec2u { let c = P.color_limit; let s = ld(P.o_cstart + c); return vec2u(s, s + ld(P.o_ccount + c)); }
// overflow contacts (a body with more contacts than colours, e.g. a big falling piece): all in parallel,
// with mass splitting; ovf_apply then averages the velocity changes per body
fn ovf_item(g: u32) -> u32 { let r = ovf_range(); if (r.x + g >= r.y) { return 0xffffffffu; } return ld(P.o_order + r.x + g); }
// (grid-stride, sized by the recent overflow count like the colours; sized by the contact capacity every
// round of every sub-step ran a thread per possible contact)
@compute @workgroup_size(64) fn solve_ovf(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  var k = g.x; loop { let m = ovf_item(k); if (m == 0xffffffffu) { break; } solve_man(m, true, true); k += nw.x * 64u; }
}
@compute @workgroup_size(64) fn relax_ovf(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  var k = g.x; loop { let m = ovf_item(k); if (m == 0xffffffffu) { break; } solve_man(m, false, true); k += nw.x * 64u; }
}
@compute @workgroup_size(1) fn restitute_ovf(@builtin(workgroup_id) wg: vec3u) {
  if (wg.x != 0u) { return; }
  let r = ovf_range(); for (var k = r.x; k < r.y; k++) { restitute_man(ld(P.o_order + k)); }
}

// ---------------------------------------------------------------- end of step, sleeping
// hard impacts this step, for the damage model: [body a, body b, shape a, shape b, point xyz, normal xyz,
// impulse (N s), approach speed (m/s)] as raw words
// Determinism: when there are more than the list holds, the ones kept were whoever got there first. Now they are
// counted by energy (in eighths of a doubling over the threshold), the lowest bands that do not fit are left
// out whole (ev_cut), and the rest written.
fn ev_band(e: f32) -> u32 { return u32(clamp(log2(max(e / P.ev_energy, 1.0)) * 8.0, 0.0, 127.0)); }
fn ev_energy_of(mi: u32) -> f32 {
  let out = man[(P.cur_base + mi) * MS + M_OUT];
  if (out.z < P.ev_speed || 0.5 * out.y * out.z < P.ev_energy) { return -1.0; }
  return 0.5 * out.y * out.z;
}
@compute @workgroup_size(64) fn ev_count(@builtin(global_invocation_id) g: vec3u) {
  let mi = g.x; if (mi >= min(ld(C_MAN), P.max_man)) { return; }
  let e = ev_energy_of(mi); if (e < 0.0) { return; }
  atomicAdd(&ints[P.o_evhist + ev_band(e)], 1u);
}
@compute @workgroup_size(1) fn ev_cut() {
  let room = P.max_events - min(ld(C_EVENTS), P.max_events);
  var tot = 0u; var cut = 128u;
  for (var b = 127; b >= 0; b--) { tot += ld(P.o_evhist + u32(b)); if (tot > room) { break; } cut = u32(b); }
  st(P.o_evhist + 128u, cut);
}
@compute @workgroup_size(64) fn events(@builtin(global_invocation_id) g: vec3u) {
  let mi = g.x; if (mi >= min(ld(C_MAN), P.max_man)) { return; }
  let m = P.cur_base + mi; let base = m * MS;
  let out = man[base + M_OUT];
  let e = ev_energy_of(mi); if (e < 0.0 || ev_band(e) < ld(P.o_evhist + 128u)) { return; }
  let k = atomicAdd(&ints[C_EVENTS], 1u); if (k >= P.max_events) { return; }
  let hd = man[base]; let a = u32(hd.x); let cnt = u32(hd.z) & 7u;
  var pt = vec3f(0.0); for (var q = 0u; q < cnt; q++) { pt += man[base + M_PT + 2u * q].xyz; }
  pt = pt / f32(max(cnt, 1u)) + B(a, F_P).xyz;
  let n = man[base + M_N].xyz;
  let o = P.o_events + k * 12u;
  st(o, a); st(o + 1u, u32(hd.y)); st(o + 2u, u32(cold[m * CS + 4u].w)); { let sb = u32(cold[m * CS + 5u].w); st(o + 3u, select(sb, 0u, sb == SUP_KEY)); }
  st(o + 4u, bitcast<u32>(pt.x)); st(o + 5u, bitcast<u32>(pt.y)); st(o + 6u, bitcast<u32>(pt.z));
  st(o + 7u, bitcast<u32>(n.x)); st(o + 8u, bitcast<u32>(n.y)); st(o + 9u, bitcast<u32>(n.z));
  st(o + 10u, bitcast<u32>(out.y)); st(o + 11u, bitcast<u32>(out.z));
}

// Crushing: a light piece pressed by a body many times heavier (a building's upper part landing on rubble cut
// from its own base) is crushed, as concrete is, when the stress at the contact passes its crushing strength. It then collides only with
// the static world (the heavy body goes through it, no rubble shot out sideways at 5-9 m/s) and is listed for
// the page, which turns it into rubble particles where it is.
const C_NCRUSH = 15u; const MAX_CRUSH = 256u;
// A crushed piece stops where it is (the page turns it into rubble at rest there): the step that crushed it may
// have thrown it, as the solver could not settle the squeeze (GPU test chain: a 223 kg piece under a 259 t part
// went up at 20.6 m/s in the step it was crushed, and flew for the 3 frames until the page heard of it).
// (and marks the bodies it was pressing, see emit)
@compute @workgroup_size(64) fn crush_release(@builtin(global_invocation_id) g: vec3u) {
  let mi = g.x; if (mi >= min(ld(C_MAN), P.max_man)) { return; }
  let hd = man[(P.cur_base + mi) * MS]; let a = u32(hd.x); let b = u32(hd.y);
  if (crushed(a) && moving(b) && !crushed(b)) { st(P.o_released + b, P.step + 1u); }
  if (crushed(b) && moving(a) && !crushed(a)) { st(P.o_released + a, P.step + 1u); }
}
fn crush_stop(i: u32) { setS(i, S_V, vec4f(0.0, 0.0, 0.0, S(i, S_V).w)); setS(i, S_W, vec4f(0.0, 0.0, 0.0, S(i, S_W).w)); }
// crushing strength (MPa) by material: concrete, masonry, steel, glass, timber (across the grain), test, ground,
// reference
const CRUSH_MPA = array<f32, 8>(30.0, 10.0, 250.0, 50.0, 8.0, 1.0, 1e9, 10.0);
@compute @workgroup_size(64) fn crush(@builtin(global_invocation_id) g: vec3u) {
  let mi = g.x; if (mi >= min(ld(C_MAN), P.max_man)) { return; }
  let m = P.cur_base + mi; let base = m * MS;
  let hd = man[base]; if (u32(hd.z) >= 8u) { return; }
  let a = u32(hd.x); let b = u32(hd.y);
  if (!moving(a) || !moving(b)) { return; }
  if (u32(cold[m * CS + 5u].w) == SUP_KEY) { return; }   // (a body held up through particles: see p_support)
  let out = man[base + M_OUT];
  if (out.z < 1.0) { return; }
  let ia = B(a, F_P).w; let ib = B(b, F_P).w;
  let light = select(b, a, ia > ib); let il = max(ia, ib); let ih = min(ia, ib);
  if (il < P.crush_ratio * ih) { return; }
  // squeezed: the impulse it took is well beyond what its own motion needs (a free piece hit by a heavy one is
  // only knocked away, about 1.1 x its approach)
  if (out.y * il < 1.5 * out.z) { return; }
  // and the stress on it: this step's normal impulse over the step, over the contact patch (radius kept in the
  // manifold), against the material's crushing strength. A building's upper part landing on rubble gives
  // hundreds of MPa; a steel beam set down on blocks a few, and they hold it.
  let side = select(cold[m * CS + 5u].w, cold[m * CS + 4u].w, light == a);
  let mat = u32(Hs(u32(side), H_ID).y);
  let rad = max(man[base + M_JT].w, 0.05);
  // (the solver limits the mass ratio to P_RATIO, so the impulse it took is that much smaller than what really
  // stops the heavy body: the real one is scaled back)
  let real = out.y * max(1.0, il / (P_RATIO * ih));
  let stress = real / P.dt / (3.14159 * rad * rad);
  if (stress < CRUSH_MPA[min(mat, 7u)] * 1e6) { return; }
  if (atomicExchange(&ints[P.o_crushed + light], 1u) != 0u) { return; }
  let v = S(light, S_V).xyz; crush_stop(light);
  let k = atomicAdd(&ints[C_NCRUSH], 1u); if (k >= MAX_CRUSH) { return; }
  let o = P.o_crush + k * 12u; let p = B(light, F_P).xyz; let q = B(light, F_Q);
  st(o, light); st(o + 1u, P.step);   // (the step: a record that arrives after the slot was reused is not about the new body)
  st(o + 2u, bitcast<u32>(p.x)); st(o + 3u, bitcast<u32>(p.y)); st(o + 4u, bitcast<u32>(p.z));
  st(o + 5u, bitcast<u32>(q.x)); st(o + 6u, bitcast<u32>(q.y)); st(o + 7u, bitcast<u32>(q.z)); st(o + 8u, bitcast<u32>(q.w));
  st(o + 9u, bitcast<u32>(v.x)); st(o + 10u, bitcast<u32>(v.y)); st(o + 11u, bitcast<u32>(v.z));
}

// Pinched: a light piece squeezed between two bodies, or a body and the static world. The crush test above sees
// one contact at a time and needs an approach of 1 m/s; a piece wedged under a falling part, or in a pocket of
// it, is pressed from both sides at almost no approach speed, takes impulses hundreds of times its own momentum
// (lab collapse: a 49 kg piece took 76,000 N s in one step between the ground and a 773 kg body) and the solver,
// which cannot settle that in a few iterations, throws it out spinning (30 rad/s, then up at 7 m/s). Here each
// moving body sums the normal impulses on it as vectors and as magnitudes (fixed point: the same sums in any
// order); the part that cancels is the squeeze. A piece is crushed when it is light against what squeezes it
// (P.crush_ratio, the static world counting as infinitely heavy), the squeeze is more than P.crush_dv of its
// own momentum, and the stress, the squeeze over the step over the largest contact patch, passes the weakest
// crushing strength among its shapes in those contacts.
// Per body (8 words): sum of impulse vectors (3 x i32, 1/16 N s), sum of magnitudes (u32, 1/16 N s), heaviest
// partner's mass (f32 bits, max), largest patch radius (f32 bits, max), 1 / crushing strength (f32 bits, max).
const PINCH_FX = 16.0;
fn pinch_add(i: u32, n: vec3f, j: f32, heavy: f32, rad: f32, mat: u32) {
  let o = P.o_pinch + i * 8u; let jj = n * j * PINCH_FX;
  atomicAdd(&ints[o], bitcast<u32>(i32(round(jj.x)))); atomicAdd(&ints[o + 1u], bitcast<u32>(i32(round(jj.y))));
  atomicAdd(&ints[o + 2u], bitcast<u32>(i32(round(jj.z)))); atomicAdd(&ints[o + 3u], u32(round(j * PINCH_FX)));
  atomicMax(&ints[o + 4u], bitcast<u32>(heavy)); atomicMax(&ints[o + 5u], bitcast<u32>(rad));
  atomicMax(&ints[o + 6u], bitcast<u32>(1.0 / CRUSH_MPA[min(mat, 7u)]));
}
@compute @workgroup_size(64) fn pinch_acc(@builtin(global_invocation_id) g: vec3u) {
  let mi = g.x; if (mi >= min(ld(C_MAN), P.max_man)) { return; }
  let m = P.cur_base + mi; let base = m * MS;
  let hd = man[base]; if (u32(hd.z) >= 8u) { return; }
  let a = u32(hd.x); let b = u32(hd.y);
  let out = man[base + M_OUT]; if (out.y <= 0.0 || u32(cold[m * CS + 5u].w) == SUP_KEY) { return; }
  let j = min(out.y, 1e7); let n = man[base + M_N].xyz; let rad = max(man[base + M_JT].w, 0.05);
  // (mass of each side: the static world and sleeping bodies as infinitely heavy)
  let ma = select(3e38, 1.0 / max(B(a, F_P).w, 1e-30), moving(a)); let mb = select(3e38, 1.0 / max(B(b, F_P).w, 1e-30), moving(b));
  // (the normal points from a to b: the impulse pushes b along it and a against it)
  if (moving(b) && !crushed(b)) { pinch_add(b, n, j, ma, rad, u32(Hs(u32(cold[m * CS + 5u].w), H_ID).y)); }
  if (moving(a) && !crushed(a)) { pinch_add(a, -n, j, mb, rad, u32(Hs(u32(cold[m * CS + 4u].w), H_ID).y)); }
}
@compute @workgroup_size(64) fn pinch(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i == 0u || i >= P.n1 || !moving(i) || crushed(i)) { return; }
  let o = P.o_pinch + i * 8u;
  let tot = f32(ld(o + 3u)) / PINCH_FX; if (tot <= 0.0) { return; }
  let net = vec3f(f32(bitcast<i32>(ld(o))), f32(bitcast<i32>(ld(o + 1u))), f32(bitcast<i32>(ld(o + 2u)))) / PINCH_FX;
  let squeeze = 0.5 * max(tot - length(net), 0.0);
  let im = B(i, F_P).w;
  if (bitcast<f32>(ld(o + 4u)) * im < P.crush_ratio) { return; }
  if (squeeze * im < P.crush_dv) { return; }
  let rad = bitcast<f32>(ld(o + 5u));
  if (squeeze / P.dt / (3.14159 * rad * rad) * bitcast<f32>(ld(o + 6u)) < 1e6) { return; }
  if (atomicExchange(&ints[P.o_crushed + i], 1u) != 0u) { return; }
  let v = S(i, S_V).xyz; crush_stop(i);
  let k = atomicAdd(&ints[C_NCRUSH], 1u); if (k >= MAX_CRUSH) { return; }
  let c = P.o_crush + k * 12u; let p = B(i, F_P).xyz; let q = B(i, F_Q);
  st(c, i); st(c + 1u, P.step);
  st(c + 2u, bitcast<u32>(p.x)); st(c + 3u, bitcast<u32>(p.y)); st(c + 4u, bitcast<u32>(p.z));
  st(c + 5u, bitcast<u32>(q.x)); st(c + 6u, bitcast<u32>(q.y)); st(c + 7u, bitcast<u32>(q.z)); st(c + 8u, bitcast<u32>(q.w));
  st(c + 9u, bitcast<u32>(v.x)); st(c + 10u, bitcast<u32>(v.y)); st(c + 11u, bitcast<u32>(v.z));
}

// Breaking on the GPU, in the step of the hit. A falling piece can carry a plan from the engine (GpuWorld.setSnap):
// its weakest section, the normal impulse in one step that breaks it there, and the two bodies it breaks into
// (the part it keeps and a spare body for the other, made in advance with that part's mass). When this step's
// contacts on it pass the impulse, it breaks at once: the spare takes the other part's shapes and the motion of
// the piece at its centre, the piece keeps the rest with that part's mass. The engine is told afterwards (the
// list at o_snap) and refines the break when it answers; it used to answer a hit about 4 frames late.
// Plan (SN words per body): impulse (f32 N s), spare body, state (1 armed, 2 broke this step, 3 done), impulse
// this step (u32 N s), kept part: centre of mass (3), rotation (4) in the body's frame, inverse mass, inverse
// inertia (3); other part: centre of mass (3), rotation (4).
const SN = 24u; const C_NSNAP_MAX = 256u;
fn plan(i: u32, k: u32) -> u32 { return ld(P.o_plan + i * SN + k); }
fn planf(i: u32, k: u32) -> f32 { return bitcast<f32>(plan(i, k)); }
fn plan3(i: u32, k: u32) -> vec3f { return vec3f(planf(i, k), planf(i, k + 1u), planf(i, k + 2u)); }
fn plan4(i: u32, k: u32) -> vec4f { return vec4f(planf(i, k), planf(i, k + 1u), planf(i, k + 2u), planf(i, k + 3u)); }
@compute @workgroup_size(64) fn snap_acc(@builtin(global_invocation_id) g: vec3u) {
  let mi = g.x; if (mi >= min(ld(C_MAN), P.max_man)) { return; }
  let m = P.cur_base + mi; let base = m * MS;
  let hd = man[base]; let j = u32(man[base + M_OUT].y + 0.5);
  if (j == 0u) { return; }
  for (var k = 0u; k < 2u; k++) {
    let i = u32(select(hd.y, hd.x, k == 0u));
    if (i != 0u && plan(i, 2u) == 1u) { atomicAdd(&ints[P.o_plan + i * SN + 3u], j); }
  }
}
// The load on particles holding bodies up: each support contact's impulse this step, added up per shape (N s).
// (its slot from emit does not do: the contacts are reordered after the narrow phase, and the slot then named
// another contact, of next to no impulse: the 120 t block on 25 crumbs read 0 Pa and stayed held, GPU test heap)
@compute @workgroup_size(64) fn sup_acc(@builtin(global_invocation_id) g: vec3u) {
  let mi = g.x; if (mi >= min(ld(C_MAN), P.max_man) || P.o_psup == 0u) { return; }
  let m = P.cur_base + mi;
  if (u32(cold[m * CS + 5u].w) != SUP_KEY) { return; }
  let s = u32(cold[m * CS + 4u].w); if (s == 0u || s >= P.psup_n) { return; }
  atomicAdd(&ints[P.o_psup + 8u * P.psup_n + s], u32(clamp(man[m * MS + M_OUT].y, 0.0, 4.0e9) + 0.5));
}
@compute @workgroup_size(64) fn snap_split(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + 1u; if (i >= P.n1) { return; }
  let state = plan(i, 2u);
  if (state == 2u) { st(P.o_plan + i * SN + 2u, 3u); return; }
  if (state != 1u) { return; }
  let acc = f32(plan(i, 3u)); st(P.o_plan + i * SN + 3u, 0u);
  if (acc < planf(i, 0u)) { return; }
  let k = atomicAdd(&ints[P.o_snap], 1u); if (k >= C_NSNAP_MAX) { return; }
  let j = plan(i, 1u);
  let p = B(i, F_P); let q = B(i, F_Q); let v = S(i, S_V); let w = S(i, S_W).xyz;
  // the other part: the spare, where that part is, moving as the piece did there
  let pb = p.xyz + qrot(q, plan3(i, 15u)); let qb = normalize(qmul(q, plan4(i, 18u)));
  setB(j, F_P, vec4f(pb, B(j, F_P).w)); setB(j, F_Q, qb); setB(j, F_FL, vec4f(0.0, pb)); setB(j, F_RQ, qb);
  let ilb = B(j, F_IL); setB(j, F_IL, vec4f(ilb.xyz, 0.0));
  setS(j, S_V, vec4f(v.xyz + cross(w, pb - p.xyz), 1.0)); setS(j, S_W, vec4f(w, 0.0));
  // the piece keeps its own part, with that part's mass
  let pa = p.xyz + qrot(q, plan3(i, 4u)); let qa = normalize(qmul(q, plan4(i, 7u)));
  setB(i, F_P, vec4f(pa, planf(i, 11u))); setB(i, F_Q, qa); setB(i, F_IL, vec4f(plan3(i, 12u), 0.0));
  setB(i, F_FL, vec4f(0.0, pa)); setB(i, F_RQ, qa);
  setS(i, S_V, vec4f(v.xyz + cross(w, pa - p.xyz), v.w));
  st(P.o_plan + i * SN + 2u, 2u);
  let o = P.o_snap + 1u + 4u * k; st(o, i); st(o + 1u, j); st(o + 2u, P.cur_base); st(o + 3u, bitcast<u32>(acc));
}
// the shapes of a piece that broke this step go to their part's body, in its frame
@compute @workgroup_size(64) fn snap_shapes(@builtin(global_invocation_id) g: vec3u) {
  let s = g.x + 1u; if (s >= P.ns1 || dead(s)) { return; }
  let i = sbody(s); if (i == 0u || plan(i, 2u) != 2u) { return; }
  let other = ld(P.o_spart + s) == 2u;
  let c = select(plan3(i, 4u), plan3(i, 15u), other); let r = select(plan4(i, 7u), plan4(i, 18u), other);
  let rc = vec4f(-r.xyz, r.w);
  let id = Hs(s, H_ID);
  if (other) { setHs(s, H_ID, vec4f(f32(plan(i, 1u)), id.yzw)); }
  setHs(s, H_LP, vec4f(qrot(rc, Hs(s, H_LP).xyz - c), 0.0));
  setHs(s, H_LQ, normalize(qmul(rc, Hs(s, H_LQ))));
}

@compute @workgroup_size(64) fn finish(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + 1u; if (i >= P.n1) { return; }
  let v4 = S(i, S_V); if (v4.w == 0.0) { return; }
  let p = B(i, F_P); let np = p.xyz + S(i, S_DP).xyz; setB(i, F_P, vec4f(np, p.w));
  let w = S(i, S_W).xyz;
  let il = B(i, F_IL);
  // at rest = it stays near where it was when the timer restarted (solver chatter of a few cm/s goes
  // nowhere and must not keep a settled pile awake), and is not moving fast
  let fl = B(i, F_FL); let q = B(i, F_Q); let rq = B(i, F_RQ);
  let dq = qmul(q, vec4f(-rq.xyz, rq.w));
  let near = distance(np, fl.yzw) < P.sleep_dist && 2.0 * length(dq.xyz) < P.sleep_rot;
  let slowish = dot(v4.xyz, v4.xyz) < 100.0 * P.sleep_lin * P.sleep_lin && dot(w, w) < 100.0 * P.sleep_ang * P.sleep_ang;
  var t = 0.0;
  if (near && slowish) { t = il.w + P.dt; } else { setB(i, F_FL, vec4f(fl.x, np)); setB(i, F_RQ, q); }
  setB(i, F_IL, vec4f(il.xyz, t));
}
fn find(x0: u32) -> u32 {
  var x = x0;
  loop {
    let p = ld(P.o_par + x); if (p == x) { return x; }
    let gp = ld(P.o_par + p);
    if (gp != p) { atomicCompareExchangeWeak(&ints[P.o_par + x], p, gp); }
    x = p;
  }
}
@compute @workgroup_size(64) fn uf_union(@builtin(global_invocation_id) g: vec3u) {
  let mi = g.x; if (mi >= min(ld(C_MAN), P.max_man)) { return; }
  let ab = mhdr(P.cur_base + mi);
  if (!dyn_awake(ab.x) || !dyn_awake(ab.y)) { return; }
  var x = ab.x; var y = ab.y;
  for (var it = 0u; it < 256u; it++) {
    x = find(x); y = find(y);
    if (x == y) { return; }
    if (x < y) { let t = x; x = y; y = t; }
    let r = atomicCompareExchangeWeak(&ints[P.o_par + x], x, y);
    if (r.exchanged) { return; }
  }
}
@compute @workgroup_size(64) fn uf_min(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + 1u; if (i >= P.n1 || !dyn_awake(i)) { return; }
  atomicMin(&ints[P.o_mint + find(i)], bitcast<u32>(B(i, F_IL).w));
}
@compute @workgroup_size(64) fn uf_sleep(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + 1u; if (i >= P.n1 || !dyn_awake(i)) { return; }
  // the whole island at rest, or this body itself at rest for twice as long (a pile with a few creeping pieces
  // must not stay awake as a whole: the still ones sleep and act as static support for the rest)
  if (bitcast<f32>(ld(P.o_mint + find(i))) >= P.sleep_time || B(i, F_IL).w >= 2.0 * P.sleep_time) {
    let fl = B(i, F_FL); setB(i, F_FL, vec4f(1.0, fl.yzw));
    setS(i, S_V, vec4f(0.0, 0.0, 0.0, S(i, S_V).w)); setS(i, S_W, vec4f(0.0));
  }
}
`;

// Granular debris: crumbs as particles (spheres, drawn as the bricks they are). Position-based dynamics: each
// sub-step predicts, then pushes particles out of each other, of the ground and of every rigid shape (one way: a
// particle never pushes a body), with Coulomb friction from the push; velocity is the move over the sub-step.
// Particles at rest sleep and cost a few grid look-ups; a moving body nearby, a blast or wakeAll wakes them.
const WGSL_P = WGSL + /* wgsl */`
@group(2) @binding(0) var<storage, read_write> prt: array<vec4f>;
@group(2) @binding(1) var<storage, read_write> pint: array<atomic<u32>>;
// particle record (7 vec4): position + radius, velocity + state (0 free, 1 moving, 2 asleep, 3 crushed, 4 waiting
// to be placed from a moving body), rotation
// (for drawing), sub-step start position + rest timer, spin (drawing) + material, corrected position + touching,
// where the rest timer last restarted + crushing
const PR = 7u;
fn R(i: u32, f: u32) -> vec4f { return prt[i * PR + f]; }
fn setR(i: u32, f: u32, v: vec4f) { prt[i * PR + f] = v; }
fn pg() -> u32 { return P.pg_mask + 1u; }
fn pcellc(p: vec3f) -> vec3i { return vec3i(floor(p / P.pcell)); }
fn phc(c: vec3i) -> u32 { return ((u32(c.x) * 73856093u) ^ (u32(c.y) * 19349663u) ^ (u32(c.z) * 83492791u)) & P.pg_mask; }
fn pld(i: u32) -> u32 { return atomicLoad(&pint[i]); }
fn pst(i: u32, v: u32) { atomicStore(&pint[i], v); }
// a moving rigid shape in a grid cell around x (any level)
fn near_awake(x: vec3f) -> bool {
  for (var L = 0u; L < P.levels; L++) {
    if (ld(C_LEVEL + L) == 0u) { continue; }
    let c = cellc(x, L);
    for (var cz = c.z - 1; cz <= c.z + 1; cz++) { for (var cy = c.y - 1; cy <= c.y + 1; cy++) { for (var cx = c.x - 1; cx <= c.x + 1; cx++) {
      if ((ld(P.o_cawake + chash(vec3i(cx, cy, cz), L)) & 2u) != 0u) { return true; }
    } } }
  }
  return false;
}
@compute @workgroup_size(64) fn p_prep(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= P.np1) { return; }
  let r1 = R(i, 1); var state = u32(r1.w); if (state == 0u || state >= 3u) { return; }
  let x = R(i, 0).xyz;
  var v = r1.xyz;
  var hit = false;
  for (var b = 0u; b < P.n_blast; b++) {
    let c = P.blast[2u * b]; let d = x - c.xyz; let l = length(d);
    if (l < c.w) { hit = true; v += (d / max(l, 1e-3) + vec3f(0.0, 0.3, 0.0)) * P.blast[2u * b + 1u].x * (1.0 - l / c.w); }
  }
  if (state == 2u && (hit || P.p_wake != 0u || near_awake(x))) { state = 1u; let r3 = R(i, 3); setR(i, 3, vec4f(r3.xyz, 0.0)); }
  if (state == 1u) { atomicAdd(&ints[12], 1u); setR(i, 1, vec4f(v, 1.0)); }
  atomicAdd(&ints[13], 1u);
  let hc = phc(pcellc(x));
  pst(3u * pg() + P.p_cap + i, hc);
  atomicAdd(&pint[hc], 1u);
}
// particles cut from a moving body: placed like its pieces (see inherit), from entries whose kid has the high bit
// (kid = particle slot + 1); they wait (state 4) until then
@compute @workgroup_size(64) fn p_place(@builtin(global_invocation_id) g: vec3u) {
  let k = g.x; if (k >= P.n_inherit) { return; }
  let kid = ld(P.o_inherit + k * 12u); let praw = ld(P.o_inherit + k * 12u + 1u);
  if ((kid & 0x80000000u) == 0u) { return; }
  let i = (kid & 0x7fffffffu) - 1u; let par = praw & 0x3fffffffu; let absolute = (praw & 0x80000000u) != 0u;
  let pr = vec3f(inh(k, 2u), inh(k, 3u), inh(k, 4u));
  let qr = vec4f(inh(k, 5u), inh(k, 6u), inh(k, 7u), inh(k, 8u));
  let dv = vec3f(inh(k, 9u), inh(k, 10u), inh(k, 11u));
  let pn = B(par, F_P).xyz; let qn = B(par, F_Q);
  let rd = normalize(qmul(qn, vec4f(-qr.xyz, qr.w)));
  let r0 = R(i, 0);
  let x = pn + qrot(rd, r0.xyz - pr);
  var v = S(par, S_V).xyz + cross(S(par, S_W).xyz, x - pn) + qrot(rd, dv);
  if (absolute) { v = dv; }
  setR(i, 0, vec4f(x, r0.w)); setR(i, 1, vec4f(v, 1.0)); setR(i, 2, normalize(qmul(rd, R(i, 2))));
  setR(i, 3, vec4f(x, 0.0)); setR(i, 5, vec4f(x, 0.0)); setR(i, 6, vec4f(x, 0.0));
}
@compute @workgroup_size(256) fn p_scan(@builtin(local_invocation_index) t: u32) {
  let G = pg(); let per = G / 256u; let base = t * per;
  var s = 0u;
  for (var k = 0u; k < per; k++) { s += pld(base + k); }
  var run = wg_scan(t, s);
  for (var k = 0u; k < per; k++) { pst(G + base + k, run); run += pld(base + k); }
}
@compute @workgroup_size(64) fn p_scatter(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= P.np1) { return; }
  let st = R(i, 1).w; if (st == 0.0 || st >= 3.0) { return; }
  let G = pg(); let hc = pld(3u * G + P.p_cap + i);
  pst(3u * G + pld(G + hc) + atomicAdd(&pint[2u * G + hc], 1u), i);
}
// (each cell's particles in index order: the candidate lists, capped, are the same every run)
@compute @workgroup_size(64) fn p_sort(@builtin(global_invocation_id) g: vec3u) {
  let G = pg(); let hc = g.x; if (hc >= G) { return; }
  let s0 = 3u * G + pld(G + hc); let n = pld(hc);
  for (var a = 1u; a < n; a++) {
    let v = pld(s0 + a); var b = a;
    while (b > 0u && pld(s0 + b - 1u) > v) { pst(s0 + b, pld(s0 + b - 1u)); b--; }
    pst(s0 + b, v);
  }
}
@compute @workgroup_size(64) fn p_integrate(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= P.np1 || R(i, 1).w != 1.0) { return; }
  var v = R(i, 1).xyz + P.gravity * P.p_h;
  let l = length(v); if (l > 40.0) { v *= 40.0 / l; }
  let r0 = R(i, 0);
  setR(i, 3, vec4f(r0.xyz, R(i, 3).w));
  setR(i, 0, vec4f(r0.xyz + v * P.p_h, r0.w));
  setR(i, 1, vec4f(v, 1.0));
}
// friction: take back the sliding of this sub-step, up to mu x the push (Coulomb)
fn pfric(disp: vec3f, n: vec3f, pen: f32, mu: f32) -> vec3f {
  let t = disp - n * dot(disp, n); let tl = length(t);
  if (tl < 1e-9) { return vec3f(0.0); }
  return -t * min(mu * pen / tl, 1.0);
}
// Candidates, gathered once per step: the particles and shapes a particle can touch during the step (its
// sub-steps only test these). Per particle in pint after the grid: [neighbours, shapes, 24 neighbours, 12 shapes]
const P_NN = 24u; const P_NS = 12u; const P_CS = 41u;   // (+2: resting on the ground, this sub-step and the last; see p_contact; +1: the load it holds, see p_sup_load)
fn pcand(i: u32) -> u32 { return 3u * pg() + 2u * P.p_cap + i * P_CS; }
@compute @workgroup_size(64) fn p_gather(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= P.np1 || R(i, 1).w != 1.0) { return; }
  let r0 = R(i, 0); let xi = r0.xyz; let ri = r0.w;
  // how far it can get this step (and a little for the others' motion)
  let m = length(R(i, 1).xyz) * P.dt + 0.02;
  let o = pcand(i);
  var nn = 0u;
  let G = pg(); let c = pcellc(xi);
  var seen: array<u32, 27>; var ns = 0u;
  for (var cz = c.z - 1; cz <= c.z + 1; cz++) { for (var cy = c.y - 1; cy <= c.y + 1; cy++) { for (var cx = c.x - 1; cx <= c.x + 1; cx++) {
    let hc = phc(vec3i(cx, cy, cz));
    var dup = false; for (var q = 0u; q < ns; q++) { if (seen[q] == hc) { dup = true; } }
    if (dup) { continue; }
    seen[ns] = hc; ns++;
    let s0 = pld(G + hc); let s1 = s0 + pld(hc);
    for (var s = s0; s < s1; s++) {
      let j = pld(3u * G + s); if (j == i) { continue; }
      let rj = R(j, 0); let rr = ri + rj.w + m + length(R(j, 1).xyz) * P.dt + 0.01;
      let d = xi - rj.xyz;
      if (dot(d, d) < rr * rr && nn < P_NN) { pst(o + 2u + nn, j); nn++; }
    }
  } } }
  var nsh = 0u;
  for (var L = 0u; L < P.levels; L++) {
    if (ld(C_LEVEL + L) == 0u) { continue; }
    let cc = cellc(xi, L);
    for (var cz = cc.z - 1; cz <= cc.z + 1; cz++) { for (var cy = cc.y - 1; cy <= cc.y + 1; cy++) { for (var cx = cc.x - 1; cx <= cc.x + 1; cx++) {
      let cell = vec3i(cx, cy, cz); let hc = chash(cell, L);
      let s0 = ld(P.o_gstart + hc); let s1 = s0 + ld(P.o_gcount + hc);
      for (var s = s0; s < s1; s++) {
        let j = ld(P.o_sorted + s);
        let wj = Hs(j, H_WP);
        if (u32(wj.w) != L || any(cellc(wj.xyz, L) != cell)) { continue; }
        let e = vec3f(ri + m + 0.01);
        if (any(xi + e < Hs(j, H_LO).xyz) || any(xi - e > Hs(j, H_HI).xyz)) { continue; }
        if (nsh < P_NS) { pst(o + 2u + P_NN + nsh, j); nsh++; }
      }
    } } }
  }
  pst(o, nn); pst(o + 1u, nsh);
}
// push limits: a sub-step's own approach is undone in full; overlap beyond it (made by a spawn, or by a body
// driven into a heap) separates at 2 m/s, so nothing is launched from a deep overlap
const P_SEP = 2.0;
@compute @workgroup_size(64) fn p_contact(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= P.np1 || R(i, 1).w != 1.0) { return; }
  let r0 = R(i, 0); let xi = r0.xyz; let ri = r0.w; let xs = R(i, 3).xyz;
  let mui = P.mat[u32(R(i, 4).w)].x;
  var corr = vec3f(0.0); var n = 0.0; var touch = 0.0; var bearing = false;
  var grounded = false; var gPen = 0.0; var gN = vec3f(0.0);
  let cap = P_SEP * P.p_h;
  let o = pcand(i);
  // other particles first (a sleeping one does not give way); several pushes are averaged, over-relaxed (Jacobi)
  let nn0 = pld(o);
  for (var q = 0u; q < nn0; q++) {
    let j = pld(o + 2u + q);
    let rj = R(j, 0); let d = xi - rj.xyz; let d2 = dot(d, d); let rr = ri + rj.w;
    // (within 1 cm counts as touching, for the rest test: a resting particle alternates between a hair's
    // overlap and none from one sub-step to the next)
    if (d2 >= (rr + 0.01) * (rr + 0.01) || d2 < 1e-12) { continue; }
    let dist = sqrt(d2); let nn = d / dist;
    // (resting, through j, on the ground: j is under it and itself rests on the ground, as of the last sub-step)
    let jg = pld(pcand(j) + 38u) != 0u && nn.y > 0.5;
    if (jg) { grounded = true; }
    touch = 1.0; if (dist >= rr) { continue; }
    if (jg && rr - dist > gPen) { gPen = rr - dist; gN = nn; }
    let moving = R(j, 1).w == 1.0;
    let wgt = select(1.0, 0.5, moving);
    let dj = select(vec3f(0.0), rj.xyz - R(j, 3).xyz, moving);
    let pen = min((rr - dist) * wgt, (max(-dot((xi - xs) - dj, nn), 0.0) + cap) * wgt);
    corr += nn * pen + pfric((xi - xs) - dj, nn, pen, mui) * wgt; n += 1.0;
  }
  var x = xi + corr * select(1.0, 1.5 / n, n > 1.5);
  // a particle pushed hard from opposite sides by two bodies (a box set down on it on the floor) is crushed
  var lastN = vec3f(0.0); var lastPush = 0.0; var crushed = R(i, 6).w;
  var lastS = 0xffffffffu; var lastFixed = true;   // (the shape that pushed hardest so far; the ground: fixed)
  // then the ground and the rigid shapes, in sequence, each in full: a particle never ends inside a body
  let pg0 = ri - x.y;
  if (pg0 > -0.01) { touch = 1.0; grounded = true; }
  // (pressed onto particles that rest on the ground: as good as the ground, for holding a body up)
  if (gPen > 0.0 && pg0 <= 0.0) { lastPush = gPen; lastN = gN; }
  if (pg0 > 0.0) { let nn = vec3f(0.0, 1.0, 0.0); x += nn * pg0 + pfric(x - xs, nn, pg0, 0.5 * (mui + P.mat[P.ground_mat].x)); lastPush = pg0; lastN = nn; }
  let ns0 = pld(o + 1u);
  for (var q = 0u; q < ns0; q++) {
    let j = pld(o + 2u + P_NN + q);
    let wj = Hs(j, H_WP);
    let Rm = qmat(Hs(j, H_WQ)); let h = Hs(j, H_H).xyz;
    let dl = x - wj.xyz;
    let lc = vec3f(dot(Rm[0], dl), dot(Rm[1], dl), dot(Rm[2], dl));
    let cl = clamp(lc, -h, h); let df = lc - cl; let d2 = dot(df, df);
    var nl = vec3f(0.0); var pen = 0.0;
    if (d2 > 1e-12) {
      let dist = sqrt(d2); if (dist >= ri + 0.01) { continue; }
      touch = 1.0; if (dist >= ri) { continue; }
      nl = df / dist; pen = ri - dist;
    } else {
      let dd = h - abs(lc);
      if (dd.x <= dd.y && dd.x <= dd.z) { nl = vec3f(select(-1.0, 1.0, lc.x >= 0.0), 0.0, 0.0); pen = dd.x + ri; }
      else if (dd.y <= dd.z) { nl = vec3f(0.0, select(-1.0, 1.0, lc.y >= 0.0), 0.0); pen = dd.y + ri; }
      else { nl = vec3f(0.0, 0.0, select(-1.0, 1.0, lc.z >= 0.0)); pen = dd.z + ri; }
    }
    let nn = Rm[0] * nl.x + Rm[1] * nl.y + Rm[2] * nl.z;
    let b = sbody(j);
    let vb = S(b, S_V).xyz + cross(S(b, S_W).xyz, x - B(b, F_P).xyz);
    // out along the normal: at most as far as the body moves into the particle plus the separation limit,
    // the rest goes as a slide along the surface (no launch from a deep overlap)
    let push = min(pen, max(dot(vb * P.p_h - (x - xs), nn), 0.0) + cap);
    let mu = 0.5 * (mui + P.mat[u32(Hs(j, H_ID).y)].x);
    x += nn * push + pfric((x - xs) - vb * P.p_h, nn, push, mu); touch = 1.0;
    let fixed = !moving(b) || asleep(b);
    if (fixed && nn.y > 0.5) { grounded = true; }
    if (lastPush > 0.0 && pen > 0.0 && dot(nn, lastN) < -0.5) {
      // pressed between a moving body and something fixed (the ground, a static or sleeping shape): the particle
      // holds the body up (p_support makes it a contact of the body's at the next step). Crushed as dust
      // instead, a heap under a falling part was deleted under it: the lab's mixed scenario lost 254 m3 so
      // The particle holds up the moving shape pressing it (the upper one, when both move) against what is under
      // it: the static world, or the other body (a part resting on rubble on a floor slab that is still awake).
      // (under an overloaded shape, see p_sup_load, it is not held up: it sinks in, and a particle squeezed deep
      // between it and what holds the particle is crushed; the rest still count, for the next step's load)
      // (between two moving bodies it was crushed once both pressed it 0.3 r deep, and a part resting on a heap
      // on an awake slab sank through the heap: lab mixed lost 48 m3 of particles that way, 3.1% of the building)
      if (fixed != lastFixed || !fixed) {
        var sh = select(j, lastS, fixed); var low = select(lastS, j, fixed);
        if (!fixed && !lastFixed && nn.y > lastN.y) { sh = lastS; low = j; }
        var lowB = 0xffffffffu; if (!fixed && !lastFixed && low != 0xffffffffu) { lowB = sbody(low); }
        if (sh != 0xffffffffu && ld(P.o_psup + 11u * P.psup_n + sh) >= 3u && min(pen, lastPush) > 0.3 * ri) { crushed = 1.0; }
        else if (sh != 0xffffffffu && (lowB == 0xffffffffu || sbody(sh) != lowB)) { sup_record(sh, i, min(pen, lastPush), lowB); bearing = true; }
      }
    }
    if (pen > lastPush) { lastPush = pen; lastN = nn; lastS = j; lastFixed = fixed; }
  }
  setR(i, 5, vec4f(x, select(touch, 2.0, bearing)));   // (2: holding up a moving body, see p_finish)
  pst(pcand(i) + 39u, select(0u, 1u, grounded));
  if (crushed > 0.0) { let r6 = R(i, 6); setR(i, 6, vec4f(r6.xyz, 1.0)); }
}
// Particles holding bodies up. Per shape, four slots (by the quarter of the shape the particle is in, in x and
// z): the deepest-pressed particle in each, packed (depth, particle) for atomicMax, so the same in any order.
// (low: the body under the particle, or none for the static world; one per shape, the static world first)
fn sup_record(s: u32, i: u32, depth: f32, low: u32) {
  if (s == 0xffffffffu || i > 0xffffu || P.o_psup == 0u) { return; }
  atomicMax(&ints[P.o_psup + 12u * P.psup_n + s], select(low + 1u, 0xffffffffu, low == 0xffffffffu));
  let wp = Hs(s, H_WP).xyz; let x = R(i, 0).xyz;
  let q = select(0u, 1u, x.x > wp.x) + select(0u, 2u, x.z > wp.z);
  atomicMax(&ints[P.o_psup + s * 4u + q], (u32(clamp(depth / 0.5, 0.0, 1.0) * 32767.0 + 1.0) << 16u) | i);
  // (and the cross-section of all the particles holding this shape up, cm2: the load is shared among them all)
  let r = R(i, 0).w; atomicAdd(&ints[P.o_psup + 9u * P.psup_n + s], u32(3.14159 * r * r * 1e4 + 0.5));
}
// At the next step, after the narrow phase: each shape's slots become one contact of its body against the static
// world at those particles (up to four points), solved with the rest: friction, warm start, push limits. The
// particle stays where it is, between the body and what holds it.
@compute @workgroup_size(64) fn p_support(@builtin(global_invocation_id) g: vec3u) {
  let s = g.x; if (s == 0u || s >= P.ns1) { return; }
  var c: Contact; c.cnt = 0u; var nsum = vec3f(0.0); var pm = 0u;
  let body = sbody(s);
  let wp = Hs(s, H_WP).xyz; let Rm = qmat(Hs(s, H_WQ)); let h = Hs(s, H_H).xyz;
  for (var q = 0u; q < 4u; q++) {
    let v = ld(P.o_psup + s * 4u + q); if (v == 0u) { continue; }
    st(P.o_psup + s * 4u + q, 0u);
    let i = v & 0xffffu; let r0 = R(i, 0); let x = r0.xyz;
    if (R(i, 1).w == 0.0 || R(i, 1).w >= 3.0) { continue; }
    let dl = x - wp; let lc = vec3f(dot(Rm[0], dl), dot(Rm[1], dl), dot(Rm[2], dl));
    let cl = clamp(lc, -h, h); var nl = lc - cl; var dist = length(nl);
    // (a particle whose centre is inside the shape holds nothing up: it is on its way out of it)
    if (dist < 1e-6) { continue; }
    nl = nl / dist;
    let n = Rm[0] * nl.x + Rm[1] * nl.y + Rm[2] * nl.z;
    let cp = wp + Rm[0] * cl.x + Rm[1] * cl.y + Rm[2] * cl.z;
    // (the contact comes a step late, by then the body is into the particle, and the heap under it springs back
    // as its particles separate: pushed out of that, a slab landing at 6.6 m/s came back up at 1.95 m/s (GPU
    // test heap). It is stopped where it is instead, with no push, no restitution, no warm start, and particles
    // whose centre is inside the shape are left out: 0.48 m/s)
    let sep = max(dist - r0.w, 0.0); if (sep > 0.05) { continue; }
    // (one normal for the contact: the particles' must agree; one on the far side of the body is left out)
    if (c.cnt > 0u && dot(n, nsum) <= 0.5 * length(nsum)) { continue; }
    st(P.o_psup + 4u * P.psup_n + s * 4u + c.cnt, i);
    c.pts[c.cnt] = vec4f(cp, sep); c.cnt++; nsum += n; pm = u32(R(i, 4).w);
  }
  let lw = ld(P.o_psup + 12u * P.psup_n + s); st(P.o_psup + 12u * P.psup_n + s, 0u);
  if (c.cnt == 0u || !moving(body) || asleep(body)) { return; }
  // (against the body under the particles, while it moves; against the static world otherwise)
  var low = 0u; if (lw != 0u && lw != 0xffffffffu) { low = lw - 1u; if (!moving(low) || asleep(low) || low == body) { low = 0u; } }
  // (normals summing to nothing made a NaN normal, NaN bodies, and the engine stopped: lab collapse frame 155)
  let ln = length(nsum); if (!(ln > 1e-3)) { return; }
  c.n = nsum / ln;
  sup_mat = pm;
  // The particles bear at most what loose rubble bears (RUBBLE_BEAR) over their cross-section (as p_contact
  // counted it in the last step's sub-steps), shared by the points, per sub-step: a body heavier than that sinks
  // in, decelerated by it, and the particles squeezed deep are crushed (p_contact) until enough of them bear it.
  let area = f32(ld(P.o_psup + 9u * P.psup_n + s)) * 1e-4 * P.p_h / P.dt;
  sup_cap = RUBBLE_BEAR * area * P.h / f32(c.cnt);
  emit(s, SUP_KEY, body, low, c);
  sup_mat = 0xffffffffu;
}
// After the bodies' solve: the load on the particles holding each shape up (its support contacts' impulse over
// the step, see sup_acc). The contact pushes at most RUBBLE_BEAR (like gravel fill) over the particles'
// cross-section (p_support); a shape that took that much is overloaded, and the particles pressed deep between
// it and what holds them are crushed (p_contact) as it sinks in, until enough of them bear it. A heap holds a
// slab (2.9 t: kPa) but not a building: crumbs left in the cut under a 2,225 t tower held it up, and its roof came
// down 4 cm (lab collapse) instead of 10 m. (Held outright, with the particles crushed past the limit instead:
// the overload, judged on a landing's spike, crushed every particle pressed for 30 steps; a 21.6 t block set
// down on crumbs on a slab went through to it, 239 crushed, GPU test heapcreep.)
const RUBBLE_BEAR = 1.0e6;
@compute @workgroup_size(64) fn p_sup_load(@builtin(global_invocation_id) g: vec3u) {
  let s = g.x; if (s == 0u || s >= P.ns1) { return; }
  // (added at each of the particles' sub-steps: over their number, the cross-section; counted once per sub-step it
  // was 4 times the real one, and the 120 t block's 1.5 MPa on 25 crumbs read 0.38 MPa, held)
  let area = f32(ld(P.o_psup + 9u * P.psup_n + s)) * 1e-4 * P.p_h / P.dt; st(P.o_psup + 9u * P.psup_n + s, 0u);
  let imp = f32(ld(P.o_psup + 8u * P.psup_n + s)); st(P.o_psup + 8u * P.psup_n + s, 0u);
  // (overloaded: the contact pushed at its limit, see p_support; the particles then squeezed deep are crushed)
  let load = imp / P.dt; st(P.o_psup + 10u * P.psup_n + s, bitcast<u32>(load));
  // (counted in steps: the particles crush once it has lasted 3 steps (50 ms), time for the rest to come to bear.
  // At once, a 21.6 t block set on the peak of a sleeping heap crushed the 30 it first came down on, each woken
  // that step and still 0.3 r deep; GPU test heapcreep, slab awake)
  let wasOver = ld(P.o_psup + 11u * P.psup_n + s);
  st(P.o_psup + 11u * P.psup_n + s, select(0u, min(wasOver + 1u, 1000u), area > 0.0 && load > 0.95 * RUBBLE_BEAR * area));
}
@compute @workgroup_size(64) fn p_apply(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= P.np1 || R(i, 1).w != 1.0) { return; }
  let x = R(i, 5).xyz; let r0 = R(i, 0);
  pst(pcand(i) + 38u, pld(pcand(i) + 39u));
  var v = (x - R(i, 3).xyz) / P.p_h;
  let l = length(v); if (l > 40.0) { v *= 40.0 / l; }
  setR(i, 0, vec4f(x, r0.w)); setR(i, 1, vec4f(v, 1.0));
}
@compute @workgroup_size(64) fn p_finish(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= P.np1) { return; }
  let r0 = R(i, 0); let r1 = R(i, 1); var q = R(i, 2); let st = r1.w;
  if (st == 1.0) {
    let r3 = R(i, 3); let r4 = R(i, 4); let touch = R(i, 5).w;
    var v = r1.xyz; var w = r4.xyz; var t = 0.0; var s = 1.0;
    // at rest = touching something and staying within 1 cm of where the timer restarted (a particle wedged among
    // sleeping ones may jitter in place; that goes nowhere)
    let r6 = R(i, 6);
    if (touch > 0.0) { w = cross(vec3f(0.0, 1.0, 0.0), v) / max(r0.w, 0.01); }
    // (nor while it holds up a moving body: asleep it would no longer bear it, and the body would sink through the
    // heap; it sleeps once the body does. GPU test heapcreep, slab awake: through to the slab, 220 crushed)
    if (touch > 0.0 && touch < 1.5 && distance(r0.xyz, r6.xyz) < 0.01 && length(v) < 2.0) { t = r3.w + P.dt; } else { setR(i, 6, vec4f(r0.xyz, r6.w)); }
    if (t > 0.5 || r0.y < -50.0) { s = 2.0; v = vec3f(0.0); w = vec3f(0.0); }
    if (r6.w > 0.0) { s = 3.0; v = vec3f(0.0); }
    q = normalize(q + 0.5 * P.dt * qmul(vec4f(w, 0.0), q));
    setR(i, 1, vec4f(v, s)); setR(i, 2, q); setR(i, 3, vec4f(r3.xyz, t)); setR(i, 4, vec4f(w, r4.w));
  }
  let v1 = R(i, 1);
  // what is drawn, packed after the records: 3 vec4 per particle
  let o = P.p_cap * PR + i * 3u;
  prt[o] = vec4f(r0.xyz, v1.x); prt[o + 1u] = vec4f(v1.yz, q.xy); prt[o + 2u] = vec4f(q.zw, v1.w, r0.w);
}
`;

// Writers of the indirect-dispatch arguments, in passes of their own so the args buffer is never
// writable and indirect in the same dispatch.
const WGSL_ARGS = WGSL.replace(/\/\/ -{10,} broad phase[\s\S]*$/, '') + /* wgsl */`
@group(2) @binding(0) var<storage, read_write> args: array<u32>;
@compute @workgroup_size(1) fn man_args() {
  let n = min(ld(C_MAN), P.max_man);
  args[96] = (n + 63u) / 64u; args[97] = 1u; args[98] = 1u;
}
@compute @workgroup_size(1) fn pair_args() {
  let n = min(ld(C_NPAIR), P.max_pairs);
  args[99] = (n + 63u) / 64u; args[100] = 1u; args[101] = 1u;
}
@compute @workgroup_size(1) fn hull_args() {
  let n = min(ld(P.o_hl), P.max_pairs);
  args[102] = (n + 63u) / 64u; args[103] = 1u; args[104] = 1u;
}
@compute @workgroup_size(1) fn color_scan() {
  var run = 0u;
  for (var c = 0u; c <= P.color_limit; c++) {
    let n = ld(P.o_ccount + c);
    st(P.o_cstart + c, run); run += n;
    args[c * 3u] = (n + 63u) / 64u; args[c * 3u + 1u] = 1u; args[c * 3u + 2u] = 1u;
  }
}
`;

// ---- small rigid-body maths for building bodies on the CPU
const q2m = (q) => { const [x, y, z, w] = q; return [
  [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]; };   // row-major
const m2q = (m) => {
  const tr = m[0][0] + m[1][1] + m[2][2]; let q;
  if (tr > 0) { const s = Math.sqrt(tr + 1) * 2; q = [(m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s, 0.25 * s]; }
  else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) { const s = Math.sqrt(1 + m[0][0] - m[1][1] - m[2][2]) * 2; q = [0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s, (m[2][1] - m[1][2]) / s]; }
  else if (m[1][1] > m[2][2]) { const s = Math.sqrt(1 + m[1][1] - m[0][0] - m[2][2]) * 2; q = [(m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s, (m[0][2] - m[2][0]) / s]; }
  else { const s = Math.sqrt(1 + m[2][2] - m[0][0] - m[1][1]) * 2; q = [(m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s, (m[1][0] - m[0][1]) / s]; }
  const l = Math.hypot(...q); return q.map(v => v / l);
};
const qmulJ = (a, b) => [a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1], a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
                         a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3], a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]];
const qconj = (q) => [-q[0], -q[1], -q[2], q[3]];
const mv = (m, v) => [0, 1, 2].map(r => m[r][0] * v[0] + m[r][1] * v[1] + m[r][2] * v[2]);
const mtv = (m, v) => [0, 1, 2].map(c => m[0][c] * v[0] + m[1][c] * v[1] + m[2][c] * v[2]);
// symmetric 3x3 eigen-decomposition (cyclic Jacobi): returns eigenvalues and eigenvector columns (row-major V)
function eigSym(A) {
  const a = A.map(r => r.slice()); const V = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 32; sweep++) {
    const off = a[0][1] ** 2 + a[0][2] ** 2 + a[1][2] ** 2;
    if (off < 1e-24 * (a[0][0] ** 2 + a[1][1] ** 2 + a[2][2] ** 2 + 1e-30)) break;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
      if (Math.abs(a[p][q]) < 1e-30) continue;
      const th = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < 3; k++) { const akp = a[k][p], akq = a[k][q]; a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq; }
      for (let k = 0; k < 3; k++) { const apk = a[p][k], aqk = a[q][k]; a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk; }
      for (let k = 0; k < 3; k++) { const vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq; }
    }
  }
  // right-handed frame
  const det = V[0][0] * (V[1][1] * V[2][2] - V[1][2] * V[2][1]) - V[0][1] * (V[1][0] * V[2][2] - V[1][2] * V[2][0]) + V[0][2] * (V[1][0] * V[2][1] - V[1][1] * V[2][0]);
  if (det < 0) for (let k = 0; k < 3; k++) V[k][2] = -V[k][2];
  return { vals: [a[0][0], a[1][1], a[2][2]], V };
}

// Build GPU body + shape records from bodies given in world space:
//   { shapes: [{ p: [x,y,z], q: [x,y,z,w], h: [hx,hy,hz], mat }], fixed?: bool, v?: [..], w?: [..] }
// ---- convex chunks: a library of fracture patterns of the cube [-1,1]^3 (Voronoi cells of 2-6 seeds), each
// cell a convex hull. A shape is a cell stretched to its brick (the cells of one brick tile it exactly). Each
// cell is stored in its own bounding box, mapped to [-1,1]^3, so the shape's h is the cell's half size.
const HULL = { V: 16, F: 12, FV: 8, S: 24, D: 12, STRIDE: 80 };   // limits, and vec4s per hull
function hullLibrary(nPatterns = 16, seed = 7) {
  let a = seed >>> 0;
  const rnd = () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const sub = (p, q) => [p[0] - q[0], p[1] - q[1], p[2] - q[2]], dot = (p, q) => p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
  const cross = (p, q) => [p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0]];
  const len = (p) => Math.hypot(p[0], p[1], p[2]), nrm = (p) => { const l = len(p); return [p[0] / l, p[1] / l, p[2] / l]; };
  // a convex polytope from half-spaces n.x <= d: vertices, faces (ordered polygons), segments, edge directions
  function polytope(planes) {
    const V = [], eps = 1e-6;
    for (let i = 0; i < planes.length; i++) for (let j = i + 1; j < planes.length; j++) for (let k = j + 1; k < planes.length; k++) {
      const [a1, b1, c1] = [planes[i], planes[j], planes[k]];
      const det = dot(a1.n, cross(b1.n, c1.n)); if (Math.abs(det) < 1e-9) continue;
      const x = [0, 1, 2].map(q => (a1.d * cross(b1.n, c1.n)[q] + b1.d * cross(c1.n, a1.n)[q] + c1.d * cross(a1.n, b1.n)[q]) / det);
      if (planes.every(p => dot(p.n, x) <= p.d + 1e-7) && !V.some(v => len(sub(v, x)) < 1e-6)) V.push(x);
    }
    const faces = [];
    for (const p of planes) {
      const idx = V.map((v, i) => [v, i]).filter(([v]) => Math.abs(dot(p.n, v) - p.d) < 1e-6).map(([, i]) => i);
      if (idx.length < 3) continue;
      const c = [0, 1, 2].map(q => idx.reduce((s, i) => s + V[i][q], 0) / idx.length);
      const u = nrm(sub(V[idx[0]], c)), w = cross(p.n, u);
      idx.sort((i, j) => Math.atan2(dot(sub(V[i], c), w), dot(sub(V[i], c), u)) - Math.atan2(dot(sub(V[j], c), w), dot(sub(V[j], c), u)));
      faces.push({ n: p.n, d: p.d, v: idx });
    }
    const seg = [];
    for (const f of faces) for (let k = 0; k < f.v.length; k++) { const i = f.v[k], j = f.v[(k + 1) % f.v.length]; if (!seg.some(([x, y]) => (x === i && y === j) || (x === j && y === i))) seg.push([i, j]); }
    const dirs = [];
    for (const [i, j] of seg) { const d = nrm(sub(V[j], V[i])); if (!dirs.some(e => Math.abs(dot(e, d)) > 0.9995)) dirs.push(d); }
    return { V, faces, seg, dirs };
  }
  // volume, centroid and covariance per unit volume about the centroid (in the cell's box coordinates): from
  // tetrahedra (centroid of the vertices, face triangle); a tetrahedron's integral of x x^T is V/20 (sum of the
  // corners' x x^T + their sum's outer product)
  function massProps(V, faces) {
    const o = [0, 1, 2].map(q => V.reduce((s, v) => s + v[q], 0) / V.length);
    let vol = 0; const m1 = [0, 0, 0], m2 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (const f of faces) for (let t = 1; t + 1 < f.v.length; t++) {
      const P4 = [o, V[f.v[0]], V[f.v[t]], V[f.v[t + 1]]];
      const tv = Math.abs(dot(sub(P4[1], o), cross(sub(P4[2], o), sub(P4[3], o)))) / 6;
      vol += tv;
      const S = [0, 1, 2].map(q => P4.reduce((s, v) => s + v[q], 0));
      for (let q = 0; q < 3; q++) m1[q] += tv * S[q] / 4;
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) m2[r][c] += tv / 20 * (P4.reduce((s, v) => s + v[r] * v[c], 0) + S[r] * S[c]);
    }
    const cen = m1.map(x => x / vol);
    const cov = m2.map((row, r) => row.map((x, c) => x / vol - cen[r] * cen[c]));
    return { vol, cen, cov };
  }
  const cube = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]].map(n => ({ n, d: 1 }));
  const hulls = [polytope(cube)], patterns = [];
  hulls[0].box = { c: [0, 0, 0], h: [1, 1, 1] }; hulls[0].vol = 1; hulls[0].cen = [0, 0, 0]; hulls[0].cov = [[1 / 3, 0, 0], [0, 1 / 3, 0], [0, 0, 1 / 3]];
  for (let p = 0, tries = 0; p < nPatterns && tries < 40000; tries++) {
    const k = 2 + (p % 5);   // 2-6 cells
    const seeds = Array.from({ length: k }, () => [0, 1, 2].map(() => rnd() * 1.6 - 0.8));
    const cells = [];
    let ok = true;
    for (let i = 0; i < k && ok; i++) {
      const planes = cube.slice();
      for (let j = 0; j < k; j++) if (j !== i) {
        const n = nrm(sub(seeds[j], seeds[i])), m = [0, 1, 2].map(q => (seeds[i][q] + seeds[j][q]) / 2);
        planes.push({ n, d: dot(n, m) });
      }
      const P = polytope(planes);
      const lo = [0, 1, 2].map(q => Math.min(...P.V.map(v => v[q]))), hi = [0, 1, 2].map(q => Math.max(...P.V.map(v => v[q])));
      const c = lo.map((l, q) => (l + hi[q]) / 2), h = lo.map((l, q) => (hi[q] - l) / 2);
      if (P.V.length > HULL.V || P.faces.length > HULL.F || P.seg.length > HULL.S || P.dirs.length > HULL.D || P.faces.some(f => f.v.length > HULL.FV) || h.some(x => x < 0.15)) { ok = false; break; }
      // into the cell's own box, [-1,1]^3 (planes: n.(c + h x) <= d  ->  (n h).x <= d - n.c)
      const V = P.V.map(v => v.map((x, q) => (x - c[q]) / h[q]));
      const faces = P.faces.map(f => { const n0 = f.n.map((x, q) => x * h[q]), l = len(n0); return { n: n0.map(x => x / l), d: (f.d - dot(f.n, c)) / l, v: f.v }; });
      const dirs = P.dirs.map(d => nrm(d.map((x, q) => x / h[q])));
      const mp = massProps(V, faces);
      cells.push({ V, faces, seg: P.seg, dirs, box: { c, h }, vol: mp.vol / 8, cen: mp.cen, cov: mp.cov });
    }
    if (!ok) continue;
    patterns.push(cells.map(cl => { hulls.push(cl); return hulls.length - 1; }));
    p++;
  }
  // packed for the GPU: per hull 80 vec4: [nv, nf, nseg, ndir] · 16 vertices · 12 planes (n, d) · 12 faces x 2
  // (8 vertex indices) · 24 segments (2 per vec4) · 12 directions · face vertex counts
  const data = new Float32Array(hulls.length * HULL.STRIDE * 4);
  hulls.forEach((H, i) => {
    const o = i * HULL.STRIDE * 4;
    data.set([H.V.length, H.faces.length, H.seg.length, H.dirs.length], o);
    H.V.forEach((v, k) => data.set([...v, 0], o + 4 * (1 + k)));
    H.faces.forEach((f, k) => {
      data.set([...f.n, f.d], o + 4 * (17 + k));
      const ix = f.v.concat(new Array(8 - f.v.length).fill(0));
      data.set(ix.slice(0, 4), o + 4 * (29 + 2 * k)); data.set(ix.slice(4, 8), o + 4 * (30 + 2 * k));
    });
    data.set([H.V.length, H.faces.length, H.seg.length, H.dirs.length], o);
    H.seg.forEach(([x, y], k) => { data[o + 4 * (53 + (k >> 1)) + 2 * (k & 1)] = x; data[o + 4 * (53 + (k >> 1)) + 2 * (k & 1) + 1] = y; });
    H.dirs.forEach((d, k) => data.set([...d, 0], o + 4 * (65 + k)));
    // the face vertex counts: 12 numbers in vec4s 77..79
    H.faces.forEach((f, k) => { data[o + 4 * 77 + k] = f.v.length; });
  });
  return { hulls, patterns, data, count: hulls.length };
}
const HULLS = hullLibrary();

function buildBodies(list) {
  const bodies = [], shapes = [];
  for (const b of list) {
    let M = 0, com = [0, 0, 0];
    const ms = b.shapes.map(s => (b.fixed ? 0 : MATERIALS[s.mat].density) * 8 * s.h[0] * s.h[1] * s.h[2] * (s.hull ? HULLS.hulls[s.hull].vol : 1));
    // where each shape's mass is: a chunk's centroid (its box centre can lie outside a wedge)
    const mc = b.shapes.map(s => { if (!s.hull) return s.p; const c = HULLS.hulls[s.hull].cen, R = q2m(s.q), o = c.map((x, k) => x * s.h[k]);
      return [0, 1, 2].map(i => s.p[i] + R[i][0] * o[0] + R[i][1] * o[1] + R[i][2] * o[2]); });
    b.shapes.forEach((s, k) => { M += ms[k]; for (let i = 0; i < 3; i++) com[i] += ms[k] * mc[k][i]; });
    if (M > 0) com = com.map(c => c / M); else { com = [0, 0, 0]; b.shapes.forEach(s => { for (let i = 0; i < 3; i++) com[i] += s.p[i] / b.shapes.length; }); }
    let q = [0, 0, 0, 1], il = [0, 0, 0];
    if (M > 0) {
      const I = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      b.shapes.forEach((s, k) => {
        // the shape's inertia about its own mass centre, in its frame: from the covariance (a box: h^2/3), stretched
        // by h; then turned into the body frame and moved to the body's centre of mass
        const m = ms[k], cov = (s.hull ? HULLS.hulls[s.hull] : HULLS.hulls[0]).cov;
        const C = cov.map((row, r) => row.map((x, c) => m * x * s.h[r] * s.h[c]));
        const tr = C[0][0] + C[1][1] + C[2][2];
        const Il = C.map((row, r) => row.map((x, c) => (r === c ? tr : 0) - x));
        const R = q2m(s.q), d = [0, 1, 2].map(i => mc[k][i] - com[i]), dd = d[0] ** 2 + d[1] ** 2 + d[2] ** 2;
        for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
          let v = 0; for (let a2 = 0; a2 < 3; a2++) for (let b2 = 0; b2 < 3; b2++) v += R[r][a2] * Il[a2][b2] * R[c][b2];
          I[r][c] += v + m * ((r === c ? dd : 0) - d[r] * d[c]);
        }
      });
      const { vals, V } = eigSym(I);
      q = m2q(V); il = vals.map(v => v > 1e-12 ? 1 / v : 0);
    }
    const Rb = q2m(q);
    const bi = bodies.length + 1;
    bodies.push({ p: com, q, im: M > 0 ? 1 / M : 0, il, v: b.v || [0, 0, 0], w: b.w || [0, 0, 0], mass: M });
    for (const s of b.shapes) {
      shapes.push({ body: bi, mat: s.mat, col: s.col ?? s.mat, hull: s.hull || 0, lp: mtv(Rb, [0, 1, 2].map(i => s.p[i] - com[i])), lq: qmulJ(qconj(q), s.q), h: s.h });
    }
  }
  return { bodies, shapes };
}

class GpuWorld {
  // list: bodies as for buildBodies
  // the convex chunk library: { hulls (vertices, faces, box, volume share), patterns (lists of hull ids that tile
  // a brick: give each piece shape { hull, p, h } from hullPiece) }
  static get hulls() { return HULLS; }
  // the shapes of a brick (centre p, rotation q, half size h) broken by a pattern: one per cell, tiling it
  static hullShapes(p, q, h, pattern, mat, col) {
    const R = q2m(q);
    return HULLS.patterns[pattern % HULLS.patterns.length].map(id => {
      const bx = HULLS.hulls[id].box, o = bx.c.map((c, k) => c * h[k]);
      return { p: [0, 1, 2].map(i => p[i] + R[i][0] * o[0] + R[i][1] * o[1] + R[i][2] * o[2]), q, h: bx.h.map((x, k) => x * h[k]), hull: id, mat, col };
    });
  }
  static async create(device, list, opt = {}) {
    const w = new GpuWorld();
    await w.init(device, list, opt);
    if (w.manIndirect == null) await w.calibrateDispatch();
    return w;
  }
  // The kernels run once a step over the contacts are sized either by capacity (direct: threads past the real
  // count return at once) or by the count itself (indirect: the browser validates each such dispatch). Which costs
  // less depends on the GPU and the browser, so it is measured here, once: 40 of each, the fastest of 3 runs.
  // (Only the timing differs: the kernels skip work past their real counts either way.)
  async calibrateDispatch() {
    const d = this.device;
    try {
      const mod = d.createShaderModule({ code: '@group(0) @binding(0) var<storage, read_write> o: array<u32>; @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3u) { if (g.x >= o[0]) { return; } o[1] = o[1] + 1u; }' });
      const pl = d.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'main' } });
      const buf = d.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      d.queue.writeBuffer(buf, 0, new Uint32Array([1000, 0, 0, 0]));
      const args = d.createBuffer({ size: 16, usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST });
      d.queue.writeBuffer(args, 0, new Uint32Array([16, 1, 1, 0]));
      const bg = d.createBindGroup({ layout: pl.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: buf } }] });
      const perMan = Math.max(1, Math.ceil(this.maxMan / 64));
      const time = async (indirect) => {
        const e = d.createCommandEncoder(); const p = e.beginComputePass(); p.setPipeline(pl); p.setBindGroup(0, bg);
        for (let k = 0; k < 40; k++) { if (indirect) p.dispatchWorkgroupsIndirect(args, 0); else p.dispatchWorkgroups(perMan); }
        p.end(); const t = performance.now(); d.queue.submit([e.finish()]); await d.queue.onSubmittedWorkDone(); return performance.now() - t;
      };
      await time(false); await time(true);
      let a = Infinity, b = Infinity; for (let k = 0; k < 3; k++) { a = Math.min(a, await time(false)); b = Math.min(b, await time(true)); }
      buf.destroy(); args.destroy();
      this.manIndirect = b < a; this.dispatchCal = { direct_ms: +a.toFixed(3), indirect_ms: +b.toFixed(3), workgroups: perMan };
    } catch (e) { this.manIndirect = false; }
  }
  // single-box bodies from 10-float poses (p, q, h), as the wasm pose() export gives them
  static boxes(poses, mat = MAT.reference) {
    const list = [];
    for (let i = 0; i < poses.length / 10; i++) {
      const o = i * 10;
      list.push({ shapes: [{ p: [poses[o], poses[o + 1], poses[o + 2]], q: [poses[o + 3], poses[o + 4], poses[o + 5], poses[o + 6]], h: [poses[o + 7], poses[o + 8], poses[o + 9]], mat }] });
    }
    return list;
  }

  async init(device, list, opt) {
    const d = this.device = device;
    const { bodies, shapes } = buildBodies(list);
    this.bodies = bodies; this.shapes = shapes;
    const n = bodies.length, ns = shapes.length;
    this.n = n; this.n1 = n + 1; this.ns = ns; this.ns1 = ns + 1;
    // capacity: bodies and shapes can be added later (debris), up to these counts
    this.capB = Math.max(n, opt.maxBodies || 0); this.capS = Math.max(ns, opt.maxShapes || 0);
    const cb1 = this.capB + 1, cs1 = this.capS + 1;
    this.substeps = opt.substeps || 12;
    this.colorCheck = !!opt.colorCheck;
    this.colorLimit = opt.colorLimit || 24;
    // most sequential colours per solve; contacts beyond go to the parallel (Jacobi) pass. Each colour is a
    // dispatch per sub-step, and dispatch count is what a step costs at this scale
    this.maxColors = opt.maxColors || 16;
    this.colorLimit = Math.min(this.colorLimit, this.maxColors);
    this.indirect = !!opt.indirect;   // the old indirect-dispatch path (for comparison)
    this.manIndirect = opt.manIndirect ?? null;   // (the once-a-step contact kernels: see calibrateDispatch)
    this.anchor = opt.anchor ?? true;
    this.sleep = opt.sleep ?? true;
    this.wakeSpeed = opt.wakeSpeed ?? 0.25;   // m/s at the contact: slower bodies do not wake what they touch
    this.levels = 4;
    this.maxMan = opt.maxContacts || Math.max(64, Math.ceil((opt.manPerShape || 4) * this.capS));
    // (pairs beyond capacity are lost: a lost ground pair let a chunk fall through the ground; stats.pairsLost)
    this.maxPairs = opt.maxPairs || Math.max(128, Math.ceil((opt.pairsPerShape || 24) * this.capS));
    let G = 256; while (G < 2 * cs1) G *= 2; this.G = G;
    let T = 256; while (T < 2 * this.maxMan) T *= 2; this.T = T;
    // base cell: a typical brick with motion margin; larger shapes go to coarser levels (x4 each)
    const ext = shapes.map(s => 2 * Math.hypot(...s.h)).sort((a, b) => a - b);
    this.cell = Math.max(0.5, (ext[Math.floor(ext.length * 0.5)] || 1) + 0.25);
    let o = 16;
    const L = {};
    const take = (k, len) => { L[k] = o; o += len; };
    take('gcount', G); take('gstart', G); take('gfill', G); take('cawake', G); take('cnear', CNEAR + 1);
    take('sorted', cs1); take('cell', cs1); take('mask', cb1);
    take('ccount', 33); take('cstart', 33); take('cfill', 33);
    take('order', this.maxMan); take('mcolor', this.maxMan);
    take('t0', T); take('t1', T);
    take('pairs', 2 * this.maxPairs); take('bcount', cb1); take('bstart', cb1); take('bfill', cb1);
    take('blist', 2 * this.maxMan); take('par', cb1); take('mint', cb1); take('ocount', cb1); take('acc', 6 * cb1);
    // contact reduction: a table of body pairs (key, count, reduced index), the reduced pairs' picks, each
    // contact's pair slot
    let TP = 1024; while (TP < 2 * this.maxMan) TP *= 2; this.TP = TP;
    take('ptab', 3 * TP); take('big', 64 * 4096); take('mslot', this.maxMan);
    take('hl', 1 + this.maxPairs); take('crushed', cb1); take('released', cb1); take('crush', 12 * 256); take('pinch', 8 * cb1); take('level', cb1); take('level2', cb1); if (opt.maxParticles) take('psup', 13 * cs1);
    this.maxEvents = 512; take('events', 12 * this.maxEvents);
    this.maxInherit = 4096; take('inherit', 12 * this.maxInherit);
    take('plan', 24 * cb1); take('spart', cs1); take('snap', 1 + 4 * 256);
    take('claim', 32 * cb1); take('claim2', 32 * cb1); take('sorted2', cs1); take('evhist', 129); if (this.colorCheck) take('cchk', 1 + 32 * cb1); take('mtry', this.maxMan + 1);
    this.L = L; this.nInts = o;

    const bod = new Float32Array(cb1 * 20), sol = new Float32Array(cb1 * 24), shp = new Float32Array(cs1 * 32);
    bodies.forEach((b, k) => {
      const i = k + 1;
      bod.set([...b.p, b.im, ...b.q, ...b.il, 0, 0, ...b.p, ...b.q], i * 20);
      sol.set([...b.v, 0, ...b.w, 0], i * 24);
    });
    shapes.forEach((s, k) => {
      const i = k + 1;
      shp.set([s.body, s.mat, 0, s.hull || 0, ...s.lp, 0, ...s.lq, ...s.h, 0], i * 32);
    });
    const U = GPUBufferUsage;
    const buf = (size, usage, data) => { const b = d.createBuffer({ size, usage }); if (data) d.queue.writeBuffer(b, 0, data); return b; };
    this.bodBuf = buf(bod.byteLength, U.STORAGE | U.COPY_DST | U.COPY_SRC, bod);
    this.solBuf = buf(sol.byteLength, U.STORAGE | U.COPY_DST | U.COPY_SRC, sol);
    this.shpBuf = buf(shp.byteLength + HULLS.data.byteLength, U.STORAGE | U.COPY_DST | U.COPY_SRC, shp);
    d.queue.writeBuffer(this.shpBuf, shp.byteLength, HULLS.data); this.oHull = shp.byteLength / 16;
    this.intBuf = buf(this.nInts * 4, U.STORAGE | U.COPY_DST | U.COPY_SRC);
    this.manBuf = buf(2 * this.maxMan * 17 * 16, U.STORAGE | U.COPY_SRC);
    this.coldBuf = buf(2 * this.maxMan * 6 * 16, U.STORAGE | U.COPY_SRC);
    this.parBuf = buf(1024, U.UNIFORM | U.COPY_DST);
    const cu = new Uint32Array(64 * 33); for (let c = 0; c < 33; c++) cu[c * 64] = c;
    this.colBuf = buf(256 * 33, U.UNIFORM | U.COPY_DST, cu);
    this.argBuf = buf(128 * 4, U.STORAGE | U.INDIRECT);
    this.evSpeed = opt.eventSpeed ?? 2.0; this.evEnergy = opt.eventEnergy ?? 0;   // events off unless asked for
    // a ring of stats read-back buffers: the browser delivers mappings late, so one pending read must not hold
    // back the next step's stats and impact events
    this.statBufs = [0, 1, 2].map(() => ({ buf: buf(64 + 48 * this.maxEvents + 33 * 4 + 48 * 256 + 4 + 16 * 256, U.COPY_DST | U.MAP_READ), busy: false }));
    this.snaps = [];
    this.crushes = [];
    this.colCounts = new Array(33).fill(0);
    this.statBuf = this.statBufs[0].buf; this.statSeq = 0; this.statLatest = 0; this.inputs = 0;
    this.events = [];
    this.stats = { manifolds: 0, overflow: 0, colors: 0, pairs: 0, dropped: 0, candidates: 0, awake: n };
    this.statPending = false;

    const sto = (binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } });
    const bgl0 = d.createBindGroupLayout({ entries: [sto(0), sto(1), sto(2), sto(3),
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }, sto(5), sto(6)] });
    const bgl1 = d.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 4 } }] });
    const bgl2 = d.createBindGroupLayout({ entries: [sto(0)] });
    const mainL = d.createPipelineLayout({ bindGroupLayouts: [bgl0, bgl1] });
    const argsL = d.createPipelineLayout({ bindGroupLayouts: [bgl0, bgl1, bgl2] });
    const ent = (i, b) => ({ binding: i, resource: { buffer: b } });
    this.bg0 = d.createBindGroup({ layout: bgl0, entries: [ent(0, this.bodBuf), ent(1, this.intBuf), ent(2, this.manBuf), ent(3, this.coldBuf), ent(4, this.parBuf), ent(5, this.solBuf), ent(6, this.shpBuf)] });
    this.bg1 = d.createBindGroup({ layout: bgl1, entries: [{ binding: 0, resource: { buffer: this.colBuf, size: 4 } }] });
    this.bg2 = d.createBindGroup({ layout: bgl2, entries: [ent(0, this.argBuf)] });
    const compile = async (code, tag) => {
      const m = d.createShaderModule({ code });
      const errs = (await m.getCompilationInfo()).messages.filter(x => x.type === 'error');
      if (errs.length) throw new Error(tag + ': ' + errs.map(x => `${x.lineNum}:${x.linePos} ${x.message}`).join('\n'));
      return m;
    };
    // particles (granular debris)
    this.capP = opt.maxParticles || 0; this.np = 0; this.freeParticles = []; this.pBlasts = [];
    this.psub = opt.particleSubsteps || 4;
    this.pcell = 2 * (opt.particleMaxRadius || 0.3) + 0.1;
    let PG = 256; while (PG < 2 * Math.max(1, this.capP)) PG *= 2; this.PG = PG; this.PGeff = 256;
    const cp = Math.max(1, this.capP);
    this.prtBuf = buf(cp * 112 + cp * 48, U.STORAGE | U.COPY_DST | U.COPY_SRC);
    this.pintBuf = buf((3 * PG + 2 * cp + 41 * cp) * 4, U.STORAGE | U.COPY_DST);
    const bglP = d.createBindGroupLayout({ entries: [sto(0), sto(1)] });
    const partL = d.createPipelineLayout({ bindGroupLayouts: [bgl0, bgl1, bglP] });
    this.bgP = d.createBindGroup({ layout: bglP, entries: [ent(0, this.prtBuf), ent(1, this.pintBuf)] });
    const mod = await compile(WGSL, 'WGSL'), modA = await compile(WGSL_ARGS, 'WGSL args');
    const modP = this.capP ? await compile(WGSL_P, 'WGSL particles') : null;
    const mk = (entry, layout = mainL, module = mod) => d.createComputePipelineAsync({ layout, compute: { module, entryPoint: entry } });
    const names = ['prep_bodies', 'prep_shapes', 'scan', 'scatter', 'pairs', 'narrow', 'narrow_hull', 'color_init', 'color_round', 'color_claim_s', 'color_claim', 'color_tie', 'color_resolve', 'color_count', 'color_check', 'ev_count', 'ev_cut', 'blist_sort', 'ovf_sort', 'sort_cells', 'color_scatter', 'adj_scan', 'integrate_warm',
                   'integrate_pos', 'finish', 'solve', 'relax', 'restitute', 'solve_ovf', 'relax_ovf', 'restitute_ovf', 'uf_union', 'uf_min', 'uf_sleep', 'ovf_apply', 'events', 'wake_all', 'inherit', 'retire', 'red_count', 'red_big', 'red_score', 'red_pick', 'red_apply', 'crush', 'crush_release', 'pinch_acc', 'pinch', 'level_pass', 'snap_acc', 'sup_acc', 'snap_split', 'snap_shapes'];
    this.pl = {};
    await Promise.all([...names.map(async k => { this.pl[k] = await mk(k); }),
      ...['man_args', 'pair_args', 'hull_args', 'color_scan'].map(async k => { this.pl[k] = await mk(k, argsL, modA); }),
      ...(modP ? ['p_gather', 'p_place', 'p_prep', 'p_scan', 'p_scatter', 'p_sort', 'p_integrate', 'p_contact', 'p_apply', 'p_finish', 'p_support', 'p_sup_load'] : []).map(async k => { this.pl[k] = await mk(k, partL, modP); })]);
    this.ts = d.features.has('timestamp-query');
    this.stages = ['broad', 'narrow', 'chunks', 'colour', 'sub-steps', 'particles', 'sleep'];
    if (this.ts) {
      this.qs = d.createQuerySet({ type: 'timestamp', count: 32 });
      this.qBuf = buf(256, U.QUERY_RESOLVE | U.COPY_SRC);
      this.qRead = buf(256, U.COPY_DST | U.MAP_READ);
      this.qPending = false; this.gpuMs = null;
    }
    this.frame = 0;
  }

  params(dt) {
    const sub = this.substeps, h = dt / sub, invH = 1 / h;
    const hz = invH / 6, omega = 2 * Math.PI * hz, zeta = 10;
    const a1 = 2 * zeta + h * omega, a2 = h * omega * a1, a3 = 1 / (1 + a2);
    const f = new Float32Array(256), u = new Uint32Array(f.buffer);
    const cur = this.frame & 1, L = this.L;
    f.set([0, -9.81, 0, h, invH, omega / a1, a2 * a3, a3, this.maxPush ?? 1.0, this.restOff ? 1e9 : 1.0, 0.02, dt, this.cell]);
    // the grid is sized by the shapes there are, not by capacity (it is cleared and scanned every step)
    let g = 256; while (g < 2 * this.ns1 && g < this.G) g *= 2; this.Geff = g;
    u[13] = this.n1; u[14] = g - 1; u[15] = this.maxMan;
    u[16] = this.T - 1; u[17] = this.colorLimit; f[18] = 0.25 * Math.PI * invH; u[19] = cur * this.maxMan;
    u[20] = (1 - cur) * this.maxMan; u[21] = L.gcount; u[22] = L.gstart; u[23] = L.gfill;
    u[24] = L.sorted; u[25] = L.cell; u[26] = L.mask; u[27] = L.ccount;
    u[28] = L.cstart; u[29] = L.cfill; u[30] = L.order; u[31] = L.mcolor;
    u[32] = cur ? L.t1 : L.t0; u[33] = cur ? L.t0 : L.t1; u[34] = this.anchor ? 1 : 0; u[35] = this.maxPairs;
    u[36] = L.pairs; u[37] = L.bcount; u[38] = L.bstart; u[39] = L.bfill;
    u[40] = L.blist; u[41] = L.par; u[42] = L.mint; u[43] = this.sleep ? 1 : 0;
    f[44] = 0.5; f[45] = 0.05; f[46] = 0.08; u[47] = this.ns1;
    u[48] = this.levels; u[49] = GROUND_MAT; f[50] = 0.01; f[51] = 0.02;
    u[52] = L.ocount; u[53] = L.acc; u[54] = L.events; u[55] = this.maxEvents; u[58] = L.cawake; u[59] = L.cnear;
    f[56] = this.evSpeed; f[57] = this.evEnergy;
    u[60] = this.inheritCount || 0; u[61] = this.L.inherit; f[62] = this.wakeSpeed;
    // the pair table sized by the contacts there were (as the grid), up to its capacity
    let tp = 1024; while (tp < 2 * ((this.stats.manifolds || 0) * 1.5 + 2048) && tp < this.TP) tp *= 2; this.TPeff = tp;
    u[128] = L.ptab; u[129] = tp - 1; u[130] = L.big; u[131] = L.mslot;
    u[136] = this.oHull; u[137] = L.hl; f[139] = this.ovfPush ?? 1; u[138] = (this.dbgBits || 0) | (this.shock === false ? 16 : 0);
    u[140] = L.plan; u[141] = L.spart; u[142] = L.snap; u[143] = L.claim; u[144] = L.mtry; u[145] = L.claim2; u[146] = L.sorted2; u[147] = L.evhist; u[148] = L.cchk || 0; u[149] = L.pinch; u[150] = this.frame; u[151] = L.level; u[152] = L.level2; u[153] = this.capP && this.support !== false ? L.psup : 0; u[154] = this.capS + 1; u[155] = L.released;   // (psup's stride: the shape capacity, so its regions stay put as shapes are added)
    u[132] = L.crushed; u[133] = L.crush; f[134] = this.crushRatio ?? 20; f[135] = this.crushDv ?? 3;
    MATERIALS.forEach((m, i) => f.set([m.mu, m.e, 0, 0], 64 + i * 4));
    if (this.capP) {
      let pgE = 256; while (pgE < 2 * this.np && pgE < this.PG) pgE *= 2; this.PGeff = pgE;
      u[96] = this.np; f[97] = this.pcell; u[98] = pgE - 1; u[99] = this.wakeReq ? 1 : 0;
      f[100] = dt / this.psub; u[102] = this.capP;
      const bl = (this.pBlasts || []).splice(0, 3); u[101] = bl.length;
      bl.forEach((b, k) => { f.set([...b.c, b.reach], 104 + k * 8); f[108 + k * 8] = b.speed; });
    }
    return f;
  }

  // Encode one 60 Hz step into `enc` (several compute passes; one timestamp pair per stage).
  encode(enc, dt = 1 / 60) {
    const d = this.device;
    // Nothing awake at the start of a step whose stats we have, and nothing changed from JS since: the world is
    // at rest and stays so; the step costs nothing (a settled city of rubble is free)
    if (this.sleep && this.stats.awake === 0 && !(this.stats.pActive > 0) && !this.pBlasts.length && this.statsInputs === this.inputs && !this.wakeReq && !(this.pendingInherit && this.pendingInherit.length)) {
      this.frame++; this.skipped = (this.skipped || 0) + 1; return;
    }
    if (this.pendingInherit && this.pendingInherit.length) {
      // (one generation a step: an entry whose parent is placed by this same batch would read that parent while
      // it is being placed, in whatever order the threads ran; it waits for the next step)
      const kids = new Set(), parents = new Set(); let cut = 0;
      for (const e of this.pendingInherit) {
        if (cut >= this.maxInherit || kids.has(e[1] & 0x3fffffff) || parents.has(e[0])) break;
        parents.add(e[1] & 0x3fffffff);
        if (e[0] !== 0 && (e[0] & 0x80000000) === 0) kids.add(e[0]);   // (high bit: a particle, never a parent)
        cut++;
      }
      const list = this.pendingInherit.splice(0, cut);
      // (the parents this batch retires: their slots are free after it, not before, whatever waits)
      this.retiredNow = list.filter(e => e[0] === 0).map(e => e[1] & 0x3fffffff);
      const u = new Uint32Array(list.length * 12), f = new Float32Array(u.buffer);
      list.forEach((e, k) => { u[k * 12] = e[0]; u[k * 12 + 1] = e[1]; for (let q = 2; q < 12; q++) f[k * 12 + q] = e[q]; });
      d.queue.writeBuffer(this.intBuf, this.L.inherit * 4, u);
      this.inheritCount = list.length;
    }
    const pBusy = this.statsInputs !== this.inputs || this.wakeReq || this.pBlasts.length > 0;
    d.queue.writeBuffer(this.parBuf, 0, this.params(dt));
    const L = this.L, cur = this.frame & 1;
    const clr = (off, len) => enc.clearBuffer(this.intBuf, off * 4, len * 4);
    // counters; impact events keep accumulating until a step copies them out (a pending read-back skips the copy)
    if (this.statCopied === false) { clr(0, 7); clr(8, 7); } else { clr(0, 16); clr(L.snap, 1); }   // (events, crushes and breaks gather until copied)
    clr(L.evhist, 128); clr(L.pinch, 8 * this.n1); clr(L.level, this.n1); clr(L.level2, this.n1);
    clr(L.gcount, this.Geff); clr(L.gfill, this.Geff); clr(L.cawake, this.Geff); clr(L.cnear, CNEAR + 1); clr(L.ccount, 33); clr(L.cfill, 33); clr(cur ? L.t1 : L.t0, this.T);
    clr(L.hl, 1); clr(L.bcount, this.n1); clr(L.bfill, this.n1); clr(L.ocount, this.n1);
    if (this.reduce !== false) { clr(L.ptab, 3 * this.TPeff); clr(L.big, 64 * 4096); }
    if (this.capP && this.np) { enc.clearBuffer(this.pintBuf, 0, this.PGeff * 4); enc.clearBuffer(this.pintBuf, 2 * this.PGeff * 4, this.PGeff * 4); }
    if (!this.n) { this.frame++; return; }
    const nb = Math.ceil(this.n / 64) || 1, nsh = Math.ceil(this.ns / 64) || 1;
    const stamp = this.ts && !this.qPending;
    const pass = (stage, args = false) => {
      let desc = {};
      if (stamp && stage !== null) { const si = this.stages.indexOf(stage); desc = { timestampWrites: { querySet: this.qs, beginningOfPassWriteIndex: 2 * si, endOfPassWriteIndex: 2 * si + 1 } }; }
      const p = enc.beginComputePass(desc);
      p.setBindGroup(0, this.bg0); p.setBindGroup(1, this.bg1, [0]);
      if (args) p.setBindGroup(2, this.bg2);
      return p;
    };
    const run = (p, k, x) => { p.setPipeline(this.pl[k]); p.dispatchWorkgroups(x); };
    // Indirect dispatches are validated one by one by the browser (on its GPU process, which also draws the
    // page); at ~800 per step that stalled whole frames. By default every dispatch is direct: sized by capacity
    // for the once-per-step kernels, by the recent contact count (with margin) in the solver loops; the
    // kernels skip work past their real counts.
    const wgCap = (n) => Math.max(1, Math.ceil(n / 64));
    // Sizes that can never be too small (stats arrive late, so estimates from them could drop work): a body is
    // in at most one contact per colour, so a colour holds at most n contacts; everything else by capacity
    const perColour = wgCap(this.n + 1), perMan = wgCap(this.maxMan);
    // colour passes loop over their contacts (grid stride), so they can be sized by the recent count with margin:
    // a colour that grew is still solved in full, only by fewer threads
    const colWg = this.colCounts.map(c => Math.min(perColour, wgCap(c * 1.5 + 256)));
    // narrow loops over its pairs: by the recent count with margin (stats arrive a few steps late)
    const pairWg = Math.min(wgCap(this.maxPairs), wgCap(Math.max(8192, 2 * (this.stats.candidates || 0))));
    const dirWg = { 396: pairWg, 384: perMan };   // narrow (args 99), per-manifold (args 96)
    const ind = this.indirect
      ? (p, k, off) => { p.setPipeline(this.pl[k]); p.dispatchWorkgroupsIndirect(this.argBuf, off); }
      : (p, k, off) => { p.setPipeline(this.pl[k]); if (this.manIndirect && (off === 384 || off === 396)) p.dispatchWorkgroupsIndirect(this.argBuf, off); else p.dispatchWorkgroups(k === 'restitute_ovf' ? 1 : dirWg[off] ?? perMan); };
    let p;
    if (this.wakeReq) { this.wakeReq = false; p = pass(null); run(p, 'wake_all', nb); p.end(); }
    if (this.inheritCount) {
      p = pass(null); run(p, 'inherit', Math.ceil(this.inheritCount / 64));
      if (this.capP) { p.setBindGroup(2, this.bgP); run(p, 'p_place', Math.ceil(this.inheritCount / 64)); }
      run(p, 'retire', Math.ceil(this.inheritCount / 64)); p.end();
    }
    p = pass('broad'); run(p, 'prep_bodies', nb); run(p, 'prep_shapes', nsh); run(p, 'scan', 1); run(p, 'scatter', nsh); run(p, 'sort_cells', nsh); run(p, 'pairs', nsh); p.end();
    p = pass(null, true); run(p, 'pair_args', 1); p.end();
    p = pass('narrow'); ind(p, 'narrow', 99 * 4); p.end();
    // the chunk pairs listed by narrow, always sized by their real count (one indirect dispatch)
    p = pass(null, true); run(p, 'hull_args', 1); p.end();
    p = pass('chunks'); p.setPipeline(this.pl.narrow_hull); p.dispatchWorkgroupsIndirect(this.argBuf, 102 * 4); p.end();
    // (particles holding bodies up, from the last step's particle stage: contacts of their bodies now)
    if (this.capP && this.support !== false) { p = pass(null); p.setBindGroup(2, this.bgP); run(p, 'p_support', nsh); p.end(); }
    p = pass(null, true); run(p, 'man_args', 1); p.end();
    p = pass('colour');
    if (this.reduce !== false) { ind(p, 'red_count', 96 * 4); run(p, 'red_big', Math.ceil(this.TPeff / 64)); for (const k of ['red_score', 'red_pick', 'red_apply']) ind(p, k, 96 * 4); }
    ind(p, 'color_init', 96 * 4);
    // (rounds: 2 for supports, then 4 for all. Each is 4 dispatches, and on the software GPU a dispatch costs
    // milliseconds: 12 rounds made the colour stage of the lab's collapse 38 -> 880 ms at its slowest step. With
    // 2 + 4 the 2,000-box pile left 10 contacts to the overflow pass against 1; with 1 + 3, 67)
    const RS = this.colRoundsS ?? 2, RA = this.colRounds ?? 4;
    for (let r = 0; r < RS + RA; r++) { ind(p, r < RS ? 'color_claim_s' : 'color_claim', 96 * 4); ind(p, 'color_tie', 96 * 4); ind(p, 'color_resolve', 96 * 4); run(p, 'color_round', 1); }
    ind(p, 'color_count', 96 * 4); run(p, 'adj_scan', 1);
    if (this.shock !== false) { for (let k = 0; k < 6; k++) { p.setBindGroup(1, this.bg1, [k * 256]); ind(p, 'level_pass', 96 * 4); } p.setBindGroup(1, this.bg1, [0]); }
    if (this.colorCheck) { p.end(); enc.clearBuffer(this.intBuf, (L.cchk + 1) * 4, 32 * this.n1 * 4); p = pass(null); ind(p, 'color_check', 96 * 4); }
    p.end();
    p = pass(null, true); run(p, 'color_scan', 1); p.end();
    p = pass('sub-steps');
    ind(p, 'color_scatter', 96 * 4); run(p, 'blist_sort', nb); if (!globalThis.NO_OVF_SORT) run(p, 'ovf_sort', 1);
    const C = this.colorLimit;
    const colors = (k) => {
      p.setPipeline(this.pl[k]);
      const iters = k === 'relax' ? (this.relaxIters ?? 1) : 1;
      for (let it = 0; it < iters; it++) for (let cc = 0; cc < C; cc++) { const c = it & 1 ? C - 1 - cc : cc; p.setBindGroup(1, this.bg1, [c * 256]); if (this.indirect) p.dispatchWorkgroupsIndirect(this.argBuf, c * 12); else p.dispatchWorkgroups(colWg[c]); }
      p.setBindGroup(1, this.bg1, [0]);
      if (k === 'restitute') { ind(p, 'restitute_ovf', C * 12); return; }
      // overflow contacts: two Jacobi rounds with mass splitting
      const ovfWg = Math.min(perMan, wgCap(this.colCounts[C] * 1.5 + 256));
      for (let it = 0; it < (k === 'relax' ? (this.relaxRounds ?? 2) : 2); it++) {
        if (this.indirect) ind(p, k + '_ovf', C * 12); else run(p, k + '_ovf', ovfWg);
        run(p, 'ovf_apply', nb);
      }
    };
    for (let s = 0; s < this.substeps; s++) {
      run(p, 'integrate_warm', nb);
      colors('solve');
      run(p, 'integrate_pos', nb);
      colors('relax');
    }
    colors('restitute');
    run(p, 'finish', nb);
    if (this.evEnergy > 0) { ind(p, 'ev_count', 96 * 4); run(p, 'ev_cut', 1); ind(p, 'events', 96 * 4); }
    if (this.crushing !== false) { ind(p, 'crush', 96 * 4); if (this.pinching === true) { ind(p, 'pinch_acc', 96 * 4); run(p, 'pinch', nb); } ind(p, 'crush_release', 96 * 4); }   // (pinching off by default: docs/GPU.md, Crushing on contact)
    if (this.capP && this.support !== false) ind(p, 'sup_acc', 96 * 4);
    if (this.plans) { ind(p, 'snap_acc', 96 * 4); run(p, 'snap_split', nb); run(p, 'snap_shapes', nsh); }
    p.end();
    // particles, after the bodies moved (they see this step's shapes and body velocities)
    p = pass('particles');
    if (this.capP && this.np) {
      p.setBindGroup(2, this.bgP);
      const np = Math.ceil(this.np / 64);
      if (this.support !== false) run(p, 'p_sup_load', nsh);
      run(p, 'p_prep', np); run(p, 'p_scan', 1); run(p, 'p_scatter', np); run(p, 'p_sort', Math.ceil(this.PGeff / 64));
      // with no particle moving at the last stats and no new input, only the wake check runs (a particle woken now
      // starts moving a few steps later, when the stats show it)
      if (this.stats.pActive > 0 || pBusy) {
        run(p, 'p_gather', np);
        for (let k = 0; k < this.psub; k++) { run(p, 'p_integrate', np); run(p, 'p_contact', np); run(p, 'p_apply', np); }
        run(p, 'p_finish', np);
      }
    }
    p.end();
    if (this.sleep) { p = pass('sleep'); ind(p, 'uf_union', 96 * 4); run(p, 'uf_min', nb); run(p, 'uf_sleep', nb); p.end(); }
    if (stamp) {
      const k = this.stages.length - (this.sleep ? 0 : 1);
      enc.resolveQuerySet(this.qs, 0, 2 * k, this.qBuf, 0);
      enc.copyBufferToBuffer(this.qBuf, 0, this.qRead, 0, 16 * k);
      this.qCopied = k;
    } else this.qCopied = 0;
    const sb = this.statBufs.find(b => !b.busy);
    if (sb) {
      sb.busy = true; this.statNext = sb; sb.inputs = this.inputs;
      enc.copyBufferToBuffer(this.intBuf, 0, sb.buf, 0, 64);
      if (this.evEnergy > 0) enc.copyBufferToBuffer(this.intBuf, L.events * 4, sb.buf, 64, 48 * this.maxEvents);
      enc.copyBufferToBuffer(this.intBuf, L.ccount * 4, sb.buf, 64 + 48 * this.maxEvents, 33 * 4);
      enc.copyBufferToBuffer(this.intBuf, L.crush * 4, sb.buf, 64 + 48 * this.maxEvents + 33 * 4, 48 * 256);
      enc.copyBufferToBuffer(this.intBuf, L.snap * 4, sb.buf, 64 + 48 * this.maxEvents + 33 * 4 + 48 * 256, 4 + 16 * 256);
      this.statCopied = true;
    } else this.statCopied = false;
    if (this.inheritCount) {
      this.inheritCount = 0;
      // the retired parents' slots are free from now on (their pieces have been placed from them)
      for (const b of this.retiredNow || []) this.freeBody(b);
      this.retiredNow = [];
    }
    this.frame++; this.realFrame = this.frame;   // (the last step that moved anything: read-backs since show the same)
  }

  // ---- incremental edits (debris, a changing static structure)
  // Append bodies (same format as create); returns the index of the first new body (0-based).
  addBodies(list) {
    this.inputs++;
    // never let a non-finite number in: one NaN body spreads NaN to everything it touches
    const fin = (a) => a && a.every(Number.isFinite);
    for (const b of list) {
      b.shapes = b.shapes.filter(s => fin(s.p) && fin(s.q) && fin(s.h));
      if (!fin(b.v || [0, 0, 0]) || !fin(b.w || [0, 0, 0])) { b.v = [0, 0, 0]; b.w = [0, 0, 0]; this.rejected = (this.rejected || 0) + 1; }
    }
    const keepIdx = []; list.forEach((b, k) => { if (b.shapes.length) keepIdx.push(k); });
    const all = list.length;
    list = keepIdx.map(k => list[k]);
    if (!list.length) return new Array(all).fill(0);
    const { bodies, shapes } = buildBodies(list);
    this.freeBodies = this.freeBodies || []; this.freeShapes = this.freeShapes || [];
    const nb = Math.max(0, bodies.length - this.freeBodies.length), nsh = Math.max(0, shapes.length - this.freeShapes.length);
    if (this.n + nb > this.capB || this.ns + nsh > this.capS) throw new Error('GpuWorld capacity');
    const q = this.device.queue;
    // slots of retired bodies and shapes are reused first, then the arrays grow
    const bslot = bodies.map(() => this.takeBodySlot());
    this.bodyShapes = this.bodyShapes || new Map();
    for (let k = 0; k < bodies.length; k++) {
      const b = bodies[k], i = bslot[k];
      this.bodyShapes.set(i, []);
      if (globalThis.__parkDbg) (globalThis.__slotLog = globalThis.__slotLog || []).push(['add', this.frame, i]);
      (this.bodyBorn = this.bodyBorn || [])[i] = this.frame;   // (records of steps before this are about the slot's old body)
      q.writeBuffer(this.intBuf, (this.L.crushed + i) * 4, new Uint32Array([0]));
      q.writeBuffer(this.bodBuf, i * 80, new Float32Array([...b.p, b.im, ...b.q, ...b.il, 0, 0, ...b.p, ...b.q]));
      q.writeBuffer(this.solBuf, i * 96, new Float32Array([...b.v, 0, ...b.w, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]));
      this.bodies[i - 1] = b;
    }
    for (const s of shapes) {
      const slot = this.freeShapes.length ? this.freeShapes.pop() : (this.ns++, this.ns1++, this.ns1 - 1);
      this.shapeGen = this.shapeGen || []; this.shapeGen[slot - 1] = (this.shapeGen[slot - 1] || 0) + 1;
      const body = bslot[s.body - 1];
      q.writeBuffer(this.shpBuf, slot * 128, new Float32Array([body, s.mat, 0, s.hull || 0, ...s.lp, 0, ...s.lq, ...s.h, 0]));
      this.shapes[slot - 1] = { ...s, body };
      (this.bodyShapes = this.bodyShapes || new Map());
      if (!this.bodyShapes.has(body)) this.bodyShapes.set(body, []);
      this.bodyShapes.get(body).push(slot);
    }
    // slots in the order given (0 for a body that had nothing valid)
    const out = new Array(all).fill(0); keepIdx.forEach((k, q) => { out[k] = bslot[q]; });
    return out;
  }
  // Static shapes of fixed body `body` (1-based body index): set (create when slot is null) or kill.
  // centre and half extents in metres, axis-aligned; returns the slot (1-based shape index).
  setStaticShape(slot, body, centre, half, mat, hull = 0) {
    this.inputs++;
    const b = this.bodies[body - 1];
    if (slot == null) {
      if (this.freeShapes && this.freeShapes.length) slot = this.freeShapes.pop();
      else {
        if (this.ns + 1 > this.capS) throw new Error('GpuWorld capacity');
        slot = this.ns1; this.ns++; this.ns1++;
        this.shapes.push(null);
      }
    }
    // a new shape in this slot gets a new generation, so read-backs never take it for the slot's old shape
    if (!this.shapes[slot - 1] || this.shapes[slot - 1].dead) { this.shapeGen = this.shapeGen || []; this.shapeGen[slot - 1] = (this.shapeGen[slot - 1] || 0) + 1; }
    const lp = [centre[0] - b.p[0], centre[1] - b.p[1], centre[2] - b.p[2]];
    const rec = { body, mat, lp, lq: [0, 0, 0, 1], h: half, hull };
    this.shapes[slot - 1] = rec;
    this.device.queue.writeBuffer(this.shpBuf, slot * 128, new Float32Array([body, mat, 0, hull, ...lp, 0, 0, 0, 0, 1, ...half, 0]));
    return slot;
  }
  // Retire a body: it stops moving and colliding (its slot is not reused).
  killBody(body) {
    if (globalThis.__parkDbg) (globalThis.__slotLog = globalThis.__slotLog || []).push(['kill', this.frame, body]);
    this.inputs++;
    const b = this.bodies[body - 1]; if (!b || b.dead) return;
    this.clearSnap(body);
    b.dead = true;
    const own = this.bodyShapes && this.bodyShapes.get(body);
    if (own) { for (const k of own) if (this.shapes[k - 1] && this.shapes[k - 1].body === body) this.killShape(k); this.bodyShapes.delete(body); }
    else this.shapes.forEach((s, k) => { if (s && s.body === body && !s.dead) this.killShape(k + 1); });
    this.freeBody(body);
    this.device.queue.writeBuffer(this.bodBuf, body * 80, new Float32Array([0, -1000, 0, 0]));
    this.device.queue.writeBuffer(this.bodBuf, body * 80 + 48, new Float32Array([1, 0, -1000, 0]));
  }
  // ---- breaking on the GPU (see snap_split)
  // Arm body `body` to break in the step its contacts' normal impulse passes `impulse` (N s): the shapes in
  // `other` (slots) go to a spare body, made now with their mass. Returns the spare (0 if there is no room).
  setSnap(body, impulse, other) {
    this.bodyShapes = this.bodyShapes || new Map();
    const b = this.bodies[body - 1];
    if (!this.bodyShapes.has(body)) this.bodyShapes.set(body, this.shapes.map((x, k) => x && !x.dead && x.body === body ? k + 1 : 0).filter(Boolean));
    const own = this.bodyShapes.get(body);
    if (!b || b.dead || !other.length || other.length >= own.length) return 0;
    const set = new Set(other);
    // the two parts' mass, centre and principal frame, in the body's frame (shapes at their local poses)
    const part = (want) => own.filter(k => set.has(k) === want).map(k => { const s = this.shapes[k - 1]; return { p: s.lp, q: s.lq, h: s.h, mat: s.mat, hull: s.hull }; });
    const { bodies: pb } = buildBodies([{ shapes: part(false) }, { shapes: part(true) }]);
    const [A, Bp] = pb;
    if (!(A.im > 0) || !(Bp.im > 0)) return 0;
    this.freeBodies = this.freeBodies || [];
    if (!this.freeBodies.length && this.n + 1 > this.capB) return 0;
    const j = this.takeBodySlot();
    this.inputs++;
    const q = this.device.queue;
    // (parked asleep, out of the way, until it is needed)
    q.writeBuffer(this.bodBuf, j * 80, new Float32Array([0, -1000, 0, Bp.im, ...Bp.q, ...Bp.il, 0, 1, 0, -1000, 0, ...Bp.q]));
    q.writeBuffer(this.solBuf, j * 96, new Float32Array(24));
    q.writeBuffer(this.intBuf, (this.L.crushed + j) * 4, new Uint32Array([0]));
    this.bodies[j - 1] = { p: [0, -1000, 0], q: Bp.q, im: Bp.im, il: Bp.il, v: [0, 0, 0], w: [0, 0, 0], mass: Bp.mass, spare: body };
    if (globalThis.__parkDbg) (globalThis.__slotLog = globalThis.__slotLog || []).push(['spare', this.frame, j, body]);
    this.bodyShapes.set(j, []);
    const w = new Uint32Array(24), wf = new Float32Array(w.buffer);
    wf[0] = impulse; w[1] = j; w[2] = 1; w[3] = 0;
    wf.set([...A.p, ...A.q, A.im, ...A.il, ...Bp.p, ...Bp.q], 4);
    q.writeBuffer(this.intBuf, (this.L.plan + body * 24) * 4, w);
    for (const k of own) q.writeBuffer(this.intBuf, (this.L.spart + k) * 4, new Uint32Array([set.has(k) ? 2 : 1]));
    this.planOf = this.planOf || new Map();
    this.planOf.set(body, { spare: j, other: set, A, B: Bp, impulse });
    this.plans = (this.plans || 0) + 1;
    return j;
  }
  // a plan no longer wanted (the body is broken or retired another way): its spare is freed
  clearSnap(body) {
    const pl = this.planOf && this.planOf.get(body); if (!pl) return;
    this.planOf.delete(body); this.plans--;
    this.device.queue.writeBuffer(this.intBuf, (this.L.plan + body * 24 + 2) * 4, new Uint32Array([0]));
    if (!pl.done) { const sp = this.bodies[pl.spare - 1]; if (globalThis.__parkDbg) (globalThis.__slotLog = globalThis.__slotLog || []).push(['giveup', this.frame, pl.spare, body]); if (sp && !sp.dead && sp.spare === body) { sp.dead = true; this.bodyShapes.delete(pl.spare); this.freeBody(pl.spare); } }
  }
  // a read-back says body i broke on the GPU into itself and spare j: the JS records follow
  snapped(i, j, impulse) {
    const pl = this.planOf && this.planOf.get(i); if (!pl || pl.done) { if (globalThis.__parkDbg) (globalThis.__slotLog = globalThis.__slotLog || []).push(['snap-ignored', this.frame, j, i]); return; }
    pl.done = true; this.planOf.delete(i); this.plans--;
    if (globalThis.__parkDbg) (globalThis.__slotLog = globalThis.__slotLog || []).push(['snapped', this.frame, j, i]);
    const own = this.bodyShapes.get(i) || [], keep = [], moved = [];
    for (const k of own) {
      const s = this.shapes[k - 1]; if (!s || s.dead) continue;
      const other = pl.other.has(k), c = other ? pl.B.p : pl.A.p, r = other ? pl.B.q : pl.A.q, rc = [-r[0], -r[1], -r[2], r[3]];
      s.lp = mv(q2m(rc), [s.lp[0] - c[0], s.lp[1] - c[1], s.lp[2] - c[2]]); s.lq = qmulJ(rc, s.lq);
      if (other) { s.body = j; moved.push(k); } else keep.push(k);
    }
    this.bodyShapes.set(i, keep); this.bodyShapes.set(j, moved);
    const b = this.bodies[i - 1]; if (b) { b.im = pl.A.im; b.il = pl.A.il; b.mass = pl.A.mass; }
    this.snaps.push({ body: i, spare: j, A: pl.A, B: pl.B, impulse });
  }
  // Wait until every step's stats submitted so far are in (events, crushes, breaks, counts). The page reads them
  // as they come; the lab waits for them each frame, so what a frame sees does not depend on how fast the
  // browser maps buffers (runs differed by it).
  async flushStats() { this.pollStats(); while (this.statPending && this.statPending.size) await Promise.all([...this.statPending]); }
  // the pieces broken on the GPU since the last call: [{ body, spare, A, B (the parts' centre p and rotation q in
  // the body's frame), impulse }]
  takeSnaps() { const s = this.snaps; this.snaps = []; return s; }
  // Take the impact events collected since the last call.
  // pieces made by breakBody that the next step still has to place (a read-back now would show them wrong)
  placing() { return !!(this.pendingInherit && this.pendingInherit.length); }
  // Break a moving body into new ones (see the `inherit` kernel): `kids` are slots from addBodies, made from
  // the parent's pose (pr, qr) of an older read-back; dv[k] is extra velocity for kid k (world frame then).
  // The parent stops colliding now and is retired at the next step, after its pieces have been placed.
  breakBody(parent, kids, pr, qr, dvs, parts = [], pdvs = []) {
    this.inputs++;
    const b = this.bodies[parent - 1]; if (!b || b.dead) return;
    this.clearSnap(parent);
    if (globalThis.__parkDbg) (globalThis.__slotLog = globalThis.__slotLog || []).push(['break', this.frame, parent]);
    b.dead = true;
    const own = this.bodyShapes && this.bodyShapes.get(parent);
    if (own) { for (const k of own) if (this.shapes[k - 1] && this.shapes[k - 1].body === parent) this.killShape(k); this.bodyShapes.delete(parent); }
    this.pendingInherit = this.pendingInherit || [];
    kids.forEach((kid, k) => {
      if (!kid) return;
      // dvs[k]: [dx, dy, dz] beyond the parent's motion, { v } for a velocity of its own, { noRebound: n } to
      // move on with the parent but without a rebound along the unit direction n
      const d = dvs[k] || [0, 0, 0];
      if (!Array.isArray(d) && d.v) this.pendingInherit.push([kid, (parent | 0x80000000) >>> 0, ...pr, ...qr, ...d.v]);
      else if (!Array.isArray(d) && d.noRebound) this.pendingInherit.push([kid, (parent | 0x40000000) >>> 0, ...pr, ...qr, ...d.noRebound]);
      else this.pendingInherit.push([kid, parent, ...pr, ...qr, ...d]);
    });
    // particles made from it (before the retire entry: they are placed from the parent's pose)
    parts.forEach((slot, k) => {
      if (slot < 0) return;
      const d = pdvs[k] || [0, 0, 0];
      if (!Array.isArray(d) && d.v) this.pendingInherit.push([(slot + 1) | 0x80000000, (parent | 0x80000000) >>> 0, ...pr, ...qr, ...d.v]);
      else this.pendingInherit.push([((slot + 1) | 0x80000000) >>> 0, parent, ...pr, ...qr, ...d]);
    });
    this.pendingInherit.push([0, parent, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]);
  }
  // Body slots. Each assignment of a slot (a new body, a spare) gets a new generation: the page keeps the one it
  // knew, so an answer about the slot's old body is not applied to the new one. A slot freed twice would be given
  // out twice (lab collapse: a spare given up after the slot had been freed as a broken body; a new body and
  // another spare then shared slot 98, and the new body's pieces were drawn 1,000 m below the ground).
  takeBodySlot() {
    this.freeBodies = this.freeBodies || []; this.freeSet = this.freeSet || new Set(); this.bodyGen = this.bodyGen || [];
    const i = this.freeBodies.length ? this.freeBodies.pop() : (this.n++, this.n1++, this.n1 - 1);
    this.freeSet.delete(i); this.bodyGen[i] = (this.bodyGen[i] || 0) + 1;
    return i;
  }
  freeBody(i) {
    this.freeBodies = this.freeBodies || []; this.freeSet = this.freeSet || new Set();
    if (this.freeSet.has(i)) { this.doubleFree = (this.doubleFree || 0) + 1; return; }
    this.freeSet.add(i); this.freeBodies.push(i);
  }
  genOf(i) { return this.bodyGen ? this.bodyGen[i] || 0 : 0; }
  // free capacity: bodies and shapes (retired slots are reused)
  roomBodies() { return this.capB - this.n + (this.freeBodies ? this.freeBodies.length : 0); }
  roomShapes() { return this.capS - this.ns + (this.freeShapes ? this.freeShapes.length : 0); }
  // Set a body's velocity and wake it (p, q: its current pose, which a wake records as its rest pose)
  setVelocity(body, v, w, p, q) {
    this.inputs++;
    if (![...v, ...w, ...p, ...q].every(Number.isFinite)) { this.rejected = (this.rejected || 0) + 1; return; }
    const qd = this.device.queue;
    qd.writeBuffer(this.solBuf, body * 96, new Float32Array([...v, 1, ...w, 0]));
    qd.writeBuffer(this.bodBuf, body * 80 + 48, new Float32Array([0, ...p, ...q]));
  }
  // Wake all sleeping bodies at the next step (call when static shapes under them were removed).
  wakeAll() { this.wakeReq = true; this.inputs++; }
  // ---- particles: crumbs as granular debris. list: { p, v, q, r (collision radius), mat, wait }; returns slots (-1
  // when full). wait: made from a moving body's older pose, placed by breakBody (pass the slots there). They are
  // drawn from readParticles().
  addParticles(list) {
    this.inputs++;
    const out = [];
    for (const e of list) {
      if (![...e.p, ...e.v].every(Number.isFinite)) { this.rejected = (this.rejected || 0) + 1; out.push(-1); continue; }
      let i = this.freeParticles.length ? this.freeParticles.pop() : (this.np < this.capP ? this.np++ : -1);
      if (i < 0) { out.push(-1); continue; }
      const q = e.q || [0, 0, 0, 1], w = e.w || [0, 0, 0], r = Math.min(e.r, this.pcell / 2 - 0.05);
      this.device.queue.writeBuffer(this.prtBuf, i * 112, new Float32Array([...e.p, r, ...e.v, e.wait ? 4 : 1, ...q, ...e.p, 0, ...w, e.mat || 0, ...e.p, 0, ...e.p, 0]));
      out.push(i);
    }
    return out;
  }
  killParticle(i) {
    this.inputs++;
    this.device.queue.writeBuffer(this.prtBuf, i * 112 + 28, new Float32Array([0]));
    this.freeParticles.push(i);
  }
  roomParticles() { return this.capP - this.np + this.freeParticles.length; }
  // push particles within `reach` of c outward (m/s at the centre, falling off with distance)
  blastParticles(c, reach, speed) { if (this.capP) this.pBlasts.push({ c, reach, speed }); }
  // [x, y, z, vx][vy, vz, qx, qy][qz, qw, state, r] per particle slot (state 0 = free, 3 = crushed to dust: the
  // caller frees it with killParticle)
  async readParticles() {
    const n = this.np, frame = this.frame;
    if (!n) return { data: new Float32Array(0), n: 0, frame };
    return { data: await this.read(this.prtBuf, n * 48, Math.max(1, this.capP) * 112), n, frame };
  }
  takeEvents() { const e = this.events; this.events = []; return e; }
  // bodies crushed by much heavier ones (see the crush kernel): [{ body, by, p, q, v }] at the step it happened;
  // they collide only with the static world until the caller breaks them (into particles, say)
  takeCrushed() { const c = this.crushes; this.crushes = []; return c; }
  killShape(slot) {
    this.inputs++;
    const s = this.shapes[slot - 1]; if (!s || s.dead) return;
    s.dead = true;
    (this.freeShapes = this.freeShapes || []).push(slot);
    this.device.queue.writeBuffer(this.shpBuf, slot * 128, new Float32Array([s.body, s.mat, 1, 0]));
  }

  step(dt = 1 / 60) {
    const enc = this.device.createCommandEncoder();
    this.encode(enc, dt);
    this.device.queue.submit([enc.finish()]);
    this.pollStats();
  }

  pollTimes() {
    if (!this.ts || !this.qCopied || this.qPending) return;
    const k = this.qCopied;
    this.qPending = true;
    this.qRead.mapAsync(GPUMapMode.READ).then(() => {
      const t = new BigUint64Array(this.qRead.getMappedRange().slice(0, 16 * k));
      this.qRead.unmap();
      const ms = {};
      this.stages.slice(0, k).forEach((s, i) => { ms[s] = Number(t[2 * i + 1] - t[2 * i]) / 1e6; });
      if (!this.gpuMs) this.gpuMs = ms; else for (const s in ms) this.gpuMs[s] = 0.8 * (this.gpuMs[s] ?? ms[s]) + 0.2 * ms[s];
      this.qPending = false;
    }, () => { this.qPending = false; });
  }

  pollStats() {
    this.pollTimes();
    if (!this.statCopied || !this.statNext) return;
    const sb = this.statNext, seq = ++this.statSeq; this.statNext = null;
    const pr = sb.buf.mapAsync(GPUMapMode.READ).then(() => {
      const raw = sb.buf.getMappedRange().slice(0);
      sb.buf.unmap(); sb.busy = false;
      const u = new Uint32Array(raw), f = new Float32Array(raw);
      // events are never lost: every copied step's events count, even when an older copy arrives late
      if (this.evEnergy > 0) {
        const n = Math.min(u[7], this.maxEvents), batch = [];
        for (let k = 0; k < n; k++) {
          const o = 16 + k * 12;
          batch.push({ a: u[o], b: u[o + 1], sa: u[o + 2], sb: u[o + 3], p: [f[o + 4], f[o + 5], f[o + 6]], n: [f[o + 7], f[o + 8], f[o + 9]], impulse: f[o + 10], approach: f[o + 11] });
        }
        // (in the order of their shapes: the GPU lists them in whatever order its threads got there)
        batch.sort((x, y) => x.sa - y.sa || x.sb - y.sb || x.a - y.a || x.b - y.b);
        for (const e of batch) this.events.push(e);
      }
      // crushed bodies, never lost either
      { const n = Math.min(u[15], 256), o0 = (64 + 48 * this.maxEvents + 33 * 4) / 4;
        const batch = [];
        for (let k = 0; k < n; k++) { const o = o0 + k * 12; if (this.bodyBorn && (this.bodyBorn[u[o]] ?? -1) > u[o + 1]) continue;   // (its slot was reused since: the new body is not crushed)
          batch.push({ body: u[o], step: u[o + 1], p: [f[o + 2], f[o + 3], f[o + 4]], q: [f[o + 5], f[o + 6], f[o + 7], f[o + 8]], v: [f[o + 9], f[o + 10], f[o + 11]] }); }
        batch.sort((x, y) => x.body - y.body); for (const c of batch) this.crushes.push(c); }
      // pieces broken on the GPU, never lost: their shapes' records here follow (as the kernels did)
      { const o0 = (64 + 48 * this.maxEvents + 33 * 4 + 48 * 256) / 4, n = Math.min(u[o0], 256);
        const batch = [];
        for (let k = 0; k < n; k++) { const o = o0 + 1 + 4 * k; batch.push([u[o], u[o + 1], f[o + 3]]); }
        batch.sort((x, y) => x[0] - y[0]); for (const b of batch) this.snapped(...b); }
      if (seq < this.statLatest) return;
      this.statLatest = seq; this.statsInputs = sb.inputs;
      this.stats = { manifolds: u[0], overflow: u[1], colors: u[2], pairs: u[3], dropped: u[4], candidates: u[5], awake: u[6], events: u[7], pActive: u[12], pLive: u[13], pairsLost: Math.max(0, u[5] - this.maxPairs) };
      if (u[4] > 0 || u[5] > this.maxPairs) this.lost = (this.lost || 0) + 1;   // steps that lost contacts or pairs
      this.colorLimit = Math.min(this.maxColors, 31, Math.max(4, u[2] + (u[1] > 0 ? 3 : 1)));
      // per-colour contact counts (they size the direct dispatches): keep a slowly decaying maximum
      const cc = new Uint32Array(raw, 64 + 48 * this.maxEvents, 33);
      for (let c = 0; c < 33; c++) this.colCounts[c] = Math.max(cc[c], Math.floor(this.colCounts[c] * 0.95));
    }, () => { sb.busy = false; });
    // (the lab waits for them each frame, so a frame sees the same stats every run: see flushStats)
    this.statPending = this.statPending || new Set(); this.statPending.add(pr); pr.finally(() => this.statPending.delete(pr));
  }

  // Copy `size` bytes of `buffer` back. Staging buffers are kept per source buffer and reused (grown when
  // needed); a read while that one is still mapping gets a temporary buffer.
  async read(buffer, size, offset = 0) {
    this.staging = this.staging || new Map();
    let st = this.staging.get(buffer), temp = false;
    if (!st || st.busy || st.size < size) {
      if (st && st.busy) temp = true;
      else { if (st) st.buf.destroy(); st = { buf: this.device.createBuffer({ size: Math.max(size, 4096) * 2, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }), size: Math.max(size, 4096) * 2, busy: false }; this.staging.set(buffer, st); }
    }
    const rb = temp ? this.device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }) : st.buf;
    if (!temp) st.busy = true;
    const enc = this.device.createCommandEncoder(); enc.copyBufferToBuffer(buffer, offset, rb, 0, size);
    this.device.queue.submit([enc.finish()]);
    try {
      await rb.mapAsync(GPUMapMode.READ, 0, size);
      const f = new Float32Array(rb.getMappedRange(0, size).slice(0));
      rb.unmap();
      return f;
    } finally { if (temp) rb.destroy(); else st.busy = false; }
  }

  // Shapes as 10 floats each (p, q, h) for rendering, plus per-body state.
  // minBody > 0 returns only the shapes of bodies from that (1-based) index on, e.g. debris, with their
  // materials in `mats`; dead shapes are skipped.
  async readPose(minBody = 0) {
    const frame = this.frame;   // the step this read-back shows
    // the world as of now: which shapes exist, of which bodies (a body broken or made while the read is in
    // flight must not vanish or appear early: the list and the poses are of the same moment)
    const n1 = this.n1, nb = this.n;
    const list = [], slots = [], gens = [];
    // Which shapes: those the JS records list now. Each one's body and local pose: the GPU's own, copied in the
    // same moment as the bodies. (A break on the GPU moves shapes to another body in the step it happens; the JS
    // records follow when that step's stats come back. Poses from the live records, at whatever time the read
    // resolved, made lab runs differ; from records copied at the read, pieces drawn from a body that had just
    // broken jumped by its new centre, 8.7 m: lab collapse, 6,709 jumps.)
    this.shapes.forEach((s, k) => { if (s && s.body < n1 && (!minBody || (!s.dead && s.body >= minBody))) { list.push({ mat: s.mat, col: s.col }); slots.push(k + 1); gens.push(this.shapeGen ? this.shapeGen[k] || 0 : 0); } });
    const [bod, sol, shp] = await Promise.all([this.read(this.bodBuf, n1 * 80), this.read(this.solBuf, n1 * 96), this.read(this.shpBuf, this.ns1 * 128)]);
    const t0 = performance.now();
    const pose = new Float32Array(list.length * 10), mats = new Uint8Array(list.length), cols = new Uint16Array(list.length);
    list.forEach((s, k) => {
      const o = slots[k] * 32, bi = Math.round(shp[o]);
      const b = bi * 20, p = [bod[b], bod[b + 1], bod[b + 2]], q = [bod[b + 4], bod[b + 5], bod[b + 6], bod[b + 7]];
      const lp = [shp[o + 4], shp[o + 5], shp[o + 6]], lq = [shp[o + 8], shp[o + 9], shp[o + 10], shp[o + 11]], h = [shp[o + 12], shp[o + 13], shp[o + 14]];
      const wp = mv(q2m(q), lp), wq = qmulJ(q, lq);
      pose.set([p[0] + wp[0], p[1] + wp[1], p[2] + wp[2], ...wq, ...h], k * 10);
      if (globalThis.__parkDbg && p[1] < -500 && frame > 105 && (globalThis.__parked = globalThis.__parked || []).length < 60) globalThis.__parked.push({ frame, issuedAt: this.frame, slot: slots[k], gpuBody: bi, jsBody: this.shapes[slots[k] - 1] && this.shapes[slots[k] - 1].body, dead: !!(this.shapes[slots[k] - 1] || {}).dead, born: this.bodyBorn && this.bodyBorn[bi], gen: gens[k], bodyRec: Array.from(bod.subarray(b, b + 4)).map(x => +x.toFixed(2)) });
      mats[k] = s.mat; cols[k] = s.col ?? s.mat;
    });
    const body = new Float32Array(nb * 7), vel = new Float32Array(nb * 3), ang = new Float32Array(nb * 3), sleeping = new Uint8Array(nb), timer = new Float32Array(nb);
    for (let i = 0; i < nb; i++) {
      const b = (i + 1) * 20, s = (i + 1) * 24;
      body.set(bod.subarray(b, b + 3), i * 7); body.set(bod.subarray(b + 4, b + 8), i * 7 + 3);
      vel.set(sol.subarray(s, s + 3), i * 3); ang.set(sol.subarray(s + 4, s + 7), i * 3);
      sleeping[i] = bod[b + 12] > 0.5 ? 1 : 0; timer[i] = bod[b + 11];
    }
    return { pose, mats, cols, slots: Uint32Array.from(slots), gens: Uint32Array.from(gens), body, vel, ang, sleeping, timer, frame, n1, jsMs: performance.now() - t0 };
  }

  // This step's contacts: bodies, restitution, normal impulse total (N s) and approach speed (m/s) per contact,
  // for the damage model (impact energy ~ 1/2 J v).
  async readContacts() {
    const count = Math.min(this.stats.manifolds, this.maxMan);
    const cur = (this.frame - 1) & 1;
    const f = await this.read(this.manBuf, 2 * this.maxMan * 17 * 16);
    const out = [];
    for (let k = 0; k < count; k++) {
      const b = (cur * this.maxMan + k) * 68;
      out.push({ a: f[b], b: f[b + 1], impulse: f[b + 16 * 4 + 1], approach: f[b + 16 * 4 + 2], restitution: f[b + 16 * 4] });
    }
    return out;
  }

  destroy() {
    for (const b of [this.bodBuf, this.solBuf, this.shpBuf, this.intBuf, this.manBuf, this.coldBuf, this.parBuf, this.colBuf, this.argBuf, this.prtBuf, this.pintBuf, ...this.statBufs.map(b => b.buf), this.qBuf, this.qRead]) if (b) b.destroy();
    if (this.qs) this.qs.destroy();
  }
}
if (typeof window !== 'undefined') { window.GpuWorld = GpuWorld; window.MATERIALS = MATERIALS; window.MAT = MAT; }
