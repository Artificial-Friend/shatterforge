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
  ev_speed: f32, ev_energy: f32, o_cawake: u32, pad5: u32,
  n_inherit: u32, o_inherit: u32, wake_speed: f32, pad7: u32,
  mat: array<vec4f, 8>,   // friction, restitution, -, -
  np1: u32, pcell: f32, pg_mask: u32, p_wake: u32,
  p_h: f32, n_blast: u32, p_cap: u32, pad9: u32,
  blast: array<vec4f, 6>,   // particle blasts: (centre, reach), (speed, -, -, -)
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
  st(P.o_par + i, i);
  st(P.o_mint + i, 0x7f7fffffu);
}
fn dead(s: u32) -> bool { return Hs(s, H_ID).z > 0.5; }
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
  if (dyn_awake(b)) { atomicOr(&ints[P.o_cawake + hc], 1u); }
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
  st(P.o_sorted + ld(P.o_gstart + hc) + atomicAdd(&ints[P.o_gfill + hc], 1u), s);
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

fn emit(sa: u32, sb: u32, a: u32, b: u32, c0: Contact) {
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
  var mb = P.mat[P.ground_mat]; if (sb != 0u) { mb = P.mat[u32(Hs(sb, H_ID).y)]; }
  let mu = sqrt(ma.x * mb.x); let rest = max(ma.y, mb.y);
  var cen = vec3f(0.0);
  for (var k = 0u; k < c.cnt; k++) { cen += c.pts[k].xyz; }
  cen /= f32(c.cnt);
  var rad = 0.0; for (var k = 0u; k < c.cnt; k++) { rad += length(c.pts[k].xyz - cen); }
  rad /= f32(c.cnt);
  let old = find_old(sa, sb);
  var jt1 = 0.0; var jt2 = 0.0; var jtw = 0.0;
  var fa = cen - pa; var fb = cen - pb; var e0 = vec3f(0.0);
  var laf = transpose(Ra) * fa; var lbf = select(transpose(Rb) * fb, cen, b == 0u);
  var ojn = vec4f(0.0); var ocnt = 0u;
  if (old != 0xffffffffu) {
    let ob = old * MS; let oc = old * CS;
    let ohd = man[ob]; ocnt = u32(ohd.z); ojn = man[ob + M_JN];
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
  man[base + M_RV] = rv;
  man[base + M_OUT] = vec4f(rest, 0.0, approach, 0.0);
  cold[cb + 4u] = vec4f(laf, f32(sa)); cold[cb + 5u] = vec4f(lbf, f32(sb));
  insert_new(sa, sb, mi);
}

fn wake(i: u32) {
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
}

// wake every sleeping body (the static structure under them changed)
@compute @workgroup_size(64) fn wake_all(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + 1u; if (i >= P.n1) { return; }
  if (B(i, F_P).w == 0.0) { return; }
  wake(i);
}

@compute @workgroup_size(64) fn narrow(@builtin(global_invocation_id) g: vec3u) {
  let k = g.x; if (k >= min(ld(C_NPAIR), P.max_pairs)) { return; }
  let si = ld(P.o_pairs + 2u * k); let sj = ld(P.o_pairs + 2u * k + 1u);
  let bi = sbody(si);
  let vi = S(bi, S_V).xyz;
  if (sj == 0u) {
    let gc = ground_contact(Hs(si, H_WP).xyz, Hs(si, H_WQ), Hs(si, H_H).xyz, P.spec + max(-vi.y, 0.0) * P.dt);
    if (gc.cnt > 0u) { emit(si, 0u, bi, 0u, gc); }
    return;
  }
  atomicAdd(&ints[C_PAIRS], 1u);
  let bj = sbody(sj);
  let hi_ = Hs(si, H_H).xyz; let hj_ = Hs(sj, H_H).xyz;
  let ri = length(Hs(si, H_LP).xyz) + length(hi_); let rj = length(Hs(sj, H_LP).xyz) + length(hj_);
  let rel = length(S(bj, S_V).xyz - vi) + length(S(bi, S_W).xyz) * ri + length(S(bj, S_W).xyz) * rj;
  let c = box_box(Hs(si, H_WP).xyz, qmat(Hs(si, H_WQ)), hi_, Hs(sj, H_WP).xyz, qmat(Hs(sj, H_WQ)), hj_, P.spec + rel * P.dt);
  if (c.cnt > 0u) {
    // a moving body wakes a sleeping one it touches (it joins the solver next step) when it moves enough to
    // matter: a piece just woken (at rest) or creeping does not wake its neighbours, so a pile wakes only as far
    // as the disturbance really goes
    let spi = length(vi) + length(S(bi, S_W).xyz) * ri; let spj = length(S(bj, S_V).xyz) + length(S(bj, S_W).xyz) * rj;
    if (dyn_awake(bi) && spi > P.wake_speed) { wake(bj); }
    if (dyn_awake(bj) && spj > P.wake_speed) { wake(bi); }
    emit(si, sj, bi, bj, c);
  }
}

// ---------------------------------------------------------------- colouring
fn mhdr(m: u32) -> vec2u { let h = man[m * MS]; return vec2u(u32(h.x), u32(h.y)); }
@compute @workgroup_size(64) fn color(@builtin(global_invocation_id) g: vec3u) {
  let mi = g.x; if (mi >= min(ld(C_MAN), P.max_man)) { return; }
  let ab = mhdr(P.cur_base + mi); let a = ab.x; let b = ab.y;
  let da = dyn_awake(a); let db = dyn_awake(b);
  var col = P.color_limit;
  for (var c = 0u; c < P.color_limit; c++) {
    let bit = 1u << c;
    if (da) { let o = atomicOr(&ints[P.o_mask + a], bit); if ((o & bit) != 0u) { continue; } }
    if (db) {
      let o = atomicOr(&ints[P.o_mask + b], bit);
      if ((o & bit) != 0u) { if (da) { atomicAnd(&ints[P.o_mask + a], ~bit); } continue; }
    }
    col = c; break;
  }
  st(P.o_mcolor + mi, col);
  atomicAdd(&ints[P.o_ccount + col], 1u);
  if (col == P.color_limit) {
    atomicAdd(&ints[C_OVF], 1u);
    if (da) { atomicAdd(&ints[P.o_ocount + a], 1u); }
    if (db) { atomicAdd(&ints[P.o_ocount + b], 1u); }
  } else { atomicMax(&ints[C_COLORS], col + 1u); }
  if (da) { atomicAdd(&ints[P.o_bcount + a], 1u); }
  if (db) { atomicAdd(&ints[P.o_bcount + b], 1u); }
}
@compute @workgroup_size(64) fn color_scatter(@builtin(global_invocation_id) g: vec3u) {
  let mi = g.x; if (mi >= min(ld(C_MAN), P.max_man)) { return; }
  let c = ld(P.o_mcolor + mi);
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
    let cnt = u32(man[base].z);
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
fn solve_man(m: u32, use_bias: bool, split: bool) {
  let base = m * MS;
  let hd = man[base]; let a = u32(hd.x); let b = u32(hd.y); let cnt = u32(hd.z);
  let va4 = S(a, S_V); let vb4 = S(b, S_V);
  var va = va4.xyz; var wa = S(a, S_W).xyz; var vb = vb4.xyz; var wb = S(b, S_W).xyz;
  let va0 = va; let wa0 = wa; let vb0 = vb; let wb0 = wb;
  var ka = 1.0; var kb = 1.0;
  if (split) { ka = f32(max(ld(P.o_ocount + a), 1u)); kb = f32(max(ld(P.o_ocount + b), 1u)); }
  let ima = va4.w * ka; let imb = vb4.w * kb;
  let A0 = S(a, S_I0) * ka; let A1 = S(a, S_I1) * ka; let B0 = S(b, S_I0) * kb; let B1 = S(b, S_I1) * kb;
  let tha = S(a, S_TH).xyz; let thb = S(b, S_TH).xyz;
  let dpab = S(b, S_DP).xyz - S(a, S_DP).xyz;
  let n4 = man[base + M_N]; let n = n4.xyz;
  let dn = dot(dpab, n);
  var jn = man[base + M_JN];
  var tot = 0.0;
  for (var k = 0u; k < cnt; k++) {
    let pa = man[base + M_PT + 2u * k]; let pb = man[base + M_PT + 2u * k + 1u];
    let raxn = cross(pa.xyz, n); let rbxn = cross(pb.xyz, n);
    let s = pa.w + dn + dot(thb, rbxn) - dot(tha, raxn);
    var bias = 0.0; var ms = 1.0; var is_ = 0.0;
    if (s > 0.0) { bias = s * P.inv_h; }
    else if (use_bias) { bias = max(P.bias_rate * s, -P.max_push); ms = P.mass_scale; is_ = P.imp_scale; }
    let vn = dot(vb - va, n) + dot(wb, rbxn) - dot(wa, raxn);
    let j0 = jn[k];
    var nm = pb.w;
    if (split) { let kk = ima + imb + dot(raxn, imul(A0, A1, raxn)) + dot(rbxn, imul(B0, B1, rbxn)); nm = select(0.0, 1.0 / kk, kk > 0.0); }
    let nj = max(j0 - nm * ms * (vn + bias) - is_ * j0, 0.0);
    let imp = nj - j0;
    jn[k] = nj; tot += nj;
    va -= n * (imp * ima); wa -= imul(A0, A1, raxn) * imp;
    vb += n * (imp * imb); wb += imul(B0, B1, rbxn) * imp;
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
  if (split) {
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
  if (split) {
    if (ima > 0.0) { acc_add(a, va - va0, wa - wa0); }
    if (imb > 0.0) { acc_add(b, vb - vb0, wb - wb0); }
    return;
  }
  if (ima > 0.0) { setS(a, S_V, vec4f(va, ima)); setS(a, S_W, vec4f(wa, 0.0)); }
  if (imb > 0.0) { setS(b, S_V, vec4f(vb, imb)); setS(b, S_W, vec4f(wb, 0.0)); }
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
  let inv = 1.0 / f32(c);
  setS(i, S_V, vec4f(v4.xyz + dv * inv, v4.w)); setS(i, S_W, vec4f(S(i, S_W).xyz + dw * inv, 0.0));
}

// restitution after the sub-steps (Box2D v3 style): points that approached faster than the threshold get
// their normal velocity reflected by the pair's restitution
fn restitute_man(m: u32) {
  let base = m * MS;
  // only contacts that pushed during the step (the last sub-step may already be separating)
  let out = man[base + M_OUT]; let rest = out.x; if (rest == 0.0 || out.y == 0.0) { return; }
  let hd = man[base]; let a = u32(hd.x); let b = u32(hd.y); let cnt = u32(hd.z);
  let va4 = S(a, S_V); let vb4 = S(b, S_V);
  var va = va4.xyz; var wa = S(a, S_W).xyz; var vb = vb4.xyz; var wb = S(b, S_W).xyz;
  let ima = va4.w; let imb = vb4.w;
  let A0 = S(a, S_I0); let A1 = S(a, S_I1); let B0 = S(b, S_I0); let B1 = S(b, S_I1);
  let n = man[base + M_N].xyz; let rv = man[base + M_RV];
  var jn = man[base + M_JN];
  // the points of a patch are coupled (one point's bounce changes the others' velocity): iterate
  for (var it = 0u; it < 8u; it++) { for (var k = 0u; k < cnt; k++) {
    if (rv[k] > -P.rest_threshold) { continue; }
    let pa = man[base + M_PT + 2u * k]; let pb = man[base + M_PT + 2u * k + 1u];
    let raxn = cross(pa.xyz, n); let rbxn = cross(pb.xyz, n);
    let vn = dot(vb - va, n) + dot(wb, rbxn) - dot(wa, raxn);
    let nj = max(jn[k] - pb.w * (vn + rest * rv[k]), 0.0);
    let imp = nj - jn[k]; jn[k] = nj;
    va -= n * (imp * ima); wa -= imul(A0, A1, raxn) * imp;
    vb += n * (imp * imb); wb += imul(B0, B1, rbxn) * imp;
  } }
  man[base + M_JN] = jn;
  if (ima > 0.0) { setS(a, S_V, vec4f(va, ima)); setS(a, S_W, vec4f(wa, 0.0)); }
  if (imb > 0.0) { setS(b, S_V, vec4f(vb, imb)); setS(b, S_W, vec4f(wb, 0.0)); }
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
@compute @workgroup_size(64) fn solve_ovf(@builtin(global_invocation_id) g: vec3u) { let m = ovf_item(g.x); if (m != 0xffffffffu) { solve_man(m, true, true); } }
@compute @workgroup_size(64) fn relax_ovf(@builtin(global_invocation_id) g: vec3u) { let m = ovf_item(g.x); if (m != 0xffffffffu) { solve_man(m, false, true); } }
@compute @workgroup_size(1) fn restitute_ovf(@builtin(workgroup_id) wg: vec3u) {
  if (wg.x != 0u) { return; }
  let r = ovf_range(); for (var k = r.x; k < r.y; k++) { restitute_man(ld(P.o_order + k)); }
}

// ---------------------------------------------------------------- end of step, sleeping
// hard impacts this step, for the damage model: [body a, body b, shape a, shape b, point xyz, normal xyz,
// impulse (N s), approach speed (m/s)] as raw words
@compute @workgroup_size(64) fn events(@builtin(global_invocation_id) g: vec3u) {
  let mi = g.x; if (mi >= min(ld(C_MAN), P.max_man)) { return; }
  let m = P.cur_base + mi; let base = m * MS;
  let out = man[base + M_OUT];
  if (out.z < P.ev_speed || 0.5 * out.y * out.z < P.ev_energy) { return; }
  let k = atomicAdd(&ints[C_EVENTS], 1u); if (k >= P.max_events) { return; }
  let hd = man[base]; let a = u32(hd.x); let cnt = u32(hd.z);
  var pt = vec3f(0.0); for (var q = 0u; q < cnt; q++) { pt += man[base + M_PT + 2u * q].xyz; }
  pt = pt / f32(max(cnt, 1u)) + B(a, F_P).xyz;
  let n = man[base + M_N].xyz;
  let o = P.o_events + k * 12u;
  st(o, a); st(o + 1u, u32(hd.y)); st(o + 2u, u32(cold[m * CS + 4u].w)); st(o + 3u, u32(cold[m * CS + 5u].w));
  st(o + 4u, bitcast<u32>(pt.x)); st(o + 5u, bitcast<u32>(pt.y)); st(o + 6u, bitcast<u32>(pt.z));
  st(o + 7u, bitcast<u32>(n.x)); st(o + 8u, bitcast<u32>(n.y)); st(o + 9u, bitcast<u32>(n.z));
  st(o + 10u, bitcast<u32>(out.y)); st(o + 11u, bitcast<u32>(out.z));
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
      if (ld(P.o_cawake + chash(vec3i(cx, cy, cz), L)) != 0u) { return true; }
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
const P_NN = 24u; const P_NS = 12u; const P_CS = 38u;
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
  var corr = vec3f(0.0); var n = 0.0; var touch = 0.0;
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
    touch = 1.0; if (dist >= rr) { continue; }
    let moving = R(j, 1).w == 1.0;
    let wgt = select(1.0, 0.5, moving);
    let dj = select(vec3f(0.0), rj.xyz - R(j, 3).xyz, moving);
    let pen = min((rr - dist) * wgt, (max(-dot((xi - xs) - dj, nn), 0.0) + cap) * wgt);
    corr += nn * pen + pfric((xi - xs) - dj, nn, pen, mui) * wgt; n += 1.0;
  }
  var x = xi + corr * select(1.0, 1.5 / n, n > 1.5);
  // a particle pushed hard from opposite sides by two bodies (a box set down on it on the floor) is crushed
  var lastN = vec3f(0.0); var lastPush = 0.0; var crushed = R(i, 6).w;
  // then the ground and the rigid shapes, in sequence, each in full: a particle never ends inside a body
  let pg0 = ri - x.y;
  if (pg0 > -0.01) { touch = 1.0; }
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
    if (lastPush > 0.3 * ri && pen > 0.3 * ri && dot(nn, lastN) < -0.5) { crushed = 1.0; }
    if (pen > lastPush) { lastPush = pen; lastN = nn; }
  }
  setR(i, 5, vec4f(x, touch));
  if (crushed > 0.0) { let r6 = R(i, 6); setR(i, 6, vec4f(r6.xyz, 1.0)); }
}
@compute @workgroup_size(64) fn p_apply(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= P.np1 || R(i, 1).w != 1.0) { return; }
  let x = R(i, 5).xyz; let r0 = R(i, 0);
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
    if (touch > 0.0 && distance(r0.xyz, r6.xyz) < 0.01 && length(v) < 2.0) { t = r3.w + P.dt; } else { setR(i, 6, vec4f(r0.xyz, r6.w)); }
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
function buildBodies(list) {
  const bodies = [], shapes = [];
  for (const b of list) {
    let M = 0, com = [0, 0, 0];
    const ms = b.shapes.map(s => (b.fixed ? 0 : MATERIALS[s.mat].density) * 8 * s.h[0] * s.h[1] * s.h[2]);
    b.shapes.forEach((s, k) => { M += ms[k]; for (let i = 0; i < 3; i++) com[i] += ms[k] * s.p[i]; });
    if (M > 0) com = com.map(c => c / M); else { com = [0, 0, 0]; b.shapes.forEach(s => { for (let i = 0; i < 3; i++) com[i] += s.p[i] / b.shapes.length; }); }
    let q = [0, 0, 0, 1], il = [0, 0, 0];
    if (M > 0) {
      const I = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      b.shapes.forEach((s, k) => {
        const m = ms[k], [x2, y2, z2] = s.h.map(v => 4 * v * v);
        const D = [m * (y2 + z2) / 12, m * (x2 + z2) / 12, m * (x2 + y2) / 12];
        const R = q2m(s.q), d = [0, 1, 2].map(i => s.p[i] - com[i]), dd = d[0] ** 2 + d[1] ** 2 + d[2] ** 2;
        for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
          let v = 0; for (let k2 = 0; k2 < 3; k2++) v += R[r][k2] * D[k2] * R[c][k2];
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
      shapes.push({ body: bi, mat: s.mat, col: s.col ?? s.mat, lp: mtv(Rb, [0, 1, 2].map(i => s.p[i] - com[i])), lq: qmulJ(qconj(q), s.q), h: s.h });
    }
  }
  return { bodies, shapes };
}

class GpuWorld {
  // list: bodies as for buildBodies
  static async create(device, list, opt = {}) {
    const w = new GpuWorld();
    await w.init(device, list, opt);
    return w;
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
    this.colorLimit = opt.colorLimit || 24;
    // most sequential colours per solve; contacts beyond go to the parallel (Jacobi) pass. Each colour is a
    // dispatch per sub-step, and dispatch count is what a step costs at this scale
    this.maxColors = opt.maxColors || 16;
    this.colorLimit = Math.min(this.colorLimit, this.maxColors);
    this.indirect = !!opt.indirect;   // the old indirect-dispatch path (for comparison)
    this.anchor = opt.anchor ?? true;
    this.sleep = opt.sleep ?? true;
    this.wakeSpeed = opt.wakeSpeed ?? 0.25;   // m/s at the contact: slower bodies do not wake what they touch
    this.levels = 4;
    this.maxMan = opt.maxContacts || Math.max(64, Math.ceil((opt.manPerShape || 4) * this.capS));
    this.maxPairs = opt.maxPairs || Math.max(128, Math.ceil((opt.pairsPerShape || 8) * this.capS));
    let G = 256; while (G < 2 * cs1) G *= 2; this.G = G;
    let T = 256; while (T < 2 * this.maxMan) T *= 2; this.T = T;
    // base cell: a typical brick with motion margin; larger shapes go to coarser levels (x4 each)
    const ext = shapes.map(s => 2 * Math.hypot(...s.h)).sort((a, b) => a - b);
    this.cell = Math.max(0.5, (ext[Math.floor(ext.length * 0.5)] || 1) + 0.25);
    let o = 16;
    const L = {};
    const take = (k, len) => { L[k] = o; o += len; };
    take('gcount', G); take('gstart', G); take('gfill', G); take('cawake', G);
    take('sorted', cs1); take('cell', cs1); take('mask', cb1);
    take('ccount', 33); take('cstart', 33); take('cfill', 33);
    take('order', this.maxMan); take('mcolor', this.maxMan);
    take('t0', T); take('t1', T);
    take('pairs', 2 * this.maxPairs); take('bcount', cb1); take('bstart', cb1); take('bfill', cb1);
    take('blist', 2 * this.maxMan); take('par', cb1); take('mint', cb1); take('ocount', cb1); take('acc', 6 * cb1);
    this.maxEvents = 512; take('events', 12 * this.maxEvents);
    this.maxInherit = 4096; take('inherit', 12 * this.maxInherit);
    this.L = L; this.nInts = o;

    const bod = new Float32Array(cb1 * 20), sol = new Float32Array(cb1 * 24), shp = new Float32Array(cs1 * 32);
    bodies.forEach((b, k) => {
      const i = k + 1;
      bod.set([...b.p, b.im, ...b.q, ...b.il, 0, 0, ...b.p, ...b.q], i * 20);
      sol.set([...b.v, 0, ...b.w, 0], i * 24);
    });
    shapes.forEach((s, k) => {
      const i = k + 1;
      shp.set([s.body, s.mat, 0, 0, ...s.lp, 0, ...s.lq, ...s.h, 0], i * 32);
    });
    const U = GPUBufferUsage;
    const buf = (size, usage, data) => { const b = d.createBuffer({ size, usage }); if (data) d.queue.writeBuffer(b, 0, data); return b; };
    this.bodBuf = buf(bod.byteLength, U.STORAGE | U.COPY_DST | U.COPY_SRC, bod);
    this.solBuf = buf(sol.byteLength, U.STORAGE | U.COPY_DST | U.COPY_SRC, sol);
    this.shpBuf = buf(shp.byteLength, U.STORAGE | U.COPY_DST | U.COPY_SRC, shp);
    this.intBuf = buf(this.nInts * 4, U.STORAGE | U.COPY_DST | U.COPY_SRC);
    this.manBuf = buf(2 * this.maxMan * 17 * 16, U.STORAGE | U.COPY_SRC);
    this.coldBuf = buf(2 * this.maxMan * 6 * 16, U.STORAGE);
    this.parBuf = buf(512, U.UNIFORM | U.COPY_DST);
    const cu = new Uint32Array(64 * 33); for (let c = 0; c < 33; c++) cu[c * 64] = c;
    this.colBuf = buf(256 * 33, U.UNIFORM | U.COPY_DST, cu);
    this.argBuf = buf(128 * 4, U.STORAGE | U.INDIRECT);
    this.evSpeed = opt.eventSpeed ?? 2.0; this.evEnergy = opt.eventEnergy ?? 0;   // events off unless asked for
    // a ring of stats read-back buffers: the browser delivers mappings late, so one pending read must not hold
    // back the next step's stats and impact events
    this.statBufs = [0, 1, 2].map(() => ({ buf: buf(64 + 48 * this.maxEvents + 33 * 4, U.COPY_DST | U.MAP_READ), busy: false }));
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
    this.pintBuf = buf((3 * PG + 2 * cp + 38 * cp) * 4, U.STORAGE | U.COPY_DST);
    const bglP = d.createBindGroupLayout({ entries: [sto(0), sto(1)] });
    const partL = d.createPipelineLayout({ bindGroupLayouts: [bgl0, bgl1, bglP] });
    this.bgP = d.createBindGroup({ layout: bglP, entries: [ent(0, this.prtBuf), ent(1, this.pintBuf)] });
    const mod = await compile(WGSL, 'WGSL'), modA = await compile(WGSL_ARGS, 'WGSL args');
    const modP = this.capP ? await compile(WGSL_P, 'WGSL particles') : null;
    const mk = (entry, layout = mainL, module = mod) => d.createComputePipelineAsync({ layout, compute: { module, entryPoint: entry } });
    const names = ['prep_bodies', 'prep_shapes', 'scan', 'scatter', 'pairs', 'narrow', 'color', 'color_scatter', 'adj_scan', 'integrate_warm',
                   'integrate_pos', 'finish', 'solve', 'relax', 'restitute', 'solve_ovf', 'relax_ovf', 'restitute_ovf', 'uf_union', 'uf_min', 'uf_sleep', 'ovf_apply', 'events', 'wake_all', 'inherit', 'retire'];
    this.pl = {};
    await Promise.all([...names.map(async k => { this.pl[k] = await mk(k); }),
      ...['man_args', 'pair_args', 'color_scan'].map(async k => { this.pl[k] = await mk(k, argsL, modA); }),
      ...(modP ? ['p_gather', 'p_place', 'p_prep', 'p_scan', 'p_scatter', 'p_integrate', 'p_contact', 'p_apply', 'p_finish'] : []).map(async k => { this.pl[k] = await mk(k, partL, modP); })]);
    this.ts = d.features.has('timestamp-query');
    this.stages = ['broad', 'narrow', 'colour', 'sub-steps', 'particles', 'sleep'];
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
    const f = new Float32Array(128), u = new Uint32Array(f.buffer);
    const cur = this.frame & 1, L = this.L;
    f.set([0, -9.81, 0, h, invH, omega / a1, a2 * a3, a3, 3.0, 1.0, 0.02, dt, this.cell]);
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
    u[52] = L.ocount; u[53] = L.acc; u[54] = L.events; u[55] = this.maxEvents; u[58] = L.cawake;
    f[56] = this.evSpeed; f[57] = this.evEnergy;
    u[60] = this.inheritCount || 0; u[61] = this.L.inherit; f[62] = this.wakeSpeed;
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
      const list = this.pendingInherit.splice(0, this.maxInherit);
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
    if (this.statCopied === false) { clr(0, 7); clr(8, 8); } else clr(0, 16);
    clr(L.gcount, this.Geff); clr(L.gfill, this.Geff); clr(L.cawake, this.Geff); clr(L.ccount, 33); clr(L.cfill, 33); clr(cur ? L.t1 : L.t0, this.T);
    clr(L.bcount, this.n1); clr(L.bfill, this.n1); clr(L.ocount, this.n1);
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
    const dirWg = { 396: wgCap(this.maxPairs), 384: perMan };   // narrow (args 99), per-manifold (args 96)
    const ind = this.indirect
      ? (p, k, off) => { p.setPipeline(this.pl[k]); p.dispatchWorkgroupsIndirect(this.argBuf, off); }
      : (p, k, off) => { p.setPipeline(this.pl[k]); p.dispatchWorkgroups(k === 'restitute_ovf' ? 1 : dirWg[off] ?? perMan); };
    let p;
    if (this.wakeReq) { this.wakeReq = false; p = pass(null); run(p, 'wake_all', nb); p.end(); }
    if (this.inheritCount) {
      p = pass(null); run(p, 'inherit', Math.ceil(this.inheritCount / 64));
      if (this.capP) { p.setBindGroup(2, this.bgP); run(p, 'p_place', Math.ceil(this.inheritCount / 64)); }
      run(p, 'retire', Math.ceil(this.inheritCount / 64)); p.end();
    }
    p = pass('broad'); run(p, 'prep_bodies', nb); run(p, 'prep_shapes', nsh); run(p, 'scan', 1); run(p, 'scatter', nsh); run(p, 'pairs', nsh); p.end();
    p = pass(null, true); run(p, 'pair_args', 1); p.end();
    p = pass('narrow'); ind(p, 'narrow', 99 * 4); p.end();
    p = pass(null, true); run(p, 'man_args', 1); p.end();
    p = pass('colour'); ind(p, 'color', 96 * 4); run(p, 'adj_scan', 1); p.end();
    p = pass(null, true); run(p, 'color_scan', 1); p.end();
    p = pass('sub-steps');
    ind(p, 'color_scatter', 96 * 4);
    const C = this.colorLimit;
    const colors = (k) => {
      p.setPipeline(this.pl[k]);
      for (let c = 0; c < C; c++) { p.setBindGroup(1, this.bg1, [c * 256]); if (this.indirect) p.dispatchWorkgroupsIndirect(this.argBuf, c * 12); else p.dispatchWorkgroups(colWg[c]); }
      p.setBindGroup(1, this.bg1, [0]);
      if (k === 'restitute') { ind(p, 'restitute_ovf', C * 12); return; }
      // overflow contacts: two Jacobi rounds with mass splitting
      for (let it = 0; it < 2; it++) { ind(p, k + '_ovf', C * 12); run(p, 'ovf_apply', nb); }
    };
    for (let s = 0; s < this.substeps; s++) {
      run(p, 'integrate_warm', nb);
      colors('solve');
      run(p, 'integrate_pos', nb);
      colors('relax');
    }
    colors('restitute');
    run(p, 'finish', nb);
    if (this.evEnergy > 0) ind(p, 'events', 96 * 4);
    p.end();
    // particles, after the bodies moved (they see this step's shapes and body velocities)
    p = pass('particles');
    if (this.capP && this.np) {
      p.setBindGroup(2, this.bgP);
      const np = Math.ceil(this.np / 64);
      run(p, 'p_prep', np); run(p, 'p_scan', 1); run(p, 'p_scatter', np);
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
      this.statCopied = true;
    } else this.statCopied = false;
    if (this.inheritCount) {
      this.inheritCount = 0;
      // the retired parents' slots are free from now on (their pieces have been placed from them)
      for (const b of this.retiring) (this.freeBodies = this.freeBodies || []).push(b);
      this.retiring = [];
    }
    this.frame++;
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
    const bslot = bodies.map(() => this.freeBodies.length ? this.freeBodies.pop() : (this.n++, this.n1++, this.n1 - 1));
    this.bodyShapes = this.bodyShapes || new Map();
    for (let k = 0; k < bodies.length; k++) {
      const b = bodies[k], i = bslot[k];
      this.bodyShapes.set(i, []);
      q.writeBuffer(this.bodBuf, i * 80, new Float32Array([...b.p, b.im, ...b.q, ...b.il, 0, 0, ...b.p, ...b.q]));
      q.writeBuffer(this.solBuf, i * 96, new Float32Array([...b.v, 0, ...b.w, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]));
      this.bodies[i - 1] = b;
    }
    for (const s of shapes) {
      const slot = this.freeShapes.length ? this.freeShapes.pop() : (this.ns++, this.ns1++, this.ns1 - 1);
      this.shapeGen = this.shapeGen || []; this.shapeGen[slot - 1] = (this.shapeGen[slot - 1] || 0) + 1;
      const body = bslot[s.body - 1];
      q.writeBuffer(this.shpBuf, slot * 128, new Float32Array([body, s.mat, 0, 0, ...s.lp, 0, ...s.lq, ...s.h, 0]));
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
  setStaticShape(slot, body, centre, half, mat) {
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
    const rec = { body, mat, lp, lq: [0, 0, 0, 1], h: half };
    this.shapes[slot - 1] = rec;
    this.device.queue.writeBuffer(this.shpBuf, slot * 128, new Float32Array([body, mat, 0, 0, ...lp, 0, 0, 0, 0, 1, ...half, 0]));
    return slot;
  }
  // Retire a body: it stops moving and colliding (its slot is not reused).
  killBody(body) {
    this.inputs++;
    const b = this.bodies[body - 1]; if (!b || b.dead) return;
    b.dead = true;
    const own = this.bodyShapes && this.bodyShapes.get(body);
    if (own) { for (const k of own) if (this.shapes[k - 1] && this.shapes[k - 1].body === body) this.killShape(k); this.bodyShapes.delete(body); }
    else this.shapes.forEach((s, k) => { if (s && s.body === body && !s.dead) this.killShape(k + 1); });
    (this.freeBodies = this.freeBodies || []).push(body);
    this.device.queue.writeBuffer(this.bodBuf, body * 80, new Float32Array([0, -1000, 0, 0]));
    this.device.queue.writeBuffer(this.bodBuf, body * 80 + 48, new Float32Array([1, 0, -1000, 0]));
  }
  // Take the impact events collected since the last call.
  // pieces made by breakBody that the next step still has to place (a read-back now would show them wrong)
  placing() { return !!(this.pendingInherit && this.pendingInherit.length); }
  // Break a moving body into new ones (see the `inherit` kernel): `kids` are slots from addBodies, made from
  // the parent's pose (pr, qr) of an older read-back; dv[k] is extra velocity for kid k (world frame then).
  // The parent stops colliding now and is retired at the next step, after its pieces have been placed.
  breakBody(parent, kids, pr, qr, dvs, parts = [], pdvs = []) {
    this.inputs++;
    const b = this.bodies[parent - 1]; if (!b || b.dead) return;
    b.dead = true;
    const own = this.bodyShapes && this.bodyShapes.get(parent);
    if (own) { for (const k of own) if (this.shapes[k - 1] && this.shapes[k - 1].body === parent) this.killShape(k); this.bodyShapes.delete(parent); }
    this.pendingInherit = this.pendingInherit || []; this.retiring = this.retiring || [];
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
    this.retiring.push(parent);
  }
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
    sb.buf.mapAsync(GPUMapMode.READ).then(() => {
      const raw = sb.buf.getMappedRange().slice(0);
      sb.buf.unmap(); sb.busy = false;
      const u = new Uint32Array(raw), f = new Float32Array(raw);
      // events are never lost: every copied step's events count, even when an older copy arrives late
      if (this.evEnergy > 0) {
        const n = Math.min(u[7], this.maxEvents);
        for (let k = 0; k < n; k++) {
          const o = 16 + k * 12;
          this.events.push({ a: u[o], b: u[o + 1], sa: u[o + 2], sb: u[o + 3], p: [f[o + 4], f[o + 5], f[o + 6]], n: [f[o + 7], f[o + 8], f[o + 9]], impulse: f[o + 10], approach: f[o + 11] });
        }
      }
      if (seq < this.statLatest) return;
      this.statLatest = seq; this.statsInputs = sb.inputs;
      this.stats = { manifolds: u[0], overflow: u[1], colors: u[2], pairs: u[3], dropped: u[4], candidates: u[5], awake: u[6], events: u[7], pActive: u[12], pLive: u[13] };
      this.colorLimit = Math.min(this.maxColors, 31, Math.max(4, u[2] + (u[1] > 0 ? 3 : 1)));
      // per-colour contact counts (they size the direct dispatches): keep a slowly decaying maximum
      const cc = new Uint32Array(raw, 64 + 48 * this.maxEvents, 33);
      for (let c = 0; c < 33; c++) this.colCounts[c] = Math.max(cc[c], Math.floor(this.colCounts[c] * 0.95));
    }, () => { sb.busy = false; });
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
    this.shapes.forEach((s, k) => { if (s && s.body < n1 && (!minBody || (!s.dead && s.body >= minBody))) { list.push(s); slots.push(k + 1); gens.push(this.shapeGen ? this.shapeGen[k] || 0 : 0); } });
    const bod = await this.read(this.bodBuf, n1 * 80), sol = await this.read(this.solBuf, n1 * 96);
    const t0 = performance.now();
    const pose = new Float32Array(list.length * 10), mats = new Uint8Array(list.length), cols = new Uint16Array(list.length);
    list.forEach((s, k) => {
      if (!s) return;
      const b = s.body * 20, p = [bod[b], bod[b + 1], bod[b + 2]], q = [bod[b + 4], bod[b + 5], bod[b + 6], bod[b + 7]];
      const wp = mv(q2m(q), s.lp), wq = qmulJ(q, s.lq);
      pose.set([p[0] + wp[0], p[1] + wp[1], p[2] + wp[2], ...wq, ...s.h], k * 10);
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
