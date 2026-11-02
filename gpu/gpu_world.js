// Shatterforge GPU rigid bodies: the same model as engine/src/phys.rs, on WebGPU.
//
// One step, all on the GPU, no read-back:
//   broad     prep (inertia, AABB, grid bucket), bucket scan, scatter, candidate pairs
//   narrow    one thread per pair: box-box SAT, clipping, 4-point reduction, warm-start match, masses
//   colour    greedy graph colouring with atomic per-body masks; per-body contact lists
//   sub-steps integrate + warm start (per body), solve per colour, integrate positions, relax per colour
//   sleep     islands by GPU union-find; an island at rest for 0.5 s sleeps; moving bodies wake what they hit
//
// Data is laid out for bandwidth: contacts are 240 B (derived Jacobians are recomputed, not stored) and the
// solver touches only a 96 B record per body (velocity, displacement, inverse inertia).
'use strict';

const WGSL = /* wgsl */`
struct Params {
  gravity: vec3f, h: f32,
  inv_h: f32, bias_rate: f32, mass_scale: f32, imp_scale: f32,
  max_push: f32, friction: f32, spec: f32, dt: f32,
  cell: f32, n1: u32, grid_mask: u32, max_man: u32,
  tbl_mask: u32, color_limit: u32, wmax: f32, cur_base: u32,
  old_base: u32, o_gcount: u32, o_gstart: u32, o_gfill: u32,
  o_sorted: u32, o_cell: u32, o_mask: u32, o_ccount: u32,
  o_cstart: u32, o_cfill: u32, o_order: u32, o_mcolor: u32,
  o_tnew: u32, o_told: u32, anchor: u32, max_pairs: u32,
  o_pairs: u32, o_bcount: u32, o_bstart: u32, o_bfill: u32,
  o_blist: u32, o_par: u32, o_mint: u32, allow_sleep: u32,
  sleep_time: f32, sleep_lin: f32, sleep_ang: f32, pad0: u32,
}
struct ColorU { c: u32 }

// static body record (6 vec4): position + inverse mass, rotation, half extents + asleep flag,
// local inverse inertia + sleep timer, AABB min, AABB max. Body 0 is the static ground / dummy.
const BS = 6u;
const F_P = 0u; const F_Q = 1u; const F_H = 2u; const F_IL = 3u; const F_LO = 4u; const F_HI = 5u;
// solver record (6 vec4): velocity + effective inverse mass (0 when static or asleep), angular velocity,
// displacement, rotation over the step, inverse inertia (xx, yy, zz, xy), (xz, yz, -, -)
const SS = 6u;
const S_V = 0u; const S_W = 1u; const S_DP = 2u; const S_TH = 3u; const S_I0 = 4u; const S_I1 = 5u;
// contact record (15 vec4): header (a, b, count, stick) as floats; normal + mu; friction anchor arms
// fa + tm1, fb + tm2; anchor offset e0 + twm; (jt1, jt2, jtw, rad); 4 x [(ra, adj), (rb, nm)]; jn x 4
const MS = 15u;
const M_N = 1u; const M_FA = 2u; const M_FB = 3u; const M_E0 = 4u; const M_JT = 5u; const M_PT = 6u; const M_JN = 14u;
// counters in ints[0..16]
const C_MAN = 0u; const C_OVF = 1u; const C_COLORS = 2u; const C_PAIRS = 3u; const C_DROP = 4u; const C_NPAIR = 5u; const C_AWAKE = 6u;

@group(0) @binding(0) var<storage, read_write> bod: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> ints: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> man: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> cold: array<vec4f>;
@group(0) @binding(4) var<uniform> P: Params;
@group(0) @binding(5) var<storage, read_write> sol: array<vec4f>;
@group(1) @binding(0) var<uniform> CU: ColorU;

fn ld(i: u32) -> u32 { return atomicLoad(&ints[i]); }
fn st(i: u32, v: u32) { atomicStore(&ints[i], v); }
fn B(i: u32, f: u32) -> vec4f { return bod[i * BS + f]; }
fn setB(i: u32, f: u32, v: vec4f) { bod[i * BS + f] = v; }
fn S(i: u32, f: u32) -> vec4f { return sol[i * SS + f]; }
fn setS(i: u32, f: u32, v: vec4f) { sol[i * SS + f] = v; }
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
fn cellc(p: vec3f) -> vec3i { return vec3i(floor(p / P.cell)); }
fn chash(c: vec3i) -> u32 { return ((u32(c.x) * 73856093u) ^ (u32(c.y) * 19349663u) ^ (u32(c.z) * 83492791u)) & P.grid_mask; }
fn phash(a: u32, b: u32) -> u32 { return ((a * 2654435761u) ^ ((b + 0x9e3779b9u) * 2246822519u)) & P.tbl_mask; }
fn asleep(i: u32) -> bool { return B(i, F_H).w > 0.5; }
fn dyn_awake(i: u32) -> bool { return S(i, S_V).w > 0.0; }
fn tangents(n: vec3f) -> mat2x3f {
  var t1: vec3f;
  if (abs(n.x) > 0.57) { t1 = vec3f(n.y, -n.x, 0.0); } else { t1 = vec3f(0.0, n.z, -n.y); }
  t1 = normalize(t1);
  return mat2x3f(t1, cross(n, t1));
}

// ---------------------------------------------------------------- broad phase
@compute @workgroup_size(64) fn prep(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + 1u; if (i >= P.n1) { return; }
  let p = B(i, F_P); let q = B(i, F_Q); let h4 = B(i, F_H); let h = h4.xyz; let il = B(i, F_IL).xyz;
  let R = qmat(q);
  let sleeping = h4.w > 0.5;
  var v = S(i, S_V).xyz; var w = S(i, S_W).xyz;
  if (sleeping || p.w == 0.0) {
    v = vec3f(0.0); w = vec3f(0.0);
    setS(i, S_V, vec4f(0.0)); setS(i, S_W, vec4f(0.0)); setS(i, S_I0, vec4f(0.0)); setS(i, S_I1, vec4f(0.0));
  } else {
    let a = R[0] * il.x; let b = R[1] * il.y; let c = R[2] * il.z;
    let c0 = a * R[0].x + b * R[1].x + c * R[2].x;
    let c1 = a * R[0].y + b * R[1].y + c * R[2].y;
    let c2 = a * R[0].z + b * R[1].z + c * R[2].z;
    setS(i, S_V, vec4f(v, p.w));
    setS(i, S_I0, vec4f(c0.x, c1.y, c2.z, c1.x)); setS(i, S_I1, vec4f(c2.x, c2.y, 0.0, 0.0));
    atomicAdd(&ints[C_AWAKE], 1u);
  }
  setS(i, S_DP, vec4f(0.0)); setS(i, S_TH, vec4f(0.0));
  let e = abs(R[0]) * h.x + abs(R[1]) * h.y + abs(R[2]) * h.z;
  let m = (length(v) + length(w) * length(h)) * P.dt + 0.5 * P.spec;
  setB(i, F_LO, vec4f(p.xyz - e - vec3f(m), 0.0));
  setB(i, F_HI, vec4f(p.xyz + e + vec3f(m), 0.0));
  let hc = chash(cellc(p.xyz));
  st(P.o_cell + i, hc);
  atomicAdd(&ints[P.o_gcount + hc], 1u);
  st(P.o_mask + i, 0u);
  st(P.o_par + i, i);
  st(P.o_mint + i, 0x7f7fffffu);
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
  let i = g.x + 1u; if (i >= P.n1) { return; }
  let hc = ld(P.o_cell + i);
  st(P.o_sorted + ld(P.o_gstart + hc) + atomicAdd(&ints[P.o_gfill + hc], 1u), i);
}

fn push_pair(a: u32, b: u32) {
  let k = atomicAdd(&ints[C_NPAIR], 1u);
  if (k < P.max_pairs) { st(P.o_pairs + 2u * k, a); st(P.o_pairs + 2u * k + 1u, b); }
}
@compute @workgroup_size(64) fn pairs(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + 1u; if (i >= P.n1) { return; }
  let lo = B(i, F_LO).xyz; let hi = B(i, F_HI).xyz;
  let ia = dyn_awake(i);
  if (ia && lo.y < 0.0) { push_pair(i, 0u); }
  let c0 = cellc(B(i, F_P).xyz);
  var seen: array<u32, 27>; var ns = 0u;
  for (var dz = -1; dz <= 1; dz++) { for (var dy = -1; dy <= 1; dy++) { for (var dx = -1; dx <= 1; dx++) {
    let hc = chash(c0 + vec3i(dx, dy, dz));
    var dup = false; for (var q = 0u; q < ns; q++) { if (seen[q] == hc) { dup = true; } }
    if (dup) { continue; }
    seen[ns] = hc; ns++;
    let s0 = ld(P.o_gstart + hc); let s1 = s0 + ld(P.o_gcount + hc);
    for (var s = s0; s < s1; s++) {
      let j = ld(P.o_sorted + s);
      if (j <= i) { continue; }
      if (!ia && !dyn_awake(j)) { continue; }   // nothing moves: no contact to solve
      let lj = B(j, F_LO).xyz; let hj = B(j, F_HI).xyz;
      if (any(lo > hj) || any(lj > hi)) { continue; }
      push_pair(i, j);
    }
  } } }
}

// ---------------------------------------------------------------- narrow phase
struct Contact { n: vec3f, cnt: u32, pts: array<vec4f, 8> }  // xyz point, w separation

fn ground_contact(i: u32, margin: f32) -> Contact {
  var c: Contact; c.cnt = 0u; c.n = vec3f(0.0, -1.0, 0.0);
  let p = B(i, F_P).xyz; let R = qmat(B(i, F_Q)); let h = B(i, F_H).xyz;
  for (var k = 0u; k < 8u; k++) {
    let s = vec3f(select(-1.0, 1.0, (k & 1u) != 0u), select(-1.0, 1.0, (k & 2u) != 0u), select(-1.0, 1.0, (k & 4u) != 0u));
    let q = p + R * (s * h);
    if (q.y < margin) { c.pts[c.cnt] = vec4f(q.x, 0.5 * q.y, q.z, q.y); c.cnt++; }
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


fn find_old(a: u32, b: u32) -> u32 {
  var s = phash(a, b);
  for (var probe = 0u; probe < 64u; probe++) {
    let v = ld(P.o_told + s);
    if (v == 0u) { return 0xffffffffu; }
    let m = v - 1u;
    let hd = man[m * MS];
    if (u32(hd.x) == a && u32(hd.y) == b) { return m; }
    s = (s + 1u) & P.tbl_mask;
  }
  return 0xffffffffu;
}
fn insert_new(a: u32, b: u32, idx: u32) {
  var s = phash(a, b);
  for (var probe = 0u; probe < 64u; probe++) {
    let r = atomicCompareExchangeWeak(&ints[P.o_tnew + s], 0u, idx + 1u);
    if (r.exchanged) { return; }
    if (r.old_value == 0u) { continue; }
    s = (s + 1u) & P.tbl_mask;
  }
}

fn emit(a: u32, b: u32, c0: Contact) {
  var c = c0;
  reduce4(&c);
  let slot = atomicAdd(&ints[C_MAN], 1u);
  if (slot >= P.max_man) { atomicAdd(&ints[C_DROP], 1u); return; }
  let mi = P.cur_base + slot;
  let base = mi * MS;
  let n = c.n;
  let tt = tangents(n); let t1 = tt[0]; let t2 = tt[1];
  let pa = B(a, F_P).xyz; let Ra = qmat(B(a, F_Q));
  var pb = vec3f(0.0); var Rb = mat3x3f(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0));
  if (b != 0u) { pb = B(b, F_P).xyz; Rb = qmat(B(b, F_Q)); }
  let ima = S(a, S_V).w; let imb = S(b, S_V).w;
  let A0 = S(a, S_I0); let A1 = S(a, S_I1); let B0 = S(b, S_I0); let B1 = S(b, S_I1);
  var cen = vec3f(0.0);
  for (var k = 0u; k < c.cnt; k++) { cen += c.pts[k].xyz; }
  cen /= f32(c.cnt);
  var rad = 0.0; for (var k = 0u; k < c.cnt; k++) { rad += length(c.pts[k].xyz - cen); }
  rad /= f32(c.cnt);
  let old = find_old(a, b);
  var jt1 = 0.0; var jt2 = 0.0; var jtw = 0.0;
  var fa = cen - pa; var fb = cen - pb; var e0 = vec3f(0.0);
  var laf = transpose(Ra) * fa; var lbf = select(transpose(Rb) * fb, cen, b == 0u);
  var ojn = vec4f(0.0); var ocnt = 0u;
  if (old != 0xffffffffu) {
    let ob = old * MS; let oc = old * 6u;
    let ohd = man[ob]; ocnt = u32(ohd.z); ojn = man[ob + M_JN];
    let olaf = cold[oc + 4u].xyz; let olbf = cold[oc + 5u].xyz;
    if (ohd.w > 0.5) {
      let wa = pa + Ra * olaf;
      var wb = olbf; if (b != 0u) { wb = pb + Rb * olbf; }
      let lim = max(0.05, rad);
      if (dot(wa - cen, wa - cen) < lim * lim && dot(wb - wa, wb - wa) < 0.03 * 0.03) {
        laf = olaf; lbf = olbf; fa = wa - pa; fb = wb - pb; e0 = wb - wa;
      }
    }
    let on = man[ob + M_N].xyz; let ot = tangents(on); let ojt = man[ob + M_JT];
    let f = ot[0] * ojt.x + ot[1] * ojt.y;
    jt1 = dot(f, t1); jt2 = dot(f, t2); jtw = ojt.z * dot(on, n);
  }
  var jn = vec4f(0.0);
  for (var k = 0u; k < 4u; k++) {
    if (k >= c.cnt) { man[base + M_PT + 2u * k] = vec4f(0.0); man[base + M_PT + 2u * k + 1u] = vec4f(0.0); continue; }
    let p = c.pts[k].xyz; let s = c.pts[k].w;
    let ra = p - pa; let rb = p - pb;
    let raxn = cross(ra, n); let rbxn = cross(rb, n);
    let kk = ima + imb + dot(raxn, imul(A0, A1, raxn)) + dot(rbxn, imul(B0, B1, rbxn));
    let la = transpose(Ra) * ra;
    if (old != 0xffffffffu) {
      var best = 0.03 * 0.03;
      for (var q = 0u; q < ocnt; q++) { let d = cold[old * 6u + q].xyz - la; if (dot(d, d) < best) { best = dot(d, d); jn[k] = ojn[q]; } }
    }
    man[base + M_PT + 2u * k] = vec4f(ra, s);
    man[base + M_PT + 2u * k + 1u] = vec4f(rb, select(0.0, 1.0 / kk, kk > 0.0));
    cold[mi * 6u + k] = vec4f(la, 0.0);
  }
  let ca1 = cross(fa, t1); let ca2 = cross(fa, t2); let cb1 = cross(fb, t1); let cb2 = cross(fb, t2);
  let k1 = ima + imb + dot(ca1, imul(A0, A1, ca1)) + dot(cb1, imul(B0, B1, cb1));
  let k2 = ima + imb + dot(ca2, imul(A0, A1, ca2)) + dot(cb2, imul(B0, B1, cb2));
  let kt = dot(n, imul(A0, A1, n)) + dot(n, imul(B0, B1, n));
  man[base] = vec4f(f32(a), f32(b), f32(c.cnt), f32(P.anchor));
  man[base + M_N] = vec4f(n, P.friction);
  man[base + M_FA] = vec4f(fa, select(0.0, 1.0 / k1, k1 > 0.0));
  man[base + M_FB] = vec4f(fb, select(0.0, 1.0 / k2, k2 > 0.0));
  man[base + M_E0] = vec4f(e0, select(0.0, 1.0 / kt, kt > 0.0));
  man[base + M_JT] = vec4f(jt1, jt2, jtw, rad);
  man[base + M_JN] = jn;
  cold[mi * 6u + 4u] = vec4f(laf, 0.0); cold[mi * 6u + 5u] = vec4f(lbf, 0.0);
  insert_new(a, b, mi);
}

fn wake(i: u32) { if (i != 0u && asleep(i)) { let h = B(i, F_H); setB(i, F_H, vec4f(h.xyz, 0.0)); let il = B(i, F_IL); setB(i, F_IL, vec4f(il.xyz, 0.0)); } }

@compute @workgroup_size(64) fn narrow(@builtin(global_invocation_id) g: vec3u) {
  let k = g.x; if (k >= min(ld(C_NPAIR), P.max_pairs)) { return; }
  let i = ld(P.o_pairs + 2u * k); let j = ld(P.o_pairs + 2u * k + 1u);
  let vi = S(i, S_V).xyz;
  if (j == 0u) {
    let gc = ground_contact(i, P.spec + max(-vi.y, 0.0) * P.dt);
    if (gc.cnt > 0u) { emit(i, 0u, gc); }
    return;
  }
  atomicAdd(&ints[C_PAIRS], 1u);
  let hi_ = B(i, F_H).xyz; let hj_ = B(j, F_H).xyz;
  let rel = length(S(j, S_V).xyz - vi) + length(S(i, S_W).xyz) * length(hi_) + length(S(j, S_W).xyz) * length(hj_);
  let c = box_box(B(i, F_P).xyz, qmat(B(i, F_Q)), hi_, B(j, F_P).xyz, qmat(B(j, F_Q)), hj_, P.spec + rel * P.dt);
  if (c.cnt > 0u) {
    // a moving body wakes a sleeping one it touches (it joins the solver next step)
    if (dyn_awake(i) && B(i, F_IL).w == 0.0) { wake(j); }
    if (dyn_awake(j) && B(j, F_IL).w == 0.0) { wake(i); }
    emit(i, j, c);
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
  if (col == P.color_limit) { atomicAdd(&ints[C_OVF], 1u); } else { atomicMax(&ints[C_COLORS], col + 1u); }
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

fn solve_man(m: u32, use_bias: bool) {
  let base = m * MS;
  let hd = man[base]; let a = u32(hd.x); let b = u32(hd.y); let cnt = u32(hd.z);
  let va4 = S(a, S_V); let vb4 = S(b, S_V);
  var va = va4.xyz; var wa = S(a, S_W).xyz; var vb = vb4.xyz; var wb = S(b, S_W).xyz;
  let ima = va4.w; let imb = vb4.w;
  let A0 = S(a, S_I0); let A1 = S(a, S_I1); let B0 = S(b, S_I0); let B1 = S(b, S_I1);
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
    let nj = max(j0 - pb.w * ms * (vn + bias) - is_ * j0, 0.0);
    let imp = nj - j0;
    jn[k] = nj; tot += nj;
    va -= n * (imp * ima); wa -= imul(A0, A1, raxn) * imp;
    vb += n * (imp * imb); wb += imul(B0, B1, rbxn) * imp;
  }
  man[base + M_JN] = jn;
  // patch friction at the anchor, soft-anchored to where the patch first stuck
  let fa4 = man[base + M_FA]; let fb4 = man[base + M_FB]; let e04 = man[base + M_E0]; var jt = man[base + M_JT];
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
  var j1 = o1 - fa4.w * ms * (v1 + b1) - is_ * o1;
  var j2 = o2 - fb4.w * ms * (v2 + b2) - is_ * o2;
  let l2 = j1 * j1 + j2 * j2;
  if (l2 > lim * lim) { let kk = lim * inverseSqrt(max(l2, 1e-30)); j1 *= kk; j2 *= kk; stick = 0.0; }
  let d1 = j1 - o1; let d2 = j2 - o2;
  let pt = t1 * d1 + t2 * d2;
  va -= pt * ima; wa -= imul(A0, A1, ca1 * d1 + ca2 * d2);
  vb += pt * imb; wb += imul(B0, B1, cb1 * d1 + cb2 * d2);
  // twist
  let wl = lim * jt.w;
  let jtw = clamp(jt.z - e04.w * dot(wb - wa, n), -wl, wl);
  let dtw = jtw - jt.z;
  wa -= imul(A0, A1, n) * dtw; wb += imul(B0, B1, n) * dtw;
  man[base + M_JT] = vec4f(j1, j2, jtw, jt.w);
  if (stick != hd.w) { man[base] = vec4f(hd.xyz, stick); }
  if (ima > 0.0) { setS(a, S_V, vec4f(va, ima)); setS(a, S_W, vec4f(wa, 0.0)); }
  if (imb > 0.0) { setS(b, S_V, vec4f(vb, imb)); setS(b, S_W, vec4f(wb, 0.0)); }
}

fn color_item(g: u32) -> u32 {
  let c = CU.c;
  if (g >= ld(P.o_ccount + c)) { return 0xffffffffu; }
  return ld(P.o_order + ld(P.o_cstart + c) + g);
}
@compute @workgroup_size(64) fn solve(@builtin(global_invocation_id) g: vec3u) { let m = color_item(g.x); if (m != 0xffffffffu) { solve_man(m, true); } }
@compute @workgroup_size(64) fn relax(@builtin(global_invocation_id) g: vec3u) { let m = color_item(g.x); if (m != 0xffffffffu) { solve_man(m, false); } }
// overflow contacts (no free colour): one thread, in sequence
fn ovf_range() -> vec2u { let c = P.color_limit; let s = ld(P.o_cstart + c); return vec2u(s, s + ld(P.o_ccount + c)); }
@compute @workgroup_size(1) fn solve_ovf() { let r = ovf_range(); for (var k = r.x; k < r.y; k++) { solve_man(ld(P.o_order + k), true); } }
@compute @workgroup_size(1) fn relax_ovf() { let r = ovf_range(); for (var k = r.x; k < r.y; k++) { solve_man(ld(P.o_order + k), false); } }

// ---------------------------------------------------------------- end of step, sleeping
@compute @workgroup_size(64) fn finish(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + 1u; if (i >= P.n1) { return; }
  let v4 = S(i, S_V); if (v4.w == 0.0) { return; }
  let p = B(i, F_P); setB(i, F_P, vec4f(p.xyz + S(i, S_DP).xyz, p.w));
  let w = S(i, S_W).xyz;
  let il = B(i, F_IL);
  var t = 0.0;
  if (dot(v4.xyz, v4.xyz) < P.sleep_lin * P.sleep_lin && dot(w, w) < P.sleep_ang * P.sleep_ang) { t = il.w + P.dt; }
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
  if (bitcast<f32>(ld(P.o_mint + find(i))) >= P.sleep_time) {
    let h = B(i, F_H); setB(i, F_H, vec4f(h.xyz, 1.0));
    setS(i, S_V, vec4f(0.0, 0.0, 0.0, S(i, S_V).w)); setS(i, S_W, vec4f(0.0));
  }
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
    // the overflow bucket is one sequential thread: one workgroup if it holds anything
    args[c * 3u] = select((n + 63u) / 64u, min(n, 1u), c == P.color_limit); args[c * 3u + 1u] = 1u; args[c * 3u + 2u] = 1u;
  }
}
`;

class GpuWorld {
  // bodies: Float32Array of 10 per body (p.xyz, q.xyzw, h.xyz); density kg/m^3 (0 = static)
  static async create(device, bodies, density, opt = {}) {
    const w = new GpuWorld();
    await w.init(device, bodies, density, opt);
    return w;
  }

  async init(device, bodies, density, opt) {
    const d = this.device = device;
    const n = bodies.length / 10;
    this.n = n; this.n1 = n + 1;
    this.substeps = opt.substeps || 12;
    this.friction = opt.friction ?? 0.6;
    this.colorLimit = opt.colorLimit || 24;
    this.anchor = opt.anchor ?? true;
    this.sleep = opt.sleep ?? true;
    this.maxMan = Math.max(64, Math.ceil((opt.manPerBody || 4) * n));
    this.maxPairs = Math.max(128, Math.ceil((opt.pairsPerBody || 8) * n));
    let G = 256; while (G < 2 * this.n1) G *= 2; this.G = G;
    let T = 256; while (T < 2 * this.maxMan) T *= 2; this.T = T;
    // grid cell: at least the largest AABB (with margin), so overlapping bodies sit in neighbour cells
    let maxExt = 0;
    for (let i = 0; i < n; i++) maxExt = Math.max(maxExt, 2 * Math.hypot(bodies[i * 10 + 7], bodies[i * 10 + 8], bodies[i * 10 + 9]));
    this.cell = Math.max(0.5, maxExt + 0.25);
    let o = 16;
    const L = {};
    const take = (k, len) => { L[k] = o; o += len; };
    take('gcount', G); take('gstart', G); take('gfill', G);
    take('sorted', this.n1); take('cell', this.n1); take('mask', this.n1);
    take('ccount', 33); take('cstart', 33); take('cfill', 33);
    take('order', this.maxMan); take('mcolor', this.maxMan);
    take('t0', T); take('t1', T);
    take('pairs', 2 * this.maxPairs); take('bcount', this.n1); take('bstart', this.n1); take('bfill', this.n1);
    take('blist', 2 * this.maxMan); take('par', this.n1); take('mint', this.n1);
    this.L = L; this.nInts = o;

    const bod = new Float32Array(this.n1 * 6 * 4);
    for (let i = 0; i < n; i++) {
      const s = i * 10, b = (i + 1) * 24;
      const hx = bodies[s + 7], hy = bodies[s + 8], hz = bodies[s + 9];
      const m = density * 8 * hx * hy * hz;
      bod.set([bodies[s], bodies[s + 1], bodies[s + 2], m > 0 ? 1 / m : 0], b);
      bod.set([bodies[s + 3], bodies[s + 4], bodies[s + 5], bodies[s + 6]], b + 4);
      bod.set([hx, hy, hz, 0], b + 8);
      const x2 = 4 * hx * hx, y2 = 4 * hy * hy, z2 = 4 * hz * hz;
      if (m > 0) bod.set([12 / (m * (y2 + z2)), 12 / (m * (x2 + z2)), 12 / (m * (x2 + y2)), 0], b + 12);
    }
    const U = GPUBufferUsage;
    this.bodBuf = d.createBuffer({ size: bod.byteLength, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    d.queue.writeBuffer(this.bodBuf, 0, bod);
    this.solBuf = d.createBuffer({ size: this.n1 * 6 * 16, usage: U.STORAGE | U.COPY_SRC });
    this.intBuf = d.createBuffer({ size: this.nInts * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    this.manBuf = d.createBuffer({ size: 2 * this.maxMan * 15 * 16, usage: U.STORAGE });
    this.coldBuf = d.createBuffer({ size: 2 * this.maxMan * 6 * 16, usage: U.STORAGE });
    this.parBuf = d.createBuffer({ size: 256, usage: U.UNIFORM | U.COPY_DST });
    this.colBuf = d.createBuffer({ size: 256 * 33, usage: U.UNIFORM | U.COPY_DST });
    const cu = new Uint32Array(64 * 33); for (let c = 0; c < 33; c++) cu[c * 64] = c;
    d.queue.writeBuffer(this.colBuf, 0, cu);
    this.argBuf = d.createBuffer({ size: 128 * 4, usage: U.STORAGE | U.INDIRECT });
    this.statBuf = d.createBuffer({ size: 64, usage: U.COPY_DST | U.MAP_READ });
    this.stats = { manifolds: 0, overflow: 0, colors: 0, pairs: 0, dropped: 0, candidates: 0, awake: n };
    this.statPending = false;

    const sto = (binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } });
    const bgl0 = d.createBindGroupLayout({ entries: [sto(0), sto(1), sto(2), sto(3),
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }, sto(5)] });
    const bgl1 = d.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 4 } }] });
    const bgl2 = d.createBindGroupLayout({ entries: [sto(0)] });
    const mainL = d.createPipelineLayout({ bindGroupLayouts: [bgl0, bgl1] });
    const argsL = d.createPipelineLayout({ bindGroupLayouts: [bgl0, bgl1, bgl2] });
    this.bg0 = d.createBindGroup({ layout: bgl0, entries: [
      { binding: 0, resource: { buffer: this.bodBuf } }, { binding: 1, resource: { buffer: this.intBuf } },
      { binding: 2, resource: { buffer: this.manBuf } }, { binding: 3, resource: { buffer: this.coldBuf } },
      { binding: 4, resource: { buffer: this.parBuf } }, { binding: 5, resource: { buffer: this.solBuf } }] });
    this.bg1 = d.createBindGroup({ layout: bgl1, entries: [{ binding: 0, resource: { buffer: this.colBuf, size: 4 } }] });
    this.bg2 = d.createBindGroup({ layout: bgl2, entries: [{ binding: 0, resource: { buffer: this.argBuf } }] });
    const compile = async (code, tag) => {
      const m = d.createShaderModule({ code });
      const errs = (await m.getCompilationInfo()).messages.filter(x => x.type === 'error');
      if (errs.length) throw new Error(tag + ': ' + errs.map(x => `${x.lineNum}:${x.linePos} ${x.message}`).join('\n'));
      return m;
    };
    const mod = await compile(WGSL, 'WGSL'), modA = await compile(WGSL_ARGS, 'WGSL args');
    const mk = (entry, layout = mainL, module = mod) => d.createComputePipelineAsync({ layout, compute: { module, entryPoint: entry } });
    const names = ['prep', 'scan', 'scatter', 'pairs', 'narrow', 'color', 'color_scatter', 'adj_scan', 'integrate_warm',
                   'integrate_pos', 'finish', 'solve', 'relax', 'solve_ovf', 'relax_ovf', 'uf_union', 'uf_min', 'uf_sleep'];
    this.pl = {};
    await Promise.all([...names.map(async k => { this.pl[k] = await mk(k); }),
      ...['man_args', 'pair_args', 'color_scan'].map(async k => { this.pl[k] = await mk(k, argsL, modA); })]);
    // GPU timestamps per stage, when the device has them
    this.ts = d.features.has('timestamp-query');
    this.stages = ['broad', 'narrow', 'colour', 'sub-steps', 'sleep'];
    if (this.ts) {
      this.qs = d.createQuerySet({ type: 'timestamp', count: 32 });
      this.qBuf = d.createBuffer({ size: 256, usage: U.QUERY_RESOLVE | U.COPY_SRC });
      this.qRead = d.createBuffer({ size: 256, usage: U.COPY_DST | U.MAP_READ });
      this.qPending = false; this.gpuMs = null;
    }
    this.frame = 0;
  }

  params(dt) {
    const sub = this.substeps, h = dt / sub, invH = 1 / h;
    const hz = invH / 6, omega = 2 * Math.PI * hz, zeta = 10;
    const a1 = 2 * zeta + h * omega, a2 = h * omega * a1, a3 = 1 / (1 + a2);
    const f = new Float32Array(64), u = new Uint32Array(f.buffer);
    const cur = this.frame & 1, L = this.L;
    f.set([0, -9.81, 0, h, invH, omega / a1, a2 * a3, a3, 3.0, this.friction, 0.02, dt, this.cell]);
    u[13] = this.n1; u[14] = this.G - 1; u[15] = this.maxMan;
    u[16] = this.T - 1; u[17] = this.colorLimit; f[18] = 0.25 * Math.PI * invH; u[19] = cur * this.maxMan;
    u[20] = (1 - cur) * this.maxMan; u[21] = L.gcount; u[22] = L.gstart; u[23] = L.gfill;
    u[24] = L.sorted; u[25] = L.cell; u[26] = L.mask; u[27] = L.ccount;
    u[28] = L.cstart; u[29] = L.cfill; u[30] = L.order; u[31] = L.mcolor;
    u[32] = cur ? L.t1 : L.t0; u[33] = cur ? L.t0 : L.t1; u[34] = this.anchor ? 1 : 0; u[35] = this.maxPairs;
    u[36] = L.pairs; u[37] = L.bcount; u[38] = L.bstart; u[39] = L.bfill;
    u[40] = L.blist; u[41] = L.par; u[42] = L.mint; u[43] = this.sleep ? 1 : 0;
    f[44] = 0.5; f[45] = 0.05; f[46] = 0.08;
    return f;
  }

  // Encode one 60 Hz step into `enc` (several compute passes; one timestamp pair per stage).
  encode(enc, dt = 1 / 60) {
    const d = this.device;
    d.queue.writeBuffer(this.parBuf, 0, this.params(dt));
    const L = this.L, cur = this.frame & 1;
    const clr = (off, len) => enc.clearBuffer(this.intBuf, off * 4, len * 4);
    clr(0, 16); clr(L.gcount, this.G); clr(L.gfill, this.G); clr(L.ccount, 33); clr(L.cfill, 33); clr(cur ? L.t1 : L.t0, this.T);
    clr(L.bcount, this.n1); clr(L.bfill, this.n1);
    const nb = Math.ceil(this.n / 64) || 1;
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
    const ind = (p, k, off) => { p.setPipeline(this.pl[k]); p.dispatchWorkgroupsIndirect(this.argBuf, off); };
    let p = pass('broad'); run(p, 'prep', nb); run(p, 'scan', 1); run(p, 'scatter', nb); run(p, 'pairs', nb); p.end();
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
      for (let c = 0; c < C; c++) { p.setBindGroup(1, this.bg1, [c * 256]); p.dispatchWorkgroupsIndirect(this.argBuf, c * 12); }
      p.setBindGroup(1, this.bg1, [0]);
      ind(p, k + '_ovf', C * 12);
    };
    for (let s = 0; s < this.substeps; s++) {
      run(p, 'integrate_warm', nb);
      colors('solve');
      run(p, 'integrate_pos', nb);
      colors('relax');
    }
    run(p, 'finish', nb);
    p.end();
    if (this.sleep) {
      p = pass('sleep'); ind(p, 'uf_union', 96 * 4); run(p, 'uf_min', nb); run(p, 'uf_sleep', nb); p.end();
    }
    if (stamp) {
      const k = this.stages.length - (this.sleep ? 0 : 1);
      enc.resolveQuerySet(this.qs, 0, 2 * k, this.qBuf, 0);
      enc.copyBufferToBuffer(this.qBuf, 0, this.qRead, 0, 16 * k);
      this.qCopied = k;
    } else this.qCopied = 0;
    if (!this.statPending) { enc.copyBufferToBuffer(this.intBuf, 0, this.statBuf, 0, 64); this.statCopied = true; } else this.statCopied = false;
    this.frame++;
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
      // exponential average, so a single read is not noise
      if (!this.gpuMs) this.gpuMs = ms; else for (const s in ms) this.gpuMs[s] = 0.8 * (this.gpuMs[s] ?? ms[s]) + 0.2 * ms[s];
      this.qPending = false;
    }, () => { this.qPending = false; });
  }

  pollStats() {
    this.pollTimes();
    if (!this.statCopied || this.statPending) return;
    this.statPending = true;
    this.statBuf.mapAsync(GPUMapMode.READ).then(() => {
      const u = new Uint32Array(this.statBuf.getMappedRange().slice(0));
      this.statBuf.unmap();
      this.stats = { manifolds: u[0], overflow: u[1], colors: u[2], pairs: u[3], dropped: u[4], candidates: u[5], awake: u[6] };
      // colour dispatches follow what the scene needs (one spare, three after an overflow); the rest go to the overflow bucket
      this.colorLimit = Math.min(31, Math.max(4, u[2] + (u[1] > 0 ? 3 : 1)));
      this.statPending = false;
    }, () => { this.statPending = false; });
  }

  // Read back bodies as 10 floats each (p, q, h) like the wasm pose() export, plus velocities and sleep flags.
  async readPose() {
    const bs = this.n1 * 6 * 16, ss = this.n1 * 6 * 16;
    const rb = this.device.createBuffer({ size: bs + ss, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.bodBuf, 0, rb, 0, bs); enc.copyBufferToBuffer(this.solBuf, 0, rb, bs, ss);
    this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const f = new Float32Array(rb.getMappedRange());
    const out = new Float32Array(this.n * 10), vel = new Float32Array(this.n * 3), sleeping = new Uint8Array(this.n);
    const so = bs / 4;
    for (let i = 0; i < this.n; i++) {
      const b = (i + 1) * 24, s = so + (i + 1) * 24;
      out.set(f.subarray(b, b + 3), i * 10); out.set(f.subarray(b + 4, b + 8), i * 10 + 3); out.set(f.subarray(b + 8, b + 11), i * 10 + 7);
      vel.set(f.subarray(s, s + 3), i * 3);
      sleeping[i] = f[b + 11] > 0.5 ? 1 : 0;
    }
    rb.unmap(); rb.destroy();
    return { pose: out, vel, sleeping };
  }

  destroy() {
    for (const b of [this.bodBuf, this.solBuf, this.intBuf, this.manBuf, this.coldBuf, this.parBuf, this.colBuf, this.argBuf, this.statBuf, this.qBuf, this.qRead]) if (b) b.destroy();
    if (this.qs) this.qs.destroy();
  }
}
if (typeof window !== 'undefined') window.GpuWorld = GpuWorld;
