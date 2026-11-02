// Shatterforge GPU rigid bodies (phase 2 prototype): the same model as engine/src/phys.rs, on WebGPU.
//
// One step, all on the GPU, no read-back:
//   prep          per body: world inertia, AABB, grid bucket, clear colour mask
//   scan          bucket prefix sum (one workgroup)
//   scatter       bodies sorted by bucket
//   collide       per body: ground + 27 neighbour cells -> box-box SAT, clipping, 4-point reduction,
//                 warm-start match against last step's manifolds, constraint preparation
//   colour        greedy graph colouring with atomic per-body masks (bodies in one colour are disjoint)
//   sub-steps     integrate velocities, warm start, solve (soft), integrate positions, relax (rigid);
//                 one indirect dispatch per colour, so the solve is Gauss-Seidel across colours
//   finish        positions written back
// Manifolds and the pair hash table are double-buffered between steps for warm starting.
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
  o_tnew: u32, o_told: u32, anchor: u32, pad0: u32,
}
struct Pt { raxn: vec4f, rbxn: vec4f, iaxn: vec4f, ibxn: vec4f }  // w: adj, nm, jn, -
struct Man {
  a: u32, b: u32, cnt: u32, stick: u32,
  n: vec4f,   // w mu
  t1: vec4f,  // w rad
  t2: vec4f,  // w twm
  e0: vec4f,  // w tm1
  ca1: vec4f, // w tm2
  ca2: vec4f, // w jt1
  cb1: vec4f, // w jt2
  cb2: vec4f, // w jtw
  ia1: vec4f, // w ima
  ia2: vec4f, // w imb
  ib1: vec4f, ib2: vec4f, ian: vec4f, ibn: vec4f,
  p: array<Pt, 4>,
}
struct Cold { la: array<vec4f, 4>, laf: vec4f, lbf: vec4f }
struct ColorU { c: u32 }

// body record: 13 vec4 per body; body 0 is the static ground / dummy
const BS = 13u;
const F_P = 0u; const F_Q = 1u; const F_H = 2u; const F_IL = 3u; const F_V = 4u; const F_W = 5u;
const F_DP = 6u; const F_TH = 7u; const F_I0 = 8u; const F_LO = 11u; const F_HI = 12u;
// counters in ints[0..16]
const C_MAN = 0u; const C_OVF = 1u; const C_COLORS = 2u; const C_PAIRS = 3u; const C_DROP = 4u;

@group(0) @binding(0) var<storage, read_write> bod: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> ints: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> man: array<Man>;
@group(0) @binding(3) var<storage, read_write> cold: array<Cold>;
@group(0) @binding(4) var<uniform> P: Params;
@group(1) @binding(0) var<uniform> CU: ColorU;

fn ld(i: u32) -> u32 { return atomicLoad(&ints[i]); }
fn st(i: u32, v: u32) { atomicStore(&ints[i], v); }
fn B(i: u32, f: u32) -> vec4f { return bod[i * BS + f]; }
fn setB(i: u32, f: u32, v: vec4f) { bod[i * BS + f] = v; }

fn qrot(q: vec4f, v: vec3f) -> vec3f { let u = q.xyz; let t = 2.0 * cross(u, v); return v + q.w * t + cross(u, t); }
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
fn iiw(i: u32) -> mat3x3f { return mat3x3f(B(i, F_I0).xyz, B(i, F_I0 + 1u).xyz, B(i, F_I0 + 2u).xyz); }
fn cellc(p: vec3f) -> vec3i { return vec3i(floor(p / P.cell)); }
fn chash(c: vec3i) -> u32 { return ((u32(c.x) * 73856093u) ^ (u32(c.y) * 19349663u) ^ (u32(c.z) * 83492791u)) & P.grid_mask; }
fn phash(a: u32, b: u32) -> u32 { return ((a * 2654435761u) ^ ((b + 0x9e3779b9u) * 2246822519u)) & P.tbl_mask; }

// ---------------------------------------------------------------- prep
@compute @workgroup_size(64) fn prep(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + 1u; if (i >= P.n1) { return; }
  let p = B(i, F_P); let q = B(i, F_Q); let h = B(i, F_H).xyz; let il = B(i, F_IL).xyz;
  let R = qmat(q);
  // world inverse inertia R diag(il) R^T (symmetric; stored as columns)
  let a = R[0] * il.x; let b = R[1] * il.y; let c = R[2] * il.z;
  setB(i, F_I0, vec4f(a * R[0].x + b * R[1].x + c * R[2].x, 0.0));
  setB(i, F_I0 + 1u, vec4f(a * R[0].y + b * R[1].y + c * R[2].y, 0.0));
  setB(i, F_I0 + 2u, vec4f(a * R[0].z + b * R[1].z + c * R[2].z, 0.0));
  let e = abs(R[0]) * h.x + abs(R[1]) * h.y + abs(R[2]) * h.z;
  let v = B(i, F_V).xyz; let w = B(i, F_W).xyz;
  let m = (length(v) + length(w) * length(h)) * P.dt + 0.5 * P.spec;
  setB(i, F_LO, vec4f(p.xyz - e - vec3f(m), 0.0));
  setB(i, F_HI, vec4f(p.xyz + e + vec3f(m), 0.0));
  setB(i, F_DP, vec4f(0.0)); setB(i, F_TH, vec4f(0.0));
  let hc = chash(cellc(p.xyz));
  st(P.o_cell + i, hc);
  atomicAdd(&ints[P.o_gcount + hc], 1u);
  st(P.o_mask + i, 0u);
}

var<workgroup> wsum: array<u32, 256>;
@compute @workgroup_size(256) fn scan(@builtin(local_invocation_index) t: u32) {
  let G = P.grid_mask + 1u; let per = G / 256u; let base = t * per;
  var s = 0u;
  for (var k = 0u; k < per; k++) { s += ld(P.o_gcount + base + k); }
  wsum[t] = s;
  workgroupBarrier();
  for (var off = 1u; off < 256u; off *= 2u) {
    var v = 0u; if (t >= off) { v = wsum[t - off]; }
    workgroupBarrier(); wsum[t] += v; workgroupBarrier();
  }
  var run = 0u; if (t > 0u) { run = wsum[t - 1u]; }
  for (var k = 0u; k < per; k++) { st(P.o_gstart + base + k, run); run += ld(P.o_gcount + base + k); }
}

@compute @workgroup_size(64) fn scatter(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + 1u; if (i >= P.n1) { return; }
  let hc = ld(P.o_cell + i);
  let slot = ld(P.o_gstart + hc) + atomicAdd(&ints[P.o_gfill + hc], 1u);
  st(P.o_sorted + slot, i);
}

// ---------------------------------------------------------------- narrow phase
struct Contact { n: vec3f, cnt: u32, pts: array<vec4f, 8> }  // xyz point, w separation

fn tangents(n: vec3f) -> mat2x3f {
  var t1: vec3f;
  if (abs(n.x) > 0.57) { t1 = vec3f(n.y, -n.x, 0.0); } else { t1 = vec3f(0.0, n.z, -n.y); }
  t1 = normalize(t1);
  return mat2x3f(t1, cross(n, t1));
}

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
    if (man[m].a == a && man[m].b == b) { return m; }
    s = (s + 1u) & P.tbl_mask;
  }
  return 0xffffffffu;
}
fn insert_new(a: u32, b: u32, idx: u32) {
  var s = phash(a, b);
  for (var probe = 0u; probe < 64u; probe++) {
    let r = atomicCompareExchangeWeak(&ints[P.o_tnew + s], 0u, idx + 1u);
    if (r.exchanged) { return; }
    if (r.old_value == 0u) { continue; } // spurious failure: retry the same slot
    s = (s + 1u) & P.tbl_mask;
  }
}

fn emit(a: u32, b: u32, c0: Contact) {
  var c = c0;
  reduce4(&c);
  let slot = atomicAdd(&ints[C_MAN], 1u);
  if (slot >= P.max_man) { atomicAdd(&ints[C_DROP], 1u); return; }
  let mi = P.cur_base + slot;
  let n = c.n;
  let tt = tangents(n); let t1 = tt[0]; let t2 = tt[1];
  let pa = B(a, F_P).xyz; let Ra = qmat(B(a, F_Q));
  var pb = vec3f(0.0); var Rb = mat3x3f(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0));
  if (b != 0u) { pb = B(b, F_P).xyz; Rb = qmat(B(b, F_Q)); }
  let ima = B(a, F_P).w; let imb = B(b, F_P).w;
  let Ia = iiw(a); let Ib = iiw(b);
  var m: Man; var cd: Cold;
  m.a = a; m.b = b; m.cnt = c.cnt; m.stick = P.anchor;
  var cen = vec3f(0.0);
  for (var k = 0u; k < c.cnt; k++) { cen += c.pts[k].xyz; }
  cen /= f32(c.cnt);
  var rad = 0.0; for (var k = 0u; k < c.cnt; k++) { rad += length(c.pts[k].xyz - cen); }
  rad /= f32(c.cnt);
  let old = find_old(a, b);
  var jt1 = 0.0; var jt2 = 0.0; var jtw = 0.0;
  var anchored = false; var fa = cen - pa; var fb = cen - pb; var e0 = vec3f(0.0);
  var laf = transpose(Ra) * fa; var lbf = select(transpose(Rb) * fb, cen, b == 0u);
  if (old != 0xffffffffu) {
    let om = man[old]; let oc = cold[old];
    if (om.stick != 0u) {
      let wa = pa + Ra * oc.laf.xyz;
      var wb = oc.lbf.xyz; if (b != 0u) { wb = pb + Rb * oc.lbf.xyz; }
      let lim = max(0.05, rad);
      if (dot(wa - cen, wa - cen) < lim * lim && dot(wb - wa, wb - wa) < 0.03 * 0.03) {
        laf = oc.laf.xyz; lbf = oc.lbf.xyz; fa = wa - pa; fb = wb - pb; e0 = wb - wa; anchored = true;
      }
    }
    let f = om.t1.xyz * om.ca2.w + om.t2.xyz * om.cb1.w;
    jt1 = dot(f, t1); jt2 = dot(f, t2); jtw = om.cb2.w * dot(om.n.xyz, n);
  }
  for (var k = 0u; k < 4u; k++) { m.p[k].iaxn = vec4f(0.0); }
  for (var k = 0u; k < c.cnt; k++) {
    let p = c.pts[k].xyz; let s = c.pts[k].w;
    let ra = p - pa; let rb = p - pb;
    let raxn = cross(ra, n); let rbxn = cross(rb, n);
    let iaxn = Ia * raxn; let ibxn = Ib * rbxn;
    let kk = ima + imb + dot(raxn, iaxn) + dot(rbxn, ibxn);
    let la = transpose(Ra) * ra;
    var jn = 0.0;
    if (old != 0xffffffffu) {
      var best = 0.03 * 0.03;
      for (var q = 0u; q < man[old].cnt; q++) { let d = cold[old].la[q].xyz - la; if (dot(d, d) < best) { best = dot(d, d); jn = man[old].p[q].iaxn.w; } }
    }
    m.p[k].raxn = vec4f(raxn, s);
    m.p[k].rbxn = vec4f(rbxn, select(0.0, 1.0 / kk, kk > 0.0));
    m.p[k].iaxn = vec4f(iaxn, jn);
    m.p[k].ibxn = vec4f(ibxn, 0.0);
    cd.la[k] = vec4f(la, 0.0);
  }
  let ca1 = cross(fa, t1); let ca2 = cross(fa, t2); let cb1 = cross(fb, t1); let cb2 = cross(fb, t2);
  let ia1 = Ia * ca1; let ia2 = Ia * ca2; let ib1 = Ib * cb1; let ib2 = Ib * cb2;
  let k1 = ima + imb + dot(ca1, ia1) + dot(cb1, ib1);
  let k2 = ima + imb + dot(ca2, ia2) + dot(cb2, ib2);
  let ian = Ia * n; let ibn = Ib * n;
  let kt = dot(n, ian) + dot(n, ibn);
  m.n = vec4f(n, P.friction);
  m.t1 = vec4f(t1, rad);
  m.t2 = vec4f(t2, select(0.0, 1.0 / kt, kt > 0.0));
  m.e0 = vec4f(e0, select(0.0, 1.0 / k1, k1 > 0.0));
  m.ca1 = vec4f(ca1, select(0.0, 1.0 / k2, k2 > 0.0));
  m.ca2 = vec4f(ca2, jt1);
  m.cb1 = vec4f(cb1, jt2);
  m.cb2 = vec4f(cb2, jtw);
  m.ia1 = vec4f(ia1, ima); m.ia2 = vec4f(ia2, imb);
  m.ib1 = vec4f(ib1, 0.0); m.ib2 = vec4f(ib2, 0.0); m.ian = vec4f(ian, 0.0); m.ibn = vec4f(ibn, 0.0);
  cd.laf = vec4f(laf, 0.0); cd.lbf = vec4f(lbf, 0.0);
  man[mi] = m; cold[mi] = cd;
  insert_new(a, b, mi);
}

@compute @workgroup_size(64) fn collide(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + 1u; if (i >= P.n1) { return; }
  let lo = B(i, F_LO).xyz; let hi = B(i, F_HI).xyz;
  let pi = B(i, F_P); let vi = B(i, F_V).xyz; let wi = B(i, F_W).xyz; let hi_ = B(i, F_H).xyz;
  if (lo.y < 0.0) {
    let gc = ground_contact(i, P.spec + max(-vi.y, 0.0) * P.dt);
    if (gc.cnt > 0u) { emit(i, 0u, gc); }
  }
  let Ri = qmat(B(i, F_Q));
  let c0 = cellc(pi.xyz);
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
      let lj = B(j, F_LO).xyz; let hj = B(j, F_HI).xyz;
      if (any(lo > hj) || any(lj > hi)) { continue; }
      atomicAdd(&ints[C_PAIRS], 1u);
      let pj = B(j, F_P); let hj_ = B(j, F_H).xyz;
      let rel = length(B(j, F_V).xyz - vi) + length(wi) * length(hi_) + length(B(j, F_W).xyz) * length(hj_);
      let margin = P.spec + rel * P.dt;
      let c = box_box(pi.xyz, Ri, hi_, pj.xyz, qmat(B(j, F_Q)), hj_, margin);
      if (c.cnt > 0u) { emit(i, j, c); }
    }
  } } }
}

// ---------------------------------------------------------------- colouring
@compute @workgroup_size(64) fn color(@builtin(global_invocation_id) g: vec3u) {
  let mi = g.x; if (mi >= ld(C_MAN) || mi >= P.max_man) { return; }
  let m = P.cur_base + mi;
  let a = man[m].a; let b = man[m].b;
  var col = P.color_limit;
  for (var c = 0u; c < P.color_limit; c++) {
    let bit = 1u << c;
    if (a != 0u) { let o = atomicOr(&ints[P.o_mask + a], bit); if ((o & bit) != 0u) { continue; } }
    if (b != 0u) {
      let o = atomicOr(&ints[P.o_mask + b], bit);
      if ((o & bit) != 0u) { if (a != 0u) { atomicAnd(&ints[P.o_mask + a], ~bit); } continue; }
    }
    col = c; break;
  }
  st(P.o_mcolor + mi, col);
  atomicAdd(&ints[P.o_ccount + col], 1u);
  if (col == P.color_limit) { atomicAdd(&ints[C_OVF], 1u); } else { atomicMax(&ints[C_COLORS], col + 1u); }
}
@compute @workgroup_size(64) fn color_scatter(@builtin(global_invocation_id) g: vec3u) {
  let mi = g.x; if (mi >= ld(C_MAN) || mi >= P.max_man) { return; }
  let c = ld(P.o_mcolor + mi);
  st(P.o_order + ld(P.o_cstart + c) + atomicAdd(&ints[P.o_cfill + c], 1u), P.cur_base + mi);
}

// ---------------------------------------------------------------- solver
@compute @workgroup_size(64) fn integrate_vel(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + 1u; if (i >= P.n1) { return; }
  if (B(i, F_P).w > 0.0) { setB(i, F_V, vec4f(B(i, F_V).xyz + P.gravity * P.h, 0.0)); }
  let w = B(i, F_W).xyz; let wl = length(w);
  if (wl > P.wmax) { setB(i, F_W, vec4f(w * (P.wmax / wl), 0.0)); }
}
@compute @workgroup_size(64) fn integrate_pos(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + 1u; if (i >= P.n1) { return; }
  let v = B(i, F_V).xyz; let w = B(i, F_W).xyz;
  setB(i, F_DP, vec4f(B(i, F_DP).xyz + v * P.h, 0.0));
  setB(i, F_TH, vec4f(B(i, F_TH).xyz + w * P.h, 0.0));
  let q = B(i, F_Q);
  let dq = qmul(vec4f(w, 0.0), q);
  setB(i, F_Q, normalize(q + 0.5 * P.h * dq));
}
@compute @workgroup_size(64) fn finish(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + 1u; if (i >= P.n1) { return; }
  let p = B(i, F_P); setB(i, F_P, vec4f(p.xyz + B(i, F_DP).xyz, p.w));
}

fn warm_man(m: u32) {
  let a = man[m].a; let b = man[m].b;
  var va = B(a, F_V).xyz; var wa = B(a, F_W).xyz; var vb = B(b, F_V).xyz; var wb = B(b, F_W).xyz;
  var pn = 0.0;
  for (var k = 0u; k < man[m].cnt; k++) {
    let pt = man[m].p[k]; let jn = pt.iaxn.w;
    pn += jn; wa -= pt.iaxn.xyz * jn; wb += pt.ibxn.xyz * jn;
  }
  let M = man[m];
  let jt1 = M.ca2.w; let jt2 = M.cb1.w; let jtw = M.cb2.w;
  let p = M.n.xyz * pn + M.t1.xyz * jt1 + M.t2.xyz * jt2;
  va -= p * M.ia1.w; vb += p * M.ia2.w;
  wa -= M.ia1.xyz * jt1 + M.ia2.xyz * jt2 + M.ian.xyz * jtw;
  wb += M.ib1.xyz * jt1 + M.ib2.xyz * jt2 + M.ibn.xyz * jtw;
  if (a != 0u) { setB(a, F_V, vec4f(va, 0.0)); setB(a, F_W, vec4f(wa, 0.0)); }
  if (b != 0u) { setB(b, F_V, vec4f(vb, 0.0)); setB(b, F_W, vec4f(wb, 0.0)); }
}

fn solve_man(m: u32, use_bias: bool) {
  let a = man[m].a; let b = man[m].b;
  var va = B(a, F_V).xyz; var wa = B(a, F_W).xyz; var vb = B(b, F_V).xyz; var wb = B(b, F_W).xyz;
  let tha = B(a, F_TH).xyz; let thb = B(b, F_TH).xyz;
  let dpab = B(b, F_DP).xyz - B(a, F_DP).xyz;
  let n = man[m].n.xyz; let t1 = man[m].t1.xyz; let t2 = man[m].t2.xyz;
  let ima = man[m].ia1.w; let imb = man[m].ia2.w;
  let dn = dot(dpab, n);
  var tot = 0.0;
  let cnt = man[m].cnt;
  for (var k = 0u; k < cnt; k++) {
    let pt = man[m].p[k];
    let s = pt.raxn.w + dn + dot(thb, pt.rbxn.xyz) - dot(tha, pt.raxn.xyz);
    var bias = 0.0; var ms = 1.0; var is_ = 0.0;
    if (s > 0.0) { bias = s * P.inv_h; }
    else if (use_bias) { bias = max(P.bias_rate * s, -P.max_push); ms = P.mass_scale; is_ = P.imp_scale; }
    let vn = dot(vb - va, n) + dot(wb, pt.rbxn.xyz) - dot(wa, pt.raxn.xyz);
    let jn0 = pt.iaxn.w;
    let imp0 = -pt.rbxn.w * ms * (vn + bias) - is_ * jn0;
    let nj = max(jn0 + imp0, 0.0);
    let imp = nj - jn0;
    man[m].p[k].iaxn.w = nj; tot += nj;
    va -= n * (imp * ima); wa -= pt.iaxn.xyz * imp;
    vb += n * (imp * imb); wb += pt.ibxn.xyz * imp;
  }
  let M = man[m];
  let lim = M.n.w * tot;
  let dv = vb - va;
  let v1 = dot(dv, t1) + dot(wb, M.cb1.xyz) - dot(wa, M.ca1.xyz);
  let v2 = dot(dv, t2) + dot(wb, M.cb2.xyz) - dot(wa, M.ca2.xyz);
  var b1 = 0.0; var b2 = 0.0; var ms = 1.0; var is_ = 0.0;
  if (use_bias && M.stick != 0u) {
    let e1 = dot(M.e0.xyz, t1) + dot(dpab, t1) + dot(thb, M.cb1.xyz) - dot(tha, M.ca1.xyz);
    let e2 = dot(M.e0.xyz, t2) + dot(dpab, t2) + dot(thb, M.cb2.xyz) - dot(tha, M.ca2.xyz);
    b1 = clamp(P.bias_rate * e1, -P.max_push, P.max_push); b2 = clamp(P.bias_rate * e2, -P.max_push, P.max_push);
    ms = P.mass_scale; is_ = P.imp_scale;
  }
  let o1 = M.ca2.w; let o2 = M.cb1.w;
  var j1 = o1 - M.e0.w * ms * (v1 + b1) - is_ * o1;
  var j2 = o2 - M.ca1.w * ms * (v2 + b2) - is_ * o2;
  let l2 = j1 * j1 + j2 * j2;
  if (l2 > lim * lim) { let kk = lim * inverseSqrt(max(l2, 1e-30)); j1 *= kk; j2 *= kk; man[m].stick = 0u; }
  man[m].ca2.w = j1; man[m].cb1.w = j2;
  let d1 = j1 - o1; let d2 = j2 - o2;
  let p = t1 * d1 + t2 * d2;
  va -= p * ima; wa -= M.ia1.xyz * d1 + M.ia2.xyz * d2;
  vb += p * imb; wb += M.ib1.xyz * d1 + M.ib2.xyz * d2;
  let wl = lim * M.t1.w;
  let ot = M.cb2.w;
  let jt = clamp(ot - M.t2.w * dot(wb - wa, n), -wl, wl);
  man[m].cb2.w = jt;
  let dtw = jt - ot;
  wa -= M.ian.xyz * dtw; wb += M.ibn.xyz * dtw;
  if (a != 0u) { setB(a, F_V, vec4f(va, 0.0)); setB(a, F_W, vec4f(wa, 0.0)); }
  if (b != 0u) { setB(b, F_V, vec4f(vb, 0.0)); setB(b, F_W, vec4f(wb, 0.0)); }
}

fn color_item(g: u32) -> u32 {
  let c = CU.c;
  if (g >= ld(P.o_ccount + c)) { return 0xffffffffu; }
  return ld(P.o_order + ld(P.o_cstart + c) + g);
}
@compute @workgroup_size(64) fn warm(@builtin(global_invocation_id) g: vec3u) { let m = color_item(g.x); if (m != 0xffffffffu) { warm_man(m); } }
@compute @workgroup_size(64) fn solve(@builtin(global_invocation_id) g: vec3u) { let m = color_item(g.x); if (m != 0xffffffffu) { solve_man(m, true); } }
@compute @workgroup_size(64) fn relax(@builtin(global_invocation_id) g: vec3u) { let m = color_item(g.x); if (m != 0xffffffffu) { solve_man(m, false); } }
// overflow manifolds (no free colour): one thread, in sequence
fn ovf_range() -> vec2u { let c = P.color_limit; let s = ld(P.o_cstart + c); return vec2u(s, s + ld(P.o_ccount + c)); }
@compute @workgroup_size(1) fn warm_ovf() { let r = ovf_range(); for (var k = r.x; k < r.y; k++) { warm_man(ld(P.o_order + k)); } }
@compute @workgroup_size(1) fn solve_ovf() { let r = ovf_range(); for (var k = r.x; k < r.y; k++) { solve_man(ld(P.o_order + k), true); } }
@compute @workgroup_size(1) fn relax_ovf() { let r = ovf_range(); for (var k = r.x; k < r.y; k++) { solve_man(ld(P.o_order + k), false); } }
`;

// Writers of the indirect-dispatch arguments (bound in their own passes so the args buffer is never
// writable and indirect in the same dispatch).
const WGSL_ARGS = WGSL.replace(/@compute[\s\S]*$/, '') + /* wgsl */`
@group(2) @binding(0) var<storage, read_write> args: array<u32>;
@compute @workgroup_size(1) fn man_args() {
  let n = min(ld(C_MAN), P.max_man);
  args[96] = (n + 63u) / 64u; args[97] = 1u; args[98] = 1u;
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

class GpuWorld {
  // bodies: Float32Array of 10 per body (p.xyz, q.xyzw, h.xyz); density kg/m^3
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
    this.maxMan = Math.max(64, Math.ceil((opt.manPerBody || 4) * n));
    let G = 256; while (G < 2 * this.n1) G *= 2; this.G = G;
    let T = 256; while (T < 2 * this.maxMan) T *= 2; this.T = T;
    // grid cell: at least the largest AABB (with margin), so overlapping bodies sit in neighbour cells
    let maxExt = 0;
    for (let i = 0; i < n; i++) maxExt = Math.max(maxExt, 2 * Math.hypot(bodies[i * 10 + 7], bodies[i * 10 + 8], bodies[i * 10 + 9]));
    this.cell = Math.max(0.5, maxExt + 0.25);
    // ints layout
    let o = 16;
    const L = {};
    const take = (k, len) => { L[k] = o; o += len; };
    take('gcount', G); take('gstart', G); take('gfill', G);
    take('sorted', this.n1); take('cell', this.n1); take('mask', this.n1);
    take('ccount', 33); take('cstart', 33); take('cfill', 33);
    take('order', this.maxMan); take('mcolor', this.maxMan);
    take('t0', T); take('t1', T);
    this.L = L; this.nInts = o;

    const BS = 13;
    const bod = new Float32Array(this.n1 * BS * 4);
    for (let i = 0; i < n; i++) {
      const s = i * 10, b = (i + 1) * BS * 4;
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
    this.intBuf = d.createBuffer({ size: this.nInts * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    this.manBuf = d.createBuffer({ size: 2 * this.maxMan * 496, usage: U.STORAGE });
    this.coldBuf = d.createBuffer({ size: 2 * this.maxMan * 96, usage: U.STORAGE });
    this.parBuf = d.createBuffer({ size: 256, usage: U.UNIFORM | U.COPY_DST });
    this.colBuf = d.createBuffer({ size: 256 * 33, usage: U.UNIFORM | U.COPY_DST });
    const cu = new Uint32Array(64 * 33); for (let c = 0; c < 33; c++) cu[c * 64] = c;
    d.queue.writeBuffer(this.colBuf, 0, cu);
    this.argBuf = d.createBuffer({ size: 128 * 4, usage: U.STORAGE | U.INDIRECT });
    this.statBuf = d.createBuffer({ size: 64, usage: U.COPY_DST | U.MAP_READ });
    this.stats = { manifolds: 0, overflow: 0, colors: 0, pairs: 0, dropped: 0 };
    this.statPending = false;

    const bgl0 = d.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }] });
    const bgl1 = d.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 4 } }] });
    const bgl2 = d.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }] });
    const mainL = d.createPipelineLayout({ bindGroupLayouts: [bgl0, bgl1] });
    const argsL = d.createPipelineLayout({ bindGroupLayouts: [bgl0, bgl1, bgl2] });
    this.bg0 = d.createBindGroup({ layout: bgl0, entries: [
      { binding: 0, resource: { buffer: this.bodBuf } }, { binding: 1, resource: { buffer: this.intBuf } },
      { binding: 2, resource: { buffer: this.manBuf } }, { binding: 3, resource: { buffer: this.coldBuf } },
      { binding: 4, resource: { buffer: this.parBuf } }] });
    this.bg1 = d.createBindGroup({ layout: bgl1, entries: [{ binding: 0, resource: { buffer: this.colBuf, size: 4 } }] });
    this.bg2 = d.createBindGroup({ layout: bgl2, entries: [{ binding: 0, resource: { buffer: this.argBuf } }] });
    const mod = d.createShaderModule({ code: WGSL });
    const modA = d.createShaderModule({ code: WGSL_ARGS });
    const info = await mod.getCompilationInfo();
    const errs = info.messages.filter(m => m.type === 'error');
    if (errs.length) throw new Error('WGSL: ' + errs.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join('\n'));
    const infoA = await modA.getCompilationInfo();
    const errsA = infoA.messages.filter(m => m.type === 'error');
    if (errsA.length) throw new Error('WGSL args: ' + errsA.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join('\n'));
    const mk = (entry, layout = mainL, module = mod) => d.createComputePipelineAsync({ layout, compute: { module, entryPoint: entry } });
    const names = ['prep', 'scan', 'scatter', 'collide', 'color', 'color_scatter', 'integrate_vel', 'integrate_pos', 'finish',
                   'warm', 'solve', 'relax', 'warm_ovf', 'solve_ovf', 'relax_ovf'];
    this.pl = {};
    await Promise.all(names.map(async k => { this.pl[k] = await mk(k); }));
    this.pl.man_args = await mk('man_args', argsL, modA);
    this.pl.color_scan = await mk('color_scan', argsL, modA);
    this.frame = 0;
  }

  params(dt) {
    const sub = this.substeps, h = dt / sub, invH = 1 / h;
    const hz = invH / 6, omega = 2 * Math.PI * hz, zeta = 10;
    const a1 = 2 * zeta + h * omega, a2 = h * omega * a1, a3 = 1 / (1 + a2);
    const f = new Float32Array(64), u = new Uint32Array(f.buffer);
    const cur = this.frame & 1;
    f.set([0, -9.81, 0, h, invH, omega / a1, a2 * a3, a3, 3.0, this.friction, 0.02, dt, this.cell]);
    u[13] = this.n1; u[14] = this.G - 1; u[15] = this.maxMan;
    u[16] = this.T - 1; u[17] = this.colorLimit; f[18] = 0.25 * Math.PI * invH; u[19] = cur * this.maxMan;
    const L = this.L;
    u[20] = (1 - cur) * this.maxMan; u[21] = L.gcount; u[22] = L.gstart; u[23] = L.gfill;
    u[24] = L.sorted; u[25] = L.cell; u[26] = L.mask; u[27] = L.ccount;
    u[28] = L.cstart; u[29] = L.cfill; u[30] = L.order; u[31] = L.mcolor;
    u[32] = cur ? L.t1 : L.t0; u[33] = cur ? L.t0 : L.t1; u[34] = this.anchor ? 1 : 0;
    return f;
  }

  // Encode one 60 Hz step into `enc` (several compute passes).
  encode(enc, dt = 1 / 60) {
    const d = this.device;
    d.queue.writeBuffer(this.parBuf, 0, this.params(dt));
    const L = this.L, cur = this.frame & 1;
    const clr = (off, len) => enc.clearBuffer(this.intBuf, off * 4, len * 4);
    clr(0, 16); clr(L.gcount, this.G); clr(L.gfill, this.G); clr(L.ccount, 33); clr(L.cfill, 33); clr(cur ? L.t1 : L.t0, this.T);
    const nb = Math.ceil(this.n / 64) || 1;
    const run = (pass, k, x) => { pass.setPipeline(this.pl[k]); pass.dispatchWorkgroups(x); };
    let p = enc.beginComputePass();
    p.setBindGroup(0, this.bg0); p.setBindGroup(1, this.bg1, [0]);
    run(p, 'prep', nb); run(p, 'scan', 1); run(p, 'scatter', nb); run(p, 'collide', nb);
    p.end();
    p = enc.beginComputePass(); p.setBindGroup(0, this.bg0); p.setBindGroup(1, this.bg1, [0]); p.setBindGroup(2, this.bg2);
    run(p, 'man_args', 1); p.end();
    p = enc.beginComputePass(); p.setBindGroup(0, this.bg0); p.setBindGroup(1, this.bg1, [0]);
    p.setPipeline(this.pl.color); p.dispatchWorkgroupsIndirect(this.argBuf, 96 * 4); p.end();
    p = enc.beginComputePass(); p.setBindGroup(0, this.bg0); p.setBindGroup(1, this.bg1, [0]); p.setBindGroup(2, this.bg2);
    run(p, 'color_scan', 1); p.end();
    p = enc.beginComputePass(); p.setBindGroup(0, this.bg0); p.setBindGroup(1, this.bg1, [0]);
    p.setPipeline(this.pl.color_scatter); p.dispatchWorkgroupsIndirect(this.argBuf, 96 * 4);
    const C = this.colorLimit;
    const colors = (k) => {
      p.setPipeline(this.pl[k]);
      for (let c = 0; c < C; c++) { p.setBindGroup(1, this.bg1, [c * 256]); p.dispatchWorkgroupsIndirect(this.argBuf, c * 12); }
      p.setBindGroup(1, this.bg1, [0]);
      run(p, k + '_ovf', 1);
    };
    for (let s = 0; s < this.substeps; s++) {
      run(p, 'integrate_vel', nb);
      colors('warm'); colors('solve');
      run(p, 'integrate_pos', nb);
      colors('relax');
    }
    run(p, 'finish', nb);
    p.end();
    if (!this.statPending) { enc.copyBufferToBuffer(this.intBuf, 0, this.statBuf, 0, 64); this.statCopied = true; } else this.statCopied = false;
    this.frame++;
  }

  step(dt = 1 / 60) {
    const enc = this.device.createCommandEncoder();
    this.encode(enc, dt);
    this.device.queue.submit([enc.finish()]);
    this.pollStats();
  }

  pollStats() {
    if (!this.statCopied || this.statPending) return;
    this.statPending = true;
    this.statBuf.mapAsync(GPUMapMode.READ).then(() => {
      const u = new Uint32Array(this.statBuf.getMappedRange().slice(0));
      this.statBuf.unmap();
      this.stats = { manifolds: u[0], overflow: u[1], colors: u[2], pairs: u[3], dropped: u[4] };
      // adapt the number of colour dispatches to what the scene needs (with headroom)
      this.colorLimit = Math.min(31, Math.max(8, u[2] + 3));
      this.statPending = false;
    }, () => { this.statPending = false; });
  }

  // Read back bodies as 10 floats each (p, q, h) like the wasm pose() export.
  async readPose() {
    const size = this.n1 * 13 * 16;
    const rb = this.device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder(); enc.copyBufferToBuffer(this.bodBuf, 0, rb, 0, size);
    this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const f = new Float32Array(rb.getMappedRange());
    const out = new Float32Array(this.n * 10), vel = new Float32Array(this.n * 3);
    for (let i = 0; i < this.n; i++) {
      const b = (i + 1) * 52;
      out.set(f.subarray(b, b + 3), i * 10); out.set(f.subarray(b + 4, b + 8), i * 10 + 3); out.set(f.subarray(b + 8, b + 11), i * 10 + 7);
      vel.set(f.subarray(b + 16, b + 19), i * 3);
    }
    rb.unmap(); rb.destroy();
    return { pose: out, vel };
  }

  destroy() { for (const b of [this.bodBuf, this.intBuf, this.manBuf, this.coldBuf, this.parBuf, this.colBuf, this.argBuf, this.statBuf]) b.destroy(); }
}
if (typeof window !== 'undefined') window.GpuWorld = GpuWorld;
