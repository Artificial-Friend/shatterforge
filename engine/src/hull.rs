//! The library of fracture patterns: Voronoi cells of 2-6 seeds in the cube [-1,1]^3, each cell convex. The same
//! library as the GPU's (`hullLibrary` in engine/gpu/gpu_world.js: the same generator, seed and limits), so a cell's hull
//! id means the same shape in the engine, on the page and on the GPU. Hull 0 is the whole cube.
//!
//! Everything here is in the cube's coordinates; a cell of a brick is the cell mapped into the brick's box
//! (x = centre + half * x_cube), which keeps planes planar and convex cells convex.
use crate::math::sqrt64;
use alloc::vec;
use alloc::vec::Vec;

/// A face of a cell: its plane (outward normal, n.x = d), its polygon (indices into the cell's vertices, in
/// order), and what is on the other side: a cube face (0..6: +x, -x, +y, -y, +z, -z) or another cell of the
/// pattern (6 + its index in the pattern).
#[derive(Clone, Debug)]
pub struct Face { pub n: [f64; 3], pub d: f64, pub v: Vec<u8>, pub other: u8 }
/// A cell: vertices, faces, its bounding box (centre, half size), its share of the cube's volume and its centroid.
#[derive(Clone, Debug)]
pub struct Cell { pub v: Vec<[f64; 3]>, pub faces: Vec<Face>, pub bc: [f64; 3], pub bh: [f64; 3], pub vol: f64, pub cen: [f64; 3], pub pattern: u16, pub slot: u8 }
pub struct Lib { pub cells: Vec<Cell>, pub patterns: Vec<Vec<u16>> }

// the GPU's limits on a hull (gpu_world.js HULL): patterns beyond them are skipped there, so here too
const MAXV: usize = 16; const MAXF: usize = 12; const MAXFV: usize = 8; const MAXS: usize = 24; const MAXD: usize = 12;

fn sub(p: [f64; 3], q: [f64; 3]) -> [f64; 3] { [p[0] - q[0], p[1] - q[1], p[2] - q[2]] }
fn dot(p: [f64; 3], q: [f64; 3]) -> f64 { p[0] * q[0] + p[1] * q[1] + p[2] * q[2] }
fn cross(p: [f64; 3], q: [f64; 3]) -> [f64; 3] { [p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0]] }
fn len(p: [f64; 3]) -> f64 { sqrt64(dot(p, p)) }
fn nrm(p: [f64; 3]) -> [f64; 3] { let l = len(p); [p[0] / l, p[1] / l, p[2] / l] }
/// a monotonic stand-in for atan2(y, x) (only the order of the angles is used): 0..4 round the circle
fn pseudo_angle(y: f64, x: f64) -> f64 {
    let p = x / (x.abs() + y.abs());
    if y < 0.0 { 3.0 + p } else { 1.0 - p }
}

struct Poly { v: Vec<[f64; 3]>, faces: Vec<(usize, Vec<usize>)>, nseg: usize, ndir: usize }
/// The polytope of half-spaces n.x <= d: vertices (from triples of planes), faces (vertex loops), and the counts
/// of edges and of distinct edge directions (for the GPU's limits).
fn polytope(planes: &[([f64; 3], f64)]) -> Poly {
    let mut v: Vec<[f64; 3]> = Vec::new();
    let np = planes.len();
    for i in 0..np { for j in i + 1..np { for k in j + 1..np {
        let (a, b, c) = (planes[i], planes[j], planes[k]);
        let bc = cross(b.0, c.0);
        let det = dot(a.0, bc); if det.abs() < 1e-9 { continue; }
        let (ca, ab) = (cross(c.0, a.0), cross(a.0, b.0));
        let x = [(a.1 * bc[0] + b.1 * ca[0] + c.1 * ab[0]) / det, (a.1 * bc[1] + b.1 * ca[1] + c.1 * ab[1]) / det, (a.1 * bc[2] + b.1 * ca[2] + c.1 * ab[2]) / det];
        if planes.iter().all(|p| dot(p.0, x) <= p.1 + 1e-7) && !v.iter().any(|w| len(sub(*w, x)) < 1e-6) { v.push(x); }
    } } }
    let mut faces = Vec::new();
    for (pi, p) in planes.iter().enumerate() {
        let mut idx: Vec<usize> = (0..v.len()).filter(|&i| (dot(p.0, v[i]) - p.1).abs() < 1e-6).collect();
        if idx.len() < 3 { continue; }
        let mut c = [0.0; 3];
        for &i in &idx { for q in 0..3 { c[q] += v[i][q]; } }
        for q in 0..3 { c[q] /= idx.len() as f64; }
        let u = nrm(sub(v[idx[0]], c)); let w = cross(p.0, u);
        let ang = |i: usize| { let r = sub(v[i], c); pseudo_angle(dot(r, w), dot(r, u)) };
        idx.sort_by(|&i, &j| ang(i).partial_cmp(&ang(j)).unwrap_or(core::cmp::Ordering::Equal));
        faces.push((pi, idx));
    }
    let mut seg: Vec<(usize, usize)> = Vec::new();
    for (_, f) in &faces { for k in 0..f.len() { let (i, j) = (f[k], f[(k + 1) % f.len()]); if !seg.iter().any(|&(x, y)| (x == i && y == j) || (x == j && y == i)) { seg.push((i, j)); } } }
    let mut dirs: Vec<[f64; 3]> = Vec::new();
    for &(i, j) in &seg { let d = nrm(sub(v[j], v[i])); if !dirs.iter().any(|e| dot(*e, d).abs() > 0.9995) { dirs.push(d); } }
    Poly { v, faces, nseg: seg.len(), ndir: dirs.len() }
}

/// volume and centroid of a convex polytope (tetrahedra from the vertices' mean to each face triangle)
pub fn mass_props(v: &[[f64; 3]], faces: &[Vec<usize>]) -> (f64, [f64; 3]) {
    let mut o = [0.0; 3];
    for p in v { for q in 0..3 { o[q] += p[q]; } }
    for q in 0..3 { o[q] /= v.len() as f64; }
    let (mut vol, mut m1) = (0.0, [0.0; 3]);
    for f in faces { for t in 1..f.len().saturating_sub(1) {
        let (a, b, c) = (v[f[0]], v[f[t]], v[f[t + 1]]);
        let tv = dot(sub(a, o), cross(sub(b, o), sub(c, o))).abs() / 6.0;
        vol += tv;
        for q in 0..3 { m1[q] += tv * (o[q] + a[q] + b[q] + c[q]) / 4.0; }
    } }
    (vol, [m1[0] / vol, m1[1] / vol, m1[2] / vol])
}

/// Volume, centroid and inertia tensor about the centroid (per unit density) of a convex polytope: tetrahedra from
/// the vertices' mean to each face triangle, with the second moments of a tetrahedron with a corner at the origin,
/// integral of x_i x_j = V / 20 (sum over its corners of x_i x_j + (sum x_i)(sum x_j)).
pub fn moments(v: &[[f64; 3]], faces: &[Vec<usize>]) -> (f64, [f64; 3], [[f64; 3]; 3]) {
    let mut o = [0.0; 3];
    for p in v { for q in 0..3 { o[q] += p[q]; } }
    for q in 0..3 { o[q] /= v.len() as f64; }
    let (mut vol, mut m1, mut m2) = (0.0, [0.0; 3], [[0.0; 3]; 3]);
    for f in faces { for t in 1..f.len().saturating_sub(1) {
        let (a, b, c) = (sub(v[f[0]], o), sub(v[f[t]], o), sub(v[f[t + 1]], o));
        let tv = dot(a, cross(b, c)).abs() / 6.0;
        vol += tv;
        let s = [a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2]];
        for i in 0..3 {
            m1[i] += tv * s[i] / 4.0;
            for j in 0..3 { m2[i][j] += tv / 20.0 * (a[i] * a[j] + b[i] * b[j] + c[i] * c[j] + s[i] * s[j]); }
        }
    } }
    let d = [m1[0] / vol, m1[1] / vol, m1[2] / vol];
    // second moments about the centroid, then I = tr(M) 1 - M
    let mut m = [[0.0; 3]; 3];
    for i in 0..3 { for j in 0..3 { m[i][j] = m2[i][j] - vol * d[i] * d[j]; } }
    let tr = m[0][0] + m[1][1] + m[2][2];
    let mut ii = [[0.0; 3]; 3];
    for i in 0..3 { for j in 0..3 { ii[i][j] = if i == j { tr - m[i][j] } else { -m[i][j] }; } }
    (vol, [o[0] + d[0], o[1] + d[1], o[2] + d[2]], ii)
}

/// A convex shape: vertices, and faces as (outward unit normal, offset n.x = d, vertex loop counter-clockwise seen
/// from outside).
#[derive(Clone, Debug)]
pub struct Shape { pub v: Vec<[f64; 3]>, pub faces: Vec<([f64; 3], f64, Vec<usize>)> }
/// The convex hull of points (m), within the GPU's limits on a hull (16 corners, 12 faces of up to 8 corners, 24
/// edges in up to 12 directions); none if the points are flat (no volume) or the hull is beyond the limits.
/// Faces come straight from the points: a plane through three of them with every point on one side (within 1 um
/// per m of size) is a face, its corners the points on it in order round it (points along an edge dropped). The
/// corners are the points given, not planes' intersections: corners kept in f32 are coplanar only to ~1e-7 m, and
/// a face missed at a tighter tolerance left the hull a face short (a piece cut four ways came out 9% too big).
pub fn convex(pts_in: &[[f64; 3]]) -> Option<Shape> {
    if pts_in.len() < 4 { return None; }
    let mut lo = [f64::MAX; 3]; let mut hi = [f64::MIN; 3];
    for p in pts_in { for q in 0..3 { lo[q] = lo[q].min(p[q]); hi[q] = hi[q].max(p[q]); } }
    let size = (0..3).map(|q| hi[q] - lo[q]).fold(0.0, f64::max);
    if size <= 0.0 { return None; }
    let eps = 1e-6 * size.max(1.0);
    // the points, without repeats
    let mut pts: Vec<[f64; 3]> = Vec::new();
    for p in pts_in { if !pts.iter().any(|q| len(sub(*q, *p)) < eps) { pts.push(*p); } }
    let n = pts.len();
    if n < 4 { return None; }
    let mut faces: Vec<([f64; 3], f64, Vec<usize>)> = Vec::new();
    let mut seen: Vec<Vec<usize>> = Vec::new();
    for i in 0..n { for j in i + 1..n { for k in j + 1..n {
        let c = cross(sub(pts[j], pts[i]), sub(pts[k], pts[i]));
        if len(c) < 1e-6 * size * size { continue; }
        let nn = nrm(c); let d = dot(nn, pts[i]);
        let (mut above, mut below) = (false, false);
        let mut on: Vec<usize> = Vec::new();
        for (m, p) in pts.iter().enumerate() { let e = dot(nn, *p) - d; if e > eps { above = true; } else if e < -eps { below = true; } else { on.push(m); } }
        let out = match (above, below) { (false, true) => nn, (true, false) => [-nn[0], -nn[1], -nn[2]], _ => continue };
        if seen.contains(&on) { continue; }
        seen.push(on.clone());
        // the face's corners in order round it, counter-clockwise seen from outside, points along an edge dropped
        let u = nrm(sub(pts[on[1]], pts[on[0]]));
        let w = cross(out, u);
        let p2: Vec<(f64, f64)> = on.iter().map(|&m| (dot(pts[m], u), dot(pts[m], w))).collect();
        let mut ord: Vec<usize> = (0..on.len()).collect();
        ord.sort_by(|&a, &b| p2[a].partial_cmp(&p2[b]).unwrap_or(core::cmp::Ordering::Equal));
        let turn = |o: (f64, f64), a: (f64, f64), b: (f64, f64)| (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0);
        let tol = eps * size;
        let mut hull: Vec<usize> = Vec::new();
        for pass in 0..2 {
            let start = hull.len();
            let seq: Vec<usize> = if pass == 0 { ord.clone() } else { ord.iter().rev().cloned().collect() };
            for &m in &seq {
                while hull.len() >= start + 2 && turn(p2[hull[hull.len() - 2]], p2[hull[hull.len() - 1]], p2[m]) <= tol { hull.pop(); }
                hull.push(m);
            }
            hull.pop();
        }
        if hull.len() < 3 { continue; }
        let lp: Vec<usize> = hull.iter().map(|&m| on[m]).collect();
        let dd = lp.iter().map(|&m| dot(out, pts[m])).sum::<f64>() / lp.len() as f64;
        faces.push((out, dd, lp));
    } } }
    if faces.len() < 4 { return None; }
    // only the corners the faces use, renumbered
    let mut used: Vec<usize> = faces.iter().flat_map(|f| f.2.iter().cloned()).collect();
    used.sort_unstable(); used.dedup();
    let v: Vec<[f64; 3]> = used.iter().map(|&m| pts[m]).collect();
    let faces: Vec<([f64; 3], f64, Vec<usize>)> = faces.into_iter().map(|(nn, d, lp)| (nn, d, lp.iter().map(|m| used.binary_search(m).unwrap()).collect())).collect();
    // the GPU's limits: corners, faces, corners a face, edges and their directions
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for f in &faces { for k in 0..f.2.len() { let (a, b) = (f.2[k], f.2[(k + 1) % f.2.len()]); let e = (a.min(b), a.max(b)); if !edges.contains(&e) { edges.push(e); } } }
    let mut dirs: Vec<[f64; 3]> = Vec::new();
    for &(a, b) in &edges { let d = nrm(sub(v[b], v[a])); if !dirs.iter().any(|q| dot(*q, d).abs() > 1.0 - 1e-6) { dirs.push(d); } }
    if v.len() > MAXV || faces.len() > MAXF || faces.iter().any(|f| f.2.len() > MAXFV) || edges.len() > MAXS || dirs.len() > MAXD { return None; }
    Some(Shape { v, faces })
}

fn build(n_patterns: usize, seed: u32) -> Lib {
    let mut a = seed;
    let mut rnd = || -> f64 {
        a = a.wrapping_add(0x6D2B79F5);
        let mut t = a;
        t = (t ^ (t >> 15)).wrapping_mul(t | 1);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
        (t ^ (t >> 14)) as f64 / 4294967296.0
    };
    let cube: Vec<([f64; 3], f64)> = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]].iter().map(|&n| (n, 1.0)).collect();
    let whole = {
        let p = polytope(&cube);
        let faces = p.faces.iter().map(|(pi, f)| Face { n: cube[*pi].0, d: 1.0, v: f.iter().map(|&i| i as u8).collect(), other: *pi as u8 }).collect();
        Cell { v: p.v, faces, bc: [0.0; 3], bh: [1.0; 3], vol: 1.0, cen: [0.0; 3], pattern: u16::MAX, slot: 0 }
    };
    let mut cells = vec![whole];
    let mut patterns = Vec::new();
    let (mut p, mut tries) = (0usize, 0usize);
    while p < n_patterns && tries < 40000 {
        tries += 1;
        let k = 2 + (p % 5);
        let seeds: Vec<[f64; 3]> = (0..k).map(|_| { let x = rnd() * 1.6 - 0.8; let y = rnd() * 1.6 - 0.8; let z = rnd() * 1.6 - 0.8; [x, y, z] }).collect();
        let mut out = Vec::new();
        let mut ok = true;
        for i in 0..k {
            let mut planes = cube.clone();
            let mut other: Vec<u8> = (0..6).collect();
            for j in 0..k { if j != i {
                let n = nrm(sub(seeds[j], seeds[i]));
                let m = [(seeds[i][0] + seeds[j][0]) / 2.0, (seeds[i][1] + seeds[j][1]) / 2.0, (seeds[i][2] + seeds[j][2]) / 2.0];
                planes.push((n, dot(n, m))); other.push(6 + j as u8);
            } }
            let pp = polytope(&planes);
            let mut lo = [f64::MAX; 3]; let mut hi = [f64::MIN; 3];
            for v in &pp.v { for q in 0..3 { lo[q] = lo[q].min(v[q]); hi[q] = hi[q].max(v[q]); } }
            let h = [(hi[0] - lo[0]) / 2.0, (hi[1] - lo[1]) / 2.0, (hi[2] - lo[2]) / 2.0];
            let c = [(hi[0] + lo[0]) / 2.0, (hi[1] + lo[1]) / 2.0, (hi[2] + lo[2]) / 2.0];
            if pp.v.len() > MAXV || pp.faces.len() > MAXF || pp.nseg > MAXS || pp.ndir > MAXD || pp.faces.iter().any(|f| f.1.len() > MAXFV) || h.iter().any(|&x| x < 0.15) { ok = false; break; }
            let fl: Vec<Vec<usize>> = pp.faces.iter().map(|f| f.1.clone()).collect();
            let (vol, cen) = mass_props(&pp.v, &fl);
            let faces = pp.faces.iter().map(|(pi, f)| Face { n: planes[*pi].0, d: planes[*pi].1, v: f.iter().map(|&i| i as u8).collect(), other: other[*pi] }).collect();
            out.push(Cell { v: pp.v, faces, bc: c, bh: h, vol: vol / 8.0, cen, pattern: patterns.len() as u16, slot: i as u8 });
        }
        if !ok { continue; }
        let ids: Vec<u16> = out.into_iter().map(|c| { cells.push(c); (cells.len() - 1) as u16 }).collect();
        patterns.push(ids);
        p += 1;
    }
    Lib { cells, patterns }
}

// built once, on first use (wasm is single-threaded; native tests may race to it, so a spin flag)
static mut LIB: Option<Lib> = None;
static STATE: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(0);
/// The library: 16 patterns, the GPU's seed 7.
#[allow(static_mut_refs)]
pub fn lib() -> &'static Lib {
    use core::sync::atomic::Ordering::*;
    loop {
        match STATE.compare_exchange(0, 1, Acquire, Acquire) {
            Ok(_) => { unsafe { LIB = Some(build(16, 7)); } STATE.store(2, Release); }
            Err(2) => return unsafe { LIB.as_ref().unwrap() },
            Err(_) => core::hint::spin_loop(),
        }
    }
}

/// A convex polygon's area, centroid and unit normal (vertices in order, counter-clockwise seen from the normal).
pub fn polygon(p: &[[f64; 3]]) -> (f64, [f64; 3], [f64; 3]) {
    let mut s = [0.0; 3]; let mut c = [0.0; 3]; let mut a2 = 0.0;
    for t in 1..p.len().saturating_sub(1) {
        let x = cross(sub(p[t], p[0]), sub(p[t + 1], p[0]));
        let ar = len(x);
        for q in 0..3 { s[q] += x[q]; c[q] += ar * (p[0][q] + p[t][q] + p[t + 1][q]) / 3.0; }
        a2 += ar;
    }
    if a2 <= 0.0 { return (0.0, p.first().copied().unwrap_or([0.0; 3]), [0.0, 0.0, 1.0]); }
    (a2 / 2.0, [c[0] / a2, c[1] / a2, c[2] / a2], nrm(s))
}
/// Second moments of a planar polygon about its centroid along two in-plane unit axes u, w: (I_uu, I_ww) =
/// (integral of (r.w)^2, integral of (r.u)^2) (bending about u, about w), and the largest |r.w|, |r.u| (extreme
/// fibres).
pub fn polygon_section(p: &[[f64; 3]], c: [f64; 3], u: [f64; 3], w: [f64; 3]) -> ([f64; 2], [f64; 2]) {
    // triangles from the centroid: for a triangle (0, a, b) in 2D the integral of y^2 is A (ya^2 + ya yb + yb^2) / 6
    let (mut iu, mut iw) = (0.0, 0.0);
    let (mut cu, mut cw) = (0.0f64, 0.0f64);
    let n = p.len();
    for t in 0..n {
        let (ra, rb) = (sub(p[t], c), sub(p[(t + 1) % n], c));
        let (au, aw, bu, bw) = (dot(ra, u), dot(ra, w), dot(rb, u), dot(rb, w));
        let ar = (au * bw - aw * bu).abs() / 2.0;
        iu += ar * (aw * aw + aw * bw + bw * bw) / 6.0;
        iw += ar * (au * au + au * bu + bu * bu) / 6.0;
        cu = cu.max(aw.abs()); cw = cw.max(au.abs());
    }
    ([iu, iw], [cu, cw])
}
/// Clip convex polygon `p` (in a plane) to the half-plane of points x with dot(x, n) <= d.
pub fn clip(p: &[[f64; 3]], n: [f64; 3], d: f64) -> Vec<[f64; 3]> {
    let mut out = Vec::new();
    let m = p.len();
    for i in 0..m {
        let (a, b) = (p[i], p[(i + 1) % m]);
        let (da, db) = (dot(a, n) - d, dot(b, n) - d);
        if da <= 0.0 { out.push(a); }
        if (da < 0.0 && db > 0.0) || (da > 0.0 && db < 0.0) {
            let t = da / (da - db);
            out.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])]);
        }
    }
    out
}
/// The overlap of two convex polygons in one plane (normal `n`; either may wind either way about it), in the
/// order of `a`.
pub fn intersect(a: &[[f64; 3]], b: &[[f64; 3]], n: [f64; 3]) -> Vec<[f64; 3]> {
    if a.len() < 3 || b.len() < 3 { return Vec::new(); }
    let sb = if dot(polygon(b).2, n) >= 0.0 { 1.0 } else { -1.0 };
    let mut q = a.to_vec();
    let m = b.len();
    for k in 0..m {
        let (p0, p1) = (b[k], b[(k + 1) % m]);
        // inside: sb ((n x e) . (x - p0)) >= 0, as clip's dot(x, m) <= d
        let e = sub(p1, p0); let ne = cross(n, e);
        let mm = [-sb * ne[0], -sb * ne[1], -sb * ne[2]];
        q = clip(&q, mm, dot(mm, p0));
        if q.len() < 3 { return Vec::new(); }
    }
    q
}
/// Clip convex polygon `p` to an axis-aligned rectangle [lo, hi] in the plane normal to `axis`.
pub fn clip_rect(p: &[[f64; 3]], axis: usize, lo: [f64; 3], hi: [f64; 3]) -> Vec<[f64; 3]> {
    let mut q = p.to_vec();
    for d in 0..3 { if d == axis { continue; }
        let mut e = [0.0; 3]; e[d] = 1.0; q = clip(&q, e, hi[d]);
        let mut e = [0.0; 3]; e[d] = -1.0; q = clip(&q, e, -lo[d]);
        if q.len() < 3 { return Vec::new(); }
    }
    q
}

impl Cell {
    /// the face's polygon mapped into a box (centre, half size), in order
    pub fn face_in(&self, f: &Face, c: [f64; 3], h: [f64; 3]) -> Vec<[f64; 3]> {
        f.v.iter().map(|&i| { let p = self.v[i as usize]; [c[0] + h[0] * p[0], c[1] + h[1] * p[1], c[2] + h[2] * p[2]] }).collect()
    }
    /// the cell's polygon on cube face `side` (0..6), if it has one
    pub fn side(&self, side: u8) -> Option<&Face> { self.faces.iter().find(|f| f.other == side) }
    /// the face it shares with cell `slot` of its pattern
    pub fn shared(&self, slot: u8) -> Option<&Face> { self.faces.iter().find(|f| f.other == 6 + slot) }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    #[test] fn the_patterns_tile_the_cube() {
        let l = lib();
        assert_eq!(l.patterns.len(), 16);
        std::println!("hull library: {} cells in {} patterns (cells per pattern {:?})", l.cells.len(), l.patterns.len(), l.patterns.iter().map(|p| p.len()).collect::<std::vec::Vec<_>>());
        for (pi, p) in l.patterns.iter().enumerate() {
            // volumes add up to the cube; the cells' faces on each cube face add up to that face (area 4)
            let v: f64 = p.iter().map(|&c| l.cells[c as usize].vol).sum();
            assert!((v - 1.0).abs() < 1e-9, "pattern {pi}: volume {v}");
            for side in 0..6u8 {
                let a: f64 = p.iter().filter_map(|&c| { let cl = &l.cells[c as usize]; cl.side(side).map(|f| polygon(&cl.face_in(f, [0.0; 3], [1.0; 3])).0) }).sum();
                assert!((a - 4.0).abs() < 1e-9, "pattern {pi}, side {side}: {a}");
            }
            // a shared face is the same polygon seen from both cells, normals opposite
            for (s, &c) in p.iter().enumerate() { for (t, &e) in p.iter().enumerate() { if s < t {
                let (x, y) = (&l.cells[c as usize], &l.cells[e as usize]);
                match (x.shared(t as u8), y.shared(s as u8)) {
                    (Some(f), Some(g)) => {
                        let (a, ca, na) = polygon(&x.face_in(f, [0.0; 3], [1.0; 3]));
                        let (b, cb, nb) = polygon(&y.face_in(g, [0.0; 3], [1.0; 3]));
                        assert!((a - b).abs() < 1e-9 && len(sub(ca, cb)) < 1e-9 && (dot(na, nb) + 1.0).abs() < 1e-9 && dot(na, f.n) > 0.999999, "pattern {pi}: face {s}-{t}");
                    }
                    (None, None) => {}
                    _ => panic!("pattern {pi}: face {s}-{t} on one side only"),
                }
            } } }
        }
    }
}
