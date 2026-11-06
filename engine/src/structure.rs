//! Load paths through the joint graph and joint failure (first cut of the structural solver).
//!
//! Every brick carries a load (its weight, or its inertia in an impact). The load flows through joints to the
//! supports like current through a resistor network: a potential per brick, conductance = joint area (infill
//! such as glass and stud partitions conducts very little, so loads go round it), supports at potential 0.
//! The flow through a joint is a force. Its direction against the load direction says how the joint works:
//! along the load = compression, against it = tension, across it = shear. A joint fails when the stress
//! (force / area) exceeds the weaker brick's strength in that mode.
//!
//! Not modelled yet: bending moments (a scalar flow has none), yielding and damage accumulation.
use crate::brick::{mat, Bricks, F_INFILL, NONE, TICK};
use crate::math::*;
use alloc::vec;
use alloc::vec::Vec;

/// Strength in Pa: (compression, tension, shear). Reinforced concrete joints get rebar continuity in tension.
pub fn strength(m: u8) -> [f32; 3] {
    match m {
        mat::CONCRETE => [30e6, 4e6, 4e6],
        mat::MASONRY => [10e6, 0.3e6, 0.5e6],
        mat::STEEL => [250e6, 250e6, 150e6],
        mat::GLASS => [50e6, 7e6, 7e6],
        // structural timber (C24 / GL24h: compression 21, tension 14, shear 4 MPa along the grain). It was 5, 5
        // and 1, a stud partition's: with bending checked at the compressed fibre too (the spring network), a
        // 5 m timber cantilever came out at 4 times its strength
        _ => [21e6, 14e6, 4e6],
    }
}
/// Energy (J/m^3) to crush the material to rubble.
/// How strong a joint is relative to the weaker of its two bricks, in [compression, tension, shear]: a joint
/// is a connection between two elements, and connections are usually weaker than what they join.
/// - a seam inside one element (two bricks of one slab, one column) is the material itself
/// - concrete slab to column or core: a cast connection, sound in compression; in shear it is the slab's
///   punching strength (about 0.33 sqrt(f_c) = 1.8 MPa against 4 MPa) and in tension what the connection
///   steel carries
/// - timber elements joined to each other: nailed or bolted, a fraction of the wood
/// - anything else between different elements: somewhat weaker than the material
pub fn connection(b: &Bricks, j: u32) -> [f32; 3] {
    let (a, c) = (b.ja[j as usize] as usize, b.jb[j as usize] as usize);
    let (ma, mc) = (b.mat[a], b.mat[c]);
    let (pa, pc) = (b.paint[a], b.paint[c]);
    if pa == pc && ma == mc { return [1.0, 1.0, 1.0]; }
    if ma == mat::TIMBER && mc == mat::TIMBER { return [1.0, 0.1, 0.25]; }
    if ma == mat::CONCRETE && mc == mat::CONCRETE {
        let slab = |p: u8| p == 0;
        let support = |p: u8| p == 1 || p == 3;
        if (slab(pa) && support(pc)) || (slab(pc) && support(pa)) { return [1.0, 0.5, 0.45]; }
    }
    [1.0, 0.7, 0.7]
}

pub fn crush_energy(m: u8) -> f32 {
    match m { mat::CONCRETE => 1.0e6, mat::MASONRY => 0.4e6, mat::STEEL => 20e6, mat::GLASS => 0.05e6, _ => 0.2e6 }
}
pub fn mass(b: &Bricks, i: u32) -> f32 {
    let s = b.size(i);
    (s[0] * s[1] * s[2]) as f32 * TICK * TICK * TICK * mat::density(b.mat[i as usize]) * b.vol_share(i) as f32
}

/// Flexural strength (Pa) for bending of a brick: reinforced concrete gets its rebar.
/// Bending strength of brick `i` (Pa over its own depth). A timber floor (paint 8) is boards on joists
/// (50 x 225 mm at 400 mm, 24 MPa), not solid wood: per metre of width it carries 2.5 x 0.05 x 0.225^2 / 6 x
/// 24 MPa = 25 kNm, which over the plate's 0.25 m depth is 2.4 MPa. As solid timber (20 MPa) a floor bay
/// carried 300 kPa of rubble.
pub fn flexural_of(b: &Bricks, i: usize) -> f32 { if b.mat[i] == mat::TIMBER && b.paint[i] == 8 { 2.4e6 } else { flexural(b.mat[i]) } }
/// A joint at which a timber floor meets another material (the core's concrete, a brick wall): the joists bear on a
/// ledger or in pockets, a hinge. It carries shear but no bending (checked as one plate, it failed under the
/// floor's own weight).
/// Also steel bearing on brick (a beam's end or a lintel on a pier): it sits on a padstone, a hinge (bonded, its
/// end rotation bent the brick pier under it: 1.31 of strength in the mixed building).
fn hinge(b: &Bricks, x: usize, y: usize) -> bool {
    let tf = |i: usize| b.mat[i] == mat::TIMBER && b.paint[i] == 8;
    let (mx, my) = (b.mat[x], b.mat[y]);
    ((tf(x) || tf(y)) && mx != my) || (mx == mat::STEEL && my == mat::MASONRY) || (mx == mat::MASONRY && my == mat::STEEL)
}
pub fn flexural(m: u8) -> f32 {
    match m { mat::CONCRETE => 12e6, mat::MASONRY => 0.5e6, mat::STEEL => 250e6, mat::GLASS => 40e6, _ => 20e6 }
}

/// A crack through a brick from bending: split brick `brick` at tick `at` across `axis`.
#[derive(Clone, Copy, Debug)]
pub struct Crack { pub brick: u32, pub axis: usize, pub at: i32, pub ratio: f32 }

// ======================================================================================================
// Structural analysis: a rigid-body-spring network over cells.
//
// Big bricks are single records in the store, but a 22 m slab must be able to bend and crack inside. For the
// analysis each brick is cut into cells of at most CELL ticks (2 m) across x and z, and at the edges of what
// bears on it. Each cell is a rigid body with 6 unknowns (3 displacements, 3 rotations); cells of one brick are
// joined through the brick's own section, and joints between bricks become springs between the cells that
// share the face (normal, shear, bending and torsion stiffness from the materials' moduli, two halves in
// series). One solve gives every face's force and moment; then every face is checked:
//   - joints between bricks (sub-faces summed about the joint's centre): tension, compression, Mohr-Coulomb
//     shear and torsion on the face as a section; over strength the joint fails (or, brick, cracks)
//   - faces inside a brick: the brick's own section; over strength it cracks there
// ======================================================================================================

pub const CELL: i32 = 16;

pub struct Analysis {
    pub failed: Vec<(u32, f32)>,  // joints over strength, with their worst ratio
    pub cracks: Vec<Crack>,       // bricks cracking inside
    pub max_ratio: f32,
    pub iterations: usize,
    pub nodes: usize,
    /// the worst link inside a brick: (ratio, brick, axis, flow N, moment N m, section width m)
    pub worst_in: (f32, u32, usize, f32, f32, f32),
    /// every joint checked, with its ratio
    pub ratios: Vec<(u32, f32)>,
}

struct Node { brick: u32, lo: [i32; 3], hi: [i32; 3], load: f32, sink: bool, c: [f64; 3] }
/// A link: between two cells of one brick (`joint` NONE), or across a joint. `face_lo..face_hi` is its face on the
/// tick grid; a face where a convex cell meets something is a polygon, and `geo` holds it.
struct Link { a: u32, b: u32, joint: u32, axis: usize, face_lo: [i32; 3], face_hi: [i32; 3], geo: Option<Geo> }
/// A face as a section: centre (m), frame (rows: normal, u, w), area, second moments about u and w, the extreme
/// fibres' distances for each, the polar moment and the farthest point (m, m^2, m^4).
#[derive(Clone, Copy)]
struct Geo { c: [f64; 3], fr: [[f64; 3]; 3], s: Sect }
#[derive(Clone, Copy)]
struct Sect { area: f64, iu: f64, iw: f64, zu: f64, zw: f64, j: f64, rmax: f64 }
impl Sect {
    /// a rectangle eu x ew (along u and w)
    fn rect(eu: f64, ew: f64, area: f64) -> Sect {
        let (eu, ew) = (eu.max(1e-3), ew.max(1e-3));
        Sect { area: area.max(1e-6), iu: eu * ew * ew * ew / 12.0, iw: ew * eu * eu * eu / 12.0, zu: ew / 2.0, zw: eu / 2.0, j: eu * ew * (eu * eu + ew * ew) / 12.0, rmax: 0.5 * sqrt64(eu * eu + ew * ew) }
    }
}
/// a polygon face (m) as a section; `ax` < 3 for one on an axis plane (its frame is then the axes')
fn geo_of(poly: &[[f64; 3]], ax: usize) -> Option<Geo> {
    let (area, c, n) = crate::hull::polygon(poly);
    if area < 1e-6 { return None; }
    let fr = if ax < 3 {
        let e = |d: usize| { let mut v = [0.0; 3]; v[d] = 1.0; v };
        [e(ax), e((ax + 1) % 3), e((ax + 2) % 3)]
    } else {
        // u: the axis least along the normal, made perpendicular
        let k = (0..3).min_by(|&a, &b| n[a].abs().partial_cmp(&n[b].abs()).unwrap()).unwrap();
        let mut e = [0.0; 3]; e[k] = 1.0;
        let d = e[0] * n[0] + e[1] * n[1] + e[2] * n[2];
        let u = [e[0] - d * n[0], e[1] - d * n[1], e[2] - d * n[2]];
        let l = sqrt64(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
        let u = [u[0] / l, u[1] / l, u[2] / l];
        let w = [n[1] * u[2] - n[2] * u[1], n[2] * u[0] - n[0] * u[2], n[0] * u[1] - n[1] * u[0]];
        [n, u, w]
    };
    let ([iu, iw], [zu, zw]) = crate::hull::polygon_section(poly, c, fr[1], fr[2]);
    let mut rmax = 0.0f64;
    for p in poly { let r = [p[0] - c[0], p[1] - c[1], p[2] - c[2]]; rmax = rmax.max(sqrt64(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])); }
    Some(Geo { c, fr, s: Sect { area, iu: iu.max(1e-12), iw: iw.max(1e-12), zu: zu.max(1e-4), zw: zw.max(1e-4), j: (iu + iw).max(1e-12), rmax } })
}
fn rot(fr: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] { [fr[0][0] * v[0] + fr[0][1] * v[1] + fr[0][2] * v[2], fr[1][0] * v[0] + fr[1][1] * v[1] + fr[1][2] * v[2], fr[2][0] * v[0] + fr[2][1] * v[1] + fr[2][2] * v[2]] }
fn rot_t(fr: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] { [fr[0][0] * v[0] + fr[1][0] * v[1] + fr[2][0] * v[2], fr[0][1] * v[0] + fr[1][1] * v[1] + fr[2][1] * v[2], fr[0][2] * v[0] + fr[1][2] * v[1] + fr[2][2] * v[2]] }

fn cells_of(t: &crate::brick::TBox) -> (Vec<i32>, Vec<i32>) {
    let cut = |lo: i32, hi: i32| { let n = ((hi - lo + CELL - 1) / CELL).max(1); let mut v = vec![lo]; for k in 1..n { v.push(lo + (hi - lo) * k / n); } v.push(hi); v };
    (cut(t.min[0], t.max[0]), cut(t.min[2], t.max[2]))
}

/// Analyse `set` under per-brick loads `load` (N) acting along `dir`, supported at `sink` bricks.
/// Moments are evaluated for vertical loads (gravity).
pub fn analyse(b: &Bricks, set: &[u32], load: &[f32], sink: &[bool], dir: V3) -> Analysis { analyse_at(b, set, load, &[], sink, dir) }
/// As `analyse`, plus loads at points: (index into `set`, x, z in ticks, N). A point load goes to the cell of
/// that brick under the point (a heap of rubble on one bay of a 22 m slab loads that bay, not the whole slab).
pub fn analyse_at(b: &Bricks, set: &[u32], load: &[f32], point: &[(u32, f32, f32, f32)], sink: &[bool], dir: V3) -> Analysis { analyse_warm(b, set, load, point, sink, dir, None) }
/// The last solution of a load case, per cell (brick, cell's corner x, z ticks): the next solve of the standing
/// building starts from it. Between two checks most of the building moves the same.
#[derive(Default)]
pub struct Warm(crate::Map<(u32, i32, i32), [f64; 6]>);
/// As `analyse_at`, starting from (and keeping) the last solution in `warm`.
pub fn analyse_warm(b: &Bricks, set: &[u32], load: &[f32], point: &[(u32, f32, f32, f32)], sink: &[bool], dir: V3, warm: Option<&mut Warm>) -> Analysis {
    let gravity = fabs(dir.y) > 0.99;
    let mut nodes: Vec<Node> = Vec::new();
    let mut first = vec![NONE; b.len()];     // brick -> first node
    let mut grid: Vec<(usize, usize)> = vec![(0, 0); set.len()];
    let mut links: Vec<Link> = Vec::new();
    let mut cuts: Vec<(Vec<i32>, Vec<i32>)> = Vec::with_capacity(set.len());   // each brick's cell boundaries
    let mut slot = vec![0u32; b.len()];   // brick -> its place in `set`
    for (k, &i) in set.iter().enumerate() { slot[i as usize] = k as u32; }
    for (k, &i) in set.iter().enumerate() {
        let t = b.tbox(i);
        let infill = b.flags[i as usize] & F_INFILL != 0;
        // (a convex cell is one node: it is no bigger than fracture makes bricks)
        let (xs, zs) = if infill || b.hull[i as usize] != 0 { (vec![t.min[0], t.max[0]], vec![t.min[2], t.max[2]]) } else {
            // the edges of the supports it rests on are where the moment peaks: cells are cut there too
            let (mut xs, mut zs) = cells_of(&t);
            let mut j = b.jhead[i as usize];
            let (mut ex, mut ez) = (Vec::new(), Vec::new());
            while j != NONE {
                let oi = b.joint_other(j, i);
                let o = b.tbox(oi);
                // (infill carries nothing: glass, cladding and partitions under a slab are no supports)
                if b.jaxis[j as usize] == 1 && o.max[1] == t.min[1] && b.flags[oi as usize] & F_INFILL == 0 {
                    for v in [o.min[0], o.max[0]] { if v > t.min[0] && v < t.max[0] { ex.push(v); } }
                    for v in [o.min[2], o.max[2]] { if v > t.min[2] && v < t.max[2] { ez.push(v); } }
                }
                j = b.joint_next(j, i);
            }
            let merge = |base: &mut Vec<i32>, extra: Vec<i32>| {
                for v in extra { base.push(v); }
                base.sort_unstable(); base.dedup();
            };
            merge(&mut xs, ex); merge(&mut zs, ez);
            (xs, zs)
        };
        first[i as usize] = nodes.len() as u32;
        grid[k] = (xs.len() - 1, zs.len() - 1);
        cuts.push((xs.clone(), zs.clone()));
        let vol = ((t.max[0] - t.min[0]) * (t.max[2] - t.min[2])) as f32;
        for cz in 0..zs.len() - 1 { for cx in 0..xs.len() - 1 {
            let (lo, hi) = ([xs[cx], t.min[1], zs[cz]], [xs[cx + 1], t.max[1], zs[cz + 1]]);
            let share = ((hi[0] - lo[0]) * (hi[2] - lo[2])) as f32 / vol;
            let tk = TICK as f64;
            let c = if b.hull[i as usize] != 0 { b.centroid(i) } else { [0.5 * (lo[0] + hi[0]) as f64 * tk, 0.5 * (lo[1] + hi[1]) as f64 * tk, 0.5 * (lo[2] + hi[2]) as f64 * tk] };
            nodes.push(Node { brick: i, lo, hi, load: load[k] * share, sink: sink[k], c });
        } }
        // links inside the brick
        let (nx, nz) = grid[k];
        let base = first[i as usize];
        for cz in 0..nz { for cx in 0..nx {
            let id = base + (cz * nx + cx) as u32;
            let n = &nodes[id as usize];
            if cx + 1 < nx {
                let (lo, hi) = ([n.hi[0], n.lo[1], n.lo[2]], [n.hi[0], n.hi[1], n.hi[2]]);
                links.push(Link { a: id, b: id + 1, joint: NONE, axis: 0, face_lo: lo, face_hi: hi, geo: None });
            }
            if cz + 1 < nz {
                let (lo, hi) = ([n.lo[0], n.lo[1], n.hi[2]], [n.hi[0], n.hi[1], n.hi[2]]);
                links.push(Link { a: id, b: id + nx as u32, joint: NONE, axis: 2, face_lo: lo, face_hi: hi, geo: None });
            }
        } }
    }
    // links across joints, cell to cell where the face overlaps
    for (k, &i) in set.iter().enumerate() {
        let mut j = b.jhead[i as usize];
        while j != NONE {
            let o = b.joint_other(j, i);
            if first[o as usize] != NONE && b.ja[j as usize] == i && b.jaxis[j as usize] == 3 {
                // two cells of one fractured brick, on their shared face
                let (ia, ib) = (first[i as usize], first[o as usize]);
                if let Some(g) = geo_of(&b.joint_poly(j), 3) {
                    let tk = TICK as f64;
                    let q = |v: f64| floor64(v / tk + 0.5) as i32;
                    let lo = [q(g.c[0]), q(g.c[1]), q(g.c[2])];
                    links.push(Link { a: ia, b: ib, joint: j, axis: 3, face_lo: lo, face_hi: lo, geo: Some(g) });
                }
            } else if first[o as usize] != NONE && b.ja[j as usize] == i {
                let ax = b.jaxis[j as usize] as usize;
                // a face of a convex cell: the joint's polygon, cut to each pair of cells below
                let poly = if b.hull[i as usize] != 0 || b.hull[o as usize] != 0 { Some(b.joint_poly(j)) } else { None };
                let ko = slot[o as usize] as usize;
                let (na, nb) = (grid[k], grid[ko]);
                let plane = b.tbox(i).max[ax];
                // the cells of o that overlap a cell of i: found by the cell boundaries (both are grids)
                let (bx, bz) = (&cuts[ko].0, &cuts[ko].1);
                let range = |c: &Vec<i32>, lo: i32, hi: i32| -> (usize, usize) {
                    let n = c.len() - 1;
                    let a = c.partition_point(|&v| v <= lo).saturating_sub(1).min(n);
                    let z = c.partition_point(|&v| v < hi).min(n);
                    (a, z.max(a))
                };
                let mut pairs: Vec<(usize, usize)> = Vec::new();
                for ca in 0..na.0 * na.1 {
                    let p = &nodes[(first[i as usize] + ca as u32) as usize];
                    if ax != 1 && p.hi[ax] != plane { continue; }
                    let (x0, x1) = if ax == 0 { (0, nb.0) } else { range(bx, p.lo[0], p.hi[0]) };
                    let (z0, z1) = if ax == 2 { (0, nb.1) } else { range(bz, p.lo[2], p.hi[2]) };
                    for cz in z0..z1 { for cx in x0..x1 { pairs.push((ca, cz * nb.0 + cx)); } }
                }
                for (ca, cb) in pairs {
                    let (ia, ib) = (first[i as usize] + ca as u32, first[o as usize] + cb as u32);
                    let (p, q) = (&nodes[ia as usize], &nodes[ib as usize]);
                    let mut lo = [0; 3]; let mut hi = [0; 3]; let mut ok = true;
                    for d in 0..3 {
                        if d == ax { lo[d] = plane; hi[d] = plane; continue; }
                        lo[d] = p.lo[d].max(q.lo[d]); hi[d] = p.hi[d].min(q.hi[d]);
                        if hi[d] <= lo[d] { ok = false; }
                    }
                    if !ok { continue; }
                    let geo = match &poly {
                        None => None,
                        Some(pl) => {
                            let tk = TICK as f64;
                            let (mut rl, mut rh) = ([0.0; 3], [0.0; 3]);
                            for d in 0..3 { rl[d] = lo[d] as f64 * tk; rh[d] = hi[d] as f64 * tk; }
                            match geo_of(&crate::hull::clip_rect(pl, ax, rl, rh), ax) { Some(g) => Some(g), None => continue }
                        }
                    };
                    links.push(Link { a: ia, b: ib, joint: j, axis: ax, face_lo: lo, face_hi: hi, geo });
                }
            }
            j = b.joint_next(j, i);
        }
    }
    for &(k, x, z, f) in point {
        let i = set[k as usize];
        let (fst, (nx, nz)) = (first[i as usize], grid[k as usize]);
        if fst == NONE { continue; }
        let (xs, zs) = (&cuts[k as usize].0, &cuts[k as usize].1);
        let cx = xs.partition_point(|&v| (v as f32) <= x).saturating_sub(1).min(nx - 1);
        let cz = zs.partition_point(|&v| (v as f32) <= z).saturating_sub(1).min(nz - 1);
        nodes[(fst + (cz * nx + cx) as u32) as usize].load += f;
    }
    let n = nodes.len();
    // nodes connected to a support
    let mut head = vec![NONE; n]; let mut next = vec![NONE; 2 * links.len()];
    for (l, k) in links.iter().enumerate() {
        next[2 * l] = head[k.a as usize]; head[k.a as usize] = (2 * l) as u32;
        next[2 * l + 1] = head[k.b as usize]; head[k.b as usize] = (2 * l + 1) as u32;
    }
    let other = |e: u32| -> u32 { let l = (e / 2) as usize; if e % 2 == 0 { links[l].b } else { links[l].a } };
    let mut live = vec![false; n];
    let mut st: Vec<u32> = (0..n as u32).filter(|&k| nodes[k as usize].sink).collect();
    for &k in &st { live[k as usize] = true; }
    while let Some(k) = st.pop() { let mut e = head[k as usize]; while e != NONE { let o = other(e); if !live[o as usize] { live[o as usize] = true; st.push(o); } e = next[e as usize]; } }
    // ---- the spring network: each cell a rigid body (3 displacements, 3 rotations), each link a spring at the
    // centre of its face, solved for the displacements under the loads with the ground cells fixed
    let springs: Vec<Spring> = links.iter().map(|l| spring(b, &nodes, l)).collect();
    let fixed: Vec<bool> = (0..n).map(|k| nodes[k].sink || !live[k]).collect();
    let mut f = vec![[0.0f64; 6]; n];
    for k in 0..n { let w = nodes[k].load as f64; f[k][0] = w * dir.x as f64; f[k][1] = w * dir.y as f64; f[k][2] = w * dir.z as f64; }
    let ends: Vec<(u32, u32)> = links.iter().map(|l| (l.a, l.b)).collect();
    let tk = TICK as f64;
    let _ = tk;
    let pos: Vec<[f64; 3]> = nodes.iter().map(|nd| nd.c).collect();
    let key = |k: usize| (nodes[k].brick, nodes[k].lo[0], nodes[k].lo[2]);
    let x0: Option<Vec<[f64; 6]>> = warm.as_ref().map(|w| (0..n).map(|k| if fixed[k] { [0.0; 6] } else { w.0.get(&key(k)).copied().unwrap_or([0.0; 6]) }).collect());
    let (x, it) = solve_network(&springs, &ends, &fixed, &pos, &f, x0);
    if let Some(w) = warm { w.0.clear(); for k in 0..n { if !fixed[k] { w.0.insert(key(k), x[k]); } } }
    let mut out = Analysis { failed: Vec::new(), cracks: Vec::new(), max_ratio: 0.0, iterations: it, nodes: n, worst_in: (0.0, 0, 0, 0.0, 0.0, 0.0), ratios: Vec::new() };
    // forces on each face (on the max side's cell, from the min side's): force and moment about the face centre
    let face_load = |l: usize| -> ([f64; 3], [f64; 3]) {
        let sp = &springs[l];
        let (a, c) = (links[l].a as usize, links[l].b as usize);
        let (d, r) = sp.strain(&x[a], &x[c]);
        sp.force(d, r)
    };
    // a link's or joint's force and moment in its face's frame (normal, u, w)
    let local = |ax: usize, geo: &Option<Geo>, v: [f64; 3]| -> [f64; 3] {
        match geo { Some(g) => rot(&g.fr, v), None => [v[ax], v[(ax + 1) % 3], v[(ax + 2) % 3]] }
    };
    // joints: their sub-links summed about the joint's face centre, then checked as one section
    let mut jsum: Vec<(u32, [f64; 3], [f64; 3], [f64; 3], f64)> = Vec::new();   // joint, force, moment, force-weighted.., area
    let mut jslot = crate::Map::default();
    let mut parts: Vec<Vec<usize>> = Vec::new();
    for (l, k) in links.iter().enumerate() {
        if k.joint == NONE || !live[k.a as usize] || !live[k.b as usize] { continue; }
        let s = *jslot.entry(k.joint).or_insert_with(|| { jsum.push((k.joint, [0.0; 3], [0.0; 3], [0.0; 3], 0.0)); parts.push(Vec::new()); jsum.len() - 1 });
        parts[s].push(l);
    }
    for (s, ls) in parts.iter().enumerate() {
        let j = jsum[s].0; let ju = j as usize;
        let ax = b.jaxis[ju] as usize;
        // the joint's face: the union of its sub-faces (or the polygon where a convex cell meets it); its centre,
        // area and section
        let (mut lo, mut hi) = ([i32::MAX; 3], [i32::MIN; 3]);
        for &l in ls { for d in 0..3 { lo[d] = lo[d].min(links[l].face_lo[d]); hi[d] = hi[d].max(links[l].face_hi[d]); } }
        let tk = TICK as f64;
        let jgeo = if links[ls[0]].geo.is_some() { geo_of(&b.joint_poly(j), ax) } else { None };
        let c = match &jgeo { Some(g) => g.c, None => [0.5 * (lo[0] + hi[0]) as f64 * tk, 0.5 * (lo[1] + hi[1]) as f64 * tk, 0.5 * (lo[2] + hi[2]) as f64 * tk] };
        let (mut ft, mut mt) = ([0.0f64; 3], [0.0f64; 3]);
        for &l in ls {
            let (fo, mo) = face_load(l);
            let p = springs[l].c;
            let r = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
            for i in 0..3 { ft[i] += fo[i]; mt[i] += mo[i]; }
            let rx = [r[1] * fo[2] - r[2] * fo[1], r[2] * fo[0] - r[0] * fo[2], r[0] * fo[1] - r[1] * fo[0]];
            for i in 0..3 { mt[i] += rx[i]; }
        }
        let (ia, ib) = (b.ja[ju] as usize, b.jb[ju] as usize);
        let (ma, mb) = (b.mat[ia], b.mat[ib]);
        let sect = match &jgeo { Some(g) => g.s, None => { let (u, w) = ((ax + 1) % 3, (ax + 2) % 3); Sect::rect((hi[u] - lo[u]) as f64 * tk, (hi[w] - lo[w]) as f64 * tk, b.jarea[ju] as f64 * tk * tk) } };
        let conn = connection(b, j);
        let cracked_masonry = ma == mat::MASONRY || mb == mat::MASONRY;
        let residual: [f32; 3] = if b.jdmg[ju] == 0 { [1.0; 3] } else if cracked_masonry { [1.0, 0.05, 0.4] } else { [0.5; 3] };
        let sa = strength(ma); let sb = strength(mb);
        let cap = [sa[0].min(sb[0]) * conn[0] * residual[0], sa[1].min(sb[1]) * conn[1] * residual[1], sa[2].min(sb[2]) * conn[2] * residual[2]];
        let flex = flexural_of(b, ia).min(flexural_of(b, ib)) * conn[1] * residual[1];
        let hinged = hinge(b, ia, ib);
        let (fl, mut ml) = (local(ax, &jgeo, ft), local(ax, &jgeo, mt));
        if hinged { ml = [0.0; 3]; }
        let ratio = section_ratio(&sect, fl, ml, cap, flex);
        out.ratios.push((j, ratio));
        if ratio > out.max_ratio { out.max_ratio = ratio; }
        if ratio > 1.0 { out.failed.push((j, ratio)); }
    }
    // faces inside bricks: the brick's own section; over strength it cracks there
    for (l, k) in links.iter().enumerate() {
        if k.joint != NONE || !live[k.a as usize] { continue; }
        let i = nodes[k.a as usize].brick;
        let m = b.mat[i as usize];
        let (u, w) = ((k.axis + 1) % 3, (k.axis + 2) % 3);
        let tk = TICK as f64;
        let (eu, ew) = ((k.face_hi[u] - k.face_lo[u]) as f64 * tk, (k.face_hi[w] - k.face_lo[w]) as f64 * tk);
        let (fo, mo) = face_load(l);
        let ratio = section_ratio(&Sect::rect(eu, ew, eu * ew), local(k.axis, &None, fo), local(k.axis, &None, mo), strength(m), flexural_of(b, i as usize));
        if ratio > out.worst_in.0 { out.worst_in = (ratio, i, k.axis, fo[k.axis] as f32, (mo[u] * mo[u] + mo[w] * mo[w]) as f32, 0.0); }
        if ratio > out.max_ratio { out.max_ratio = ratio; }
        if ratio > 1.0 { out.cracks.push(Crack { brick: i, axis: k.axis, at: k.face_lo[k.axis], ratio }); }
    }
    let _ = gravity;
    out
}

/// A spring at a face between two cells: stiffness per axis in world axes (the faces are axis-aligned) for the
/// relative displacement at the face centre (`k`: normal along the face's axis, shear along the others) and the
/// relative rotation (`kr`: torsion about the axis, bending about the others), and the arms from each cell's
/// centre to the face centre (m).
/// A face that is not on an axis plane (between two cells of a fractured brick) has its frame in `fr`, and `k`, `kr`
/// are then along its normal, u and w.
#[derive(Clone)]
struct Spring { c: [f64; 3], ra: [f64; 3], rb: [f64; 3], k: [f64; 3], kr: [f64; 3], fr: Option<[[f64; 3]; 3]> }
impl Spring {
    /// force and moment (world) for relative displacement d and rotation r
    #[inline] fn force(&self, d: [f64; 3], r: [f64; 3]) -> ([f64; 3], [f64; 3]) {
        match &self.fr {
            None => ([self.k[0] * d[0], self.k[1] * d[1], self.k[2] * d[2]], [self.kr[0] * r[0], self.kr[1] * r[1], self.kr[2] * r[2]]),
            Some(fr) => {
                let (dl, rl) = (rot(fr, d), rot(fr, r));
                (rot_t(fr, [self.k[0] * dl[0], self.k[1] * dl[1], self.k[2] * dl[2]]), rot_t(fr, [self.kr[0] * rl[0], self.kr[1] * rl[1], self.kr[2] * rl[2]]))
            }
        }
    }
    /// the 6 x 6 stiffness of the face (force and moment against displacement and rotation), world axes
    fn d6(&self) -> M6 {
        let mut d = [[0.0; 6]; 6];
        match &self.fr {
            None => { for i in 0..3 { d[i][i] = self.k[i]; d[3 + i][3 + i] = self.kr[i]; } }
            Some(fr) => { for i in 0..3 { for j in 0..3 { for a in 0..3 { d[i][j] += fr[a][i] * self.k[a] * fr[a][j]; d[3 + i][3 + j] += fr[a][i] * self.kr[a] * fr[a][j]; } } } }
        }
        d
    }
    /// relative displacement at the face centre and relative rotation (cell b against cell a)
    fn strain(&self, xa: &[f64; 6], xb: &[f64; 6]) -> ([f64; 3], [f64; 3]) {
        let pt = |x: &[f64; 6], r: &[f64; 3]| [x[0] + x[4] * r[2] - x[5] * r[1], x[1] + x[5] * r[0] - x[3] * r[2], x[2] + x[3] * r[1] - x[4] * r[0]];
        let (pa, pb) = (pt(xa, &self.ra), pt(xb, &self.rb));
        ([pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]], [xb[3] - xa[3], xb[4] - xa[4], xb[5] - xa[5]])
    }
}
/// Young's modulus of brick `i` (Pa). A timber floor is joists (50 x 225 mm at 400 mm): its plate has the
/// bending stiffness of the joists, about a tenth of solid timber's.
pub fn modulus_of(b: &Bricks, i: usize) -> f64 {
    match b.mat[i] { mat::CONCRETE => 30e9, mat::MASONRY => 5e9, mat::STEEL => 200e9, mat::GLASS => 70e9, _ => if b.paint[i] == 8 { 1e9 } else { 11e9 } }
}
fn spring(b: &Bricks, nodes: &[Node], l: &Link) -> Spring {
    let (na, nb) = (&nodes[l.a as usize], &nodes[l.b as usize]);
    let t = TICK as f64;
    let (pa, pb) = (na.c, nb.c);
    let ax = l.axis;
    // the face as a section, in its frame (normal, u, w): a rectangle on the grid, or a convex cell's polygon
    let (c, fr, sec) = match &l.geo {
        Some(g) => (g.c, g.fr, g.s),
        None => {
            let (u, w) = ((ax + 1) % 3, (ax + 2) % 3);
            let (eu, ew) = ((l.face_hi[u] - l.face_lo[u]) as f64 * t, (l.face_hi[w] - l.face_lo[w]) as f64 * t);
            let e = |d: usize| { let mut v = [0.0; 3]; v[d] = 1.0; v };
            ([0.5 * (l.face_lo[0] + l.face_hi[0]) as f64 * t, 0.5 * (l.face_lo[1] + l.face_hi[1]) as f64 * t, 0.5 * (l.face_lo[2] + l.face_hi[2]) as f64 * t], [e(ax), e(u), e(w)], Sect::rect(eu, ew, eu * ew))
        }
    };
    let n = fr[0];
    let along = |p: [f64; 3]| ((c[0] - p[0]) * n[0] + (c[1] - p[1]) * n[1] + (c[2] - p[2]) * n[2]).abs().max(0.5 * t);
    let (la, lb) = (along(pa), along(pb));
    let (ea, eb) = (modulus_of(b, na.brick as usize), modulus_of(b, nb.brick as usize));
    // the two halves in series (compliance per unit E: length / E)
    let comp = la / ea + lb / eb;
    let gcomp = comp / 0.4;
    // in the face's frame: normal, two shears; torsion, bending about u (stresses vary along w), about w
    let kl = [sec.area / comp, sec.area / gcomp, sec.area / gcomp];
    let krl = [sec.j / gcomp, sec.iu / comp, sec.iw / comp];
    let (mut k, mut kr, gen) = if ax < 3 {
        let (u, w) = ((ax + 1) % 3, (ax + 2) % 3);
        let (mut k, mut kr) = ([0.0; 3], [0.0; 3]);
        k[ax] = kl[0]; k[u] = kl[1]; k[w] = kl[2]; kr[ax] = krl[0]; kr[u] = krl[1]; kr[w] = krl[2];
        (k, kr, None)
    } else { (kl, krl, Some(fr)) };
    // the links' own factors: infill conducts little, a stair flight is flexible, steel in a brick pocket bears only
    // vertically, a timber floor's seam with another material is a hinge
    let (ia, ib) = (na.brick as usize, nb.brick as usize);
    let infill = (b.flags[ia] | b.flags[ib]) & F_INFILL != 0;
    let mono = (b.flags[ia] | b.flags[ib]) & crate::brick::F_MONO != 0;
    let (mi, mo) = (b.mat[ia], b.mat[ib]);
    let pocket = l.joint != NONE && ax != 1 && ax < 3 && ((mi == mat::STEEL && mo == mat::MASONRY) || (mi == mat::MASONRY && mo == mat::STEEL));
    let f = if infill || pocket { 0.02 } else if mono { 0.05 } else { 1.0 };
    for i in 0..3 { k[i] *= f; kr[i] *= f; }
    if l.joint != NONE && hinge(b, ia, ib) { if ax < 3 { kr[(ax + 1) % 3] = 0.0; kr[(ax + 2) % 3] = 0.0; } else { kr[1] = 0.0; kr[2] = 0.0; } }
    // a joist floor spans one way, along its joists (x: from the walls to the beams): across them only the
    // boards bend, and open joists hardly twist, each about a twentieth as stiff as a solid plate. As a plate
    // bending both ways, a 3.25 m bay under 40 kPa came out at a third of its strength; with the boards soft but
    // the deck still twisting as a solid plate, 0.68 (torsion about the joists carried a third of the load to the
    // walls along the bay's sides)
    let tf = |i: usize| b.mat[i] == mat::TIMBER && b.paint[i] == 8;
    if tf(ia) && tf(ib) && ax == 2 { kr[0] *= 0.05; kr[2] *= 0.05; }
    if tf(ia) && tf(ib) && ax == 0 { kr[0] *= 0.05; }
    Spring { c, ra: [c[0] - pa[0], c[1] - pa[1], c[2] - pa[2]], rb: [c[0] - pb[0], c[1] - pb[1], c[2] - pb[2]], k, kr, fr: gen }
}
/// Stress over strength of a face (section `s`) under force `f` and moment `m` on the max side, in the face's frame
/// (normal, u, w; N and N m): the worst of
/// - tension at the extreme fibre, N / A + M / S, against the tensile strength, or the flexural strength where
///   bending makes most of it
/// - compression at the other extreme fibre, against the compressive strength
/// - shear (V / A, plus torsion) against cohesion plus friction on the compression: 0.4 sigma_n (Mohr-Coulomb;
///   a brick wall under load is stronger in shear than mortar alone)
fn section_ratio(s: &Sect, f: [f64; 3], m: [f64; 3], cap: [f32; 3], flex: f32) -> f32 {
    // normal force: positive = the max side pulled away (tension: the spring is stretched)
    let sn = f[0] / s.area;
    let bend = m[1].abs() * s.zu / s.iu + m[2].abs() * s.zw / s.iw;
    let top = sn + bend;          // the most tensile fibre
    let bottom = sn - bend;       // the most compressed fibre
    let mut r = 0.0f64;
    if top > 0.0 {
        let tens = sn.max(0.0);
        let fe = (tens * cap[1] as f64 + bend * flex as f64) / (tens + bend).max(1e-9);
        r = r.max(top / fe.max(1.0));
    }
    if bottom < 0.0 { r = r.max(-bottom / cap[0] as f64); }
    let v = sqrt64(f[1] * f[1] + f[2] * f[2]) / s.area;
    let tor = m[0].abs() * s.rmax / s.j;
    let cohesion = cap[2] as f64 + 0.4 * (-sn).max(0.0);
    r = r.max((v + tor) / cohesion);
    r as f32
}
/// Conjugate gradients for K x = f on the spring network (6 unknowns per cell, the fixed ones held at 0),
/// preconditioned by a multigrid V-cycle. The levels: cells grouped into blocks of 4, 8, 16... m, each block
/// rigid. A spring between two blocks becomes the same spring with its arms measured from the blocks'
/// centres (the Galerkin coarse problem of a rigid grouping is again a spring network); springs inside a block
/// drop out. Damped block-Jacobi smoothing (each cell's 6 x 6 block inverted) on every level.
/// With block Jacobi alone the 12-storey tower took 1,552 iterations and 1.2 s (the long, slow bending modes of a
/// tall building converge slowly).
fn solve_network(sp: &[Spring], ends: &[(u32, u32)], fixed: &[bool], pos: &[[f64; 3]], f: &[[f64; 6]], x0: Option<Vec<[f64; 6]>>) -> (Vec<[f64; 6]>, usize) {
    let n = fixed.len();
    #[cfg(test)] let t0 = { extern crate std; std::time::Instant::now() };
    let mut levels = vec![Level::from_springs(sp, ends, fixed.to_vec(), pos.to_vec())];
    let mut bin = 3.0f64;
    while levels.last().unwrap().n > 64 && levels.len() < 8 {
        let c = levels.last().unwrap().coarsen(bin);
        if c.n as f64 > 0.7 * levels.last().unwrap().n as f64 { bin *= 2.0; if bin > 256.0 { break; } continue; }
        levels.push(c); bin *= 2.0;
    }
    // (the finest level's blocks were needed only to build the next one)
    levels[0].off = Vec::new();
    #[cfg(test)] { extern crate std; if std::env::var("SF_MG").is_ok() { std::println!("  levels built in {:.0} ms", t0.elapsed().as_secs_f64() * 1e3); for (i, l) in levels.iter().enumerate() { std::println!("  level {i}: {} cells, {} couplings", l.n, l.off.len()); } } }
    let dot = |a: &[[f64; 6]], b: &[[f64; 6]]| -> f64 { let mut s = 0.0; for k in 0..a.len() { for i in 0..6 { s += a[k][i] * b[k][i]; } } s };
    let mut r: Vec<[f64; 6]> = (0..n).map(|k| if fixed[k] { [0.0; 6] } else { f[k] }).collect();
    let mut ap = vec![[0.0f64; 6]; n];
    // the tolerance is measured against the residual from zero, so that a warm start stops at the same accuracy
    let z0 = vcycle(&levels, 0, &r);
    let r0 = dot(&r, &z0).max(1e-30);
    let mut x = match x0 {
        Some(x0) => { levels[0].apply(&x0, &mut ap); for k in 0..n { for i in 0..6 { r[k][i] -= ap[k][i]; } } x0 }
        None => vec![[0.0f64; 6]; n],
    };
    let mut z = vcycle(&levels, 0, &r);
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    if rz < TOL * r0 { return (x, 0); }
    let mut it = 0;
    while it < MAX_IT {
        levels[0].apply(&p, &mut ap);
        let pap = dot(&p, &ap); if pap <= 0.0 { break; }
        let alpha = rz / pap;
        for k in 0..n { for i in 0..6 { x[k][i] += alpha * p[k][i]; r[k][i] -= alpha * ap[k][i]; } }
        it += 1;
        z = vcycle(&levels, 0, &r);
        let rz2 = dot(&r, &z);
        if rz2 < TOL * r0 { break; }
        let beta = rz2 / rz; rz = rz2;
        for k in 0..n { for i in 0..6 { p[k][i] = z[k][i] + beta * p[k][i]; } }
    }
    #[cfg(test)] { extern crate std; if std::env::var("SF_MG").is_ok() { std::println!("  solved in {:.0} ms", t0.elapsed().as_secs_f64() * 1e3); } }
    (x, it)
}
const MAX_IT: usize = 2000;
// (r z down by 1e-5 from the start: every building's worst ratio and the validation cases agree with 1e-8 to the
// digits the tests print, in 62 iterations against 104 for the 12-storey tower)
const TOL: f64 = 1e-5;
const SMOOTH: usize = 1;
const OMEGA: f64 = 0.8;

type M6 = [[f64; 6]; 6];
fn mmul(a: &M6, b: &M6) -> M6 { let mut c = [[0.0; 6]; 6]; for i in 0..6 { for k in 0..6 { let v = a[i][k]; if v != 0.0 { for j in 0..6 { c[i][j] += v * b[k][j]; } } } } c }
fn tr(a: &M6) -> M6 { let mut c = [[0.0; 6]; 6]; for i in 0..6 { for j in 0..6 { c[j][i] = a[i][j]; } } c }
/// the rigid map from a body's 6 unknowns to a point at offset d: [I, -[d]x; 0, I]
fn rigid(d: &[f64; 3]) -> M6 {
    let mut t = [[0.0; 6]; 6];
    for i in 0..6 { t[i][i] = 1.0; }
    // u_point = u + theta x d = u - [d]x theta
    t[0][4] = d[2]; t[0][5] = -d[1]; t[1][3] = -d[2]; t[1][5] = d[0]; t[2][3] = d[1]; t[2][4] = -d[0];
    t
}
/// A level of the solve as a block matrix: each cell's own 6 x 6 block and one block per pair of coupled cells.
/// The finest level also keeps its springs (`sp`, `ends`) and applies K from them directly: 15 numbers a spring
/// against a 6 x 6 block, and the solve is bound by memory (the 12-storey tower: 4.3 ms an iteration with blocks).
struct Level { n: usize, diag: Vec<M6>, off: Vec<(u32, u32, M6)>, fixed: Vec<bool>, pos: Vec<[f64; 3]>, inv: Vec<M6>, up: Vec<u32>, sp: Vec<Spring>, ends: Vec<(u32, u32)>, reg: Vec<f64> }
impl Level {
    /// the finest level, from the springs: strain e = B_b x_b - B_a x_a (B the rigid map to the face centre),
    /// energy e^T D e / 2, so K_aa = B_a^T D B_a, K_bb = B_b^T D B_b, K_ab = -B_a^T D B_b
    fn from_springs(sp: &[Spring], ends: &[(u32, u32)], fixed: Vec<bool>, pos: Vec<[f64; 3]>) -> Level {
        let n = fixed.len();
        let mut diag = vec![[[0.0; 6]; 6]; n];
        let mut off = Vec::with_capacity(sp.len());
        for (s, &(a, c)) in sp.iter().zip(ends) {
            let d = s.d6();
            let (ba, bb) = (rigid(&s.ra), rigid(&s.rb));
            let (dba, dbb) = (mmul(&d, &ba), mmul(&d, &bb));
            let (au, cu) = (a as usize, c as usize);
            if !fixed[au] { let m = mmul(&tr(&ba), &dba); for x in 0..6 { for y in 0..6 { diag[au][x][y] += m[x][y]; } } }
            if !fixed[cu] { let m = mmul(&tr(&bb), &dbb); for x in 0..6 { for y in 0..6 { diag[cu][x][y] += m[x][y]; } } }
            if !fixed[au] && !fixed[cu] {
                let mut m = mmul(&tr(&ba), &dbb); for x in 0..6 { for y in 0..6 { m[x][y] = -m[x][y]; } }
                off.push((a, c, m));
            }
        }
        let mut l = Level::finish(n, diag, off, fixed, pos);
        l.sp = sp.to_vec(); l.ends = ends.to_vec();
        l
    }
    fn finish(n: usize, mut diag: Vec<M6>, off: Vec<(u32, u32, M6)>, fixed: Vec<bool>, pos: Vec<[f64; 3]>) -> Level {
        // (a little added to the diagonal: a cell held only by hinges or infill would be free to turn)
        let mut inv = vec![[[0.0; 6]; 6]; n];
        let mut reg = vec![0.0; n];
        for k in 0..n {
            if fixed[k] { continue; }
            let big = (0..6).map(|i| diag[k][i][i].abs()).fold(0.0, f64::max).max(1.0);
            reg[k] = 1e-7 * big;
            for i in 0..6 { diag[k][i][i] += reg[k]; }
            inv[k] = invert6(diag[k]);
        }
        Level { n, diag, off, fixed, pos, inv, up: Vec::new(), sp: Vec::new(), ends: Vec::new(), reg }
    }
    /// y = K x (fixed cells: y = 0)
    fn apply(&self, x: &[[f64; 6]], y: &mut [[f64; 6]]) {
        if !self.sp.is_empty() {
            for k in 0..self.n { let r = self.reg[k]; for i in 0..6 { y[k][i] = r * x[k][i]; } }
            for (s, &(a, c)) in self.sp.iter().zip(&self.ends) {
                let (a, c) = (a as usize, c as usize);
                let (d, r) = s.strain(&x[a], &x[c]);
                let (f, m) = s.force(d, r);
                // on c: f, and m plus the arm's moment rb x f; on a the opposite with ra
                let (ra, rb) = (&s.ra, &s.rb);
                let mb = [m[0] + rb[1] * f[2] - rb[2] * f[1], m[1] + rb[2] * f[0] - rb[0] * f[2], m[2] + rb[0] * f[1] - rb[1] * f[0]];
                let ma = [m[0] + ra[1] * f[2] - ra[2] * f[1], m[1] + ra[2] * f[0] - ra[0] * f[2], m[2] + ra[0] * f[1] - ra[1] * f[0]];
                for i in 0..3 { y[c][i] += f[i]; y[c][3 + i] += mb[i]; y[a][i] -= f[i]; y[a][3 + i] -= ma[i]; }
            }
            for k in 0..self.n { if self.fixed[k] { y[k] = [0.0; 6]; } }
            return;
        }
        for k in 0..self.n {
            if self.fixed[k] { y[k] = [0.0; 6]; continue; }
            let m = &self.diag[k]; let v = &x[k];
            for i in 0..6 { let mut s = 0.0; for j in 0..6 { s += m[i][j] * v[j]; } y[k][i] = s; }
        }
        for &(a, c, ref m) in &self.off {
            let (a, c) = (a as usize, c as usize);
            let (xa, xc) = (x[a], x[c]);
            for i in 0..6 { let mut s = 0.0; let mut t = 0.0; for j in 0..6 { s += m[i][j] * xc[j]; t += m[j][i] * xa[j]; } y[a][i] += s; y[c][i] += t; }
        }
    }
    fn jacobi(&self, r: &[[f64; 6]], z: &mut [[f64; 6]], w: f64) {
        for k in 0..self.n { if self.fixed[k] { continue; } for i in 0..6 { let mut s = 0.0; for j in 0..6 { s += self.inv[k][i][j] * r[k][j]; } z[k][i] += w * s; } }
    }
    /// the next level: free cells grouped into blocks of `bin` m, each block rigid (K_c = T^T K T, T the rigid
    /// map from the block's centre to each cell); fixed cells stay out (they are not unknowns)
    fn coarsen(&self, bin: f64) -> Level {
        let mut map = crate::Map::default();
        let mut up = vec![NONE; self.n];
        let mut pos: Vec<[f64; 3]> = Vec::new(); let mut cnt: Vec<f64> = Vec::new();
        for k in 0..self.n {
            if self.fixed[k] { continue; }
            let p = self.pos[k];
            let key = (floor64(p[0] / bin) as i64, floor64(p[1] / bin) as i64, floor64(p[2] / bin) as i64);
            let c = *map.entry(key).or_insert_with(|| { pos.push([0.0; 3]); cnt.push(0.0); pos.len() - 1 });
            up[k] = c as u32; for d in 0..3 { pos[c][d] += p[d]; } cnt[c] += 1.0;
        }
        for c in 0..pos.len() { for d in 0..3 { pos[c][d] /= cnt[c]; } }
        let nc = pos.len();
        let t: Vec<M6> = (0..self.n).map(|k| if up[k] == NONE { [[0.0; 6]; 6] } else { let c = up[k] as usize; rigid(&[self.pos[k][0] - pos[c][0], self.pos[k][1] - pos[c][1], self.pos[k][2] - pos[c][2]]) }).collect();
        let mut diag = vec![[[0.0; 6]; 6]; nc];
        let add = |m: &mut M6, a: &M6| { for x in 0..6 { for y in 0..6 { m[x][y] += a[x][y]; } } };
        for k in 0..self.n { if up[k] == NONE { continue; } let m = mmul(&tr(&t[k]), &mmul(&self.diag[k], &t[k])); add(&mut diag[up[k] as usize], &m); }
        let mut pairs: crate::Map<(u32, u32), M6> = crate::Map::default();
        for &(a, c, ref m) in &self.off {
            let (ua, uc) = (up[a as usize], up[c as usize]);
            if ua == NONE || uc == NONE { continue; }
            let q = mmul(&tr(&t[a as usize]), &mmul(m, &t[c as usize]));
            if ua == uc { let qt = tr(&q); add(&mut diag[ua as usize], &q); add(&mut diag[ua as usize], &qt); }
            else if ua < uc { add(pairs.entry((ua, uc)).or_insert([[0.0; 6]; 6]), &q); }
            else { add(pairs.entry((uc, ua)).or_insert([[0.0; 6]; 6]), &tr(&q)); }
        }
        let off: Vec<(u32, u32, M6)> = pairs.into_iter().map(|((a, c), m)| (a, c, m)).collect();
        // (the diagonal already holds the small additions of the level above, carried down)
        let fixed = vec![false; nc];
        let mut l = Level::finish(nc, diag, off, fixed, pos);
        l.up = up;
        l
    }
}
/// One V-cycle on level `l` for residual `r`: smooth, correct on the next level, smooth.
fn vcycle(levels: &[Level], l: usize, r: &[[f64; 6]]) -> Vec<[f64; 6]> {
    let lv = &levels[l];
    let n = lv.n;
    let mut z = vec![[0.0f64; 6]; n];
    let mut kz = vec![[0.0f64; 6]; n];
    let coarsest = l + 1 >= levels.len();
    let sweeps = if coarsest { 30 } else { SMOOTH };
    let mut rr = vec![[0.0f64; 6]; n];
    let mut smooth = |z: &mut Vec<[f64; 6]>, kz: &mut Vec<[f64; 6]>, first: bool| {
        for s in 0..sweeps {
            // (from z = 0 the first residual is r itself)
            if first && s == 0 { lv.jacobi(r, z, OMEGA); continue; }
            lv.apply(z, kz);
            for k in 0..n { for i in 0..6 { rr[k][i] = r[k][i] - kz[k][i]; } }
            lv.jacobi(&rr, z, OMEGA);
        }
    };
    smooth(&mut z, &mut kz, true);
    if coarsest { return z; }
    let nx = &levels[l + 1];
    // restrict the residual: forces summed, their moments about the block's centre added
    lv.apply(&z, &mut kz);
    let mut rc = vec![[0.0f64; 6]; nx.n];
    for k in 0..n {
        let c = nx.up[k]; if c == NONE { continue; }
        let c = c as usize;
        let mut v = [0.0; 6]; for i in 0..6 { v[i] = r[k][i] - kz[k][i]; }
        let d = [lv.pos[k][0] - nx.pos[c][0], lv.pos[k][1] - nx.pos[c][1], lv.pos[k][2] - nx.pos[c][2]];
        for i in 0..3 { rc[c][i] += v[i]; }
        rc[c][3] += v[3] + d[1] * v[2] - d[2] * v[1];
        rc[c][4] += v[4] + d[2] * v[0] - d[0] * v[2];
        rc[c][5] += v[5] + d[0] * v[1] - d[1] * v[0];
    }
    let zc = vcycle(levels, l + 1, &rc);
    // prolong: each cell moves with its block
    for k in 0..n {
        let c = nx.up[k]; if c == NONE { continue; }
        let c = c as usize;
        let d = [lv.pos[k][0] - nx.pos[c][0], lv.pos[k][1] - nx.pos[c][1], lv.pos[k][2] - nx.pos[c][2]];
        let t = [zc[c][3], zc[c][4], zc[c][5]];
        z[k][0] += zc[c][0] + t[1] * d[2] - t[2] * d[1];
        z[k][1] += zc[c][1] + t[2] * d[0] - t[0] * d[2];
        z[k][2] += zc[c][2] + t[0] * d[1] - t[1] * d[0];
        for i in 3..6 { z[k][i] += zc[c][i]; }
    }
    smooth(&mut z, &mut kz, false);
    z
}
fn invert6(mut m: [[f64; 6]; 6]) -> [[f64; 6]; 6] {
    let mut inv = [[0.0f64; 6]; 6];
    for i in 0..6 { inv[i][i] = 1.0; }
    for c in 0..6 {
        let mut p = c; for r in c + 1..6 { if m[r][c].abs() > m[p][c].abs() { p = r; } }
        if m[p][c].abs() < 1e-300 { continue; }
        m.swap(c, p); inv.swap(c, p);
        let d = m[c][c];
        for j in 0..6 { m[c][j] /= d; inv[c][j] /= d; }
        for r in 0..6 { if r != c { let q = m[r][c]; if q != 0.0 { for j in 0..6 { m[r][j] -= q * m[c][j]; inv[r][j] -= q * inv[c][j]; } } } }
    }
    inv
}

#[cfg(test)]
mod analysis_tests {
    use super::*;
    use crate::tower;
    extern crate std;
    fn gravity(b: &Bricks) -> Analysis {
        let set: Vec<u32> = (0..b.len() as u32).filter(|&i| b.alive(i)).collect();
        let load: Vec<f32> = set.iter().map(|&i| mass(b, i) * 9.81).collect();
        let sink: Vec<bool> = set.iter().map(|&i| b.y[i as usize] <= 0).collect();
        analyse(b, &set, &load, &sink, v3(0.0, -1.0, 0.0))
    }
    #[test] fn a_pier_carries_its_load_through_a_fractured_brick() {
        // a 1 m square concrete pier of five 1 m blocks with 2,000 kN on top (2 MPa): with its middle block split
        // into the convex cells of each of the 16 patterns, the load goes down through the cells (their faces on the
        // blocks above and below tile the block's) and stands. The faces between cells are inclined, and on a plane
        // at angle t to the load the shear is sigma sin t cos t, at most sigma / 2: that, against the shear
        // strength, bounds the ratio.
        let build = || { let mut b = Bricks::new(); for k in 0..5 { b.add([0, k * 8, 0], [8, 8, 8], mat::CONCRETE, 0, 1); } b.build_joints(); b };
        let run = |b: &Bricks| {
            let set: Vec<u32> = (0..b.len() as u32).filter(|&i| b.alive(i)).collect();
            let top = set.iter().copied().max_by_key(|&i| b.y[i as usize]).unwrap();
            let load: Vec<f32> = set.iter().map(|&i| mass(b, i) * 9.81 + if i == top { 2.0e6 } else { 0.0 }).collect();
            let sink: Vec<bool> = set.iter().map(|&i| b.y[i as usize] <= 0).collect();
            analyse(b, &set, &load, &sink, v3(0.0, -1.0, 0.0))
        };
        let whole = run(&build());
        let mut worst = 0.0f32; let mut its = 0;
        for p in 0..16u32 {
            let mut b = build();
            // (the pattern comes from the brick's index: pad with dead bricks to reach each)
            let mut k = 0; while crate::brick::frac_cells(b.len() as u32, 1).0 != p { b.add([100 + 2 * k, 0, 100], [2, 2, 2], mat::CONCRETE, 0, 0); k += 1; }
            let m = b.add([0, 16, 0], [8, 8, 8], mat::CONCRETE, 0, 1);
            b.remove(2); for i in 5..m { b.remove(i); }
            b.build_joints();
            let (cells, _) = b.shatter(m);
            assert_eq!(crate::brick::frac_cells(m, 1).0, p);
            let a = run(&b);
            assert!(a.failed.is_empty() && a.cracks.is_empty(), "pattern {p}: {} cells, worst {}", cells.len(), a.max_ratio);
            worst = worst.max(a.max_ratio); its = its.max(a.iterations);
        }
        std::println!("pier under 2,000 kN: whole {:.3}, through each pattern's cells at most {worst:.3} ({its} iterations at most)", whole.max_ratio);
        let bound = 0.5 * 2.1e6 / strength(mat::CONCRETE)[2];
        assert!(worst <= 1.05 * bound.max(whole.max_ratio), "at most sigma / 2 in shear: {bound:.3}");
    }
    #[test] fn cantilever_as_one_brick() {
        for (len, expect) in [(60, true), (16, false)] {
            let mut b = Bricks::new();
            b.add([0, 0, 0], [8, 8, 8], mat::CONCRETE, 0, 0);
            b.add([0, 8, 0], [8 + len, 2, 8], mat::CONCRETE, 0, 0);
            b.build_joints();
            let a = gravity(&b);
            std::println!("cantilever {} m: worst {:.2}, cracks {:?}, failed joints {}", len as f32 * TICK, a.max_ratio, a.cracks.iter().map(|c| (c.at, c.ratio)).collect::<std::vec::Vec<_>>(), a.failed.len());
            assert_eq!(!a.cracks.is_empty() || !a.failed.is_empty(), expect);
        }
    }
    #[test] fn simply_supported_span() {
        // a slab strip 1 m wide bonded to two 1 m blocks (between clamped and simply supported): 7.5 m holds
        // (w L^2 / 8 = 41 kNm per m, 3.9 MPa even simply supported); 22.5 m does not (clamped: 5.9 x 23.5^2 / 12 =
        // 271 kNm, 26 MPa). (15 m, which the load-flow model failed at 15.9 MPa simply supported, holds bonded:
        // 0.81 of strength.)
        for (span, expect) in [(60, false), (180, true)] {
            let mut b = Bricks::new();
            b.add([0, 0, 0], [8, 8, 8], mat::CONCRETE, 0, 0);
            b.add([8 + span, 0, 0], [8, 8, 8], mat::CONCRETE, 0, 0);
            b.add([0, 8, 0], [16 + span, 2, 8], mat::CONCRETE, 0, 0);
            b.build_joints();
            let a = gravity(&b);
            std::println!("span {} m: worst {:.2}", span as f32 * TICK, a.max_ratio);
            assert_eq!(a.max_ratio > 1.0, expect);
        }
    }
    #[test] fn a_slab_on_one_off_centre_column_tips_and_a_centred_one_holds() {
        // a 6 m x 6 m concrete slab (0.25 m) on one 0.5 m column 3 m high. Centred: it stands (the resultant is
        // over the column). At a corner: the slab's weight (212 kN) acts 3.9 m off the column, which can bend at
        // most f b d^2 / 6 = 12 MPa x 0.5^3 / 6 = 250 kNm, e = 1.2 m: it tips. Before the overturning check both
        // stood (the scalar flow carried the slab down the corner column).
        for (off, expect) in [(20, false), (0, true)] {
            let mut b = Bricks::new();
            b.add([0, 0, 0], [80, 2, 80], mat::CONCRETE, 0, 0);          // ground slab (the support)
            b.add([off, 2, off], [4, 24, 4], mat::CONCRETE, 1, 0);         // column
            b.add([0, 26, 0], [48, 2, 48], mat::CONCRETE, 0, 0);          // slab above
            b.build_joints();
            let a = gravity(&b);
            std::println!("column at {} m: worst {:.2}, failed {:?}", off as f32 * TICK, a.max_ratio, a.failed.iter().map(|x| x.1).collect::<std::vec::Vec<_>>());
            assert_eq!(!a.failed.is_empty(), expect, "column at {off}");
        }
    }
    #[test] fn intact_tower_stands() {
        let b = tower::build(12, 1);
        let t = std::time::Instant::now();
        let a = gravity(&b);
        std::println!("tower: {} nodes, worst {:.2}, {} joints and {} cracks over, {} CG iterations, {:.0} ms", a.nodes, a.max_ratio, a.failed.len(), a.cracks.len(), a.iterations, t.elapsed().as_secs_f64() * 1e3);
        assert!(a.failed.is_empty() && a.cracks.is_empty());
    }
}

