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
        _ => [5e6, 5e6, 1e6], // timber / stud partitions
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
    (s[0] * s[1] * s[2]) as f32 * TICK * TICK * TICK * mat::density(b.mat[i as usize])
}

pub struct Flow {
    /// joints of the set and the force through each (N, positive from the min-side brick to the max-side one)
    pub joints: Vec<(u32, f32)>,
    /// worst stress / strength ratio and the joints above 1
    pub max_ratio: f32,
    pub failed: Vec<u32>,
    pub iterations: usize,
}

/// Solve the load flow over `set` (brick indices). `load[k]` is the load (N) on set[k], `sink[k]` marks
/// supports. `dir` is the direction the loads act in (gravity: down). Bricks with no path to a support are
/// left out (they are falling anyway).
pub fn solve(b: &Bricks, set: &[u32], load: &[f32], sink: &[bool], dir: V3) -> Flow {
    let n = set.len();
    let mut local = vec![NONE; b.len()];
    for (k, &i) in set.iter().enumerate() { local[i as usize] = k as u32; }
    // edges (k, l, conductance, joint)
    let mut edges: Vec<(u32, u32, f32, u32)> = Vec::new();
    let mut head = vec![NONE; n]; let mut next: Vec<u32> = Vec::new(); let mut other: Vec<(u32, f32)> = Vec::new();
    for (k, &i) in set.iter().enumerate() {
        let mut j = b.jhead[i as usize];
        while j != NONE {
            let o = b.joint_other(j, i);
            let l = local[o as usize];
            if l != NONE && (l as usize) > k {
                let infill = (b.flags[i as usize] | b.flags[o as usize]) & F_INFILL != 0;
                let c = b.jarea[j as usize] as f32 * TICK * TICK * if infill { 0.02 } else { 1.0 };
                edges.push((k as u32, l, c, j));
                for (x, y) in [(k as u32, l), (l, k as u32)] { other.push((y, c)); next.push(head[x as usize]); head[x as usize] = (other.len() - 1) as u32; }
            }
            j = b.joint_next(j, i);
        }
    }
    // only bricks connected to a support take part
    let mut live = vec![false; n];
    let mut stack: Vec<u32> = (0..n as u32).filter(|&k| sink[k as usize]).collect();
    for &k in &stack { live[k as usize] = true; }
    while let Some(k) = stack.pop() {
        let mut e = head[k as usize];
        while e != NONE { let (o, _) = other[e as usize]; if !live[o as usize] { live[o as usize] = true; stack.push(o); } e = next[e as usize]; }
    }
    // conjugate gradient on L phi = load (Jacobi preconditioned), phi = 0 on supports
    let free = |k: usize| live[k] && !sink[k];
    let mut diag = vec![0.0f32; n];
    for &(k, l, c, _) in &edges { diag[k as usize] += c; diag[l as usize] += c; }
    let apply = |x: &[f32], y: &mut [f32]| {
        for k in 0..n { y[k] = if free(k) { diag[k] * x[k] } else { 0.0 }; }
        for &(k, l, c, _) in &edges {
            let (k, l) = (k as usize, l as usize);
            if free(k) { y[k] -= c * x[l]; }
            if free(l) { y[l] -= c * x[k]; }
        }
    };
    let mut phi = vec![0.0f32; n];
    let mut r: Vec<f32> = (0..n).map(|k| if free(k) { load[k] } else { 0.0 }).collect();
    let mut z: Vec<f32> = (0..n).map(|k| if free(k) && diag[k] > 0.0 { r[k] / diag[k] } else { 0.0 }).collect();
    let mut p = z.clone();
    let mut ap = vec![0.0f32; n];
    let dot = |a: &[f32], b: &[f32]| -> f64 { a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum() };
    let mut rz = dot(&r, &z);
    let r0 = dot(&r, &r).max(1e-30);
    let mut it = 0;
    while it < 2000 {
        apply(&p, &mut ap);
        let pap = dot(&p, &ap);
        if pap <= 0.0 { break; }
        let alpha = (rz / pap) as f32;
        for k in 0..n { phi[k] += alpha * p[k]; r[k] -= alpha * ap[k]; }
        it += 1;
        if dot(&r, &r) < 1e-10 * r0 { break; }
        for k in 0..n { z[k] = if free(k) && diag[k] > 0.0 { r[k] / diag[k] } else { 0.0 }; }
        let rz2 = dot(&r, &z);
        let beta = (rz2 / rz) as f32;
        rz = rz2;
        for k in 0..n { p[k] = z[k] + beta * p[k]; }
    }
    // forces and failure
    let mut out = Flow { joints: Vec::with_capacity(edges.len()), max_ratio: 0.0, failed: Vec::new(), iterations: it };
    for &(k, l, c, j) in &edges {
        if !live[k as usize] { continue; }
        let ju = j as usize;
        // flow from brick k to brick l, then expressed from the joint's min-side brick (ja) to its max side
        let f_kl = c * (phi[k as usize] - phi[l as usize]);
        let f = if b.ja[ju] == set[k as usize] { f_kl } else { -f_kl };
        out.joints.push((j, f));
        let mut axis = V3::ZERO; axis.set(b.jaxis[ju] as usize, 1.0);
        let along = f * axis.dot(dir);            // > 0: the load moves along its own direction here
        let mode = if along > 0.5 * fabs(f) { 0 } else if along < -0.5 * fabs(f) { 1 } else { 2 };
        let area = b.jarea[ju] as f32 * TICK * TICK;
        let (ma, mb) = (b.mat[b.ja[ju] as usize], b.mat[b.jb[ju] as usize]);
        let cap = strength(ma)[mode].min(strength(mb)[mode]);
        let ratio = fabs(f) / (area * cap);
        if ratio > out.max_ratio { out.max_ratio = ratio; }
        if ratio > 1.0 { out.failed.push(j); }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tower;
    extern crate std;

    #[test] fn column_stack_compression() {
        // a 10 m concrete column (0.5 m square) of 1 m blocks on the ground: the bottom joint carries the
        // weight of everything above it
        let mut b = Bricks::new();
        for k in 0..10 { b.add([0, k * 8, 0], [4, 8, 4], mat::CONCRETE, 0, 0); }
        b.build_joints();
        let set: Vec<u32> = (0..10).collect();
        let load: Vec<f32> = set.iter().map(|&i| mass(&b, i) * 9.81).collect();
        let sink: Vec<bool> = set.iter().map(|&i| i == 0).collect();
        let f = solve(&b, &set, &load, &sink, v3(0.0, -1.0, 0.0));
        let w1 = mass(&b, 1) * 9.81;
        for &(j, force) in &f.joints {
            let above = 10 - b.jb[j as usize] as usize;          // bricks above this joint
            assert!((fabs(force) - above as f32 * w1).abs() < 1.0, "joint force {force} vs {}", above as f32 * w1);
        }
        assert!(f.max_ratio < 0.01 && f.failed.is_empty());
    }

    #[test] fn the_tower_stands_and_fails_without_columns() {
        let b = tower::build(12, 1);
        let set: Vec<u32> = (0..b.len() as u32).filter(|&i| b.alive(i)).collect();
        let load: Vec<f32> = set.iter().map(|&i| mass(&b, i) * 9.81).collect();
        let sink: Vec<bool> = set.iter().map(|&i| b.y[i as usize] <= 0).collect();
        let t = std::time::Instant::now();
        let f = solve(&b, &set, &load, &sink, v3(0.0, -1.0, 0.0));
        std::println!("tower under its own weight: worst joint at {:.0}% of strength, {} failed, {} CG iterations, {:.1} ms",
            f.max_ratio * 100.0, f.failed.len(), f.iterations, t.elapsed().as_secs_f64() * 1e3);
        assert!(f.failed.is_empty(), "an intact tower must stand");
    }
}

/// Flexural strength (Pa) for bending of a brick: reinforced concrete gets its rebar.
pub fn flexural(m: u8) -> f32 {
    match m { mat::CONCRETE => 12e6, mat::MASONRY => 0.5e6, mat::STEEL => 250e6, mat::GLASS => 40e6, _ => 20e6 }
}

/// A crack through a brick from bending: split brick `brick` at tick `at` across `axis`.
#[derive(Clone, Copy, Debug)]
pub struct Crack { pub brick: u32, pub axis: usize, pub at: i32, pub ratio: f32 }

/// Bending inside bricks under gravity. Each brick is in equilibrium between its weight (spread over its
/// length) and the vertical forces from its joints (the flow through a joint is a vertical force where the
/// faces meet, spread over the face). The moment at a section across a horizontal axis, divided by the
/// section modulus (width x depth^2 / 6), is the bending stress. Sections are tried on the course grid.
pub fn bending(b: &Bricks, flow: &Flow, sink: &[bool], set: &[u32]) -> Vec<Crack> {
    // bricks resting on a support (the ground) are carried along their whole underside: no span to bend
    let mut on_support = vec![false; b.len()];
    for (k, &i) in set.iter().enumerate() { if sink[k] { on_support[i as usize] = true; } }
    // vertical up-force on each brick from each joint: the load leaving the brick through it
    let mut forces: Vec<Vec<(u32, f32)>> = Vec::new();
    let mut slot = vec![NONE; b.len()];
    for &(j, f) in &flow.joints {
        let ju = j as usize;
        for (i, out) in [(b.ja[ju], f), (b.jb[ju], -f)] {
            if slot[i as usize] == NONE { slot[i as usize] = forces.len() as u32; forces.push(Vec::new()); }
            forces[slot[i as usize] as usize].push((j, out));
        }
    }
    let mut cracks = Vec::new();
    for i in 0..b.len() as u32 {
        let s = slot[i as usize];
        if s == NONE || b.flags[i as usize] & F_INFILL != 0 || on_support[i as usize] { continue; }
        let t = b.tbox(i);
        let w = mass(b, i) * 9.81;
        let course = mat::course(b.mat[i as usize]).max(1);
        for ax in [0usize, 2] {
            let other = 2 - ax;
            let (lo, hi) = (t.min[ax], t.max[ax]);
            if hi - lo < 2 * course { continue; }
            let depth = (t.max[1] - t.min[1]) as f32 * TICK;
            let width = (t.max[other] - t.min[other]) as f32 * TICK;
            let modulus = width * depth * depth / 6.0;
            let cap = flexural(b.mat[i as usize]);
            // each force: (u0, u1, F) along ax, in ticks
            let segs: Vec<(f32, f32, f32)> = forces[s as usize].iter().map(|&(j, f)| {
                let o = b.tbox(b.joint_other(j, i));
                let (u0, u1) = (t.min[ax].max(o.min[ax]) as f32, t.max[ax].min(o.max[ax]) as f32);
                (u0, u1.max(u0), f)
            }).collect();
            // moment of the forces left of a section (support forces spread over their faces, weight over the
            // brick), and the share of the support force left of it
            let left = |sx: f32| -> (f32, f32) {
                let (mut m, mut sup) = (0.0f32, 0.0f32);
                for &(u0, u1, f) in &segs {
                    if u0 >= sx && !(u1 == u0 && u0 == sx) { continue; }
                    let (frac, c) = if u1 > u0 { (((sx.min(u1)) - u0) / (u1 - u0), 0.5 * (u0 + sx.min(u1))) } else { (1.0, u0) };
                    m += f * frac * (sx - c) * TICK;
                    if f > 0.0 { sup += f * frac; }
                }
                let part = (sx - lo as f32) / (hi - lo) as f32;
                m -= w * part * 0.5 * (sx - lo as f32) * TICK;
                (m, sup)
            };
            // a scalar flow balances forces but not moments: the leftover moment is carried where the brick is
            // supported (a clamp), in proportion to the support force there
            let (resid, sup_total) = left(hi as f32);
            let mut best = (0.0f32, 0i32);
            let mut at = lo + course;
            while at < hi {
                let sx = at as f32;
                let (ml, sl) = left(sx);
                let m = ml - if sup_total > 0.0 { resid * sl / sup_total } else { 0.0 };
                let ratio = fabs(m) / modulus / cap;
                if ratio > best.0 { best = (ratio, at); }
                at += course;
            }
            if best.0 > 1.0 { cracks.push(Crack { brick: i, axis: ax, at: best.1, ratio: best.0 }); }
        }
    }
    cracks
}

#[cfg(test)]
mod bending_tests {
    use super::*;
    extern crate std;
    #[test] fn cantilever_breaks_at_the_support_and_a_short_one_holds() {
        for (len, expect) in [(60, true), (16, false)] {
            // a 0.25 m concrete slab strip 1 m wide, sticking out `len` ticks from a 1 m block on the ground
            let mut b = Bricks::new();
            b.add([0, 0, 0], [8, 8, 8], mat::CONCRETE, 0, 0);
            b.add([0, 8, 0], [8 + len, 2, 8], mat::CONCRETE, 0, 0);
            b.build_joints();
            let set = [0u32, 1];
            let load: Vec<f32> = set.iter().map(|&i| mass(&b, i) * 9.81).collect();
            let f = solve(&b, &set, &load, &[true, false], v3(0.0, -1.0, 0.0));
            let c = bending(&b, &f, &[true, false], &set);
            // a 7.5 m cantilever: M = w L^2 / 2 = 5.9 kN/m x 7.5^2 / 2 per m width = 165 kNm, stress 15.9 MPa > 12
            std::println!("cantilever {} m: cracks {:?}", len as f32 * TICK, c.iter().map(|c| (c.at, c.ratio)).collect::<std::vec::Vec<_>>());
            assert_eq!(!c.is_empty(), expect);
            if expect { assert!(c[0].at <= 10, "cracks near the support, at {}", c[0].at); }
        }
    }
}

// ======================================================================================================
// Structural analysis on a cell graph (phase 3 core).
//
// Big bricks are single records in the store, but a 22 m slab must be able to bend and crack inside. For the
// analysis each brick is cut horizontally into cells of at most CELL ticks (2 m); cells of one brick are
// linked through the brick's own section, and joints between bricks become links between the cells that
// share the face. Loads flow over this graph (as in `solve`). Moments are then carried down the flow:
// a link passes on the first moment of all the load that reaches it about its own position, so a cantilever
// collects w L^2 / 2 at its root, and loads arriving from opposite sides cancel. Every link is checked:
//   - joints between bricks: compression / tension / shear on the face, and bending of the face as a section
//   - links inside a brick: shear and bending of the brick's section there; a failure cracks the brick
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
}

struct Node { brick: u32, lo: [i32; 3], hi: [i32; 3], load: f32, sink: bool }
struct Link { a: u32, b: u32, cond: f32, joint: u32, axis: usize, face_lo: [i32; 3], face_hi: [i32; 3] }

/// Stiffness of a link for vertical load (the flow is shear or axial force, the potential a deflection):
/// bearing (vertical link) is axial, E A / L; a horizontal link carries vertical load by bending, 12 E I / L^3
/// = E w d^3 / L^3 (E cancels). Without this, loads would wander sideways through slabs instead of going
/// straight down the column below.
fn stiffness(axis: usize, lo: [i32; 3], hi: [i32; 3], na: &Node, nb: &Node) -> f32 {
    let (u, w) = ((axis + 1) % 3, (axis + 2) % 3);
    let ext = |d: usize| (hi[d] - lo[d]) as f32 * TICK;
    let centre = |n: &Node, d: usize| 0.5 * (n.lo[d] + n.hi[d]) as f32 * TICK;
    let len = fabs(centre(na, axis) - centre(nb, axis)).max(TICK);
    if axis == 1 { ext(u) * ext(w) / len } else {
        let (width, depth) = (ext(2 - axis), ext(1));
        width * depth * depth * depth / (len * len * len)
    }
}

fn cells_of(t: &crate::brick::TBox) -> (Vec<i32>, Vec<i32>) {
    let cut = |lo: i32, hi: i32| { let n = ((hi - lo + CELL - 1) / CELL).max(1); let mut v = vec![lo]; for k in 1..n { v.push(lo + (hi - lo) * k / n); } v.push(hi); v };
    (cut(t.min[0], t.max[0]), cut(t.min[2], t.max[2]))
}

/// The width of plate that carries a moment across a section. The cell graph sends a bay's load through the
/// faces of the cells next to a support, but in a plate the moment spreads sideways about as far as its lever
/// arm (M / F) on each side: the section is the face widened by that, up to the plate's own width.
fn spread(face: f32, m: f32, f: f32, extent: f32) -> f32 {
    let arm = if fabs(f) > 1e-3 { fabs(m / f) } else { 0.0 };
    // at most a column strip: the width of plate that carries a bay's moment to its supports is about half the
    // span between them (flat-slab design: 3.75 m for 7.5 m bays). Uncapped, a section deep in a bay, where the
    // shear F is small and M / F large, was as wide as the whole 22 m slab: 60 kPa of rubble on one bay came out
    // at 0.70 of strength where a 7.5 m span gives 3.7 times.
    (face + 2.0 * arm).min(extent).min(face + STRIP).max(face)
}
const STRIP: f32 = 3.75;

/// Analyse `set` under per-brick loads `load` (N) acting along `dir`, supported at `sink` bricks.
/// Moments are evaluated for vertical loads (gravity).
pub fn analyse(b: &Bricks, set: &[u32], load: &[f32], sink: &[bool], dir: V3) -> Analysis { analyse_at(b, set, load, &[], sink, dir) }
/// As `analyse`, plus loads at points: (index into `set`, x, z in ticks, N). A point load goes to the cell of
/// that brick under the point (a heap of rubble on one bay of a 22 m slab loads that bay, not the whole slab).
pub fn analyse_at(b: &Bricks, set: &[u32], load: &[f32], point: &[(u32, f32, f32, f32)], sink: &[bool], dir: V3) -> Analysis {
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
        let (xs, zs) = if infill { (vec![t.min[0], t.max[0]], vec![t.min[2], t.max[2]]) } else {
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
            nodes.push(Node { brick: i, lo, hi, load: load[k] * share, sink: sink[k] });
        } }
        // links inside the brick
        let (nx, nz) = grid[k];
        let base = first[i as usize];
        for cz in 0..nz { for cx in 0..nx {
            let id = base + (cz * nx + cx) as u32;
            let n = &nodes[id as usize];
            if cx + 1 < nx {
                let (lo, hi) = ([n.hi[0], n.lo[1], n.lo[2]], [n.hi[0], n.hi[1], n.hi[2]]);
                let cond = stiffness(0, lo, hi, &nodes[id as usize], &nodes[id as usize + 1]);
                links.push(Link { a: id, b: id + 1, cond, joint: NONE, axis: 0, face_lo: lo, face_hi: hi });
            }
            if cz + 1 < nz {
                let (lo, hi) = ([n.lo[0], n.lo[1], n.hi[2]], [n.hi[0], n.hi[1], n.hi[2]]);
                let cond = stiffness(2, lo, hi, &nodes[id as usize], &nodes[id as usize + nx]);
                links.push(Link { a: id, b: id + nx as u32, cond, joint: NONE, axis: 2, face_lo: lo, face_hi: hi });
            }
        } }
    }
    // links across joints, cell to cell where the face overlaps
    for (k, &i) in set.iter().enumerate() {
        let mut j = b.jhead[i as usize];
        while j != NONE {
            let o = b.joint_other(j, i);
            if first[o as usize] != NONE && b.ja[j as usize] == i {
                let ax = b.jaxis[j as usize] as usize;
                let infill = (b.flags[i as usize] | b.flags[o as usize]) & F_INFILL != 0;
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
                    let cond = stiffness(ax, lo, hi, p, q);
                    // infill carries its own weight only; a stair flight is an inclined slab, far more flexible
                    // than its stepped bricks suggest, so it does not take floor loads either
                    let mono = (b.flags[i as usize] | b.flags[o as usize]) & crate::brick::F_MONO != 0;
                    let k = if infill { 0.02 } else if mono { 0.05 } else { 1.0 };
                    links.push(Link { a: ia, b: ib, cond: cond * k, joint: j, axis: ax, face_lo: lo, face_hi: hi });
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
    // adjacency
    let mut head = vec![NONE; n]; let mut next = vec![NONE; 2 * links.len()];
    for (l, k) in links.iter().enumerate() {
        next[2 * l] = head[k.a as usize]; head[k.a as usize] = (2 * l) as u32;
        next[2 * l + 1] = head[k.b as usize]; head[k.b as usize] = (2 * l + 1) as u32;
    }
    let other = |e: u32| -> (usize, u32) { let l = (e / 2) as usize; (l, if e % 2 == 0 { links[l].b } else { links[l].a }) };
    // nodes connected to a support
    let mut live = vec![false; n];
    let mut st: Vec<u32> = (0..n as u32).filter(|&k| nodes[k as usize].sink).collect();
    for &k in &st { live[k as usize] = true; }
    while let Some(k) = st.pop() { let mut e = head[k as usize]; while e != NONE { let (_, o) = other(e); if !live[o as usize] { live[o as usize] = true; st.push(o); } e = next[e as usize]; } }
    let free = |k: usize| live[k] && !nodes[k].sink;
    // CG on L phi = load
    let mut diag = vec![0.0f32; n];
    for l in &links { diag[l.a as usize] += l.cond; diag[l.b as usize] += l.cond; }
    let apply = |x: &[f32], y: &mut [f32]| {
        for k in 0..n { y[k] = if free(k) { diag[k] * x[k] } else { 0.0 }; }
        for l in &links { let (a, c) = (l.a as usize, l.b as usize); if free(a) { y[a] -= l.cond * x[c]; } if free(c) { y[c] -= l.cond * x[a]; } }
    };
    let dotp = |a: &[f32], c: &[f32]| -> f64 { a.iter().zip(c).map(|(x, y)| *x as f64 * *y as f64).sum() };
    let mut phi = vec![0.0f32; n];
    let mut r: Vec<f32> = (0..n).map(|k| if free(k) { nodes[k].load } else { 0.0 }).collect();
    let pre = |r: &[f32], z: &mut Vec<f32>| { for k in 0..n { z[k] = if free(k) && diag[k] > 0.0 { r[k] / diag[k] } else { 0.0 }; } };
    let mut z = vec![0.0f32; n]; pre(&r, &mut z);
    let mut p = z.clone(); let mut ap = vec![0.0f32; n];
    let mut rz = dotp(&r, &z); let r0 = dotp(&r, &r).max(1e-30);
    let mut it = 0;
    while it < 3000 {
        apply(&p, &mut ap);
        let pap = dotp(&p, &ap); if pap <= 0.0 { break; }
        let alpha = (rz / pap) as f32;
        for k in 0..n { phi[k] += alpha * p[k]; r[k] -= alpha * ap[k]; }
        it += 1;
        if dotp(&r, &r) < 1e-8 * r0 { break; }
        pre(&r, &mut z);
        let rz2 = dotp(&r, &z); let beta = (rz2 / rz) as f32; rz = rz2;
        for k in 0..n { p[k] = z[k] + beta * p[k]; }
    }
    // flows (from a to b) and moments carried down the flow
    let flow: Vec<f32> = links.iter().map(|l| if live[l.a as usize] { l.cond * (phi[l.a as usize] - phi[l.b as usize]) } else { 0.0 }).collect();
    let centre = |lo: [i32; 3], hi: [i32; 3]| [0.5 * (lo[0] + hi[0]) as f32 * TICK, 0.5 * (lo[2] + hi[2]) as f32 * TICK];
    let mut mlink = vec![[0.0f32; 2]; links.len()];
    if gravity {
        let mut order: Vec<usize> = (0..n).filter(|&k| live[k]).collect();
        order.sort_by(|&x, &y| phi[y].partial_cmp(&phi[x]).unwrap_or(core::cmp::Ordering::Equal));
        for &k in &order {
            let c = centre(nodes[k].lo, nodes[k].hi);
            // first moment of everything arriving here, about this node's centre (own load acts at the centre)
            let (mut s, mut fin) = ([0.0f32; 2], nodes[k].load);
            let mut fout = 0.0f32;
            let mut e = head[k];
            while e != NONE {
                let (l, _) = other(e);
                let f = if e % 2 == 0 { flow[l] } else { -flow[l] };   // outflow from k
                if f < 0.0 {
                    let pl = centre(links[l].face_lo, links[l].face_hi);
                    s[0] += mlink[l][0] + (-f) * (c[0] - pl[0]);
                    s[1] += mlink[l][1] + (-f) * (c[1] - pl[1]);
                    fin += -f;
                } else { fout += f; }
                e = next[e as usize];
            }
            if fout <= 0.0 { continue; }
            let mut e = head[k];
            while e != NONE {
                let (l, _) = other(e);
                let f = if e % 2 == 0 { flow[l] } else { -flow[l] };
                // bearing (a vertical link) takes the load out of the plate: the moment ends at the support,
                // where the plate's own section has been checked on the way in
                if f > 0.0 && links[l].axis != 1 {
                    let pl = centre(links[l].face_lo, links[l].face_hi);
                    let share = f / fout;
                    mlink[l] = [share * s[0] + f * (pl[0] - c[0]), share * s[1] + f * (pl[1] - c[1])];
                }
                e = next[e as usize];
            }
            let _ = fin;
        }
    }
    // checks
    let mut out = Analysis { failed: Vec::new(), cracks: Vec::new(), max_ratio: 0.0, iterations: it, nodes: n, worst_in: (0.0, 0, 0, 0.0, 0.0, 0.0) };
    // joints: sum their sub-links
    let mut jsum: Vec<(u32, f32, [f32; 2], [i32; 3], [i32; 3])> = Vec::new();
    let mut jslot = hashbrown::HashMap::new();
    for (l, k) in links.iter().enumerate() {
        if k.joint == NONE || !live[k.a as usize] { continue; }
        let s = *jslot.entry(k.joint).or_insert_with(|| { jsum.push((k.joint, 0.0, [0.0; 2], k.face_lo, k.face_hi)); jsum.len() - 1 });
        let e = &mut jsum[s];
        e.1 += flow[l]; e.2[0] += mlink[l][0]; e.2[1] += mlink[l][1];
        for d in 0..3 { e.3[d] = e.3[d].min(k.face_lo[d]); e.4[d] = e.4[d].max(k.face_hi[d]); }
    }
    for &(j, f, m, lo, hi) in &jsum {
        let ju = j as usize;
        // the link flow is from the min-side cell to the max-side one: a -> b of the joint
        let ax = b.jaxis[ju] as usize;
        let mut axis = V3::ZERO; axis.set(ax, 1.0);
        let along = f * axis.dot(dir);
        let mode = if along > 0.5 * fabs(f) { 0 } else if along < -0.5 * fabs(f) { 1 } else { 2 };
        let area = b.jarea[ju] as f32 * TICK * TICK;
        let (ma, mb) = (b.mat[b.ja[ju] as usize], b.mat[b.jb[ju] as usize]);
        let infill = (b.flags[b.ja[ju] as usize] | b.flags[b.jb[ju] as usize]) & F_INFILL != 0;
        let cap = strength(ma)[mode].min(strength(mb)[mode]) * connection(b, j)[mode];
        let mut ratio = fabs(f) / (area * cap);
        // bending across a joint between two bricks of one plate (side by side): the face is the section
        let mono = (b.flags[b.ja[ju] as usize] | b.flags[b.jb[ju] as usize]) & crate::brick::F_MONO != 0;
        // a cracked reinforced joint has half its strength left (the steel across the crack)
        let residual = if b.jdmg[ju] > 0 { 0.5 } else { 1.0 };
        if gravity && ax != 1 && !infill && !mono {
            let depth = (hi[1] - lo[1]) as f32 * TICK;
            let mc = if ax == 0 { m[0] } else { m[1] };
            let (ta, tb) = (b.tbox(b.ja[ju]), b.tbox(b.jb[ju]));
            let extent = ((ta.max[2 - ax] - ta.min[2 - ax]).min(tb.max[2 - ax] - tb.min[2 - ax])) as f32 * TICK;
            let width = spread((hi[2 - ax] - lo[2 - ax]) as f32 * TICK, mc, f, extent);
            ratio = ratio.max(fabs(mc) / (width * depth * depth / 6.0) / flexural(ma).min(flexural(mb)));
        }
        ratio /= residual;
        if ratio > out.max_ratio { out.max_ratio = ratio; }
        if ratio > 1.0 { out.failed.push((j, ratio)); }
    }
    if gravity { for (j, r) in overturning(b, &nodes, &links, &live) {
        if r > out.max_ratio { out.max_ratio = r; }
        if r > 1.0 && !out.failed.iter().any(|&(x, _)| x == j) { out.failed.push((j, r)); }
    } }
    // links inside bricks: shear and bending of the brick's own section
    for (l, k) in links.iter().enumerate() {
        if k.joint != NONE || !live[k.a as usize] { continue; }
        let i = nodes[k.a as usize].brick;
        let m = b.mat[i as usize];
        let (u, w) = ((k.axis + 1) % 3, (k.axis + 2) % 3);
        let area = ((k.face_hi[u] - k.face_lo[u]) * (k.face_hi[w] - k.face_lo[w])) as f32 * TICK * TICK;
        let mut ratio = fabs(flow[l]) / (area * strength(m)[2]);
        if gravity {
            let depth = (k.face_hi[1] - k.face_lo[1]) as f32 * TICK;
            let mc = if k.axis == 0 { mlink[l][0] } else { mlink[l][1] };
            let t = b.tbox(i);
            let width = spread((k.face_hi[2 - k.axis] - k.face_lo[2 - k.axis]) as f32 * TICK, mc, flow[l], (t.max[2 - k.axis] - t.min[2 - k.axis]) as f32 * TICK);
            ratio = ratio.max(fabs(mc) / (width * depth * depth / 6.0) / flexural(m));
            if ratio > out.worst_in.0 { out.worst_in = (ratio, i, k.axis, flow[l], mc, width); }
        }
        if ratio > out.max_ratio { out.max_ratio = ratio; }
        if ratio > 1.0 { out.cracks.push(Crack { brick: i, axis: k.axis, at: k.face_lo[k.axis], ratio }); }
    }
    out
}

/// Overturning at each bearing level. The scalar flow sends every load down some path to the ground, but it has
/// no moment balance: a building whose ground storey is gone except one column was carried by that column like a
/// table on one leg. At each horizontal plane where bricks bear on bricks (a storey's floor line), each
/// connected part above it has a weight W whose resultant lies at its load centroid. The supports bearing at
/// that plane can hold it only if the resultant lies inside the area they cover (their convex hull), or
/// outside by no more than they can carry in bending: e = sum(f_flex b d^2 / 6) / W. Beyond that the part tips
/// about its supports: every support joint at that plane fails (ratio = how far out / allowed), the part comes
/// loose and the rigid-body solver tips it. Returns (joint, ratio) for the joints of the planes that fail.
fn overturning(b: &Bricks, nodes: &[Node], links: &[Link], live: &[bool]) -> Vec<(u32, f32)> {
    let n = nodes.len();
    let structural = |l: &Link| {
        let (p, q) = (nodes[l.a as usize].brick as usize, nodes[l.b as usize].brick as usize);
        (b.flags[p] | b.flags[q]) & (F_INFILL | crate::brick::F_MONO) == 0
    };
    // bearing planes: the heights of bearing joints between structural bricks
    let mut planes: Vec<i32> = links.iter().filter(|l| l.axis == 1 && l.joint != NONE && live[l.a as usize] && structural(l)).map(|l| l.face_lo[1]).collect();
    planes.sort_unstable(); planes.dedup();
    let mut out = Vec::new();
    let mut comp = vec![NONE; n];
    for &y in &planes {
        // the parts above the plane (nodes wholly above it, joined by links that stay above it)
        for c in comp.iter_mut() { *c = NONE; }
        let above = |k: usize| live[k] && nodes[k].lo[1] >= y;
        // union-find over links among nodes above
        let mut par: Vec<u32> = (0..n as u32).collect();
        fn find(p: &mut [u32], mut x: u32) -> u32 { while p[x as usize] != x { p[x as usize] = p[p[x as usize] as usize]; x = p[x as usize]; } x }
        for l in links { let (a, c) = (l.a as usize, l.b as usize); if above(a) && above(c) { let (ra, rc) = (find(&mut par, l.a), find(&mut par, l.b)); if ra != rc { par[ra as usize] = rc; } } }
        let mut ncomp = 0u32;
        let mut root_of = vec![NONE; n];
        for k in 0..n { if above(k) { let r = find(&mut par, k as u32) as usize; if root_of[r] == NONE { root_of[r] = ncomp; ncomp += 1; } comp[k] = root_of[r]; } }
        if ncomp == 0 { continue; }
        // weight and its first moments per part
        let mut w = vec![0.0f64; ncomp as usize]; let mut mx = vec![0.0f64; ncomp as usize]; let mut mz = vec![0.0f64; ncomp as usize];
        for k in 0..n { let c = comp[k]; if c == NONE { continue; }
            let nd = &nodes[k]; let (cx, cz) = (0.5 * (nd.lo[0] + nd.hi[0]) as f64, 0.5 * (nd.lo[2] + nd.hi[2]) as f64);
            w[c as usize] += nd.load as f64; mx[c as usize] += nd.load as f64 * cx; mz[c as usize] += nd.load as f64 * cz; }
        // its supports at the plane: bearing links from a node above to one below, carrying load down
        let mut sup: Vec<Vec<usize>> = vec![Vec::new(); ncomp as usize];
        for (li, l) in links.iter().enumerate() {
            if l.axis != 1 || l.joint == NONE || l.face_lo[1] != y || !structural(l) { continue; }
            let (a, c) = (l.a as usize, l.b as usize);
            let up = if comp[c] != NONE && comp[a] == NONE { c } else if comp[a] != NONE && comp[c] == NONE { a } else { continue };
            sup[comp[up] as usize].push(li);
        }
        for c in 0..ncomp as usize {
            if w[c] <= 0.0 || sup[c].is_empty() { continue; }
            let (gx, gz) = (mx[c] / w[c], mz[c] / w[c]);
            // the supports' faces, as corners in ticks, and what they carry in bending (N m)
            let mut pts: Vec<(f64, f64)> = Vec::new();
            let mut mcap = 0.0f64;
            for &li in &sup[c] {
                let l = &links[li];
                for (x, z) in [(l.face_lo[0], l.face_lo[2]), (l.face_hi[0], l.face_lo[2]), (l.face_lo[0], l.face_hi[2]), (l.face_hi[0], l.face_hi[2])] { pts.push((x as f64, z as f64)); }
                let (bx, bz) = ((l.face_hi[0] - l.face_lo[0]) as f32 * TICK, (l.face_hi[2] - l.face_lo[2]) as f32 * TICK);
                let (p, q) = (nodes[l.a as usize].brick as usize, nodes[l.b as usize].brick as usize);
                let f = flexural(b.mat[p]).min(flexural(b.mat[q])) * if b.jdmg[l.joint as usize] > 0 { 0.5 } else { 1.0 };
                // the weaker way of bending the face
                mcap += (f * bx.max(bz) * bx.min(bz) * bx.min(bz) / 6.0) as f64;
            }
            let out_by = outside_hull(&mut pts, gx, gz) * TICK as f64;   // m
            if out_by <= 0.0 { continue; }
            let allowed = mcap / w[c];
            let ratio = (out_by / allowed.max(1e-6)) as f32;
            if ratio <= 1.0 { continue; }
            for &li in &sup[c] { out.push((links[li].joint, ratio)); }
        }
    }
    out.sort_unstable_by_key(|x| x.0); out.dedup_by_key(|x| x.0);
    out
}

/// How far (in the units of the points) (x, z) lies outside the convex hull of `pts`; 0 inside.
fn outside_hull(pts: &mut Vec<(f64, f64)>, x: f64, z: f64) -> f64 {
    pts.sort_by(|a, c| a.partial_cmp(c).unwrap_or(core::cmp::Ordering::Equal)); pts.dedup();
    if pts.len() == 1 { let d = ((pts[0].0 - x).powi(2) + (pts[0].1 - z).powi(2)).sqrt(); return d; }
    let cross = |o: (f64, f64), a: (f64, f64), c: (f64, f64)| (a.0 - o.0) * (c.1 - o.1) - (a.1 - o.1) * (c.0 - o.0);
    let mut h: Vec<(f64, f64)> = Vec::new();
    for &p in pts.iter() { while h.len() >= 2 && cross(h[h.len() - 2], h[h.len() - 1], p) <= 0.0 { h.pop(); } h.push(p); }
    let lower = h.len() + 1;
    for &p in pts.iter().rev().skip(1) { while h.len() >= lower && cross(h[h.len() - 2], h[h.len() - 1], p) <= 0.0 { h.pop(); } h.push(p); }
    h.pop();
    // distance to the hull: 0 if inside (counter-clockwise hull: all edges have the point on their left)
    let seg = |a: (f64, f64), c: (f64, f64)| { let (dx, dz) = (c.0 - a.0, c.1 - a.1); let l2 = dx * dx + dz * dz; let t = if l2 > 0.0 { (((x - a.0) * dx + (z - a.1) * dz) / l2).clamp(0.0, 1.0) } else { 0.0 }; ((a.0 + t * dx - x).powi(2) + (a.1 + t * dz - z).powi(2)).sqrt() };
    if h.len() < 3 { return if h.len() == 2 { seg(h[0], h[1]) } else { 0.0 }; }
    let inside = (0..h.len()).all(|i| cross(h[i], h[(i + 1) % h.len()], (x, z)) >= 0.0);
    if inside { return 0.0; }
    (0..h.len()).map(|i| seg(h[i], h[(i + 1) % h.len()])).fold(f64::MAX, f64::min)
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
        // a 7.5 m slab strip 1 m wide on two blocks: M = w L^2 / 8 = 5.9 x 56 / 8 = 41 kNm per m: 3.9 MPa, holds;
        // a 15 m span (165 kNm, 15.9 MPa) does not
        for (span, expect) in [(60, false), (120, true)] {
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

/// Floor panels under heavy loads (rubble lying on them). The cell graph underestimates the bending of a bay
/// (its moments split over parallel paths), so a loaded bay is checked as in flat-slab design: in each direction
/// the total static moment of the bay, q L^2 / 8 per metre of width, must be carried by the sections over its
/// supports and at midspan together, 2 m_p with m_p = f_flex d^2 / 6 (a free edge: q L^2 / 2 against m_p).
/// L runs between the faces of the supports under the slab that are nearest the load on each side (within
/// 6 m across). A bay over that cracks along its support lines (both directions) and comes down.
/// `point`: (brick, x, z ticks, N). Returns cracks (ratio > 1) and the worst ratio.
pub fn panels(b: &Bricks, point: &[(u32, f32, f32, f32)]) -> (Vec<Crack>, f32) {
    let mut out = Vec::new();
    let mut worst = 0.0f32;
    let mut bricks: Vec<u32> = point.iter().map(|p| p.0).collect();
    bricks.sort_unstable(); bricks.dedup();
    for &i in &bricks {
        let iu = i as usize;
        if b.flags[iu] & (F_INFILL | crate::brick::F_MONO) != 0 { continue; }
        let t = b.tbox(i);
        let depth = (t.max[1] - t.min[1]) as f32 * TICK;
        // plates only: wider than 4 times their depth both ways
        if ((t.max[0] - t.min[0]).min(t.max[2] - t.min[2]) as f32) * TICK < 4.0 * depth { continue; }
        // supports: structural bricks it bears on
        let mut sup: Vec<[i32; 4]> = Vec::new();
        let mut j = b.jhead[iu];
        while j != NONE {
            let o = b.joint_other(j, i); let ot = b.tbox(o);
            if b.jaxis[j as usize] == 1 && ot.max[1] == t.min[1] && b.flags[o as usize] & F_INFILL == 0 {
                sup.push([ot.min[0].max(t.min[0]), ot.min[2].max(t.min[2]), ot.max[0].min(t.max[0]), ot.max[2].min(t.max[2])]);
            }
            j = b.joint_next(j, i);
        }
        let mp = flexural(b.mat[iu]) * depth * depth / 6.0;
        let selfw = mat::density(b.mat[iu]) * depth * 9.81;   // Pa
        let band = 6.0 / TICK;
        // each loaded point picks its bay; points in one bay are checked once
        let mut seen: Vec<[i32; 4]> = Vec::new();
        for p in point.iter().filter(|p| p.0 == i) {
            let (x, z) = (p.1, p.2);
            // x span: supports within the band across (z)
            let near_z = |s: &[i32; 4]| (s[1] as f32) <= z + band && (s[3] as f32) >= z - band;
            let near_x = |s: &[i32; 4]| (s[0] as f32) <= x + band && (s[2] as f32) >= x - band;
            let left = sup.iter().filter(|s| near_z(s) && (s[2] as f32) <= x).map(|s| s[2]).max();
            let right = sup.iter().filter(|s| near_z(s) && (s[0] as f32) >= x).map(|s| s[0]).min();
            let lo = sup.iter().filter(|s| near_x(s) && (s[3] as f32) <= z).map(|s| s[3]).max();
            let hi = sup.iter().filter(|s| near_x(s) && (s[1] as f32) >= z).map(|s| s[1]).min();
            let bay = [left.unwrap_or(t.min[0]), lo.unwrap_or(t.min[2]), right.unwrap_or(t.max[0]), hi.unwrap_or(t.max[2])];
            if seen.contains(&bay) { continue; }
            seen.push(bay);
            let area = ((bay[2] - bay[0]).max(1) * (bay[3] - bay[1]).max(1)) as f32 * TICK * TICK;
            let on: f32 = point.iter().filter(|q| q.0 == i && q.1 >= bay[0] as f32 && q.1 <= bay[2] as f32 && q.2 >= bay[1] as f32 && q.2 <= bay[3] as f32).map(|q| q.3).sum();
            let q = selfw + on / area;
            let ratio_dir = |l: i32, both: bool| { let l = l as f32 * TICK; if both { q * l * l / 8.0 / (2.0 * mp) } else { q * l * l / 2.0 / mp } };
            let rx = ratio_dir(bay[2] - bay[0], left.is_some() && right.is_some());
            let rz = ratio_dir(bay[3] - bay[1], lo.is_some() && hi.is_some());
            let r = rx.max(rz);
            worst = worst.max(r);
            if r <= 1.0 { continue; }
            // it comes down along its support lines
            for (axis, at) in [(0usize, bay[0]), (0, bay[2]), (2, bay[1]), (2, bay[3])] {
                if at > t.min[axis] && at < t.max[axis] { out.push(Crack { brick: i, axis, at, ratio: r }); }
            }
        }
    }
    (out, worst)
}
