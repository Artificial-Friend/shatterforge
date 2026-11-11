//! Spike: can the standing structure be stepped as a dynamic model, in the step of a change, instead of analysed
//! statically a few frames late? (docs/DYNAMIC.md)
//!
//! The network is the one `analyse` solves: rigid cells, a spring per face. Implicit Euler on it is
//!     (K + M / h^2) x' = f + (M / h^2) (x + h v),   v' = (x' - x) / h
//! and M / h^2 is a spring from each cell's centre to the ground (translation m / h^2, rotation I / h^2), so the
//! dynamic step is the static network with one more spring a cell, and both solvers below run on it unchanged:
//!   - `gs`: block Gauss-Seidel, a cell at a time (its 6 x 6 block inverted): what a GPU solves per colour
//!     (vertex block descent on a linear network is this sweep);
//!   - `mg`: conjugate gradients preconditioned by the multigrid of `solve_network`.
extern crate std;
use super::*;
use std::{cell::RefCell, println, time::Instant, vec, vec::Vec};

#[derive(Clone)]
pub struct Net { pub sp: Vec<Spring>, pub ends: Vec<(u32, u32)>, pub fixed: Vec<bool>, pub pos: Vec<[f64; 3]>, pub f: Vec<[f64; 6]>, pub dims: Vec<[f64; 4]> }
std::thread_local! { static NET: RefCell<Option<Net>> = RefCell::new(None); static ON: RefCell<bool> = RefCell::new(false); }
pub fn capture(sp: &[Spring], ends: &[(u32, u32)], fixed: &[bool], pos: &[[f64; 3]], f: &[[f64; 6]], dims: &[[f64; 4]]) {
    if !ON.with(|o| *o.borrow()) { return; }
    NET.with(|n| *n.borrow_mut() = Some(Net { sp: sp.to_vec(), ends: ends.to_vec(), fixed: fixed.to_vec(), pos: pos.to_vec(), f: f.to_vec(), dims: dims.to_vec() }));
}
pub fn network_of(b: &Bricks) -> Net {
    ON.with(|o| *o.borrow_mut() = true);
    let set: Vec<u32> = (0..b.len() as u32).filter(|&i| b.flags[i as usize] & crate::brick::F_ALIVE != 0).collect();
    let load: Vec<f32> = set.iter().map(|&i| mass(b, i) * 9.81).collect();
    let sink: Vec<bool> = set.iter().map(|&i| b.y[i as usize] <= 0).collect();
    let _ = analyse(b, &set, &load, &sink, crate::math::v3(0.0, -1.0, 0.0));
    ON.with(|o| *o.borrow_mut() = false);
    NET.with(|n| n.borrow_mut().take().unwrap())
}

type X = Vec<[f64; 6]>;
fn dot(a: &X, b: &X) -> f64 { let mut s = 0.0; for k in 0..a.len() { for i in 0..6 { s += a[k][i] * b[k][i]; } } s }

/// The levels of a network (finest first), as `solve_network` builds them.
fn levels_of(sp: &[Spring], ends: &[(u32, u32)], fixed: &[bool], pos: &[[f64; 3]]) -> Vec<Level> {
    let mut levels = vec![Level::from_springs(sp, ends, fixed.to_vec(), pos.to_vec())];
    let mut bin = 3.0f64;
    while levels.last().unwrap().n > 64 && levels.len() < 8 {
        let c = levels.last().unwrap().coarsen(bin);
        if c.n as f64 > 0.7 * levels.last().unwrap().n as f64 { bin *= 2.0; if bin > 256.0 { break; } continue; }
        levels.push(c); bin *= 2.0;
    }
    levels
}
/// Multigrid-preconditioned CG from `x`, at most `max_it` iterations or until r.z falls `tol` times from the
/// right-hand side's; returns the iterations.
fn mg(levels: &[Level], b: &X, x: &mut X, max_it: usize, tol: f64) -> usize {
    let n = x.len();
    let mut ap = vec![[0.0f64; 6]; n];
    let fixed = &levels[0].fixed;
    let rhs: X = (0..n).map(|k| if fixed[k] { [0.0; 6] } else { b[k] }).collect();
    let z0 = vcycle(levels, 0, &rhs);
    let r0 = dot(&rhs, &z0).max(1e-30);
    levels[0].apply(x, &mut ap);
    let mut r: X = (0..n).map(|k| { let mut v = rhs[k]; for i in 0..6 { v[i] -= ap[k][i]; } if fixed[k] { [0.0; 6] } else { v } }).collect();
    let mut z = vcycle(levels, 0, &r);
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let mut it = 0;
    while it < max_it && rz > tol * r0 {
        levels[0].apply(&p, &mut ap);
        let pap = dot(&p, &ap); if pap <= 0.0 { break; }
        let alpha = rz / pap;
        for k in 0..n { for i in 0..6 { x[k][i] += alpha * p[k][i]; r[k][i] -= alpha * ap[k][i]; } }
        it += 1;
        z = vcycle(levels, 0, &r);
        let rz2 = dot(&r, &z);
        let beta = rz2 / rz; rz = rz2;
        for k in 0..n { for i in 0..6 { p[k][i] = z[k][i] + beta * p[k][i]; } }
    }
    it
}
/// Cell -> the springs on it.
struct Adj { head: Vec<u32>, next: Vec<u32> }
fn adj_of(n: usize, ends: &[(u32, u32)]) -> Adj {
    let mut head = vec![NONE; n]; let mut next = vec![NONE; 2 * ends.len()];
    for (l, &(a, b)) in ends.iter().enumerate() {
        next[2 * l] = head[a as usize]; head[a as usize] = (2 * l) as u32;
        next[2 * l + 1] = head[b as usize]; head[b as usize] = (2 * l + 1) as u32;
    }
    Adj { head, next }
}
/// `sweeps` block Gauss-Seidel sweeps over the cells, in order then in reverse.
fn gs(lv: &Level, adj: &Adj, b: &X, x: &mut X, sweeps: usize) {
    let n = x.len();
    for s in 0..sweeps {
        for kk in 0..n {
            let k = if s % 2 == 0 { kk } else { n - 1 - kk };
            if lv.fixed[k] { continue; }
            // the residual of cell k: b - (K x)_k, from its springs
            let mut r = b[k];
            for i in 0..6 { r[i] -= lv.reg[k][i / 3] * x[k][i]; }
            let mut e = adj.head[k];
            while e != NONE {
                let l = (e / 2) as usize; let sp = &lv.sp[l]; let (a, c) = lv.ends[l];
                let (d, rr) = sp.strain(&x[a as usize], &x[c as usize]);
                let (f, m) = sp.force(d, rr);
                if e % 2 == 1 { let rb = &sp.rb; let mb = [m[0] + rb[1] * f[2] - rb[2] * f[1], m[1] + rb[2] * f[0] - rb[0] * f[2], m[2] + rb[0] * f[1] - rb[1] * f[0]]; for i in 0..3 { r[i] -= f[i]; r[3 + i] -= mb[i]; } }
                else { let ra = &sp.ra; let ma = [m[0] + ra[1] * f[2] - ra[2] * f[1], m[1] + ra[2] * f[0] - ra[0] * f[2], m[2] + ra[0] * f[1] - ra[1] * f[0]]; for i in 0..3 { r[i] += f[i]; r[3 + i] += ma[i]; } }
                e = adj.next[e as usize];
            }
            for i in 0..6 { let mut s = 0.0; for j in 0..6 { s += lv.inv[k][i][j] * r[j]; } x[k][i] += s; }
        }
    }
}

/// The dynamic network for a step h: the static one plus each cell's inertia as a spring to a fixed anchor (the
/// last cell). Returns it with each cell's M / h^2 (3 translations, 3 rotations).
fn dynamic(net: &Net, h: f64) -> (Net, Vec<[f64; 6]>) {
    let n = net.fixed.len();
    let mut d = net.clone();
    let mut mh = vec![[0.0; 6]; n + 1];
    d.fixed.push(true); d.pos.push([0.0, -100.0, 0.0]); d.f.push([0.0; 6]); d.dims.push([0.0; 4]);
    for k in 0..n {
        if net.fixed[k] { continue; }
        let [a, b, c, w] = net.dims[k]; let m = w / 9.81;
        let i = [m * (b * b + c * c) / 12.0, m * (a * a + c * c) / 12.0, m * (a * a + b * b) / 12.0];
        let (kt, kr) = (m / (h * h), [i[0] / (h * h), i[1] / (h * h), i[2] / (h * h)]);
        mh[k] = [kt, kt, kt, kr[0], kr[1], kr[2]];
        d.sp.push(Spring { c: net.pos[k], ra: [0.0; 3], rb: [0.0; 3], k: [kt; 3], kr, fr: None, frr: None });
        d.ends.push((n as u32, k as u32));
    }
    (d, mh)
}
/// every spring's force (N) for displacements x
fn forces(net: &Net, x: &X) -> Vec<[f64; 3]> { net.sp.iter().zip(&net.ends).map(|(s, &(a, c))| { let (d, r) = s.strain(&x[a as usize], &x[c as usize]); s.force(d, r).0 }).collect() }
fn norm(v: [f64; 3]) -> f64 { (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt() }

#[derive(Clone, Copy, Debug)]
enum Solver { Gs(usize), Mg(usize), Exact }

/// Step the network `steps` frames of 1/60 s (each in `sub` sub-steps) from displacements x0 at rest, with the
/// solver given; returns each frame's displacements and the ms a frame.
fn run(net: &Net, x0: &X, steps: usize, sub: usize, solver: Solver) -> (Vec<X>, f64) {
    let h = 1.0 / 60.0 / sub as f64;
    let (d, mh) = dynamic(net, h);
    let n = net.fixed.len();
    let levels = levels_of(&d.sp, &d.ends, &d.fixed, &d.pos);
    let adj = adj_of(n + 1, &d.ends);
    let mut x: X = x0.clone(); x.push([0.0; 6]);
    let mut v: X = vec![[0.0; 6]; n + 1];
    let mut out = Vec::new();
    let t = Instant::now();
    for _ in 0..steps {
        for _ in 0..sub {
            let mut b = d.f.clone();
            let mut y = x.clone();
            for k in 0..n { for i in 0..6 { y[k][i] = x[k][i] + h * v[k][i]; b[k][i] += mh[k][i] * y[k][i]; } }
            let mut xn = y;
            match solver {
                Solver::Gs(s) => gs(&levels[0], &adj, &b, &mut xn, s),
                Solver::Mg(it) => { mg(&levels, &b, &mut xn, it, 0.0); }
                Solver::Exact => { mg(&levels, &b, &mut xn, 400, 1e-16); }
            }
            for k in 0..n { for i in 0..6 { v[k][i] = (xn[k][i] - x[k][i]) / h; } }
            x = xn;
        }
        out.push(x[..n].to_vec());
    }
    (out, t.elapsed().as_secs_f64() * 1e3 / steps as f64)
}
fn static_solve(net: &Net) -> (X, usize) {
    let levels = levels_of(&net.sp, &net.ends, &net.fixed, &net.pos);
    let mut x = vec![[0.0; 6]; net.fixed.len()];
    let it = mg(&levels, &net.f, &mut x, 2000, 1e-14);
    (x, it)
}
/// The network with a ground-storey column taken out: the cell under the hardest-pressed spring within `ymax` of
/// the ground is cut from everything and held (it is gone). Returns it and a cell beside the cut to watch.
fn cut_column(net: &Net, x: &X, ymax: f64) -> (Net, usize, usize) {
    let fo = forces(net, x);
    let (mut best, mut bl) = (0.0, 0);
    for (l, &(a, c)) in net.ends.iter().enumerate() {
        let (a, c) = (a as usize, c as usize);
        if net.pos[a][1] > ymax || net.pos[c][1] > ymax { continue; }
        if fo[l][1].abs() > best { best = fo[l][1].abs(); bl = l; }
    }
    let (a, c) = net.ends[bl];
    let (gone, watch) = if net.pos[a as usize][1] < net.pos[c as usize][1] { (a as usize, c as usize) } else { (c as usize, a as usize) };
    let gone = if net.fixed[gone] { watch } else { gone };
    let mut d = net.clone();
    let mut probe = watch;
    for (l, &(a, c)) in net.ends.iter().enumerate() {
        if a as usize == gone || c as usize == gone {
            d.sp[l].k = [0.0; 3]; d.sp[l].kr = [0.0; 3];
            let o = if a as usize == gone { c } else { a } as usize;
            if !net.fixed[o] && net.pos[o][1] > net.pos[gone][1] { probe = o; }
        }
    }
    d.fixed[gone] = true; d.f[gone] = [0.0; 6];
    (d, gone, probe)
}

fn report(name: &str, b: &Bricks, ymax: f64, frames: usize) {
    let net = network_of(b);
    let n = net.fixed.len();
    let free = net.fixed.iter().filter(|f| !**f).count();
    let (xs, it) = static_solve(&net);
    let fs = forces(&net, &xs);
    let fmax = fs.iter().map(|f| norm(*f)).fold(0.0, f64::max);
    let dmax = xs.iter().map(|x| norm([x[0], x[1], x[2]])).fold(0.0, f64::max);
    println!("\n== {name}: {n} cells ({free} free), {} springs; static: {it} iterations, largest displacement {:.3} mm, largest spring force {:.0} kN", net.sp.len(), dmax * 1e3, fmax / 1e3);
    let ferr = |net: &Net, x: &X, fr: &[[f64; 3]], scale: f64| -> f64 { forces(net, x).iter().zip(fr).map(|(a, b)| norm([a[0] - b[0], a[1] - b[1], a[2] - b[2]])).fold(0.0, f64::max) / scale };

    // 1. standing: from the static solution at rest, 60 s; how far does it drift, and what does a frame cost?
    println!("-- standing from the static solution, {frames} frames: drift (largest displacement change, um) and force error (largest spring-force change / largest force)");
    for (sub, s) in [(1, Solver::Gs(1)), (6, Solver::Gs(1)), (1, Solver::Mg(1))] {
        let (tr, ms) = run(&net, &xs, frames, sub, s);
        let last = tr.last().unwrap();
        let drift = last.iter().zip(&xs).map(|(a, b)| norm([a[0] - b[0], a[1] - b[1], a[2] - b[2]])).fold(0.0, f64::max);
        println!("   {:?} x {sub} sub-steps: drift {:.4} um, force error {:.2e}, {:.2} ms a frame", s, drift * 1e6, ferr(&net, last, &fs, fmax), ms);
    }

    // 2. from nothing: the building put down unloaded, its weight switched on. How long until the forces are right?
    println!("-- weight switched on at frame 0 (from no displacement): force error against the static solution at frames 1, 6, 30, 120, 600");
    let zero = vec![[0.0; 6]; n];
    let fr = [1usize, 6, 30, 120, 600];
    for (sub, s) in [(1, Solver::Gs(2)), (1, Solver::Gs(8)), (6, Solver::Gs(1)), (6, Solver::Gs(4)), (1, Solver::Mg(1)), (1, Solver::Mg(3)), (1, Solver::Exact)] {
        let (tr, ms) = run(&net, &zero, 600, sub, s);
        let e: Vec<std::string::String> = fr.iter().map(|&k| std::format!("{:.3}", ferr(&net, &tr[k - 1], &fs, fmax))).collect();
        println!("   {:?} x {sub}: {}   ({:.2} ms a frame)", s, e.join("  "), ms);
    }

    // 3. a column taken out: against the dynamic step solved exactly, how wrong are the forces frame by frame, and
    //    how far has the cell above come of the way to its new resting place?
    let (cut, gone, probe) = cut_column(&net, &xs, ymax);
    let (x2, it2) = static_solve(&cut);
    let f2 = forces(&cut, &x2);
    let f2max = f2.iter().map(|f| norm(*f)).fold(0.0, f64::max);
    let sag = x2[probe][1] - xs[probe][1];
    println!("-- column cell {gone} at ({:.1}, {:.1}, {:.1}) m taken out; new static: {it2} iterations, the cell above moves {:.3} mm, largest spring force {:.0} -> {:.0} kN", net.pos[gone][0], net.pos[gone][1], net.pos[gone][2], sag * 1e3, fmax / 1e3, f2max / 1e3);
    let (exact, ms_exact) = run(&cut, &xs, 120, 1, Solver::Exact);
    let fex: Vec<Vec<[f64; 3]>> = exact.iter().map(|x| forces(&cut, x)).collect();
    let peak = fex.iter().map(|f| f.iter().map(|v| norm(*v)).fold(0.0, f64::max)).fold(0.0, f64::max);
    let fr = [1usize, 2, 4, 8, 15, 30, 60, 120];
    let row = |tr: &Vec<X>| -> (std::string::String, std::string::String) {
        let a: Vec<std::string::String> = fr.iter().map(|&k| std::format!("{:.2}", (tr[k - 1][probe][1] - xs[probe][1]) / sag)).collect();
        let b: Vec<std::string::String> = fr.iter().map(|&k| std::format!("{:.3}", ferr(&cut, &tr[k - 1], &fex[k - 1], f2max))).collect();
        (a.join(" "), b.join(" "))
    };
    println!("   frames {:?}; exact dynamic step: peak spring force {:.0} kN ({:.2} of the new static), {:.1} ms a frame", fr, peak / 1e3, peak / f2max, ms_exact);
    let (a, _) = row(&exact);
    println!("   {:<22} sag/static: {a}", "exact");
    for (sub, s) in [(1, Solver::Gs(1)), (1, Solver::Gs(4)), (1, Solver::Gs(16)), (6, Solver::Gs(1)), (6, Solver::Gs(4)), (1, Solver::Mg(1)), (1, Solver::Mg(2)), (1, Solver::Mg(4))] {
        let (tr, ms) = run(&cut, &xs, 120, sub, s);
        let (a, b) = row(&tr);
        println!("   {:<22} sag/static: {a} | force error vs exact: {b} | {:.2} ms", std::format!("{:?} x {sub}", s), ms);
    }
}

#[test] #[ignore]
fn spike_tower() { report("concrete frame, 12 storeys", &crate::tower::build(12, 1), 4.0, 3600); }
#[test] #[ignore]
fn spike_tower4() { report("concrete frame, 4 storeys", &crate::tower::build(4, 1), 4.0, 3600); }
#[test] #[ignore]
fn spike_complex() { report("complex building", &crate::complex::build(), 4.0, 3600); }
