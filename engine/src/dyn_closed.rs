//! The dynamic step against closed forms (docs/DYNPLAN.md, stage 1), as `validation` has for the solve at rest.
//! Each case is a few bricks stepped by `structure::analyse_dyn` directly: no game, no failures applied.
//! `cargo test --release dyn_closed -- --nocapture` prints the numbers.
extern crate std;
use crate::brick::{mat, Bricks, TICK};
use crate::math::*;
use crate::structure::{analyse_dyn, mass, modulus_of, Analysis, Dyn, Warm, DYN_TURN};
use alloc::vec::Vec;
use std::println;

struct Rig { b: Bricks, set: Vec<u32>, sink: Vec<bool>, warm: Warm, d: Dyn, t: f64, cells: Vec<([i32; 5], [f64; 3], [f64; 6])> }
impl Rig {
    /// `fixed`: the bricks held (the supports)
    fn new(b: Bricks, fixed: &[u32]) -> Rig {
        let set: Vec<u32> = (0..b.len() as u32).filter(|&i| b.alive(i)).collect();
        let sink = set.iter().map(|i| fixed.contains(i)).collect();
        let mut d = Dyn::default(); d.want_cells = true; d.max_it = 400;
        Rig { b, set, sink, warm: Warm::default(), d, t: 0.0, cells: Vec::new() }
    }
    /// one step of `h` s (0: the solve at rest) under `g` m/s^2 on every brick's mass plus `extra` (brick, N)
    fn step(&mut self, h: f64, g: f32, extra: &[(u32, f32)]) -> Analysis {
        let mut load: Vec<f32> = self.set.iter().map(|&i| mass(&self.b, i) * g).collect();
        for &(i, f) in extra { load[self.set.iter().position(|&s| s == i).unwrap()] += f; }
        self.d.h = h; self.t += h;
        let a = analyse_dyn(&self.b, &self.set, &load, &[], &self.sink, v3(0.0, -1.0, 0.0), Some(&mut self.warm), Some(&mut self.d));
        self.cells = a.cells.clone();
        a
    }
    /// the cell farthest along x: how far down it is (m) and its turn about z (rad)
    fn tip(&self) -> (f64, f64) { let c = self.cells.iter().max_by(|a, b| a.1[0].partial_cmp(&b.1[0]).unwrap()).unwrap(); (-c.2[1], c.2[5]) }
    fn of(&self, brick: u32) -> [f64; 6] { self.cells.iter().find(|c| c.0[0] == brick as i32).unwrap().2 }
}
/// a strip 1 m wide and `deep` ticks deep of `n` bricks `step` ticks long, the first one held: a cantilever from
/// that brick's centre (a cell is rigid, and its springs carry the half lengths each side of a face)
fn cantilever(m: u8, n: i32, step: i32, deep: i32) -> Rig {
    let mut b = Bricks::new();
    for k in 0..n { b.add([k * step, 8, 0], [step, deep, 8], m, 0, 0); }
    b.build_joints();
    Rig::new(b, &[0])
}
/// the turning points of a series, each as (time, value), a parabola through the three samples round it
fn peaks(s: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    for i in 1..s.len() - 1 {
        let (a, b, c) = (s[i - 1].1, s[i].1, s[i + 1].1);
        if (b - a) * (c - b) < 0.0 || (b != a && c == b) {
            let den = a - 2.0 * b + c;
            let o = if den != 0.0 { 0.5 * (a - c) / den } else { 0.0 };
            out.push((s[i].0 + o * (s[i].0 - s[i - 1].0), b - 0.25 * (a - c) * o));
        }
    }
    out
}

const H: f64 = 1.0 / 60.0;
const G: f32 = 9.81;
// the strip of the first cases: concrete, 0.25 m deep, 31 bricks of 0.25 m, 7.5 m from the held brick's centre to
// the last one's
fn strip() -> (Rig, f64, f64, f64) {
    let r = cantilever(mat::CONCRETE, 31, 2, 2);
    let (l, dp) = (30.0 * 2.0 * TICK as f64 + TICK as f64, 2.0 * TICK as f64);   // to the tip's end: 7.625 m
    let e = modulus_of(&r.b, 1);
    let rho = mass(&r.b, 1) as f64 / (2.0 * TICK as f64 * dp * 1.0);
    // first period of a cantilever: 2 pi / (1.875^2 sqrt(E I / (rho A L^4))), I / A = d^2 / 12
    let t1 = 2.0 * core::f64::consts::PI / (1.8751f64.powi(2) * (e * dp * dp / (12.0 * rho * l.powi(4))).sqrt());
    // deflection under its weight at the last brick's centre, x from the held end: w x^2 (6 L^2 - 4 L x + x^2) / (24 E I)
    let x = l - TICK as f64;
    let ds = rho * G as f64 * dp * x * x * (6.0 * l * l - 4.0 * l * x + x * x) / (24.0 * e * dp.powi(3) / 12.0);
    (r, t1, ds, l)
}

#[test] fn a_step_of_100_s_is_the_solve_at_rest() {
    let (mut s, _, ds, _) = strip();
    let a0 = s.step(0.0, G, &[]);
    let (d0, r0) = (s.tip().0, a0.max_ratio as f64);
    let (mut r, ..) = strip();
    r.step(0.0, 0.0, &[]);
    let a1 = r.step(100.0, G, &[]);
    let (d1, r1) = (r.tip().0, a1.max_ratio as f64);
    println!("cantilever 7.6 m under its weight: tip {:.3} mm at rest (beam theory {:.3} mm), {:.3} mm after one step of 100 s ({:+.3}%); worst joint {r0:.4} and {r1:.4} ({:+.3}%); {} iterations", d0 * 1e3, ds * 1e3, d1 * 1e3, (d1 / d0 - 1.0) * 100.0, (r1 / r0 - 1.0) * 100.0, r.d.its);
    // (the solve at rest held every cell to the ground by a weak spring, `Level::finish`: 53.1 mm here, 15% under
    // beam theory, and the worst joint 1.152 for 1.313)
    assert!((d0 / ds - 1.0).abs() < 0.03, "the solve at rest is beam theory's: {d0} / {ds}");
    assert!((d1 / d0 - 1.0).abs() < 1e-3 && (r1 / r0 - 1.0).abs() < 1e-3, "within 0.1%");
}

#[test] fn a_load_let_go_peaks_at_twice_its_static_force_and_rings_at_the_beams_period() {
    // the strip is held level without weight, and its weight is let go: it swings about where it would rest, to
    // twice that (less what the material's damping takes in half a period), at the cantilever's first period
    let (mut s, t1, _, _) = strip();
    let a0 = s.step(0.0, G, &[]);
    let (ds, rs) = (s.tip().0, a0.max_ratio as f64);
    for (h, name) in [(H, "1/60"), (H / 4.0, "1/240")] {
        let (mut r, ..) = strip();
        r.step(0.0, 0.0, &[]);
        let (mut series, mut rmax, mut capped) = (Vec::new(), 0.0f64, 0);
        series.push((0.0, 0.0));
        while r.t < 2.6 * t1 { let a = r.step(h, G, &[]); series.push((r.t, r.tip().0)); rmax = rmax.max(a.max_ratio as f64); capped += r.d.capped as usize; }
        let p = peaks(&series);
        let top: Vec<&(f64, f64)> = p.iter().filter(|p| p.1 > ds).collect();
        let (peak, period) = (top[0].1 / ds, top[1].0 - top[0].0);
        println!("weight let go, step {name} s: tip peaks at {peak:.3} of its rest deflection, worst joint at {:.3} of its rest ratio; period {period:.4} s, theory {t1:.4} s ({:+.1}%); {capped} steps at the cap", rmax / rs, (period / t1 - 1.0) * 100.0);
        assert!(peak >= 1.8 && peak <= 2.0, "peak 1.8 to 2.0: {peak}");
        // (the moment at the held end is not one mode's: the first carries 0.89 of it at rest, the others ring
        // faster than the step resolves and are damped, so the end sees less than twice. The force of one mode
        // is checked on the block below.)
        assert!(rmax / rs >= 1.7 && rmax / rs <= 2.0, "and the held end's moment with it: {}", rmax / rs);
        assert!((period / t1 - 1.0).abs() < 0.05, "period within 5%: {period} / {t1}");
    }
}

#[test] fn without_damping_a_ringing_strip_keeps_its_energy() {
    // 10 s of ringing with the material's damping off: what is lost is the step's own (the generalised-alpha step
    // damps what it cannot resolve, `DYN_RHO`), under 5% a period, and it never gains
    let (mut r, t1, ..) = strip();
    r.d.undamped = true;
    r.step(0.0, 0.0, &[]);
    r.step(100.0, G, &[]);
    let at_rest = -0.5 * r.d.work;   // the energy at rest under the load, from where it was let go
    let (mut r, ..) = strip();
    r.d.undamped = true;
    r.step(0.0, 0.0, &[]);
    let mut e = Vec::new();
    while r.t < 10.0 { r.step(H, G, &[]); e.push(r.d.energy - at_rest); }
    let e0 = -at_rest;
    let per = (t1 / H).round() as usize;
    let (mut gain, mut loss) = (0.0f64, 0.0f64);
    let mut wobble = 0.0f64;
    for i in 1..e.len() { wobble = wobble.max((e[i] - e[i - 1]) / e0); }
    for i in per..e.len() { loss = loss.max(1.0 - e[i] / e[i - per]); gain = gain.max(e[i] / e[i - per] - 1.0); }
    println!("ringing undamped for 10 s ({:.1} periods): energy of the swing {:.1} J let go, {:.1} J after the first step, {:.1} J after a period, {:.1} J after 10 s; from one step to the next it rises by {:.2e} of it at most; over a period it loses {:.2}% at most and gains {:.2e} at most", 10.0 / t1, e0, e[0], e[per], e[e.len() - 1], wobble, loss * 100.0, gain);
    // (measured over a period: at a step's end the energy wobbles by a few parts in 10,000 with the phase of the
    // swing, the step's own energy being another quadratic form; and the first step, implicit Euler, takes what
    // the sudden load put into motions the step cannot follow)
    assert!(gain <= 0.0, "never gains over a period: {gain}");
    assert!(loss < 0.05, "loses under 5% a period: {loss}");
    assert!(e[0] / e0 > 0.9 && e[e.len() - 1] / e0 > 0.8);
}

#[test] fn a_beam_hinged_at_one_end_and_let_go_turns_as_a_falling_rod() {
    // a timber floor strip 4 m long bears on a concrete block at one end (a hinge: structure::hinge) and nothing
    // else: let go, it turns about the hinge as a rod about its end, at 3 g / 2 L, and passes `DYN_TURN` at
    // sqrt(4 DYN_TURN L / 3 g)
    let mut b = Bricks::new();
    b.add([0, 0, 0], [8, 16, 8], mat::CONCRETE, 0, 0);
    b.add([8, 10, 0], [32, 2, 8], mat::TIMBER, 8, 0);
    b.build_joints();
    let l = 32.0 * TICK as f64;
    let theory = (4.0 * DYN_TURN * l / (3.0 * G as f64)).sqrt();
    for (h, name) in [(H / 4.0, "1/240"), (H, "1/60")] {
        let mut r = Rig::new(b.clone(), &[0]);
        r.step(0.0, 0.0, &[]);
        let (mut last, mut at, mut watched) = (0.0, 0.0, 0.0);
        while r.t < 0.25 {
            r.step(h, G, &[]);
            let turn = r.tip().1.abs();
            if at == 0.0 && turn >= DYN_TURN { at = r.t - h * (turn - DYN_TURN) / (turn - last); }
            if watched == 0.0 && !r.d.beyond.is_empty() { watched = r.t; }
            last = turn;
            if at > 0.0 && watched > 0.0 { break; }
        }
        println!("hinged strip let go, step {name} s: turned {DYN_TURN} rad at {:.1} ms, theory {:.1} ms ({:+.1}%); watched from the step ending at {:.1} ms", at * 1e3, theory * 1e3, (at / theory - 1.0) * 100.0, watched * 1e3);
        if h < H { assert!((at / theory - 1.0).abs() < 0.10, "within 10%: {at} / {theory}"); }
        // (at 1/60 s the turn is passed in the 4th step: within a step of theory)
        else { assert!((at - theory).abs() < h, "within a step: {at} / {theory}"); }
        assert!(watched > 0.0 && watched - at < h * 1.001, "and it is watched from that step");
    }
}

#[test] fn a_mass_arriving_at_speed_peaks_as_a_mass_dropped() {
    // a concrete block of 4.8 t on the end of a timber strip 7 m long, which hangs level without weight: one mass on
    // one spring, 3 Hz. Its weight let go (a drop from no height) takes it to twice its rest deflection, and the
    // strip's worst joint to twice its rest force. The block's
    // weight comes on with the speed of a fall from height h (as a load of m v / step in the first step, which is
    // how a hit reaches the structure: DYNPLAN stage 5): the strip's end goes down by
    // delta (1 + sqrt(1 + 2 h / delta)), delta its deflection at rest under the block.
    // (g is 1 m/s^2 here, to stay in the linear range: `DYN_TURN`.)
    let build = || {
        let mut b = Bricks::new();
        for k in 0..14 { b.add([k * 4, 8, 0], [4, 4, 8], mat::TIMBER, 0, 0); }
        let m = b.add([48, 12, 0], [8, 16, 8], mat::CONCRETE, 0, 0);
        b.build_joints();
        (Rig::new(b, &[0]), m)
    };
    let g = 1.0f64;
    let (mut s, m) = build();
    let w = mass(&s.b, m) as f64 * g;
    let a0 = s.step(0.0, 0.0, &[(m, w as f32)]);
    let (delta, rs) = (-s.of(m)[1], a0.max_ratio as f64);
    let beam: f64 = (1..14).map(|i| mass(&s.b, i) as f64).sum();
    for ratio in [0.0, 4.0, 12.0] {
        let drop = 0.5 * ratio * delta;
        let v = (2.0 * g * drop).sqrt();
        let theory = 1.0 + (1.0 + ratio).sqrt();
        for (h, name) in [(H, "1/60"), (H / 4.0, "1/240")] {
            let (mut r, m) = build();
            r.step(0.0, 0.0, &[]);
            let (mut peak, mut rmax) = (0.0f64, 0.0f64);
            let mut first = true;
            while r.t < 1.5 {
                let kick = if first { mass(&r.b, m) as f64 * v / h } else { 0.0 };
                first = false;
                let a = r.step(h, 0.0, &[(m, (w + kick) as f32)]);
                peak = peak.max(-r.of(m)[1]); if !first || kick == 0.0 { rmax = rmax.max(a.max_ratio as f64); }
            }
            println!("block of {:.0} kg on a strip of {beam:.0} kg, rest deflection {:.3} mm, dropped from {:.2} mm ({v:.3} m/s), step {name} s: peak {:.3} of the rest deflection, theory {theory:.3} ({:+.1}%); worst joint {:.3} of its rest force", w / g, delta * 1e3, drop * 1e3, peak / delta, (peak / delta / theory - 1.0) * 100.0, rmax / rs);
            if ratio == 0.0 { assert!(peak / delta >= 1.8 && peak / delta <= 2.0 && rmax / rs >= 1.8 && rmax / rs <= 2.0, "a load let go: 1.8 to 2.0 in deflection and in force: {} {}", peak / delta, rmax / rs); }
            assert!((peak / delta / theory - 1.0).abs() < 0.10, "within 10%: {} / {theory}", peak / delta);
        }
    }
}
