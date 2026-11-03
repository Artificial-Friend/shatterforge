//! Validation of the structural model against textbook answers (beam theory, axial stress).
//! Each case builds a small structure, analyses it under its weight (or a given load) and compares the
//! worst stress ratio the engine reports with the analytic one. `cargo test --release validation -- --nocapture`
//! prints the table.
#![cfg(test)]
extern crate std;
use crate::brick::{mat, Bricks};
use crate::math::*;
use crate::structure::{analyse, flexural, mass, strength, Analysis};
use alloc::vec::Vec;

const G: f32 = 9.81;
fn density(m: u8) -> f32 { match m { mat::CONCRETE => 2400.0, mat::MASONRY => 1900.0, mat::STEEL => 7850.0, mat::TIMBER => 500.0, _ => 2500.0 } }

/// Analyse `b` under weights plus extra loads (brick, N); bricks on the ground are the supports.
fn run(b: &Bricks, extra: &[(u32, f32)]) -> Analysis {
    let set: Vec<u32> = (0..b.len() as u32).filter(|&i| b.alive(i)).collect();
    let mut load: Vec<f32> = set.iter().map(|&i| mass(b, i) * G).collect();
    for &(i, f) in extra { load[i as usize] += f; }
    let sink: Vec<bool> = set.iter().map(|&i| b.y[i as usize] <= 0).collect();
    analyse(b, &set, &load, &sink, v3(0.0, -1.0, 0.0))
}

struct Row { case: std::string::String, engine: f32, theory: f32 }
fn report(rows: &[Row], tol: f32) {
    std::println!("{:<52} {:>8} {:>8} {:>7}", "case", "engine", "theory", "error");
    for r in rows {
        let e = (r.engine - r.theory) / r.theory;
        std::println!("{:<52} {:>8.3} {:>8.3} {:>6.1}%", r.case, r.engine, r.theory, e * 100.0);
    }
    for r in rows { assert!(((r.engine - r.theory) / r.theory).abs() < tol, "{}: engine {:.3}, theory {:.3}", r.case, r.engine, r.theory); }
}

/// A cantilever strip 1 m wide, depth h ticks, sticking out `len` ticks from a 1 m block on the ground.
/// Theory: M = w L^2 / 2 with w = rho g h per metre width, section modulus h^2 / 6: sigma = 3 rho g L^2 / h.
fn cantilever(m: u8, len: i32, h: i32) -> Row {
    let mut b = Bricks::new();
    b.add([0, 0, 0], [8, 8, 8], m, 0, 0);
    b.add([0, 8, 0], [8 + len, h, 8], m, 0, 0);
    b.build_joints();
    let a = run(&b, &[]);
    let (l, d) = (len as f32 * 0.125, h as f32 * 0.125);
    let sigma = 3.0 * density(m) * G * l * l / d;
    Row { case: std::format!("cantilever, {} m long, {:.2} m deep, {}", l, d, name(m)), engine: a.max_ratio, theory: sigma / flexural(m) }
}
/// A strip on two supports `bear` ticks wide, clear span `span` ticks. Theory, with an even pressure under each
/// bearing (half clear span c, bearing width b): the midspan moment is w (c^2 + b c) / 2, so
/// sigma = 3 rho g (c^2 + b c) / h. The engine ends the moment at the bearing faces (w c^2 / 2), which misses the
/// b c part: about b / c low, a few percent for column-width bearings.
fn span(m: u8, span: i32, h: i32, bear: i32) -> Row {
    let mut b = Bricks::new();
    b.add([0, 0, 0], [bear, 8, 8], m, 0, 0);
    b.add([bear + span, 0, 0], [bear, 8, 8], m, 0, 0);
    b.add([0, 8, 0], [2 * bear + span, h, 8], m, 0, 0);
    b.build_joints();
    let a = run(&b, &[]);
    let (c, bw, d) = (span as f32 * 0.0625, bear as f32 * 0.125, h as f32 * 0.125);
    let sigma = 3.0 * density(m) * G * (c * c + bw * c) / d;
    Row { case: std::format!("simple span, {} m clear on {:.2} m bearings, {:.2} m deep, {}", span as f32 * 0.125, bw, d, name(m)), engine: a.max_ratio, theory: sigma / flexural(m) }
}
/// A column of `side` x `side` ticks, `tall` ticks high, with a load P (N) on a cap brick on top.
/// Theory: the base carries P plus its own weight: sigma = (P + W) / A against the compressive strength.
fn column(m: u8, side: i32, tall: i32, p: f32) -> Row {
    let mut b = Bricks::new();
    b.add([0, 0, 0], [side, tall, side], m, 0, 0);
    let cap = b.add([0, tall, 0], [side, 2, side], m, 0, 0);
    b.build_joints();
    let a = run(&b, &[(cap, p)]);
    let s = side as f32 * 0.125;
    let w = (mass(&b, 0) + mass(&b, cap)) * G;
    let sigma = (p + w) / (s * s);
    Row { case: std::format!("column {:.2} m square, {:.0} kN on top, {}", s, p / 1e3, name(m)), engine: a.max_ratio, theory: sigma / strength(m)[0] }
}
fn name(m: u8) -> &'static str { match m { mat::CONCRETE => "concrete", mat::MASONRY => "masonry", mat::TIMBER => "timber", mat::STEEL => "steel", _ => "glass" } }

#[test] fn validation_beams_and_columns() {
    let mut rows = Vec::new();
    for len in [24, 40, 56] { rows.push(cantilever(mat::CONCRETE, len, 2)); }
    rows.push(cantilever(mat::CONCRETE, 40, 4));
    rows.push(cantilever(mat::TIMBER, 40, 2));
    for sp in [48, 96] { rows.push(span(mat::CONCRETE, sp, 2, 2)); }
    rows.push(span(mat::TIMBER, 96, 4, 2));
    rows.push(column(mat::CONCRETE, 4, 24, 5.0e6));
    rows.push(column(mat::MASONRY, 4, 24, 2.0e6));
    rows.push(column(mat::TIMBER, 4, 24, 1.0e6));
    report(&rows, 0.10);
}
