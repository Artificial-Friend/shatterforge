//! The city (kind 5): blocks of lots between streets, each lot a building of one of the other kinds (concrete
//! frame, masonry, mass timber, mixed, the complex building), of its own height, turned to face its street. Each is
//! built where it was designed (`tower::build_kind`, `complex::build`) and set on its lot (`Bricks::place`); they
//! share the ground and nothing else.
use crate::brick::{Bricks, TICK};
use alloc::vec::Vec;

/// A lot: the building on it and where it stands.
#[derive(Clone, Copy, Debug)]
pub struct Lot { pub kind: u8, pub storeys: i32, pub quarter: u8, pub off: [i32; 3] }
/// The city's layout: its lots and the tile (lot and half a street each side, ticks) the engine meshes them by.
#[derive(Clone, Debug, Default)]
pub struct Layout { pub n: i32, pub tile: i32, pub lots: Vec<Lot> }

/// A lot's pitch: the buildings are 22.5 m square (the complex one reaches 14.75 m from its centre with its wing), and
/// the streets between them 14 m or more.
pub const PITCH_M: f32 = 40.0;
pub fn pitch() -> i32 { (PITCH_M / TICK) as i32 }

/// The layout of an n x n city: lot (i, j) centred at ((i - (n-1)/2) pitch, (j - (n-1)/2) pitch). The building on
/// each is drawn from a small fixed sequence (the same city every time), all kinds about equally, the complex one
/// at most once in five lots; its height within what the kind is built to (the concrete frame 3 to 12 storeys,
/// masonry 2 to 5, timber 3 to 6, mixed 3 to 5), its turn one of four.
pub fn layout(n: i32) -> Layout {
    let n = n.clamp(1, 16);
    let mut seed: u32 = 0x5f37_59df;
    let mut rnd = |m: u32| -> u32 { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; seed % m };
    let p = pitch();
    let mut lots = Vec::new();
    let mut complexes = 0u32;
    for j in 0..n { for i in 0..n {
        let k = (j * n + i) as u32;
        // (the first lot is the complex building, and at most one lot in five after it)
        let mut kind = if k == 0 { 4 } else { rnd(5) as u8 };
        if kind == 4 && 5 * complexes > k { kind = rnd(4) as u8; }
        if kind == 4 { complexes += 1; }
        let storeys = match kind { 0 => 3 + rnd(10) as i32, 1 => 2 + rnd(4) as i32, 2 => 3 + rnd(4) as i32, 3 => 3 + rnd(3) as i32, _ => 2 };
        let quarter = rnd(4) as u8;
        let c = |a: i32| (2 * a - (n - 1)) * p / 2;
        lots.push(Lot { kind, storeys, quarter, off: [c(i), 0, c(j)] });
    } }
    Layout { n, tile: p, lots }
}

/// Build the city's bricks from its layout, with joints.
pub fn build_layout(l: &Layout) -> Bricks {
    let mut b = Bricks::new();
    let mut sid: u16 = 0;
    for lot in &l.lots {
        let o = if lot.kind == 4 { crate::complex::build() } else { crate::tower::build_kind(lot.storeys, 1, lot.kind) };
        let top = o.structure.iter().copied().max().unwrap_or(0);
        b.place(&o, lot.off, lot.quarter, sid);
        sid = sid.wrapping_add(top).wrapping_add(1);
    }
    b.build_joints();
    b.log.clear();
    b
}
pub fn build(n: i32) -> Bricks { build_layout(&layout(n)) }

#[cfg(all(test, sf_engine))]   // (run by the engine's tests, which take this file as a fixture: engine/src/lib.rs)
mod tests {
    use super::*;
    extern crate std;
    #[test] fn a_turned_copy_keeps_its_joints() {
        // The complex building set on a lot in each of the four turns: all its bricks and joints, and only its 171
        // pieces of furniture loose. (The free pieces' rotation was turned the other way from their positions: a
        // quarter turn kept 669 of the 1,250 joints, and 287 parts came loose.)
        let o = crate::complex::build();
        for q in 0u8..4 {
            let mut b = Bricks::new(); b.place(&o, [320, 0, -320], q, 0); b.build_joints();
            std::println!("turned {q}: {} bricks of {}, {} joints of {}, {} detached", b.alive_count(), o.alive_count(), b.live_joints, o.live_joints, b.detached().len());
            assert_eq!(b.alive_count(), o.alive_count());
            assert_eq!(b.live_joints, o.live_joints);
            assert_eq!(b.detached().len(), o.detached().len());
        }
    }
    #[test] fn footprints() {
        for kind in 0u8..5 {
            let b = if kind == 4 { crate::complex::build() } else { crate::tower::build_kind(4, 1, kind) };
            let (mut lo, mut hi) = ([i32::MAX; 3], [i32::MIN; 3]);
            for i in 0..b.len() as u32 { if !b.alive(i) { continue; } let t = b.tbox(i); for k in 0..3 { lo[k] = lo[k].min(t.min[k]); hi[k] = hi[k].max(t.max[k]); } }
            let m = |v: [i32; 3]| v.map(|x| x as f32 * TICK);
            std::println!("kind {kind}: {:?} .. {:?} m, {} bricks", m(lo), m(hi), b.alive_count());
        }
    }
    #[test] fn the_city_is_what_it_should_be_and_stands() {
        // Cities of 1, 3 and 5 lots a side: their lots, each building on its own (nothing joins two lots), every
        // brick held up, and the first structure step: nothing fails under its own weight. (Analysed all at once, the
        // 5 x 5 city failed 2 joints, arches of two of its complex buildings at 2.25 and 1.19, in 5.6 s; each lot on
        // its own, Sim::structure_step: none, worst 0.831, in 1.7 s.)
        for n in [1, 3, 5] {
            let l = layout(n);
            let t = std::time::Instant::now();
            let b = build_layout(&l);
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let kinds: std::vec::Vec<(u8, i32, u8)> = l.lots.iter().map(|o| (o.kind, o.storeys, o.quarter)).collect();
            std::println!("city {n} x {n}: {} bricks ({} free pieces), {} joints, built in {ms:.0} ms; lots (kind, storeys, turn) {kinds:?}", b.alive_count(), b.pieces.len(), b.live_joints);
            assert_eq!(l.lots.len(), (n * n) as usize);
            assert!(l.lots.iter().any(|o| o.kind == 4), "the complex building");
            if n >= 3 { for k in 0u8..4 { assert!(l.lots.iter().any(|o| o.kind == k), "every kind: {k}"); } }
            // nothing joins two lots: a joint's bricks are on the same lot
            let lot = |i: u32| { let c = b.centroid(i); let f = |v: f64| ((v / PITCH_M as f64) + n as f64 / 2.0).floor() as i32; (f(c[0]), f(c[2])) };
            for j in 0..b.ja.len() { let (a, c) = (b.ja[j], b.jb[j]); if a == crate::brick::NONE { continue; } assert_eq!(lot(a), lot(c), "a joint across lots"); }
            // the loose contents of the complex building are its only detached bricks
            let det = b.detached();
            assert!(det.iter().all(|d| d.iter().all(|&i| b.flags[i as usize] & crate::brick::F_LOOSE != 0)), "only loose contents are detached");
            let mut g = crate::tower::new_kind(n, 5);
            let t = std::time::Instant::now();
            let f = g.structure_step(64);
            let t1 = t.elapsed().as_secs_f64() * 1e3;
            assert_eq!(f, 0, "nothing fails under its own weight");
            // a second step, wear on (as in the page): the lots with nothing changing and nothing wearing are not
            // analysed again
            g.struct_dt = 0.25;
            let t = std::time::Instant::now(); g.structure_step(64); let t2 = t.elapsed().as_secs_f64() * 1e3;
            let t = std::time::Instant::now(); g.structure_step(64); let t3 = t.elapsed().as_secs_f64() * 1e3;
            let on = g.active.iter().filter(|&&a| a).count();
            std::println!("  structure steps: first {t1:.0} ms (worst ratio {:.3}, {f} failed), then {t2:.0} and {t3:.0} ms with {on} of {} lots active; {} mesh chunks", g.max_ratio, g.active.len(), g.chunks.len());
        }
    }
    #[test] #[ignore] fn bigger_cities_cost() {
        // (by hand: cargo test --release bigger_cities -- --ignored --nocapture)
        for n in [5, 8] {
            let t = std::time::Instant::now();
            let mut g = crate::tower::new_kind(n, 5);
            let built = t.elapsed().as_secs_f64();
            let t = std::time::Instant::now(); let f = g.structure_step(64); let s1 = t.elapsed().as_secs_f64();
            std::println!("city {n} x {n}: {} bricks, {} B, game made in {built:.1} s (meshes included), first structure step {s1:.1} s, {f} failed, worst {:.3}", g.b.alive_count(), (g.b.total_bytes_per_brick() * g.b.alive_count() as f32) as u64, g.max_ratio);
        }
    }
}
