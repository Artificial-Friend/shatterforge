//! The level: the Quarter. Nine lots, three by three, 40 m apart, each a building of its own kind and material,
//! modelled on a real one or a real type. They share the ground and nothing else: the engine solves each lot on its
//! own (`sim::Tiles`).
//!
//! ```text
//!   north (+z)
//!   temple (marble, dry stone)      office tower (concrete frame)        tenement (load-bearing brick)
//!   aqueduct (stone arches)         market hall (braced steel, arcade)   mill, chimney, water tower (brick, steel)
//!   white villa (concrete, render)  glass house (steel, glass)           barn (timber frame)
//!   west (-x)                                                            east (+x)
//! ```
use crate::brick::{Bricks, TICK};
use alloc::vec::Vec;

pub const PITCH_M: f32 = 40.0;
pub const N: i32 = 3;
pub fn pitch() -> i32 { (PITCH_M / TICK) as i32 }

/// What stands on a lot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum What { Villa, GlassHouse, Barn, Aqueduct, MarketHall, Mill, Chimney, WaterTower, Temple, Office, Tenement }
/// A thing on a lot: what, the lot (i east, j north), where its design origin goes from the lot's centre (m), its turn
/// (quarter turns about the vertical).
#[derive(Clone, Copy, Debug)]
pub struct Placed { pub what: What, pub lot: (i32, i32), pub at: (f32, f32), pub quarter: u8 }
pub const OFFICE_STOREYS: i32 = 10;
pub const MILL_STOREYS: i32 = 4;
pub const TENEMENT_STOREYS: i32 = 4;
pub fn layout() -> Vec<Placed> {
    use What::*;
    alloc::vec![
        Placed { what: Villa, lot: (0, 0), at: (0.0, 0.0), quarter: 2 },
        Placed { what: GlassHouse, lot: (1, 0), at: (3.375, 4.5), quarter: 0 },
        Placed { what: Barn, lot: (2, 0), at: (0.0, 0.0), quarter: 1 },
        Placed { what: Aqueduct, lot: (0, 1), at: (0.0, 0.0), quarter: 0 },
        Placed { what: MarketHall, lot: (1, 1), at: (0.0, -2.5), quarter: 0 },
        Placed { what: Mill, lot: (2, 1), at: (-3.0, 0.0), quarter: 0 },
        Placed { what: Chimney, lot: (2, 1), at: (13.0, -12.0), quarter: 0 },
        Placed { what: WaterTower, lot: (2, 1), at: (13.5, 10.0), quarter: 0 },
        Placed { what: Temple, lot: (0, 2), at: (0.0, 0.0), quarter: 0 },
        Placed { what: Office, lot: (1, 2), at: (0.0, 0.0), quarter: 0 },
        Placed { what: Tenement, lot: (2, 2), at: (0.0, 0.0), quarter: 1 },
    ]
}
/// A lot's centre (m).
pub fn lot_centre(lot: (i32, i32)) -> (f32, f32) { ((lot.0 as f32 - (N - 1) as f32 / 2.0) * PITCH_M, (lot.1 as f32 - (N - 1) as f32 / 2.0) * PITCH_M) }
/// Where a placed thing's design origin is in the world (m).
pub fn origin(p: &Placed) -> (f32, f32) { let c = lot_centre(p.lot); (c.0 + p.at.0, c.1 + p.at.1) }
/// The lot's index as the engine counts its tiles (`sim::Tiles::tile_of`).
pub fn lot_index(lot: (i32, i32)) -> i32 { lot.1 * N + lot.0 }
/// A point of a placed thing's design (m) in the world: turned with it and moved to its place.
pub fn to_world(p: &Placed, x: f32, y: f32, z: f32) -> [f32; 3] {
    let (c, s) = [(1.0f32, 0.0f32), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)][(p.quarter & 3) as usize];
    let o = origin(p);
    [o.0 + c * x - s * z, y, o.1 + s * x + c * z]
}
pub fn bricks_of(w: What) -> Bricks {
    match w {
        What::Villa => crate::savoye::build(),
        What::GlassHouse => crate::farnsworth::build(),
        What::Barn => crate::barn::build(),
        What::Aqueduct => crate::aqueduct::build(),
        What::MarketHall => crate::complex::build(),
        What::Mill => crate::tower::build_kind(MILL_STOREYS, 1, 3),
        What::Chimney => crate::works::chimney(),
        What::WaterTower => crate::works::water_tower(),
        What::Temple => crate::temple::build(),
        What::Office => crate::tower::build_kind(OFFICE_STOREYS, 1, 0),
        What::Tenement => crate::tower::build_kind(TENEMENT_STOREYS, 1, 1),
    }
}
/// The quarter's bricks, with joints.
pub fn build() -> Bricks {
    let mut b = Bricks::new();
    let mut sid: u16 = 0;
    for p in layout() {
        let o = bricks_of(p.what);
        let top = o.structure.iter().copied().max().unwrap_or(0);
        let at = origin(&p);
        b.place(&o, [(at.0 / TICK) as i32, 0, (at.1 / TICK) as i32], p.quarter, sid);
        sid = sid.wrapping_add(top).wrapping_add(1);
    }
    b.build_joints();
    b.log.clear();
    b
}
/// The level's simulation.
pub fn new() -> crate::sim::Sim {
    let b = build();
    let top = (0..b.len() as u32).filter(|&i| b.alive(i)).map(|i| b.tbox(i).max[1]).max().unwrap_or(0);
    crate::sim::Sim::from_bricks(b, crate::sim::Tiles { tile: pitch(), n: N, slices: top / crate::sim::BAND + 2 })
}
/// Where the player starts: on the crossing south-east of the market hall, looking at it.
pub fn start() -> crate::math::V3 { crate::math::v3(20.0, 0.26, -20.0) }

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    #[test] fn the_quarter_is_what_it_should_be_and_stands() {
        let t = std::time::Instant::now();
        let b = build();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let free = (0..b.len() as u32).filter(|&i| b.alive(i) && b.piece(i).is_some()).count();
        std::println!("the quarter: {} bricks ({free} free pieces), {} joints, built in {ms:.0} ms", b.alive_count(), b.live_joints);
        for p in layout() { let o = bricks_of(p.what); let free = (0..o.len() as u32).filter(|&i| o.alive(i) && o.piece(i).is_some()).count(); std::println!("  {:?}: {} bricks ({free} free pieces), {} joints", p.what, o.alive_count(), o.live_joints); }
        // each thing's own count of bricks is in it: nothing lost or doubled in placing
        let sum: usize = layout().iter().map(|p| bricks_of(p.what).alive_count()).sum();
        assert_eq!(b.alive_count(), sum);
        // everything on its own lot, inside its tile; nothing joins two lots
        let half = PITCH_M as f64 * N as f64 / 2.0;
        let lot = |i: u32| { let c = b.centroid(i); let f = |v: f64| ((v + half) / PITCH_M as f64).floor() as i32; (f(c[0]), f(c[2])) };
        for i in 0..b.len() as u32 { if !b.alive(i) { continue; } let (lo, hi) = b.box_m(i); let _ = (lo, hi); let l = lot(i); assert!(l.0 >= 0 && l.0 < N && l.1 >= 0 && l.1 < N, "a brick off the map"); }
        for j in 0..b.ja.len() { let (a, c) = (b.ja[j], b.jb[j]); if a == crate::brick::NONE { continue; } assert_eq!(lot(a), lot(c), "a joint across lots"); }
        // only loose contents are detached
        let det = b.detached();
        assert!(det.iter().all(|d| d.iter().all(|&i| b.flags[i as usize] & crate::brick::F_LOOSE != 0)), "only loose contents are detached");
        // it stands: the first structure step fails nothing, at rest and as the dynamic structure
        for dynamic in [false, true] {
            let mut g = new();
            g.dynamic = dynamic;
            let t = std::time::Instant::now();
            let f = g.structure_step(64);
            let t1 = t.elapsed().as_secs_f64() * 1e3;
            let mut more = 0;
            let t = std::time::Instant::now();
            for _ in 0..8 { g.struct_dt = 0.25; more += g.structure_step(64); }
            let t2 = t.elapsed().as_secs_f64() * 1e3 / 8.0;
            std::println!("  {}: first step {t1:.0} ms, {f} failed, worst ratio {:.3}; then {t2:.0} ms a step, {more} failed, {} lots active; {} mesh chunks",
                if dynamic { "dynamic" } else { "at rest" }, g.max_ratio, g.active.iter().filter(|&&a| a).count(), g.chunks.len());
            assert_eq!(f + more, 0, "nothing fails under its own weight");
        }
    }
    #[test] fn nothing_in_the_quarter_wears_left_alone() {
        // A joint loaded past 0.7 of its strength wears (engine: wear_joints) and gives way in time: a building that
        // stands near its limit comes down by itself. Left alone for 10 s of structure checks, no joint of the
        // quarter is wearing and every lot has gone quiet. (The villa, turned half round on its lot, stood at 0.79
        // where it stands at 0.69 as designed: its ramp's spine wall touched the roof slabs' edges. 15 Nov.)
        let mut g = new();
        g.structure_step(64);
        for _ in 0..40 { g.struct_dt = 0.25; g.structure_step(64); }
        std::println!("the quarter after 10 s: worst ratio {:.3}, wearing {}, worn {}, failed {}, active {:?}", g.max_ratio, g.wearing, g.worn_total, g.failed_total, g.active);
        assert!(g.wearing == 0 && g.worn_total == 0 && g.failed_total == 0);
        assert!(g.active.iter().all(|&a| !a), "every lot is quiet");
    }
    #[test] fn a_design_point_lands_where_its_brick_does() {
        // the villa is turned half round: its entrance (design z = -4.75 - 4.5, the half round of glass) faces north
        let l = layout();
        let v = l.iter().find(|p| p.what == What::Villa).unwrap();
        let w = to_world(v, 0.0, 1.0, -9.25);
        assert!((w[0] + 40.0).abs() < 1e-4 && (w[2] - (-40.0 + 9.25)).abs() < 1e-4, "{w:?}");
        let mut g = new();
        // a ray down the aqueduct's line at the height of its channel's floor meets it at its west end
        let a = l.iter().find(|p| p.what == What::Aqueduct).unwrap();
        let o = to_world(a, -30.0, 8.375, 0.0);
        let t = g.raycast(crate::math::v3(o[0], o[1], o[2]), crate::math::v3(1.0, 0.0, 0.0), 100.0);
        assert!((t - (30.0 - 12.75)).abs() < 0.01, "the channel's west end, 12.75 m from the aqueduct's middle: {t}");
        // the barn is turned a quarter: its great door (the south gable in its design) faces east. A ray from the east
        // along its middle, 1.3 m up, goes in through the door, over the cart and past the bales, to the far gable's
        // boards (design z = 11.5)
        let bn = l.iter().find(|p| p.what == What::Barn).unwrap();
        let o = to_world(bn, 0.0, 1.3, -20.0);
        assert!((o[0] - 60.0).abs() < 1e-4 && (o[2] + 40.0).abs() < 1e-4, "{o:?}");
        let t = g.raycast(crate::math::v3(o[0], o[1], o[2]), crate::math::v3(-1.0, 0.0, 0.0), 100.0);
        assert!((t - (20.0 + 11.5)).abs() < 0.01, "through the door to the far gable: {t}");
    }
}
