//! WebAssembly exports for the phase-0 benchmarks. The host times the calls with its own clock.
use crate::brick::{self, Bricks};
use crate::math::*;
use crate::phys::{self, World};
use alloc::vec::Vec;

static mut WORLD: Option<World> = None;
static mut BRICKS: Option<Bricks> = None;
static mut POSE: Vec<f32> = Vec::new();

#[allow(static_mut_refs)]
fn world() -> &'static mut World { unsafe { WORLD.get_or_insert_with(World::new) } }

/// Build the split-gate panel (kept until the next call).
#[no_mangle] pub extern "C" fn split_setup() { unsafe { BRICKS = Some(brick::panel().0); } }
/// Refine the panel around (cx, 15, 2) ticks with a 4-tick radius to 1-tick bricks. Returns the brick count.
#[allow(static_mut_refs)]
#[no_mangle] pub extern "C" fn split_run(cx: f32) -> u32 {
    let b = unsafe { BRICKS.as_mut().unwrap() };
    b.refine_sphere([cx, 15.0, 2.0], 4.0, 1);
    b.alive_count() as u32
}
#[allow(static_mut_refs)]
#[no_mangle] pub extern "C" fn split_bytes_per_brick() -> f32 { unsafe { BRICKS.as_ref().map(|b| b.bytes_per_brick()).unwrap_or(0.0) } }

/// Throughput scenario: n brick-sized boxes dropped in a loose pile.
#[no_mangle] pub extern "C" fn pile_setup(n: u32, substeps: u32) {
    let mut w = phys::pile(n as usize);
    w.prm.allow_sleep = false; w.prm.substeps = substeps;
    unsafe { WORLD = Some(w); }
}
/// The stacking gate: n 0.5 m cubes stacked exactly.
#[no_mangle] pub extern "C" fn stack_setup(n: u32, substeps: u32, sleep: u32) {
    let mut w = World::new();
    w.prm.substeps = substeps; w.prm.allow_sleep = sleep != 0;
    for k in 0..n { w.add_box(v3(0.0, 0.25 + 0.5 * k as f32, 0.0), Quat::IDENTITY, v3(0.25, 0.25, 0.25), 1900.0); }
    unsafe { WORLD = Some(w); }
}
#[no_mangle] pub extern "C" fn step(dt: f32) -> u32 { let w = world(); w.step(dt); w.stats.awake }
#[no_mangle] pub extern "C" fn body_count() -> u32 { world().bodies.len() as u32 }
#[no_mangle] pub extern "C" fn manifolds() -> u32 { world().stats.manifolds }
#[no_mangle] pub extern "C" fn points() -> u32 { world().stats.points }
#[no_mangle] pub extern "C" fn max_speed() -> f32 { world().bodies.iter().map(|b| b.v.len()).fold(0.0, f32::max) }
#[no_mangle] pub extern "C" fn body_pos(i: u32, axis: u32) -> f32 { world().bodies[i as usize].p.get(axis as usize) }
/// Fill and return a pointer to 10 floats per body: position, rotation quaternion (x, y, z, w), half extents.
#[allow(static_mut_refs)]
#[no_mangle] pub extern "C" fn pose() -> *const f32 {
    let w = world();
    let p = unsafe { &mut POSE };
    p.clear();
    for b in &w.bodies { p.extend_from_slice(&[b.p.x, b.p.y, b.p.z, b.q.x, b.q.y, b.q.z, b.q.w, b.h.x, b.h.y, b.h.z]); }
    p.as_ptr()
}
