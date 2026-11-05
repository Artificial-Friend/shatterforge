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

// ---------------------------------------------------------------- phase-1 playground (game.rs)
use crate::game::Game;
static mut GAME: Option<Game> = None;
#[allow(static_mut_refs)]
fn game() -> &'static mut Game { unsafe { GAME.as_mut().unwrap() } }

#[no_mangle] pub extern "C" fn game_new(storeys: u32) { unsafe { GAME = Some(Game::new(storeys as i32)); } }
/// A tower of `kind`: 0 concrete frame, 1 masonry, 2 timber.
#[no_mangle] pub extern "C" fn game_new_kind(storeys: u32, kind: u32) { unsafe { GAME = Some(Game::new_kind(storeys as i32, kind as u8)); } }
// ---- handing the brick store between the simulation worker and the page
static mut SNAP: Vec<u8> = Vec::new();
#[allow(static_mut_refs)]
#[no_mangle] pub extern "C" fn snap_save() -> u32 { unsafe { game().b.save(&mut SNAP); SNAP.len() as u32 } }
#[allow(static_mut_refs)]
#[no_mangle] pub extern "C" fn snap_ptr() -> *mut u8 { unsafe { SNAP.as_mut_ptr() } }
#[allow(static_mut_refs)]
#[no_mangle] pub extern "C" fn snap_in(len: u32) -> *mut u8 { unsafe { SNAP.clear(); SNAP.resize(len as usize, 0); SNAP.as_mut_ptr() } }
/// Replace the store with the one written by `snap_in`, then remesh chunks lo..=hi.
#[allow(static_mut_refs)]
#[no_mangle] pub extern "C" fn snap_load(lo: i32, hi: i32) {
    let g = game();
    g.b = unsafe { crate::brick::Bricks::load(&SNAP) };
    for k in lo..=hi { g.remesh(k); }
    g.dirty = (lo.max(0), hi.min(g.chunks.len() as i32 - 1));
}
/// Settings: 0 crumbs kept per impacts call, 1 rubble fragments kept, 2 crush depth (m).
#[no_mangle] pub extern "C" fn set_param(id: u32, v: f32) {
    let g = game();
    match id { 0 => g.crumb_keep = (v.max(0.0) as usize).max(2), 1 => g.rubble_keep = v.max(0.0) as usize, 2 => g.crush_depth = v.max(0.0), _ => {} }
}
#[no_mangle] pub extern "C" fn game_mesh(on: u32) { game().mesh = on != 0; }
#[no_mangle] pub extern "C" fn chunk_count() -> u32 { game().chunks.len() as u32 }
#[no_mangle] pub extern "C" fn chunk_vptr(k: u32) -> *const f32 { game().chunks[k as usize].v.as_ptr() }
#[no_mangle] pub extern "C" fn chunk_vlen(k: u32) -> u32 { game().chunks[k as usize].v.len() as u32 }
#[no_mangle] pub extern "C" fn chunk_iptr(k: u32) -> *const u32 { game().chunks[k as usize].i.as_ptr() }
#[no_mangle] pub extern "C" fn chunk_ilen(k: u32) -> u32 { game().chunks[k as usize].i.len() as u32 }
/// Volume (m³) of the static bricks chunk k shows (the same bricks its mesh is built from), for the lab's check
/// that what is drawn never loses volume except to dust.
#[no_mangle] pub extern "C" fn chunk_volume(k: i32) -> f32 {
    let g = game();
    let (y0, y1) = (k * crate::tower::STOREY, (k + 1) * crate::tower::STOREY);
    let mut v = 0.0f64;
    for i in 0..g.b.len() {
        if g.b.flags[i] & (crate::brick::F_ALIVE | crate::brick::F_DYNAMIC) != crate::brick::F_ALIVE { continue; }
        let y = g.b.y[i] as i32; if y < y0 || y >= y1 { continue; }
        let t = g.b.tbox(i as u32);
        v += ((t.max[0] - t.min[0]) as f64) * ((t.max[1] - t.min[1]) as f64) * ((t.max[2] - t.min[2]) as f64);
    }
    let t = crate::brick::TICK as f64; (v * t * t * t) as f32
}
#[no_mangle] pub extern "C" fn dirty_lo() -> i32 { game().dirty.0 }
#[no_mangle] pub extern "C" fn dirty_hi() -> i32 { game().dirty.1 }
#[no_mangle] pub extern "C" fn out_ptr() -> *const f32 { game().out.as_ptr() }
#[no_mangle] pub extern "C" fn out_len() -> u32 { game().out.len() as u32 }
#[no_mangle] pub extern "C" fn rest_load(x: f32, y: f32, z: f32, f: f32) { game().rest.push([x, y, z, f]); }
#[no_mangle] pub extern "C" fn rest_applied() -> f32 { game().rest_applied }
#[no_mangle] pub extern "C" fn cracked_total() -> u32 { game().cracked_total as u32 }
#[no_mangle] pub extern "C" fn bricks_dump() { game().dump(); }
#[no_mangle] pub extern "C" fn blast(x: f32, y: f32, z: f32, r: f32) -> u32 { game().blast(v3(x, y, z), r) as u32 }
#[no_mangle] pub extern "C" fn raycast(ox: f32, oy: f32, oz: f32, dx: f32, dy: f32, dz: f32, max: f32) -> f32 { game().raycast(v3(ox, oy, oz), v3(dx, dy, dz), max) }
#[no_mangle] pub extern "C" fn bricks_alive() -> u32 { game().b.alive_count() as u32 }
#[no_mangle] pub extern "C" fn joints_live() -> u32 { game().b.live_joints as u32 }
#[no_mangle] pub extern "C" fn bytes_per_brick() -> f32 { game().b.total_bytes_per_brick() }
#[no_mangle] pub extern "C" fn player_set(x: f32, y: f32, z: f32) { let g = game(); g.player.p = v3(x, y, z); g.player.v = V3::ZERO; }
#[no_mangle] pub extern "C" fn player_step(dt: f32, wx: f32, wy: f32, wz: f32, jump: u32, fly: u32) {
    let g = game(); g.player.step(&g.b, dt, v3(wx, wy, wz), jump != 0, fly != 0);
}
#[no_mangle] pub extern "C" fn player_get(axis: u32) -> f32 { game().player.p.get(axis as usize) }
#[no_mangle] pub extern "C" fn player_on_ground() -> u32 { game().player.on_ground as u32 }
#[no_mangle] pub extern "C" fn structure_step(max_fail: u32) -> u32 { game().structure_step(max_fail as usize) as u32 }
#[no_mangle] pub extern "C" fn structure_ratio() -> f32 { game().max_ratio }
/// Room for `n` impact records (11 floats each, see `Game::impacts`); returns where to write them.
#[no_mangle] pub extern "C" fn impacts_in(n: u32) -> *mut f32 { let g = game(); g.inp.clear(); g.inp.resize(n as usize * 11, 0.0); g.inp.as_mut_ptr() }
#[no_mangle] pub extern "C" fn impacts(dt: f32) -> u32 { game().impacts(dt) as u32 }
#[no_mangle] pub extern "C" fn piece_blast(piece: u32, x: f32, y: f32, z: f32, r: f32) -> u32 { game().blast_piece(piece, v3(x, y, z), r) as u32 }
#[no_mangle] pub extern "C" fn dust_bricks() -> u32 { game().dust as u32 }
#[no_mangle] pub extern "C" fn piece_impact(piece: u32, px: f32, py: f32, pz: f32, dx: f32, dy: f32, dz: f32, impulse: f32, approach: f32, dt: f32) -> u32 {
    game().impact(piece, v3(px, py, pz), v3(dx, dy, dz), impulse, approach, dt) as u32
}
