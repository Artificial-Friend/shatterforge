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
/// Settings: 0 crumbs kept per impacts call, 1 rubble fragments kept, 2 crush depth (m), 3 standing bricks split
/// into convex cells at most (0: fracture leaves boxes), 7 the standing structure stepped dynamically (docs/DYNAMIC.md).
#[no_mangle] pub extern "C" fn set_param(id: u32, v: f32) {
    let g = game();
    match id { 0 => g.crumb_keep = (v.max(0.0) as usize).max(2), 1 => g.rubble_keep = v.max(0.0) as usize, 2 => g.crush_depth = v.max(0.0), 3 => g.frac_max = v.max(0.0) as usize, 4 => g.struct_dt = v.max(0.0), 5 => g.spill_on = v > 0.5, 6 => g.masonry_parts = v > 0.5, 7 => g.dynamic = v > 0.5,
        // (the view asked for: the next structure step analyses every lot, so that all of them are drawn)
        8 => { g.view_want = v > 0.5; if g.view_want { for a in g.active.iter_mut() { *a = true; } for st in g.dynst.values_mut() { st.1.viewed = false; } } }
        _ => {} }
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
    let t = g.tiles;
    let mut v = 0.0f64;
    for i in 0..g.b.len() {
        if g.b.flags[i] & (crate::brick::F_ALIVE | crate::brick::F_DYNAMIC) != crate::brick::F_ALIVE { continue; }
        if !t.holds(k, g.b.x[i] as i32, g.b.y[i] as i32, g.b.z[i] as i32) { continue; }
        let t = g.b.tbox(i as u32);
        // (a convex cell: its share of its box)
        v += ((t.max[0] - t.min[0]) as f64) * ((t.max[1] - t.min[1]) as f64) * ((t.max[2] - t.min[2]) as f64) * g.b.vol_share(i as u32);
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
/// a free piece's shape (Game::piece_shape) into its own buffer: its length; then piece_ptr
#[no_mangle] pub extern "C" fn piece_shape(k: u32) -> u32 { game().piece_shape(k) as u32 }
#[no_mangle] pub extern "C" fn piece_ptr() -> *const f32 { game().piece_out.as_ptr() }
#[no_mangle] pub extern "C" fn blast(x: f32, y: f32, z: f32, r: f32) -> u32 { let g = game(); g.plan_out.clear(); let n = g.blast(v3(x, y, z), r) as u32; g.finish_plans(); n }
#[no_mangle] pub extern "C" fn raycast(ox: f32, oy: f32, oz: f32, dx: f32, dy: f32, dz: f32, max: f32) -> f32 { game().raycast(v3(ox, oy, oz), v3(dx, dy, dz), max) }
#[no_mangle] pub extern "C" fn bricks_alive() -> u32 { game().b.alive_count() as u32 }
#[no_mangle] pub extern "C" fn joints_live() -> u32 { game().b.live_joints as u32 }
#[no_mangle] pub extern "C" fn bytes_per_brick() -> f32 { game().b.total_bytes_per_brick() }
#[no_mangle] pub extern "C" fn player_set(x: f32, y: f32, z: f32) { let g = game(); g.player.p = v3(x, y, z); g.player.v = V3::ZERO; }
#[no_mangle] pub extern "C" fn player_step(dt: f32, wx: f32, wy: f32, wz: f32, jump: u32, fly: u32) {
    let g = game(); g.player.step(&g.b, dt, v3(wx, wy, wz), jump != 0, fly != 0);
}
/// the debris near the player (from the page's read-back), as oriented boxes the player stands on and walks into
#[no_mangle] pub extern "C" fn player_debris_clear() { game().player.extra.clear(); }
#[no_mangle] pub extern "C" fn player_debris_box(cx: f32, cy: f32, cz: f32, qx: f32, qy: f32, qz: f32, qw: f32, hx: f32, hy: f32, hz: f32) { game().player.extra.push(crate::player::Obstacle::obb([cx, cy, cz], [qx, qy, qz, qw], [hx, hy, hz])); }
#[no_mangle] pub extern "C" fn player_get(axis: u32) -> f32 { game().player.p.get(axis as usize) }
#[no_mangle] pub extern "C" fn player_on_ground() -> u32 { game().player.on_ground as u32 }
#[no_mangle] pub extern "C" fn structure_step(max_fail: u32) -> u32 { let g = game(); g.plan_out.clear(); let n = g.structure_step(max_fail as usize) as u32; g.finish_plans(); n }
/// pieces the last structure step sent off: with no failures, loose contents (the page applies its output then too)
/// the building has loose bricks (room contents) not yet sent off: the page runs a first structure step
#[no_mangle] pub extern "C" fn loose_pending() -> u32 { game().loose_pending as u32 }
#[no_mangle] pub extern "C" fn last_detached() -> u32 { game().last_detached as u32 }
#[no_mangle] pub extern "C" fn structure_ratio() -> f32 { game().max_ratio }
/// The structure as the engineer sees it (Game::view_out, view_apply): what the last structure step wrote, and in
/// the page's copy room to take it, drawing it, and leaving the view.
#[no_mangle] pub extern "C" fn view_ptr() -> *const f32 { game().view_out.as_ptr() }
#[no_mangle] pub extern "C" fn view_len() -> u32 { game().view_out.len() as u32 }
#[no_mangle] pub extern "C" fn view_in(n: u32) -> *mut f32 { let g = game(); g.view_in.clear(); g.view_in.resize(n as usize, 0.0); g.view_in.as_mut_ptr() }
#[no_mangle] pub extern "C" fn view_apply(scale: f32, stress: u32) { game().view_apply(scale, stress != 0); }
#[no_mangle] pub extern "C" fn view_off() { game().view_off(); }
/// The chunks the last view_apply or view_off remeshed.
#[no_mangle] pub extern "C" fn view_chunk_count() -> u32 { game().view_chunks.len() as u32 }
#[no_mangle] pub extern "C" fn view_chunk(i: u32) -> i32 { game().view_chunks.get(i as usize).copied().unwrap_or(-1) }
/// The dynamic structure after the last step: f 0 still moving (another step is due), 1 its fastest cell (m/s),
/// 2 the solver's iterations (the lot that took most), 3 the cells' kinetic energy (J), 4 the largest displacement (m),
/// 5 the cells that left the structure in the step, 6 the joints and faces cut for it, 7 a solve stopped at its cap
#[no_mangle] pub extern "C" fn structure_dyn(f: u32) -> f32 {
    let g = game();
    match f { 0 => g.moving as u32 as f32, 1 => g.dyn_speed, 2 => g.dyn_its as f32, 3 => g.dyn_kinetic, 4 => g.dynst.values().map(|s| s.0.max_disp()).fold(0.0, f64::max) as f32, 5 => g.dyn_left as f32, 6 => g.dyn_cut as f32, 7 => g.dyn_capped as u32 as f32, _ => 0.0 }
}
#[no_mangle] pub extern "C" fn structure_wearing() -> u32 { game().wearing as u32 }
#[no_mangle] pub extern "C" fn worn_total() -> u32 { game().worn_total as u32 }
#[no_mangle] pub extern "C" fn dup_bricks() -> u32 { game().dup_bricks as u32 }
#[no_mangle] pub extern "C" fn spill_why(k: u32) -> u32 { game().spill_why[(k as usize).min(3)] as u32 }
#[no_mangle] pub extern "C" fn spill_mat(k: u32) -> u32 { if k == 8 { game().spill_single as u32 } else { game().spill_mat[(k as usize).min(7)] as u32 } }
/// Room for `n` impact records (11 floats each, see `Game::impacts`); returns where to write them.
#[no_mangle] pub extern "C" fn impacts_in(n: u32) -> *mut f32 { let g = game(); g.inp.clear(); g.inp.resize(n as usize * 11, 0.0); g.inp.as_mut_ptr() }
#[no_mangle] pub extern "C" fn impacts(dt: f32) -> u32 { let g = game(); g.plan_out.clear(); let n = g.impacts(dt) as u32; g.finish_plans(); n }
#[no_mangle] pub extern "C" fn piece_blast(piece: u32, x: f32, y: f32, z: f32, r: f32) -> u32 { let g = game(); g.plan_out.clear(); let n = g.blast_piece(piece, v3(x, y, z), r) as u32; g.finish_plans(); n }
#[no_mangle] pub extern "C" fn dust_bricks() -> u32 { game().dust as u32 }
#[no_mangle] pub extern "C" fn piece_impact(piece: u32, px: f32, py: f32, pz: f32, dx: f32, dy: f32, dz: f32, impulse: f32, approach: f32, dt: f32) -> u32 {
    game().impact(piece, v3(px, py, pz), v3(dx, dy, dz), impulse, approach, dt) as u32
}

/// The engine's hull library, for the check against the GPU's (gpu/test_hulls.cjs): the number of hulls, and
/// per hull k = 0..2 its box centre, 3..5 half size, 6 volume share, 7..9 centroid (cube coordinates), 10 pattern
#[no_mangle] pub extern "C" fn hull_count() -> u32 { crate::hull::lib().cells.len() as u32 }
#[no_mangle] pub extern "C" fn hull_value(id: u32, k: u32) -> f64 {
    let c = &crate::hull::lib().cells[id as usize];
    match k { 0..=2 => c.bc[k as usize], 3..=5 => c.bh[k as usize - 3], 6 => c.vol, 7..=9 => c.cen[k as usize - 7], _ => c.pattern as f64 }
}

/// The GPU broke falling piece `piece` by its plan (see Game::snap); the result in `out`.
#[no_mangle] pub extern "C" fn piece_snap(piece: u32) -> u32 { let g = game(); g.plan_out.clear(); let n = g.snap(piece) as u32; g.finish_plans(); n }
/// The break plans made by the last call (see Game::plan_out)
#[no_mangle] pub extern "C" fn plan_ptr() -> *const f32 { game().plan_out.as_ptr() }
#[no_mangle] pub extern "C" fn plan_len() -> u32 { game().plan_out.len() as u32 }
/// The city's lots (kind 5, `n` a side): how many, and lot k's field f: 0 kind, 1 storeys, 2 quarter turns, 3 x, 4 z
/// (m, its centre).
#[no_mangle] pub extern "C" fn city_lots(n: u32) -> u32 { crate::city::layout(n as i32).lots.len() as u32 }
#[no_mangle] pub extern "C" fn city_lot(n: u32, k: u32, f: u32) -> f32 {
    let l = crate::city::layout(n as i32);
    let Some(o) = l.lots.get(k as usize) else { return 0.0 };
    match f { 0 => o.kind as f32, 1 => o.storeys as f32, 2 => o.quarter as f32, 3 => o.off[0] as f32 * crate::brick::TICK, 4 => o.off[2] as f32 * crate::brick::TICK, _ => 0.0 }
}
