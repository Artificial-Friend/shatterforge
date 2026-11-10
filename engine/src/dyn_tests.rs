//! The standing structure as a dynamic model (structure::Dyn, docs/DYNAMIC.md): what a game sees of it.
extern crate std;
use crate::game::Game;
use crate::math::v3;
use std::{println, time::Instant, vec::Vec};

const DT: f32 = 1.0 / 60.0;
fn largest_piece(g: &Game) -> usize { g.pieces.iter().map(|p| p.len()).max().unwrap_or(0) }
fn disp(g: &Game) -> f64 { g.dynst.values().map(|s| s.0.max_disp()).fold(0.0, f64::max) }
/// the ground storey blasted away except round (kx, kz)
fn cut_but(g: &mut Game, keep: Option<(f32, f32)>) {
    let mut x = -11.6f32;
    while x <= 11.7 { let mut z = -11.6f32; while z <= 11.7 {
        if keep.map_or(true, |k| (x - k.0).abs() > 1.3 || (z - k.1).abs() > 1.3) { g.blast(v3(x, 1.6, z), 0.7); }
        z += 0.8; } x += 0.8; }
}

#[test] fn a_standing_building_costs_no_iterations_and_does_not_drift() {
    // the 12-storey tower, solved at rest, then 600 dynamic steps (10 s): no step needs an iteration, nothing
    // moves and nothing fails
    let mut g = Game::new_kind(12, 0);
    g.dynamic = true;
    assert_eq!(g.structure_step(64), 0, "the tower stands");
    let (d0, r0) = (disp(&g), g.max_ratio);
    let t = Instant::now();
    let (mut its, mut failed, mut moved) = (0, 0, 0);
    for _ in 0..600 { g.struct_dt = DT; failed += g.structure_step(64); its += g.dyn_its; if g.moving { moved += 1; } }
    let ms = t.elapsed().as_secs_f64() * 1e3 / 600.0;
    println!("12-storey tower at rest: largest displacement {:.3} mm before, change {:.2e} mm after 600 steps; worst {r0:.3} / {:.3}; {its} iterations, {moved} steps moving, {failed} failures, {ms:.1} ms a step", d0 * 1e3, (disp(&g) - d0).abs() * 1e3, g.max_ratio);
    assert_eq!((its, failed, moved), (0, 0, 0));
    assert!((disp(&g) - d0).abs() < 1e-9, "no drift");
    assert!((g.max_ratio - r0).abs() < 1e-6);
}

#[test] fn a_column_gone_is_answered_within_the_steps_it_takes_to_move() {
    // a corner column of the 4-storey concrete frame blasted out. Solved at rest (round after round until nothing
    // fails) and stepped dynamically (2 s of steps): the same joints give way, and the worst joint left ends the
    // same. Every dynamic step is bounded.
    let run = |dynamic: bool| -> (Game, Vec<(usize, f32, usize, f32)>, f64) {
        let mut g = Game::new_kind(4, 0);
        g.dynamic = dynamic;
        assert_eq!(g.structure_step(64), 0);
        for y in [0.8f32, 1.6, 2.4] { g.blast(v3(10.7, y, -10.7), 0.8); }
        let mut log = Vec::new();
        let mut worst_ms = 0.0f64;
        for k in 0..240 {
            g.struct_dt = if dynamic { DT } else { 0.0 };
            let t = Instant::now();
            let n = g.structure_step(64);
            worst_ms = worst_ms.max(t.elapsed().as_secs_f64() * 1e3);
            log.push((n, g.max_ratio, if dynamic { g.dyn_its } else { g.last_iterations }, g.dyn_speed));
            if !dynamic && n == 0 { break; }
            if dynamic && n == 0 && !g.moving && k > 4 { break; }
        }
        (g, log, worst_ms)
    };
    let (gs, ls, ms_s) = run(false);
    let (gd, ld, ms_d) = run(true);
    println!("at rest:   {} rounds, {} joints failed, worst left {:.3}, iterations a round {:?}, slowest round {ms_s:.0} ms", ls.len(), gs.failed_total, gs.max_ratio, ls.iter().map(|l| l.2).collect::<Vec<_>>());
    println!("dynamic:   {} steps to rest, {} joints failed, worst left {:.3}, slowest step {ms_d:.0} ms", ld.len(), gd.failed_total, gd.max_ratio);
    println!("  first steps (failures, worst ratio, iterations, fastest cell mm/s): {:?}", ld.iter().take(12).map(|l| (l.0, (l.1 * 1e3).round() / 1e3, l.2, (l.3 * 1e4).round() / 10.0)).collect::<Vec<_>>());
    assert!(ld.len() < 240, "comes to rest");
    assert!(ld.iter().all(|l| l.2 <= 9 * crate::structure::DYN_IT), "every step bounded");
    assert!((gd.max_ratio - gs.max_ratio).abs() < 0.05 * gs.max_ratio.max(0.2), "the same worst joint left: {} / {}", gd.max_ratio, gs.max_ratio);
    assert!((gd.failed_total as f32 - gs.failed_total as f32).abs() <= (0.1 * gs.failed_total as f32).max(3.0), "the same failures: {} / {}", gd.failed_total, gs.failed_total);
}

#[test] fn a_tower_on_one_column_comes_down_in_bounded_steps() {
    // game::tests::a_tower_left_on_one_column_comes_down, stepped dynamically. At rest the first solve of this
    // mechanism takes 340 iterations on the 12-storey tower (docs/STRUCTURE.md); a dynamic step is bounded, and the
    // floors come down all the same.
    for (storeys, kind, keep) in [(4, 0u8, (-10.7f32, -10.7f32)), (4, 1, (-3.7, -3.7)), (12, 0, (-3.7, -3.7))] {
        let mut out = Vec::new();
        for dynamic in [false, true] {
            let mut g = Game::new_kind(storeys, kind);
            g.dynamic = dynamic;
            assert_eq!(g.structure_step(64), 0);
            cut_but(&mut g, Some(keep));
            let (mut steps, mut worst_ms, mut worst_it, mut total_ms) = (0, 0.0f64, 0, 0.0);
            while steps < 600 {
                g.struct_dt = if dynamic { DT } else { 0.0 };
                let t = Instant::now();
                let n = g.structure_step(64);
                let ms = t.elapsed().as_secs_f64() * 1e3;
                worst_ms = worst_ms.max(ms); total_ms += ms;
                worst_it = worst_it.max(if dynamic { g.dyn_its } else { g.last_iterations });
                steps += 1;
                if n == 0 && !(dynamic && g.moving) { break; }
            }
            println!("{storeys} storeys, kind {kind}, {}: {} joints failed in {steps} steps, biggest falling piece {} bricks; most iterations in a step {worst_it}, slowest step {worst_ms:.0} ms, {total_ms:.0} ms in all", if dynamic { "dynamic" } else { "at rest" }, g.failed_total, largest_piece(&g));
            out.push((largest_piece(&g), worst_it, steps));
        }
        assert!(out[1].0 > 300, "the floors above come down");
        assert!(out[1].1 <= 9 * crate::structure::DYN_IT, "bounded");
        assert!(out[1].2 < 600, "and it ends");
    }
}

#[test] fn the_complex_building_comes_to_rest_after_a_column_is_shot_out() {
    // the lab's complexcol (the north-east corner column shot out, four shots), stepped as the lab steps it (a
    // step every 4 frames). The complex building's stones stand on open and hinged beds: with the beds started
    // closed at every step, no step found it at rest (70 steps in 280 frames, each re-opening them, and the lab's
    // settled check failed); kept between steps, it rests.
    let shoot = |g: &mut Game| { let (o, d) = (v3(16.0, 1.5, 6.0), v3(-1.0, 0.0, 0.0)); let t = g.raycast(o, d, 100.0); if t < 100.0 { g.blast(o + d * t, 0.7); } };
    let mut out = Vec::new();
    for dynamic in [false, true] {
        let mut g = Game::new_kind(3, 4);
        g.dynamic = dynamic;
        g.structure_step(64);
        let r0 = g.max_ratio;
        let (mut steps, mut its, mut last_moving) = (0, 0, 0);
        for k in 0..120 {
            if k < 4 { shoot(&mut g); }
            g.struct_dt = if dynamic { 4.0 / 60.0 } else { 0.0 };
            let n = g.structure_step(48);
            steps += 1; its += if dynamic { g.dyn_its } else { g.last_iterations };
            if n > 0 || g.moving { last_moving = k; }
            if dynamic && std::env::var("SF_DYN_TRACE").is_ok() { let st = &g.dynst[&0]; println!("  step {k}: its {} speed {:.2} mm/s kinetic {:.3} J bed changes {} beds open/hinged {} ratio {:.3}", st.1.its, st.1.speed * 1e3, st.1.kinetic, st.1.cut, st.1.beds.len(), g.max_ratio); }
            if !dynamic && k >= 4 && n == 0 { break; }
        }
        println!("complex building, corner column shot out, {}: worst {r0:.3} -> {:.3}, {} joints failed, {steps} steps, {its} iterations, last step moving or failing {last_moving}", if dynamic { "dynamic" } else { "at rest" }, g.max_ratio, g.failed_total);
        out.push((g.max_ratio, g.failed_total, last_moving));
    }
    assert!(out[1].2 < 75, "at rest within 5 s (75 steps of 4 frames): still moving at step {}", out[1].2);
    assert!((out[1].0 - out[0].0).abs() < 0.05 * out[0].0, "the same worst joint: {} / {}", out[1].0, out[0].0);
    assert!((out[1].1 as f32 - out[0].1 as f32).abs() <= (0.1 * out[0].1 as f32).max(3.0), "the same failures: {} / {}", out[1].1, out[0].1);
}

#[test] fn the_view_draws_the_structure_where_its_solution_puts_it() {
    // the 4-storey frame, its structure step asked for the view; the page's copy takes it (here the same game) and
    // draws the displacements 100 times as large, coloured by load. The roof comes down by 100 times what the top
    // cells moved, faces are cut along the cells (a slab drawn sagging bends), and bricks take their load's colour.
    let mut g = Game::new_kind(4, 0);
    g.dynamic = true; g.view_want = true;
    assert_eq!(g.structure_step(64), 0);
    let top = |g: &Game| g.chunks.iter().flat_map(|m| m.v.chunks(9).map(|v| v[1])).fold(f32::MIN, f32::max);
    let quads = |g: &Game| g.chunks.iter().map(|m| m.quads).sum::<usize>();
    for k in 0..g.chunks.len() as i32 { g.remesh(k); }
    let (top0, q0) = (top(&g), quads(&g));
    let cells = g.view_out[1 + 2 * g.view_out[0] as usize] as usize;
    g.view_in = g.view_out.clone();
    g.view_apply(100.0, true);
    let (top1, q1) = (top(&g), quads(&g));
    let sag = disp(&g) as f32;
    let worst = g.view.ratio.values().copied().fold(0.0, f32::max);
    println!("view: {} bricks, {cells} cells; largest displacement {:.2} mm; roof drawn at {top0:.3} -> {top1:.3} m at x 100; {q0} -> {q1} quads; worst brick at {worst:.2} of strength (the building's worst {:.2})", g.view_out[0], sag * 1e3, g.max_ratio);
    assert!(cells > 1000 && q1 > q0, "faces cut along the cells");
    assert!(top0 - top1 > 0.0 && top0 - top1 < 100.0 * sag + 1e-3, "the roof drawn lower, by no more than the largest displacement x 100: {}", top0 - top1);
    // (the cell that moved most: its centre is drawn 100 times its displacement away)
    let (bi, c) = g.view.cells.iter().flat_map(|(i, cs)| cs.iter().map(move |c| (*i, *c))).max_by(|a, b| (a.1.u[1].abs()).partial_cmp(&b.1.u[1].abs()).unwrap()).unwrap();
    let q = g.view.moved(bi, c.c);
    assert!(((c.c[1] - q[1]) - 100.0 * -c.u[1]).abs() < 1e-4 && (c.u[1].abs() - sag).abs() < 0.2 * sag, "cell drawn {} m lower, moved {} m", c.c[1] - q[1], c.u[1]);
    assert!((worst - g.max_ratio).abs() < 0.02 || worst >= g.max_ratio, "bricks carry the worst joint's ratio");
    let tinted = g.chunks.iter().flat_map(|m| m.v.chunks(9)).filter(|v| { let t = crate::mesh::View::tint(0.0); (v[6] - t[0]).abs() < 0.2 && v[8] > v[6] }).count();
    assert!(tinted > 0, "lightly loaded bricks are blue");
    g.view_off();
    assert!((top(&g) - top0).abs() < 1e-6 && quads(&g) == q0, "and the building as built again");
}

#[test] fn a_city_is_stepped_lot_by_lot() {
    // the 3 x 3 city: solved at rest lot by lot, then stepped dynamically. At rest nothing moves or fails; the
    // complex building's corner column shot out moves its lot only, and the city rests again.
    let mut g = Game::new_kind(3, 5);
    g.dynamic = true; g.view_want = true;
    let t = Instant::now();
    assert_eq!(g.structure_step(64), 0, "the city stands");
    let first = t.elapsed().as_secs_f64() * 1e3;
    let lots = g.dynst.len();
    let (d0, r0) = (disp(&g), g.max_ratio);
    let (bricks, cells) = (g.view_out[0] as usize, g.view_out[1 + 2 * g.view_out[0] as usize] as usize);
    let mut its = 0;
    let t = Instant::now();
    for _ in 0..30 { g.struct_dt = DT; assert_eq!(g.structure_step(64), 0); its += g.dyn_its; assert!(!g.moving); }
    let rest = t.elapsed().as_secs_f64() * 1e3 / 30.0;
    println!("city 3 x 3: {lots} lots, {bricks} bricks and {cells} cells in the view, first step {first:.0} ms, worst {r0:.3}, largest displacement {:.2} mm; 30 steps at rest: {its} iterations, {rest:.2} ms a step", d0 * 1e3);
    assert_eq!(its, 0);
    assert!(lots == 9 && d0 > 1e-4 && cells > 5000);
    // the first lot's complex building: a column of its frame shot out (the lab's complexcol, moved to its lot)
    let l0 = crate::city::layout(3).lots[0].off;
    let (ox, oz) = (l0[0] as f32 * crate::brick::TICK, l0[2] as f32 * crate::brick::TICK);
    let (mut moved_lots, mut steps, mut worst_ms) = (0usize, 0, 0.0f64);
    for k in 0..240 {
        if k < 4 { for (o, d) in [(v3(ox + 16.0, 1.5, oz + 6.0), v3(-1.0, 0.0, 0.0)), (v3(ox - 6.0, 1.5, oz + 16.0), v3(0.0, 0.0, -1.0)), (v3(ox - 16.0, 1.5, oz - 6.0), v3(1.0, 0.0, 0.0)), (v3(ox + 6.0, 1.5, oz - 16.0), v3(0.0, 0.0, 1.0))] { let t = g.raycast(o, d, 12.0); if t < 12.0 { g.blast(o + d * t, 0.7); } } }
        g.struct_dt = DT;
        let t = Instant::now();
        let n = g.structure_step(64);
        worst_ms = worst_ms.max(t.elapsed().as_secs_f64() * 1e3);
        moved_lots = moved_lots.max(g.dynst.values().filter(|s| s.1.moving()).count());
        steps = k + 1;
        if k >= 4 && n == 0 && !g.moving { break; }
    }
    println!("  a column shot out on lot 0: at rest again after {steps} steps, at most {moved_lots} lot moving at once, slowest step {worst_ms:.0} ms, worst {:.3}, {} joints failed", g.max_ratio, g.failed_total);
    assert!(steps < 240 && moved_lots == 1);
}

#[test] fn the_view_of_a_city_at_rest_is_written_once() {
    // the 8 x 8 city with the view on. Reported: 3 frames a second, where the city plays without the view. Every
    // structure step wrote every analysed lot, and the lots with furniture or rubble on their floors are analysed
    // every second: 279,622 numbers each time, and the page remeshed the range from the first chunk named to the
    // last, most of the city's 960 (672 ms of meshing, native). Now a lot at rest is written once, and only the
    // chunks of the bricks named are remeshed: a step at rest writes nothing, and a column shot out on one lot
    // remeshes that lot's chunks.
    let mut g = Game::new_kind(8, 5);
    g.dynamic = true;
    g.structure_step(48);
    g.view_want = true; for a in g.active.iter_mut() { *a = true; } for st in g.dynst.values_mut() { st.1.viewed = false; }
    g.struct_dt = DT;
    let t = Instant::now(); g.structure_step(48); let step = t.elapsed().as_secs_f64() * 1e3;
    let first = g.view_out.len();
    g.view_in = g.view_out.clone();
    let t = Instant::now(); g.view_apply(100.0, true); let apply = t.elapsed().as_secs_f64() * 1e3;
    let all = g.view_chunks.len();
    let mut seq = Vec::new();
    for _ in 0..8 { g.struct_dt = 1.0; g.structure_step(48); seq.push((g.view_out.len(), g.dyn_its, g.moving)); }
    println!("  steps at rest (numbers written, iterations, moving): {seq:?}");
    let rest = seq[4..].iter().map(|s| s.0).max().unwrap();
    // a column of lot 0's building shot out
    let l0 = crate::city::layout(8).lots[0].off;
    let (ox, oz) = (l0[0] as f32 * crate::brick::TICK, l0[2] as f32 * crate::brick::TICK);
    let (o, d) = (v3(ox + 16.0, 1.5, oz + 6.0), v3(-1.0, 0.0, 0.0));
    let tt = g.raycast(o, d, 12.0); assert!(tt < 12.0); g.blast(o + d * tt, 0.7);
    g.struct_dt = DT; g.structure_step(48);
    let changed = g.view_out.len();
    g.view_in = g.view_out.clone();
    let t = Instant::now(); g.view_apply(100.0, true); let apply1 = t.elapsed().as_secs_f64() * 1e3;
    println!("city 8 x 8, view on: first step {step:.0} ms writes {first} numbers, drawn in {apply:.0} ms ({all} chunks); steps at rest write {rest} at most once settled; a column shot out on lot 0 writes {changed}, drawn in {apply1:.1} ms ({} chunks)", g.view_chunks.len());
    assert!(first > 1_000_000 && all == g.chunks.len(), "every lot written once");
    assert!(rest <= 2, "nothing written at rest: {rest}");
    assert!(changed > 2 && changed < first / 20 && g.view_chunks.len() <= g.chunks.len() / 64 + 1, "one lot written and remeshed: {} chunks", g.view_chunks.len());
}
