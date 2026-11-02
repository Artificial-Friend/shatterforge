use sf_engine::math::*;
use sf_engine::phys::World;
fn e(k: &str, d: f32) -> f32 { std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }
fn main() {
    let n = e("N", 20.0) as usize;
    let mut w = World::new();
    w.prm.allow_sleep = e("SLEEP", 0.0) > 0.5; w.prm.substeps = e("SUB", 8.0) as u32; w.prm.contact_hertz = e("HZ", 0.0);
    w.prm.damping_ratio = e("ZETA", 10.0); w.prm.anchor = e("ANCH", 1.0) > 0.5; w.prm.friction = e("MU", 0.6);
    for k in 0..n { w.add_box(v3(0.0, 0.25 + 0.5 * k as f32, 0.0), Quat::IDENTITY, v3(0.25, 0.25, 0.25), 1900.0); }
    for f in 0..600 {
        w.step(1.0 / 60.0);
        if f % 60 == 59 {
            let (mut mv, mut mi) = (0.0, 0); for (i, b) in w.bodies.iter().enumerate() { let s = b.v.len(); if s > mv { mv = s; mi = i; } }
            let t = w.bodies[n - 1];
            println!("t{:>2}s top dx {:+.2} dy {:+.2} mm | fastest #{mi} {:.2} mm/s v {:?} w {:?}", (f + 1) / 60, t.p.x * 1e3, (t.p.y - (0.25 + 0.5 * (n - 1) as f32)) * 1e3, mv * 1e3, w.bodies[mi].v, w.bodies[mi].w);
        }
    }
}
