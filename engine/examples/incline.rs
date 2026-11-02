// A box resting on a fixed slope below the friction angle: does it sit still, creep, or chatter?
use sf_engine::math::*;
use sf_engine::phys::World;
fn main() {
    let deg: f32 = std::env::args().nth(1).and_then(|v| v.parse().ok()).unwrap_or(25.0);
    let mut w = World::new();
    w.prm.allow_sleep = false;
    if std::env::var("NOANCH").is_ok() { w.prm.anchor = false; }
    if let Ok(v) = std::env::var("AP") { w.prm.anchor_push = v.parse().unwrap(); }
    let th = deg.to_radians();
    let q = Quat::from_axis_angle(v3(0.0, 0.0, 1.0), th);
    let n = v3(-sin(th), cos(th), 0.0);
    w.add_box(v3(0.0, 3.0, 0.0), q, v3(4.0, 0.25, 1.0), 0.0);
    let c = v3(0.0, 3.0, 0.0) + n * 0.5;
    w.add_box(c, q, v3(0.25, 0.25, 0.25), 1900.0);
    let mut maxv = 0.0f32;
    for f in 0..600 {
        w.step(1.0 / 60.0);
        let b = &w.bodies[1];
        if f >= 120 { maxv = maxv.max(b.v.len()); }
        if f >= 590 { println!("f{f} v ({:+.4},{:+.4},{:+.4}) w {:.4}", b.v.x, b.v.y, b.v.z, b.w.len()); }
    }
    let b = &w.bodies[1];
    println!("{deg} deg: moved {:.2} mm, max speed after 2 s {:.2} mm/s", (b.p - c).len() * 1e3, maxv * 1e3);
}
