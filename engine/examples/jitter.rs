// Which bodies keep a settled pile awake? Track the fastest ones over a few frames.
use sf_engine::phys::pile;
fn main() {
    let mut w = pile(2000);
    w.prm.allow_sleep = false;
    if std::env::var("NOANCH").is_ok() { w.prm.anchor = false; }
    if let Ok(v) = std::env::var("AP") { w.prm.anchor_push = v.parse().unwrap(); }
    if let Ok(v) = std::env::var("SLOP") { w.prm.anchor_slop = v.parse().unwrap(); }
    if let Ok(v) = std::env::var("SUB") { w.prm.substeps = v.parse().unwrap(); }
    for _ in 0..(14 * 60) { w.step(1.0 / 60.0); }
    let mut idx: Vec<usize> = (0..w.bodies.len()).collect();
    idx.sort_by(|&a, &b| w.bodies[b].v.len().partial_cmp(&w.bodies[a].v.len()).unwrap());
    let watch: Vec<usize> = idx[..6].to_vec();
    let fast = w.bodies.iter().filter(|b| b.v.len() > 0.02).count();
    println!("bodies above 2 cm/s at 14 s: {fast}, max {:.3}", w.bodies.iter().map(|b| b.v.len()).fold(0.0, f32::max));
    for f in 0..6 {
        w.step(1.0 / 60.0);
        let mut line = format!("f{f:>2}");
        for &i in &watch { let b = &w.bodies[i]; line += &format!(" | #{i} y{:.3} v({:+.3},{:+.3},{:+.3}) w{:.2}", b.p.y, b.v.x, b.v.y, b.v.z, b.w.len()); }
        println!("{line}");
    }
}
