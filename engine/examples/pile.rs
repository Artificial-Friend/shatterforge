use sf_engine::phys::pile;
use std::time::Instant;
fn clk() -> f64 { use std::sync::OnceLock; static T: OnceLock<Instant> = OnceLock::new(); T.get_or_init(Instant::now).elapsed().as_secs_f64() * 1e3 }
fn main() {
    let n: usize = std::env::args().nth(1).and_then(|v| v.parse().ok()).unwrap_or(2000);
    let steps: usize = std::env::args().nth(2).and_then(|v| v.parse().ok()).unwrap_or(120);
    let sub: u32 = std::env::args().nth(3).and_then(|v| v.parse().ok()).unwrap_or(12);
    let mut w = pile(n);
    w.prm.allow_sleep = false;
    w.prm.substeps = sub;
    w.clock = clk;
    let (mut tc, mut ts, mut tn, mut k) = (0.0, 0.0, 0.0, 0);
    let mut t = Vec::new();
    for f in 0..steps { let s = Instant::now(); w.step(1.0 / 60.0); if f >= steps / 3 { t.push(s.elapsed().as_secs_f64() * 1e3); tc += w.stats.t_broad; ts += w.stats.t_solve; tn += w.stats.t_narrow; k += 1; } }
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("pile {n}, {sub} substeps: median {:.2} ms (collide {:.2} of which narrow {:.2}, solve {:.2}), awake {}, manifolds {}, points {}", t[t.len() / 2], tc / k as f64, tn / k as f64, ts / k as f64, w.stats.awake, w.stats.manifolds, w.stats.points);
}
