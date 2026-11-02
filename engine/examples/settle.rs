// How a big pile settles: awake count and speed distribution per second (sleeping on).
use sf_engine::phys::pile;
fn main() {
    let n: usize = std::env::args().nth(1).and_then(|v| v.parse().ok()).unwrap_or(20000);
    let secs: usize = std::env::args().nth(2).and_then(|v| v.parse().ok()).unwrap_or(20);
    let mut w = pile(n);
    w.prm.allow_sleep = true;
    if let Ok(v) = std::env::var("SD") { w.prm.sleep_dist = v.parse().unwrap(); }
    if let Ok(v) = std::env::var("SR") { w.prm.sleep_rot = v.parse().unwrap(); }
    for s in 0..secs {
        for _ in 0..60 { w.step(1.0 / 60.0); }
        let mut sp: Vec<f32> = w.bodies.iter().filter(|b| b.awake).map(|b| b.v.len()).collect();
        sp.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |p: f32| if sp.is_empty() { 0.0 } else { sp[((sp.len() - 1) as f32 * p) as usize] };
        let fast = sp.iter().filter(|&&v| v > 0.05).count();
        let low: Vec<&sf_engine::phys::Body> = w.bodies.iter().filter(|b| b.awake && b.sleep_t < 0.5).collect();
        let creep = low.iter().map(|b| b.v.len()).fold(0.0f32, f32::max);
        let worst = low.iter().map(|b| { let dq = b.q.mul(b.sleep_q.conj()); ((b.p - b.sleep_p).len(), 2.0 * (dq.x * dq.x + dq.y * dq.y + dq.z * dq.z).sqrt(), b.sleep_t) }).fold((0.0f32, 0.0f32, 9.0f32), |a, x| (a.0.max(x.0), a.1.max(x.1), a.2.min(x.2)));
        print!("[{} not still, fastest {:.3} m/s, drift {:.4} m, turn {:.4} rad] ", low.len(), creep, worst.0, worst.1);
        println!("t {:>2}s awake {:>5}  >5cm/s {:>5}  speed p50 {:.3} p99 {:.3} max {:.3} m/s", s + 1, sp.len(), fast, q(0.5), q(0.99), q(1.0));
    }
}
