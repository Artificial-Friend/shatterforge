use sf_engine::{tower, structure, math::*, brick::TICK};
fn main() {
    let b = tower::build(12, 1);
    let set: Vec<u32> = (0..b.len() as u32).filter(|&i| b.alive(i)).collect();
    let load: Vec<f32> = set.iter().map(|&i| structure::mass(&b, i) * 9.81).collect();
    let sink: Vec<bool> = set.iter().map(|&i| b.y[i as usize] <= 0).collect();
    let f = structure::solve(&b, &set, &load, &sink, v3(0.0, -1.0, 0.0));
    let mut v: Vec<(f32, u32, f32)> = f.joints.iter().map(|&(j, force)| {
        let ju = j as usize; let area = b.jarea[ju] as f32 * TICK * TICK;
        let mut ax = V3::ZERO; ax.set(b.jaxis[ju] as usize, 1.0); let along = force * ax.dot(v3(0.0, -1.0, 0.0));
        let mode = if along > 0.5 * force.abs() { 0 } else if along < -0.5 * force.abs() { 1 } else { 2 };
        let cap = structure::strength(b.mat[b.ja[ju] as usize])[mode].min(structure::strength(b.mat[b.jb[ju] as usize])[mode]);
        (force.abs() / area / cap, j, mode as f32) }).collect();
    v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    for &(mpa, j, force) in v.iter().take(8) {
        let ju = j as usize; let (a, bb) = (b.ja[ju], b.jb[ju]);
        println!("ratio {mpa:.2} mode {force} axis {} area {:.3} m2 | a mat {} paint {} {:?} | b mat {} paint {} {:?}", b.jaxis[ju], b.jarea[ju] as f32 * TICK * TICK,
            b.mat[a as usize], b.paint[a as usize], b.tbox(a), b.mat[bb as usize], b.paint[bb as usize], b.tbox(bb));
    }
}
