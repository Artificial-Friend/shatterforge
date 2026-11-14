use sf_demo::{tower, structure, math::*};
fn main() {
    let b = tower::build(12, 1);
    let set: Vec<u32> = (0..b.len() as u32).filter(|&i| b.alive(i)).collect();
    let load: Vec<f32> = set.iter().map(|&i| structure::mass(&b, i) * 9.81).collect();
    let sink: Vec<bool> = set.iter().map(|&i| b.y[i as usize] <= 0).collect();
    let a = structure::analyse(&b, &set, &load, &sink, v3(0.0, -1.0, 0.0));
    let mut f = a.failed.clone(); f.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
    let mut kinds = std::collections::BTreeMap::new();
    for &(j, _) in &f { let ju = j as usize; *kinds.entry((b.jaxis[ju], b.paint[b.ja[ju] as usize], b.paint[b.jb[ju] as usize], b.flags[b.ja[ju] as usize] & 2, b.flags[b.jb[ju] as usize] & 2)).or_insert(0) += 1; }
    for (k, v) in &kinds { println!("{v} failed joints: axis {} paint a {} b {} infill {} {}", k.0, k.1, k.2, k.3, k.4); }
    for &(j, r) in f.iter().take(6) { let ju = j as usize; println!("ratio {r:.1} axis {} a {:?} p{} b {:?} p{}", b.jaxis[ju], b.tbox(b.ja[ju]), b.paint[b.ja[ju] as usize], b.tbox(b.jb[ju]), b.paint[b.jb[ju] as usize]); }
    for c in a.cracks.iter().take(5) { println!("crack ratio {:.2} brick {:?} axis {} at {}", c.ratio, b.tbox(c.brick), c.axis, c.at); }
}
