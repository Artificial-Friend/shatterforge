use sf_engine::{tower, structure, math::*};
fn main() {
    let b = tower::build(12, 1);
    let set: Vec<u32> = (0..b.len() as u32).filter(|&i| b.alive(i)).collect();
    let load: Vec<f32> = set.iter().map(|&i| structure::mass(&b, i) * 9.81).collect();
    let sink: Vec<bool> = set.iter().map(|&i| b.y[i as usize] <= 0).collect();
    let f = structure::solve(&b, &set, &load, &sink, v3(0.0, -1.0, 0.0));
    let mut c = structure::bending(&b, &f, &sink, &set);
    c.sort_by(|a, b| b.ratio.partial_cmp(&a.ratio).unwrap());
    let mut kinds = std::collections::BTreeMap::new();
    for x in &c { *kinds.entry((b.mat[x.brick as usize], b.paint[x.brick as usize], b.size(x.brick))).or_insert(0) += 1; }
    for (k, v) in &kinds { println!("{v} cracks in mat {} paint {} size {:?}", k.0, k.1, k.2); }
    for x in c.iter().take(5) { println!("ratio {:.2} brick {:?} axis {} at {}", x.ratio, b.tbox(x.brick), x.axis, x.at); }
}
