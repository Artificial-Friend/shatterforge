// The engine's hull library (engine/src/hull.rs, through the wasm) against the GPU's (hullLibrary in
// gpu_world.js): the same cells, boxes, volumes and centroids. (The GPU keeps a cell's volume as a share of its
// own box, the engine as a share of the cube.) A cell's hull id must mean the same shape in the
// engine (which splits fractured bricks into cells) and on the GPU (which collides them).
//   node engine/gpu/test_hulls.cjs      (after the wasm build)
const fs = require('fs'), path = require('path');
const src = fs.readFileSync(path.join(__dirname, 'gpu_world.js'), 'utf8');
const a = src.indexOf('const HULL = '), b = src.indexOf('const HULLS = hullLibrary();');
const HULLS = new Function(src.slice(a, b) + '\nreturn hullLibrary();')();
const wasm = fs.readFileSync(path.join(__dirname, '../../target/wasm32-unknown-unknown/release/sf_demo.wasm'));
WebAssembly.instantiate(wasm, {}).then(({ instance }) => {
  const E = instance.exports, n = E.hull_count();
  let worst = 0, bad = [];
  if (n !== HULLS.count) bad.push(`count ${n} against ${HULLS.count}`);
  for (let id = 0; id < Math.min(n, HULLS.count); id++) {
    const H = HULLS.hulls[id], c = H.box.c, h = H.box.h;
    const js = [...c, ...h, H.vol * h[0] * h[1] * h[2], ...H.cen.map((x, k) => c[k] + x * h[k])];
    for (let k = 0; k < 10; k++) { const d = Math.abs(E.hull_value(id, k) - js[k]); worst = Math.max(worst, d); if (d > 1e-9) bad.push(`hull ${id} value ${k}: ${E.hull_value(id, k)} against ${js[k]}`); }
  }
  for (let p = 0; p < HULLS.patterns.length; p++) for (const id of HULLS.patterns[p]) if (E.hull_value(id, 10) !== p) bad.push(`hull ${id}: pattern ${E.hull_value(id, 10)} against ${p}`);
  console.log(`${bad.length ? 'FAIL' : 'PASS'} hull library: ${n} hulls, engine and GPU agree to ${worst.toExponential(1)}${bad.length ? ': ' + bad.slice(0, 5).join('; ') : ''}`);
  process.exit(bad.length ? 1 : 0);
});
