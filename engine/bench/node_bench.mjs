// Phase-0 gates in WebAssembly under Node: split timing, brick memory, 20-stack, pile throughput.
// Usage: node bench/node_bench.mjs [path/to/sf_engine.wasm]
import { readFileSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
const path = process.argv[2] || new URL('../../target/wasm32-unknown-unknown/release/sf_demo.wasm', import.meta.url);
const { instance } = await WebAssembly.instantiate(readFileSync(path), {});
const e = instance.exports;
const pct = (a, p) => { const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(s.length * p))]; };

// split gate
const st = [];
let bricks = 0;
for (let k = 0; k < 100; k++) { e.split_setup(); e.split_run(8 + (k % 45)); }
for (let k = 0; k < 400; k++) { e.split_setup(); const t = performance.now(); bricks = e.split_run(8 + (k % 45)); st.push(performance.now() - t); }
console.log(`split: ${bricks} bricks after refine; median ${pct(st, .5).toFixed(3)} ms, p99 ${pct(st, .99).toFixed(3)} ms (gate 1 ms); ${e.split_bytes_per_brick().toFixed(1)} B/brick (gate 96)`);

// stacking gate
for (const sleep of [0, 1]) {
  e.stack_setup(20, 12, sleep);
  const top0 = e.body_pos(19, 1);
  let mv = 0;
  for (let f = 0; f < 600; f++) { e.step(1 / 60); if (f >= 540) mv = Math.max(mv, e.max_speed()); }
  const dx = e.body_pos(19, 0), dy = e.body_pos(19, 1) - top0, dz = e.body_pos(19, 2);
  console.log(`stack 20 (sleep ${sleep}): drift ${(Math.hypot(dx, dy, dz) * 1e3).toFixed(2)} mm (gate 10), max speed last second ${(mv * 1e3).toFixed(2)} mm/s (gate 10)`);
}

// throughput
for (const [n, sub] of [[2000, 4], [2000, 12], [20000, 4], [20000, 12]]) {
  e.pile_setup(n, sub);
  const t = [];
  const frames = n > 5000 ? 90 : 180;
  for (let f = 0; f < frames; f++) { const a = performance.now(); e.step(1 / 60); if (f >= frames / 3) t.push(performance.now() - a); }
  console.log(`pile ${n}, ${sub} substeps: median ${pct(t, .5).toFixed(2)} ms, p95 ${pct(t, .95).toFixed(2)} ms per 60 Hz step; ${e.manifolds()} manifolds, ${e.points()} points`);
}
