import fs from 'fs';
const { instance } = await WebAssembly.instantiate(fs.readFileSync(new URL('../target/wasm32-unknown-unknown/release/shatterforge.wasm', import.meta.url)), { env: { now: () => performance.now() } });
const e = instance.exports;
const F = (p, n) => new Float32Array(e.memory.buffer, p, n);
const pl = () => Array.from(F(e.ptr_player(), 4));
e.set_ai(0); e.init();
const [sx, sz, route] = JSON.parse(process.argv[2]);
e.dbg_tp(sx, 0.2, sz);
for (let i = 0; i < 30; i++) e.step(1 / 60);
for (const [tx, tz] of route) {
  let n = 0;
  for (; n < 600; n++) {
    const p = pl(); const dx = tx - p[0], dz = tz - p[2]; const d = Math.hypot(dx, dz);
    if (d < 0.15) break;
    const sp = Math.min(1, d * 2);
    e.set_move(dx / d * sp, dz / d * sp, 0, 0); e.step(1 / 60); e.clear_events();
  }
  const p = pl();
  console.log(`-> (${tx},${tz}) at ${p.slice(0, 3).map(v => v.toFixed(2)).join(',')} ${n >= 600 ? 'STUCK' : ''}`);
}
