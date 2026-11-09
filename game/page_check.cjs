// Checks of the page itself, as a player uses it (not the lab, which reads back every frame and so never rests):
//   node game/page_check.cjs rest     the world at rest, then a click: the ball must fly and be read back
//   node game/page_check.cjs debris   the player dropped onto a slab on the GPU (debris): it must stand on it
// SF_NODE_MODULES: a node_modules folder with three (and Playwright installed), as for game/lab.cjs.
//
// rest: the complex building (kind 4) has furniture that falls asleep while the page still steps; the GPU then skips
// its steps, no read-backs are issued, and the page, which steps at most 3 past the newest read-back, stopped
// stepping for good: a ball fired then never stepped or showed ("clicking does nothing"; the benchmark stood still
// at 39.6 s the same way). Before the fix: 3 steps skipped, no read-back in the 20 s after the click; after: read-backs
// resume and the ball hits the arcade.
const { chromium } = require(process.env.PW || 'playwright'); const fs = require('fs'), path = require('path');
const root = path.resolve(__dirname, '..');
const which = process.argv[2] || 'rest';
(async () => {
  const html = path.join(root, 'lab-results', 'page_check.html');
  fs.mkdirSync(path.dirname(html), { recursive: true });
  fs.writeFileSync(html, '<!doctype html><html><head><meta charset="utf-8"></head><body style="margin:0">' + fs.readFileSync(path.join(root, 'dist/tower.html'), 'utf8') + '</body></html>');
  const b = await chromium.launch({ args: ['--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--enable-features=Vulkan'] });
  const p = await b.newPage({ viewport: { width: 400, height: 250 } });
  let err = null; p.on('pageerror', e => { err = e.message; });
  let threeJs; try { threeJs = fs.readFileSync(require.resolve('three/build/three.min.js'), 'utf8'); } catch (e) { threeJs = fs.readFileSync(path.join(process.env.SF_NODE_MODULES || '', 'three/build/three.min.js'), 'utf8'); }
  await p.route('**/three.min.js', r => r.fulfill({ body: threeJs, contentType: 'text/javascript' }));
  await p.addInitScript(() => { localStorage.setItem('sf-settings', JSON.stringify({ kind: 4 })); });
  await p.goto('file://' + html);
  for (let i = 0; i < 180; i++) { await new Promise(r => setTimeout(r, 1000)); if (await p.evaluate(() => !!gpu && !$('go').disabled)) break; }
  await p.evaluate(() => $('go').click());
  if (which === 'debris') {
    // debris: a 3 x 0.3 x 3 m slab held 5 m up beside the building (a fixed body: debris that does not fall), the player
    // dropped onto it from 7 m. Before: the walker saw only the standing bricks and fell to the ground (feet at 0);
    // now it stands on the slab, its feet at 5.15 m.
    await p.evaluate(() => { gpu.addBodies([{ fixed: true, shapes: [{ p: [20, 5, 0], q: [0, 0, 0, 1], h: [1.5, 0.15, 1.5], mat: 0 }], v: [0, 0, 0], w: [0, 0, 0] }]); lastActive = clock(); E.player_set(20, 7, 0); fly = false; });
    for (let i = 0; i < 30; i++) { await new Promise(r => setTimeout(r, 1000)); await p.evaluate(() => { lastActive = clock(); }); }
    const y = await p.evaluate(() => E.player_get(1));
    const ok = !err && Math.abs(y - 5.15) < 0.05;
    console.log(`${ok ? 'PASS' : 'FAIL'} debris: the player's feet at ${y.toFixed(3)} m on a slab whose top is 5.15 m${err ? '; page error: ' + err : ''}`);
    await b.close(); process.exit(ok ? 0 : 1);
  }
  const st = () => p.evaluate(() => ({ bodies: gpu.bodies.length, awake: gpu.stats.awake, skipped: gpu.skipped || 0, stepsDone, stepsAtRead, readSeq }));
  let s = await st();
  for (let i = 0; i < 600 && s.awake !== 0; i++) { await new Promise(r => setTimeout(r, 1000)); s = await st(); }
  // (a fast GPU gets here while the page still steps, its last activity under 2 s ago: so does this check)
  for (let i = 0; i < 120; i++) { await p.evaluate(() => { lastActive = clock(); }); await new Promise(r => setTimeout(r, 500)); s = await st(); if (s.skipped >= 4) break; }
  await p.evaluate(() => { locked = true; setRadius('ball'); });
  await p.mouse.click(200, 125);
  const r0 = s.readSeq;
  for (let i = 0; i < 20; i++) { await new Promise(r => setTimeout(r, 1000)); }
  const s1 = await st();
  const ok = !err && s.skipped >= 3 && s1.readSeq > r0 && s1.bodies > s.bodies;
  console.log(`${ok ? 'PASS' : 'FAIL'} rest: at rest ${JSON.stringify(s)}; 20 s after a click ${JSON.stringify(s1)}${err ? '; page error: ' + err : ''}`);
  await b.close();
  process.exit(ok ? 0 : 1);
})();
