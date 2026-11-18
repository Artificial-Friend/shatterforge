// several views of the page in one session: node shots.cjs <outdir> name:x,y,z,tx,ty,tz ...
const path = require('path'), fs = require('fs');
const root = '/home/claude/shatterforge';
const NM = process.env.SF_NODE_MODULES;
const { chromium } = require('playwright');
const [out, ...views] = process.argv.slice(2);
(async () => {
  const page0 = fs.readFileSync(path.join(root, 'dist/tower.html'), 'utf8');
  const html = path.join(out, 'shot_page.html');
  fs.writeFileSync(html, '<!doctype html><html><head><meta charset="utf-8"></head><body style="margin:0">' + page0 + '</body></html>');
  const b = await chromium.launch({ args: ['--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--enable-features=Vulkan'] });
  const p = await b.newPage({ viewport: { width: 1100, height: 650 } });
  p.on('pageerror', e => console.log('page error: ' + e.message));
  const threeJs = fs.readFileSync(path.join(NM, 'three/build/three.min.js'), 'utf8');
  await p.route('**/three.min.js', r => r.fulfill({ body: threeJs, contentType: 'text/javascript' }));
  await p.addInitScript(() => { localStorage.removeItem('sf-lab'); });
  await p.goto('file://' + html);
  for (let i = 0; i < 180; i++) { await new Promise(r => setTimeout(r, 1000)); if (await p.evaluate(() => typeof gpu !== 'undefined' && !!gpu && !document.getElementById('go').disabled)) break; }
  await p.evaluate(() => { document.getElementById('start').hidden = true; document.getElementById('hud').style.display = 'none'; fly = true; });
  for (const v of views) {
    const [name, nums] = v.split(':'); const [x, y, z, tx, ty, tz] = nums.split(',').map(Number);
    await p.evaluate(([x, y, z, tx, ty, tz]) => {
      const dx = tx - x, dy = ty - y, dz = tz - z, hl = Math.hypot(dx, dz);
      E.player_set(x, y, z); yaw = Math.atan2(-dx, -dz); pitch = Math.atan2(dy, hl);
    }, [x, y, z, tx, ty, tz]);
    await new Promise(r => setTimeout(r, 6000));
    await p.screenshot({ path: path.join(out, name + '.png') });
    console.log('shot ' + name);
  }
  await b.close();
})();
