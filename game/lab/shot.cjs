// screenshot of the page with a given building kind (software GPU): node shot.cjs <kind> <out.png> [wait s] [eval]
const path = require('path'), fs = require('fs');
const NM = process.env.SF_NODE_MODULES;
const { chromium } = require('playwright');
const [kind, out, wait, ev] = process.argv.slice(2);
(async () => {
  const page0 = fs.readFileSync(path.join(__dirname, '..', '..', 'dist/tower.html'), 'utf8');
  const html = path.join(path.dirname(out), 'shot_page.html');
  fs.writeFileSync(html, '<!doctype html><html><head><meta charset="utf-8"></head><body style="margin:0">' + page0 + '</body></html>');
  const b = await chromium.launch({ args: ['--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--enable-features=Vulkan'] });
  const p = await b.newPage({ viewport: { width: 1100, height: 650 } });
  p.on('pageerror', e => console.log('page error: ' + e.message));
  const threeJs = fs.readFileSync(path.join(NM, 'three/build/three.min.js'), 'utf8');
  await p.route('**/three.min.js', r => r.fulfill({ body: threeJs, contentType: 'text/javascript' }));
  await p.addInitScript((k) => { localStorage.setItem('sf-settings', JSON.stringify({ kind: +k })); localStorage.removeItem('sf-lab'); }, kind);
  await p.goto('file://' + html);
  await new Promise(r => setTimeout(r, (+wait || 40) * 1000));
  if (ev) { await p.evaluate(ev); await new Promise(r => setTimeout(r, 20000)); }
  await p.screenshot({ path: out });
  await b.close();
})();
