// The lab, headless: node game/lab.cjs [quick|full|scenario ...] [--read-lag N] [--job-lag N] [--out DIR]
// Runs each scenario of the page's lab (game/lab.js) in headless Chrome on the software GPU, on the lab's
// virtual clock (so slowness changes nothing but the wall time), prints the checks and writes the full reports
// (events, time series) as JSON. Exit code 1 when a check fails.
const { chromium } = require('playwright');
const fs = require('fs'), path = require('path');
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); if (i < 0) return d; const v = args[i + 1]; args.splice(i, 2); return v; };
const readLag = +opt('--read-lag', 2), jobLag = +opt('--job-lag', 4), outDir = opt('--out', path.join(__dirname, '..', 'lab-results'));
const frames = opt('--frames', null), trace = opt('--trace', null);
const root = path.join(__dirname, '..');
const page0 = fs.readFileSync(path.join(root, 'dist/tower.html'), 'utf8');
const SUITES = { quick: ['collapse', 'ball'], full: ['collapse', 'ball', 'masonry', 'timber', 'mixed', 'rubble'] };
const names = args.length ? args.flatMap(a => SUITES[a] || [a]) : SUITES.quick;
(async () => {
  fs.mkdirSync(outDir, { recursive: true });
  const html = path.join(outDir, 'lab_page.html');
  fs.writeFileSync(html, '<!doctype html><html><head><meta charset="utf-8"></head><body style="margin:0">' + page0 + '</body></html>');
  const b = await chromium.launch({ args: ['--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--enable-features=Vulkan'] });
  const results = []; let failed = false;
  for (const name of names) {
    const p = await b.newPage({ viewport: { width: 800, height: 500 } });
    p.on('pageerror', e => console.log(`[${name}] page error: ${e.message}`));
    let threeJs; try { threeJs = fs.readFileSync(require.resolve('three/build/three.min.js'), 'utf8'); } catch (e) { threeJs = fs.readFileSync(path.join(process.env.SF_NODE_MODULES || '', 'three/build/three.min.js'), 'utf8'); }
    await p.route('**/three.min.js', r => r.fulfill({ body: threeJs, contentType: 'text/javascript' }));
    // one scenario per page load, queued the way the page's own button queues them (no render: nothing looks)
    await p.addInitScript((nf) => { if (nf) window.__noFracCells = true; }, !!process.env.SF_NO_FRAC_CELLS);
    await p.addInitScript((ns) => { if (ns) window.__noSnap = true; }, !!process.env.SF_NO_SNAP);   // (compare without breaking on the GPU)   // (compare without irregular fracture)
    await p.addInitScript(([name, readLag, jobLag, frames, trace]) => {
      if (sessionStorage.getItem('lab-set')) return; sessionStorage.setItem('lab-set', '1');
      localStorage.setItem('sf-lab', JSON.stringify({ queue: [name], done: [], results: [], readLag, jobLag, render: false, external: true, frames, trace }));
    }, [name, readLag, jobLag, frames ? +frames : null, trace ? trace.split(':').map(Number) : null]);
    const t0 = Date.now();
    await p.goto('file://' + html);
    let last = '';
    for (;;) {
      await new Promise(r => setTimeout(r, 5000));
      const st = await p.evaluate(() => ({ done: !!window.LAB_RESULT, err: window.LAB_ERROR || null, s: LAB.status || null }));
      if (st.err) { console.log(`[${name}] lab error: ${st.err}`); failed = true; break; }
      if (st.done) break;
      const line = st.s ? `[${name}] frame ${st.s.f} of ${st.s.frames} · ${((Date.now() - t0) / 1000).toFixed(0)} s` : '';
      if (line && Date.now() - (this.lastPrint || 0) > 60000) { console.log(line); this.lastPrint = Date.now(); }
    }
    const rep = await p.evaluate(() => window.LAB_RESULT || null);
    if (rep) {
      results.push(rep); if (!rep.pass) failed = true;
      console.log(await p.evaluate((r) => labSummary(r), rep));
      fs.writeFileSync(path.join(outDir, `${name}.json`), JSON.stringify(rep, null, 1));
    }
    await p.close();
  }
  await b.close();
  fs.writeFileSync(path.join(outDir, 'all.json'), JSON.stringify(results));
  console.log(`\n${results.filter(r => r.pass).length} of ${names.length} scenarios passed · reports in ${outDir}`);
  process.exit(failed ? 1 : 0);
})();
