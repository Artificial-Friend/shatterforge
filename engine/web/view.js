// ---- the structure as the engineer sees it (key V; docs/DYNAMIC.md): each brick coloured by how near its worst joint
// or section is to its strength, and the building drawn where the structure's solution puts it, its displacements
// VIEW_SCALE[setting.viewScale] times as large. The engine writes it with each structure step (set_param 8); the
// page's copy of the engine draws it (view_apply) into the chunks' meshes.
const VIEW_SCALE = [0, 1, 10, 100, 1000];
// (13 Nov: with the dynamic structure the building is drawn where it has sagged without the view too, in its own
// colours and at its true size: the engine then writes only the lots that have moved 2 cm, set_param 8 at 2)
function drawView() { if (+setting.view) E.view_apply(VIEW_SCALE[setting.viewScale] || 0, 1); else E.view_apply(1, 0); }
function viewParam() { return +setting.view ? 1 : (+setting.dynamic ? 2 : 0); }
// the motion of the structure's cell at a point, for a part that leaves it there (from the view's cells: only lots
// that have sagged to see are in it)
function motionAt(c) {
  if (!E.view_motion || !viewParam() || !E.view_motion(c[0], c[1], c[2])) return { v: [0, 0, 0], w: [0, 0, 0] };
  const m = new Float32Array(E.memory.buffer, E.motion_ptr(), 12);
  const sc = +setting.view ? (VIEW_SCALE[setting.viewScale] || 0) : 1;
  return { u: [m[0] * sc, m[1] * sc, m[2] * sc], r: [m[3] * sc, m[4] * sc, m[5] * sc], v: [m[6], m[7], m[8]], w: [m[9], m[10], m[11]] };
}
function showView(v) {
  if (!E.view_in || (v[0] === 0 && v[1] === 0)) return;   // (nothing changed in the step)
  const p = E.view_in(v.length); new Float32Array(E.memory.buffer, p, v.length).set(v);
  drawView();
  // (drawn at once: remeshDirty waits for the read-back that shows the debris of a change, and a building at rest
  // has none coming)
  // (without the view a lot is written because it moved, and what moved may be about to leave as bodies: its chunks
  // wait for the read-back that shows those, as every change's do. Drawn at once, the lab's ball had 26 frames
  // with less drawn than exists, 13 Nov.)
  const later = gpu && !+setting.view;
  for (let i = 0, n = E.view_chunk_count(); i < n; i++) { if (later) pendingChunks.push({ k: E.view_chunk(i), need: readSeq + 1 }); else chunkMesh(E.view_chunk(i)); }
}
let viewShown = false;
function applyView() {
  if (!E || !E.view_off) return;
  const was = viewShown;
  viewShown = !!+setting.view;
  if (worker) worker.postMessage({ op: 'param', id: 8, v: viewParam() }); else E.set_param(8, viewParam());
  // (a structure step is asked for only when the view itself is switched: at start-up the settings are applied
  // before the building's first step, and a step asked for here ran ahead of it: its loose furniture, 208 bodies,
  // never reached the GPU; page check rest, 13 Nov)
  if (viewParam()) { if (was || viewShown) { drawView(); for (let k = 0; k < E.chunk_count(); k++) chunkMesh(k); jobs.structure = true; pump(); } }
  else { E.view_off(); for (let k = 0; k < E.chunk_count(); k++) chunkMesh(k); }
}

