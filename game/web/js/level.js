// the level: the Quarter (kind 6; game/src/quarter.rs). The lab's scenarios may ask for one of the engine's test
// buildings instead (kinds 0 to 5, see LAB_SCENARIOS); the game has the one level.
let KIND = 6;
if (LABSTATE) KIND = LAB_SCENARIOS[LABSTATE.queue[0]].kind;
// (a test building's storeys; the test city's size, lots a side)
const STOREYS = LABSTATE ? (LAB_SCENARIOS[LABSTATE.queue[0]].storeys || 3) : 3;
const KIND_NAMES = ['concrete frame', 'masonry', 'mass timber', 'mixed', 'complex', 'city', 'the Quarter'];
// (levels of several lots: the test city and the Quarter)
const TILED = KIND === 5 || KIND === 6;
// what stands in the Quarter, in the order of quarter::What, and where (from the engine, once it is there)
const QUARTER_WHAT = ['villa', 'glass', 'barn', 'aqueduct', 'hall', 'mill', 'chimney', 'tower', 'temple', 'office', 'tenement'];
let QUARTER = null;
function quarter() {
  if (QUARTER) return QUARTER;
  QUARTER = {};
  if (E && E.quarter_things) for (let k = 0; k < E.quarter_things(); k++) QUARTER[QUARTER_WHAT[E.quarter_thing(k, 0)]] = { lot: E.quarter_thing(k, 1), x: E.quarter_thing(k, 2), z: E.quarter_thing(k, 3), q: E.quarter_thing(k, 4) };
  return QUARTER;
}
// a point of one of its buildings' designs (m) in the world: turned with the building and moved to its place
function quarterAt(name, x, y, z) {
  const t = quarter()[name]; if (!t) return [x, y, z];
  const [c, s] = [[1, 0], [0, 1], [-1, 0], [0, -1]][t.q & 3];
  return [t.x + c * x - s * z, y, t.z + s * x + c * z];
}
// a direction of a building's design in the world
function quarterDir(name, x, y, z) { const t = quarter()[name]; if (!t) return [x, y, z]; const [c, s] = [[1, 0], [0, 1], [-1, 0], [0, -1]][t.q & 3]; return [c * x - s * z, y, s * x + c * z]; }
// the lots: where each building stands (one at the origin but in the city and the Quarter), from the engine's layout
let LOTS = null;
function lots() {
  if (LOTS) return LOTS;
  if (KIND === 6 && E && E.quarter_lot) { LOTS = []; for (let k = 0; k < 9; k++) LOTS.push({ kind: 6, x: E.quarter_lot(k, 0), z: E.quarter_lot(k, 1) }); return LOTS; }
  if (KIND !== 5 || !E || !E.city_lots) return (LOTS = [{ x: 0, z: 0, kind: KIND }]);
  LOTS = []; for (let k = 0; k < E.city_lots(STOREYS); k++) LOTS.push({ kind: E.city_lot(STOREYS, k, 0), storeys: E.city_lot(STOREYS, k, 1), x: E.city_lot(STOREYS, k, 3), z: E.city_lot(STOREYS, k, 4) });
  return LOTS;
}
let E = null, gpu = null, dev = null, slots = new Map(), debrisFrom = 2;
let radius = 0.7, fly = false, locked = false, yaw = Math.PI, pitch = 0, physicsPaused = true, lastActive = -1e9;
const keys = new Set();

