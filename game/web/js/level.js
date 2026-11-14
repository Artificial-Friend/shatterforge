// the building (remembered): 0 concrete frame, 1 masonry, 2 timber, each at a height it can stand
let KIND = 0; try { KIND = +(JSON.parse(localStorage.getItem('sf-settings') || '{}').kind || 0); } catch (e) {}
if (![0, 1, 2, 3, 4, 5].includes(KIND)) KIND = 0;
let CITY = 3; try { CITY = +(JSON.parse(localStorage.getItem('sf-settings') || '{}').city || 3); } catch (e) {}
if (!(CITY >= 1 && CITY <= 16)) CITY = 3;
if (LABSTATE) KIND = LAB_SCENARIOS[LABSTATE.queue[0]].kind;
// (the city: its size, lots a side, goes to the engine as its storeys)
const STOREYS = LABSTATE ? LAB_SCENARIOS[LABSTATE.queue[0]].storeys : [12, 5, 6, 5, 3, CITY][KIND];
const KIND_NAMES = ['concrete frame', 'masonry', 'mass timber', 'mixed', 'complex', 'city'];
// the lots: where each building stands (one at the origin but in the city), from the engine's layout
let LOTS = null;
function lots() {
  if (LOTS) return LOTS;
  if (KIND !== 5 || !E || !E.city_lots) return (LOTS = [{ x: 0, z: 0, kind: KIND }]);
  LOTS = []; for (let k = 0; k < E.city_lots(STOREYS); k++) LOTS.push({ kind: E.city_lot(STOREYS, k, 0), storeys: E.city_lot(STOREYS, k, 1), x: E.city_lot(STOREYS, k, 3), z: E.city_lot(STOREYS, k, 4) });
  return LOTS;
}
let E = null, gpu = null, dev = null, slots = new Map(), debrisFrom = 2;
let radius = 0.7, fly = false, locked = false, yaw = Math.PI, pitch = 0, physicsPaused = true, lastActive = -1e9;
const keys = new Set();

