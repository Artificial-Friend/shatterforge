//! Shatterforge engine: the brick store, the structure (at rest and dynamic), break plans, meshes and the
//! simulation that ties them (`sim::Sim`). No levels, no player, no page: those are the demo's (game/).
#![cfg_attr(target_arch = "wasm32", no_std)]
extern crate alloc;

pub mod math;

/// Hash maps and sets with a fixed hasher (FxHash). The default one is seeded per run, and the order a map is
/// walked in changed sums over its entries (the solver's coarse levels) in their last bits: the same game gave
/// different results run to run.
#[derive(Default, Clone, Copy)]
pub struct Fx(u64);
const FX_K: u64 = 0x517c_c1b7_2722_0a95;
impl core::hash::Hasher for Fx {
    #[inline] fn finish(&self) -> u64 { self.0 }
    #[inline] fn write(&mut self, b: &[u8]) { for &x in b { self.0 = (self.0.rotate_left(5) ^ x as u64).wrapping_mul(FX_K); } }
    #[inline] fn write_u8(&mut self, i: u8) { self.0 = (self.0.rotate_left(5) ^ i as u64).wrapping_mul(FX_K); }
    #[inline] fn write_u32(&mut self, i: u32) { self.0 = (self.0.rotate_left(5) ^ i as u64).wrapping_mul(FX_K); }
    #[inline] fn write_u64(&mut self, i: u64) { self.0 = (self.0.rotate_left(5) ^ i).wrapping_mul(FX_K); }
    #[inline] fn write_i32(&mut self, i: i32) { self.write_u32(i as u32); }
    #[inline] fn write_usize(&mut self, i: usize) { self.write_u64(i as u64); }
}
pub type Map<K, V> = hashbrown::HashMap<K, V, core::hash::BuildHasherDefault<Fx>>;
pub type Set<K> = hashbrown::HashSet<K, core::hash::BuildHasherDefault<Fx>>;
pub mod brick;
pub mod phys;
pub mod mesh;
pub mod sim;
pub mod structure;
pub mod hull;
mod validation;

// The engine's tests run on the demo's buildings: its level files are taken in as fixtures, for tests only. They
// reach the engine as `crate::brick` and so on, which is what the demo crate re-exports under the same names.
#[cfg(test)] #[path = "../../game/src/tower.rs"] pub mod tower;
#[cfg(test)] #[path = "../../game/src/complex.rs"] pub mod complex;
#[cfg(test)] #[path = "../../game/src/city.rs"] pub mod city;
#[cfg(test)] #[path = "../../game/src/player.rs"] pub mod player;
#[cfg(test)] mod dyn_tests;
#[cfg(test)] mod dyn_closed;
