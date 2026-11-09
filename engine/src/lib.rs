//! Shatterforge engine (phase 0 prototypes): adaptive brick store and box rigid-body solver.
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
pub mod tower;
pub mod complex;
pub mod city;
pub mod mesh;
pub mod player;
pub mod game;
pub mod structure;
pub mod hull;
mod validation;

#[cfg(target_arch = "wasm32")]
#[global_allocator]
static ALLOC: dlmalloc::GlobalDlmalloc = dlmalloc::GlobalDlmalloc;

#[cfg(all(target_arch = "wasm32", not(test)))]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { core::arch::wasm32::unreachable() }

#[cfg(target_arch = "wasm32")]
pub mod wasm;
