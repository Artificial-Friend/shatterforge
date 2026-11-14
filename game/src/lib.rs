//! Shatterforge demo game: what is built (towers of five kinds, the complex building, the city), the player, and
//! the functions the page calls (wasm.rs). The engine is `sf-engine` (engine/); its modules are named here as the
//! level files name them, which are also the engine's test fixtures (engine/src/lib.rs).
#![cfg_attr(target_arch = "wasm32", no_std)]
extern crate alloc;

pub use sf_engine::{brick, hull, math, mesh, phys, sim, structure, Map, Set};
pub mod tower;
pub mod complex;
pub mod city;
pub mod player;

#[cfg(target_arch = "wasm32")]
#[global_allocator]
static ALLOC: dlmalloc::GlobalDlmalloc = dlmalloc::GlobalDlmalloc;

#[cfg(all(target_arch = "wasm32", not(test)))]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { core::arch::wasm32::unreachable() }

#[cfg(target_arch = "wasm32")]
pub mod wasm;

#[cfg(test)]
mod tests {
    #[test] fn every_level_builds_and_stands() {
        // the demo's own check: each kind of level makes a simulation whose first structure step fails nothing
        for (kind, storeys) in [(0u8, 4), (1, 3), (2, 3), (3, 3), (4, 3), (5, 2)] {
            let mut g = crate::tower::new_kind(storeys, kind);
            let alive = g.b.alive_count();
            g.structure_step(usize::MAX);
            assert!(alive > 100 && g.failed_total == 0, "kind {kind}: {alive} bricks, {} joints failed standing", g.failed_total);
        }
    }
}
