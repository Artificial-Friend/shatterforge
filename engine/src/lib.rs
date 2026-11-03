//! Shatterforge engine (phase 0 prototypes): adaptive brick store and box rigid-body solver.
#![cfg_attr(target_arch = "wasm32", no_std)]
extern crate alloc;

pub mod math;
pub mod brick;
pub mod phys;
pub mod tower;
pub mod mesh;
pub mod player;
pub mod game;
pub mod structure;
mod validation;

#[cfg(target_arch = "wasm32")]
#[global_allocator]
static ALLOC: dlmalloc::GlobalDlmalloc = dlmalloc::GlobalDlmalloc;

#[cfg(all(target_arch = "wasm32", not(test)))]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { core::arch::wasm32::unreachable() }

#[cfg(target_arch = "wasm32")]
pub mod wasm;
