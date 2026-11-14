// `sf_engine` is set when this crate is the one being compiled: the demo's level files (game/src), which the
// engine's tests take in as fixtures, keep their own tests for this build.
fn main() { println!("cargo:rustc-cfg=sf_engine"); println!("cargo:rerun-if-changed=build.rs"); }
