# Phase 0 results

The phase 0 prototypes live in `engine/`. They are a new crate, separate from the retired engine in `src/`. They cover the adaptive brick store (`brick.rs`) and the box rigid-body solver (`phys.rs`). Measurements were taken in the build sandbox, a 2-core 2.8 GHz Xeon VM, on one thread, both natively and in WebAssembly. The browser bench page (`dist/bench.html`, built by `bench/build_page.py`) runs the same gates on any machine.

## Gates

| Gate | Target | Native | WebAssembly | Verdict |
| --- | --- | --- | --- | --- |
| Split | p99 < 1 ms | median 0.13 ms, p99 0.23 ms | median 0.13–0.20 ms, p99 0.40–0.69 ms (Node, headless Chrome) | Pass |
| Brick memory | ≤ 96 B/brick | 79 B (running-bond wall: 232 bricks, 637 joints) | 76 B (refined panel) | Pass |
| Stack of 20 | < 10 mm drift, < 10 mm/s after 10 s | 6.9 mm, 1.7 mm/s (asleep after settling) | the same | Pass, at 12 sub-steps |
| Throughput | 20,000 awake bricks < 5 ms | 118 ms (4 sub-steps), 226 ms (12) | 120 ms, 256 ms | Miss: GPU work for phase 2 |

The split scenario refines a 7.5 × 3.75 × 0.25 m masonry panel (one brick) to 0.125 m within 0.5 m of a hit. That takes 160 splits and leaves 165 bricks. The joint graph after splitting is checked to equal a rebuild from geometry, and joint areas to the outside are conserved.

## What the prototypes taught us

**1. Soft contacts need stiffness for tall stacks.** With Box2D v3's defaults (4 sub-steps, 30 Hz contacts), a stack of 20 cubes sags 11 cm and then topples. This is real physics of the model, not a bug.
- A soft contact sags by load ÷ (effective mass × ω²).
- At 30 Hz the stack is an inverted pendulum on springs that are too soft to hold it.
- What works is 12 sub-steps with contact stiffness at one sixth of the sub-step rate (120 Hz). Near the stability limit of a quarter, the stack keeps swaying.
- Lighter partial-leak and no-leak variants were tried and were unstable.

**2. Friction must be anchored.** Per-point friction let stacks creep sideways. The solver now uses patch friction at the contact centroid: two tangents plus twist. It is anchored to a point fixed on each body while the patch sticks, which removed the creep.

**3. One CPU thread is 25–50× short of the throughput target.** Breakdown at 20,000 bricks with 4 sub-steps:
- **Collision: 52 ms** (37 ms of it in the narrow phase).
- **Solver: 64 ms.** About 690 instructions per contact manifold per pass, and manifolds are about 660 bytes.

  The 20,000 case scales worse than linearly because manifolds stop fitting in cache. Even a desktop CPU twice as fast stays far from 5 ms.

**4. Threads are not available on a hosted page.** WebAssembly threads need SharedArrayBuffer, which needs cross-origin isolation (COOP/COEP headers), and an artifact page cannot set those. WebGPU compute needs no special headers.

## Consequences for phase 2 (rigid-body engine)

1. **WebGPU solver as the main path.** Graph-coloured soft-step solver and narrow phase in compute shaders. The CPU path stays as the reference and the fallback.
2. **Adaptive sub-steps per island.** Low rubble islands, which make up most debris, step at 4 sub-steps. Tall stacks and standing debris step at 12. On the CPU this alone is about 2× for typical scenes.
3. **Compact contact data.** Hot and cold manifold data split, variable point counts, and a target of ≤ 200 B per manifold. SIMD batches of 4 for the CPU fallback (128-bit wasm SIMD).
4. **Cheaper collisions.** Persistent pairs, a cached separating axis per pair, and an axis-aligned fast path for bricks that have not rotated.
5. **Structure index.** The brick store's spatial hash is a prototype convenience. Phase 1 replaces it with a per-structure grid or BVH whose memory is counted in the budget.

## Reproduce

```bash
cd engine
cargo test --release -- --nocapture                    # unit tests and the split, memory and stack gates
cargo run --release --example pile -- 20000 90 4       # native throughput: bodies, frames, sub-steps
cargo run --release --example stack                    # stack trace (env N, SUB, HZ, ZETA, ANCH, MU, SLEEP)
RUSTC_BOOTSTRAP=1 cargo build --release --target wasm32-unknown-unknown -Zbuild-std=core,alloc,panic_abort
cd .. && node bench/node_bench.mjs                     # gates in WebAssembly
python3 bench/build_page.py                            # dist/bench.html
```
