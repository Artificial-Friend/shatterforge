# WebGPU rigid-body solver (phase 2, first cut)

`gpu/gpu_world.js` runs the same model as `engine/src/phys.rs` entirely on the GPU. The model covers:
- box bodies
- speculative contacts
- the soft step with 12 sub-steps at contact stiffness of one sixth of the sub-step rate
- anchored patch friction with twist

Nothing is read back during a step.

| Stage | Kernel | Work |
| --- | --- | --- |
| Broad phase | `prep`, `scan`, `scatter` | Bodies hashed into a uniform grid (cell ≥ largest AABB) and counting-sorted by bucket |
| Narrow phase | `collide` | One thread per body: ground plus the 27 neighbour cells. Box-box SAT, face clipping and 4-point reduction. Warm-start match against last step's manifolds through a GPU hash table, then constraint preparation |
| Colouring | `color`, `color_scan`, `color_scatter` | Greedy colouring with atomic per-body bit masks. Manifolds in one colour share no dynamic body, so a colour is solved in parallel without races. Rare overflow manifolds are solved by one thread |
| Sub-steps | `integrate_vel`, `warm`, `solve`, `integrate_pos`, `relax` | Gauss-Seidel across colours: one indirect dispatch per colour |

Contacts are 240 B hot (the moment arms and inverse-inertia products are recomputed in the shader, not stored) plus 96 B cold for warm-start matching. The solver reads a 96 B record per body (velocity, displacement, inverse inertia), separate from the 96 B static record (pose, size, AABB). Contacts and the pair table are double-buffered between steps.

**Sleeping.** Islands are found with a GPU union-find (atomic compare-and-swap hooking) over contacts between awake bodies. An island whose slowest-to-settle body has been still for 0.5 s sleeps. A sleeping body is static to the solver, costs no contacts with other sleepers, and wakes when a moving body touches it.

## Checks against the CPU reference (headless Chrome, SwiftShader WebGPU)

`node gpu/test_gpu.cjs` reports:

| Check | CPU | GPU |
| --- | --- | --- |
| Stack of 20, 10 s | 6.93 mm drift, 1.7 mm/s | 6.93 mm drift, 1.0 mm/s (4 colours) |
| Offset column of 10 | topples, all at 0.25 m | topples, all at 0.25 m |
| Pile of 2,000 after 2 s | y 0.117–4.828 m, 3,580 contacts | y 0.118–4.826 m, 3,759 contacts, 11 colours, 0 overflow |

SwiftShader runs WebGPU on the CPU, so its timings mean nothing. Throughput is measured by the bench page on real GPUs.

Sleep check (`node gpu/test_gpu.cjs sleep`): a 300-brick pile settles and sleeps. A box dropped from 3 m above it wakes about 210 bricks. Everything is asleep again within 5 s.

## Not yet on the GPU

- Bodies larger than the grid cell: a multi-level grid is needed for large debris chunks.
