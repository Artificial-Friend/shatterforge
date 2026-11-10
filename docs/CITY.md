# The city (kind 5)

A city of n × n lots on a 40 m pitch, a building on each, streets between (`engine/src/city.rs`). In the page: Settings → Building → city, and Settings → City size (2 × 2 to 8 × 8 lots; 3 × 3 by default). The engine takes the size as its storeys (`game_new_kind(n, 5)`).

## Layout

- Lot (i, j) is centred at ((i − (n−1)/2) × 40 m, (j − (n−1)/2) × 40 m). The first lot (south-west corner) holds the complex building, and at most one lot in five after it.
- The other lots draw a kind and a height from a fixed sequence (xorshift, fixed seed), so a city of a given size is the same every time:
  - concrete frame: 3 to 12 storeys
  - masonry: 2 to 5 storeys
  - mass timber: 3 to 6 storeys
  - mixed: 3 to 5 storeys
- Each building is turned by 0 to 3 quarter turns.
- The buildings are 22.5 m square; the complex one reaches 14.75 m from its centre with its wing. Streets are 14 m wide or more.
- Each building is built where it was designed (`tower::build_kind`, `complex::build`), then set on its lot with `Bricks::place`: boxes turned and moved, free pieces with their shapes and rotations, flags and damage kept, member ids (`structure`) raised so that no two buildings share one. Joints are then built for the whole city. Nothing joins two lots; they share only the ground.

Test `city::tests::a_turned_copy_keeps_its_joints`: the complex building in each of the four turns keeps all its 798 bricks and 1,250 joints, with only its 171 pieces of furniture loose. (The free pieces' rotation was first turned the other way from their positions: a quarter turn kept 669 of the 1,250 joints.)

## Structure checks by lot

- Each lot is analysed on its own (`Game::structure_step`). Analysed all at once, the solver's accuracy is relative to the whole city: in a 5 × 5 city, the arches of two complex buildings came out as mechanisms (2.25 and 1.19 of strength), where alone they stand at 0.66.
- A lot is analysed again only when it has changed (a brick removed, split or detached there), rubble lies on it, or it has joints wearing. A lot left alone gives the same answer every time.
- Each lot keeps its last worst ratio; the city's worst is the largest.

Test `city::tests::the_city_is_what_it_should_be_and_stands` (native, release, build machine):

| City | Bricks | Free pieces | Joints | Build | First structure step | Worst ratio | Later steps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 × 1 | 798 | 395 | 1,250 | 13 ms | 30 ms | 0.662 | 0 of 1 lots, 0-8 ms |
| 3 × 3 | 7,539 | 395 | 21,486 | 203 ms | 680 ms | 0.829 | 0 of 9 lots, 0-21 ms |
| 5 × 5 | 20,460 | 1,975 | 55,778 | 503 ms | 1,478 ms | 0.917 | 0 of 25 lots, 0-115 ms |

- In the 3 × 3 city, every kind appears and nothing fails under its own weight.
- An 8 × 8 city (`bigger_cities_cost`, ignored by default; same machine) has 57,924 bricks and 5.3 MB of bricks and joints. It builds in 1.8 s with meshes, and its first structure step takes 5.0 s (20.8 s with all lots analysed at once).

## The dynamic structure by lot (branch `test`, docs/DYNAMIC.md)

- Each lot keeps its own displacements, velocities and beds (`Game::dynst`), and is stepped on its own. A lot still moving stays active; the others cost nothing.
- Test `dyn_tests::a_city_is_stepped_lot_by_lot` (native, release, build machine): the 3 × 3 city's first step solves its 9 lots at rest in 704–990 ms; at rest a step takes 0.2 ms. A column of the complex building shot out moves lot 0 only: at rest again after 87 steps (1.5 s), the slowest 28 ms.
- The engineer's view (key V) colours every brick of the city by its load and draws the lots where their solutions put them: 7,539 bricks and 32,819 cells, 475 kB when it is switched on (every lot analysed once), then only the lots that change.

## Meshes by lot

The static world is meshed in chunks a storey high, and in the city also by lot: chunk k is lot k / slices, slice k % slices (`game::Tiles`). A brick is in the chunk of its minimum corner. A change remeshes its own lot's slices, not the whole city's. The 3 × 3 city has 90 chunks.

## In the page

- Each lot is paved; the streets are bare ground.
- The sun's shadow follows the player, because the city is wider than the shadow map.
- The player starts in the street east of the first lot.
- The benchmark's cut blows out every lot's ground storey (36 blasts a lot).
- The lab's `cut` takes a list of lot indices: scenario `city` cuts lots 0 and 1.

## Lab

Scenarios `citystand` (3 × 3, left alone) and `city` (3 × 3, the complex building and the 7-storey frame beside it cut); suite `city`. Results are in docs/LAB.md.

## Limits

- Lots are 40 m apart: a building larger than 26 m across would reach the street's far side.
- The city's sizes run up to 16 × 16 in the engine (`layout` clamps); the page offers up to 8 × 8.
- A lot is meshed and analysed with what stands on it. Debris that falls on a neighbouring lot is a body until it rests; if it rests there, it loads that lot (rubble on floors), not its own.
