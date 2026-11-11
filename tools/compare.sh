#!/usr/bin/env bash
# The dynamic structure against the solve at rest on one lab scenario: see tools/compare.cjs.
cd "$(dirname "$0")/.." && exec node tools/compare.cjs "$@"
