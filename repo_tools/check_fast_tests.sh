#!/bin/bash
# Pre-push fast tier: run `ctest -L fast` when a local debug build tree
# exists. Skips silently otherwise (fresh clone, no build) so the hook never
# blocks a push just because the tree is missing.
set -u

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(git -C "$script_dir" rev-parse --show-toplevel 2>/dev/null) || exit 0
tree="$root/build/debug"
if [ ! -d "$tree" ]; then
    exit 0
fi

ctest --test-dir "$tree" -L fast -j 4 --output-on-failure
