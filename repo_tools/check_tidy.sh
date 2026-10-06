#!/bin/bash
# clang-tidy gate for allowlisted modules (shared by CI and local use).
#
# The allowlist below grows one module at a time as code is cleaned; files
# outside it are skipped, so the job stays green while coverage expands.
# Header-only modules are exercised through a driver translation unit: the
# plain `qta_test_util` TU pulls `tests/support/*.h` and the util headers into
# one compile, so a changed support/util header is checked through it.
#
# Allowlisted (all verified clean under clang-tidy 22.1.8 and the repo
# .clang-tidy config): include/quantape/util/**, tests/support/*.h (top
# level), tests/unit/util/*, tests/unit/infra/*, and the individual
# datetime/math TUs whose include closure is clean. The remaining converted
# plain TUs are deliberately NOT allowlisted yet: their include closures trip
# pre-existing library findings that the migration did not touch (e.g.
# datetime/Calendar.h and math/Random/PCGRandom.hpp branch-clones/typedefs,
# Interpolations CRTP ctors, Optimization/LevenbergMarquardt.h std::move,
# Optimization/SLSQP.h use-after-move). Promote them here as the owning
# modules are cleaned; Stan/TBB/AD profiles stay out entirely (clang-tidy
# time/memory explodes) and so does tests/support/fixtures/* (its only
# includers are Stan targets).
#
# Usage:
#   repo_tools/check_tidy.sh                 # full allowlisted set
#   repo_tools/check_tidy.sh --changed REF   # only files changed vs REF's tip
#
# Environment: BUILD_DIR (default build-ci), CLANG_TIDY (default clang-tidy).
# On macOS with an upstream (pip/brew) clang-tidy, export
#   SDKROOT=$(xcrun --show-sdk-path)
# so it finds the system libc++ headers.
set -uo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(git -C "$script_dir" rev-parse --show-toplevel 2>/dev/null) || {
    echo "error: not inside a git repository" >&2
    exit 2
}
cd "$root" || exit 2

BUILD_DIR=${BUILD_DIR:-build-ci}
CLANG_TIDY=${CLANG_TIDY:-clang-tidy}
COMPILE_DB="$BUILD_DIR/compile_commands.json"

if [ ! -f "$COMPILE_DB" ]; then
    echo "error: $COMPILE_DB not found (configure with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON)" >&2
    exit 2
fi

# Paths whose findings are enforced. Whole util/infra directories, the
# top-level support headers, and the individually verified clean TUs.
ALLOWLIST_REGEX='^include/quantape/util/|^tests/support/[^/]+\.h$|^tests/unit/(util|infra)/[^/]+\.cpp$|^tests/unit/datetime/test_date\.cpp$|^tests/unit/math/test_precision\.cpp$|^tests/unit/math/test_ziggurat\.cpp$'

# Header -> driver: headers are only checked through a TU that includes them.
DRIVER_TU=tests/unit/util/test_util.cpp

DEFAULT_TARGETS=(
    "$DRIVER_TU"
    tests/unit/util/test_number_format.cpp
    tests/unit/infra/test_logging.cpp
    tests/unit/datetime/test_date.cpp
    tests/unit/math/test_precision.cpp
    tests/unit/math/test_ziggurat.cpp
)

targets=()
if [ "${1:-}" = "--changed" ]; then
    base=${2:?usage: check_tidy.sh --changed <ref>}
    changed=$(git diff --name-only --diff-filter=ACMR "$base"...HEAD -- '*.h' '*.hpp' '*.cpp' || true)
    for f in $changed; do
        if printf '%s\n' "$f" | grep -Eq "$ALLOWLIST_REGEX"; then
            case "$f" in
                *.cpp) targets+=("$f") ;;
                *.h|*.hpp) targets+=("$DRIVER_TU") ;; # header: check via driver
            esac
        fi
    done
    # de-duplicate (portable: mapfile is bash 4+, macOS ships 3.2)
    if [ "${#targets[@]}" -gt 0 ]; then
        targets=($(printf '%s\n' "${targets[@]}" | sort -u))
    fi
    if [ "${#targets[@]}" -eq 0 ]; then
        echo "clang-tidy: no allowlisted changes; nothing to check"
        exit 0
    fi
else
    targets=("${DEFAULT_TARGETS[@]}")
fi

echo "clang-tidy: checking ${targets[*]}"
"$CLANG_TIDY" -p "$BUILD_DIR" --warnings-as-errors='*' "${targets[@]}"
