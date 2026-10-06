#!/bin/bash
# clang-tidy gate for allowlisted modules (shared by CI and local use).
#
# The allowlist below grows one module at a time as code is cleaned; files
# outside it are skipped, so the job stays green while coverage expands.
# Header-only modules are exercised through a driver translation unit.
#
# Usage:
#   repo_tools/check_tidy.sh                 # default target (allowlist driver)
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

# Allowlisted path prefixes and their driver TUs for header-only modules.
ALLOWLIST_REGEX='^include/quantape/util/|^tests/test_util\.cpp$'
DEFAULT_TARGETS=(tests/test_util.cpp)

targets=()
if [ "${1:-}" = "--changed" ]; then
    base=${2:?usage: check_tidy.sh --changed <ref>}
    changed=$(git diff --name-only --diff-filter=ACMR "$base"...HEAD -- '*.h' '*.hpp' '*.cpp' || true)
    for f in $changed; do
        if printf '%s\n' "$f" | grep -Eq "$ALLOWLIST_REGEX"; then
            case "$f" in
                *.cpp) targets+=("$f") ;;
                *.h|*.hpp) targets+=("tests/test_util.cpp") ;; # header: check via driver
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
