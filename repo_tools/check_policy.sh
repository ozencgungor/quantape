#!/bin/bash
# Repo policy checks shared by CI (policy.yml) and pre-commit.
#
# Enforced:
#   1. no -ffast-math compile flags (comments mentioning it are fine)
#   2. no AGPL code and no XAD includes (AGPL design reference only)
#   3. self-containment: quantape/util and quantape/ad include only themselves
#      (plus the standard library / Eigen in dedicated adapter headers)
#   4. CamelCase file names under include/ and src/
#   5. include-guard convention in the new low-level modules
#
# Usage: repo_tools/check_policy.sh
set -uo pipefail

root=$(git rev-parse --show-toplevel 2>/dev/null) || {
    echo "error: not inside a git repository" >&2
    exit 2
}
cd "$root" || exit 2

failures=0
report() {
    printf 'policy: %s\n' "$*" >&2
    failures=1
}

# ── 1. -ffast-math flags (not prose) ────────────────────────────────────────
while IFS= read -r line; do
    report "fast-math flag: $line"
done < <(git grep -n -e '-ffast-math' -- 'CMakeLists.txt' '*.cmake' 2>/dev/null |
    grep -v -E ':[0-9]+:[[:space:]]*#' || true)

# ── 2. AGPL / XAD ───────────────────────────────────────────────────────────
while IFS= read -r line; do
    report "AGPL mention: $line"
done < <(git grep -n -i 'agpl' 2>/dev/null || true)
while IFS= read -r line; do
    report "XAD include: $line"
done < <(git grep -n -E '#[[:space:]]*include[[:space:]]*[<"]xad/' 2>/dev/null || true)

# ── 3. self-containment of util/ (and the future ad/) ───────────────────────
while IFS= read -r file; do
    module=${file#include/quantape/}
    module=${module%%/*}
    while IFS= read -r inc; do
        target=${inc#*\#include \"quantape/}
        target=${target%%\"*}
        case "$target" in
            "$module"/*|"$module") ;;
            *) report "self-containment: $file includes quantape/$target" ;;
        esac
    done < <(grep -n '#include "quantape/' "$file" || true)
done < <(git ls-files 'include/quantape/util/**' 'include/quantape/ad/**')

# ── 4. CamelCase file names under include/ and src/ ─────────────────────────
while IFS= read -r file; do
    base=${file##*/}
    case "$base" in
        [A-Z]*) ;;
        *) report "file name not CamelCase: $file" ;;
    esac
    case "$base" in
        *[!A-Za-z0-9.]*) report "file name not CamelCase: $file" ;;
    esac
done < <(git ls-files 'include/**' 'src/**' | grep -E '\.(h|hpp|cpp)$' || true)

# ── 5. include guards in the new modules ────────────────────────────────────
while IFS= read -r file; do
    if grep -q '#pragma once' "$file"; then
        continue
    fi
    guard=$(grep -m1 -E '^#ifndef[[:space:]]+[A-Za-z_][A-Za-z0-9_]*' "$file" | awk '{print $2}')
    if [ -z "$guard" ]; then
        report "missing include guard: $file"
    elif [[ "$guard" != QUANTAPE_* ]]; then
        report "guard must start with QUANTAPE_: $file ($guard)"
    fi
done < <(git ls-files 'include/quantape/util/**' 'include/quantape/ad/**')

if [ "$failures" -eq 0 ]; then
    echo "policy: ok"
fi
exit "$failures"
