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
#   6. lightape engine self-containment: lightape/include and lightape/src
#      include only lightape headers, the standard library and Eigen (Eigen
#      confined to adapter headers); never quantape, Stan, Boost, TBB or other
#      project code. Benchmarks (and the test harness) may include quantape
#      header-only for cross-project comparisons.
#   7. no redundant `inline` on lightape function templates: templates are
#      ODR-exempt and clang-tidy has no check for this
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
# Scan code files only: the policy script, NOTICE and pre-commit config
# legitimately mention AGPL when stating that it must not be used.
while IFS= read -r line; do
    report "AGPL mention: $line"
done < <(git grep -n -i 'agpl' -- '*.h' '*.hpp' '*.cpp' '*.cc' 'CMakeLists.txt' '*.cmake' 2>/dev/null || true)
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

# ── 6. lightape self-containment ────────────────────────────────────────────
while IFS= read -r line; do
    report "lightape self-containment: $line"
done < <(git grep -n -E \
    '#[[:space:]]*include[[:space:]]*[<"](quantape|stan|boost|tbb|sundials|quill|zmij)(/|")' \
    -- 'lightape/include/**' 'lightape/src/**' 2>/dev/null || true)

# ── 7. redundant `inline` on lightape function templates ────────────────────
# Function templates are exempt from the one-definition rule and do not need
# `inline`; clang-tidy has no check for it (readability-redundant-inline-
# specifier only covers in-class definitions), so it is guarded here. A
# template head may carry a `requires(...)` clause on the following line.
# `inline constexpr`/`inline static` variables stay (load-bearing). Files are
# taken from the index when tracked, else from the working tree so the check
# also covers lightape while it is still untracked.
_lightape_headers=$(git ls-files 'lightape/include/**' 2>/dev/null)
if [ -z "$_lightape_headers" ] && [ -d lightape/include ]; then
    _lightape_headers=$(find lightape/include -type f -name '*.h' 2>/dev/null)
fi
while IFS= read -r file; do
    [ -n "$file" ] || continue
    while IFS= read -r line; do
        report "redundant inline on function template: $line"
    done < <(awk '
        FNR == 1 { prev = "" }
        prev ~ /^[[:space:]]*template[[:space:]]*</ &&
        $0 ~ /^[[:space:]]*inline[[:space:]]/ &&
        $0 !~ /inline[[:space:]]+(constexpr|static)/ {
            printf "%s:%d:%s\n", FILENAME, FNR, $0
        }
        prev ~ /^[[:space:]]*requires[[:space:]]*\(/ &&
        $0 ~ /^[[:space:]]*inline[[:space:]]/ {
            printf "%s:%d:%s\n", FILENAME, FNR, $0
        }
        { prev = $0 }
    ' "$file")
done <<< "$_lightape_headers"

if [ "$failures" -eq 0 ]; then
    echo "policy: ok"
fi
exit "$failures"
