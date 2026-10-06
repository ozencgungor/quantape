#!/bin/bash
# Developer bootstrap: tooling setup + debug/release build trees.
# Usage: repo_tools/setup_dev.sh
#
# Safe to re-run; existing build directories are left untouched.
set -u

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(git -C "$script_dir" rev-parse --show-toplevel 2>/dev/null) || {
    echo "error: not inside a git repository" >&2
    exit 2
}
cd "$root" || exit 2

status=0
required_clang_format="21.1.6"

# ── clang-format (must match the CI pin in .github/workflows/format.yml) ────
if command -v clang-format >/dev/null 2>&1; then
    have=$(clang-format --version | awk '{print $3}')
    case "$have" in
        "$required_clang_format"|21.*) echo "clang-format: $have (ok)" ;;
        *)
            echo "warning: clang-format $have found; CI pins $required_clang_format" >&2
            status=1
            ;;
    esac
else
    echo "error: clang-format not found" >&2
    echo "  macOS: brew install clang-format" >&2
    echo "  Linux: python3 -m pip install --user clang-format==$required_clang_format" >&2
    status=1
fi

# ── pre-commit hooks ────────────────────────────────────────────────────────
if ! command -v pre-commit >/dev/null 2>&1; then
    echo "pre-commit not found; installing with pip --user"
    python3 -m pip install --user pre-commit || status=1
fi
if command -v pre-commit >/dev/null 2>&1; then
    pre-commit install >/dev/null && echo "pre-commit hooks installed"
fi

# ── pinned third-party sources (Eigen, Stan Math) ───────────────────────────
"$script_dir/fetch_third_party.sh"

# ── build trees: debug (no benches), release (benches on) ───────────────────
configure() {
    local dir=$1 type=$2 benches=$3
    if [ -f "$dir/CMakeCache.txt" ]; then
        echo "build tree $dir already configured"
        return
    fi
    cmake -S . -B "$dir" -DCMAKE_BUILD_TYPE="$type" -DQUANTAPE_BUILD_BENCHMARKS="$benches" || status=1
}
configure build/debug Debug OFF
configure build/release Release ON

echo
echo "next steps:"
echo "  cmake --build build/debug -j"
echo "  ctest --test-dir build/debug -j"
echo "  # or presets: cmake --preset debug && cmake --build --preset debug"
echo "  repo_tools/format_code.sh --check"
echo "  repo_tools/check_policy.sh"
exit "$status"
