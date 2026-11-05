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
# lightape (the standalone header-only AD engine) is checked in full through
# its own compile database: every test TU plus the benchmarks, under its own
# lightape/.clang-tidy. Its headers are pulled in through the all-headers TU
# (lightape/tests/test_all_headers.cpp), so a changed lightape/include header
# is checked through that driver. Configure the database with:
#   cmake -S lightape -B build-lightape -DCMAKE_BUILD_TYPE=Release \
#     -DLIGHTAPE_BUILD_TESTS=ON -DLIGHTAPE_BUILD_BENCHMARKS=ON \
#     -DLIGHTAPE_VECTOR_MATH=libm -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
# Eigen-conditional test TUs are skipped automatically when the lightape
# compile database does not contain them.
#
# Usage:
#   repo_tools/check_tidy.sh                 # full allowlisted set
#   repo_tools/check_tidy.sh --changed REF   # only files changed vs REF's tip
#
# Environment: BUILD_DIR (default build-ci), LIGHTAPE_BUILD_DIR (default
# build-lightape), CLANG_TIDY (default clang-tidy). On macOS with an upstream
# (pip/brew) clang-tidy, export
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
LIGHTAPE_BUILD_DIR=${LIGHTAPE_BUILD_DIR:-build-lightape}
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

# lightape: own database, own config (lightape/.clang-tidy), own header
# filter. The driver TU is the include-all TU, so a changed header is checked
# through it.
LIGHTAPE_COMPILE_DB="$LIGHTAPE_BUILD_DIR/compile_commands.json"
LIGHTAPE_HEADER_FILTER='lightape/.*'
LIGHTAPE_DRIVER_TU=lightape/tests/test_all_headers.cpp
LIGHTAPE_ALLOWLIST_REGEX='^lightape/(include/.*\.h|tests/.*\.cpp|benchmarks/.*\.cpp)$'
LIGHTAPE_TARGETS=(
    "$LIGHTAPE_DRIVER_TU"
    lightape/tests/test_lean_reverse.cpp
    lightape/tests/test_forward_scalar.cpp
    lightape/tests/test_forward_ops.cpp
    lightape/tests/test_forward_completion.cpp
    lightape/tests/test_reverse_ops.cpp
    lightape/tests/test_external_ops.cpp
    lightape/tests/test_softmax.cpp
    lightape/tests/test_hessian.cpp
    lightape/tests/test_second_order.cpp
    lightape/tests/test_callbacks.cpp
    lightape/tests/test_tape_lifecycle.cpp
    lightape/tests/test_p1_functions.cpp
    lightape/tests/test_bessel_ad.cpp
    lightape/tests/test_noncentral_chi_square_ad.cpp
    lightape/tests/test_bvn_ad.cpp
    lightape/tests/test_elementary_extras.cpp
    lightape/tests/test_record_opt.cpp
    lightape/tests/test_mode_mixing.cpp
    lightape/tests/test_polygamma_ad.cpp
    lightape/tests/test_batch_tape.cpp
    lightape/tests/test_batch_functions.cpp
    lightape/tests/test_vector_math.cpp
    lightape/tests/test_sparsity.cpp
    lightape/tests/test_p2_functions.cpp
    lightape/tests/test_third_order.cpp
    lightape/tests/test_determinism.cpp
    lightape/tests/test_inv_inc_beta.cpp
    lightape/tests/test_bessel_jy_ad.cpp
    lightape/tests/test_revolve.cpp
    lightape/tests/test_checkpoint.cpp
    lightape/tests/test_generic_ops.cpp
    lightape/tests/test_probability.cpp
    lightape/tests/test_digamma.cpp
    lightape/tests/test_exec.cpp
    lightape/tests/test_math_gaps.cpp
    lightape/tests/test_revtapet.cpp
    lightape/tests/test_revscalarn.cpp
    # Eigen adapters: present only when the lightape configure found Eigen;
    # the presence filter below drops them otherwise.
    lightape/tests/test_eigen_scalars.cpp
    lightape/tests/test_eigen_numext.cpp
    lightape/tests/test_matrix_blocks.cpp
    lightape/tests/test_eigen_select.cpp
    lightape/benchmarks/bench_lean_reverse.cpp
    lightape/benchmarks/bench_batch_tape.cpp
    lightape/benchmarks/bench_vector_math.cpp
)

targets=()
lightape_targets=()
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
        if printf '%s\n' "$f" | grep -Eq "$LIGHTAPE_ALLOWLIST_REGEX"; then
            case "$f" in
                *.cpp) lightape_targets+=("$f") ;;
                *.h|*.hpp) lightape_targets+=("$LIGHTAPE_DRIVER_TU") ;;
            esac
        fi
    done
    # de-duplicate (portable: mapfile is bash 4+, macOS ships 3.2)
    if [ "${#targets[@]}" -gt 0 ]; then
        targets=($(printf '%s\n' "${targets[@]}" | sort -u))
    fi
    if [ "${#lightape_targets[@]}" -gt 0 ]; then
        lightape_targets=($(printf '%s\n' "${lightape_targets[@]}" | sort -u))
    fi
    if [ "${#targets[@]}" -eq 0 ] && [ "${#lightape_targets[@]}" -eq 0 ]; then
        echo "clang-tidy: no allowlisted changes; nothing to check"
        exit 0
    fi
else
    targets=("${DEFAULT_TARGETS[@]}")
    lightape_targets=("${LIGHTAPE_TARGETS[@]}")
fi

rc=0

if [ "${#targets[@]}" -gt 0 ]; then
    echo "clang-tidy: checking ${targets[*]}"
    "$CLANG_TIDY" -p "$BUILD_DIR" --warnings-as-errors='*' "${targets[@]}" || rc=$?
fi

if [ "${#lightape_targets[@]}" -gt 0 ]; then
    if [ ! -f "$LIGHTAPE_COMPILE_DB" ]; then
        echo "error: $LIGHTAPE_COMPILE_DB not found (configure lightape with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON)" >&2
        exit 2
    fi
    # Drop targets the database does not contain (Eigen-conditional TUs).
    present=()
    for t in "${lightape_targets[@]}"; do
        if grep -qF -- "$t" "$LIGHTAPE_COMPILE_DB"; then
            present+=("$t")
        fi
    done
    if [ "${#present[@]}" -gt 0 ]; then
        echo "clang-tidy: checking lightape (${present[*]})"
        "$CLANG_TIDY" -p "$LIGHTAPE_BUILD_DIR" --warnings-as-errors='*' \
            --header-filter="$LIGHTAPE_HEADER_FILTER" "${present[@]}" || rc=$?
    fi
fi

exit "$rc"
