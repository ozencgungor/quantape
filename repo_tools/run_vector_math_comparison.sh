#!/bin/bash
# x86-64 vector-math comparison driver for lightape (libm vs SLEEF explicit ISAs).
#
# Run on a real x86-64 Linux machine (cloud VM or bare metal); emulated CPUs
# give meaningless timings. Results land in <root>/comparison-results/.
#
# Usage: repo_tools/run_vector_math_comparison.sh [--tests] [build-root]
#   --tests       also build/run the full ctest suites of both build flavors,
#                 the scalar AD bench, and cross-check the libm-reference
#                 fingerprints between the two builds
#   build-root    build dir root (default: build/comparison)
#
# The per-ISA SLEEF kernel TUs are compiled by lightape/CMakeLists.txt with
# their own -mavx2/-mavx512f flags; there is deliberately no -mavx* in
# CMAKE_C_FLAGS/CMAKE_CXX_FLAGS here. The ISA mix is selected at run time via
# LIGHTAPE_VECTOR_ISA (auto|avx512f|avx2|avx2128|generic), and VectorMath<W>
# picks the widest kernel that fits W, so one build exercises the whole ladder.
#
# See internal_docs/simd_vector_math_plan.md and decisions.md #28.
set -uo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(git -C "$script_dir" rev-parse --show-toplevel 2>/dev/null || dirname -- "$script_dir")
cd "$root" || exit 2

run_tests=0
jobs=${JOBS:-2}
build_root="build/comparison"
for arg in "$@"; do
    case "$arg" in
        --tests) run_tests=1 ;;
        *) build_root=$arg ;;
    esac
done

out="$root/comparison-results"
mkdir -p "$out"

{
    echo "## CPU"
    grep -m1 'model name' /proc/cpuinfo || true
    echo "flags (simd subset):"
    grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | grep -E '^(avx|avx2|avx512|fma)' | sort -u || true
    echo
    cmake --version | head -1
} | tee "$out/environment.txt"

cmake -S lightape -B "${build_root}/libm" -DCMAKE_BUILD_TYPE=Release \
    -DLIGHTAPE_BUILD_BENCHMARKS=ON -DLIGHTAPE_BUILD_TESTS="$run_tests" \
    -DLIGHTAPE_VECTOR_MATH=libm || exit 2
cmake -S lightape -B "${build_root}/sleef" -DCMAKE_BUILD_TYPE=Release \
    -DLIGHTAPE_BUILD_BENCHMARKS=ON -DLIGHTAPE_BUILD_TESTS="$run_tests" \
    -DLIGHTAPE_VECTOR_MATH=sleef || exit 2

cmake --build "${build_root}/libm" --target lightape_vector_math_bench lightape_batch_bench lightape_bench -j"$jobs" || exit 2
cmake --build "${build_root}/sleef" --target lightape_vector_math_bench lightape_batch_bench lightape_bench -j"$jobs" || exit 2
if [ "$run_tests" = 1 ]; then
    cmake --build "${build_root}/libm" --target lightape_tests -j"$jobs" || exit 2
    cmake --build "${build_root}/sleef" --target lightape_tests lightape_tests_sleef -j"$jobs" || exit 2
    ctest --test-dir "${build_root}/libm" -j4 --output-on-failure | tee "$out/ctest-libm.txt"
    ctest --test-dir "${build_root}/sleef" -j4 --output-on-failure | tee "$out/ctest-sleef.txt"

    # TU-split invariant: the libm reference TUs carry no ISA flags in either
    # build, so the SLEEF build's libm-table fingerprint must equal the
    # portable libm build's (see tests/test_vector_math.cpp).
    "./${build_root}/libm/lightape_tests" --gtest_filter='VectorMathTest.FingerprintPerResolvedTable' \
        | tee "$out/fingerprint-libm.txt"
    "./${build_root}/sleef/lightape_tests_sleef" --gtest_filter='VectorMathTest.FingerprintPerResolvedTable' \
        | tee "$out/fingerprint-sleef.txt"
    fp_libm=$(grep -m1 'table=libm' "$out/fingerprint-libm.txt" | grep -o '0x[0-9a-f]*' | head -1)
    fp_sleef=$(grep -m1 'table=libm' "$out/fingerprint-sleef.txt" | grep -o '0x[0-9a-f]*' | head -1)
    echo "libm fingerprint (portable libm build): ${fp_libm:-missing}"
    echo "libm fingerprint (SLEEF build):         ${fp_sleef:-missing}"
    if [ -z "$fp_libm" ] || [ "$fp_libm" != "$fp_sleef" ]; then
        echo "FAIL: the SLEEF build's libm reference is not the portable libm reference" >&2
        exit 1
    fi
fi

{
    echo "### scalar AD bench (libm build)"
    "./${build_root}/libm/lightape_bench"
    echo "### scalar AD bench (SLEEF build)"
    "./${build_root}/sleef/lightape_bench"
} | tee "$out/scalar_ad.txt"

for i in auto avx512f avx2 avx2128 generic; do
    for table in auto sleef libm; do
        echo "### ISA=${i} TABLE=${table}" | tee -a "$out/vector_math.txt"
        LIGHTAPE_VECTOR_ISA="${i}" LIGHTAPE_MATH_TABLE="${table}" \
            "./${build_root}/sleef/lightape_vector_math_bench" | tee -a "$out/vector_math.txt"
    done
done

{
    echo "### batch bench (libm build)"
    "./${build_root}/libm/lightape_batch_bench"
    echo "### batch bench (SLEEF build, auto table)"
    "./${build_root}/sleef/lightape_batch_bench"
    echo "### batch bench (SLEEF build, forced libm)"
    LIGHTAPE_MATH_TABLE=libm "./${build_root}/sleef/lightape_batch_bench"
} | tee "$out/batch_bench.txt"

echo "comparison results: $out"
