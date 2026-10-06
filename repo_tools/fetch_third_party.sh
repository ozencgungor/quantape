#!/bin/bash
# Fetch the pinned third-party libraries that CMake expects under third-party/
# but does not download itself (Eigen and Stan Math). Idempotent; verifies
# SHA256 against the pins below. Boost/TBB/SUNDIALS/quill/zmij are handled by
# CMake's FetchContent at configure time.
#
# Usage: repo_tools/fetch_third_party.sh
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(git -C "$script_dir" rev-parse --show-toplevel 2>/dev/null) || {
    echo "error: not inside a git repository" >&2
    exit 2
}
cd "$root"

EIGEN_VERSION="3.4.0"
EIGEN_URL="https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VERSION}/eigen-${EIGEN_VERSION}.tar.gz"
EIGEN_SHA256="8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72"

STAN_MATH_VERSION="4.9.0"
STAN_MATH_URL="https://github.com/stan-dev/math/archive/refs/tags/v${STAN_MATH_VERSION}.tar.gz"
STAN_MATH_SHA256="876881b71dee6fec32f2b4aa52692994cccd2a19c6de1b986d7fc457155f219c"

sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

fetch_verified() { # <url> <sha256> <destination-tarball>
    local url=$1 expected=$2 archive=$3
    echo "fetching $url"
    curl -fsSL "$url" -o "$archive"
    local actual
    actual=$(sha256 "$archive")
    if [ "$actual" != "$expected" ]; then
        echo "error: SHA256 mismatch for $url" >&2
        echo "  expected: $expected" >&2
        echo "  actual:   $actual" >&2
        rm -f "$archive"
        exit 1
    fi
}

mkdir -p third-party
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

if [ ! -f third-party/eigen/Eigen/Core ]; then
    fetch_verified "$EIGEN_URL" "$EIGEN_SHA256" "$tmp/eigen.tar.gz"
    mkdir -p third-party/eigen
    tar -xzf "$tmp/eigen.tar.gz" -C third-party/eigen --strip-components=1
else
    echo "Eigen ${EIGEN_VERSION}: present"
fi

if [ -f "third-party/stan-math/math-${STAN_MATH_VERSION}/stan/math.hpp" ] ||
   [ -f third-party/stan-math/stan/math.hpp ]; then
    echo "Stan Math ${STAN_MATH_VERSION}: present"
else
    fetch_verified "$STAN_MATH_URL" "$STAN_MATH_SHA256" "$tmp/stan-math.tar.gz"
    mkdir -p third-party/stan-math
    tar -xzf "$tmp/stan-math.tar.gz" -C third-party/stan-math
fi

echo "third-party ready"
