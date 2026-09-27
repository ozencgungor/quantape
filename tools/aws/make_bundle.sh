#!/usr/bin/env sh
# Bundle the standalone Sobol tools (standard library only) for a remote
# build on a bare Linux machine:
#
#   ./tools/aws/make_bundle.sh [output.tar.gz]
#   scp sobol_extender.tar.gz user@host:~/
#   ssh user@host 'tar xzf sobol_extender.tar.gz && ./build.sh'
#
# Builds extend_sobol (table extension) and refine_sobol (iterative refinement).
# No Stan, Eigen, TBB or the quantape library are required; g++ >= 10 suffices.
set -e

output="${1:-sobol_extender.tar.gz}"
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT

mkdir -p "$stage/tools" "$stage/include/quantape/math/Random/Sobol"
cp "$repo_root/tools/extend_sobol.cpp" "$stage/tools/"
cp "$repo_root/tools/refine_sobol.cpp" "$stage/tools/"
cp "$repo_root"/include/quantape/math/Random/Sobol/*.h "$stage/include/quantape/math/Random/Sobol/"

cat > "$stage/build.sh" <<'EOF'
#!/usr/bin/env sh
set -e
if ! command -v g++ >/dev/null 2>&1; then
    echo "g++ not found" >&2
    exit 1
fi
g++ -O3 -std=c++20 -pthread -I include tools/extend_sobol.cpp -o extend_sobol
echo "built ./extend_sobol"
g++ -O3 -std=c++20 -pthread -I include tools/refine_sobol.cpp -o refine_sobol
echo "built ./refine_sobol"
EOF
chmod +x "$stage/build.sh"

tar -C "$stage" -czf "$output" .
echo "wrote $output ($(du -h "$output" | cut -f1))"
echo "contents:"
tar -tzf "$output"
