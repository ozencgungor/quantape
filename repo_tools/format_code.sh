#!/bin/bash

# Script to format all C++ source files using clang-format (repo-wide).
# Usage: repo_tools/format_code.sh [--check]
#
# Works from any directory: the repository root is resolved from the script
# location (git top-level when available, otherwise the script's parent).
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if ! root=$(git -C "$script_dir" rev-parse --show-toplevel 2>/dev/null); then
    root=$(dirname -- "$script_dir")
fi
cd "$root" || exit 2

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if clang-format is installed
if ! command -v clang-format &> /dev/null; then
    echo -e "${RED}Error: clang-format is not installed${NC}"
    echo ""
    echo "To install clang-format on macOS:"
    echo "  brew install clang-format"
    echo ""
    echo "To install on Linux:"
    echo "  sudo apt-get install clang-format   # Debian/Ubuntu"
    echo "  sudo yum install clang-tools-extra  # RHEL/CentOS"
    echo ""
    exit 1
fi

if [[ "${1:-}" == "--check" ]]; then
    echo -e "${GREEN}Checking C++ formatting...${NC}"
    find . -type f \( -name "*.cpp" -o -name "*.h" \) \
        ! -path "./cmake*/*" \
        ! -path "./build*/*" \
        ! -path "./third-party/*" \
        -print0 | xargs -0 clang-format --dry-run --Werror || {
        echo -e "${RED}Formatting check failed: run repo_tools/format_code.sh${NC}"
        exit 1
    }
    echo -e "${GREEN}Formatting check passed.${NC}"
    exit 0
fi

echo -e "${GREEN}Formatting C++ source files...${NC}"
echo ""

# Find all .cpp and .h files and format them
find . -type f \( -name "*.cpp" -o -name "*.h" \) \
    ! -path "./cmake*/*" \
    ! -path "./build*/*" \
    ! -path "./third-party/*" \
    -print0 | while IFS= read -r -d '' file; do
    echo -e "${YELLOW}Formatting:${NC} $file"
    clang-format -i "$file"
done

echo ""
echo -e "${GREEN}Done!${NC}"
