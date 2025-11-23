#!/bin/bash

# Script to format all C++ source files using clang-format
# Usage: ./format_code.sh

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

echo -e "${GREEN}Formatting C++ source files...${NC}"
echo ""

# Find all .cpp and .h files and format them
find . -type f \( -name "*.cpp" -o -name "*.h" \) \
    ! -path "./cmake*/*" \
    ! -path "./build/*" \
    ! -path "./third-party/*" \
    -print0 | while IFS= read -r -d '' file; do
    echo -e "${YELLOW}Formatting:${NC} $file"
    clang-format -i "$file"
done

echo ""
echo -e "${GREEN}Done!${NC}"
