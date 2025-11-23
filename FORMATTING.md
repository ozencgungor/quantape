# Code Formatting Guide

This project uses `clang-format` for automatic C++ code formatting to maintain consistent style.

## Installation

### macOS
```bash
brew install clang-format
```

### Linux (Debian/Ubuntu)
```bash
sudo apt-get install clang-format
```

### Linux (RHEL/CentOS)
```bash
sudo yum install clang-tools-extra
```

## Usage

### Format all project files
Run the provided script from the project root:
```bash
./format_code.sh
```

This will automatically format all `.cpp` and `.h` files in the project, excluding:
- `third-party/` directory
- `build/` directory

### Format a single file
```bash
clang-format -i path/to/file.cpp
```

### Check formatting without modifying files
```bash
clang-format --dry-run --Werror path/to/file.cpp
```

### Format specific files
```bash
clang-format -i Models/*.cpp Math/**/*.h
```

## IDE Integration

### Visual Studio Code
1. Install the "C/C++" extension by Microsoft
2. Add to your `.vscode/settings.json`:
```json
{
    "editor.formatOnSave": true,
    "C_Cpp.clang_format_style": "file",
    "C_Cpp.clang_format_fallbackStyle": "LLVM"
}
```

### CLion / IntelliJ IDEA
1. Go to Settings → Editor → Code Style → C/C++
2. Select "Set from..." → "Predefined Style" → "LLVM"
3. Check "Enable ClangFormat (only for C/C++/Objective-C)"
4. Select "Use .clang-format file"

### Vim/Neovim
Add to your `.vimrc` or `init.vim`:
```vim
" Format on save
autocmd BufWritePre *.cpp,*.h :silent! %!clang-format

" Or use a keybinding
map <C-K> :pyf /usr/local/share/clang/clang-format.py<cr>
```

### Emacs
```elisp
(require 'clang-format)
(global-set-key [C-M-tab] 'clang-format-region)
(add-hook 'c++-mode-hook
    (lambda () (add-hook 'before-save-hook 'clang-format-buffer nil 'local)))
```

## Formatting Style Summary

The project uses a style based on LLVM with these key features:

- **Indentation**: 4 spaces (no tabs)
- **Line length**: 100 characters
- **Braces**: K&R style (opening brace on same line)
- **Pointer/Reference**: Left-aligned (`int* ptr`, `int& ref`)
- **Namespace**: No indentation inside namespaces
- **Include ordering**: Project headers → Standard library → Third-party → Others

## Pre-commit Hook (Optional)

To automatically format code before each commit, create `.git/hooks/pre-commit`:

```bash
#!/bin/bash

# Format all staged C++ files
for file in $(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(cpp|h)$'); do
    if [ -f "$file" ]; then
        clang-format -i "$file"
        git add "$file"
    fi
done
```

Then make it executable:
```bash
chmod +x .git/hooks/pre-commit
```

## Configuration

The formatting rules are defined in `.clang-format` at the project root.
To modify the style, edit this file and re-run `./format_code.sh`.

For more details on available options, see:
https://clang.llvm.org/docs/ClangFormatStyleOptions.html
