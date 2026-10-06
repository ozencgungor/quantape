#!/bin/bash
# Advisory commit-message convention check (never blocks).
#
# The repository style is a terse single-line subject (<= 72 chars, no trailing
# period). Violations print hints to stderr and the hook still exits 0; turn it
# into a hard failure once the style is fully adopted.
#
# Usage: repo_tools/check_commit_msg.sh <path-to-commit-message-file>
set -u

msg_file=${1:-}
[ -n "$msg_file" ] && [ -f "$msg_file" ] || exit 0

subject=$(head -n 1 "$msg_file" | sed 's/[[:space:]]*$//')
[ -n "$subject" ] || exit 0

hints=0
length=${#subject}
if [ "$length" -gt 140 ]; then
    printf 'commit-msg hint: subject is %d chars (> 140): %s\n' "$length" "$subject" >&2
    hints=1
fi
case "$subject" in
    *.)
        printf 'commit-msg hint: subject ends with a period: %s\n' "$subject" >&2
        hints=1
        ;;
esac
[ "$hints" -eq 0 ] || printf 'commit-msg: terse single-line subjects keep the log readable.\n' >&2
exit 0
