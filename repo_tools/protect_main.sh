#!/bin/bash
# Protect the main branch with a repository ruleset (requires admin + gh auth).
#
#   repo_tools/protect_main.sh [--dry-run] [--strict] [--bypass-user LOGIN]
#
# Sets, for refs/heads/main:
#   - block deletions and non-fast-forward (force) pushes
#   - require a pull request before merging (0 approvals: solo maintainer)
#   - require linear history
#   - require the CI/analysis checks below
#
# By default the authenticated user is added as a bypass actor with mode
# "always": the owner can still push directly to main, everyone else must open
# a pull request. Use --strict to remove all bypass actors, or
# --bypass-user LOGIN to bypass a different account.
#
# Run again after changing the check list: the ruleset is updated in place.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(git -C "$script_dir" rev-parse --show-toplevel 2>/dev/null) || {
    echo "error: not inside a git repository" >&2
    exit 2
}
cd "$root"

DRY_RUN=0
STRICT=0
BYPASS_USER=${QUANTAPE_BYPASS_USER:-}
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --strict) STRICT=1 ;;
        --bypass-user)
            shift
            BYPASS_USER=${1:?--bypass-user needs a login}
            ;;
        *)
            echo "usage: protect_main.sh [--dry-run] [--strict] [--bypass-user LOGIN]" >&2
            exit 2
            ;;
    esac
    shift
done

# Matrix jobs surface as "<job> (<os>, <build_type>)" check names.
REQUIRED_CHECKS=(
    "build-test (macos-14, Debug)"
    "build-test (macos-14, Release)"
    "build-test (ubuntu-24.04, Debug)"
    "build-test (ubuntu-24.04, Release)"
    "clang-format"
    "policy"
    "asan-ubsan"
    "clang-tidy"
)

checks_json=""
for c in "${REQUIRED_CHECKS[@]}"; do
    checks_json+="${checks_json:+,}{\"context\":\"$c\"}"
done

# Bypass actor: the owner keeps direct-push access; everyone else must PR.
bypass_json=""
if [ "$STRICT" -eq 0 ]; then
    actor_id=""
    if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
        if [ -n "$BYPASS_USER" ]; then
            actor_id=$(gh api "users/$BYPASS_USER" -q .id)
        else
            actor_id=$(gh api user -q .id)
        fi
    fi
    if [ -n "$actor_id" ]; then
        bypass_json="{\"actor_id\":$actor_id,\"actor_type\":\"User\",\"bypass_mode\":\"always\"}"
    elif [ "$DRY_RUN" -eq 0 ]; then
        echo "error: cannot resolve bypass user (gh unavailable/unauthenticated); use --strict" >&2
        exit 2
    fi
fi

payload=$(
    cat <<JSON
{
  "name": "main protection",
  "target": "branch",
  "enforcement": "active",
  "conditions": {
    "ref_name": { "include": ["refs/heads/main"], "exclude": [] }
  },
  "rules": [
    { "type": "deletion" },
    { "type": "non_fast_forward" },
    { "type": "required_linear_history" },
    {
      "type": "pull_request",
      "parameters": {
        "required_approving_review_count": 0,
        "dismiss_stale_reviews_on_push": false,
        "require_code_owner_review": false,
        "require_last_push_approval": false,
        "required_review_thread_resolution": false
      }
    },
    {
      "type": "required_status_checks",
      "parameters": {
        "strict_required_status_checks_policy": false,
        "required_status_checks": [$checks_json]
      }
    }
  ],
  "bypass_actors": [$bypass_json]
}
JSON
)

printf '%s' "$payload" | python3 -m json.tool > /dev/null

if [ "$DRY_RUN" -eq 1 ]; then
    printf '%s\n' "$payload"
    exit 0
fi

command -v gh >/dev/null 2>&1 || {
    echo "error: gh CLI not found" >&2
    exit 2
}
gh auth status >/dev/null 2>&1 || {
    echo "error: gh is not authenticated (run: gh auth login)" >&2
    exit 2
}

repo=$(gh repo view --json nameWithOwner -q .nameWithOwner)
existing=$(gh api "repos/$repo/rulesets" --jq '.[] | select(.name == "main protection") | .id' | head -1)

if [ -n "$existing" ]; then
    printf '%s' "$payload" | gh api -X PUT "repos/$repo/rulesets/$existing" --input - > /dev/null
    echo "updated ruleset 'main protection' (#$existing) on $repo"
else
    printf '%s' "$payload" | gh api -X POST "repos/$repo/rulesets" --input - > /dev/null
    echo "created ruleset 'main protection' on $repo"
fi

echo "required checks:"
for c in "${REQUIRED_CHECKS[@]}"; do
    echo "  - $c"
done
