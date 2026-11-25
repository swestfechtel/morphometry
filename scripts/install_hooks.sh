#!/usr/bin/env bash
# Install the pre-push hook that runs scripts/check_version.py.
#
# The hooks directory is resolved with `git rev-parse --git-path hooks`, so this also works in a
# linked worktree (where `.git` is a file) and installs into the hooks dir shared by all worktrees.
# An existing hook that differs from the one below is never overwritten.
set -euo pipefail

top=$(git rev-parse --show-toplevel)
hooks=$(git -C "$top" rev-parse --path-format=absolute --git-path hooks)
hook="$hooks/pre-push"

desired=$(cat <<'HOOK'
#!/usr/bin/env bash
# Installed by scripts/install_hooks.sh: refuse pushes that break the release rules.
top=$(git rev-parse --show-toplevel) || exit 1
if [ -x "$top/venv/bin/python" ]; then py="$top/venv/bin/python"; else py=python3; fi
"$py" "$top/scripts/check_version.py" || exit 1
HOOK
)

mkdir -p "$hooks"
if [ -e "$hook" ]; then
    if [ "$(cat "$hook")" = "$desired" ]; then
        echo "pre-push hook already installed: $hook"
        exit 0
    fi
    echo "refusing to overwrite existing, different hook: $hook" >&2
    exit 1
fi
printf '%s\n' "$desired" > "$hook"
chmod +x "$hook"
echo "installed $hook"
