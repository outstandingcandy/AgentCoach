#!/usr/bin/env bash
# GoalInsight offline container entrypoint.
#
# - Seeds /workspace/configs/ with 3 starter templates (fifa / futsal /
#   children) on first launch so a brand-new workspace shows usable
#   options in the Library config dropdown without the user having to
#   ``docker cp`` anything.
# - Existing files are NEVER overwritten — users who edited their
#   templates keep their changes across container restarts.
# - Then exec's into goalinsight-web (default) or whatever the user
#   passed via ``docker run -- <cmd>``.

set -e

TEMPLATE_DIR=/opt/goalinsight/example_configs
WORKSPACE=/workspace
WORKSPACE_CONFIGS="$WORKSPACE/configs"

declare -A SEEDS=(
    [fifa.yaml]=fifa_sample.yaml
    [futsal.yaml]=futsal_sample.yaml
    [children.yaml]=kids_soccer_sample.yaml
)

# Warn when /workspace lives inside the container's writable layer
# rather than a bind-mounted host directory. ``docker rm`` would
# throw away every pipeline output the user just produced, which is a
# common first-day footgun. ``findmnt`` returns success only when the
# path is a mount point; on the container layer it returns 2.
if command -v findmnt >/dev/null 2>&1; then
    if ! findmnt -n -T "$WORKSPACE" 2>/dev/null | grep -q "^$WORKSPACE "; then
        echo "================================================================" >&2
        echo "WARNING: /workspace is not a bind-mount. Pipeline outputs," >&2
        echo "uploaded videos, and annotations will live INSIDE the container" >&2
        echo "and disappear when you 'docker rm' it." >&2
        echo "Re-run with:  -v \"\$PWD/workspace:/workspace\"" >&2
        echo "================================================================" >&2
    fi
fi

# The Workspace.ensure() path inside goalinsight-web creates this
# dir, but it only fires AFTER exec — so we mkdir here so seeding can
# always run.
mkdir -p "$WORKSPACE_CONFIGS"
for dest_name in "${!SEEDS[@]}"; do
    src="$TEMPLATE_DIR/${SEEDS[$dest_name]}"
    dest="$WORKSPACE_CONFIGS/$dest_name"
    if [ -f "$src" ] && [ ! -e "$dest" ]; then
        cp "$src" "$dest"
        echo "seeded $dest from $src"
    fi
done

# Hand off to goalinsight-web (default) or the user's override.
exec goalinsight-web "$@"
