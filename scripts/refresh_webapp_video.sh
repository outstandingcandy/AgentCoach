#!/usr/bin/env bash
# Copy the most recent annotated.mp4 from a pipeline run-dir into the
# stable webapp video location, so the running web viewer picks it up.
#
# Usage:
#   bash scripts/refresh_webapp_video.sh
#   bash scripts/refresh_webapp_video.sh <run-dir>

set -euo pipefail

RUN_DIR="${1:-output/full_pipeline/full_v2}"
SRC="$RUN_DIR/annotated_video/annotated.mp4"
DST="webapp_videos/annotated.mp4"

if [ ! -e "$SRC" ]; then
  echo "ERROR: source not found: $SRC" >&2
  exit 1
fi

mkdir -p "$(dirname "$DST")"

# Resolve symlinks so we copy the actual mp4, not a dangling link.
RESOLVED="$(readlink -f "$SRC")"
echo "copying $RESOLVED -> $DST"

# Atomic replace: copy to a sibling tmp file then rename, so the webapp
# never sees a half-written video while serving Range requests.
TMP="$DST.tmp.$$"
cp -f "$RESOLVED" "$TMP"
mv -f "$TMP" "$DST"

ls -lh "$DST"
