#!/usr/bin/env bash
set -euo pipefail
sleep 3600
cd /workspace/LatentWire
# Commit and push any new artifacts
if ! git diff --quiet || ! git diff --cached --quiet; then
  git add -A
  if ! git diff --cached --quiet; then
    git commit -m "Upload M3 Logs (auto)"
  fi
  git push
else
  # No changes; still attempt push in case of remote updates
  git push
fi
/usr/sbin/shutdown -h now
