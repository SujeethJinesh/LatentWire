#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run a command that will fail
timeout 5 bash "$SCRIPT_DIR/../RUN_ALL.sh" experiment --phase 99 --no-interactive 2>&1 || true
