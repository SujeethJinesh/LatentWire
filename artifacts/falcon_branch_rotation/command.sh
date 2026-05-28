#!/usr/bin/env bash
set -euo pipefail

# Guarded template only. No BranchRot diagnostic runner exists yet.
# The orchestrator must set ORCHESTRATOR_GATE=1 after ParoQuant Falcon smoke
# or another explicit gate makes branch-local placement evidence necessary.
: "${ORCHESTRATOR_GATE:?Set ORCHESTRATOR_GATE=1 only after the orchestrator gates Falcon BranchRot diagnostic work.}"

cd /workspace/LatentWire
source .venv_gpu/bin/activate

echo "BranchRot diagnostic is not implemented as a runnable script yet."
echo "Use artifacts/falcon_branch_rotation/diagnostic_config.json as the implementation spec."
exit 2

