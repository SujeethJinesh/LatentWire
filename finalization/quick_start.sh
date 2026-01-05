#!/bin/bash
# ============================================================================
# Quick Start Script for LatentWire Finalization Experiments
# ============================================================================
# Usage: bash quick_start.sh
#
# This script:
#   1. Validates your environment
#   2. Submits the SLURM job to HPC
#   3. Provides monitoring commands
# ============================================================================

set -e

# Set environment for any Python commands
export PYTHONUNBUFFERED=1  # Critical: Immediate output flushing

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print header
echo ""
echo "=============================================================="
echo "   🚀 LatentWire Quick Start - Finalization Experiments"
echo "=============================================================="
echo ""

# Step 1: Environment validation
echo -e "${BLUE}Step 1: Validating environment...${NC}"
echo ""

# Check if we're in the right directory
if [[ ! -f "submit_finalization.slurm" ]]; then
    echo -e "${RED}✗ Error: submit_finalization.slurm not found${NC}"
    echo "  Please run this script from the finalization/ directory:"
    echo "    cd finalization && bash quick_start.sh"
    exit 1
fi

# Check if we can reach HPC
echo -n "Checking SSH connection to HPC... "
if ssh -o ConnectTimeout=5 -o BatchMode=yes sjinesh@marlowe.crc.nd.edu "echo connected" &>/dev/null; then
    echo -e "${GREEN}✓ Connected${NC}"
else
    echo -e "${YELLOW}⚠ Cannot verify HPC connection${NC}"
    echo "  Make sure you can SSH to marlowe.crc.nd.edu"
    echo "  You may need to set up SSH keys or VPN"
fi

echo ""
echo -e "${GREEN}✓ Environment validation complete${NC}"
echo ""

# Step 2: Submit the job
echo -e "${BLUE}Step 2: Submitting SLURM job to HPC...${NC}"
echo ""

# Show what will be submitted
echo "Job details:"
echo "  • Script: submit_finalization.slurm"
echo "  • GPUs: 4× H100"
echo "  • Time: 12 hours"
echo "  • Experiments: GSM8K, AGNews, SST-2, TREC"
echo ""

echo "Submitting job..."
echo ""

# Create submission script
cat > submit_job.sh << 'EOF'
#!/bin/bash
# This runs on the HPC

cd /projects/m000066/sujinesh/LatentWire
git pull

# Submit the job and capture the job ID
JOB_OUTPUT=$(sbatch finalization/submit_finalization.slurm 2>&1)
if [[ $? -eq 0 ]]; then
    JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP '\d+$')
    echo "SUCCESS:$JOB_ID"
else
    echo "ERROR:$JOB_OUTPUT"
fi
EOF

# Submit to HPC
RESULT=$(ssh sjinesh@marlowe.crc.nd.edu 'bash -s' < submit_job.sh 2>&1)

if [[ $RESULT == SUCCESS:* ]]; then
    JOB_ID=${RESULT#SUCCESS:}
    echo -e "${GREEN}✓ Job submitted successfully!${NC}"
    echo -e "  Job ID: ${YELLOW}${JOB_ID}${NC}"
else
    echo -e "${RED}✗ Failed to submit job${NC}"
    echo "  Error: ${RESULT#ERROR:}"
    echo ""
    echo "You can try manually:"
    echo "  ssh sjinesh@marlowe.crc.nd.edu"
    echo "  cd /projects/m000066/sujinesh/LatentWire"
    echo "  git pull"
    echo "  sbatch finalization/submit_finalization.slurm"
    exit 1
fi

echo ""

# Step 3: Monitoring instructions
echo -e "${BLUE}Step 3: Monitor your experiments${NC}"
echo ""

echo "Your experiments are now running! Here's how to monitor them:"
echo ""

echo -e "${YELLOW}📊 Quick Commands:${NC}"
echo ""

# Create monitor script
cat > monitor.sh << EOF
#!/bin/bash
# Monitor script for job $JOB_ID

echo "Checking job status..."
ssh sjinesh@marlowe.crc.nd.edu "squeue -j $JOB_ID"

echo ""
echo "Latest log output:"
ssh sjinesh@marlowe.crc.nd.edu "tail -20 /projects/m000066/sujinesh/LatentWire/runs/finalization_${JOB_ID}.log 2>/dev/null || echo 'Log not yet available'"
EOF

chmod +x monitor.sh

echo "  1. Check job status:"
echo -e "     ${GREEN}ssh sjinesh@marlowe.crc.nd.edu \"squeue -j ${JOB_ID}\"${NC}"
echo ""

echo "  2. Watch live logs:"
echo -e "     ${GREEN}ssh sjinesh@marlowe.crc.nd.edu \"tail -f /projects/m000066/sujinesh/LatentWire/runs/finalization_${JOB_ID}.log\"${NC}"
echo ""

echo "  3. Quick monitor (created for you):"
echo -e "     ${GREEN}./monitor.sh${NC}"
echo ""

echo "  4. Cancel if needed:"
echo -e "     ${GREEN}ssh sjinesh@marlowe.crc.nd.edu \"scancel ${JOB_ID}\"${NC}"
echo ""

# Step 4: Next steps
echo -e "${BLUE}What happens next:${NC}"
echo ""
echo "  1. The job will run 4 experiments in sequence:"
echo "     • GSM8K (math reasoning) - ~3 hours"
echo "     • AGNews (news classification) - ~2 hours"
echo "     • SST-2 (sentiment analysis) - ~2 hours"
echo "     • TREC (question classification) - ~2 hours"
echo ""
echo "  2. Results will be saved to:"
echo "     • runs/finalization/gsm8k_results.json"
echo "     • runs/finalization/agnews_results.json"
echo "     • runs/finalization/sst2_results.json"
echo "     • telepathy/trec_results.json"
echo ""
echo "  3. When complete, results will be automatically pushed to git"
echo ""
echo "  4. To get results locally after completion:"
echo -e "     ${GREEN}git pull${NC}"
echo ""

# Create results checker
cat > check_results.sh << 'EOF'
#!/bin/bash
# Check if results are ready

echo "Pulling latest from git..."
git pull

echo ""
echo "Checking for results files..."

if [[ -f "runs/finalization/gsm8k_results.json" ]]; then
    echo "✓ GSM8K results found"
    echo "  Accuracy: $(python -c "import json; print(json.load(open('runs/finalization/gsm8k_results.json'))['summary']['latent']['accuracy'])" 2>/dev/null || echo "parse error")"
else
    echo "✗ GSM8K results not yet available"
fi

if [[ -f "runs/finalization/agnews_results.json" ]]; then
    echo "✓ AGNews results found"
    echo "  Accuracy: $(python -c "import json; print(json.load(open('runs/finalization/agnews_results.json'))['accuracy']['latent'])" 2>/dev/null || echo "parse error")"
else
    echo "✗ AGNews results not yet available"
fi

if [[ -f "runs/finalization/sst2_results.json" ]]; then
    echo "✓ SST-2 results found"
    echo "  Accuracy: $(python -c "import json; print(json.load(open('runs/finalization/sst2_results.json'))['accuracy']['latent'])" 2>/dev/null || echo "parse error")"
else
    echo "✗ SST-2 results not yet available"
fi

if [[ -f "telepathy/trec_results.json" ]]; then
    echo "✓ TREC results found"
    echo "  Accuracy: $(python -c "import json; print(json.load(open('telepathy/trec_results.json'))['accuracy']['latent'])" 2>/dev/null || echo "parse error")"
else
    echo "✗ TREC results not yet available"
fi

echo ""
echo "Run './check_results.sh' again later to check for updates"
EOF

chmod +x check_results.sh

echo -e "${YELLOW}📁 Helper scripts created:${NC}"
echo "  • ./monitor.sh - Quick status check"
echo "  • ./check_results.sh - Check if results are ready"
echo ""

echo "=============================================================="
echo -e "${GREEN}✨ Setup complete! Your experiments are running.${NC}"
echo "=============================================================="
echo ""
echo "Estimated completion time: ~9-10 hours"
echo "Job ID: ${JOB_ID}"
echo ""

# Clean up temp file
rm -f submit_job.sh