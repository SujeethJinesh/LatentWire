# Git Configuration for HPC SLURM Jobs

This document explains the Git setup requirements for running experiments on the HPC cluster.

## Critical Requirements

### 1. Repository Configuration
- **Remote URL**: Must use SSH (`git@github.com:SujeethJinesh/LatentWire.git`)
- **Working Directory**: `/projects/m000066/sujinesh/LatentWire`
- **Authentication**: SSH key-based (no password prompts in batch jobs)

### 2. Files Committed vs Ignored

**COMMITTED** (small, text-based results):
- `runs/*.log` - Execution logs
- `runs/*.err` - Error logs
- `runs/*.json` - Results and metrics
- `figures/*.png` - Visualization plots
- `figures/*.pdf` - Generated figures

**IGNORED** (large binary files):
- `*.pt`, `*.pth` - PyTorch checkpoints
- `*.safetensors` - Model weights
- `*ckpt*` - Any checkpoint files
- Model directories - Full model snapshots

### 3. Git Commands in SLURM Scripts

#### Pull Strategy (with fallback)
```bash
# Try to pull, with stash fallback for uncommitted changes
if ! git pull; then
    echo "WARNING: git pull failed. Attempting to stash and retry..."
    git stash push -m "SLURM job $SLURM_JOB_ID auto-stash"
    git pull || echo "Pull failed - continuing with existing code"
    git stash pop 2>/dev/null || true
fi
```

#### Commit Strategy (selective adding)
```bash
# Only add specific file types (not git add -A!)
git add runs/*.log runs/*.err runs/*.json runs/**/*.log runs/**/*.json 2>/dev/null || true
git add figures/*.png figures/*.pdf 2>/dev/null || true

# Commit with descriptive message including job ID
git commit -m "results: experiment name (SLURM job $SLURM_JOB_ID)

Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.1 <noreply@anthropic.com>"
```

#### Push Strategy (with retry)
```bash
# Push with retry logic for transient failures
RETRY_COUNT=0
while [ $RETRY_COUNT -lt 3 ]; do
    if git push; then
        echo "Successfully pushed to remote"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        echo "Push attempt $RETRY_COUNT failed"
        [ $RETRY_COUNT -lt 3 ] && sleep 5 && git pull --rebase=false || true
    fi
done
```

## Setting Up SSH Keys on HPC

### 1. Generate SSH Key (if needed)
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
# Press Enter for default location
# Enter passphrase (or leave empty for no passphrase)
```

### 2. Add Public Key to GitHub
```bash
# Display public key
cat ~/.ssh/id_ed25519.pub

# Copy the output and add to GitHub:
# GitHub.com > Settings > SSH and GPG keys > New SSH key
```

### 3. Test Connection
```bash
ssh -T git@github.com
# Should see: "Hi username! You've successfully authenticated..."
```

### 4. Configure Git Identity
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Verification Script

Run the verification script to check your setup:
```bash
cd /projects/m000066/sujinesh/LatentWire
bash telepathy/verify_git_setup.sh
```

This will check:
- Git version and configuration
- SSH key presence
- GitHub authentication
- Repository status
- Pull/push capability

## Common Issues and Solutions

### Issue 1: Permission Denied (publickey)
**Symptom**: `git pull` or `git push` fails with "Permission denied (publickey)"

**Solution**:
1. Check SSH key exists: `ls ~/.ssh/id_*`
2. Add key to ssh-agent: `ssh-add ~/.ssh/id_ed25519`
3. Verify key is on GitHub: Compare `cat ~/.ssh/id_ed25519.pub` with GitHub settings

### Issue 2: Large File Rejection
**Symptom**: Push fails with "large files" error

**Solution**:
1. Check `.gitignore` includes patterns for large files
2. Remove large files from staging: `git reset HEAD large_file.pt`
3. Use selective `git add` (not `git add -A`)

### Issue 3: Merge Conflicts
**Symptom**: Pull fails due to merge conflicts

**Solution**:
1. Stash local changes: `git stash`
2. Pull remote changes: `git pull`
3. Apply stash: `git stash pop`
4. Resolve conflicts if any

### Issue 4: Authentication in SLURM Jobs
**Symptom**: Git works interactively but fails in SLURM jobs

**Solution**:
1. Ensure SSH key has no passphrase (or use ssh-agent)
2. Test with: `ssh -T git@github.com`
3. Use SSH URL, not HTTPS: `git@github.com:user/repo.git`

## Best Practices

1. **Always pull before experiments**: Ensures latest code
2. **Commit frequently**: After each experiment completes
3. **Use descriptive messages**: Include job ID and experiment type
4. **Don't commit checkpoints**: Keep repo size manageable
5. **Handle failures gracefully**: Use `|| true` for non-critical operations
6. **Test locally first**: Run `verify_git_setup.sh` before submitting jobs

## Example SLURM Script Section

```bash
#!/bin/bash
#SBATCH --job-name=my_experiment
# ... other SBATCH directives ...

# Working directory
cd /projects/m000066/sujinesh/LatentWire

# Git operations with proper error handling
echo "Setting up Git..."
git config user.name > /dev/null 2>&1 || git config user.name "SLURM Job"
git config user.email > /dev/null 2>&1 || git config user.email "slurm@hpc"

echo "Pulling latest code..."
git pull || { git stash; git pull; git stash pop 2>/dev/null || true; }

# Run experiment
python experiment.py

# Save results
echo "Saving results..."
git add runs/*.log runs/*.json
git commit -m "results: my experiment (job $SLURM_JOB_ID)" || true
git push || echo "Push failed - results saved locally"
```

## Monitoring Git Operations

View Git activity in SLURM logs:
```bash
# Check SLURM output
tail -f /projects/m000066/sujinesh/LatentWire/runs/slurm_*.log

# Check Git history
git log --oneline -10

# Check what was committed
git show --stat HEAD
```