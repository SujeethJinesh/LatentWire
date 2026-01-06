#!/bin/bash
# =============================================================================
# Git Configuration Verification Script for HPC
# =============================================================================
# Run this on HPC to verify Git is properly configured:
#   cd /projects/m000066/sujinesh/LatentWire
#   bash telepathy/verify_git_setup.sh
# =============================================================================

set -e

echo "=============================================================="
echo "Git Configuration Verification"
echo "=============================================================="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "User: $USER"
echo "Working dir: $(pwd)"
echo ""

# Check Git version
echo "1. Git Version:"
git --version
echo ""

# Check Git configuration
echo "2. Git Configuration:"
echo "  User name: $(git config user.name 2>/dev/null || echo 'NOT SET')"
echo "  User email: $(git config user.email 2>/dev/null || echo 'NOT SET')"
echo "  Remote URL: $(git config remote.origin.url 2>/dev/null || echo 'NOT SET')"
echo ""

# Check SSH configuration
echo "3. SSH Key Check:"
if [ -f ~/.ssh/id_rsa ] || [ -f ~/.ssh/id_ed25519 ]; then
    echo "  SSH keys found:"
    ls -la ~/.ssh/id_* 2>/dev/null | grep -v ".pub" | awk '{print "    " $NF}'
else
    echo "  WARNING: No SSH keys found in ~/.ssh/"
    echo "  You may need to set up SSH keys for GitHub"
fi
echo ""

# Test GitHub SSH connection
echo "4. GitHub SSH Connection Test:"
echo "  Testing connection to GitHub..."
if ssh -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
    echo "  SUCCESS: Can authenticate with GitHub"
else
    SSH_OUTPUT=$(ssh -T git@github.com 2>&1)
    if echo "$SSH_OUTPUT" | grep -q "Hi"; then
        echo "  SUCCESS: GitHub authentication working"
        echo "  $SSH_OUTPUT"
    else
        echo "  WARNING: Cannot authenticate with GitHub via SSH"
        echo "  Output: $SSH_OUTPUT"
        echo ""
        echo "  To fix, you may need to:"
        echo "    1. Generate SSH key: ssh-keygen -t ed25519 -C 'your_email@example.com'"
        echo "    2. Add key to GitHub: cat ~/.ssh/id_ed25519.pub"
        echo "    3. Add the public key to GitHub Settings > SSH Keys"
    fi
fi
echo ""

# Check current branch and status
echo "5. Repository Status:"
echo "  Current branch: $(git branch --show-current)"
echo "  Status:"
git status --short | head -10 || echo "    (clean)"
echo ""

# Test git operations
echo "6. Testing Git Operations:"

# Test pull
echo "  Testing git pull..."
if timeout 10 git pull --dry-run 2>&1 | grep -q "Already up to date\|Would fast-forward"; then
    echo "    SUCCESS: Can pull from remote"
else
    PULL_OUTPUT=$(timeout 10 git pull --dry-run 2>&1 || echo "TIMEOUT/ERROR")
    echo "    WARNING: Pull test failed or timed out"
    echo "    Output: $PULL_OUTPUT"
fi

# Check if there are any large files that shouldn't be committed
echo ""
echo "7. Large File Check (files > 50MB that might cause issues):"
find runs -type f -size +50M 2>/dev/null | head -10 || echo "  No large files found"

# Check .gitignore
echo ""
echo "8. .gitignore Check:"
echo "  Key patterns in .gitignore:"
grep -E "^\*\.pt|ckpt|\*\.safetensors" .gitignore | head -5 | sed 's/^/    /'
echo "  These patterns prevent large model files from being committed"

# Provide recommendations
echo ""
echo "=============================================================="
echo "Recommendations:"
echo "=============================================================="

# Check for potential issues
ISSUES_FOUND=false

if ! git config user.name > /dev/null 2>&1; then
    echo "• Set git user name: git config user.name 'Your Name'"
    ISSUES_FOUND=true
fi

if ! git config user.email > /dev/null 2>&1; then
    echo "• Set git email: git config user.email 'your.email@example.com'"
    ISSUES_FOUND=true
fi

if ! ssh -T git@github.com 2>&1 | grep -q "successfully authenticated\|Hi"; then
    echo "• Set up SSH keys for GitHub (see instructions above)"
    ISSUES_FOUND=true
fi

if [ "$ISSUES_FOUND" = false ]; then
    echo "✓ Git configuration looks good!"
    echo "✓ You should be able to pull and push successfully"
fi

echo ""
echo "=============================================================="
echo "Quick Commands Reference:"
echo "=============================================================="
echo "• Check status:      git status"
echo "• Pull latest:       git pull"
echo "• Add changes:       git add runs/*.log runs/*.json"
echo "• Commit:           git commit -m 'results: experiment description'"
echo "• Push:             git push"
echo "• View log:         git log --oneline -10"
echo "=============================================================="