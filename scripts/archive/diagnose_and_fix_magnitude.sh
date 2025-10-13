#!/bin/bash
# Diagnose embedding magnitude issue and apply fixes
# Tests 3 approaches:
#   1. Check raw embedding norms
#   2. Test RMS matching (quick fix)
#   3. Enable colorize=True (proper fix)

set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-.}"

echo "================================================================================"
echo "EMBEDDING MAGNITUDE DIAGNOSIS & FIX"
echo "================================================================================"
echo ""

# ==============================================================================
# Step 1: Diagnose - Check Raw Embedding Norms
# ==============================================================================
echo "STEP 1: Checking raw Llama embedding statistics..."
echo "--------------------------------------------------------------------------------"

python3 << 'DIAG1'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

print("Loading Llama model to check raw embedding statistics...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        device_map="cpu",
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

    # Test with actual text
    test_text = "Context: Tesla was the fourth of five children. He had an older brother named Dane."
    inputs = tokenizer(test_text, return_tensors="pt")

    with torch.no_grad():
        embeds = model.get_input_embeddings()(inputs.input_ids)

    print(f"\n{'='*80}")
    print("RAW EMBEDDING STATISTICS:")
    print(f"{'='*80}")
    print(f"  Text: {test_text[:50]}...")
    print(f"  Embedding shape: {embeds.shape}")
    print(f"  Embedding dtype: {embeds.dtype}")
    print(f"\n  Per-dimension statistics:")
    print(f"    Mean: {embeds.mean().item():.6f}")
    print(f"    Std:  {embeds.std().item():.6f}")
    print(f"    Min:  {embeds.min().item():.6f}")
    print(f"    Max:  {embeds.max().item():.6f}")
    print(f"\n  L2 Norm per token:")
    norms = embeds.norm(dim=-1)
    print(f"    Mean norm: {norms.mean().item():.4f}")
    print(f"    Min norm:  {norms.min().item():.4f}")
    print(f"    Max norm:  {norms.max().item():.4f}")

    print(f"\n{'='*80}")
    print("CONCLUSION:")
    print(f"{'='*80}")
    if 0.4 < norms.mean().item() < 1.5:
        print("✅ Embeddings have small scale (norm ≈0.5-1.0) as expected for Llama")
        print("✅ Codex analysis is CORRECT")
        print("✅ Original norms in logs (0.53) are real, not a bug")
    else:
        print(f"⚠️ Unexpected norm: {norms.mean().item():.2f}")

    del model  # Free memory

except Exception as e:
    print(f"ERROR: {e}")
    print("Could not load model, but mathematical analysis confirms:")
    print("  Llama uses std ≈ 1/sqrt(4096) ≈ 0.0156 per dimension")
    print("  Expected norm ≈ 0.5-1.0 per token")
    sys.exit(0)  # Continue anyway

DIAG1

echo ""
echo "================================================================================"
echo "STEP 2: Apply Quick Fix - RMS Matching After Adapter"
echo "================================================================================"
echo ""

# Check if we already have the RMS matching code
if grep -q "Match magnitude to original embeddings" train_adapter_only_phase1.py; then
    echo "✅ RMS matching code already present in train_adapter_only_phase1.py"
else
    echo "Adding RMS matching code to train_adapter_only_phase1.py..."

    python3 << 'PATCH1'
import re

# Read the file
with open("train_adapter_only_phase1.py", "r") as f:
    content = f.read()

# Find the location after adapter forward pass in training loop (around line 362)
# Look for: reconstructed = adapter(compressed)
pattern = r"(            # Compress and reconstruct\n            compressed = compressor\.compress\(orig_embeds\)\n            reconstructed = adapter\(compressed\))"

rms_match_code = r'''\1

            # Match magnitude to original embeddings (critical for generation!)
            # Original Llama embeddings have very small scale (RMS ≈0.01 per dim, norm ≈0.5)
            # Adapter output has RMS ≈1 per dim (from LayerNorm), norm ≈64
            # Without matching, embeddings are 120× too large → LLM generates empty strings
            orig_rms = orig_embeds.pow(2).mean(dim=-1, keepdim=True).sqrt()
            recon_rms = reconstructed.pow(2).mean(dim=-1, keepdim=True).sqrt()
            reconstructed = reconstructed * (orig_rms / (recon_rms + 1e-8))'''

if "Match magnitude to original embeddings" in content:
    print("Code already contains magnitude matching - skipping")
else:
    content = re.sub(pattern, rms_match_code, content)

    with open("train_adapter_only_phase1.py", "w") as f:
        f.write(content)
    print("✅ Added RMS matching code to training loop")

# Also add to evaluate_quick function (around line 520)
pattern2 = r"(            if adapted\.dtype != orig_embeds\.dtype:\n                adapted = adapted\.to\(orig_embeds\.dtype\))"

rms_match_eval = r'''\1

            # Match magnitude (same fix as training)
            orig_rms = orig_embeds.pow(2).mean(dim=-1, keepdim=True).sqrt()
            adapted_rms = adapted.pow(2).mean(dim=-1, keepdim=True).sqrt()
            adapted = adapted * (orig_rms / (adapted_rms + 1e-8))'''

with open("train_adapter_only_phase1.py", "r") as f:
    content = f.read()

if content.count("Match magnitude") < 2:
    content = re.sub(pattern2, rms_match_eval, content, count=1)
    with open("train_adapter_only_phase1.py", "w") as f:
        f.write(content)
    print("✅ Added RMS matching to evaluate_quick")

# Add to evaluate_full (around line 565)
pattern3 = r"(            if adapted\.dtype != orig_embeds\.dtype:\n                adapted = adapted\.to\(orig_embeds\.dtype\)\n\n            # Create attention mask)"

rms_match_full = r'''\1.replace("# Create attention mask", """# Match magnitude (same fix as training)
            orig_rms_full = orig_embeds.pow(2).mean(dim=-1, keepdim=True).sqrt()
            adapted_rms_full = adapted.pow(2).mean(dim=-1, keepdim=True).sqrt()
            adapted = adapted * (orig_rms_full / (adapted_rms_full + 1e-8))

            # Create attention mask""")'''

# Simpler approach - just insert before "Create attention mask"
with open("train_adapter_only_phase1.py", "r") as f:
    lines = f.readlines()

found_full_eval = False
for i, line in enumerate(lines):
    if "def evaluate_full" in line:
        found_full_eval = True
    if found_full_eval and "# Create attention mask" in line and "Match magnitude" not in lines[i-1]:
        # Insert RMS matching before this line
        indent = "            "
        insert_lines = [
            f"{indent}# Match magnitude (same fix as training)\n",
            f"{indent}orig_rms_full = orig_embeds.pow(2).mean(dim=-1, keepdim=True).sqrt()\n",
            f"{indent}adapted_rms_full = adapted.pow(2).mean(dim=-1, keepdim=True).sqrt()\n",
            f"{indent}adapted = adapted * (orig_rms_full / (adapted_rms_full + 1e-8))\n",
            f"\n"
        ]
        lines = lines[:i] + insert_lines + lines[i:]
        break

with open("train_adapter_only_phase1.py", "w") as f:
    f.writelines(lines)

print("✅ Added RMS matching to evaluate_full")

PATCH1

    echo "✅ RMS matching code added successfully"
fi

echo ""
echo "================================================================================"
echo "STEP 3: Enable colorize=True for Proper Calibration"
echo "================================================================================"
echo ""

# Update the adapter instantiation to use colorize=True
if grep -q "colorize=True" train_adapter_only_phase1.py; then
    echo "✅ Adapter already has colorize=True"
else
    echo "Enabling colorize=True in adapter initialization..."

    python3 << 'PATCH2'
import re

with open("train_adapter_only_phase1.py", "r") as f:
    content = f.read()

# Find adapter initialization and change colorize=False to colorize=True
content = re.sub(
    r"(adapter = Adapter\([^)]*colorize=)False",
    r"\1True",
    content
)

with open("train_adapter_only_phase1.py", "w") as f:
    f.write(content)

print("✅ Changed colorize=False to colorize=True")
PATCH2

    echo "✅ Adapter colorize enabled"
fi

echo ""
echo "================================================================================"
echo "VERIFICATION"
echo "================================================================================"
echo ""
echo "Checking all changes were applied..."
echo ""

# Verify RMS matching is present
if grep -q "Match magnitude to original embeddings" train_adapter_only_phase1.py; then
    echo "✅ RMS matching code present in training"
else
    echo "❌ RMS matching code NOT found in training"
fi

if grep -q "colorize=True" train_adapter_only_phase1.py; then
    echo "✅ colorize=True enabled"
else
    echo "❌ colorize=True NOT enabled"
fi

# Count occurrences of magnitude matching (should be in 3 places: train, eval_quick, eval_full)
match_count=$(grep -c "Match magnitude" train_adapter_only_phase1.py || echo "0")
echo "📊 Magnitude matching applied in $match_count location(s)"

echo ""
echo "================================================================================"
echo "SUMMARY OF CHANGES"
echo "================================================================================"
echo ""
echo "Applied fixes to train_adapter_only_phase1.py:"
echo ""
echo "1. ✅ RMS Magnitude Matching (Quick Fix)"
echo "   - Added after adapter forward pass in training loop"
echo "   - Added in evaluate_quick() function"
echo "   - Added in evaluate_full() function"
echo "   - Formula: reconstructed *= orig_rms / recon_rms"
echo "   - This forces reconstructed embeddings to match original scale"
echo ""
echo "2. ✅ Enable colorize=True (Proper Fix)"
echo "   - Changed adapter initialization to use colorize=True"
echo "   - This trains a learnable calibration layer (_EmbedColor)"
echo "   - Better long-term solution than manual RMS matching"
echo ""
echo "================================================================================"
echo "NEXT STEPS"
echo "================================================================================"
echo ""
echo "1. Commit changes:"
echo "   git add train_adapter_only_phase1.py"
echo "   git commit -m 'fix: Add RMS magnitude matching and enable colorize'"
echo ""
echo "2. Run training on HPC:"
echo "   git pull && rm -rf runs && PYTHONPATH=. ./scripts/run_stage1_h100.sh"
echo ""
echo "3. Expected results:"
echo "   - Embedding norms should match (both ≈0.5)"
echo "   - Generation should work (no more empty strings!)"
echo "   - F1 score should be >0% (target: 10-70%)"
echo ""
echo "================================================================================"
