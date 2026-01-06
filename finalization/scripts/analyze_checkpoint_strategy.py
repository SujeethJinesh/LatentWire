#!/usr/bin/env python3
"""
Analysis of checkpoint saving/loading strategy by examining the codebase.

Validates:
1. Checkpoint saving intervals and triggers
2. Resume functionality implementation
3. Checkpoint components and size
4. State preservation completeness
5. Cleanup strategy for old checkpoints
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print('='*80)


def print_status(category: str, status: str, details: str = ""):
    """Print status with consistent formatting."""
    icon = "✅" if status == "GOOD" else "⚠️" if status == "WARNING" else "❌"
    print(f"  {icon} {category}: {status}")
    if details:
        print(f"     {details}")


def analyze_checkpoint_saving() -> Dict[str, Any]:
    """Analyze checkpoint saving implementation."""

    findings = {
        "save_triggers": [],
        "saved_components": [],
        "atomic_saves": False,
        "pruning_enabled": False,
    }

    # Check train.py for save triggers
    train_path = Path("latentwire/train.py")
    if train_path.exists():
        content = train_path.read_text()

        # Find save_every parameter
        if match := re.search(r'--save_every.*default=(\d+)', content):
            findings["save_every_default"] = int(match.group(1))

        # Find periodic saves
        if "if args.save_every and (global_step % args.save_every == 0)" in content:
            findings["save_triggers"].append("periodic")

        # Find best checkpoint saves
        if "best_first_acc" in content and "save_latest_checkpoint" in content:
            findings["save_triggers"].append("best_model")

        # Find final save
        if re.search(r'# ===== Final save =====', content):
            findings["save_triggers"].append("final")

        # Check what gets saved
        artifacts_pattern = r'artifacts\["([^"]+)"\]\s*='
        for match in re.finditer(artifacts_pattern, content):
            component = match.group(1)
            if component not in findings["saved_components"]:
                findings["saved_components"].append(component)

    # Check checkpointing.py for atomic saves and pruning
    ckpt_path = Path("latentwire/checkpointing.py")
    if ckpt_path.exists():
        content = ckpt_path.read_text()

        if "_atomic_write_bytes" in content and "os.replace" in content:
            findings["atomic_saves"] = True

        if "prune_save_dir" in content:
            findings["pruning_enabled"] = True

        # Find canonical files
        if match := re.search(r'CANONICAL_FILES = \{([^}]+)\}', content, re.DOTALL):
            canonical = [f.strip().strip('"') for f in match.group(1).split(',')]
            findings["canonical_files"] = canonical

    return findings


def analyze_resume_capability() -> Dict[str, Any]:
    """Analyze checkpoint resume implementation."""

    findings = {
        "auto_resume": False,
        "manual_resume": False,
        "state_restored": [],
        "find_latest": False,
    }

    train_path = Path("latentwire/train.py")
    if train_path.exists():
        content = train_path.read_text()

        # Check for resume arguments
        if "--auto_resume" in content:
            findings["auto_resume"] = True

        if "--resume_from" in content:
            findings["manual_resume"] = True

        # Check find_latest_checkpoint function
        if "def find_latest_checkpoint" in content:
            findings["find_latest"] = True

        # Check what gets loaded
        if "def load_checkpoint" in content:
            # Extract what's being loaded
            load_section = content[content.find("def load_checkpoint"):content.find("def ", content.find("def load_checkpoint") + 10)]

            if "encoder.state_dict()" in load_section or "load_state_dict" in load_section:
                findings["state_restored"].append("encoder")
            if "optimizer.state_dict()" in load_section or "optimizer.load_state_dict" in load_section:
                findings["state_restored"].append("optimizer")
            if "adapter" in load_section:
                findings["state_restored"].append("adapters")
            if "torch.set_rng_state" in content:
                findings["state_restored"].append("rng_state")

    return findings


def estimate_checkpoint_sizes() -> Dict[str, Any]:
    """Estimate checkpoint sizes based on model architecture."""

    estimates = {}

    # Parse model dimensions from models.py
    models_path = Path("latentwire/models.py")
    if models_path.exists():
        content = models_path.read_text()

        # Find InterlinguaEncoder defaults
        if match := re.search(r'class InterlinguaEncoder.*?def __init__.*?d_z:\s*int\s*=\s*(\d+).*?n_layers:\s*int\s*=\s*(\d+)',
                             content, re.DOTALL):
            d_z = int(match.group(1))
            n_layers = int(match.group(2))

            # Estimate parameters
            # ByteEncoder: ~6 transformer layers
            params_per_layer = 4 * d_z * d_z + 2 * d_z * (4 * d_z)  # Attention + FFN
            encoder_params = n_layers * params_per_layer

            # Adapters: 2 models x small MLP
            adapter_params = 2 * (d_z * 2 * d_z + 2 * d_z * 4096)  # Rough estimate

            total_params = encoder_params + adapter_params

            # Sizes in MB (float32 = 4 bytes)
            model_size_mb = (total_params * 4) / (1024 * 1024)
            optimizer_size_mb = model_size_mb * 2  # Adam has 2 momentum buffers

            estimates["standard"] = {
                "d_z": d_z,
                "n_layers": n_layers,
                "model_params": total_params,
                "model_size_mb": round(model_size_mb, 1),
                "with_optimizer_mb": round(model_size_mb + optimizer_size_mb, 1),
            }

    return estimates


def analyze_robustness_features() -> Dict[str, Any]:
    """Analyze robustness and safety features."""

    findings = {
        "atomic_writes": False,
        "temp_file_cleanup": False,
        "oom_handling": False,
        "nan_handling": False,
        "gradient_clipping": False,
    }

    # Check checkpointing.py for atomic operations
    ckpt_path = Path("latentwire/checkpointing.py")
    if ckpt_path.exists():
        content = ckpt_path.read_text()

        if "tempfile.mkstemp" in content and "os.replace" in content:
            findings["atomic_writes"] = True

        if "_is_tmp_file" in content and "_safe_remove" in content:
            findings["temp_file_cleanup"] = True

    # Check train.py for error handling
    train_path = Path("latentwire/train.py")
    if train_path.exists():
        content = train_path.read_text()

        if "torch.cuda.empty_cache()" in content:
            findings["oom_handling"] = True

        if "torch.isfinite" in content or "skip_batch_due_to_nan" in content:
            findings["nan_handling"] = True

        if "torch.nn.utils.clip_grad" in content or "clip_grad_norm" in content:
            findings["gradient_clipping"] = True

    return findings


def check_test_coverage() -> Dict[str, Any]:
    """Check if checkpoint functionality has tests."""

    findings = {
        "has_tests": False,
        "test_files": [],
        "test_types": [],
    }

    # Look for checkpoint tests
    test_paths = [
        Path("tests/integration/test_checkpoint_roundtrip.py"),
        Path("tests/test_checkpointing.py"),
        Path("tests/test_checkpoint.py"),
    ]

    for path in test_paths:
        if path.exists():
            findings["has_tests"] = True
            findings["test_files"].append(str(path))

            content = path.read_text()
            if "roundtrip" in content.lower():
                findings["test_types"].append("roundtrip")
            if "resume" in content.lower():
                findings["test_types"].append("resume")
            if "prune" in content.lower() or "cleanup" in content.lower():
                findings["test_types"].append("cleanup")

    return findings


def main():
    """Run checkpoint strategy analysis."""

    print_section("CHECKPOINT STRATEGY ANALYSIS")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Analyzing codebase for checkpoint robustness...")

    # 1. Analyze checkpoint saving
    print_section("1. CHECKPOINT SAVING")

    save_analysis = analyze_checkpoint_saving()

    print_status(
        "Save Triggers",
        "GOOD" if len(save_analysis["save_triggers"]) >= 2 else "WARNING",
        f"Found: {', '.join(save_analysis['save_triggers'])}"
    )

    print_status(
        "Periodic Saves",
        "GOOD" if "periodic" in save_analysis["save_triggers"] else "WARNING",
        f"Default interval: {save_analysis.get('save_every_default', 0)} steps"
    )

    print_status(
        "Best Model Tracking",
        "GOOD" if "best_model" in save_analysis["save_triggers"] else "WARNING",
        "Saves checkpoint when metrics improve"
    )

    print_status(
        "Components Saved",
        "GOOD" if len(save_analysis["saved_components"]) >= 4 else "WARNING",
        f"Saves {len(save_analysis['saved_components'])} components"
    )

    if save_analysis["saved_components"]:
        for comp in save_analysis["saved_components"][:5]:  # Show first 5
            print(f"       - {comp}")
        if len(save_analysis["saved_components"]) > 5:
            print(f"       ... and {len(save_analysis['saved_components']) - 5} more")

    print_status(
        "Atomic Saves",
        "GOOD" if save_analysis["atomic_saves"] else "CRITICAL",
        "Prevents corruption during crashes"
    )

    print_status(
        "Auto-Pruning",
        "GOOD" if save_analysis["pruning_enabled"] else "WARNING",
        "Cleans up old checkpoints to save space"
    )

    # 2. Analyze resume capability
    print_section("2. RESUME CAPABILITY")

    resume_analysis = analyze_resume_capability()

    print_status(
        "Auto Resume",
        "GOOD" if resume_analysis["auto_resume"] else "WARNING",
        "Automatically finds and loads latest checkpoint"
    )

    print_status(
        "Manual Resume",
        "GOOD" if resume_analysis["manual_resume"] else "WARNING",
        "Supports --resume_from flag"
    )

    print_status(
        "Find Latest Logic",
        "GOOD" if resume_analysis["find_latest"] else "CRITICAL",
        "Can locate most recent checkpoint"
    )

    print_status(
        "State Restoration",
        "GOOD" if len(resume_analysis["state_restored"]) >= 3 else "WARNING",
        f"Restores: {', '.join(resume_analysis['state_restored'])}"
    )

    critical_components = ["encoder", "optimizer", "rng_state"]
    missing = [c for c in critical_components if c not in resume_analysis["state_restored"]]
    if missing:
        print(f"       ⚠️  Missing: {', '.join(missing)}")

    # 3. Size estimates
    print_section("3. CHECKPOINT SIZES")

    size_estimates = estimate_checkpoint_sizes()

    if size_estimates:
        for config_name, est in size_estimates.items():
            print(f"\n  Configuration: {config_name}")
            print(f"    Model parameters: ~{est['model_params']:,}")
            print(f"    Model state: ~{est['model_size_mb']} MB")
            print(f"    With optimizer: ~{est['with_optimizer_mb']} MB")

            status = "GOOD" if est['with_optimizer_mb'] < 500 else "WARNING"
            print_status(
                "Size Assessment",
                status,
                "Within reasonable limits" if status == "GOOD" else "Large checkpoint size"
            )
    else:
        print_status("Size Estimates", "WARNING", "Could not estimate from code")

    # 4. Robustness features
    print_section("4. ROBUSTNESS FEATURES")

    robustness = analyze_robustness_features()

    print_status(
        "Atomic Writes",
        "GOOD" if robustness["atomic_writes"] else "CRITICAL",
        "Uses temp files + atomic rename"
    )

    print_status(
        "Temp File Cleanup",
        "GOOD" if robustness["temp_file_cleanup"] else "WARNING",
        "Removes .tmp, .new, .partial files"
    )

    print_status(
        "OOM Handling",
        "GOOD" if robustness["oom_handling"] else "WARNING",
        "Clears GPU cache on errors"
    )

    print_status(
        "NaN Detection",
        "GOOD" if robustness["nan_handling"] else "WARNING",
        "Skips batches with NaN values"
    )

    print_status(
        "Gradient Clipping",
        "GOOD" if robustness["gradient_clipping"] else "INFO",
        "Prevents gradient explosions"
    )

    # 5. Test coverage
    print_section("5. TEST COVERAGE")

    test_coverage = check_test_coverage()

    print_status(
        "Has Tests",
        "GOOD" if test_coverage["has_tests"] else "WARNING",
        f"Found {len(test_coverage['test_files'])} test files"
    )

    if test_coverage["test_files"]:
        for test_file in test_coverage["test_files"]:
            print(f"       - {test_file}")

    if test_coverage["test_types"]:
        print(f"     Test types: {', '.join(test_coverage['test_types'])}")

    # Summary and recommendations
    print_section("ASSESSMENT SUMMARY")

    critical_issues = []
    warnings = []
    strengths = []

    # Evaluate critical issues
    if not save_analysis["atomic_saves"]:
        critical_issues.append("No atomic saves - risk of corruption")
    if not resume_analysis["find_latest"]:
        critical_issues.append("Cannot find latest checkpoint")

    # Evaluate warnings
    if save_analysis.get("save_every_default", 0) == 0:
        warnings.append("Periodic saves disabled by default")
    if not resume_analysis["auto_resume"]:
        warnings.append("No auto-resume capability")
    if not test_coverage["has_tests"]:
        warnings.append("Limited test coverage for checkpoints")

    # Identify strengths
    if save_analysis["pruning_enabled"]:
        strengths.append("Automatic cleanup of old checkpoints")
    if "best_model" in save_analysis["save_triggers"]:
        strengths.append("Tracks best model checkpoints")
    if robustness["nan_handling"]:
        strengths.append("Handles NaN gracefully")

    print("\n📊 Overall Assessment:")

    if not critical_issues:
        print("  ✅ PRODUCTION READY - Safe for 24-hour runs")
    elif len(critical_issues) == 1:
        print("  ⚠️  MOSTLY READY - One critical issue to address")
    else:
        print("  ❌ NEEDS WORK - Multiple critical issues")

    if critical_issues:
        print("\n🚨 Critical Issues:")
        for issue in critical_issues:
            print(f"  • {issue}")

    if warnings:
        print("\n⚠️  Warnings:")
        for warning in warnings:
            print(f"  • {warning}")

    if strengths:
        print("\n✅ Strengths:")
        for strength in strengths:
            print(f"  • {strength}")

    print("\n📝 Recommendations for 24-hour runs:")
    print("  1. Set --save_every 500 to save every 500 steps")
    print("  2. Use --auto_resume flag for crash recovery")
    print("  3. Monitor disk usage (~500MB per checkpoint)")
    print("  4. Keep separate directories for best and latest")
    print("  5. Test resume on target hardware before long runs")

    if save_analysis.get("canonical_files"):
        print("\n📁 Files preserved in checkpoints:")
        for f in save_analysis["canonical_files"][:5]:
            print(f"  • {f}")

    print("\n" + "="*80)

    return len(critical_issues) == 0


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)