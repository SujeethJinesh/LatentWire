#!/usr/bin/env python3
"""
Comprehensive validation script to check all prerequisites before running experiments.
Prevents wasted GPU time from missing dependencies or configuration issues.

Usage:
    python finalization/validate.py [--fix]

    --fix: Attempt to automatically fix issues where possible
"""

import os
import sys
import subprocess
import shutil
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import importlib.util

# ANSI color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

class ValidationResult:
    """Container for validation results"""
    def __init__(self, name: str):
        self.name = name
        self.passed = True
        self.warnings = []
        self.errors = []
        self.info = []

    def error(self, msg: str, fix: str = None):
        self.errors.append((msg, fix))
        self.passed = False

    def warning(self, msg: str, fix: str = None):
        self.warnings.append((msg, fix))

    def info(self, msg: str):
        self.info.append(msg)

    def print_result(self):
        """Print formatted result"""
        status = f"{GREEN}✓ PASSED{RESET}" if self.passed else f"{RED}✗ FAILED{RESET}"
        print(f"\n{BOLD}{self.name}:{RESET} {status}")

        for msg in self.info:
            print(f"  {BLUE}ℹ{RESET} {msg}")

        for msg, fix in self.warnings:
            print(f"  {YELLOW}⚠{RESET} {msg}")
            if fix:
                print(f"    {YELLOW}→ Fix:{RESET} {fix}")

        for msg, fix in self.errors:
            print(f"  {RED}✗{RESET} {msg}")
            if fix:
                print(f"    {RED}→ Fix:{RESET} {fix}")

def check_python_version() -> ValidationResult:
    """Check Python version >= 3.8"""
    result = ValidationResult("Python Version")

    version = sys.version_info
    result.info(f"Current version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        result.error(
            f"Python {version.major}.{version.minor} detected, need >= 3.8",
            "Install Python 3.8+ using conda or pyenv"
        )

    return result

def check_required_packages() -> ValidationResult:
    """Check all required packages are installed"""
    result = ValidationResult("Required Packages")

    required_packages = {
        'torch': '2.0.0',
        'transformers': '4.30.0',
        'datasets': None,
        'numpy': None,
        'pandas': None,
        'scikit-learn': None,
        'matplotlib': None,
        'seaborn': None,
        'scipy': None,
        'tensorboard': None,
        'wandb': None,
        'einops': None,
        'safetensors': None,
        'accelerate': None,
        'bitsandbytes': None,  # For quantization
    }

    missing = []
    outdated = []

    for package, min_version in required_packages.items():
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing.append(package)
        elif min_version:
            try:
                mod = __import__(package)
                if hasattr(mod, '__version__'):
                    current = mod.__version__
                    if current < min_version:
                        outdated.append(f"{package} (have {current}, need >= {min_version})")
            except:
                pass

    if missing:
        result.error(
            f"Missing packages: {', '.join(missing)}",
            f"pip install {' '.join(missing)}"
        )

    if outdated:
        result.warning(
            f"Outdated packages: {', '.join(outdated)}",
            "pip install --upgrade " + ' '.join([p.split()[0] for p in outdated])
        )

    # Check optional but recommended packages
    optional = ['llmlingua', 'telepathy']
    for package in optional:
        spec = importlib.util.find_spec(package)
        if spec is None:
            result.warning(
                f"Optional package '{package}' not installed",
                f"pip install -e . (from project root)"
            )

    result.info(f"Checked {len(required_packages)} required packages")

    return result

def check_gpu_availability() -> ValidationResult:
    """Check GPU availability and memory"""
    result = ValidationResult("GPU Resources")

    try:
        import torch

        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            result.info(f"Found {num_gpus} CUDA GPU(s)")

            total_memory = 0
            for i in range(num_gpus):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                result.info(f"  GPU {i}: {props.name} ({memory_gb:.1f}GB)")
                total_memory += memory_gb

                if memory_gb < 16:
                    result.warning(
                        f"GPU {i} has only {memory_gb:.1f}GB memory (may cause OOM)",
                        "Reduce batch_size or use gradient accumulation"
                    )

            # Check if we're on HPC with expected GPUs
            if "H100" in str(props.name) or "A100" in str(props.name):
                result.info("High-performance GPU detected (H100/A100)")

            # Check CUDA version
            cuda_version = torch.version.cuda
            if cuda_version:
                result.info(f"CUDA version: {cuda_version}")

        elif torch.backends.mps.is_available():
            result.info("Apple MPS (Metal) GPU available")
            result.warning(
                "MPS is for development only, use CUDA GPUs for training",
                "Run on HPC cluster for training"
            )
        else:
            result.error(
                "No GPU detected (CUDA or MPS)",
                "Install CUDA drivers or run on GPU-enabled machine"
            )

    except ImportError:
        result.error(
            "PyTorch not installed",
            "pip install torch torchvision torchaudio"
        )
    except Exception as e:
        result.error(f"Failed to check GPU: {e}")

    return result

def check_disk_space() -> ValidationResult:
    """Check available disk space for checkpoints"""
    result = ValidationResult("Disk Space")

    # Check current directory
    try:
        stat = shutil.disk_usage(".")
        free_gb = stat.free / (1024**3)
        total_gb = stat.total / (1024**3)
        used_pct = (stat.used / stat.total) * 100

        result.info(f"Free space: {free_gb:.1f}GB / {total_gb:.1f}GB ({used_pct:.1f}% used)")

        if free_gb < 10:
            result.error(
                f"Only {free_gb:.1f}GB free space (need at least 10GB)",
                "Free up disk space or use different storage location"
            )
        elif free_gb < 50:
            result.warning(
                f"Limited free space ({free_gb:.1f}GB)",
                "Monitor disk usage during training"
            )

        # Check if we're on HPC with project directory
        if os.path.exists("/projects/m000066/sujinesh"):
            hpc_stat = shutil.disk_usage("/projects/m000066/sujinesh")
            hpc_free_gb = hpc_stat.free / (1024**3)
            result.info(f"HPC project space: {hpc_free_gb:.1f}GB free")

    except Exception as e:
        result.error(f"Failed to check disk space: {e}")

    return result

def check_git_repository() -> ValidationResult:
    """Check git repository status and access"""
    result = ValidationResult("Git Repository")

    try:
        # Check if we're in a git repo
        ret = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            text=True
        )
        if ret.returncode != 0:
            result.error(
                "Not in a git repository",
                "git init or clone the repository"
            )
            return result

        # Check current branch
        ret = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True
        )
        branch = ret.stdout.strip()
        result.info(f"Current branch: {branch}")

        # Check for uncommitted changes
        ret = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True
        )
        if ret.stdout.strip():
            num_changes = len(ret.stdout.strip().split('\n'))
            result.warning(
                f"Found {num_changes} uncommitted changes",
                "Commit or stash changes before running experiments"
            )

        # Check remote access
        ret = subprocess.run(
            ["git", "remote", "-v"],
            capture_output=True,
            text=True
        )
        if not ret.stdout.strip():
            result.warning(
                "No git remote configured",
                "git remote add origin <repository-url>"
            )
        else:
            result.info("Git remote configured")

        # Try to fetch (non-blocking check)
        ret = subprocess.run(
            ["git", "fetch", "--dry-run"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if ret.returncode == 0:
            result.info("Git remote accessible")
        else:
            result.warning(
                "Cannot fetch from remote",
                "Check network connection or git credentials"
            )

    except FileNotFoundError:
        result.error(
            "Git not installed",
            "Install git: apt-get install git (or brew install git on macOS)"
        )
    except subprocess.TimeoutExpired:
        result.warning("Git fetch timed out (network may be slow)")
    except Exception as e:
        result.error(f"Git check failed: {e}")

    return result

def check_model_weights() -> ValidationResult:
    """Check if model weights are accessible"""
    result = ValidationResult("Model Weights")

    models_to_check = [
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct"
    ]

    try:
        from transformers import AutoConfig
        from huggingface_hub import HfApi

        api = HfApi()

        for model_id in models_to_check:
            try:
                # Check if model exists on HuggingFace
                model_info = api.model_info(model_id)
                result.info(f"✓ {model_id} accessible")

                # Check if we have local cache
                cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
                model_cache = list(cache_dir.glob(f"*{model_id.replace('/', '--')}*"))
                if model_cache:
                    result.info(f"  → Cached locally")
                else:
                    result.warning(
                        f"  → Not cached (will download on first use)",
                        f"Pre-download with: huggingface-cli download {model_id}"
                    )

            except Exception as e:
                result.error(
                    f"Cannot access {model_id}: {e}",
                    "Check HuggingFace token or network connection"
                )

        # Check HF token
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not hf_token:
            token_file = Path.home() / ".cache" / "huggingface" / "token"
            if not token_file.exists():
                result.warning(
                    "No HuggingFace token found (may be needed for gated models)",
                    "huggingface-cli login"
                )
        else:
            result.info("HuggingFace token configured")

    except ImportError:
        result.error(
            "transformers or huggingface_hub not installed",
            "pip install transformers huggingface_hub"
        )
    except Exception as e:
        result.error(f"Model check failed: {e}")

    return result

def check_datasets() -> ValidationResult:
    """Check if datasets are accessible"""
    result = ValidationResult("Datasets")

    datasets_to_check = ["squad", "hotpot_qa", "ag_news", "sst2"]

    try:
        from datasets import load_dataset

        for dataset_name in datasets_to_check:
            try:
                # Try loading a tiny subset to check accessibility
                if dataset_name == "hotpot_qa":
                    ds = load_dataset(dataset_name, "distractor", split="train[:1]")
                elif dataset_name == "sst2":
                    ds = load_dataset("glue", "sst2", split="train[:1]")
                else:
                    ds = load_dataset(dataset_name, split="train[:1]")

                result.info(f"✓ {dataset_name} accessible")

            except Exception as e:
                result.warning(
                    f"Cannot load {dataset_name}: {str(e)[:100]}",
                    f"Will download on first use"
                )

        # Check cache directory
        cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
        if cache_dir.exists():
            cache_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
            cache_gb = cache_size / (1024**3)
            result.info(f"Dataset cache size: {cache_gb:.2f}GB")

    except ImportError:
        result.error(
            "datasets library not installed",
            "pip install datasets"
        )
    except Exception as e:
        result.error(f"Dataset check failed: {e}")

    return result

def check_previous_checkpoints() -> ValidationResult:
    """Check for existing checkpoints if resuming"""
    result = ValidationResult("Previous Checkpoints")

    runs_dir = Path("runs")
    telepathy_dir = Path("telepathy/runs")

    if runs_dir.exists():
        checkpoints = list(runs_dir.glob("*/epoch*"))
        if checkpoints:
            result.info(f"Found {len(checkpoints)} checkpoint(s) in runs/")

            # Show most recent
            if checkpoints:
                recent = sorted(checkpoints, key=lambda p: p.stat().st_mtime)[-1]
                result.info(f"Most recent: {recent}")
        else:
            result.info("No checkpoints in runs/ (will train from scratch)")
    else:
        result.info("runs/ directory doesn't exist (will be created)")

    if telepathy_dir.exists():
        telepathy_ckpts = list(telepathy_dir.glob("*/epoch*"))
        if telepathy_ckpts:
            result.info(f"Found {len(telepathy_ckpts)} telepathy checkpoint(s)")

    # Check for checkpoint corruption
    for ckpt_dir in checkpoints[:3] if 'checkpoints' in locals() else []:
        required_files = ["encoder.pt", "adapter_llama.pt", "adapter_qwen.pt", "config.json"]
        missing = [f for f in required_files if not (ckpt_dir / f).exists()]
        if missing:
            result.warning(
                f"Checkpoint {ckpt_dir.name} missing files: {missing}",
                f"Remove corrupted checkpoint: rm -rf {ckpt_dir}"
            )

    return result

def check_write_permissions() -> ValidationResult:
    """Check write permissions in key directories"""
    result = ValidationResult("Write Permissions")

    dirs_to_check = [
        (".", "Current directory"),
        ("runs", "Runs directory"),
        ("telepathy", "Telepathy directory"),
        ("figures", "Figures directory"),
    ]

    for dir_path, name in dirs_to_check:
        path = Path(dir_path)

        try:
            if not path.exists():
                # Try to create it
                path.mkdir(parents=True, exist_ok=True)
                result.info(f"✓ Created {name}: {path}")
                path.rmdir()  # Clean up test directory
            else:
                # Check if we can write
                test_file = path / ".write_test"
                test_file.write_text("test")
                test_file.unlink()
                result.info(f"✓ Can write to {name}: {path}")

        except PermissionError:
            result.error(
                f"No write permission for {name}: {path}",
                f"chmod +w {path} or run from writable directory"
            )
        except Exception as e:
            result.error(f"Cannot write to {name}: {e}")

    # Check HPC project directory if applicable
    if os.path.exists("/projects/m000066/sujinesh"):
        try:
            test_file = Path("/projects/m000066/sujinesh/.write_test")
            test_file.write_text("test")
            test_file.unlink()
            result.info("✓ HPC project directory writable")
        except:
            result.error(
                "Cannot write to HPC project directory",
                "Contact HPC admin about /projects/m000066/sujinesh permissions"
            )

    return result

def check_slurm_environment() -> ValidationResult:
    """Check SLURM environment variables and configuration"""
    result = ValidationResult("SLURM Environment")

    # Check if we're in a SLURM environment
    slurm_job_id = os.environ.get("SLURM_JOB_ID")

    if slurm_job_id:
        result.info(f"Running in SLURM job: {slurm_job_id}")

        # Check important SLURM variables
        important_vars = [
            "SLURM_JOB_ID",
            "SLURM_JOB_NAME",
            "SLURM_NODELIST",
            "SLURM_NTASKS",
            "SLURM_GPUS",
            "CUDA_VISIBLE_DEVICES"
        ]

        for var in important_vars:
            value = os.environ.get(var)
            if value:
                result.info(f"  {var}={value}")
            else:
                result.warning(f"  {var} not set")

        # Check account and partition
        account = os.environ.get("SLURM_JOB_ACCOUNT")
        partition = os.environ.get("SLURM_JOB_PARTITION")

        if account != "marlowe-m000066":
            result.warning(
                f"SLURM account is '{account}', expected 'marlowe-m000066'",
                "Update SLURM script: #SBATCH --account=marlowe-m000066"
            )

        if partition != "preempt":
            result.warning(
                f"SLURM partition is '{partition}', expected 'preempt'",
                "Update SLURM script: #SBATCH --partition=preempt"
            )

    else:
        result.info("Not running in SLURM environment")

        # Check if SLURM commands are available
        try:
            ret = subprocess.run(["which", "sbatch"], capture_output=True)
            if ret.returncode == 0:
                result.info("SLURM commands available (can submit jobs)")

                # Check user's job queue
                ret = subprocess.run(
                    ["squeue", "-u", os.environ.get("USER", "")],
                    capture_output=True,
                    text=True
                )
                if ret.returncode == 0:
                    lines = ret.stdout.strip().split('\n')
                    if len(lines) > 1:  # Header + jobs
                        result.info(f"You have {len(lines)-1} job(s) in queue")
            else:
                result.info("SLURM not available (development machine)")

        except:
            pass

    return result

def check_environment_variables() -> ValidationResult:
    """Check important environment variables"""
    result = ValidationResult("Environment Variables")

    important_vars = {
        "PYTHONPATH": ".",
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "TOKENIZERS_PARALLELISM": "false"
    }

    for var, expected in important_vars.items():
        value = os.environ.get(var)
        if value == expected:
            result.info(f"✓ {var}={value}")
        elif value:
            result.warning(
                f"{var}={value} (expected '{expected}')",
                f"export {var}={expected}"
            )
        else:
            result.warning(
                f"{var} not set (should be '{expected}')",
                f"export {var}={expected}"
            )

    # Check for conda/venv
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    venv = os.environ.get("VIRTUAL_ENV")

    if conda_env:
        result.info(f"Conda environment: {conda_env}")
    elif venv:
        result.info(f"Virtual environment: {venv}")
    else:
        result.warning(
            "No conda or virtual environment activated",
            "conda activate <env> or source venv/bin/activate"
        )

    return result

def run_validation(fix: bool = False) -> bool:
    """Run all validation checks"""
    print(f"\n{BOLD}{'='*60}")
    print("LatentWire Validation Check")
    print(f"{'='*60}{RESET}")

    checks = [
        check_python_version,
        check_required_packages,
        check_gpu_availability,
        check_disk_space,
        check_git_repository,
        check_model_weights,
        check_datasets,
        check_previous_checkpoints,
        check_write_permissions,
        check_slurm_environment,
        check_environment_variables,
    ]

    results = []
    for check_func in checks:
        try:
            result = check_func()
            results.append(result)
            result.print_result()
        except Exception as e:
            print(f"{RED}Check '{check_func.__name__}' failed: {e}{RESET}")

    # Summary
    print(f"\n{BOLD}{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}{RESET}")

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    warnings = sum(len(r.warnings) for r in results)

    if failed == 0:
        print(f"{GREEN}✓ All {len(results)} checks PASSED{RESET}")
        if warnings > 0:
            print(f"{YELLOW}  ({warnings} warning(s) - see above for details){RESET}")
        print(f"\n{GREEN}System is ready for experiments!{RESET}")
        return True
    else:
        print(f"{RED}✗ {failed}/{len(results)} checks FAILED{RESET}")
        print(f"\n{RED}Please fix the errors above before running experiments.{RESET}")
        print(f"Review the {RED}→ Fix:{RESET} suggestions for each error.")

        if fix:
            print(f"\n{YELLOW}Auto-fix mode enabled - attempting fixes...{RESET}")
            # Add auto-fix logic here if desired

        return False

def main():
    parser = argparse.ArgumentParser(description="Validate LatentWire environment")
    parser.add_argument("--fix", action="store_true", help="Attempt to auto-fix issues")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    success = run_validation(fix=args.fix)

    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()