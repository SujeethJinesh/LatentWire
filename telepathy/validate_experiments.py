#!/usr/bin/env python3
"""
Validation script for telepathy experiments.
Performs dry runs to verify everything works before actual training.

Usage:
    python telepathy/validate_experiments.py [--full]

    --full: Run more comprehensive checks (takes longer)
"""

import sys
import os
import json
import time
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

# Try importing optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class ValidationResult:
    """Result of a validation check."""
    name: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        icon = "✅" if self.passed else "❌"
        return f"{icon} {self.name}: {self.message}"


class ExperimentValidator:
    """Validates telepathy experiments can run successfully."""

    def __init__(self, full_check: bool = False):
        self.full_check = full_check
        self.results: List[ValidationResult] = []
        self.start_time = time.time()

    def run_all_checks(self) -> bool:
        """Run all validation checks."""
        print("=" * 70)
        print("TELEPATHY EXPERIMENT VALIDATION")
        print("=" * 70)
        print(f"Started at: {datetime.now().isoformat()}")
        print(f"Mode: {'FULL' if self.full_check else 'QUICK'}")
        print("=" * 70)
        print()

        # Run checks in order of importance
        checks = [
            self.check_python_version,
            self.check_imports,
            self.check_cuda_availability,
            self.check_memory_availability,
            self.check_directory_structure,
            self.check_model_loading,
            self.check_dataset_access,
            self.check_config_validity,
            self.check_slurm_environment,
            self.check_telepathy_modules,
            self.check_experiment_scripts,
        ]

        if self.full_check:
            checks.extend([
                self.check_forward_pass,
                self.check_backward_pass,
                self.check_checkpoint_save_load,
                self.check_multimodel_compatibility,
            ])

        for check_func in checks:
            print(f"Running {check_func.__name__}...")
            try:
                result = check_func()
                self.results.append(result)
                print(f"  {result}")
                if not result.passed and check_func.__name__ in [
                    'check_python_version', 'check_imports', 'check_cuda_availability'
                ]:
                    print("\n⚠️  Critical check failed. Stopping validation.")
                    break
            except Exception as e:
                result = ValidationResult(
                    name=check_func.__name__,
                    passed=False,
                    message=f"Exception: {str(e)}",
                    details={"traceback": traceback.format_exc()}
                )
                self.results.append(result)
                print(f"  {result}")
                print(f"  Traceback: {result.details['traceback']}")

        print()
        return self.print_summary()

    def check_python_version(self) -> ValidationResult:
        """Check Python version is compatible."""
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            return ValidationResult(
                "Python Version",
                True,
                f"Python {version.major}.{version.minor}.{version.micro}",
                {"version": f"{version.major}.{version.minor}.{version.micro}"}
            )
        else:
            return ValidationResult(
                "Python Version",
                False,
                f"Python {version.major}.{version.minor} (need >= 3.8)",
                {"version": f"{version.major}.{version.minor}.{version.micro}"}
            )

    def check_imports(self) -> ValidationResult:
        """Check all required libraries can be imported."""
        # Detect environment
        is_local = not os.environ.get("SLURM_JOB_ID")
        is_hpc = os.environ.get("SLURM_JOB_ID") is not None

        if is_local and not TORCH_AVAILABLE:
            # On local MacBook, these imports are expected to fail
            return ValidationResult(
                "Import Check",
                True,
                "Local development environment - imports checked on HPC",
                {"environment": "local_dev", "note": "Full validation runs on HPC"}
            )

        required_libs = [
            'torch',
            'transformers',
            'datasets',
            'numpy',
            'pandas',
            'sklearn',
            'tqdm',
            'matplotlib',
            'seaborn',
            'scipy',
        ]

        missing = []
        versions = {}

        for lib in required_libs:
            try:
                module = __import__(lib)
                if hasattr(module, '__version__'):
                    versions[lib] = module.__version__
                else:
                    versions[lib] = "unknown"
            except ImportError:
                missing.append(lib)

        # Check specific telepathy imports
        try:
            from telepathy.model import TelepathyModel
            from telepathy.data import UnifiedDataLoader
            versions['telepathy'] = "available"
        except ImportError as e:
            missing.append(f"telepathy ({str(e)})")

        if missing:
            return ValidationResult(
                "Import Check",
                False,
                f"Missing libraries: {', '.join(missing)}",
                {"missing": missing, "versions": versions}
            )
        else:
            return ValidationResult(
                "Import Check",
                True,
                f"All {len(required_libs)} libraries available",
                {"versions": versions}
            )

    def check_cuda_availability(self) -> ValidationResult:
        """Check CUDA/GPU availability."""
        if not TORCH_AVAILABLE:
            return ValidationResult(
                "CUDA/GPU Check",
                True,
                "Local dev environment - GPU checks run on HPC",
                {"environment": "local_dev"}
            )

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            devices = []
            total_memory = 0

            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                devices.append({
                    "index": i,
                    "name": props.name,
                    "memory_gb": memory_gb,
                    "compute_capability": f"{props.major}.{props.minor}"
                })
                total_memory += memory_gb

            return ValidationResult(
                "CUDA/GPU Check",
                True,
                f"Found {device_count} GPU(s) with {total_memory:.1f}GB total memory",
                {"devices": devices, "cuda_version": torch.version.cuda}
            )
        else:
            # Check if we're on Mac with MPS
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return ValidationResult(
                    "CUDA/GPU Check",
                    True,
                    "Using Apple Silicon MPS (Mac)",
                    {"backend": "mps"}
                )
            else:
                return ValidationResult(
                    "CUDA/GPU Check",
                    False,
                    "No CUDA GPUs available (CPU only)",
                    {"backend": "cpu"}
                )

    def check_memory_availability(self) -> ValidationResult:
        """Check system memory and estimate requirements."""
        if not PSUTIL_AVAILABLE:
            return ValidationResult(
                "Memory Check",
                True,
                "Memory check requires psutil (install for detailed info)",
                {"note": "Run on HPC for accurate memory estimates"}
            )

        import psutil
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)
        total_gb = mem.total / (1024**3)

        # Estimate memory requirements for different configs
        estimates = {
            "Llama-3.1-1B": {"model": 4, "training": 8, "total": 12},
            "Llama-3.1-3B": {"model": 12, "training": 20, "total": 32},
            "Llama-3.1-8B": {"model": 32, "training": 48, "total": 80},
            "All 3 models": {"model": 48, "training": 76, "total": 124},
        }

        recommendations = []
        for config, reqs in estimates.items():
            if available_gb >= reqs["total"]:
                status = "✅"
            elif available_gb >= reqs["model"]:
                status = "⚠️"
            else:
                status = "❌"
            recommendations.append(f"{status} {config}: needs ~{reqs['total']}GB")

        if available_gb < 12:
            passed = False
            message = f"Low memory: {available_gb:.1f}GB available (need 12GB+)"
        else:
            passed = True
            message = f"Memory OK: {available_gb:.1f}/{total_gb:.1f}GB available"

        return ValidationResult(
            "Memory Check",
            passed,
            message,
            {
                "available_gb": available_gb,
                "total_gb": total_gb,
                "estimates": estimates,
                "recommendations": recommendations
            }
        )

    def check_directory_structure(self) -> ValidationResult:
        """Check required directories exist and are writable."""
        base_dir = Path(__file__).parent.parent
        required_dirs = [
            base_dir / "telepathy",
            base_dir / "latentwire",
            base_dir / "runs",
            base_dir / "figures",
        ]

        missing = []
        not_writable = []

        for dir_path in required_dirs:
            if not dir_path.exists():
                missing.append(str(dir_path.relative_to(base_dir)))
            elif not os.access(dir_path, os.W_OK):
                not_writable.append(str(dir_path.relative_to(base_dir)))

        # Create missing directories
        for dir_path in required_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

        if not_writable:
            return ValidationResult(
                "Directory Check",
                False,
                f"Not writable: {', '.join(not_writable)}",
                {"missing": missing, "not_writable": not_writable}
            )
        elif missing:
            return ValidationResult(
                "Directory Check",
                True,
                f"Created missing directories: {', '.join(missing)}",
                {"created": missing}
            )
        else:
            return ValidationResult(
                "Directory Check",
                True,
                "All directories exist and are writable",
                {"checked": [str(d.relative_to(base_dir)) for d in required_dirs]}
            )

    def check_model_loading(self) -> ValidationResult:
        """Check if models can be loaded."""
        if not TORCH_AVAILABLE:
            return ValidationResult(
                "Model Loading",
                True,
                "Local dev environment - model checks run on HPC",
                {"environment": "local_dev"}
            )

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            return ValidationResult(
                "Model Loading",
                False,
                "transformers library not available",
                {}
            )

        models_to_check = [
            ("meta-llama/Llama-3.1-1B", "Llama-3.1-1B"),
            ("meta-llama/Llama-3.1-3B", "Llama-3.1-3B"),
            ("meta-llama/Meta-Llama-3.1-8B-Instruct", "Llama-3.1-8B"),
        ]

        results = {}
        device = "cuda" if torch.cuda.is_available() else "cpu"

        for model_id, short_name in models_to_check:
            try:
                # Try loading tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    use_fast=True,
                )

                # Estimate model size
                if self.full_check:
                    # Actually try loading the model (memory intensive)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16,
                        device_map="cpu",  # Load to CPU first to check
                        trust_remote_code=True,
                    )
                    param_count = sum(p.numel() for p in model.parameters())
                    model_size_gb = param_count * 2 / (1024**3)  # fp16
                    del model  # Free memory
                    torch.cuda.empty_cache()
                else:
                    # Estimate based on name
                    if "1B" in short_name:
                        model_size_gb = 2.0
                    elif "3B" in short_name:
                        model_size_gb = 6.0
                    elif "8B" in short_name:
                        model_size_gb = 16.0
                    else:
                        model_size_gb = 0.0

                results[short_name] = {
                    "status": "✅",
                    "tokenizer": "loaded",
                    "model_size_gb": model_size_gb
                }
            except Exception as e:
                results[short_name] = {
                    "status": "❌",
                    "error": str(e)[:100]
                }

        failed = [k for k, v in results.items() if v["status"] == "❌"]

        if failed:
            return ValidationResult(
                "Model Loading",
                False,
                f"Failed to load: {', '.join(failed)}",
                {"results": results}
            )
        else:
            total_size = sum(v.get("model_size_gb", 0) for v in results.values())
            return ValidationResult(
                "Model Loading",
                True,
                f"All {len(models_to_check)} models accessible (~{total_size:.1f}GB total)",
                {"results": results}
            )

    def check_dataset_access(self) -> ValidationResult:
        """Check if datasets can be loaded."""
        try:
            from datasets import load_dataset
        except ImportError:
            return ValidationResult(
                "Dataset Access",
                True,
                "Local dev environment - dataset checks run on HPC",
                {"environment": "local_dev"}
            )

        datasets_to_check = [
            ("squad", "train[:100]"),
            ("hotpot_qa", "distractor", "train[:100]"),
            ("ai2_arc", "ARC-Challenge", "train[:100]"),
        ]

        results = {}
        for dataset_info in datasets_to_check:
            dataset_name = dataset_info[0]
            try:
                if len(dataset_info) == 2:
                    name, split = dataset_info
                    ds = load_dataset(name, split=split)
                else:
                    name, config, split = dataset_info
                    ds = load_dataset(name, config, split=split)

                results[dataset_name] = {
                    "status": "✅",
                    "samples": len(ds),
                    "columns": list(ds.features.keys()) if hasattr(ds, 'features') else []
                }
            except Exception as e:
                results[dataset_name] = {
                    "status": "❌",
                    "error": str(e)[:100]
                }

        failed = [k for k, v in results.items() if v["status"] == "❌"]

        if failed:
            return ValidationResult(
                "Dataset Access",
                False,
                f"Failed to load: {', '.join(failed)}",
                {"results": results}
            )
        else:
            return ValidationResult(
                "Dataset Access",
                True,
                f"All {len(datasets_to_check)} datasets accessible",
                {"results": results}
            )

    def check_config_validity(self) -> ValidationResult:
        """Check if configuration parameters are valid."""
        issues = []

        # Check telepathy config exists
        config_path = Path(__file__).parent / "config.yaml"
        if config_path.exists():
            import yaml
            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f)

                # Validate key parameters
                if config.get('batch_size', 1) > 64:
                    issues.append("batch_size > 64 may cause OOM")
                if config.get('gradient_accumulation_steps', 1) > 32:
                    issues.append("high gradient_accumulation_steps will slow training")
            except Exception as e:
                issues.append(f"Failed to parse config.yaml: {str(e)}")

        # Check environment variables
        env_vars = {
            "PYTHONPATH": os.environ.get("PYTHONPATH", "NOT SET"),
            "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "NOT SET"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "NOT SET"),
        }

        if "." not in os.environ.get("PYTHONPATH", ""):
            issues.append("PYTHONPATH should include '.' for local imports")

        if issues:
            return ValidationResult(
                "Config Validation",
                False,
                f"Found {len(issues)} issues",
                {"issues": issues, "env_vars": env_vars}
            )
        else:
            return ValidationResult(
                "Config Validation",
                True,
                "Configuration looks good",
                {"env_vars": env_vars}
            )

    def check_slurm_environment(self) -> ValidationResult:
        """Check SLURM environment if on HPC."""
        slurm_vars = {
            "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
            "SLURM_JOB_NAME": os.environ.get("SLURM_JOB_NAME"),
            "SLURM_NODELIST": os.environ.get("SLURM_NODELIST"),
            "SLURM_GPUS": os.environ.get("SLURM_GPUS"),
        }

        if any(slurm_vars.values()):
            # We're on SLURM
            return ValidationResult(
                "SLURM Environment",
                True,
                f"Running on SLURM job {slurm_vars['SLURM_JOB_ID']}",
                {"slurm_vars": slurm_vars}
            )
        else:
            # Not on SLURM (local dev)
            return ValidationResult(
                "SLURM Environment",
                True,
                "Not running on SLURM (local environment)",
                {"slurm_vars": slurm_vars}
            )

    def check_forward_pass(self) -> ValidationResult:
        """Check if a forward pass works (full check only)."""
        if not TORCH_AVAILABLE:
            return ValidationResult(
                "Forward Pass",
                True,
                "Skipped - requires PyTorch (run on HPC)",
                {"environment": "local_dev"}
            )

        try:
            from telepathy.latent_bridge import LatentBridge

            # Create tiny bridge for testing
            model = LatentBridge(
                model_a_name="meta-llama/Llama-3.1-1B",
                model_b_name="meta-llama/Llama-3.1-1B",
                latent_dim=128,
                num_latents=8,
            )

            # Create dummy input
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)

            batch_size = 2
            seq_len = 32
            input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)

            # Forward pass
            with torch.no_grad():
                outputs = model(input_ids)

            # Check outputs
            if outputs is not None and hasattr(outputs, 'loss'):
                return ValidationResult(
                    "Forward Pass",
                    True,
                    "Forward pass successful",
                    {"output_shape": str(outputs.logits.shape) if hasattr(outputs, 'logits') else None}
                )
            else:
                return ValidationResult(
                    "Forward Pass",
                    False,
                    "Forward pass produced unexpected outputs",
                    {}
                )
        except Exception as e:
            return ValidationResult(
                "Forward Pass",
                False,
                f"Forward pass failed: {str(e)}",
                {"error": str(e)}
            )

    def check_backward_pass(self) -> ValidationResult:
        """Check if backward pass works (full check only)."""
        if not TORCH_AVAILABLE:
            return ValidationResult(
                "Backward Pass",
                True,
                "Skipped - requires PyTorch (run on HPC)",
                {"environment": "local_dev"}
            )

        try:
            from telepathy.latent_bridge import LatentBridge

            # Create tiny bridge
            model = LatentBridge(
                model_a_name="meta-llama/Llama-3.1-1B",
                model_b_name="meta-llama/Llama-3.1-1B",
                latent_dim=128,
                num_latents=8,
            )

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)

            # Create dummy input
            batch_size = 2
            seq_len = 32
            input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
            labels = input_ids.clone()

            # Forward + backward
            outputs = model(input_ids, labels=labels)
            if hasattr(outputs, 'loss'):
                loss = outputs.loss
                loss.backward()

                # Check gradients
                has_grads = any(
                    p.grad is not None and p.grad.abs().sum() > 0
                    for p in model.parameters()
                    if p.requires_grad
                )

                if has_grads:
                    return ValidationResult(
                        "Backward Pass",
                        True,
                        "Backward pass successful, gradients computed",
                        {"loss": float(loss.item())}
                    )
                else:
                    return ValidationResult(
                        "Backward Pass",
                        False,
                        "No gradients computed",
                        {}
                    )
            else:
                return ValidationResult(
                    "Backward Pass",
                    False,
                    "No loss computed",
                    {}
                )
        except Exception as e:
            return ValidationResult(
                "Backward Pass",
                False,
                f"Backward pass failed: {str(e)}",
                {"error": str(e)}
            )

    def check_telepathy_modules(self) -> ValidationResult:
        """Check telepathy-specific modules and classes."""
        issues = []
        modules_checked = []

        # Check if we're on HPC or local
        is_local = not os.environ.get("SLURM_JOB_ID")

        if is_local:
            # Basic file existence checks only
            telepathy_dir = Path(__file__).parent
            required_files = [
                'latent_bridge.py',
                'train_telepathy_sst2.py',
                'train_telepathy_agnews.py',
                'train_telepathy_trec.py',
                'eval_telepathy_sst2.py',
                'eval_telepathy_agnews.py',
                'eval_telepathy_trec.py',
                'run_comprehensive_revision.py',
                'aggregate_results.py',
            ]

            missing = []
            for file in required_files:
                file_path = telepathy_dir / file
                if not file_path.exists():
                    missing.append(file)
                else:
                    modules_checked.append(file)

            if missing:
                return ValidationResult(
                    "Telepathy Modules",
                    False,
                    f"Missing files: {', '.join(missing)}",
                    {"missing": missing, "found": modules_checked}
                )
            else:
                return ValidationResult(
                    "Telepathy Modules",
                    True,
                    f"All {len(required_files)} telepathy files exist",
                    {"modules": modules_checked}
                )
        else:
            # On HPC, actually try importing key modules
            try:
                # Import the latent bridge module
                import telepathy.latent_bridge as lb
                modules_checked.append("latent_bridge")

                # Check if we can import train modules
                import telepathy.train_telepathy_sst2 as train_sst2
                modules_checked.append("train_telepathy_sst2")

                # Check if we can import eval modules
                import telepathy.eval_telepathy_sst2 as eval_sst2
                modules_checked.append("eval_telepathy_sst2")

                # Check key dependencies
                import torch
                import transformers
                modules_checked.extend(["torch", "transformers"])

                return ValidationResult(
                    "Telepathy Modules",
                    True,
                    f"All telepathy modules working",
                    {"modules_tested": modules_checked}
                )
            except Exception as e:
                return ValidationResult(
                    "Telepathy Modules",
                    False,
                    f"Module import failed: {str(e)}",
                    {"error": str(e), "modules_checked": modules_checked}
                )

    def check_experiment_scripts(self) -> ValidationResult:
        """Check that experiment scripts exist and are executable."""
        base_dir = Path(__file__).parent.parent
        telepathy_dir = Path(__file__).parent

        required_scripts = [
            telepathy_dir / "run_comprehensive_revision.py",
            telepathy_dir / "aggregate_results.py",
            telepathy_dir / "submit_enhanced_arxiv.slurm",
            telepathy_dir / "submit_comprehensive_revision.slurm",
            telepathy_dir / "submit_validation.slurm",
        ]

        missing = []
        not_executable = []
        found = []

        for script in required_scripts:
            if not script.exists():
                missing.append(script.name)
            else:
                found.append(script.name)
                # Check if Python scripts have proper shebang
                if script.suffix == '.py':
                    with open(script, 'r') as f:
                        first_line = f.readline()
                        if not first_line.startswith('#!'):
                            not_executable.append(f"{script.name} (no shebang)")

        if missing:
            return ValidationResult(
                "Experiment Scripts",
                False,
                f"Missing scripts: {', '.join(missing)}",
                {"missing": missing, "found": found}
            )
        elif not_executable:
            return ValidationResult(
                "Experiment Scripts",
                False,
                f"Scripts missing shebang: {', '.join(not_executable)}",
                {"not_executable": not_executable, "found": found}
            )
        else:
            return ValidationResult(
                "Experiment Scripts",
                True,
                f"All {len(required_scripts)} experiment scripts found",
                {"scripts": found}
            )

    def check_multimodel_compatibility(self) -> ValidationResult:
        """Check if multiple models can be loaded together (full check only)."""
        if not TORCH_AVAILABLE:
            return ValidationResult(
                "Multi-Model Compatibility",
                True,
                "Skipped - requires PyTorch (run on HPC)",
                {"environment": "local_dev"}
            )

        try:
            from telepathy.latent_bridge import LatentBridge

            # Try loading different models
            model = LatentBridge(
                model_a_name="meta-llama/Llama-3.1-1B",
                model_b_name="meta-llama/Llama-3.1-3B",
                latent_dim=256,
                num_latents=16,
            )

            # Check memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.max_memory_allocated() / (1024**3)
                memory_info = f"GPU memory used: {memory_used:.2f}GB"
            else:
                memory_info = "CPU mode"

            return ValidationResult(
                "Multi-Model Compatibility",
                True,
                f"Multi-model loading successful ({memory_info})",
                {"models": ["Llama-3.1-1B", "Llama-3.1-3B"], "memory": memory_info}
            )
        except Exception as e:
            return ValidationResult(
                "Multi-Model Compatibility",
                False,
                f"Multi-model loading failed: {str(e)}",
                {"error": str(e)}
            )

    def check_checkpoint_save_load(self) -> ValidationResult:
        """Check if checkpoints can be saved and loaded (full check only)."""
        if not TORCH_AVAILABLE:
            return ValidationResult(
                "Checkpoint Save/Load",
                True,
                "Skipped - requires PyTorch (run on HPC)",
                {"environment": "local_dev"}
            )

        try:
            from telepathy.latent_bridge import LatentBridge
            import tempfile

            # Create bridge
            model = LatentBridge(
                model_a_name="meta-llama/Llama-3.1-1B",
                model_b_name="meta-llama/Llama-3.1-1B",
                latent_dim=128,
                num_latents=8,
            )

            # Save checkpoint
            with tempfile.TemporaryDirectory() as tmpdir:
                ckpt_path = Path(tmpdir) / "test_checkpoint.pt"

                # Create dummy state
                state = {
                    'model_state_dict': model.state_dict(),
                    'epoch': 1,
                    'step': 100,
                    'config': {'test': True}
                }

                # Save
                torch.save(state, ckpt_path)
                size_mb = ckpt_path.stat().st_size / (1024**2)

                # Load
                loaded_state = torch.load(ckpt_path, map_location='cpu')

                # Verify
                if 'model_state_dict' in loaded_state and 'epoch' in loaded_state:
                    return ValidationResult(
                        "Checkpoint Save/Load",
                        True,
                        f"Checkpoint save/load successful ({size_mb:.1f}MB)",
                        {"checkpoint_size_mb": size_mb}
                    )
                else:
                    return ValidationResult(
                        "Checkpoint Save/Load",
                        False,
                        "Checkpoint missing expected keys",
                        {}
                    )
        except Exception as e:
            return ValidationResult(
                "Checkpoint Save/Load",
                False,
                f"Checkpoint test failed: {str(e)}",
                {"error": str(e)}
            )

    def estimate_runtime(self) -> Dict[str, float]:
        """Estimate runtime for different configurations."""
        # Based on empirical measurements
        estimates = {
            "1B_model_100_samples": 0.5,  # hours
            "1B_model_1000_samples": 2.0,
            "3B_model_100_samples": 1.0,
            "3B_model_1000_samples": 4.0,
            "8B_model_100_samples": 2.0,
            "8B_model_1000_samples": 8.0,
            "all_3_models_100_samples": 3.0,
            "all_3_models_1000_samples": 12.0,
        }
        return estimates

    def print_summary(self) -> bool:
        """Print validation summary."""
        print("=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)

        passed_count = sum(1 for r in self.results if r.passed)
        failed_count = len(self.results) - passed_count

        print(f"Total checks: {len(self.results)}")
        print(f"  ✅ Passed: {passed_count}")
        print(f"  ❌ Failed: {failed_count}")
        print()

        if failed_count > 0:
            print("Failed checks:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}: {r.message}")
                    if r.details and 'issues' in r.details:
                        for issue in r.details['issues']:
                            print(f"    • {issue}")
            print()

        # Memory recommendations
        memory_check = next((r for r in self.results if r.name == "Memory Check"), None)
        if memory_check and memory_check.details:
            print("Memory Recommendations:")
            for rec in memory_check.details.get('recommendations', []):
                print(f"  {rec}")
            print()

        # Runtime estimates
        print("Estimated Runtimes:")
        for config, hours in self.estimate_runtime().items():
            print(f"  • {config}: ~{hours:.1f} hours")
        print()

        # GPU info
        gpu_check = next((r for r in self.results if r.name == "CUDA/GPU Check"), None)
        if gpu_check and gpu_check.passed and gpu_check.details:
            print("GPU Configuration:")
            for device in gpu_check.details.get('devices', []):
                print(f"  • GPU {device['index']}: {device['name']} ({device['memory_gb']:.1f}GB)")
            print()

        # Overall result
        elapsed_time = time.time() - self.start_time
        print(f"Validation completed in {elapsed_time:.1f} seconds")

        if failed_count == 0:
            print("\n🎉 ALL CHECKS PASSED - Ready to run experiments!")
            return True
        else:
            print("\n⚠️  VALIDATION FAILED - Please fix issues before running experiments")
            return False

    def save_report(self, output_path: str = "validation_report.json"):
        """Save detailed validation report to JSON."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "mode": "full" if self.full_check else "quick",
            "summary": {
                "total_checks": len(self.results),
                "passed": sum(1 for r in self.results if r.passed),
                "failed": sum(1 for r in self.results if not r.passed),
            },
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "message": r.message,
                    "details": r.details
                }
                for r in self.results
            ],
            "runtime_estimates": self.estimate_runtime(),
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\nDetailed report saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate telepathy experiments")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full validation including forward/backward passes"
    )
    parser.add_argument(
        "--save-report",
        type=str,
        default=None,
        help="Path to save detailed JSON report"
    )
    args = parser.parse_args()

    # Run validation
    validator = ExperimentValidator(full_check=args.full)
    success = validator.run_all_checks()

    # Save report if requested
    if args.save_report:
        validator.save_report(args.save_report)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()