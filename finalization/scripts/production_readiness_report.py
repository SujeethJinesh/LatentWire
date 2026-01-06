#!/usr/bin/env python3
"""
Production Readiness Report Generator for LatentWire

This script generates a comprehensive report on the production readiness
of the LatentWire system for HPC deployment with SLURM.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

def check_command_exists(command: str) -> bool:
    """Check if a command exists in the system."""
    try:
        subprocess.run(["which", command], capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def check_python_modules() -> Tuple[List[str], List[str], Dict[str, str]]:
    """Check required Python modules."""
    required_modules = [
        'torch',
        'transformers',
        'datasets',
        'accelerate',
        'numpy',
        'scipy',
        'sklearn',
        'rouge_score',
        'statsmodels',
        'pandas',
        'matplotlib',
        'seaborn',
        'peft',
        'tqdm',
        'sentence_transformers'
    ]

    installed = []
    missing = []
    versions = {}

    for module in required_modules:
        try:
            mod = __import__(module)
            installed.append(module)
            if hasattr(mod, '__version__'):
                versions[module] = mod.__version__
            else:
                versions[module] = 'unknown'
        except ImportError:
            missing.append(module)

    return installed, missing, versions

def check_latentwire_imports() -> Tuple[List[str], List[str]]:
    """Check if latentwire modules can be imported."""
    modules_to_check = [
        'latentwire.train',
        'latentwire.eval',
        'latentwire.models',
        'latentwire.data',
        'latentwire.core_utils',
        'latentwire.checkpointing',
        'latentwire.data_pipeline',
        'latentwire.feature_registry',
        'latentwire.loss_bundles',
        'latentwire.losses'
    ]

    successful = []
    failed = []

    for module in modules_to_check:
        try:
            __import__(module)
            successful.append(module)
        except Exception as e:
            failed.append(f"{module}: {str(e)}")

    return successful, failed

def check_slurm_scripts() -> Dict[str, List[str]]:
    """Check SLURM scripts for correct configuration."""
    issues = {
        'missing_account': [],
        'missing_partition': [],
        'wrong_workdir': [],
        'not_found': []
    }

    slurm_dir = Path('telepathy')
    if not slurm_dir.exists():
        return issues

    for script_path in slurm_dir.glob('*.slurm'):
        try:
            content = script_path.read_text()

            if '--account=marlowe-m000066' not in content:
                issues['missing_account'].append(script_path.name)

            if '--partition=preempt' not in content:
                issues['missing_partition'].append(script_path.name)

            if 'WORK_DIR="/projects/m000066/sujinesh/LatentWire"' not in content:
                issues['wrong_workdir'].append(script_path.name)
        except Exception:
            issues['not_found'].append(script_path.name)

    return issues

def check_file_permissions() -> Dict[str, List[str]]:
    """Check file permissions for scripts."""
    issues = {
        'not_executable_sh': [],
        'not_executable_slurm': [],
        'not_found': []
    }

    # Check shell scripts
    scripts_dir = Path('scripts')
    if scripts_dir.exists():
        for script in scripts_dir.glob('*.sh'):
            if not os.access(script, os.X_OK):
                issues['not_executable_sh'].append(script.name)

    # Check SLURM scripts
    slurm_dir = Path('telepathy')
    if slurm_dir.exists():
        for script in slurm_dir.glob('*.slurm'):
            if not os.access(script, os.X_OK):
                issues['not_executable_slurm'].append(script.name)

    return issues

def check_gpu_availability():
    """Check GPU availability and configuration."""
    try:
        import torch
        gpu_info = {
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'gpus': []
        }

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_info['gpus'].append({
                    'index': i,
                    'name': props.name,
                    'memory_gb': props.total_memory / (1024**3)
                })

        return gpu_info
    except ImportError:
        return {'error': 'PyTorch not installed'}

def estimate_memory_requirements() -> Dict[str, float]:
    """Estimate memory requirements for typical configurations."""
    # Model sizes (in billions of parameters)
    llama_8b = 8e9
    qwen_7b = 7e9
    encoder = 50e6  # ~50M for encoder + adapters

    # Bytes per parameter (fp16)
    bytes_per_param = 2

    # Typical batch size
    batch_size = 64
    seq_len = 512
    hidden_dim = 4096

    # Calculate memory requirements (in GB)
    model_memory = (llama_8b + qwen_7b + encoder) * bytes_per_param / 1e9
    activation_memory = batch_size * seq_len * hidden_dim * 4 / 1e9
    optimizer_memory = model_memory * 2  # Adam optimizer states

    total_memory = model_memory + activation_memory + optimizer_memory

    return {
        'model_memory_gb': round(model_memory, 1),
        'activation_memory_gb': round(activation_memory, 1),
        'optimizer_memory_gb': round(optimizer_memory, 1),
        'total_memory_gb': round(total_memory, 1),
        'recommended_slurm_memory_gb': int(total_memory * 1.5),
        'recommended_gpus': 4
    }

def check_data_access():
    """Check if datasets can be loaded."""
    datasets_status = {}

    try:
        from latentwire.data import load_examples

        test_datasets = ['squad', 'hotpotqa', 'agnews', 'sst2', 'gsm8k']
        for dataset in test_datasets:
            try:
                examples = load_examples(dataset, limit=2)
                datasets_status[dataset] = {
                    'status': 'OK',
                    'samples_loaded': len(examples)
                }
            except Exception as e:
                datasets_status[dataset] = {
                    'status': 'FAILED',
                    'error': str(e)
                }
    except ImportError as e:
        datasets_status['error'] = f"Cannot import latentwire.data: {e}"

    return datasets_status

def generate_report() -> Dict:
    """Generate comprehensive production readiness report."""
    report = {
        'timestamp': datetime.now().isoformat(),
        'system': {
            'python_version': sys.version,
            'platform': sys.platform,
            'cwd': os.getcwd()
        },
        'dependencies': {},
        'latentwire_modules': {},
        'slurm_configuration': {},
        'file_permissions': {},
        'gpu_status': {},
        'memory_estimate': {},
        'data_access': {},
        'recommendations': [],
        'critical_issues': [],
        'warnings': []
    }

    # Check Python modules
    installed, missing, versions = check_python_modules()
    report['dependencies'] = {
        'installed': installed,
        'missing': missing,
        'versions': versions
    }
    if missing:
        report['critical_issues'].append(f"Missing Python modules: {', '.join(missing)}")
        report['recommendations'].append("Run: pip install -r requirements.txt")

    # Check latentwire imports
    successful, failed = check_latentwire_imports()
    report['latentwire_modules'] = {
        'successful': successful,
        'failed': failed
    }
    if failed:
        report['critical_issues'].append(f"Failed imports: {len(failed)} modules")

    # Check SLURM scripts
    slurm_issues = check_slurm_scripts()
    report['slurm_configuration'] = slurm_issues
    total_slurm_issues = sum(len(v) for v in slurm_issues.values())
    if total_slurm_issues > 0:
        report['warnings'].append(f"SLURM configuration issues in {total_slurm_issues} scripts")

    # Check file permissions
    permission_issues = check_file_permissions()
    report['file_permissions'] = permission_issues
    if permission_issues['not_executable_sh'] or permission_issues['not_executable_slurm']:
        report['warnings'].append("Some scripts are not executable")
        report['recommendations'].append("Run: chmod +x scripts/*.sh telepathy/*.slurm")

    # Check GPU
    report['gpu_status'] = check_gpu_availability()

    # Memory requirements
    report['memory_estimate'] = estimate_memory_requirements()

    # Data access
    report['data_access'] = check_data_access()

    # Overall readiness assessment
    is_ready = (
        len(report['critical_issues']) == 0 and
        len(report['dependencies']['missing']) == 0 and
        len(report['latentwire_modules']['failed']) == 0
    )

    report['production_ready'] = is_ready

    if is_ready:
        report['recommendations'].extend([
            "System is production ready for HPC deployment",
            "1. Push code: git add -A && git commit && git push",
            "2. On HPC: cd /projects/m000066/sujinesh/LatentWire && git pull",
            "3. Submit job: sbatch telepathy/submit_production_readiness.slurm",
            "4. Monitor: squeue -u $USER"
        ])
    else:
        report['recommendations'].insert(0, "⚠️ Fix critical issues before deployment")

    return report

def main():
    """Main function."""
    print("=" * 70)
    print("LatentWire Production Readiness Report")
    print("=" * 70)

    report = generate_report()

    # Print summary
    print(f"\nTimestamp: {report['timestamp']}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {report['system']['platform']}")

    # Dependencies
    print("\n📦 DEPENDENCIES")
    print(f"  Installed: {len(report['dependencies']['installed'])}")
    print(f"  Missing: {len(report['dependencies']['missing'])}")
    if report['dependencies']['missing']:
        for mod in report['dependencies']['missing']:
            print(f"    ❌ {mod}")

    # Modules
    print("\n📁 LATENTWIRE MODULES")
    print(f"  Successful imports: {len(report['latentwire_modules']['successful'])}")
    print(f"  Failed imports: {len(report['latentwire_modules']['failed'])}")
    if report['latentwire_modules']['failed']:
        for fail in report['latentwire_modules']['failed'][:3]:
            print(f"    ❌ {fail}")

    # GPU
    print("\n🖥️  GPU STATUS")
    gpu = report['gpu_status']
    if 'error' not in gpu:
        print(f"  PyTorch: {gpu.get('pytorch_version', 'N/A')}")
        print(f"  CUDA available: {gpu.get('cuda_available', False)}")
        print(f"  GPU count: {gpu.get('gpu_count', 0)}")
    else:
        print(f"  ❌ {gpu['error']}")

    # Memory
    print("\n💾 MEMORY REQUIREMENTS")
    mem = report['memory_estimate']
    print(f"  Total estimated: {mem['total_memory_gb']} GB")
    print(f"  Recommended SLURM memory: {mem['recommended_slurm_memory_gb']} GB")
    print(f"  Recommended GPUs: {mem['recommended_gpus']}")

    # Critical issues
    if report['critical_issues']:
        print("\n❌ CRITICAL ISSUES")
        for issue in report['critical_issues']:
            print(f"  • {issue}")

    # Warnings
    if report['warnings']:
        print("\n⚠️  WARNINGS")
        for warning in report['warnings']:
            print(f"  • {warning}")

    # Final verdict
    print("\n" + "=" * 70)
    if report['production_ready']:
        print("✅ SYSTEM IS PRODUCTION READY")
    else:
        print("❌ SYSTEM NOT READY - FIX ISSUES ABOVE")

    print("\n📋 RECOMMENDATIONS:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"  {i}. {rec}")

    # Save full report
    report_path = Path('runs/validation')
    report_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = report_path / f'production_readiness_{timestamp}.json'

    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n📄 Full report saved to: {json_path}")
    print("=" * 70)

    return 0 if report['production_ready'] else 1

if __name__ == '__main__':
    sys.exit(main())