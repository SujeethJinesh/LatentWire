#!/usr/bin/env python3
"""
System check utility to verify environment is ready for experiments.
"""

import os
import sys
import subprocess
from pathlib import Path

import torch


def check_environment():
    """Check if environment is properly configured."""
    checks = []

    # Check Python version
    python_version = sys.version_info
    checks.append({
        'name': 'Python Version',
        'status': python_version >= (3, 8),
        'message': f"Python {python_version.major}.{python_version.minor}.{python_version.micro}"
    })

    # Check PyTorch
    try:
        pytorch_version = torch.__version__
        checks.append({
            'name': 'PyTorch',
            'status': True,
            'message': f"Version {pytorch_version}"
        })
    except:
        checks.append({
            'name': 'PyTorch',
            'status': False,
            'message': "Not installed"
        })

    # Check CUDA
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        gpu_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        checks.append({
            'name': 'CUDA',
            'status': True,
            'message': f"Version {cuda_version}, {gpu_count} GPUs: {', '.join(gpu_names)}"
        })
    else:
        checks.append({
            'name': 'CUDA',
            'status': False,
            'message': "No CUDA devices available"
        })

    # Check disk space
    import shutil
    total, used, free = shutil.disk_usage("/")
    free_gb = free // (2**30)
    checks.append({
        'name': 'Disk Space',
        'status': free_gb > 50,
        'message': f"{free_gb} GB free"
    })

    # Check if on HPC
    hostname = os.environ.get('HOSTNAME', 'unknown')
    on_hpc = '/projects' in str(Path.cwd()) or 'marlowe' in hostname.lower()
    checks.append({
        'name': 'HPC Environment',
        'status': True,  # Not a failure if not on HPC
        'message': "On HPC" if on_hpc else "Local environment"
    })

    # Print results
    print("=" * 50)
    print("SYSTEM CHECK")
    print("=" * 50)

    all_pass = True
    for check in checks:
        status = "✓" if check['status'] else "✗"
        print(f"{status} {check['name']}: {check['message']}")
        if not check['status'] and check['name'] != 'HPC Environment':
            all_pass = False

    print("=" * 50)
    if all_pass:
        print("✓ System ready for experiments")
    else:
        print("✗ Some checks failed - review before proceeding")

    return all_pass


if __name__ == '__main__':
    success = check_environment()
    sys.exit(0 if success else 1)