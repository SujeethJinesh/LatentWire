#!/usr/bin/env python3
"""
Real-time monitoring for SLURM experiments on HPC.

Usage:
    python finalization/monitor.py --job-id <SLURM_JOB_ID>
    python finalization/monitor.py --latest  # Monitor most recent job
    python finalization/monitor.py --all     # Show all running jobs

Features:
- GPU utilization and memory tracking
- Training progress (epoch, step, loss curves)
- Temperature monitoring and thermal throttling detection
- Checkpoint status and size tracking
- ETA estimation based on current speed
- Preemption detection and warnings
- Live log tail with error highlighting
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Terminal colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class SLURMMonitor:
    """Monitor SLURM jobs with GPU tracking and training progress."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.base_dir = Path("/projects/m000066/sujinesh/LatentWire")
        self.runs_dir = self.base_dir / "runs"

        # Find log files for this job
        self.log_file = self.runs_dir / f"experiment_{job_id}.log"
        self.err_file = self.runs_dir / f"experiment_{job_id}.err"

        # Training metrics tracking
        self.loss_history = deque(maxlen=100)
        self.gpu_history = deque(maxlen=60)  # 1 minute of history
        self.memory_history = deque(maxlen=60)

        # Job metadata
        self.start_time = None
        self.node_name = None
        self.num_gpus = None
        self.total_epochs = None
        self.current_epoch = None
        self.current_step = None
        self.total_steps = None

    def get_job_info(self) -> Dict:
        """Get SLURM job information."""
        try:
            cmd = f"scontrol show job {self.job_id}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            if result.returncode != 0:
                return {"status": "NOT_FOUND"}

            info = {}
            for line in result.stdout.split('\n'):
                # Parse key=value pairs
                pairs = re.findall(r'(\w+)=([^\s]+)', line)
                for key, value in pairs:
                    info[key] = value

            return info
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}

    def get_gpu_stats(self, node: str) -> Dict:
        """Get GPU utilization from nvidia-smi on the compute node."""
        try:
            cmd = f"ssh {node} 'nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)

            if result.returncode != 0:
                return {}

            gpu_stats = []
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 7:
                    gpu_stats.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'utilization': float(parts[2]) if parts[2] != '[N/A]' else 0,
                        'memory_used': float(parts[3]) if parts[3] != '[N/A]' else 0,
                        'memory_total': float(parts[4]) if parts[4] != '[N/A]' else 0,
                        'temperature': float(parts[5]) if parts[5] != '[N/A]' else 0,
                        'power': float(parts[6]) if parts[6] != '[N/A]' else 0,
                    })

            return {'gpus': gpu_stats}
        except subprocess.TimeoutExpired:
            return {'error': 'timeout'}
        except Exception as e:
            return {'error': str(e)}

    def parse_training_progress(self, tail_lines: int = 100) -> Dict:
        """Parse training progress from log file."""
        if not self.log_file.exists():
            return {}

        try:
            # Read last N lines
            cmd = f"tail -n {tail_lines} {self.log_file}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            lines = result.stdout.split('\n')

            progress = {}

            for line in lines:
                # Parse epoch progress
                if "Epoch" in line and "/" in line:
                    epoch_match = re.search(r'Epoch (\d+)/(\d+)', line)
                    if epoch_match:
                        self.current_epoch = int(epoch_match.group(1))
                        self.total_epochs = int(epoch_match.group(2))

                # Parse step progress
                if "Step" in line or "step" in line:
                    step_match = re.search(r'[Ss]tep[:\s]+(\d+)/(\d+)', line)
                    if step_match:
                        self.current_step = int(step_match.group(1))
                        self.total_steps = int(step_match.group(2))
                    else:
                        # Try alternative format
                        step_match = re.search(r'[Ss]tep[:\s]+(\d+)', line)
                        if step_match:
                            self.current_step = int(step_match.group(1))

                # Parse loss values
                loss_patterns = [
                    r'loss[:\s]+([\d.]+)',
                    r'Loss[:\s]+([\d.]+)',
                    r'train_loss[:\s]+([\d.]+)',
                    r'kl_loss[:\s]+([\d.]+)',
                    r'ce_loss[:\s]+([\d.]+)',
                ]
                for pattern in loss_patterns:
                    loss_match = re.search(pattern, line)
                    if loss_match:
                        loss_val = float(loss_match.group(1))
                        if not np.isnan(loss_val) and not np.isinf(loss_val):
                            self.loss_history.append(loss_val)
                            progress['latest_loss'] = loss_val

                # Parse learning rate
                lr_match = re.search(r'lr[:\s]+([\d.e-]+)', line)
                if lr_match:
                    progress['learning_rate'] = float(lr_match.group(1))

                # Parse evaluation metrics
                if "F1" in line or "EM" in line:
                    f1_match = re.search(r'F1[:\s]+([\d.]+)', line)
                    em_match = re.search(r'EM[:\s]+([\d.]+)', line)
                    if f1_match:
                        progress['f1_score'] = float(f1_match.group(1))
                    if em_match:
                        progress['em_score'] = float(em_match.group(1))

                # Parse checkpoint saves
                if "Saving checkpoint" in line or "checkpoint saved" in line.lower():
                    progress['last_checkpoint'] = datetime.now().isoformat()

            # Add parsed values to progress
            if self.current_epoch is not None:
                progress['current_epoch'] = self.current_epoch
                progress['total_epochs'] = self.total_epochs

            if self.current_step is not None:
                progress['current_step'] = self.current_step
                if self.total_steps:
                    progress['total_steps'] = self.total_steps
                    progress['progress_pct'] = (self.current_step / self.total_steps) * 100

            if self.loss_history:
                progress['avg_loss'] = np.mean(list(self.loss_history)[-10:])  # Last 10 losses
                progress['loss_trend'] = 'decreasing' if len(self.loss_history) > 2 and self.loss_history[-1] < self.loss_history[-3] else 'stable'

            return progress

        except Exception as e:
            return {'error': str(e)}

    def check_preemption_risk(self, job_info: Dict) -> str:
        """Check if job is at risk of preemption."""
        if job_info.get('Partition') == 'preempt':
            run_time = job_info.get('RunTime', '0:00:00')
            time_limit = job_info.get('TimeLimit', '12:00:00')

            # Parse times
            def parse_time(time_str):
                parts = time_str.split(':')
                if len(parts) == 3:
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                return 0

            run_seconds = parse_time(run_time)
            limit_seconds = parse_time(time_limit)

            if limit_seconds > 0:
                remaining = limit_seconds - run_seconds
                if remaining < 1800:  # Less than 30 minutes
                    return f"{Colors.RED}HIGH - {remaining//60} minutes left{Colors.END}"
                elif remaining < 3600:  # Less than 1 hour
                    return f"{Colors.YELLOW}MEDIUM - {remaining//60} minutes left{Colors.END}"

            return f"{Colors.GREEN}LOW{Colors.END}"

        return "N/A"

    def estimate_eta(self, progress: Dict) -> Optional[str]:
        """Estimate time to completion based on current progress."""
        if not progress.get('current_step') or not progress.get('total_steps'):
            return None

        # Get job runtime
        try:
            job_info = self.get_job_info()
            run_time_str = job_info.get('RunTime', '0:00:00')

            # Parse runtime to seconds
            parts = run_time_str.split(':')
            if len(parts) == 3:
                run_seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            else:
                return None

            if run_seconds == 0 or progress['current_step'] == 0:
                return None

            # Calculate ETA
            seconds_per_step = run_seconds / progress['current_step']
            remaining_steps = progress['total_steps'] - progress['current_step']
            eta_seconds = remaining_steps * seconds_per_step

            # Format ETA
            eta = datetime.now() + timedelta(seconds=eta_seconds)
            return eta.strftime('%Y-%m-%d %H:%M:%S')

        except Exception:
            return None

    def get_checkpoint_info(self) -> Dict:
        """Get information about saved checkpoints."""
        checkpoints = {}

        # Look for checkpoint directories
        if self.runs_dir.exists():
            for run_dir in self.runs_dir.glob("*"):
                if run_dir.is_dir() and "epoch" in run_dir.name:
                    # Get checkpoint size
                    total_size = sum(f.stat().st_size for f in run_dir.rglob("*") if f.is_file())
                    checkpoints[run_dir.name] = {
                        'size_mb': total_size / (1024 * 1024),
                        'modified': datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat()
                    }

        return checkpoints

    def display_dashboard(self, refresh_rate: int = 2):
        """Display real-time monitoring dashboard."""
        try:
            while True:
                # Clear screen
                os.system('clear')

                # Get job information
                job_info = self.get_job_info()

                if job_info.get('status') == 'NOT_FOUND':
                    print(f"{Colors.RED}Job {self.job_id} not found!{Colors.END}")
                    break

                # Header
                print(f"{Colors.BOLD}{Colors.HEADER}═══════════════════════════════════════════════════════════════{Colors.END}")
                print(f"{Colors.BOLD}{Colors.CYAN}  SLURM Job Monitor - Job ID: {self.job_id}{Colors.END}")
                print(f"{Colors.BOLD}{Colors.HEADER}═══════════════════════════════════════════════════════════════{Colors.END}\n")

                # Job Status
                state = job_info.get('JobState', 'UNKNOWN')
                state_color = Colors.GREEN if state == 'RUNNING' else Colors.YELLOW if state == 'PENDING' else Colors.RED

                print(f"{Colors.BOLD}Job Information:{Colors.END}")
                print(f"  Status: {state_color}{state}{Colors.END}")
                print(f"  Node: {job_info.get('NodeList', 'N/A')}")
                print(f"  Partition: {job_info.get('Partition', 'N/A')}")
                print(f"  Runtime: {job_info.get('RunTime', 'N/A')}")
                print(f"  Time Limit: {job_info.get('TimeLimit', 'N/A')}")
                print(f"  Preemption Risk: {self.check_preemption_risk(job_info)}")
                print()

                # GPU Stats
                if state == 'RUNNING' and job_info.get('NodeList'):
                    gpu_stats = self.get_gpu_stats(job_info['NodeList'])

                    if 'gpus' in gpu_stats:
                        print(f"{Colors.BOLD}GPU Status:{Colors.END}")
                        for gpu in gpu_stats['gpus']:
                            mem_pct = (gpu['memory_used'] / gpu['memory_total'] * 100) if gpu['memory_total'] > 0 else 0

                            # Color code based on utilization
                            util_color = Colors.GREEN if gpu['utilization'] > 70 else Colors.YELLOW if gpu['utilization'] > 30 else Colors.RED
                            temp_color = Colors.RED if gpu['temperature'] > 80 else Colors.YELLOW if gpu['temperature'] > 70 else Colors.GREEN

                            print(f"  GPU {gpu['index']} ({gpu['name']}):")
                            print(f"    Utilization: {util_color}{gpu['utilization']:.1f}%{Colors.END}")
                            print(f"    Memory: {gpu['memory_used']:.0f}/{gpu['memory_total']:.0f} MB ({mem_pct:.1f}%)")
                            print(f"    Temperature: {temp_color}{gpu['temperature']:.0f}°C{Colors.END}")
                            print(f"    Power: {gpu['power']:.0f}W")

                            # Track for history
                            self.gpu_history.append(gpu['utilization'])
                            self.memory_history.append(mem_pct)

                        # Show trends
                        if len(self.gpu_history) > 10:
                            avg_gpu = np.mean(list(self.gpu_history)[-10:])
                            avg_mem = np.mean(list(self.memory_history)[-10:])
                            print(f"\n  10-sample averages: GPU {avg_gpu:.1f}%, Memory {avg_mem:.1f}%")
                    print()

                # Training Progress
                progress = self.parse_training_progress()
                if progress:
                    print(f"{Colors.BOLD}Training Progress:{Colors.END}")

                    if 'current_epoch' in progress:
                        epoch_bar = self.make_progress_bar(progress['current_epoch'], progress['total_epochs'], width=30)
                        print(f"  Epoch: {progress['current_epoch']}/{progress['total_epochs']} {epoch_bar}")

                    if 'current_step' in progress:
                        if 'total_steps' in progress:
                            step_bar = self.make_progress_bar(progress['current_step'], progress['total_steps'], width=30)
                            print(f"  Step: {progress['current_step']}/{progress['total_steps']} {step_bar} ({progress.get('progress_pct', 0):.1f}%)")
                        else:
                            print(f"  Step: {progress['current_step']}")

                    if 'latest_loss' in progress:
                        trend_icon = "↓" if progress.get('loss_trend') == 'decreasing' else "→"
                        print(f"  Loss: {progress['latest_loss']:.4f} {trend_icon} (avg: {progress.get('avg_loss', 0):.4f})")

                    if 'learning_rate' in progress:
                        print(f"  Learning Rate: {progress['learning_rate']:.2e}")

                    if 'f1_score' in progress:
                        print(f"  F1 Score: {progress['f1_score']:.3f}")

                    if 'em_score' in progress:
                        print(f"  EM Score: {progress['em_score']:.3f}")

                    # ETA
                    eta = self.estimate_eta(progress)
                    if eta:
                        print(f"  ETA: {eta}")

                    print()

                # Checkpoints
                checkpoints = self.get_checkpoint_info()
                if checkpoints:
                    print(f"{Colors.BOLD}Checkpoints:{Colors.END}")
                    for name, info in sorted(checkpoints.items())[:5]:  # Show last 5
                        print(f"  {name}: {info['size_mb']:.1f} MB")
                    print()

                # Error check
                if self.err_file.exists():
                    err_size = self.err_file.stat().st_size
                    if err_size > 0:
                        print(f"{Colors.BOLD}{Colors.RED}Errors Detected:{Colors.END}")
                        # Show last few error lines
                        cmd = f"tail -n 5 {self.err_file}"
                        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                        for line in result.stdout.split('\n'):
                            if line.strip():
                                print(f"  {Colors.RED}{line[:100]}{Colors.END}")
                        print()

                # Log tail
                if self.log_file.exists():
                    print(f"{Colors.BOLD}Recent Log Output:{Colors.END}")
                    cmd = f"tail -n 10 {self.log_file}"
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    for line in result.stdout.split('\n'):
                        if line.strip():
                            # Highlight important lines
                            if 'error' in line.lower() or 'exception' in line.lower():
                                print(f"  {Colors.RED}{line[:100]}{Colors.END}")
                            elif 'warning' in line.lower():
                                print(f"  {Colors.YELLOW}{line[:100]}{Colors.END}")
                            elif 'success' in line.lower() or 'complete' in line.lower():
                                print(f"  {Colors.GREEN}{line[:100]}{Colors.END}")
                            else:
                                print(f"  {line[:100]}")

                # Footer
                print(f"\n{Colors.BOLD}Press Ctrl+C to exit. Refreshing every {refresh_rate}s...{Colors.END}")

                time.sleep(refresh_rate)

        except KeyboardInterrupt:
            print(f"\n{Colors.BOLD}Monitoring stopped.{Colors.END}")
            sys.exit(0)

    def make_progress_bar(self, current: int, total: int, width: int = 30) -> str:
        """Create a text progress bar."""
        if total == 0:
            return ""

        percent = current / total
        filled = int(width * percent)
        bar = '█' * filled + '░' * (width - filled)
        return f"[{bar}]"


def get_latest_job() -> Optional[str]:
    """Get the most recent SLURM job ID for current user."""
    try:
        cmd = "squeue -u $USER -h -o %i -S -t | head -1"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return None


def list_all_jobs():
    """List all running jobs for current user."""
    try:
        cmd = "squeue -u $USER -o '%.10i %.20j %.8T %.10M %.6D %.20R'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"{Colors.RED}Failed to list jobs{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}Error: {e}{Colors.END}")


def main():
    parser = argparse.ArgumentParser(description="Real-time SLURM experiment monitor")
    parser.add_argument('--job-id', type=str, help='SLURM job ID to monitor')
    parser.add_argument('--latest', action='store_true', help='Monitor most recent job')
    parser.add_argument('--all', action='store_true', help='List all running jobs')
    parser.add_argument('--refresh', type=int, default=2, help='Refresh rate in seconds (default: 2)')

    args = parser.parse_args()

    if args.all:
        list_all_jobs()
        return

    # Determine job ID
    job_id = args.job_id
    if args.latest:
        job_id = get_latest_job()
        if not job_id:
            print(f"{Colors.RED}No running jobs found for current user{Colors.END}")
            return
        print(f"{Colors.CYAN}Monitoring latest job: {job_id}{Colors.END}")

    if not job_id:
        print(f"{Colors.RED}Please specify --job-id, --latest, or --all{Colors.END}")
        parser.print_help()
        return

    # Start monitoring
    monitor = SLURMMonitor(job_id)
    monitor.display_dashboard(refresh_rate=args.refresh)


if __name__ == '__main__':
    main()