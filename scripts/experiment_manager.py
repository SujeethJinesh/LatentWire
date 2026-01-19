#!/usr/bin/env python3
"""
Experiment Manager CLI for MacBook.

Analyzes experiment results, lists failures, and prepares re-runs.

Usage:
    # View status summary
    python scripts/experiment_manager.py --status

    # List experiments by status
    python scripts/experiment_manager.py --list failed
    python scripts/experiment_manager.py --list completed

    # Show experiment details
    python scripts/experiment_manager.py --details flow_matching_arc_easy_seed42

    # Mark for re-run
    python scripts/experiment_manager.py --mark-rerun flow_matching_arc_easy_seed42
    python scripts/experiment_manager.py --mark-rerun-all-failed

    # Validate result files exist
    python scripts/experiment_manager.py --validate

    # Rebuild registry from existing files
    python scripts/experiment_manager.py --rebuild-registry
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
# Import directly from module file to avoid loading torch-dependent packages
sys.path.insert(0, str(Path(__file__).parent.parent / "telepathy"))

from experiment_registry import ExperimentRegistry


def format_timestamp(ts: str) -> str:
    """Format ISO timestamp to human-readable."""
    if not ts:
        return "N/A"
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return str(ts)[:16] if ts else "N/A"


def print_status(registry: ExperimentRegistry) -> None:
    """Print experiment status summary."""
    summary = registry.get_summary()

    print()
    print("=" * 60)
    print("Experiment Status Summary")
    print("=" * 60)
    print(f"  Total experiments:  {summary['total']}")
    print(f"  Completed:          {summary['completed']}")
    print(f"  Failed:             {summary['failed']}")
    print(f"  Pending:            {summary['pending']}")
    print(f"  Running:            {summary['running']}")
    print(f"  Needs re-run:       {summary['needs_rerun']}")

    if summary.get('by_error_category'):
        print()
        print("Failed by error category:")
        for category, count in sorted(summary['by_error_category'].items()):
            print(f"  {category:15} {count}")

    print("=" * 60)


def print_list(registry: ExperimentRegistry, status: str) -> None:
    """List experiments by status."""
    experiments = registry.get_experiments_by_status(status)

    print()
    print(f"Experiments with status '{status}': {len(experiments)}")
    print("-" * 60)

    if not experiments:
        print("  (none)")
        return

    for name in sorted(experiments):
        details = registry.get_experiment_details(name)
        if not details:
            print(f"  {name}")
            continue

        error_category = details.get('error_category', '')
        error_msg = details.get('error_msg', '')
        metrics = details.get('metrics', {})

        if status == "failed" and error_msg:
            # Truncate error message
            short_error = error_msg[:60] + "..." if len(error_msg) > 60 else error_msg
            print(f"  {name}")
            print(f"    Category: {error_category}")
            print(f"    Error: {short_error}")
        elif status == "completed" and metrics:
            acc = metrics.get('accuracy', 'N/A')
            acc_str = f"{acc:.1f}%" if isinstance(acc, (int, float)) else str(acc)
            print(f"  {name}: {acc_str}")
        else:
            print(f"  {name}")

    print("-" * 60)


def print_details(registry: ExperimentRegistry, name: str) -> None:
    """Print detailed information about an experiment."""
    details = registry.get_experiment_details(name)

    if not details:
        print(f"Experiment '{name}' not found in registry.")
        return

    print()
    print("=" * 60)
    print(f"Experiment: {name}")
    print("=" * 60)
    print(f"  Status:           {details.get('status', 'unknown')}")
    print(f"  GPU:              {details.get('gpu', 'N/A')}")
    print(f"  Started:          {format_timestamp(details.get('started_at'))}")
    print(f"  Completed:        {format_timestamp(details.get('completed_at'))}")
    print(f"  Retry count:      {details.get('retry_count', 0)}")
    print(f"  Marked for rerun: {details.get('marked_for_rerun', False)}")

    if details.get('result_file'):
        result_path = Path(details['result_file'])
        exists = "exists" if result_path.exists() else "MISSING"
        print(f"  Result file:      {details['result_file']} ({exists})")

    if details.get('metrics'):
        print()
        print("  Metrics:")
        for key, value in details['metrics'].items():
            if isinstance(value, float):
                print(f"    {key}: {value:.2f}")
            else:
                print(f"    {key}: {value}")

    if details.get('error_msg'):
        print()
        print("  Error details:")
        print(f"    Category:   {details.get('error_category', 'unknown')}")
        print(f"    Exit code:  {details.get('exit_code', 'N/A')}")
        print(f"    Message:")
        # Print error message with indentation
        for line in details['error_msg'].split('\n')[:10]:
            print(f"      {line}")

    print("=" * 60)


def mark_rerun(registry: ExperimentRegistry, name: str, reason: str = "") -> None:
    """Mark an experiment for re-run."""
    if registry.mark_for_rerun(name, reason):
        print(f"Marked '{name}' for re-run.")
    else:
        print(f"Experiment '{name}' not found in registry.")


def mark_all_failed_for_rerun(registry: ExperimentRegistry, reason: str = "batch re-run") -> None:
    """Mark all failed experiments for re-run."""
    marked = registry.mark_all_failed_for_rerun(reason)
    if marked:
        print(f"Marked {len(marked)} failed experiments for re-run:")
        for name in marked:
            print(f"  - {name}")
    else:
        print("No failed experiments to mark.")


def validate_results(registry: ExperimentRegistry) -> None:
    """Validate that result files exist for completed experiments."""
    validation = registry.validate_results()

    print()
    print("=" * 60)
    print("Result File Validation")
    print("=" * 60)
    print(f"  Valid (file exists):   {len(validation['valid'])}")
    print(f"  Missing (file gone):   {len(validation['missing'])}")

    if validation['missing']:
        print()
        print("Experiments with missing result files:")
        for name in validation['missing']:
            print(f"  - {name}")
        print()
        print("Consider marking these for re-run with:")
        print("  python scripts/experiment_manager.py --mark-rerun <name>")

    print("=" * 60)


def rebuild_registry(registry: ExperimentRegistry, runs_dir: Path) -> None:
    """Rebuild registry from existing result files."""
    print(f"Scanning {runs_dir} for result files...")
    discovered = registry.rebuild_from_results(runs_dir)
    print(f"Discovered {discovered} experiments from existing result files.")


def list_recent_failures(registry: ExperimentRegistry, limit: int = 10) -> None:
    """List recent failed experiments with their errors."""
    failed = registry.get_experiments_by_status("failed")

    if not failed:
        print("No failed experiments.")
        return

    print()
    print("=" * 60)
    print(f"Recent Failed Experiments (showing up to {limit})")
    print("=" * 60)

    # Sort by completed_at (most recent first)
    failed_with_time = []
    for name in failed:
        details = registry.get_experiment_details(name)
        completed_at = details.get('completed_at', '') if details else ''
        failed_with_time.append((name, completed_at, details))

    failed_with_time.sort(key=lambda x: x[1] or '', reverse=True)

    for name, completed_at, details in failed_with_time[:limit]:
        if not details:
            print(f"\n{name}: (no details)")
            continue

        print(f"\n{name}")
        print(f"  Time: {format_timestamp(completed_at)}")
        print(f"  Category: {details.get('error_category', 'unknown')}")
        error_msg = details.get('error_msg', 'No error message')
        # Show first 100 chars of error
        short_error = error_msg[:100] + "..." if len(error_msg) > 100 else error_msg
        print(f"  Error: {short_error}")

    print()
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Experiment Manager CLI for analyzing results and preparing re-runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/experiment_manager.py --status
  python scripts/experiment_manager.py --list failed
  python scripts/experiment_manager.py --details flow_matching_arc_easy_seed42
  python scripts/experiment_manager.py --mark-rerun flow_matching_arc_easy_seed42
  python scripts/experiment_manager.py --mark-rerun-all-failed
  python scripts/experiment_manager.py --validate
  python scripts/experiment_manager.py --rebuild-registry
        """
    )

    parser.add_argument(
        "--registry", "-r",
        type=Path,
        default=Path("runs/experiment_registry.json"),
        help="Path to registry file (default: runs/experiment_registry.json)"
    )

    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Show experiment status summary"
    )

    parser.add_argument(
        "--list", "-l",
        choices=["completed", "failed", "pending", "running"],
        help="List experiments by status"
    )

    parser.add_argument(
        "--details", "-d",
        type=str,
        metavar="NAME",
        help="Show detailed info for an experiment"
    )

    parser.add_argument(
        "--mark-rerun", "-m",
        type=str,
        metavar="NAME",
        help="Mark an experiment for re-run"
    )

    parser.add_argument(
        "--mark-rerun-all-failed",
        action="store_true",
        help="Mark all failed experiments for re-run"
    )

    parser.add_argument(
        "--validate", "-v",
        action="store_true",
        help="Validate that result files exist"
    )

    parser.add_argument(
        "--rebuild-registry",
        action="store_true",
        help="Rebuild registry from existing result files"
    )

    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Directory containing run outputs (for --rebuild-registry)"
    )

    parser.add_argument(
        "--recent-failures",
        type=int,
        nargs="?",
        const=10,
        metavar="N",
        help="Show N most recent failures (default: 10)"
    )

    parser.add_argument(
        "--reason",
        type=str,
        default="",
        help="Reason for re-run (used with --mark-rerun)"
    )

    args = parser.parse_args()

    # Create registry instance
    registry = ExperimentRegistry(args.registry)

    # Handle commands
    if args.status:
        print_status(registry)
    elif args.list:
        print_list(registry, args.list)
    elif args.details:
        print_details(registry, args.details)
    elif args.mark_rerun:
        mark_rerun(registry, args.mark_rerun, args.reason)
    elif args.mark_rerun_all_failed:
        mark_all_failed_for_rerun(registry, args.reason or "batch re-run")
    elif args.validate:
        validate_results(registry)
    elif args.rebuild_registry:
        rebuild_registry(registry, args.runs_dir)
    elif args.recent_failures is not None:
        list_recent_failures(registry, args.recent_failures)
    else:
        # Default: show status
        print_status(registry)


if __name__ == "__main__":
    main()
