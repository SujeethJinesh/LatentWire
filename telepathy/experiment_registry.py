"""
Experiment Registry for tracking experiment status across SLURM jobs.

Provides atomic file operations, concurrent access safety, and Git-friendly
JSON storage for experiment tracking and resumption.
"""

import json
import os
import fcntl
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any


class ExperimentRegistry:
    """Unified experiment tracking across SLURM jobs."""

    # Error categories for automatic classification
    ERROR_CATEGORIES = {
        "oom": ["CUDA out of memory", "OutOfMemoryError", "out of memory"],
        "nan": ["NaN loss", "loss is nan", "nan detected", "NaN"],
        "import": ["ImportError", "ModuleNotFoundError", "No module named"],
        "timeout": ["SIGTERM", "timeout", "TimeoutError", "Timeout"],
        "keyerror": ["KeyError:"],
        "filenotfound": ["FileNotFoundError", "No such file", "not found"],
    }

    MAX_RETRIES = 3

    def __init__(self, registry_path: Optional[Path] = None):
        """
        Initialize the experiment registry.

        Args:
            registry_path: Path to the registry JSON file. Defaults to runs/experiment_registry.json
        """
        if registry_path is None:
            registry_path = Path("runs/experiment_registry.json")
        self.registry_path = Path(registry_path)
        self._ensure_registry_exists()

    def _ensure_registry_exists(self) -> None:
        """Create the registry file if it doesn't exist."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self._save({
                "meta": {
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "job_id": os.environ.get("SLURM_JOB_ID", "local"),
                },
                "experiments": {}
            })

    def _load(self) -> Dict[str, Any]:
        """Load the registry from disk with file locking."""
        try:
            with open(self.registry_path, 'r') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    return json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except (json.JSONDecodeError, FileNotFoundError):
            return {
                "meta": {
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "job_id": os.environ.get("SLURM_JOB_ID", "local"),
                },
                "experiments": {}
            }

    def _save(self, data: Dict[str, Any]) -> None:
        """Save the registry atomically with file locking."""
        data["meta"]["last_updated"] = datetime.now().isoformat()
        data["meta"]["job_id"] = os.environ.get("SLURM_JOB_ID", data["meta"].get("job_id", "local"))

        # Atomic write: write to temp file, then rename
        temp_fd, temp_path = tempfile.mkstemp(
            dir=self.registry_path.parent,
            prefix=".registry_",
            suffix=".tmp"
        )
        try:
            with os.fdopen(temp_fd, 'w') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(data, f, indent=2)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            os.replace(temp_path, self.registry_path)
        except Exception:
            # Clean up temp file on failure
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    def _classify_error(self, error_msg: str) -> str:
        """Classify an error message into a category."""
        if not error_msg:
            return "unknown"
        error_lower = error_msg.lower()
        for category, patterns in self.ERROR_CATEGORIES.items():
            for pattern in patterns:
                if pattern.lower() in error_lower:
                    return category
        return "unknown"

    def should_run(self, name: str) -> Tuple[bool, str]:
        """
        Check if an experiment should run.

        Args:
            name: Experiment name

        Returns:
            Tuple of (should_run: bool, reason: str)
        """
        data = self._load()
        experiments = data.get("experiments", {})

        if name not in experiments:
            return True, "new experiment"

        exp = experiments[name]
        status = exp.get("status", "pending")

        if status == "completed" and not exp.get("marked_for_rerun", False):
            return False, "already completed"

        if status == "running":
            # Check if it's a stale run (older than 24 hours)
            started_at = exp.get("started_at")
            if started_at:
                try:
                    start_time = datetime.fromisoformat(started_at)
                    elapsed = datetime.now() - start_time
                    if elapsed.total_seconds() < 24 * 3600:
                        return False, f"currently running (started {elapsed.total_seconds() / 3600:.1f}h ago)"
                except ValueError:
                    pass
            return True, "stale run (>24h)"

        if status == "failed":
            retry_count = exp.get("retry_count", 0)
            if retry_count >= self.MAX_RETRIES:
                return False, f"max retries ({self.MAX_RETRIES}) exceeded"
            return True, f"retry {retry_count + 1}/{self.MAX_RETRIES}"

        if exp.get("marked_for_rerun", False):
            return True, "marked for re-run"

        if status == "pending":
            return True, "pending experiment"

        return True, "unknown status"

    def start_experiment(self, name: str, gpu: int = 0) -> None:
        """
        Mark an experiment as running.

        Args:
            name: Experiment name
            gpu: GPU index being used
        """
        data = self._load()
        experiments = data.setdefault("experiments", {})

        # Preserve retry count if exists
        retry_count = experiments.get(name, {}).get("retry_count", 0)
        if experiments.get(name, {}).get("status") == "failed":
            retry_count += 1

        experiments[name] = {
            "status": "running",
            "result_file": None,
            "error_msg": None,
            "exit_code": None,
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
            "gpu": gpu,
            "metrics": {},
            "marked_for_rerun": False,
            "retry_count": retry_count,
        }
        self._save(data)

    def complete_experiment(
        self,
        name: str,
        result_file: Optional[Path] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Mark an experiment as completed.

        Args:
            name: Experiment name
            result_file: Path to the result file
            metrics: Dictionary of metrics (e.g., {"accuracy": 45.2})
        """
        data = self._load()
        experiments = data.setdefault("experiments", {})

        if name not in experiments:
            experiments[name] = {}

        exp = experiments[name]
        exp["status"] = "completed"
        exp["result_file"] = str(result_file) if result_file else None
        exp["error_msg"] = None
        exp["exit_code"] = 0
        exp["completed_at"] = datetime.now().isoformat()
        exp["marked_for_rerun"] = False
        if metrics:
            exp["metrics"] = metrics

        self._save(data)

    def fail_experiment(
        self,
        name: str,
        error_msg: str,
        exit_code: int = 1
    ) -> None:
        """
        Mark an experiment as failed.

        Args:
            name: Experiment name
            error_msg: Error message describing the failure
            exit_code: Exit code of the failed process
        """
        data = self._load()
        experiments = data.setdefault("experiments", {})

        if name not in experiments:
            experiments[name] = {"retry_count": 0}

        exp = experiments[name]
        exp["status"] = "failed"
        exp["error_msg"] = error_msg
        exp["error_category"] = self._classify_error(error_msg)
        exp["exit_code"] = exit_code
        exp["completed_at"] = datetime.now().isoformat()

        self._save(data)

    def mark_for_rerun(self, name: str, reason: str = "") -> bool:
        """
        Mark an experiment for re-run.

        Args:
            name: Experiment name
            reason: Reason for re-run

        Returns:
            True if experiment was found and marked, False otherwise
        """
        data = self._load()
        experiments = data.get("experiments", {})

        if name not in experiments:
            return False

        experiments[name]["marked_for_rerun"] = True
        experiments[name]["rerun_reason"] = reason
        experiments[name]["retry_count"] = 0  # Reset retry count
        self._save(data)
        return True

    def mark_all_failed_for_rerun(self, reason: str = "batch re-run") -> List[str]:
        """
        Mark all failed experiments for re-run.

        Args:
            reason: Reason for re-run

        Returns:
            List of experiment names that were marked
        """
        data = self._load()
        experiments = data.get("experiments", {})
        marked = []

        for name, exp in experiments.items():
            if exp.get("status") == "failed":
                exp["marked_for_rerun"] = True
                exp["rerun_reason"] = reason
                exp["retry_count"] = 0
                marked.append(name)

        if marked:
            self._save(data)
        return marked

    def get_experiments_by_status(self, status: str) -> List[str]:
        """
        Get all experiments with a given status.

        Args:
            status: One of "pending", "running", "completed", "failed"

        Returns:
            List of experiment names
        """
        data = self._load()
        experiments = data.get("experiments", {})
        return [
            name for name, exp in experiments.items()
            if exp.get("status") == status
        ]

    def get_experiment_details(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about an experiment.

        Args:
            name: Experiment name

        Returns:
            Experiment details dict or None if not found
        """
        data = self._load()
        return data.get("experiments", {}).get(name)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of experiment statuses.

        Returns:
            Dictionary with status counts and lists
        """
        data = self._load()
        experiments = data.get("experiments", {})

        summary = {
            "total": len(experiments),
            "completed": 0,
            "failed": 0,
            "pending": 0,
            "running": 0,
            "needs_rerun": 0,
            "by_error_category": {},
        }

        for name, exp in experiments.items():
            status = exp.get("status", "pending")
            if status in summary:
                summary[status] += 1

            if exp.get("marked_for_rerun", False):
                summary["needs_rerun"] += 1

            if status == "failed":
                category = exp.get("error_category", "unknown")
                summary["by_error_category"][category] = \
                    summary["by_error_category"].get(category, 0) + 1

        return summary

    def validate_results(self) -> Dict[str, List[str]]:
        """
        Validate that result files exist for completed experiments.

        Returns:
            Dictionary with "valid" and "missing" lists
        """
        data = self._load()
        experiments = data.get("experiments", {})

        valid = []
        missing = []

        for name, exp in experiments.items():
            if exp.get("status") != "completed":
                continue

            result_file = exp.get("result_file")
            if result_file and Path(result_file).exists():
                valid.append(name)
            else:
                missing.append(name)

        return {"valid": valid, "missing": missing}

    def rebuild_from_results(self, runs_dir: Path = Path("runs")) -> int:
        """
        Rebuild registry from existing result files.

        Args:
            runs_dir: Directory containing run outputs

        Returns:
            Number of experiments discovered
        """
        data = self._load()
        experiments = data.setdefault("experiments", {})
        discovered = 0

        # Find all result JSON files
        for result_file in runs_dir.rglob("*_results.json"):
            try:
                with open(result_file) as f:
                    result_data = json.load(f)

                # Extract experiment name from directory structure
                rel_path = result_file.relative_to(runs_dir)
                parts = rel_path.parts

                # Try to construct experiment name
                if len(parts) >= 2:
                    name = parts[-2]  # Directory name before the file
                else:
                    name = result_file.stem.replace("_results", "")

                if name not in experiments:
                    experiments[name] = {
                        "status": "completed",
                        "result_file": str(result_file),
                        "error_msg": None,
                        "exit_code": 0,
                        "started_at": None,
                        "completed_at": result_file.stat().st_mtime,
                        "gpu": None,
                        "metrics": result_data.get("final_results", {}),
                        "marked_for_rerun": False,
                        "retry_count": 0,
                    }
                    discovered += 1
            except (json.JSONDecodeError, KeyError, OSError):
                continue

        if discovered > 0:
            self._save(data)

        return discovered

    def to_json(self) -> str:
        """Export registry as formatted JSON string."""
        return json.dumps(self._load(), indent=2)


# CLI interface for bash integration
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment Registry CLI")
    parser.add_argument("--registry", type=Path, default=Path("runs/experiment_registry.json"),
                        help="Path to registry file")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # should_run command
    should_run_parser = subparsers.add_parser("should_run", help="Check if experiment should run")
    should_run_parser.add_argument("name", help="Experiment name")

    # start command
    start_parser = subparsers.add_parser("start", help="Mark experiment as running")
    start_parser.add_argument("name", help="Experiment name")
    start_parser.add_argument("--gpu", type=int, default=0, help="GPU index")

    # complete command
    complete_parser = subparsers.add_parser("complete", help="Mark experiment as completed")
    complete_parser.add_argument("name", help="Experiment name")
    complete_parser.add_argument("--result-file", type=Path, help="Path to result file")
    complete_parser.add_argument("--metrics", type=str, help="JSON metrics string")

    # fail command
    fail_parser = subparsers.add_parser("fail", help="Mark experiment as failed")
    fail_parser.add_argument("name", help="Experiment name")
    fail_parser.add_argument("--error", type=str, default="", help="Error message")
    fail_parser.add_argument("--exit-code", type=int, default=1, help="Exit code")

    # summary command
    subparsers.add_parser("summary", help="Print summary")

    args = parser.parse_args()

    registry = ExperimentRegistry(args.registry)

    if args.command == "should_run":
        should_run, reason = registry.should_run(args.name)
        print(f"{'1' if should_run else '0'}|{reason}")

    elif args.command == "start":
        registry.start_experiment(args.name, args.gpu)
        print(f"Started: {args.name}")

    elif args.command == "complete":
        metrics = json.loads(args.metrics) if args.metrics else None
        registry.complete_experiment(args.name, args.result_file, metrics)
        print(f"Completed: {args.name}")

    elif args.command == "fail":
        registry.fail_experiment(args.name, args.error, args.exit_code)
        print(f"Failed: {args.name}")

    elif args.command == "summary":
        summary = registry.get_summary()
        print(json.dumps(summary, indent=2))
