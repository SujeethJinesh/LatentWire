#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path


REQUIRED_MODULES = [
    "rosetta",
    "torch",
    "transformers",
    "datasets",
    "huggingface_hub",
    "yaml",
    "numpy",
    "tqdm",
]


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def run_cmd(cmd, cwd=None, env=None, log_file=None):
    cmd_str = " ".join(cmd)
    if log_file:
        log_file.write(f"\n$ {cmd_str}\n")
        log_file.flush()
    print(f"$ {cmd_str}")
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        if log_file:
            log_file.write(line)
    ret = proc.wait()
    if ret != 0:
        die(f"Command failed ({ret}): {cmd_str}")


def check_gpu():
    if shutil.which("nvidia-smi") is None:
        die("nvidia-smi not found. Run this script on a GPU node.")
    res = subprocess.run(["nvidia-smi", "-L"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if res.returncode != 0:
        die("No GPUs detected by nvidia-smi. Run this script on a GPU node.")


def find_conda_exe():
    return os.environ.get("CONDA_EXE") or shutil.which("conda")


def ensure_env(env_name, project_root, args):
    current_env = os.environ.get("CONDA_DEFAULT_ENV")
    if current_env == env_name:
        return

    conda_exe = find_conda_exe()
    if conda_exe is None:
        die("conda not found in PATH. Load your conda module and retry.")

    if args.no_reexec:
        die(f"Not running inside conda env '{env_name}'. Activate it and retry.")

    res = subprocess.run([conda_exe, "env", "list"], capture_output=True, text=True)
    if env_name not in res.stdout:
        print(f"Conda env '{env_name}' not found. Creating it now...")
        run_cmd([conda_exe, "create", "-n", env_name, "python=3.10", "-y"])

    script_path = Path(__file__).resolve()
    cmd = [
        conda_exe, "run", "-n", env_name, "python", str(script_path),
        "--project-root", str(project_root),
        "--env", env_name,
        "--no-reexec",
    ]
    if args.skip_gpu_check:
        cmd.append("--skip-gpu-check")
    print("Re-running inside conda env:", " ".join(cmd))
    subprocess.check_call(cmd)
    sys.exit(0)


def ensure_installed(c2c_root, log_file=None):
    try:
        import importlib
        for mod in REQUIRED_MODULES:
            importlib.import_module(mod)
        print("Dependencies already installed; skipping pip install.")
        return
    except Exception:
        pass

    run_cmd([sys.executable, "-m", "pip", "install", "-e", "."], cwd=c2c_root, log_file=log_file)
    run_cmd([sys.executable, "-m", "pip", "install", "-e", ".[training,evaluation]"], cwd=c2c_root, log_file=log_file)


def collect_env_info():
    info = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "hostname": socket.gethostname(),
        "hf_home": os.environ.get("HF_HOME"),
        "transformers_cache": os.environ.get("TRANSFORMERS_CACHE"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi_path": shutil.which("nvidia-smi"),
    }

    def add_mod(name):
        try:
            mod = __import__(name)
            info[f"{name}_version"] = getattr(mod, "__version__", None)
            info[f"{name}_path"] = getattr(mod, "__file__", None)
        except Exception as exc:
            info[f"{name}_error"] = str(exc)

    for mod in ["torch", "transformers", "datasets", "huggingface_hub", "yaml", "numpy", "tqdm", "rosetta"]:
        add_mod(mod)

    try:
        import torch
        info["torch_cuda_available"] = torch.cuda.is_available()
        info["torch_cuda_device_count"] = torch.cuda.device_count()
    except Exception as exc:
        info["torch_cuda_error"] = str(exc)

    return info


def main():
    parser = argparse.ArgumentParser(description="Run Step 0 C2C baselines on a GPU node.")
    parser.add_argument(
        "--project-root",
        default=os.getcwd(),
        help="Path to LatentWire repo root (default: current working directory)",
    )
    parser.add_argument("--env", default="rosetta", help="Conda env name to use")
    parser.add_argument("--skip-gpu-check", action="store_true", help="Skip GPU detection")
    parser.add_argument("--prep-only", action="store_true", help="Run setup + downloads only; skip evaluation")
    parser.add_argument("--no-reexec", action="store_true", help="Internal flag to avoid re-exec loops")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    if not project_root.is_dir():
        die(f"PROJECT_ROOT not found: {project_root}")

    if not args.skip_gpu_check and not args.prep_only:
        check_gpu()

    ensure_env(args.env, project_root, args)

    os.environ.setdefault("HF_HOME", f"/scratch/m000066/{os.environ.get('USER','user')}/.cache/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(os.environ["HF_HOME"], "transformers"))
    os.environ.setdefault("WANDB_DISABLED", "true")

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    run_root = project_root / "data" / "step_0_baselines" / run_tag
    run_root.mkdir(parents=True, exist_ok=True)
    for sub in ("configs", "logs", "results", "manifests"):
        (run_root / sub).mkdir(parents=True, exist_ok=True)

    log_path = run_root / "logs" / "step0.log"
    with log_path.open("a", encoding="utf-8") as log_file:
        env_info = collect_env_info()
        print("Environment info:", json.dumps(env_info, indent=2))
        env_path = run_root / "manifests" / "env_info.json"
        env_path.write_text(json.dumps(env_info, indent=2))

        run_cmd(
            ["git", "submodule", "update", "--init", "--recursive", "quantization/C2C"],
            cwd=str(project_root),
            log_file=log_file,
        )

        c2c_root = project_root / "quantization" / "C2C"
        if not c2c_root.is_dir():
            die(
                f"C2C submodule missing at {c2c_root}. "
                "Run from the LatentWire repo root or pass --project-root."
            )

        ensure_installed(str(c2c_root), log_file=log_file)

        from huggingface_hub import snapshot_download
        import yaml

        ckpt_root = Path(
            os.environ.get("C2C_CKPT_ROOT", f"/scratch/m000066/{os.environ.get('USER','user')}/c2c_checkpoints")
        )
        ckpt_root.mkdir(parents=True, exist_ok=True)

        repo_id = "nics-efc/C2C_Fuser"
        pattern = "qwen3_0.6b+qwen2.5_0.5b_Fuser/*"
        local_dir = ckpt_root / "C2C_Fuser"
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            allow_patterns=[pattern],
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )

        def git_rev(path):
            try:
                out = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"])
                return out.decode().strip()
            except Exception:
                return None

        manifest = {
            "repo_id": repo_id,
            "allow_patterns": [pattern],
            "snapshot_path": snapshot_path,
            "checkpoint_dir": str(local_dir / "qwen3_0.6b+qwen2.5_0.5b_Fuser" / "final"),
            "latentwire_commit": git_rev(project_root),
            "c2c_commit": git_rev(c2c_root),
            "hostname": socket.gethostname(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prep_only": bool(args.prep_only),
        }
        manifest_path = run_root / "manifests" / "step_0_checkpoint_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print("Wrote manifest:", manifest_path)

        # Copy eval configs
        (run_root / "configs" / "openbookqa.yaml").write_text(
            (c2c_root / "recipe" / "eval_recipe" / "unified_eval.yaml").read_text()
        )
        (run_root / "configs" / "arc_c.yaml").write_text(
            (c2c_root / "recipe" / "eval_recipe" / "unified_eval.yaml").read_text()
        )

        def patch(cfg_path, dataset, out_dir):
            cfg = yaml.safe_load(Path(cfg_path).read_text())
            cfg["model"]["rosetta_config"]["checkpoints_dir"] = manifest["checkpoint_dir"]
            cfg["output"]["output_dir"] = str(out_dir)
            cfg["eval"]["dataset"] = dataset
            Path(cfg_path).write_text(yaml.safe_dump(cfg, sort_keys=False))

        patch(run_root / "configs/openbookqa.yaml", "openbookqa", run_root / "results" / "openbookqa")
        patch(run_root / "configs/arc_c.yaml", "ai2-arc", run_root / "results" / "arc_c")

        if args.prep_only:
            print(f"Prep-only complete. Run on GPU to execute evals. Run folder: {run_root}")
            return

        if not args.skip_gpu_check:
            check_gpu()

        run_cmd(
            [sys.executable, "script/evaluation/unified_evaluator.py", "--config", str(run_root / "configs/openbookqa.yaml")],
            cwd=str(c2c_root),
            env=os.environ.copy(),
            log_file=log_file,
        )
        run_cmd(
            [sys.executable, "script/evaluation/unified_evaluator.py", "--config", str(run_root / "configs/arc_c.yaml")],
            cwd=str(c2c_root),
            env=os.environ.copy(),
            log_file=log_file,
        )

    print(f"Step 0 complete. Results in: {run_root}")


if __name__ == "__main__":
    main()
