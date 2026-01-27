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


REQUIRED_MODULES_GPU = [
    "rosetta",
    "torch",
    "transformers",
    "datasets",
    "huggingface_hub",
    "yaml",
    "numpy",
    "tqdm",
]

REQUIRED_MODULES_LOCAL = [
    "rosetta",
    "torch",
    "transformers",
    "huggingface_hub",
    "yaml",
    "numpy",
    "tqdm",
]
LOCAL_TORCH_VERSION = "2.2.2"


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


def find_conda_env_prefix(conda_exe, env_name):
    try:
        res = subprocess.run(
            [conda_exe, "env", "list", "--json"], capture_output=True, text=True, check=True
        )
        data = json.loads(res.stdout)
        for env_path in data.get("envs", []):
            if Path(env_path).name == env_name:
                return env_path
    except Exception:
        return None
    return None


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
    forwarded = [arg for arg in sys.argv[1:] if arg != "--no-reexec"]
    env_prefix = find_conda_env_prefix(conda_exe, env_name)
    python_exe = None
    if env_prefix:
        python_exe = Path(env_prefix) / "bin" / "python"
        if not python_exe.exists():
            python_exe = None
    python_cmd = str(python_exe) if python_exe else "python"
    cmd = [
        conda_exe, "run", "-n", env_name, python_cmd, str(script_path),
        "--project-root", str(project_root),
    ] + forwarded + ["--no-reexec"]
    print("Re-running inside conda env:", " ".join(cmd))
    subprocess.check_call(cmd)
    sys.exit(0)


def _module_in_repo(module, repo_root: Path) -> bool:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return False
    try:
        return Path(module_file).resolve().is_relative_to(repo_root.resolve())
    except Exception:
        return str(repo_root.resolve()) in str(Path(module_file).resolve())


def ensure_installed(
    c2c_root,
    log_file=None,
    required_modules=None,
    extras=None,
    no_deps=False,
    extra_pip=None,
):
    required_modules = required_modules or REQUIRED_MODULES_GPU
    try:
        import importlib
        imported = {mod: importlib.import_module(mod) for mod in required_modules}
        rosetta_mod = imported.get("rosetta")
        if rosetta_mod and not _module_in_repo(rosetta_mod, Path(c2c_root)):
            raise RuntimeError(f"rosetta resolved to {rosetta_mod.__file__}, expected under {c2c_root}")
        print("Dependencies already installed; skipping pip install.")
        return
    except Exception as exc:
        if log_file:
            log_file.write(f"Dependency check failed or mismatched: {exc}\n")
        pass

    pip_cmd = [sys.executable, "-m", "pip", "install", "-e", "."]
    if no_deps:
        pip_cmd.append("--no-deps")
    run_cmd(pip_cmd, cwd=c2c_root, log_file=log_file)
    if extras:
        run_cmd(
            [sys.executable, "-m", "pip", "install", "-e", f".[{extras}]"],
            cwd=c2c_root,
            log_file=log_file,
        )
    if extra_pip:
        run_cmd([sys.executable, "-m", "pip", "install"] + list(extra_pip), log_file=log_file)


def collect_model_stats(model_name):
    if not model_name:
        return None, "missing_model_name"
    try:
        from transformers import AutoConfig
        local_only = os.environ.get("TRANSFORMERS_OFFLINE", "").lower() in ("1", "true", "yes")
        cfg = AutoConfig.from_pretrained(model_name, local_files_only=local_only)
        hidden_size = getattr(cfg, "hidden_size", None)
        num_layers = getattr(cfg, "num_hidden_layers", None)
        num_heads = getattr(cfg, "num_attention_heads", None)
        num_kv_heads = getattr(cfg, "num_key_value_heads", None) or num_heads
        if not all([hidden_size, num_layers, num_heads, num_kv_heads]):
            return None, "missing_model_stats"
        head_dim = int(hidden_size) // int(num_heads)
        return {
            "hidden_size": int(hidden_size),
            "num_layers": int(num_layers),
            "num_heads": int(num_heads),
            "num_kv_heads": int(num_kv_heads),
            "head_dim": int(head_dim),
        }, None
    except Exception as exc:
        return None, str(exc)


def collect_env_info():
    info = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "hostname": socket.gethostname(),
        "hf_home": os.environ.get("HF_HOME"),
        "transformers_cache": os.environ.get("TRANSFORMERS_CACHE"),
        "hf_datasets_cache": os.environ.get("HF_DATASETS_CACHE"),
        "hf_hub_cache": os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE"),
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


def ensure_numpy_compat(log_file=None):
    try:
        import numpy as np
        major = int(np.__version__.split(".", 1)[0])
        if major < 2:
            return
    except Exception:
        pass
    run_cmd([sys.executable, "-m", "pip", "install", "numpy<2"], log_file=log_file)
    os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve())] + sys.argv[1:])


def resolve_device_for_mode(mode):
    import torch
    if mode == "local":
        if not torch.backends.mps.is_available():
            die("MPS not available. Local mode expects a Mac with MPS support.")
        return torch.device("mps")
    if not torch.cuda.is_available():
        die("CUDA not available. GPU mode expects a CUDA-capable node.")
    return torch.device("cuda")


def run_local_smoke_test(project_root, data_root, args):
    if args.prep_only:
        print("Note: --prep-only ignored in local mode.")

    run_tag = args.run_tag or time.strftime("local_smoke_%Y%m%d_%H%M%S")
    if args.output_dir:
        run_root = Path(args.output_dir).expanduser().resolve()
    else:
        run_root = data_root / "local_smoke_tests" / run_tag
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "manifests").mkdir(parents=True, exist_ok=True)

    log_path = run_root / "local_smoke.log"
    with log_path.open("a", encoding="utf-8") as log_file:
        ensure_numpy_compat(log_file=log_file)
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

        ensure_installed(
            str(c2c_root),
            log_file=log_file,
            required_modules=REQUIRED_MODULES_LOCAL,
            extras=None,
            no_deps=True,
            extra_pip=[
                f"torch=={LOCAL_TORCH_VERSION}",
                "transformers==4.52.4",
                "huggingface_hub",
                "pyyaml",
                "numpy<2",
                "tqdm",
                "tiktoken",
                "sentencepiece",
            ],
        )
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        if not hasattr(torch.nn, "RMSNorm"):
            class RMSNorm(torch.nn.Module):
                def __init__(self, hidden_size, eps=1e-6, dtype=None):
                    super().__init__()
                    self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=dtype))
                    self.eps = eps

                def forward(self, x):
                    norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
                    return x * norm * self.weight

            torch.nn.RMSNorm = RMSNorm
        from rosetta.model.projector import load_projector
        from rosetta.model.wrapper import RosettaModel
        from rosetta.utils.evaluate import set_default_chat_template

        checkpoint_dir = args.checkpoint_dir
        if checkpoint_dir is None:
            ckpt_root = data_root / "local_smoke_tests" / "checkpoints"
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
            checkpoint_dir = str(local_dir / "qwen3_0.6b+qwen2.5_0.5b_Fuser" / "final")
            manifest = {
                "repo_id": repo_id,
                "allow_patterns": [pattern],
                "snapshot_path": snapshot_path,
                "checkpoint_dir": checkpoint_dir,
            }
            (run_root / "manifests" / "checkpoint_manifest.json").write_text(
                json.dumps(manifest, indent=2)
            )

        device = resolve_device_for_mode(args.mode)
        dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
        projector_dtype = torch.float16 if device.type == "mps" else dtype

        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        set_default_chat_template(tokenizer, args.base_model)
        base_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype).eval()
        teacher_model = AutoModelForCausalLM.from_pretrained(args.teacher_model, torch_dtype=dtype).eval()
        base_model.to(device)
        teacher_model.to(device)

        ckpt_path = Path(checkpoint_dir)
        projector_list = []
        for proj_json in sorted(ckpt_path.glob("projector_*.json")):
            if proj_json.name == "projector_config.json":
                continue
            proj = load_projector(str(proj_json))
            pt_path = proj_json.with_suffix(".pt")
            if pt_path.exists():
                state_dict = torch.load(pt_path, map_location="cpu")
                for key, value in state_dict.items():
                    if torch.is_tensor(value):
                        state_dict[key] = value.to(projector_dtype)
                proj.load_state_dict(state_dict, strict=False)
            proj = proj.to(device=device, dtype=projector_dtype)
            projector_list.append(proj)

        rosetta = RosettaModel(
            model_list=[base_model, teacher_model],
            base_model_idx=0,
            projector_list=projector_list,
            include_response=True,
        ).to(device).eval()

        proj_cfg_path = ckpt_path / "projector_config.json"
        rosetta.load_projector_config(str(proj_cfg_path))

        from rosetta.utils.evaluate import build_prompt, extract_answer_from_content

        question = "What is 2+2?"
        choices = "A. 3\nB. 4\nC. 5\nD. 6\n"
        prompt_text = build_prompt(
            dataset="mmlu-redux",
            locale="",
            question=question,
            choices=choices,
            use_cot=False,
            use_template=True,
        )
        prompt = [{"role": "user", "content": prompt_text}]
        input_text = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        position_ids = inputs["attention_mask"].long().cumsum(-1) - 1

        instruction_index = (
            torch.tensor([1, 0], dtype=torch.long)
            .repeat(inputs["input_ids"].shape[1] - 1, 1)
            .unsqueeze(0)
            .to(device)
        )
        label_index = torch.tensor([-1, 0], dtype=torch.long).repeat(1, 1).unsqueeze(0).to(device)
        kv_cache_index = [instruction_index, label_index]

        with torch.no_grad():
            outputs = rosetta.generate(
                **inputs,
                position_ids=position_ids,
                kv_cache_index=kv_cache_index,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
            )

        output_ids = outputs[0]
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[input_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        full_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        print("C2C smoke test output:")
        print(generated_text)

        def strip_think_tags(text_value):
            return text_value.replace("<think>", "").replace("</think>", "").strip()

        extracted_answer = extract_answer_from_content(generated_text)
        expected_answer = "B"

        meta = {
            "device": str(device),
            "dtype": str(dtype),
            "base_model": args.base_model,
            "teacher_model": args.teacher_model,
            "checkpoint_dir": checkpoint_dir,
            "max_new_tokens": args.max_new_tokens,
            "output_dir": str(run_root),
            "input_length": input_len,
            "generated_length": int(generated_ids.shape[0]),
            "prompt_text": prompt_text,
            "generated_text": generated_text,
            "extracted_answer": extracted_answer,
            "expected_answer": expected_answer,
            "matches_expected": (extracted_answer == expected_answer)
            if extracted_answer is not None
            else (expected_answer in strip_think_tags(generated_text)),
        }
        (run_root / "output.txt").write_text(generated_text + "\n", encoding="utf-8")
        (run_root / "output_full.txt").write_text(full_text + "\n", encoding="utf-8")
        (run_root / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"Local smoke test complete. Output in: {run_root}")


def main():
    parser = argparse.ArgumentParser(description="Run Step 0 C2C baselines (GPU default, local smoke supported).")
    parser.add_argument(
        "--project-root",
        default=os.getcwd(),
        help="Path to LatentWire repo root (default: current working directory)",
    )
    parser.add_argument(
        "--mode",
        choices=["gpu", "local"],
        default="gpu",
        help="gpu for full evals (default), local for Mac smoke test",
    )
    parser.add_argument("--env", default="rosetta", help="Conda env name to use")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Local mode generation length")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B", help="Local mode base model")
    parser.add_argument("--teacher-model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Local mode teacher model")
    parser.add_argument("--checkpoint-dir", default=None, help="Local mode projector checkpoint dir")
    parser.add_argument("--run-tag", default=None, help="Override run tag")
    parser.add_argument("--output-dir", default=None, help="Local mode output dir override")
    parser.add_argument("--hf-cache", default=None, help="Optional HF cache root")
    parser.add_argument("--skip-gpu-check", action="store_true", help="Skip GPU detection")
    parser.add_argument("--prep-only", action="store_true", help="Run setup + downloads only; skip evaluation")
    parser.add_argument("--no-reexec", action="store_true", help="Internal flag to avoid re-exec loops")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    if not project_root.is_dir():
        die(f"PROJECT_ROOT not found: {project_root}")
    quant_root = project_root / "quantization"
    if not quant_root.is_dir():
        die(f"quantization/ folder not found under {project_root}")
    data_root = quant_root / "data"

    if args.mode == "gpu" and not args.skip_gpu_check and not args.prep_only:
        check_gpu()

    ensure_env(args.env, project_root, args)

    if args.hf_cache:
        os.environ["HF_HOME"] = args.hf_cache
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(args.hf_cache, "transformers")
    elif args.mode == "local":
        local_cache = data_root / "local_smoke_tests" / "hf_cache"
        os.environ.setdefault("HF_HOME", str(local_cache))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(local_cache / "transformers"))
    else:
        os.environ.setdefault("HF_HOME", f"/scratch/m000066/{os.environ.get('USER','user')}/.cache/huggingface")
        os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(os.environ["HF_HOME"], "transformers"))
    os.environ.setdefault("WANDB_DISABLED", "true")

    if args.mode == "local":
        run_local_smoke_test(project_root, data_root, args)
        return

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    if args.run_tag:
        run_tag = args.run_tag
    run_root = data_root / "step_0_baselines" / run_tag
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

        ensure_installed(
            str(c2c_root),
            log_file=log_file,
            required_modules=REQUIRED_MODULES_GPU,
            extras="training,evaluation",
        )

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

        eval_cfg = yaml.safe_load((c2c_root / "recipe" / "eval_recipe" / "unified_eval.yaml").read_text())
        rosetta_cfg = eval_cfg.get("model", {}).get("rosetta_config", {})
        base_model = rosetta_cfg.get("base_model")
        teacher_model = rosetta_cfg.get("teacher_model")
        base_model_stats, base_model_stats_error = collect_model_stats(base_model)

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
            "base_model": base_model,
            "teacher_model": teacher_model,
            "base_model_stats": base_model_stats,
            "base_model_stats_error": base_model_stats_error,
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
