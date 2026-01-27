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
    "datasets",
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
        for mod in required_modules:
            importlib.import_module(mod)
        print("Dependencies already installed; skipping pip install.")
        return
    except Exception:
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


def create_segmented_kv_cache_index(instruction_length, response_length, proportion, order_mode, device):
    import torch

    if proportion < 0.0 or proportion > 1.0:
        raise ValueError(f"proportion must be between 0.0 and 1.0, got {proportion}")
    if order_mode not in ["front", "back"]:
        raise ValueError(f"order_mode must be 'front' or 'back', got '{order_mode}'")

    instruction_positive_length = int(instruction_length * proportion)
    instruction_negative_length = instruction_length - instruction_positive_length

    complete_sequence = []
    if order_mode == "front":
        complete_sequence.extend([[1, 0]] * instruction_positive_length)
        complete_sequence.extend([[-1, 0]] * instruction_negative_length)
    else:
        complete_sequence.extend([[-1, 0]] * instruction_negative_length)
        complete_sequence.extend([[1, 0]] * instruction_positive_length)
    complete_sequence.extend([[-1, 0]] * response_length)

    if not complete_sequence:
        return []

    segments = []
    current_segment_start = 0
    current_value = complete_sequence[0]
    for i in range(1, len(complete_sequence)):
        if complete_sequence[i] != current_value:
            segment_length = i - current_segment_start
            segment = torch.tensor(current_value, dtype=torch.long).repeat(segment_length, 1).unsqueeze(0).to(device)
            segments.append(segment)
            current_segment_start = i
            current_value = complete_sequence[i]

    segment_length = len(complete_sequence) - current_segment_start
    segment = torch.tensor(current_value, dtype=torch.long).repeat(segment_length, 1).unsqueeze(0).to(device)
    segments.append(segment)
    return segments


def _choices_from_example(example, question_key):
    question_text = example.get(question_key, "")
    raw_choices = example.get("choices")
    choices_texts = []
    if isinstance(raw_choices, dict):
        choices_texts = list(raw_choices.get("text", []))
    elif isinstance(raw_choices, list):
        for item in raw_choices:
            if isinstance(item, dict):
                choices_texts.append(str(item.get("text", "")))
            else:
                choices_texts.append(str(item))

    choices_str = ""
    for i, text in enumerate(choices_texts):
        choices_str += f"{chr(65+i)}. {text}\n"
    return question_text, choices_texts, choices_str


def build_prompt_from_example(example, dataset_name, use_cot=False, use_template=True):
    from rosetta.utils.evaluate import build_prompt

    if dataset_name == "openbookqa":
        question_text, choices_texts, choices_str = _choices_from_example(example, "question_stem")
    elif dataset_name == "ai2-arc":
        question_text, choices_texts, choices_str = _choices_from_example(example, "question")
    else:
        raise ValueError(f"Unsupported local dataset: {dataset_name}")

    prompt_text = build_prompt(
        dataset=dataset_name,
        locale="",
        question=question_text,
        choices=choices_str,
        use_cot=use_cot,
        use_template=use_template,
    )
    return prompt_text, question_text, choices_texts, choices_str


def parse_answer_from_example(example):
    answer_key = example.get("answerKey")
    if isinstance(answer_key, str) and answer_key in ["A", "B", "C", "D"]:
        return answer_key
    return None


def write_status(run_root, status, extra=None):
    if run_root is None:
        return
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / "manifests" / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text())
        except Exception:
            data = {}
    data["status"] = status
    if extra:
        data.update(extra)
    manifest_path.write_text(json.dumps(data, indent=2))


def cleanup_failed(run_root, args, reason):
    if run_root is None:
        return
    write_status(run_root, "failed", {"error": reason})
    if args.cleanup_failed and run_root.exists():
        shutil.rmtree(run_root)


def load_local_dataset(dataset_name):
    from datasets import load_dataset

    if dataset_name == "openbookqa":
        return load_dataset("allenai/openbookqa")["test"]
    if dataset_name == "ai2-arc":
        return load_dataset("allenai/ai2_arc", "ARC-Challenge")["test"]
    die(f"Unsupported local dataset: {dataset_name}")


def run_local_smoke_test(project_root, data_root, kv_quant_config, args, run_root):
    if args.prep_only:
        print("Note: --prep-only ignored in local mode.")

    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "manifests").mkdir(parents=True, exist_ok=True)

    log_path = run_root / "local_smoke.log"
    with log_path.open("a", encoding="utf-8") as log_file:
        ensure_numpy_compat(log_file=log_file)
        env_info = collect_env_info()
        print("Environment info:", json.dumps(env_info, indent=2))
        (run_root / "manifests" / "env_info.json").write_text(json.dumps(env_info, indent=2))
        write_status(
            run_root,
            "running",
            {
                "mode": "local",
                "kv_quant_config": kv_quant_config,
                "receiver_only": bool(args.local_receiver_only),
                "include_response": bool(args.include_response),
                "kv_cache_proportion": args.kv_cache_proportion,
                "kv_cache_order_mode": args.kv_cache_order_mode,
            },
        )

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
                "datasets",
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
        from rosetta.utils.evaluate import set_default_chat_template, extract_answer_from_content

        checkpoint_dir = args.checkpoint_dir
        if not args.local_receiver_only:
            if checkpoint_dir is None:
                ckpt_root = data_root / "step_1_kv_ptq" / "local_smoke_tests" / "checkpoints"
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
        base_model.to(device)

        teacher_model = None
        if not args.local_receiver_only:
            teacher_model = AutoModelForCausalLM.from_pretrained(args.teacher_model, torch_dtype=dtype).eval()
            teacher_model.to(device)

        rosetta = None
        if not args.local_receiver_only:
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
                include_response=args.include_response,
                kv_quant_config=kv_quant_config,
            ).to(device).eval()

            proj_cfg_path = ckpt_path / "projector_config.json"
            rosetta.load_projector_config(str(proj_cfg_path))

        dataset = load_local_dataset(args.local_dataset)
        dataset_size = len(dataset)
        if args.local_num_samples < 1:
            die("local_num_samples must be >= 1")
        if args.local_sample_index < 0 or args.local_sample_index >= dataset_size:
            die(
                f"Sample index {args.local_sample_index} out of range for {args.local_dataset} "
                f"(size={dataset_size})"
            )
        end_index = args.local_sample_index + args.local_num_samples
        if end_index > dataset_size:
            die(
                f"Requested {args.local_num_samples} samples starting at {args.local_sample_index}, "
                f"but dataset size is {dataset_size}"
            )

        sample_indices = list(range(args.local_sample_index, end_index))
        samples_dir = run_root / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        summary_rows = []

        for sample_index in sample_indices:
            example = dataset[int(sample_index)]
            prompt_text, question_text, choices_texts, choices_str = build_prompt_from_example(
                example,
                dataset_name=args.local_dataset,
                use_cot=False,
                use_template=True,
            )

            prompt = [{"role": "user", "content": prompt_text}]
            input_text = tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            inputs = tokenizer(input_text, return_tensors="pt").to(device)
            position_ids = inputs["attention_mask"].long().cumsum(-1) - 1

            full_length = inputs["input_ids"].shape[1]
            instruction_length = full_length - 1
            kv_cache_index = create_segmented_kv_cache_index(
                instruction_length=instruction_length,
                response_length=1,
                proportion=args.kv_cache_proportion,
                order_mode=args.kv_cache_order_mode,
                device=device,
            )

            with torch.no_grad():
                if args.local_receiver_only:
                    outputs = base_model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=args.max_new_tokens,
                    )
                else:
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

            def strip_think_tags(text_value):
                return text_value.replace("<think>", "").replace("</think>", "").strip()

            extracted_answer = extract_answer_from_content(generated_text)
            expected_answer = parse_answer_from_example(example)
            matches = (
                (extracted_answer == expected_answer)
                if extracted_answer is not None and expected_answer is not None
                else (expected_answer in strip_think_tags(generated_text))
                if expected_answer is not None
                else None
            )

            kv_quant_stats = rosetta.get_kv_quant_stats() if rosetta is not None else None
            kv_quant_applied = (rosetta is not None) and kv_quant_config.get("enabled", False)
            meta = {
                "device": str(device),
                "dtype": str(dtype),
                "base_model": args.base_model,
                "teacher_model": args.teacher_model,
                "checkpoint_dir": checkpoint_dir,
                "include_response": bool(args.include_response),
                "max_new_tokens": args.max_new_tokens,
                "output_dir": str(run_root),
                "input_length": input_len,
                "generated_length": int(generated_ids.shape[0]),
                "prompt_text": prompt_text,
                "question_text": question_text,
                "choices_texts": choices_texts,
                "choices_str": choices_str,
                "generated_text": generated_text,
                "extracted_answer": extracted_answer,
                "expected_answer": expected_answer,
                "matches_expected": matches,
                "dataset": args.local_dataset,
                "sample_index": int(sample_index),
                "receiver_only": bool(args.local_receiver_only),
                "kv_quant_config": kv_quant_config,
                "kv_quant_stats": kv_quant_stats,
                "kv_quant_applied": kv_quant_applied,
                "kv_cache_proportion": args.kv_cache_proportion,
                "kv_cache_order_mode": args.kv_cache_order_mode,
            }

            sample_tag = f"sample_{sample_index:05d}"
            (samples_dir / f"{sample_tag}_output.txt").write_text(generated_text + "\n", encoding="utf-8")
            (samples_dir / f"{sample_tag}_output_full.txt").write_text(full_text + "\n", encoding="utf-8")
            (samples_dir / f"{sample_tag}_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

            summary_rows.append(
                {
                    "sample_index": int(sample_index),
                    "extracted_answer": extracted_answer,
                    "expected_answer": expected_answer,
                    "matches_expected": matches,
                }
            )

        num_correct = sum(1 for row in summary_rows if row["matches_expected"] is True)
        summary = {
            "dataset": args.local_dataset,
            "sample_indices": sample_indices,
            "num_samples": len(sample_indices),
            "num_correct": num_correct,
            "accuracy": (num_correct / len(sample_indices)) if sample_indices else None,
            "kv_quant_config": kv_quant_config,
            "kv_quant_stats": rosetta.get_kv_quant_stats() if rosetta is not None else None,
            "include_response": bool(args.include_response),
            "receiver_only": bool(args.local_receiver_only),
            "kv_cache_proportion": args.kv_cache_proportion,
            "kv_cache_order_mode": args.kv_cache_order_mode,
        }
        (run_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (run_root / "samples.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in summary_rows),
            encoding="utf-8",
        )

        if len(sample_indices) == 1:
            (run_root / "output.txt").write_text(
                (samples_dir / f"sample_{sample_indices[0]:05d}_output.txt").read_text(),
                encoding="utf-8",
            )
            (run_root / "output_full.txt").write_text(
                (samples_dir / f"sample_{sample_indices[0]:05d}_output_full.txt").read_text(),
                encoding="utf-8",
            )
            (run_root / "metadata.json").write_text(
                (samples_dir / f"sample_{sample_indices[0]:05d}_metadata.json").read_text(),
                encoding="utf-8",
            )

        write_status(run_root, "complete")

    print(f"Local smoke test complete. Output in: {run_root}")
    return run_root


def run_gpu_eval(project_root, data_root, kv_quant_config, args, run_root):
    run_root.mkdir(parents=True, exist_ok=True)
    for sub in ("configs", "logs", "results", "manifests"):
        (run_root / sub).mkdir(parents=True, exist_ok=True)

    log_path = run_root / "logs" / "step1.log"
    with log_path.open("a", encoding="utf-8") as log_file:
        env_info = collect_env_info()
        print("Environment info:", json.dumps(env_info, indent=2))
        (run_root / "manifests" / "env_info.json").write_text(json.dumps(env_info, indent=2))
        write_status(
            run_root,
            "running",
            {
                "mode": "gpu",
                "kv_quant_config": kv_quant_config,
                "kv_cache_proportion": args.kv_cache_proportion,
                "kv_cache_order_mode": args.kv_cache_order_mode,
            },
        )

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
            "kv_quant_config": kv_quant_config,
            "kv_cache_proportion": args.kv_cache_proportion,
            "kv_cache_order_mode": args.kv_cache_order_mode,
        }
        manifest_path = run_root / "manifests" / "step_1_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print("Wrote manifest:", manifest_path)

        (run_root / "configs" / "openbookqa.yaml").write_text(
            (c2c_root / "recipe" / "eval_recipe" / "unified_eval.yaml").read_text()
        )
        (run_root / "configs" / "arc_c.yaml").write_text(
            (c2c_root / "recipe" / "eval_recipe" / "unified_eval.yaml").read_text()
        )

        def patch(cfg_path, dataset, out_dir):
            cfg = yaml.safe_load(Path(cfg_path).read_text())
            cfg.setdefault("eval", {})
            cfg["model"]["rosetta_config"]["checkpoints_dir"] = manifest["checkpoint_dir"]
            cfg["model"]["rosetta_config"]["kv_quant_config"] = kv_quant_config
            cfg["output"]["output_dir"] = str(out_dir)
            cfg["eval"]["dataset"] = dataset
            cfg["eval"]["kv_cache_proportion"] = args.kv_cache_proportion
            cfg["eval"]["kv_cache_order_mode"] = args.kv_cache_order_mode
            Path(cfg_path).write_text(yaml.safe_dump(cfg, sort_keys=False))

        patch(run_root / "configs/openbookqa.yaml", "openbookqa", run_root / "results" / "openbookqa")
        patch(run_root / "configs/arc_c.yaml", "ai2-arc", run_root / "results" / "arc_c")

        if args.prep_only:
            write_status(run_root, "prep_only")
            print(f"Prep-only complete. Run on GPU to execute evals. Run folder: {run_root}")
            return run_root

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

        write_status(run_root, "complete")

    print(f"Step 1 KV-PTQ complete. Results in: {run_root}")
    return run_root


def main():
    parser = argparse.ArgumentParser(description="Run Step 1 KV-PTQ evals (GPU default, local smoke supported).")
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
    parser.add_argument("--kv-quant-scheme", choices=["int8", "int4"], default="int8", help="KV quant scheme")
    parser.add_argument("--kv-quant-axis", choices=["head", "layer"], default="head", help="KV quant axis")
    parser.add_argument("--kv-quant-eps", type=float, default=1e-6, help="KV quant epsilon")
    parser.add_argument("--kv-quant-collect-stats", action="store_true", help="Collect quant scale stats")
    parser.add_argument("--disable-kv-quant", action="store_true", help="Disable KV quantization for baseline compare")
    parser.add_argument(
        "--kv-cache-proportion",
        type=float,
        default=1.0,
        help="Proportion of instruction tokens kept for KV cache sharing (0.0-1.0)",
    )
    parser.add_argument(
        "--kv-cache-order-mode",
        choices=["front", "back"],
        default="front",
        help="Which part of instruction tokens to keep when pruning (front/back)",
    )
    parser.add_argument(
        "--local-dataset",
        choices=["openbookqa", "ai2-arc"],
        default="openbookqa",
        help="Local smoke dataset (1 sample)",
    )
    parser.add_argument("--local-sample-index", type=int, default=0, help="Local dataset sample index")
    parser.add_argument("--local-num-samples", type=int, default=1, help="Number of local samples to run")
    parser.add_argument(
        "--local-receiver-only",
        action="store_true",
        help="Local-only baseline: run the receiver model without C2C projectors",
    )
    parser.add_argument("--include-response", action="store_true", help="Enable include_response for RosettaModel")
    parser.add_argument("--skip-gpu-check", action="store_true", help="Skip GPU detection")
    parser.add_argument("--prep-only", action="store_true", help="Run setup + downloads only; skip evaluation")
    parser.add_argument("--cleanup-failed", action="store_true", help="Delete failed run folders")
    parser.add_argument("--no-reexec", action="store_true", help="Internal flag to avoid re-exec loops")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    if not project_root.is_dir():
        die(f"PROJECT_ROOT not found: {project_root}")
    quant_root = project_root / "quantization"
    if not quant_root.is_dir():
        die(f"quantization/ folder not found under {project_root}")
    data_root = quant_root / "data"

    if args.kv_cache_proportion < 0.0 or args.kv_cache_proportion > 1.0:
        die("kv_cache_proportion must be between 0.0 and 1.0")

    if args.mode == "gpu" and not args.skip_gpu_check and not args.prep_only:
        check_gpu()

    ensure_env(args.env, project_root, args)

    if args.hf_cache:
        os.environ["HF_HOME"] = args.hf_cache
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(args.hf_cache, "transformers")
    elif args.mode == "local":
        local_cache = data_root / "step_1_kv_ptq" / "local_smoke_tests" / "hf_cache"
        os.environ.setdefault("HF_HOME", str(local_cache))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(local_cache / "transformers"))
    else:
        os.environ.setdefault("HF_HOME", f"/scratch/m000066/{os.environ.get('USER','user')}/.cache/huggingface")
        os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(os.environ["HF_HOME"], "transformers"))
    os.environ.setdefault("WANDB_DISABLED", "true")

    kv_quant_enabled = not args.disable_kv_quant
    kv_quant_config = {
        "enabled": kv_quant_enabled,
        "scheme": args.kv_quant_scheme,
        "axis": args.kv_quant_axis,
        "eps": args.kv_quant_eps,
        "collect_stats": bool(args.kv_quant_collect_stats) and kv_quant_enabled,
    }

    run_root = None
    try:
        if args.mode == "local":
            if args.run_tag is None:
                args.run_tag = time.strftime("local_smoke_%Y%m%d_%H%M%S")
            if args.output_dir:
                run_root = Path(args.output_dir).expanduser().resolve()
            else:
                run_root = data_root / "step_1_kv_ptq" / "local_smoke_tests" / args.run_tag
            run_root = run_local_smoke_test(project_root, data_root, kv_quant_config, args, run_root)
        else:
            if args.run_tag is None:
                args.run_tag = f"{args.kv_quant_scheme}_{time.strftime('%Y%m%d_%H%M%S')}"
            run_root = data_root / "step_1_kv_ptq" / args.run_tag
            run_root = run_gpu_eval(project_root, data_root, kv_quant_config, args, run_root)
    except SystemExit as exc:
        cleanup_failed(run_root, args, f"exit_code={exc.code}")
        raise
    except Exception as exc:
        cleanup_failed(run_root, args, str(exc))
        raise


if __name__ == "__main__":
    main()
