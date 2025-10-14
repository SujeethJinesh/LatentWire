#!/usr/bin/env python3
"""
LatentWire Experiment Orchestrator
----------------------------------
Runs 10 independent experiments sequentially with parameter sweeps and robust logging.
This script is a reference harness you can adapt to your repo (e.g., latentwire/cli/*).
It writes a machine-parseable JSONL log and mirrors stdout/stderr to per-run files.

Usage examples
--------------
# Dry run (print commands, don't execute)
python run_latentwire_experiments.py --dry-run

# Execute end-to-end (sequential)
python run_latentwire_experiments.py --execute

# Limit to specific experiments by id (comma-separated), e.g., 1,3,7
python run_latentwire_experiments.py --execute --only 1,3,7

Notes
-----
- The default cmd templates assume `python -m latentwire.cli.train` and `...eval` exist.
  If your entrypoints differ, edit the `cmd_template` fields in EXPERIMENTS below.
- Each experiment sweep expands into multiple runs; they are executed sequentially.
- Each run logs a JSON object into `./runs_orchestrator/metrics_history.jsonl`.
- Stdout and stderr are captured into per-run text files under `./runs_orchestrator/logs/`.
- You can set environment variables (e.g., CUDA_VISIBLE_DEVICES) via `--env`.
- A global `--max_samples` and `--max_steps` can be used to keep initial passes short.
"""

import argparse, os, json, time, subprocess, uuid, itertools
from pathlib import Path

ORCH_ROOT = Path("runs_orchestrator")
LOGS_DIR = ORCH_ROOT / "logs"
HISTORY_JSONL = ORCH_ROOT / "metrics_history.jsonl"

def ensure_dirs():
    ORCH_ROOT.mkdir(exist_ok=True, parents=True)
    LOGS_DIR.mkdir(exist_ok=True, parents=True)

def now():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def expand_grid(grid_dict):
    """Cartesian product expansion of a param grid dict into list of dicts."""
    if not grid_dict:
        return [dict()]
    keys = list(grid_dict.keys())
    values = [grid_dict[k] if isinstance(grid_dict[k], (list, tuple)) else [grid_dict[k]] for k in keys]
    combos = []
    for prod in itertools.product(*values):
        combos.append({k: v for k, v in zip(keys, prod)})
    return combos

def render_args(arg_dict):
    """Convert dict to CLI args; booleans become flags; None values are skipped."""
    parts = []
    for k, v in arg_dict.items():
        if v is None:
            continue
        flag = f"--{k.replace('_','-')}"
        if isinstance(v, bool):
            if v:
                parts.append(flag)
        else:
            parts.append(flag)
            parts.append(str(v))
    return parts

def run_cmd(cmd_list, env=None, dry_run=True, run_id="run"):
    """Run a command and capture logs. Return code, stdout_tail, stderr_tail."""
    cmd_str = " ".join(cmd_list)
    stdout_path = LOGS_DIR / f"{run_id}.stdout.txt"
    stderr_path = LOGS_DIR / f"{run_id}.stderr.txt"

    if dry_run:
        print(f"[DRY-RUN] {cmd_str}")
        return 0, "", ""

    with open(stdout_path, "w") as out, open(stderr_path, "w") as err:
        print(f"[EXEC] {cmd_str}")
        proc = subprocess.Popen(cmd_list, stdout=out, stderr=err, env=env)
        rc = proc.wait()
    # Tail last few lines for quick status in JSONL
    def tail(path, n=25):
        try:
            with open(path, "r") as f:
                lines = f.readlines()
            return "".join(lines[-n:])
        except Exception:
            return ""
    return rc, tail(stdout_path), tail(stderr_path)

def write_history(entry):
    with open(HISTORY_JSONL, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def default_env(extra_kv=None):
    env = os.environ.copy()
    if extra_kv:
        env.update(extra_kv)
    return env

# -----------------------------
# Experiments and sweep configs
# -----------------------------

# Each experiment is defined by:
# - id: numeric id 1..10
# - name: short name
# - question: the research question
# - train: dict -> {cmd_template: [...], base_args: {...}, sweep: {...}}
# - eval: optional dict with similar structure (cmd_template & args & sweep)
# - notes: freeform string for quick context in the logs
#
# You should check / adapt the cmd_template lists to your codebase.
# Many arguments here follow those mentioned in LOG.md/RESEARCH_PROPOSAL.md.

EXPERIMENTS = [

# 1) Sequence-preserving latent encoder vs mean-pooling
{
    "id": 1,
    "name": "SeqEncoder_vs_MeanPool",
    "question": "Does preserving token-level structure in the latent encoder avoid collapse and improve F1 vs mean pooling?",
    "train": {
        "cmd_template": ["python", "-m", "latentwire.cli.train"],
        "base_args": {
            "model_name": "llama-3.1-8b",
            "dataset": "squad",
            "epochs": 1,
            "batch_size": 16,
            "latent_len": 32,
            "d_z": 256,
            "encoder_type": None,  # filled by sweep
            "use_deep_prefix": False,
            "log_jsonl": str(HISTORY_JSONL),
        },
        "sweep": {
            "encoder_type": ["mean_pool", "st_query"],
            "latent_len": [24, 32, 48],
            "d_z": [128, 256, 512],
        }
    },
    "eval": {
        "cmd_template": ["python", "-m", "latentwire.cli.eval"],
        "base_args": {
            "model_name": "llama-3.1-8b",
            "dataset": "squad",
            "metrics": "f1,em,nll,first_tok_top1",
        },
        "sweep": {}
    },
    "notes": "Compare mean-pooled single-vector expansion vs learned-query cross-attention encoder."
},

# 2) Latent diversity regularization
{
    "id": 2,
    "name": "Latent_Diversity_Reg",
    "question": "Do orthogonality / redundancy penalties across latent slots reduce collapse and improve acceptance?",
    "train": {
        "cmd_template": ["python", "-m", "latentwire.cli.train"],
        "base_args": {
            "model_name": "llama-3.1-8b",
            "dataset": "squad",
            "epochs": 1,
            "batch_size": 16,
            "latent_len": 32,
            "d_z": 256,
            "diversity_loss": None,   # filled by sweep
            "diversity_weight": None, # filled by sweep
            "log_jsonl": str(HISTORY_JSONL),
        },
        "sweep": {
            "diversity_loss": ["cosine_orth", "covariance", "offdiag_l1"],
            "diversity_weight": [0.01, 0.05, 0.1],
            "latent_len": [24, 32, 48],
        }
    },
    "eval": {
        "cmd_template": ["python", "-m", "latentwire.cli.eval"],
        "base_args": {
            "model_name": "llama-3.1-8b",
            "dataset": "squad",
            "metrics": "f1,em,nll,first_tok_top1",
        },
        "sweep": {}
    },
    "notes": "Encourage specialization across latent slots."
},

# 3) Deep prefix injection ablation
{
    "id": 3,
    "name": "DeepPrefix_vs_Surface",
    "question": "Does per-layer KV (deep prefix) injection outperform surface inputs_embeds at matched budgets?",
    "train": {
        "cmd_template": ["python", "-m", "latentwire.cli.train"],
        "base_args": {
            "model_name": "llama-3.1-8b",
            "dataset": "squad",
            "epochs": 1,
            "batch_size": 16,
            "latent_len": 32,
            "d_z": 256,
            "use_deep_prefix": None,   # True/False sweep
            "deep_prefix_len": None,   # sweep
            "deep_prefix_dropout": None, # sweep
            "log_jsonl": str(HISTORY_JSONL),
        },
        "sweep": {
            "use_deep_prefix": [False, True],
            "deep_prefix_len": [0, 4, 8, 16],
            "deep_prefix_dropout": [0.0, 0.1, 0.2],
        }
    },
    "eval": {
        "cmd_template": ["python", "-m", "latentwire.cli.eval"],
        "base_args": {
            "model_name": "llama-3.1-8b",
            "dataset": "squad",
            "metrics": "f1,em,nll,first_tok_top1",
        },
        "sweep": {}
    },
    "notes": "KV-range injection is expected to help 'frozen acceptance'."
},

# 4) Tiny-LoRA acceptance sweep
{
    "id": 4,
    "name": "TinyLoRA_Acceptance",
    "question": "Does a tiny per-model LoRA on early attention blocks improve soft-prompt acceptance?",
    "train": {
        "cmd_template": ["python", "-m", "latentwire.cli.train"],
        "base_args": {
            "model_name": "llama-3.1-8b",
            "dataset": "squad",
            "epochs": 1,
            "batch_size": 16,
            "latent_len": 32,
            "d_z": 256,
            "lora_rank": None,      # sweep
            "lora_layers": None,    # sweep (e.g., 0-3)
            "log_jsonl": str(HISTORY_JSONL),
        },
        "sweep": {
            "lora_rank": [0, 4, 8, 16],
            "lora_layers": ["0-1", "0-3"],
        }
    },
    "eval": {
        "cmd_template": ["python", "-m", "latentwire.cli.eval"],
        "base_args": {
            "model_name": "llama-3.1-8b",
            "dataset": "squad",
            "metrics": "f1,em,nll,first_tok_top1",
        },
        "sweep": {}
    },
    "notes": "r=0 is the frozen baseline; test early-layer LoRA only."
},

# 5) Gist reconstruction auxiliary loss
{
    "id": 5,
    "name": "Gist_Reconstruction_Head",
    "question": "Does adding a gist reconstruction head protect against information loss and aid generation?",
    "train": {
        "cmd_template": ["python", "-m", "latentwire.cli.train"],
        "base_args": {
            "model_name": "llama-3.1-8b",
            "dataset": "squad",
            "epochs": 1,
            "batch_size": 16,
            "latent_len": 32,
            "d_z": 256,
            "use_gist_head": True,
            "gist_weight": None,     # sweep
            "gist_mask_prob": None,  # sweep
            "log_jsonl": str(HISTORY_JSONL),
        },
        "sweep": {
            "gist_weight": [0.1, 0.3, 0.5, 0.8],
            "gist_mask_prob": [0.15, 0.30, 0.50],
        }
    },
    "eval": {
        "cmd_template": ["python", "-m", "latentwire.cli.eval"],
        "base_args": {
            "model_name": "llama-3.1-8b",
            "dataset": "squad",
            "metrics": "f1,em,nll,first_tok_top1",
        },
        "sweep": {}
    },
    "notes": "Reconstructs prompt embeddings to keep latent informative."
},

# 6) Curriculum / annealing schedule sweep
{
    "id": 6,
    "name": "Curriculum_Anneal",
    "question": "Which annealing schedule stabilizes joint compression+generation?",
    "train": {
        "cmd_template": ["python", "-m", "latentwire.cli.train"],
        "base_args": {
            "model_name": "llama-3.1-8b",
            "dataset": "squad",
            "epochs": 1,
            "batch_size": 16,
            "latent_len": 32,
            "d_z": 256,
            "anneal_gen_objectives": True,
            "lambda_kce": None,   # sweep
            "lambda_kd": None,    # sweep
            "anneal_epochs": None, # sweep
            "warmup_recon_epochs": None, # optional sweep
            "log_jsonl": str(HISTORY_JSONL),
        },
        "sweep": {
            "lambda_kce": [0.001, 0.01],
            "lambda_kd": [0.001, 0.01],
            "anneal_epochs": [0.3, 0.4, 0.6],
            "warmup_recon_epochs": [0.0, 1.0, 2.0],
        }
    },
    "eval": {
        "cmd_template": ["python", "-m", "latentwire.cli.eval"],
        "base_args": {
            "model_name": "llama-3.1-8b",
            "dataset": "squad",
            "metrics": "f1,em,nll,first_tok_top1",
        },
        "sweep": {}
    },
    "notes": "Linear anneal weight sweep; optional recon warm-up."
},

# 7) Anchor-guided cross-model interlingua
{
    "id": 7,
    "name": "Anchor_Guided_Interlingua",
    "question": "Do semantic anchors and cross-model alignment losses improve portability to Qwen?",
    "train": {
        "cmd_template": ["python", "-m", "latentwire.cli.train"],
        "base_args": {
            "model_name": "llama-3.1-8b,qwen2.5-7b",  # train with both
            "dataset": "squad",
            "epochs": 1,
            "batch_size": 16,
            "latent_len": 32,
            "d_z": 512,
            "use_semantic_anchor": True,
            "lambda_align": None, # sweep
            "lambda_sem": None,   # sweep
            "log_jsonl": str(HISTORY_JSONL),
        },
        "sweep": {
            "lambda_align": [0.05, 0.1, 0.2],
            "lambda_sem": [0.05, 0.1, 0.2],
            "latent_len": [24, 32, 48],
        }
    },
    "eval": {
        "cmd_template": ["python", "-m", "latentwire.cli.eval"],
        "base_args": {
            "model_name": "llama-3.1-8b,qwen2.5-7b",
            "dataset": "squad",
            "metrics": "f1,em,nll,first_tok_top1,agreement,oracle_ub",
        },
        "sweep": {}
    },
    "notes": "Align Llama and Qwen latents; measure agreement & oracle."
},

# 8) Asymmetric interlingua (shared + model-specific tail)
{
    "id": 8,
    "name": "Asymmetric_Interlingua",
    "question": "Does reserving a small model-specific tail improve each model while keeping a shared core?",
    "train": {
        "cmd_template": ["python", "-m", "latentwire.cli.train"],
        "base_args": {
            "model_name": "llama-3.1-8b,qwen2.5-7b",
            "dataset": "squad",
            "epochs": 1,
            "batch_size": 16,
            "latent_len": None,       # computed as shared + private
            "shared_len": None,       # sweep
            "private_len": None,      # sweep
            "d_z": 256,
            "log_jsonl": str(HISTORY_JSONL),
        },
        "sweep": {
            "shared_len": [16, 24, 32],
            "private_len": [0, 8, 16],
        }
    },
    "eval": {
        "cmd_template": ["python", "-m", "latentwire.cli.eval"],
        "base_args": {
            "model_name": "llama-3.1-8b,qwen2.5-7b",
            "dataset": "squad",
            "metrics": "f1,em,agreement,oracle_ub",
        },
        "sweep": {}
    },
    "notes": "Compute latent_len as shared_len + private_len internally."
},

# 9) Joint rescoring strategy
{
    "id": 9,
    "name": "Joint_Rescoring",
    "question": "Does joint rescoring with a learned combiner outperform equal-weight sum of log-probs?",
    "train": {
        "cmd_template": ["python", "-m", "latentwire.cli.train"],
        "base_args": {
            "model_name": "llama-3.1-8b,qwen2.5-7b",
            "dataset": "squad",
            "epochs": 1,
            "batch_size": 16,
            "latent_len": 32,
            "d_z": 256,
            "log_jsonl": str(HISTORY_JSONL),
        },
        "sweep": {}
    },
    "eval": {
        "cmd_template": ["python", "-m", "latentwire.cli.eval"],
        "base_args": {
            "model_name": "llama-3.1-8b,qwen2.5-7b",
            "dataset": "squad",
            "metrics": "f1,em",
            "rescoring": None,  # sweep
        },
        "sweep": {
            "rescoring": ["equal", "learned_logistic"],
        }
    },
    "notes": "Keep training fixed; only change rescoring strategy at eval."
},

# 10) Cache-augmentation approximation (deep prefix as KV augmentation proxy)
{
    "id": 10,
    "name": "CacheAug_Proxy",
    "question": "At matched latent budgets, does KV-approx (deep prefix) beat embedding-only prompting?",
    "train": {
        "cmd_template": ["python", "-m", "latentwire.cli.train"],
        "base_args": {
            "model_name": "llama-3.1-8b",
            "dataset": "squad",
            "epochs": 1,
            "batch_size": 16,
            "latent_len": 32,
            "d_z": 256,
            "use_deep_prefix": None,      # sweep
            "deep_prefix_len": None,      # sweep
            "ahead_tokens": None,         # sweep (proxy for lookahead)
            "log_jsonl": str(HISTORY_JSONL),
        },
        "sweep": {
            "use_deep_prefix": [False, True],
            "deep_prefix_len": [4, 8, 16, 32],
            "ahead_tokens": [8, 16, 32],
        }
    },
    "eval": {
        "cmd_template": ["python", "-m", "latentwire.cli.eval"],
        "base_args": {
            "model_name": "llama-3.1-8b",
            "dataset": "squad",
            "metrics": "f1,em,nll,first_tok_top1",
        },
        "sweep": {}
    },
    "notes": "Inspired by KV-cache coprocessor literature; approximates augmentation."
},

]

def compute_derived_args(exp, args):
    """
    Place for per-experiment derived parameters; e.g., compute latent_len = shared + private.
    """
    if exp["id"] == 8:  # Asymmetric interlingua
        shared = int(args.get("shared_len", 0) or 0)
        private = int(args.get("private_len", 0) or 0)
        args["latent_len"] = shared + private
    return args

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--execute", action="store_true", help="Execute commands sequentially")
    parser.add_argument("--only", type=str, default="", help="Comma-separated experiment ids to run")
    parser.add_argument("--max-samples", type=int, default=None, help="Override samples per run if your CLI supports it")
    parser.add_argument("--max-steps", type=int, default=None, help="Override steps per run if your CLI supports it")
    parser.add_argument("--env", type=str, default="", help="Comma-separated k=v to inject into environment")
    args = parser.parse_args()

    ensure_dirs()
    selected = set()
    if args.only:
        selected = {int(x.strip()) for x in args.only.split(",") if x.strip()}

    extra_env = {}
    if args.env:
        for kv in args.env.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                extra_env[k.strip()] = v.strip()

    # Record orchestrator config
    write_history({
        "ts": now(),
        "event": "orchestrator_start",
        "only": sorted(list(selected)) if selected else "ALL",
        "dry_run": args.dry_run and not args.execute,
    })

    env = os.environ.copy()
    env.update(extra_env)

    for exp in EXPERIMENTS:
        if selected and exp["id"] not in selected:
            continue

        # TRAIN SWEEP
        train = exp.get("train", {})
        train_grid = expand_grid(train.get("sweep", {}))
        for combo_idx, sweep_vals in enumerate(train_grid, start=1):
            run_uuid = str(uuid.uuid4())[:8]
            base = dict(train.get("base_args", {}))
            base.update(sweep_vals)
            base = compute_derived_args(exp, base)

            # User overrides to keep runs short if desired
            if args.max_samples is not None:
                base["samples"] = args.max_samples
            if args.max_steps is not None:
                base["max_steps"] = args.max_steps

            run_id = f"exp{exp['id']:02d}_train_{combo_idx:03d}_{run_uuid}"
            cmd = list(train.get("cmd_template", [])) + render_args(base)

            rc, out_tail, err_tail = run_cmd(cmd, env=env, dry_run=(not args.execute), run_id=run_id)
            write_history({
                "ts": now(),
                "phase": "train",
                "exp_id": exp["id"],
                "exp_name": exp["name"],
                "question": exp["question"],
                "combo_idx": combo_idx,
                "args": base,
                "cmd": cmd,
                "return_code": rc,
                "stdout_tail": out_tail,
                "stderr_tail": err_tail,
                "notes": exp.get("notes", ""),
            })

        # EVAL SWEEP
        evald = exp.get("eval", {})
        eval_grid = expand_grid(evald.get("sweep", {}))
        if not eval_grid:  # at least one eval
            eval_grid = [dict()]
        for combo_idx, sweep_vals in enumerate(eval_grid, start=1):
            run_uuid = str(uuid.uuid4())[:8]
            base = dict(evald.get("base_args", {}))
            base.update(sweep_vals)

            # Allow quick eval overrides
            if args.max_samples is not None:
                base["samples"] = args.max_samples

            run_id = f"exp{exp['id']:02d}_eval_{combo_idx:03d}_{run_uuid}"
            cmd = list(evald.get("cmd_template", [])) + render_args(base)

            rc, out_tail, err_tail = run_cmd(cmd, env=env, dry_run=(not args.execute), run_id=run_id)
            write_history({
                "ts": now(),
                "phase": "eval",
                "exp_id": exp["id"],
                "exp_name": exp["name"],
                "question": exp["question"],
                "combo_idx": combo_idx,
                "args": base,
                "cmd": cmd,
                "return_code": rc,
                "stdout_tail": out_tail,
                "stderr_tail": err_tail,
                "notes": exp.get("notes", ""),
            })

    write_history({
        "ts": now(),
        "event": "orchestrator_done"
    })

if __name__ == "__main__":
    main()
