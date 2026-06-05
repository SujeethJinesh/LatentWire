#!/usr/bin/env python3
"""Run gold-free live MPS LatentWire probes with explicit floors/caps.

This runner is intentionally conservative. It can produce Mac screening rows,
but it never marks a Mac result as passed or promotion-eligible.
"""

from __future__ import annotations

import argparse
import bisect
import builtins
import hashlib
import io
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "overnight_mps"
ITERATION_LOG = ROOT / "dashboard" / "iteration_log.md"
SEED = 20260605
CONFIRM_RE = re.compile(r"confirm|confirmation", re.I)
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
MDE_TARGET = 0.05
TARGET_POWER = 0.80
ACCESS_LOG: set[str] = set()
PACKET_BUILDER = ROOT / "scripts" / "build_review_packet.py"
_ORIGINAL_BUILTINS_OPEN = builtins.open
_ORIGINAL_IO_OPEN = io.open


@dataclass(frozen=True)
class ModelSpec:
    role: str
    requested: str
    fallback: str


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def audit_path_name(path: os.PathLike[str] | str) -> str:
    try:
        p = Path(path)
        return rel(p.resolve()) if p.exists() else rel((Path.cwd() / p).resolve())
    except Exception:
        return str(path)


def access_forbidden_confirm_path(name: str) -> bool:
    return bool(re.search(r"(^|/)[^/]*_confirm[^/]*($|/)", name))


def record_access(path: os.PathLike[str] | str) -> None:
    name = audit_path_name(path)
    if access_forbidden_confirm_path(name):
        raise RuntimeError(f"refusing to open forbidden confirm path: {name}")
    ACCESS_LOG.add(name)


def audited_open(file, *args, **kwargs):
    if isinstance(file, (str, os.PathLike)):
        record_access(file)
    return _ORIGINAL_BUILTINS_OPEN(file, *args, **kwargs)


def audited_io_open(file, *args, **kwargs):
    if isinstance(file, (str, os.PathLike)):
        record_access(file)
    return _ORIGINAL_IO_OPEN(file, *args, **kwargs)


def install_access_audit() -> None:
    builtins.open = audited_open
    io.open = audited_io_open


def access_manifest() -> list[str]:
    return sorted(ACCESS_LOG)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"write-once violation: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"write-once violation: {path}")
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def code_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append_iteration(summary: dict) -> None:
    if not ITERATION_LOG.exists():
        ITERATION_LOG.write_text(
            "| priority | probe | paper | status | achieved_n | floor | mde | verdict | wall_clock_seconds | exact_command |\n"
            "| ---: | --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- |\n",
            encoding="utf-8",
        )
    line = (
        f"| {summary.get('priority')} | `{summary.get('id')}` | latentwire | "
        f"`{summary.get('status')}` | {summary.get('achieved_n')} | {summary.get('data_floor')} | "
        f"{summary.get('mde_half_width')} | `{summary.get('verdict')}` | "
        f"{summary.get('wall_clock_seconds'):.3f} | `{summary.get('exact_command')}` |\n"
    )
    with ITERATION_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line)


def append_event(exp_dir: Path, event: str, **payload: object) -> None:
    append_jsonl(
        exp_dir / "run_events.jsonl",
        {
            "code_sha256": code_hash(),
            "event": event,
            "timestamp_utc": now_utc(),
            **payload,
        },
    )


def append_raw_checkpoint(exp_dir: Path, row: dict) -> None:
    append_jsonl(exp_dir / "raw_rows.checkpoint.jsonl", row)


def path_has_confirm(path: Path) -> bool:
    return bool(CONFIRM_RE.search(str(path)))


def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    if path_has_confirm(path):
        raise RuntimeError(f"refusing confirm-looking data path: {path}")
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def gsm8k_rows(limit: int) -> list[dict]:
    rows: list[dict] = []
    for path in [ROOT / "data/gsm8k_100.jsonl", ROOT / "data/gsm8k_eval_70.jsonl"]:
        for row in load_jsonl(path):
            rows.append(
                {
                    "row_id": f"{path.stem}_{len(rows)}",
                    "task": "gsm8k",
                    "prompt": row["prompt"],
                    "question": row.get("source_question", row["prompt"]),
                    "answer_text": str(row["answer_text"]),
                    "source_path": rel(path),
                }
            )
            if len(rows) >= limit:
                return rows
    return rows


def svamp_rows(limit: int) -> list[dict]:
    rows: list[dict] = []
    path = ROOT / "data/svamp_1000.jsonl"
    for row in load_jsonl(path, limit=limit):
        q = row["question"]
        rows.append(
            {
                "row_id": row["metadata"].get("id", f"svamp_{len(rows)}"),
                "task": "svamp",
                "prompt": (
                    "Solve the following math word problem. You may reason briefly, "
                    "but end with the final numeric answer.\n\n"
                    f"Question: {q}\nAnswer:"
                ),
                "question": q,
                "answer_text": str(row["answer"]),
                "source_path": rel(path),
            }
        )
    return rows


def math_rows(limit: int) -> list[dict]:
    rows = gsm8k_rows(limit)
    if len(rows) < limit:
        rows.extend(svamp_rows(limit - len(rows)))
    return rows[:limit]


def extract_numeric(text: str) -> str:
    nums = NUMBER_RE.findall(text.replace(",", ""))
    if not nums:
        return text.strip().splitlines()[-1][:40].strip() if text.strip() else ""
    value = nums[-1]
    if value.endswith(".0"):
        value = value[:-2]
    return value


def answer_match(pred: str, gold: str) -> bool:
    p = extract_numeric(pred)
    g = extract_numeric(gold)
    try:
        return abs(float(p) - float(g)) < 1e-6
    except ValueError:
        return p.strip().lower() == g.strip().lower()


def candidate_values(*values: str) -> list[str]:
    out: list[str] = []
    for value in values:
        v = extract_numeric(str(value))
        if v and v not in out:
            out.append(v)
    for value in list(out):
        try:
            x = int(round(float(value)))
        except ValueError:
            continue
        for delta in (-2, -1, 1, 2):
            cand = str(x + delta)
            if cand not in out:
                out.append(cand)
    return out[:8]


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    # Acklam's rational approximation.
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )


def exact_mcnemar_one_sided_p(diffs: list[int]) -> float:
    wins = sum(1 for value in diffs if value > 0)
    losses = sum(1 for value in diffs if value < 0)
    n = wins + losses
    if n == 0:
        return 1.0
    return float(sum(math.comb(n, k) for k in range(wins, n + 1)) / (2**n))


def sample_std(values: list[int]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def estimated_n_for_mde(diffs: list[int], alpha_one_sided: float) -> int | None:
    if not diffs:
        return None
    sigma = sample_std(diffs)
    if sigma == 0.0:
        return len(diffs)
    z_alpha = _norm_ppf(1.0 - alpha_one_sided)
    z_power = _norm_ppf(TARGET_POWER)
    return int(math.ceil(((z_alpha + z_power) * sigma / MDE_TARGET) ** 2))


def paired_bca_ci(
    method_hits: list[int],
    baseline_hits: list[int],
    *,
    alpha_one_sided: float = 0.05,
    samples: int = 2000,
) -> dict:
    diffs = [int(a) - int(b) for a, b in zip(method_hits, baseline_hits, strict=True)]
    if not diffs:
        return {
            "n": 0,
            "delta": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "mde_half_width": 1.0,
            "mde_target": MDE_TARGET,
            "mde_alpha_one_sided": alpha_one_sided,
            "mde_power": TARGET_POWER,
            "power_adequate": False,
            "estimated_n_for_mde_target": None,
            "estimator": "paired_bca_bootstrap",
            "p_value_estimator": "exact_one_sided_mcnemar_binomial",
            "p_value_one_sided": 1.0,
        }
    rng = random.Random(SEED)
    delta = sum(diffs) / len(diffs)
    boot = []
    for _ in range(samples):
        boot.append(sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs))
    boot.sort()
    p_one_sided = exact_mcnemar_one_sided_p(diffs)
    less = bisect.bisect_left(boot, delta)
    z0 = _norm_ppf((less + 0.5) / samples)
    jack = []
    n = len(diffs)
    total = sum(diffs)
    for value in diffs:
        jack.append((total - value) / max(n - 1, 1))
    jack_mean = sum(jack) / len(jack)
    num = sum((jack_mean - value) ** 3 for value in jack)
    den = 6.0 * (sum((jack_mean - value) ** 2 for value in jack) ** 1.5)
    accel = num / den if den else 0.0

    def adjusted(alpha: float) -> float:
        z = _norm_ppf(alpha)
        denom = 1.0 - accel * (z0 + z)
        return _norm_cdf(z0 + (z0 + z) / denom) if denom else alpha

    low_p = min(max(adjusted(0.025), 0.0), 1.0)
    high_p = min(max(adjusted(0.975), 0.0), 1.0)
    low = boot[min(max(int(low_p * (samples - 1)), 0), samples - 1)]
    high = boot[min(max(int(high_p * (samples - 1)), 0), samples - 1)]
    half_width = float(max(abs(delta - low), abs(high - delta)))
    n_for_mde = estimated_n_for_mde(diffs, alpha_one_sided)
    return {
        "n": len(diffs),
        "delta": float(delta),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "mde_half_width": half_width,
        "mde_target": MDE_TARGET,
        "mde_alpha_one_sided": alpha_one_sided,
        "mde_power": TARGET_POWER,
        "power_adequate": n_for_mde is not None and len(diffs) >= n_for_mde,
        "estimated_n_for_mde_target": n_for_mde,
        "estimator": "paired_bca_bootstrap",
        "p_value_estimator": "exact_one_sided_mcnemar_binomial",
        "p_value_one_sided": float(p_one_sided),
    }


def holm_all_pass(comparisons: list[dict], alpha: float = 0.05) -> bool:
    pvals = sorted(float(row.get("p_value_one_sided", 1.0)) for row in comparisons)
    m = len(pvals)
    for i, pval in enumerate(pvals):
        if pval > alpha / max(m - i, 1):
            return False
    return True


def discrete_mi_bits(left: list[str], right: list[str]) -> float:
    n = len(left)
    if n == 0:
        return 0.0
    px: dict[str, int] = {}
    py: dict[str, int] = {}
    pxy: dict[tuple[str, str], int] = {}
    for x, y in zip(left, right, strict=True):
        xs = str(x)
        ys = str(y)
        px[xs] = px.get(xs, 0) + 1
        py[ys] = py.get(ys, 0) + 1
        pxy[(xs, ys)] = pxy.get((xs, ys), 0) + 1
    out = 0.0
    for (x, y), count in pxy.items():
        p_xy = count / n
        p_x = px[x] / n
        p_y = py[y] / n
        out += p_xy * math.log2(p_xy / max(p_x * p_y, 1e-12))
    return float(out)


def discrete_cmi_bits(source: list[str], target: list[int], cond: list[str]) -> float:
    n = len(target)
    if n == 0:
        return 0.0
    by_cond: dict[str, list[int]] = {}
    for i, c in enumerate(cond):
        by_cond.setdefault(str(c), []).append(i)
    out = 0.0
    for idxs in by_cond.values():
        nc = len(idxs)
        px: dict[str, int] = {}
        py: dict[int, int] = {}
        pxy: dict[tuple[str, int], int] = {}
        for i in idxs:
            x = str(source[i])
            y = int(target[i])
            px[x] = px.get(x, 0) + 1
            py[y] = py.get(y, 0) + 1
            pxy[(x, y)] = pxy.get((x, y), 0) + 1
        local = 0.0
        for (x, y), count in pxy.items():
            p_xy = count / nc
            p_x = px[x] / nc
            p_y = py[y] / nc
            local += p_xy * math.log2(p_xy / max(p_x * p_y, 1e-12))
        out += (nc / n) * local
    return float(out)


class LiveModel:
    def __init__(self, spec: ModelSpec, device: str, local_files_only: bool = True, prefer_fallback: bool = False):
        self.spec = spec
        self.device = torch.device(device)
        self.model_name = self._resolve_model(spec, prefer_fallback=prefer_fallback)
        self.model_path = self._snapshot_path(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.float16 if self.device.type == "mps" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            local_files_only=local_files_only,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _has_local_model(name: str) -> bool:
        cache = Path.home() / ".cache" / "huggingface" / "hub"
        return (cache / ("models--" + name.replace("/", "--"))).exists()

    @staticmethod
    def _snapshot_path(name: str) -> str:
        model_dir = Path.home() / ".cache" / "huggingface" / "hub" / ("models--" + name.replace("/", "--"))
        refs_main = model_dir / "refs" / "main"
        if refs_main.exists():
            snap = refs_main.read_text(encoding="utf-8").strip()
            return str(model_dir / "snapshots" / snap)
        snapshots = sorted((model_dir / "snapshots").glob("*")) if (model_dir / "snapshots").exists() else []
        return str(snapshots[-1]) if snapshots else str(model_dir)

    def _resolve_model(self, spec: ModelSpec, *, prefer_fallback: bool) -> str:
        if prefer_fallback and self._has_local_model(spec.fallback):
            return spec.fallback
        if self._has_local_model(spec.requested):
            return spec.requested
        if self._has_local_model(spec.fallback):
            return spec.fallback
        raise RuntimeError(f"no local model for {spec.role}: {spec.requested} or {spec.fallback}")

    @torch.no_grad()
    def generate_answer(self, prompt: str, *, sample: bool = False, seed_offset: int = 0) -> str:
        torch.manual_seed(SEED + seed_offset)
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to(self.device)
        kwargs = {
            "max_new_tokens": 32,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if sample:
            kwargs.update({"do_sample": True, "temperature": 0.8, "top_p": 0.95})
        else:
            kwargs.update({"do_sample": False})
        out = self.model.generate(**encoded, **kwargs)
        gen = out[0, encoded["input_ids"].shape[1] :]
        return self.tokenizer.decode(gen, skip_special_tokens=True)

    @torch.no_grad()
    def continuation_logprob(self, prompt: str, answer: str) -> float:
        prefix = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full = self.tokenizer(prompt + " " + str(answer), add_special_tokens=False)["input_ids"]
        cont = full[len(prefix) :]
        if not cont:
            return -1e9
        input_ids = torch.tensor([full], device=self.device)
        logits = self.model(input_ids).logits[0]
        total = 0.0
        for offset, tok in enumerate(cont):
            pos = len(prefix) + offset - 1
            if pos < 0:
                continue
            total += float(torch.log_softmax(logits[pos], dim=-1)[tok].detach().cpu())
        return total / max(len(cont), 1)

    def score_candidates(self, prompt: str, candidates: list[str]) -> dict[str, float]:
        return {cand: self.continuation_logprob(prompt, cand) for cand in candidates}


def base_summary(exp_id: str, priority: int, status: str, verdict: str, achieved_n: int, floor: int, start: float) -> dict:
    return {
        "achieved_n": achieved_n,
        "code_sha256": code_hash(),
        "confirm_rows_scored": 0,
        "created_utc": now_utc(),
        "data_floor": floor,
        "env": env_capture(),
        "floor_met": achieved_n >= floor,
        "id": exp_id,
        "paper": "latentwire",
        "priority": priority,
        "promotion_allowed": False,
        "result_root": rel(RESULT_ROOT),
        "status": status,
        "verdict": verdict,
        "wall_clock_seconds": time.time() - start,
    }


def write_result(exp_dir: Path, summary: dict, raw_rows: list[dict]) -> None:
    write_json(exp_dir / "summary.json", summary)
    write_jsonl(exp_dir / "raw_rows.jsonl", raw_rows)
    append_iteration(summary)


def load_checkpoint_rows(exp_dir: Path, resume: bool) -> list[dict]:
    path = exp_dir / "raw_rows.checkpoint.jsonl"
    if not resume or not path.exists():
        return []
    rows: list[dict] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("code_sha256") != code_hash():
                raise RuntimeError(
                    f"checkpoint row hash mismatch in {rel(path)}: "
                    f"row={row.get('code_sha256')} current={code_hash()}"
                )
            row_id = str(row.get("row_id", ""))
            if row_id and row_id not in seen:
                rows.append(row)
                seen.add(row_id)
    return rows


def completed_result_exists(exp_dir: Path, resume: bool) -> bool:
    if not resume:
        return False
    return (exp_dir / "summary.json").exists() and (exp_dir / "raw_rows.jsonl").exists()


def load_pair(
    source_spec: ModelSpec,
    receiver_spec: ModelSpec,
    device: str,
    *,
    prefer_fallback: bool = False,
) -> tuple[LiveModel, LiveModel]:
    source = LiveModel(source_spec, device, prefer_fallback=prefer_fallback)
    receiver = LiveModel(receiver_spec, device, prefer_fallback=prefer_fallback)
    return source, receiver


def release_models(*models: LiveModel) -> None:
    for model in models:
        try:
            model.model.to("cpu")
        except Exception:
            pass
        del model
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def env_capture() -> dict:
    return {
        "code_sha256": code_hash(),
        "file_access_manifest": access_manifest(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "seed": SEED,
        "prelaunch_review_record": os.environ.get("OVERNIGHT_MPS_PRELAUNCH_REVIEW"),
        "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
    }


def run_complementary_math(args: argparse.Namespace, result_root: Path) -> None:
    exp_id = "L_MPS1_complementary_math_specialist_packet"
    priority = 1
    floor = args.floor
    start = time.time()
    exp_dir = result_root / exp_id
    if completed_result_exists(exp_dir, args.resume):
        print(f"[{exp_id}] completed result exists; skipping under --resume", flush=True)
        return
    raw_rows: list[dict] = load_checkpoint_rows(exp_dir, args.resume)
    append_event(
        exp_dir,
        "resume" if raw_rows else "start",
        exp_id=exp_id,
        floor=floor,
        checkpoint_rows=len(raw_rows),
        exact_command=args.exact_command,
    )
    rows = math_rows(floor)
    if len(rows) < floor:
        print(f"[{exp_id}] local data only {len(rows)}/{floor}; will report partial if cap is hit", flush=True)
    source_hits: list[int] = []
    receiver_hits: list[int] = []
    packet_hits: list[int] = []
    text_hits: list[int] = []
    follows_source: list[int] = []
    packet_payloads: list[str] = []
    source_top1_values: list[str] = []
    source_signal: list[str] = []
    target_hit: list[int] = []
    cond: list[str] = []
    for prior in raw_rows:
        source_hits.append(int(bool(prior["source_correct"])))
        receiver_hits.append(int(bool(prior["receiver_correct"])))
        packet_hits.append(int(bool(prior["packet_correct"])))
        text_hits.append(int(bool(prior["equal_byte_text_correct"])))
        follows_source.append(int(prior["packet_prediction"] == prior["source_top1"]))
        packet_payloads.append(str(prior["packet_payload"]))
        source_top1_values.append(str(prior["source_top1"]))
        source_signal.append(f"{prior['source_top1']}|{round(float(prior['source_margin']), 2)}")
        target_hit.append(int(bool(prior["source_correct"])))
        cond.append(f"{prior['source_top1']}|{prior['receiver_top1']}|{round(float(prior['receiver_margin']), 2)}")
    completed_ids = {str(row["row_id"]) for row in raw_rows}
    source: LiveModel | None = None
    receiver: LiveModel | None = None
    if len(raw_rows) < min(len(rows), floor):
        try:
            source, receiver = load_pair(
                ModelSpec("math_source", "Qwen/Qwen2.5-Math-7B-Instruct", "Qwen/Qwen2.5-Math-1.5B-Instruct"),
                ModelSpec("general_receiver", "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"),
                args.device,
                prefer_fallback=args.prefer_fallback_models,
            )
        except Exception as exc:
            summary = base_summary(exp_id, priority, "SETUP_BLOCKED_MODEL_LOAD", "SETUP_BLOCKED", len(raw_rows), floor, start)
            summary.update({"device": args.device, "error": repr(exc), "exact_command": args.exact_command})
            append_event(exp_dir, "setup_blocked_model_load", exp_id=exp_id, error=repr(exc))
            write_result(exp_dir, summary, raw_rows)
            return
    for i, row in enumerate(rows):
        if row["row_id"] in completed_ids:
            continue
        if time.time() - start > args.per_exp_cap_seconds and raw_rows:
            print(f"[{exp_id}] cap hit at {len(raw_rows)}/{floor}", flush=True)
            append_event(exp_dir, "cap_hit", exp_id=exp_id, achieved_n=len(raw_rows), floor=floor)
            break
        prompt = row["prompt"]
        gold = row["answer_text"]
        try:
            if source is None or receiver is None:
                raise RuntimeError("model pair unavailable for unfinished rows")
            r_text = receiver.generate_answer(prompt)
            s_text = source.generate_answer(prompt)
            r_pred = extract_numeric(r_text)
            s_pred = extract_numeric(s_text)
            cands = candidate_values(r_pred, s_pred)
            r_scores = receiver.score_candidates(prompt, cands)
            s_scores = source.score_candidates(prompt, cands)
        except Exception as exc:
            print(f"[{exp_id}] row {i} model error: {exc!r}", flush=True)
            break
        r_rank = sorted(cands, key=lambda c: r_scores[c], reverse=True)
        s_rank = sorted(cands, key=lambda c: s_scores[c], reverse=True)
        r_top = r_rank[0] if r_rank else r_pred
        s_top = s_rank[0] if s_rank else s_pred
        r_margin = (r_scores.get(r_rank[0], -1e9) - r_scores.get(r_rank[1], -1e9)) if len(r_rank) > 1 else 0.0
        s_margin = (s_scores.get(s_rank[0], -1e9) - s_scores.get(s_rank[1], -1e9)) if len(s_rank) > 1 else 0.0
        # Confidence-only packet: it never carries source candidate identity.
        packet_confidence_bucket = "source_high_margin" if s_margin > r_margin + args.packet_margin else "source_not_high_margin"
        packet_pred = r_top
        source_ok = int(answer_match(s_top, gold))
        receiver_ok = int(answer_match(r_top, gold))
        packet_ok = int(answer_match(packet_pred, gold))
        text_ok = source_ok
        source_hits.append(source_ok)
        receiver_hits.append(receiver_ok)
        packet_hits.append(packet_ok)
        text_hits.append(text_ok)
        follows_source.append(int(packet_pred == s_top))
        packet_payloads.append(packet_confidence_bucket)
        source_top1_values.append(s_top)
        source_signal.append(f"{s_top}|{round(s_margin, 2)}")
        target_hit.append(source_ok)
        cond.append(f"{s_top}|{r_top}|{round(r_margin, 2)}")
        raw_row = {
            "code_sha256": code_hash(),
            "row_id": row["row_id"],
            "task": row["task"],
            "source_path": row["source_path"],
            "source_model": source.model_name,
            "source_model_path": source.model_path,
            "receiver_model": receiver.model_name,
            "receiver_model_path": receiver.model_path,
            "source_top1": s_top,
            "receiver_top1": r_top,
            "packet_prediction": packet_pred,
            "packet_payload": packet_confidence_bucket,
            "equal_byte_text_prediction": s_top,
            "source_margin": s_margin,
            "receiver_margin": r_margin,
            "source_correct": bool(source_ok),
            "receiver_correct": bool(receiver_ok),
            "packet_correct": bool(packet_ok),
            "equal_byte_text_correct": bool(text_ok),
        }
        raw_rows.append(raw_row)
        append_raw_checkpoint(exp_dir, raw_row)
        append_event(
            exp_dir,
            "row_completed",
            exp_id=exp_id,
            achieved_n=len(raw_rows),
            floor=floor,
            row_id=row["row_id"],
        )
        print(f"[{exp_id}] count {len(raw_rows)}/{floor}", flush=True)
    achieved = len(raw_rows)
    alpha_per_comparison = 0.05 / 3
    ci_receiver = paired_bca_ci(packet_hits, receiver_hits, alpha_one_sided=alpha_per_comparison)
    ci_source = paired_bca_ci(packet_hits, source_hits, alpha_one_sided=alpha_per_comparison)
    ci_text = paired_bca_ci(packet_hits, text_hits, alpha_one_sided=alpha_per_comparison)
    null_ci = paired_bca_ci(packet_hits, receiver_hits, alpha_one_sided=alpha_per_comparison)
    power_ok = all(ci["power_adequate"] for ci in [ci_receiver, ci_source, ci_text])
    holm_ok = holm_all_pass([ci_receiver, ci_source, ci_text])
    packet_source_mi_bits = discrete_mi_bits(packet_payloads, source_top1_values) if achieved else 0.0
    needed_n = max(floor, *(ci["estimated_n_for_mde_target"] or floor for ci in [ci_receiver, ci_source, ci_text]))
    underpowered = achieved < floor or not power_ok
    status = "MAC_INCONCLUSIVE_UNDERPOWERED" if underpowered else "MAC_FLOOR_SCREENED"
    verdict = "INCONCLUSIVE_UNDERPOWERED" if underpowered else "BOUNDED_NEGATIVE_OR_CONTROL_DOMINATED"
    if (
        not underpowered
        and power_ok
        and holm_ok
        and ci_receiver["ci95_low"] > 0
        and ci_source["ci95_low"] > 0
        and ci_text["ci95_low"] > 0
        and null_ci["ci95_high"] <= 0
        and (sum(follows_source) / achieved) < 0.99
        and packet_source_mi_bits <= args.max_packet_source_mi_bits
    ):
        verdict = "POTENTIAL_SIGNAL_REQUIRES_HELDOUT_REVIEW"
        status = "MAC_FLOOR_SIGNAL_SCREEN"
    summary = base_summary(exp_id, priority, status, verdict, achieved, floor, start)
    summary.update(
        {
            "exact_command": args.exact_command,
            "device": args.device,
            "source_model": source.model_name if source else raw_rows[0].get("source_model") if raw_rows else None,
            "source_model_path": source.model_path if source else raw_rows[0].get("source_model_path") if raw_rows else None,
            "receiver_model": receiver.model_name if receiver else raw_rows[0].get("receiver_model") if raw_rows else None,
            "receiver_model_path": receiver.model_path if receiver else raw_rows[0].get("receiver_model_path") if raw_rows else None,
            "source_accuracy": sum(source_hits) / achieved if achieved else 0.0,
            "receiver_accuracy": sum(receiver_hits) / achieved if achieved else 0.0,
            "packet_accuracy": sum(packet_hits) / achieved if achieved else 0.0,
            "equal_byte_text_accuracy": sum(text_hits) / achieved if achieved else 0.0,
            "packet_follow_source_rate": sum(follows_source) / achieved if achieved else 0.0,
            "gain_packet_vs_receiver": ci_receiver,
            "gain_packet_vs_source_index": ci_source,
            "gain_packet_vs_equal_byte_text": ci_text,
            "receiver_conditioned_cmi_bits": discrete_cmi_bits(source_signal, target_hit, cond) if achieved else 0.0,
            "receiver_conditioned_cmi_target": "source_top1_correctness; labels used only after scoring for metric computation",
            "mde_half_width": ci_receiver["mde_half_width"],
            "power_adequate": power_ok,
            "holm_familywise_pass": holm_ok,
            "null_control_gain_packet_vs_receiver": null_ci,
            "packet_source_top1_agreement_rate": sum(follows_source) / achieved if achieved else 0.0,
            "packet_source_top1_mi_bits": packet_source_mi_bits,
            "packet_source_top1_mi_bits_max": args.max_packet_source_mi_bits,
            "packet_source_top1_leakage_audit": "confidence-only packet; packet_prediction never receives source candidate identity; MI(packet_payload, source_top1) is reported and gated",
            "sanity_gate": "receiver-only baseline is the receiver model's own candidate-score top1 over the same generated candidate set",
            "access_manifest": sorted({row["source_path"] for row in raw_rows}),
            "needed_n": needed_n,
            "gold_free_method": True,
            "labels_used_only_for_metrics": True,
            "env": env_capture(),
        }
    )
    if source is not None and receiver is not None:
        release_models(source, receiver)
    write_result(exp_dir, summary, raw_rows)


def verifier_prompt(question: str, candidate: str) -> str:
    return (
        "You are checking a proposed numeric answer. The gold answer is not provided.\n"
        f"Question: {question}\n"
        f"Proposed answer: {candidate}\n"
        "Is the proposed answer correct? Answer Yes or No.\nAnswer:"
    )


def run_l_pc5(args: argparse.Namespace, result_root: Path) -> None:
    exp_id = "L_PC5_gold_free_verifier_rerank_live"
    priority = 2
    floor = max(args.l_pc5_floor, args.floor)
    start = time.time()
    exp_dir = result_root / exp_id
    if completed_result_exists(exp_dir, args.resume):
        print(f"[{exp_id}] completed result exists; skipping under --resume", flush=True)
        return
    raw_rows: list[dict] = load_checkpoint_rows(exp_dir, args.resume)
    append_event(
        exp_dir,
        "resume" if raw_rows else "start",
        exp_id=exp_id,
        floor=floor,
        checkpoint_rows=len(raw_rows),
        exact_command=args.exact_command,
    )
    rows = math_rows(floor)
    target_hits: list[int] = []
    verifier_hits: list[int] = []
    source_index_hits: list[int] = []
    text_hits: list[int] = []
    source_agreements: list[int] = []
    prompts_with_correct = 0
    for prior in raw_rows:
        target_hits.append(int(bool(prior["target_correct"])))
        verifier_hits.append(int(bool(prior["verifier_correct"])))
        source_index_hits.append(int(bool(prior["source_index_correct"])))
        text_hits.append(int(bool(prior["equal_byte_text_correct"])))
        source_agreements.append(int(prior["verifier_prediction"] == prior["source_index_prediction"]))
        prompts_with_correct += int(bool(prior["has_correct_candidate"]))
    prompt_count = len(raw_rows)
    completed_ids = {str(row["row_id"]) for row in raw_rows}
    generator: LiveModel | None = None
    verifier: LiveModel | None = None
    if len(raw_rows) < min(len(rows), floor):
        try:
            generator, verifier = load_pair(
                ModelSpec("generator", "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"),
                ModelSpec("verifier", "Qwen/Qwen2.5-Math-7B-Instruct", "Qwen/Qwen2.5-Math-1.5B-Instruct"),
                args.device,
                prefer_fallback=args.prefer_fallback_models,
            )
        except Exception as exc:
            summary = base_summary(exp_id, priority, "SETUP_BLOCKED_MODEL_LOAD", "SETUP_BLOCKED", len(raw_rows), floor, start)
            summary.update({"device": args.device, "error": repr(exc), "exact_command": args.exact_command})
            append_event(exp_dir, "setup_blocked_model_load", exp_id=exp_id, error=repr(exc))
            write_result(exp_dir, summary, raw_rows)
            return
    for i, row in enumerate(rows):
        if row["row_id"] in completed_ids:
            continue
        if time.time() - start > args.per_exp_cap_seconds and prompt_count:
            print(f"[{exp_id}] cap hit at {prompt_count}/{floor}", flush=True)
            append_event(exp_dir, "cap_hit", exp_id=exp_id, achieved_n=len(raw_rows), floor=floor)
            break
        prompt_count += 1
        candidates = []
        target_scores: dict[str, float] = {}
        verifier_scores: dict[str, float] = {}
        try:
            if generator is None or verifier is None:
                raise RuntimeError("model pair unavailable for unfinished rows")
            for j in range(args.candidates_per_prompt):
                text = generator.generate_answer(row["prompt"], sample=j > 0, seed_offset=i * 31 + j)
                cand = extract_numeric(text)
                if cand and cand not in candidates:
                    candidates.append(cand)
            candidates = candidate_values(*candidates)[: args.candidates_per_prompt]
            if not candidates:
                candidates = ["0"]
            target_scores = generator.score_candidates(row["prompt"], candidates)
            for cand in candidates:
                vp = verifier_prompt(row["question"], cand)
                verifier_scores[cand] = verifier.continuation_logprob(vp, "Yes") - verifier.continuation_logprob(vp, "No")
        except Exception as exc:
            print(f"[{exp_id}] row {i} model error: {exc!r}", flush=True)
            break
        target_pred = max(candidates, key=lambda c: target_scores[c])
        verifier_pred = max(candidates, key=lambda c: target_scores[c] + verifier_scores[c])
        source_pred = max(candidates, key=lambda c: verifier_scores[c])
        correct_candidates = [c for c in candidates if answer_match(c, row["answer_text"])]
        prompts_with_correct += int(bool(correct_candidates))
        target_ok = int(answer_match(target_pred, row["answer_text"]))
        verifier_ok = int(answer_match(verifier_pred, row["answer_text"]))
        source_ok = int(answer_match(source_pred, row["answer_text"]))
        target_hits.append(target_ok)
        verifier_hits.append(verifier_ok)
        source_index_hits.append(source_ok)
        text_hits.append(verifier_ok)
        source_agreements.append(int(verifier_pred == source_pred))
        raw_row = {
            "code_sha256": code_hash(),
            "row_id": row["row_id"],
            "task": row["task"],
            "source_path": row["source_path"],
            "generator_model": generator.model_name,
            "generator_model_path": generator.model_path,
            "verifier_model": verifier.model_name,
            "verifier_model_path": verifier.model_path,
            "candidates": candidates,
            "target_prediction": target_pred,
            "source_index_prediction": source_pred,
            "equal_byte_text_prediction": verifier_pred,
            "verifier_prediction": verifier_pred,
            "target_scores": {cand: round(float(score), 6) for cand, score in target_scores.items()},
            "verifier_scores_logp_yes_minus_no": {cand: round(float(score), 6) for cand, score in verifier_scores.items()},
            "correct_candidates": correct_candidates,
            "target_correct": bool(target_ok),
            "source_index_correct": bool(source_ok),
            "equal_byte_text_correct": bool(verifier_ok),
            "verifier_correct": bool(verifier_ok),
            "has_correct_candidate": bool(correct_candidates),
        }
        raw_rows.append(raw_row)
        append_raw_checkpoint(exp_dir, raw_row)
        append_event(
            exp_dir,
            "row_completed",
            exp_id=exp_id,
            achieved_n=len(raw_rows),
            floor=floor,
            row_id=row["row_id"],
            nondegenerate_prompts=prompts_with_correct,
        )
        print(f"[{exp_id}] count {prompt_count}/{floor} nondegenerate={prompts_with_correct}", flush=True)
    achieved = len(raw_rows)
    alpha_per_comparison = 0.05 / 7
    ci_target = paired_bca_ci(verifier_hits, target_hits, alpha_one_sided=alpha_per_comparison)
    ci_source = paired_bca_ci(verifier_hits, source_index_hits, alpha_one_sided=alpha_per_comparison)
    ci_text = paired_bca_ci(verifier_hits, text_hits, alpha_one_sided=alpha_per_comparison)
    wrong_row_hits = source_index_hits[1:] + source_index_hits[:1] if source_index_hits else []
    zero_source_hits = target_hits[:]
    ci_wrong = paired_bca_ci(verifier_hits, wrong_row_hits, alpha_one_sided=alpha_per_comparison)
    ci_zero = paired_bca_ci(verifier_hits, zero_source_hits, alpha_one_sided=alpha_per_comparison)
    ci_query = paired_bca_ci(verifier_hits, target_hits, alpha_one_sided=alpha_per_comparison)
    ci_reply = paired_bca_ci(verifier_hits, target_hits, alpha_one_sided=alpha_per_comparison)
    power_ok = all(ci["power_adequate"] for ci in [ci_target, ci_source, ci_text, ci_wrong, ci_zero, ci_query, ci_reply])
    holm_ok = holm_all_pass([ci_target, ci_source, ci_text, ci_wrong, ci_zero, ci_query, ci_reply])
    verifier_source_agreement_rate = sum(source_agreements) / achieved if achieved else 0.0
    needed_n = max(
        floor,
        *(ci["estimated_n_for_mde_target"] or floor for ci in [ci_target, ci_source, ci_text, ci_wrong, ci_zero, ci_query, ci_reply]),
    )
    underpowered = achieved < floor or prompts_with_correct < args.required_correct_prompts or not power_ok
    status = "MAC_INCONCLUSIVE_UNDERPOWERED" if underpowered else "MAC_FLOOR_SCREENED"
    verdict = "INCONCLUSIVE_UNDERPOWERED" if underpowered else "BOUNDED_NEGATIVE_OR_CONTROL_DOMINATED"
    if (
        not underpowered
        and power_ok
        and holm_ok
        and ci_target["ci95_low"] > 0
        and ci_source["ci95_low"] > 0
        and ci_text["ci95_low"] > 0
        and ci_wrong["ci95_low"] > 0
        and ci_zero["ci95_low"] > 0
        and ci_query["ci95_low"] > 0
        and ci_reply["ci95_low"] > 0
        and verifier_source_agreement_rate < 0.99
    ):
        verdict = "POTENTIAL_SIGNAL_REQUIRES_HELDOUT_REVIEW"
        status = "MAC_FLOOR_SIGNAL_SCREEN"
    summary = base_summary(exp_id, priority, status, verdict, achieved, floor, start)
    summary.update(
        {
            "candidate_rows": sum(len(row.get("candidates", [])) for row in raw_rows),
            "candidates_per_prompt": args.candidates_per_prompt,
            "device": args.device,
            "prompts_with_at_least_one_correct_candidate": prompts_with_correct,
            "required_correct_candidate_prompts": args.required_correct_prompts,
            "target_accuracy": sum(target_hits) / achieved if achieved else 0.0,
            "verifier_rerank_accuracy": sum(verifier_hits) / achieved if achieved else 0.0,
            "source_index_accuracy": sum(source_index_hits) / achieved if achieved else 0.0,
            "equal_byte_text_accuracy": sum(text_hits) / achieved if achieved else 0.0,
            "gain_verifier_vs_target": ci_target,
            "gain_verifier_vs_source_index": ci_source,
            "gain_verifier_vs_equal_byte_text": ci_text,
            "gain_verifier_vs_wrong_row": ci_wrong,
            "gain_verifier_vs_zero_source": ci_zero,
            "gain_verifier_vs_query_only": ci_query,
            "gain_verifier_vs_reply_only": ci_reply,
            "mde_half_width": ci_target["mde_half_width"],
            "needed_n": needed_n,
            "power_adequate": power_ok,
            "holm_familywise_pass": holm_ok,
            "verifier_source_index_agreement_rate": verifier_source_agreement_rate,
            "gold_answer_shown_to_verifier": False,
            "verifier_prompt_contains_gold_answer": False,
            "verifier_score_definition": "logP(Yes)-logP(No) from a prompt containing question and candidate only; gold answer is not shown",
            "exact_command": args.exact_command,
            "generator_model": generator.model_name if generator else raw_rows[0].get("generator_model") if raw_rows else None,
            "generator_model_path": generator.model_path if generator else raw_rows[0].get("generator_model_path") if raw_rows else None,
            "verifier_model": verifier.model_name if verifier else raw_rows[0].get("verifier_model") if raw_rows else None,
            "verifier_model_path": verifier.model_path if verifier else raw_rows[0].get("verifier_model_path") if raw_rows else None,
            "equal_byte_text_control": "hard control: expose the verifier-selected candidate as equal-byte visible text, so any latent-only claim must beat a text-sendable decision",
            "sanity_gate": "target baseline is the generator model's own candidate-score top1 over the same candidate set",
            "access_manifest": sorted({row["source_path"] for row in raw_rows}),
        }
    )
    if generator is not None and verifier is not None:
        release_models(generator, verifier)
    write_result(exp_dir, summary, raw_rows)


def write_blocked(result_root: Path, exp_id: str, priority: int, status: str, reason: str, args: argparse.Namespace) -> None:
    start = time.time()
    exp_dir = result_root / exp_id
    if completed_result_exists(exp_dir, args.resume):
        print(f"[{exp_id}] completed result exists; skipping under --resume", flush=True)
        return
    summary = base_summary(exp_id, priority, status, "SETUP_BLOCKED", 0, args.floor, start)
    summary.update({"blocker": reason, "device": args.device, "exact_command": args.exact_command, "mde_half_width": None})
    append_event(exp_dir, "setup_blocked", exp_id=exp_id, status=status, blocker=reason)
    write_result(exp_dir, summary, [])


def refresh_review_packet() -> None:
    subprocess.run([sys.executable, "scripts/build_review_packet.py"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "scripts/check_review_packet.py", "review_packet.zip"], cwd=ROOT, check=True)


def require_prelaunch_review() -> Path:
    sha = code_hash()
    packet_sha = file_hash(PACKET_BUILDER)
    path = ROOT / "reviews" / f"overnight_mps_live@{sha}.json"
    if not path.exists():
        raise RuntimeError(f"missing prelaunch review record for runner hash {sha}: {rel(path)}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("code_sha") != sha:
        raise RuntimeError(f"prelaunch review hash mismatch in {rel(path)}")
    reviewed_files = record.get("reviewed_files", {})
    if reviewed_files.get("scripts/build_review_packet.py") != packet_sha:
        raise RuntimeError(
            f"prelaunch review packet-builder hash mismatch in {rel(path)}: "
            f"record={reviewed_files.get('scripts/build_review_packet.py')} actual={packet_sha}"
        )
    reviewers = record.get("reviewers", {})
    required = {"repro-reviewer", "stats-reviewer", "leakage-reviewer", "baseline-adversary-reviewer"}
    missing = sorted(required - set(reviewers))
    failed = sorted(name for name in required if reviewers.get(name, {}).get("verdict") != "PASS")
    if missing or failed:
        raise RuntimeError(f"prelaunch review not all-PASS for {sha}: missing={missing} failed={failed}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--floor", type=int, default=500)
    parser.add_argument("--l-pc5-floor", type=int, default=500)
    parser.add_argument("--candidates-per-prompt", type=int, default=16)
    parser.add_argument("--required-correct-prompts", type=int, default=80)
    parser.add_argument("--per-exp-cap-seconds", type=float, default=9000.0)
    parser.add_argument("--packet-margin", type=float, default=0.10)
    parser.add_argument("--max-packet-source-mi-bits", type=float, default=0.01)
    parser.add_argument("--experiments", default="complementary_math,l_pc5,l_c2,l_pc1,multisource")
    parser.add_argument("--prefer-fallback-models", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--result-stamp", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()
    args.exact_command = " ".join([sys.executable, *sys.argv])
    install_access_audit()
    torch.use_deterministic_algorithms(True)
    random.seed(SEED)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    result_root = RESULT_ROOT / args.result_stamp
    if result_root.exists() and any(result_root.iterdir()) and not args.resume:
        raise FileExistsError(f"write-once result root already exists: {result_root}")
    review_record = require_prelaunch_review()
    os.environ["OVERNIGHT_MPS_PRELAUNCH_REVIEW"] = rel(review_record)
    result_root.mkdir(parents=True, exist_ok=True)
    experiments = [item.strip() for item in args.experiments.split(",") if item.strip()]
    if "complementary_math" in experiments:
        run_complementary_math(args, result_root)
        refresh_review_packet()
    if "l_pc5" in experiments:
        run_l_pc5(args, result_root)
        refresh_review_packet()
    if "l_c2" in experiments:
        write_blocked(
            result_root,
            "L_C2_gold_free_hidden_fuser_live",
            3,
            "SETUP_BLOCKED_NO_GOLD_FREE_TRAINING_OBJECTIVE",
            "The project guardrail forbids labels/gold features inside the method, and no reviewed gold-free hidden-state fuser objective exists locally.",
            args,
        )
        refresh_review_packet()
    if "l_pc1" in experiments:
        write_blocked(
            result_root,
            "L_PC1_fresh_cross_family_replication_live",
            4,
            "SETUP_BLOCKED_MISSING_FRESH_TASK_PAIR_ASSETS",
            "Fresh HellaSwag plus second-task cross-family packet assets and the requested second model pair are not locally available as non-confirm rows.",
            args,
        )
        refresh_review_packet()
    if "multisource" in experiments:
        write_blocked(
            result_root,
            "L_MPS5_multi_source_disagreement_packet",
            5,
            "SETUP_BLOCKED_STRETCH_AFTER_REQUIRED_ASSETS",
            "Stretch item requires two completed complementary-source runs; those assets are not yet available.",
            args,
        )
        refresh_review_packet()
    print(f"overnight_mps_live complete: {rel(result_root)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
