from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from dataclasses import asdict, dataclass
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CandidateModel:
    model: str
    family: str
    params: str
    active_params: str | None
    architecture: str
    priority: str
    local_rung: str
    expected_device: str
    prompt_mode: str
    limit: int
    dtype: str
    max_new_tokens: int
    status: str
    rationale: str

    def command(self, benchmark: str, output_root: str) -> str:
        safe_name = (
            self.model.replace("/", "__")
            .replace(".", "_")
            .replace("-", "_")
            .replace("+", "_")
            .lower()
        )
        return " ".join(
            [
                "./venv_arm64/bin/python",
                "scripts/run_source_private_hidden_repair_packet_llm.py",
                f"--benchmark-jsonl {benchmark}",
                f"--output-dir {output_root}/{safe_name}",
                f"--model {self.model}",
                f"--device {self.expected_device}",
                f"--dtype {self.dtype}",
                f"--limit {self.limit}",
                "--seed 29",
                f"--max-new-tokens {self.max_new_tokens}",
                f"--prompt-mode {self.prompt_mode}",
                "--no-enable-thinking",
            ]
        )


def _model_matrix() -> list[CandidateModel]:
    return [
        CandidateModel(
            model="Qwen/Qwen3.5-0.8B",
            family="Qwen3.5 small hybrid",
            params="0.8B",
            active_params=None,
            architecture="qwen3_5 conditional generation",
            priority="P0",
            local_rung="CPU n160 passed on seeds 29/31",
            expected_device="cpu",
            prompt_mode="trace_no_hint",
            limit=64,
            dtype="float32",
            max_new_tokens=8,
            status="CPU n160 seed repeat passed after Transformers 5.7.0 upgrade; MPS backend fails before generation",
            rationale=(
                "Smallest latest Qwen3.5 candidate; first live latest-model smoke row. "
                "Transformers 5.7.0 recognizes model_type qwen3_5 and the CPU n16/n64/n160 rows passed; "
                "n160 is stable across seeds 29 and 31. "
                "Apple MPS currently fails inside the model's hybrid attention matmul before generation, "
                "so local confirmation uses CPU until the MPS backend path is fixed."
            ),
        ),
        CandidateModel(
            model="Qwen/Qwen3.5-2B",
            family="Qwen3.5 small hybrid",
            params="2B",
            active_params=None,
            architecture="qwen3_5 conditional generation",
            priority="P0",
            local_rung="CPU n16/n64/n160 passed",
            expected_device="cpu",
            prompt_mode="trace_no_hint",
            limit=160,
            dtype="float32",
            max_new_tokens=8,
            status="CPU n16/n64/n160 passed after local download; MPS skipped due Qwen3.5 backend risk",
            rationale=(
                "Second latest Qwen3.5 small size after Qwen3.5-0.8B. CPU n16/n64/n160 all reach "
                "1.000 matched accuracy versus a 0.250 target floor and no source-destroying control "
                "above 0.256 at n160, with valid packet rate 1.000."
            ),
        ),
        CandidateModel(
            model="Qwen/Qwen3.5-4B",
            family="Qwen3.5 small hybrid",
            params="4B",
            active_params=None,
            architecture="qwen3_5 conditional generation",
            priority="P1",
            local_rung="CPU n16/n64 passed; n160 only if needed",
            expected_device="cpu",
            prompt_mode="trace_no_hint",
            limit=64,
            dtype="float32",
            max_new_tokens=8,
            status="CPU n16/n64 passed; local CPU latency is high, so n160 is lower priority than MoE",
            rationale=(
                "Upper local Qwen3.5 small-hybrid row before remote 27B/MoE models. CPU n16/n64 both reach "
                "1.000 matched accuracy versus 0.250 target/control with packet valid rate 1.000; "
                "n64 p50 packet latency is about 27.2s on CPU."
            ),
        ),
        CandidateModel(
            model="Qwen/Qwen3.5-35B-A3B",
            family="Qwen3.5 MoE",
            params="35B total",
            active_params="3B activated",
            architecture="sparse MoE",
            priority="P2",
            local_rung="remote/API n32 after Qwen3.6 MoE",
            expected_device="cuda",
            prompt_mode="trace_no_hint",
            limit=32,
            dtype="bfloat16",
            max_new_tokens=8,
            status="off-machine Qwen3.5 MoE candidate",
            rationale="Adds a Qwen3.5-generation MoE row so MoE evidence is not tied only to Qwen3.6.",
        ),
        CandidateModel(
            model="Qwen/Qwen3.5-35B-A3B-FP8",
            family="Qwen3.5 MoE FP8",
            params="35B total",
            active_params="3B activated",
            architecture="sparse MoE quantized FP8",
            priority="P2",
            local_rung="remote/API n32 after Qwen3.6 FP8",
            expected_device="cuda",
            prompt_mode="trace_no_hint",
            limit=32,
            dtype="float16",
            max_new_tokens=8,
            status="off-machine Qwen3.5 MoE FP8 candidate",
            rationale="Checks whether an earlier Qwen3.5 MoE FP8 deployment preserves packet emission.",
        ),
        CandidateModel(
            model="Qwen/Qwen3.6-35B-A3B",
            family="Qwen3.6 MoE",
            params="35B total",
            active_params="3B activated",
            architecture="sparse MoE",
            priority="P1",
            local_rung="remote/API n32 then n500 if pass",
            expected_device="cuda",
            prompt_mode="trace_no_hint",
            limit=32,
            dtype="bfloat16",
            max_new_tokens=8,
            status="off-machine candidate",
            rationale=(
                "Direct MoE generalization test: the method should depend on instruction-following packet "
                "emission, not dense-vs-MoE internals."
            ),
        ),
        CandidateModel(
            model="Qwen/Qwen3.6-35B-A3B-FP8",
            family="Qwen3.6 MoE FP8",
            params="35B total",
            active_params="3B activated",
            architecture="sparse MoE quantized FP8",
            priority="P1",
            local_rung="remote/API n32 then n500 if pass",
            expected_device="cuda",
            prompt_mode="trace_no_hint",
            limit=32,
            dtype="float16",
            max_new_tokens=8,
            status="off-machine candidate",
            rationale=(
                "Tests whether low-rate diagnostic packet emission survives FP8 deployment/quantization."
            ),
        ),
        CandidateModel(
            model="Qwen/Qwen3.6-27B",
            family="Qwen3.6 dense",
            params="27B",
            active_params=None,
            architecture="dense",
            priority="P2",
            local_rung="remote/API n32",
            expected_device="cuda",
            prompt_mode="trace_no_hint",
            limit=32,
            dtype="bfloat16",
            max_new_tokens=8,
            status="off-machine dense latest-model comparator",
            rationale="Separates Qwen3.6 generation quality from MoE routing effects.",
        ),
        CandidateModel(
            model="Qwen/Qwen3-0.6B",
            family="Qwen3 small dense",
            params="0.6B",
            active_params=None,
            architecture="dense",
            priority="already-tested reference",
            local_rung="n500 done in final evidence",
            expected_device="mps",
            prompt_mode="trace_no_hint",
            limit=500,
            dtype="float32",
            max_new_tokens=8,
            status="positive reference row",
            rationale="Existing positive small-model reference for comparing latest Qwen3.5 models.",
        ),
        CandidateModel(
            model="Qwen/Qwen3-1.7B",
            family="Qwen3 small dense",
            params="1.7B",
            active_params=None,
            architecture="dense",
            priority="P2",
            local_rung="n16 if cached/downloaded",
            expected_device="mps",
            prompt_mode="trace_no_hint",
            limit=16,
            dtype="float32",
            max_new_tokens=8,
            status="optional small dense bridge row",
            rationale="Useful bridge between proven Qwen3-0.6B and newer Qwen3.5 small models.",
        ),
        CandidateModel(
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            family="DeepSeek distilled Qwen",
            params="1.5B",
            active_params=None,
            architecture="dense distillation",
            priority="P2",
            local_rung="n16 if cached",
            expected_device="mps",
            prompt_mode="trace_no_hint",
            limit=16,
            dtype="float32",
            max_new_tokens=8,
            status="optional non-Qwen-org small source emitter",
            rationale="Checks whether packet copying is robust to a reasoning-distilled Qwen derivative.",
        ),
        CandidateModel(
            model="allenai/OLMo-2-0425-1B-Instruct",
            family="OLMo 2 open instruct",
            params="1B",
            active_params=None,
            architecture="dense",
            priority="P1",
            local_rung="cross-family local n16",
            expected_device="mps",
            prompt_mode="trace_no_hint",
            limit=16,
            dtype="float32",
            max_new_tokens=8,
            status="n16 MPS failed behaviorally with 0 valid packets; pruned unless prompt contract changes",
            rationale=(
                "Open-science non-Qwen/Phi small instruct model; useful for testing whether packet "
                "emission depends on Qwen/Phi chat-template conventions."
            ),
        ),
        CandidateModel(
            model="google/gemma-3-1b-it",
            family="Gemma 3 small instruct",
            params="1B",
            active_params=None,
            architecture="dense",
            priority="P1",
            local_rung="cross-family local n16 if access/cache permits",
            expected_device="mps",
            prompt_mode="trace_no_hint",
            limit=16,
            dtype="float32",
            max_new_tokens=8,
            status="planned cross-family small-model falsification row",
            rationale="Strong non-Qwen small-family row; may require Hugging Face access acceptance.",
        ),
        CandidateModel(
            model="ibm-granite/granite-3.3-2b-instruct",
            family="Granite 3.3 instruct",
            params="2B",
            active_params=None,
            architecture="dense",
            priority="P1",
            local_rung="CPU copied-helper n160 passed; trace-no-hint n64 weaker",
            expected_device="cpu",
            prompt_mode="copied_helper",
            limit=64,
            dtype="float32",
            max_new_tokens=8,
            status="CPU copied-helper n16/n64/n160 passed; trace-no-hint n64 passes weaker; MPS backend fails",
            rationale=(
                "Enterprise/open small instruct row with long-context positioning. It is the first non-Qwen "
                "positive emitter: copied-helper CPU n160 reaches 0.800 matched accuracy vs 0.250 target and "
                "0.256 best controls. Trace-no-hint n64 is weaker but still positive at 0.578, and Apple MPS "
                "fails before generation, so keep the claim scoped."
            ),
        ),
        CandidateModel(
            model="HuggingFaceTB/SmolLM3-3B",
            family="SmolLM3",
            params="3B",
            active_params=None,
            architecture="dense",
            priority="P2",
            local_rung="cross-family local n16 if memory permits",
            expected_device="mps",
            prompt_mode="trace_no_hint",
            limit=16,
            dtype="float32",
            max_new_tokens=8,
            status="planned cross-family small-model falsification row",
            rationale=(
                "HF small-model research row; adds architecture and training-family breadth beyond Qwen/Phi."
            ),
        ),
        CandidateModel(
            model="microsoft/Phi-4-mini-instruct",
            family="Phi-4 mini instruct",
            params="3.8B",
            active_params=None,
            architecture="dense",
            priority="P2",
            local_rung="successor-family local n16 if memory permits",
            expected_device="mps",
            prompt_mode="trace_no_hint",
            limit=16,
            dtype="float16",
            max_new_tokens=8,
            status="planned Phi successor row",
            rationale=(
                "Direct successor-family check after the existing Phi-3 positive row; useful for "
                "separating protocol robustness from one frozen Phi generation."
            ),
        ),
        CandidateModel(
            model="mistralai/Ministral-3-3B-Instruct-2512-BF16",
            family="Ministral 3 instruct",
            params="3B-class",
            active_params=None,
            architecture="dense multimodal/text",
            priority="P2",
            local_rung="cross-family local/off-machine n16",
            expected_device="mps",
            prompt_mode="trace_no_hint",
            limit=16,
            dtype="float16",
            max_new_tokens=8,
            status="planned recent Mistral-family falsification row",
            rationale="Recent edge-focused Mistral-family row; useful if the local backend can run it.",
        ),
        CandidateModel(
            model="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
            family="Nemotron Nano",
            params="9B",
            active_params=None,
            architecture="hybrid Mamba-Transformer",
            priority="P2",
            local_rung="off-machine architecture-diversity n16",
            expected_device="cuda",
            prompt_mode="trace_no_hint",
            limit=16,
            dtype="bfloat16",
            max_new_tokens=8,
            status="planned off-machine architecture-diversity row",
            rationale=(
                "Hybrid Mamba-Transformer architecture row; would test packet-emitter robustness beyond "
                "standard dense transformer families."
            ),
        ),
        CandidateModel(
            model="moonshotai/Kimi-K2-Thinking",
            family="Kimi K2 MoE",
            params="1T total",
            active_params="32B activated",
            architecture="sparse MoE",
            priority="P2",
            local_rung="off-machine non-Qwen MoE stress row",
            expected_device="cuda",
            prompt_mode="trace_no_hint",
            limit=16,
            dtype="bfloat16",
            max_new_tokens=8,
            status="planned off-machine non-Qwen MoE stress row",
            rationale=(
                "Non-Qwen MoE stress test; only useful after small local rows pass because it is expensive "
                "and requires remote serving."
            ),
        ),
    ]


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Latest-Model Source-Packet Matrix",
        "",
        f"- generated: `{payload['generated']}`",
        f"- benchmark: `{payload['benchmark_jsonl']}`",
        "",
        "## Recommendation",
        "",
        payload["recommendation"],
        "",
        "## Model Matrix",
        "",
        "| Priority | Model | Family | Params | Active | Status | Rung |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for row in payload["models"]:
        lines.append(
            "| "
            f"{row['priority']} | `{row['model']}` | {row['family']} | "
            f"{row['params']} | {row['active_params'] or '-'} | "
            f"{row['status']} | {row['local_rung']} |"
        )
    lines.extend(["", "## Commands", ""])
    for row in payload["models"]:
        if row["priority"] in {"P0", "P1"}:
            lines.extend([f"### {row['model']}", "", "```bash", row["command"], "```", ""])
    lines.extend(["## Compatibility Note", "", payload["compatibility_note"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-jsonl",
        default="results/source_private_hidden_repair_packet_medium_20260429/benchmark.jsonl",
    )
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("results/source_private_latest_model_matrix_20260428"))
    parser.add_argument("--generated", default="2026-04-28")
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = args.benchmark_jsonl
    output_root = str(args.output_dir)
    models = [
        asdict(model) | {"command": model.command(benchmark, output_root)}
        for model in _model_matrix()
    ]
    payload = {
        "generated": args.generated,
        "benchmark_jsonl": benchmark,
        "recommendation": (
            "Treat MoE generalization as plausible but unproven. Qwen3.5-0.8B now has CPU n160 seed-stable "
            "passes after upgrading Transformers to 5.7.0; Qwen3.5-2B now also passes CPU n160; Qwen3.5-4B "
            "passes CPU n64; Granite copied-helper has a non-Qwen n160 pass. Next use Qwen3.6-35B-A3B "
            "and FP8 as off-machine MoE falsification rows, or run Qwen3.5-4B n160 only if local CPU time "
            "is acceptable."
        ),
        "compatibility_note": (
            "A local 2026-04-28 Qwen/Qwen3.5-0.8B smoke first failed before generation with "
            "transformers 4.51.0 because AutoConfig did not recognize model_type qwen3_5. "
            "After upgrading the repo-local environment to transformers 5.7.0, tokenizers 0.22.2, "
            "and huggingface_hub 1.12.0, Qwen3.5-0.8B CPU n16, n64, and n160 source-packet rows passed, "
            "with n160 repeated on seeds 29 and 31. Qwen3.5-2B CPU n16, n64, and n160 rows also passed "
            "on seed 29. Qwen3.5-4B CPU n16/n64 passed on seed 29 after downloading the 8.7G snapshot. The same 0.8B row still "
            "fails on Apple MPS before generation with an incompatible-dimensions matmul in the "
            "hybrid attention path, so MPS failure is logged as a backend compatibility issue rather "
            "than source-packet evidence. OLMo-2-0425-1B-Instruct is a behavioral negative at n16 "
            "with zero valid packets; Granite-3.3-2B-Instruct is a non-Qwen positive under copied-helper "
            "CPU n160 and a weaker trace-no-hint CPU n64 positive, while its MPS row is backend-blocked."
        ),
        "models": models,
    }
    (output_dir / "latest_model_matrix.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(output_dir / "latest_model_matrix.md", payload)
    manifest = {
        "command": " ".join(sys.argv),
        "artifacts": ["latest_model_matrix.json", "latest_model_matrix.md", "manifest.json", "manifest.md"],
        "artifact_sha256": {
            name: _sha256_file(output_dir / name)
            for name in ["latest_model_matrix.json", "latest_model_matrix.md"]
        },
        "script_sha256": _sha256_file(pathlib.Path(__file__)),
        "summary": {
            "model_count": len(models),
            "p0_count": sum(row["priority"] == "P0" for row in models),
            "moe_count": sum("MoE" in row["family"] for row in models),
            "off_machine_count": sum(row["local_rung"].startswith("remote") for row in models),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Latest-Model Source-Packet Matrix Manifest",
        "",
        f"- model count: `{manifest['summary']['model_count']}`",
        f"- P0 count: `{manifest['summary']['p0_count']}`",
        f"- MoE count: `{manifest['summary']['moe_count']}`",
        f"- off-machine count: `{manifest['summary']['off_machine_count']}`",
        "",
        "## Artifacts",
        "",
    ]
    lines.extend(f"- `{name}`" for name in manifest["artifacts"])
    (output_dir / "manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
