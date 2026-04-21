# LatentMAS Wrapper Inspection 2026-04-21

Scope: inspect `references/repos/LatentMAS` and define a LatentWire-side wrapper test contract without running heavy model calls or editing vendor code.

## Local Repo

- Path: `references/repos/LatentMAS`
- Entrypoint: `run.py`
- Native methods: `baseline`, `text_mas`, `latent_mas`
- Native tasks: `gsm8k`, `aime2024`, `aime2025`, `gpqa`, `arc_easy`, `arc_challenge`, `mbppplus`, `humanevalplus`, `medqa`
- Native gap: SVAMP is not a LatentMAS task, so a LatentWire-side wrapper must convert `data/svamp_eval_70.jsonl` rows into LatentMAS item dicts.

## Method Signatures

`BaselineMethod` from `methods/baseline.py`:

```python
BaselineMethod(
    model,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    generate_bs: int = 1,
    use_vllm: bool = False,
    args=None,
)
```

`TextMASMethod` from `methods/text_mas.py`:

```python
TextMASMethod(
    model,
    *,
    max_new_tokens_each: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    generate_bs: int = 1,
    args=None,
)
```

`LatentMASMethod` from `methods/latent_mas.py`:

```python
LatentMASMethod(
    model,
    *,
    latent_steps: int = 10,
    judger_max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    generate_bs: int = 1,
    args=None,
)
```

All three expose `run_batch(items: list[dict]) -> list[dict]`; `LatentMASMethod` also exposes `run_batch_vllm(items)` for the hybrid vLLM path.

## Expected Item Schema

LatentMAS dataset loaders yield:

```python
{
    "question": str,
    "solution": str,
    "gold": str,
}
```

The wrapper should preserve LatentWire IDs and aliases for telemetry while passing only LatentMAS-compatible fields into `run_batch`:

```python
{
    "id": str,
    "question": str,
    "solution": str,
    "gold": str,
    "answers": list[str],
}
```

Conversion rules:

- GSM-style rows use `prompt` when present, otherwise `question`; gold comes from `answer_text`, `answer`, or `target`.
- SVAMP rows use `question`; gold comes from `answer`; aliases should be preserved.
- Stable fallback ID can be the row offset; prefer `metadata.id` when present.

## Expected Method Output

LatentMAS methods return:

```python
{
    "question": str,
    "gold": str,
    "solution": str,
    "prediction": str | None,
    "raw_prediction": str,
    "agents": list[dict],
    "correct": bool,
}
```

Each `agents` entry usually includes `name`, `role`, `input`, `output`, and for HF paths `input_ids` plus `input_tokens`. Latent agents include `latent_steps` for non-judger agents.

## Mock Strategy

No model calls are needed for wrapper tests. Use dependency injection:

- `model_factory(args) -> object` returns a dummy object.
- `method_factory(method_name, model, args) -> fake_method` returns a fake with `run_batch`.
- Fake `run_batch` records received item batches and returns deterministic LatentMAS-shaped records.
- Use a fake clock or patched timer so latency/tokens metadata is deterministic.

The tests added in `tests/test_latentmas_competitor_eval.py` are skipped until `scripts.run_latentmas_competitor_eval` exists. When the main wrapper is implemented, they verify:

- LatentWire GSM/SVAMP JSONL rows convert to LatentMAS-compatible item records.
- Method constructor kwargs match upstream LatentMAS signatures.
- A fake-method run writes JSONL records and a `.jsonl.meta.json` summary with accuracy, task, method, trace agent count, and input-token count.

## Recommended Wrapper API

```python
load_latentwire_generation_items(path: Path, limit: int | None) -> list[dict]
make_latentmas_method(method_name: str, model: object, args: argparse.Namespace) -> object
run_eval(args: argparse.Namespace, *, model_factory=None, method_factory=None, clock=None) -> dict
```

This keeps the wrapper testable and avoids importing or loading Hugging Face models during unit tests.
