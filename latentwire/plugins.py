"""Plugin helpers: chat templates and PEFT adapters."""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Tuple, Union

__all__ = [
    "build_chat_for_qa",
    "apply_lora",
    "maybe_merge_lora",
    "apply_prefix_tuning",
    "apply_prompt_tuning",
]

# ---------------------------------------------------------------------------
# Chat template helper
# ---------------------------------------------------------------------------


def build_chat_for_qa(
    tokenizer,
    question: str,
    context: str,
    system: Optional[str] = None,
    add_generation_prompt: bool = True,
    return_tensors: str = "pt",
) -> Tuple[Dict, Optional[str]]:
    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append(
        {
            "role": "user",
            "content": (
                "Use the context to answer the question.\n\n"
                f"Context:\n{context}\n\nQuestion: {question}\nAnswer in one short span."
            ),
        }
    )
    kwargs = dict(add_generation_prompt=add_generation_prompt, return_tensors=return_tensors)
    try:
        kwargs["continue_final_message"] = True
    except Exception:
        pass
    model_inputs = tokenizer.apply_chat_template(messages, **kwargs)
    prompt_text = None
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            **{k: v for k, v in kwargs.items() if k != "return_tensors"},
        )
    except Exception:
        pass
    return model_inputs, prompt_text

# ---------------------------------------------------------------------------
# PEFT integration
# ---------------------------------------------------------------------------


def _ensure_peft():
    try:
        import peft  # noqa
    except Exception:
        import subprocess
        import sys

        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "peft>=0.12.0",
            "accelerate>=0.33.0",
        ])
    import peft  # noqa
    return peft


def _infer_default_targets(model):
    names = set()
    for n, _ in model.named_modules():
        if any(s in n for s in ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj")):
            names.add(n.split(".")[-1])
    if not names:
        names = {"q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"}
    return sorted(list(names))


def _parse_target_modules(arg: Union[str, Iterable[str]], model):
    if isinstance(arg, (list, tuple, set)):
        return list(arg), None
    s = str(arg).strip()
    firstN = None
    if "firstN:" in s:
        s, n = s.split("firstN:", 1)
        try:
            firstN = int(n)
        except Exception:
            firstN = None
    if s == "attn":
        mods = ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif s == "attn_mlp":
        mods = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ]
    elif s == "auto":
        mods = _infer_default_targets(model)
    else:
        mods = [m.strip() for m in s.split(",") if m.strip()]
    return mods, firstN


def apply_lora(
    model,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: Union[str, Iterable[str]] = "attn_mlp_firstN:16",
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
):
    peft = _ensure_peft()
    from peft import LoraConfig, get_peft_model, TaskType

    tm, firstN = _parse_target_modules(target_modules, model)
    kwargs = dict(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=tm,
        bias=bias,
        task_type=getattr(TaskType, task_type),
    )
    try:
        kwargs["layers_to_transform"] = list(range(firstN)) if firstN else None
    except Exception:
        pass

    cfg = LoraConfig(**{k: v for k, v in kwargs.items() if v is not None})
    lora_model = get_peft_model(model, cfg)
    try:
        lora_model.print_trainable_parameters()
    except Exception:
        pass
    return lora_model


def maybe_merge_lora(model):
    try:
        from peft import PeftModel

        if isinstance(model, PeftModel) and getattr(model, "peft_config", None):
            adapter_types = {cfg.peft_type for cfg in model.peft_config.values()}
            if adapter_types == {"LORA"} or "LORA" in adapter_types:
                model = model.merge_and_unload()
        return model
    except Exception:
        return model


def apply_prefix_tuning(
    model,
    num_virtual_tokens: int = 16,
    projection: bool = True,
    encoder_hidden_size: Optional[int] = None,
    task_type: str = "CAUSAL_LM",
):
    peft = _ensure_peft()
    from peft import PrefixTuningConfig, get_peft_model, TaskType

    cfg = PrefixTuningConfig(
        task_type=getattr(TaskType, task_type),
        num_virtual_tokens=int(num_virtual_tokens),
        prefix_projection=bool(projection),
        encoder_hidden_size=encoder_hidden_size,
    )
    pt_model = get_peft_model(model, cfg)
    try:
        pt_model.print_trainable_parameters()
    except Exception:
        pass
    return pt_model


def apply_prompt_tuning(
    model,
    num_virtual_tokens: int = 16,
    tokenizer=None,
    task_type: str = "CAUSAL_LM",
):
    peft = _ensure_peft()
    from peft import PromptTuningConfig, get_peft_model, TaskType

    cfg = PromptTuningConfig(
        task_type=getattr(TaskType, task_type),
        num_virtual_tokens=int(num_virtual_tokens),
        tokenizer_name_or_path=getattr(tokenizer, "name_or_path", None),
    )
    pt_model = get_peft_model(model, cfg)
    try:
        pt_model.print_trainable_parameters()
    except Exception:
        pass
    return pt_model

