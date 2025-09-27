"""Auxiliary loss utilities shared by training and evaluation."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from contextlib import nullcontext

import torch
import torch.nn.functional as F

__all__ = [
    "k_token_ce_from_prefix",
    "kd_first_k_prefix_vs_text",
    "kd_hidden_states_first_k",
]


def _maybe_to_anchor_tensor(
    anchor_ids: Optional[Sequence[int] | torch.Tensor],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if anchor_ids is None:
        return None
    if isinstance(anchor_ids, torch.Tensor):
        return anchor_ids.to(device=device, dtype=torch.long)
    if isinstance(anchor_ids, (list, tuple)):
        if len(anchor_ids) == 0:
            return None
        return torch.tensor(anchor_ids, dtype=torch.long, device=device)
    raise TypeError(f"Unsupported anchor_ids type: {type(anchor_ids)!r}")


def k_token_ce_from_prefix(
    llm,
    prefix_embeds: torch.Tensor,
    gold_ids: torch.Tensor,
    K: int = 4,
    anchor_ids: Optional[Sequence[int] | torch.Tensor] = None,
    append_bos_after_prefix: Optional[bool] = None,
    *,
    deep_prefix_past: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
) -> torch.Tensor:
    """Cross-entropy over the first `K` decoding steps under latent-prefix conditioning."""
    device = next(llm.model.parameters()).device
    total = torch.zeros((), device=device)
    steps = 0

    anchor_tensor = _maybe_to_anchor_tensor(anchor_ids, device)

    for t in range(min(K, gold_ids.size(1))):
        inputs_embeds, attn_mask, prepared_past = llm._compose_inputs_from_prefix(
            prefix_embeds,
            gold_ids[:, :t],
            anchor_ids=anchor_tensor,
            append_bos_after_prefix=append_bos_after_prefix,
            deep_prefix=deep_prefix_past,
        )
        logits = llm.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            past_key_values=prepared_past,
            use_cache=bool(prepared_past),
            return_dict=True,
        ).logits[:, -1, :]
        total = total + F.cross_entropy(logits, gold_ids[:, t], reduction="mean")
        steps += 1

    return total / max(steps, 1)


def kd_first_k_prefix_vs_text(
    student_llm,
    teacher_llm,
    prefix_embeds: torch.Tensor,
    scaffold_ids: torch.Tensor,
    gold_ids: torch.Tensor,
    K: int = 4,
    tau: float = 1.0,
    anchor_ids: Optional[Sequence[int] | torch.Tensor] = None,
    append_bos_after_prefix: Optional[bool] = None,
    *,
    deep_prefix_past: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
) -> torch.Tensor:
    """KL(student‖teacher) over first `K` steps with teacher conditioned on text prompts.

    We run the teacher once over the full text scaffold (plus answer tail) to obtain
    logits for all positions, then reuse those distributions for each latent step.
    This avoids repeated model dispatches—which were triggering CUDA launch failures
    under multi-GPU PEFT setups—and keeps the KD target stable.
    """

    student_device = next(student_llm.model.parameters()).device
    teacher_device = next(teacher_llm.model.parameters()).device
    anchor_tensor = _maybe_to_anchor_tensor(anchor_ids, student_device)

    scaffold_ids_teacher = scaffold_ids.to(teacher_device, non_blocking=True)
    gold_ids_teacher = gold_ids.to(teacher_device, non_blocking=True)

    pad_id = getattr(teacher_llm.tokenizer, "pad_token_id", None)

    if scaffold_ids_teacher.size(1) <= 1 or gold_ids_teacher.size(1) <= 1:
        return torch.zeros((), device=student_device)

    teacher_model = teacher_llm.model
    disable_adapter_ctx = nullcontext()
    disable_adapter = getattr(teacher_model, "disable_adapter", None)
    if callable(disable_adapter):
        try:
            disable_adapter_ctx = disable_adapter()
        except Exception:
            disable_adapter_ctx = nullcontext()

    teacher_logits_full = None
    with disable_adapter_ctx:
        with torch.no_grad():
            try:
                _loss, _n_tok, teacher_logits_full = teacher_llm.loss_with_text_prompt(
                    scaffold_ids_teacher,
                    gold_ids_teacher,
                    return_logits=True,
                    compute_loss=False,
                )
            except RuntimeError as exc:
                print("[WARN] KD teacher forward failed; skipping KD for this batch:", exc)
                torch.cuda.synchronize()
                teacher_logits_full = None

    if teacher_logits_full is None:
        return torch.zeros((), device=student_device)

    teacher_logits_full = teacher_logits_full.to(student_device)

    B, answer_len = gold_ids.size()
    scaffold_len = scaffold_ids_teacher.size(1)

    if pad_id is not None:
        pad_val = int(pad_id)
        prompt_mask = scaffold_ids_teacher.ne(pad_val)
        answer_mask = gold_ids_teacher.ne(pad_val)
        prompt_lengths = prompt_mask.long().sum(dim=1).to(student_device)
        answer_lengths = answer_mask.long().sum(dim=1).to(student_device)
    else:
        prompt_lengths = torch.full((B,), scaffold_len, dtype=torch.long, device=student_device)
        answer_lengths = torch.full((B,), answer_len, dtype=torch.long, device=student_device)

    prompt_lengths = prompt_lengths.clamp(min=1)

    total = torch.zeros((), device=student_device)
    steps = 0

    max_steps = min(int(K), answer_len)
    batch_indices = torch.arange(B, device=student_device)

    for t in range(max_steps):
        if answer_lengths.numel() == 0:
            break

        active_mask = answer_lengths > t
        if not active_mask.any():
            break

        gather_positions = prompt_lengths + t - 1
        gather_positions = gather_positions.clamp(min=0, max=teacher_logits_full.size(1) - 1)
        teacher_step_logits = teacher_logits_full[batch_indices, gather_positions, :]
        teacher_step_logits = teacher_step_logits / tau
        teacher_probs = F.softmax(teacher_step_logits, dim=-1)

        prev_tokens = gold_ids[:, :t].to(student_device, non_blocking=True)
        student_inputs, student_mask, prepared_past = student_llm._compose_inputs_from_prefix(
            prefix_embeds,
            prev_tokens,
            anchor_ids=anchor_tensor,
            append_bos_after_prefix=append_bos_after_prefix,
            deep_prefix=deep_prefix_past,
        )
        student_logits = student_llm.model(
            inputs_embeds=student_inputs,
            attention_mask=student_mask,
            past_key_values=prepared_past,
            use_cache=bool(prepared_past),
            return_dict=True,
        ).logits[:, -1, :]
        student_logits = student_logits.to(student_device)
        student_log_probs = F.log_softmax(student_logits / tau, dim=-1)

        if not active_mask.all():
            teacher_probs = teacher_probs[active_mask]
            student_log_probs = student_log_probs[active_mask]

        if teacher_probs.numel() == 0:
            continue

        total = total + F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (tau * tau)
        steps += 1

    if steps == 0:
        return torch.zeros((), device=student_device)

    return total / steps


def kd_hidden_states_first_k(
    wrapper,
    prefix_embeds: torch.Tensor,
    text_scaffold_ids: torch.Tensor,
    gold_ids: torch.Tensor,
    K: int = 4,
    layers: Tuple[int, ...] = (0, 1, 2),
    append_bos_after_prefix: Optional[bool] = None,
    anchor_ids: Optional[Sequence[int] | torch.Tensor] = None,
    *,
    deep_prefix_past: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
) -> torch.Tensor:
    """Match hidden states between latent and text runs for the first `K` decoding steps."""
    device = prefix_embeds.device
    model = wrapper.model
    total = torch.zeros((), device=device)
    steps = 0

    anchor_tensor = _maybe_to_anchor_tensor(anchor_ids, device)

    with torch.no_grad():
        teacher_inputs = torch.cat([text_scaffold_ids, gold_ids[:, :K]], dim=1)
        teacher_mask = torch.ones_like(teacher_inputs, dtype=torch.long, device=device)
        teacher_out = model(
            input_ids=teacher_inputs,
            attention_mask=teacher_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        teacher_states = teacher_out.hidden_states

    for t in range(min(K, gold_ids.size(1))):
        student_inputs, student_mask, prepared_past = wrapper._compose_inputs_from_prefix(
            prefix_embeds,
            gold_ids[:, :t],
            anchor_ids=anchor_tensor,
            append_bos_after_prefix=append_bos_after_prefix,
            deep_prefix=deep_prefix_past,
        )
        student_out = model(
            inputs_embeds=student_inputs,
            attention_mask=student_mask,
            output_hidden_states=True,
            past_key_values=prepared_past,
            use_cache=bool(prepared_past),
            return_dict=True,
        )
        student_states = student_out.hidden_states

        gather_index = text_scaffold_ids.size(1) + t
        for layer in layers:
            teacher_state = teacher_states[layer][:, gather_index, :]
            student_state = student_states[layer][:, -1, :]
            total = total + (student_state - teacher_state).pow(2).mean()
        steps += 1

    return total / max(steps, 1)
