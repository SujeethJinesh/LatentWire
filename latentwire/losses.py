"""Auxiliary loss utilities shared by training and evaluation."""

from __future__ import annotations

import os
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
    latent: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cross-entropy over the first `K` decoding steps under latent-prefix conditioning."""
    device = next(llm.model.parameters()).device
    total = torch.zeros((), device=device)
    steps = 0

    anchor_tensor = _maybe_to_anchor_tensor(anchor_ids, device)

    # Get pad token id for masking (critical for proper gradient computation)
    pad_id = getattr(getattr(llm, "tokenizer", None), "pad_token_id", None)
    ignore_index = int(pad_id) if pad_id is not None else -100

    # Request hidden states if using latent adapters
    output_hidden_states = llm.use_latent_adapters and latent is not None

    for t in range(min(K, gold_ids.size(1))):
        inputs_embeds, attn_mask, prepared_past = llm._compose_inputs_from_prefix(
            prefix_embeds,
            gold_ids[:, :t],
            anchor_ids=anchor_tensor,
            append_bos_after_prefix=append_bos_after_prefix,
            deep_prefix=deep_prefix_past,
        )
        out = llm.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            past_key_values=prepared_past,
            use_cache=bool(prepared_past),
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # Apply latent adapters if enabled
        if output_hidden_states and latent is not None:
            modified_hidden_states = llm._apply_latent_adapters(out.hidden_states, latent)
            final_hidden = modified_hidden_states[-1][:, -1, :]  # [B, d_model]
            logits = llm.model.lm_head(final_hidden)  # [B, vocab_size]
        else:
            logits = out.logits[:, -1, :]

        # Use ignore_index to properly mask PAD tokens
        # Move gold_ids to same device as logits (critical for multi-GPU models)
        target = gold_ids[:, t].to(logits.device)
        total = total + F.cross_entropy(logits, target, ignore_index=ignore_index, reduction="mean")
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

    max_teacher_steps = max(1, min(int(K), gold_ids_teacher.size(1)))
    gold_ids_teacher_trim = gold_ids_teacher[:, :max_teacher_steps]

    pad_id = getattr(teacher_llm.tokenizer, "pad_token_id", None)

    if scaffold_ids_teacher.size(1) <= 1 or gold_ids_teacher.size(1) <= 1:
        return torch.zeros((), device=student_device)

    # Use the new LMWrapper.disable_adapter() method for proper teacher cleanup
    disable_adapter_ctx = nullcontext()
    disable_adapter = getattr(teacher_llm, "disable_adapter", None)  # Changed from teacher_model to teacher_llm
    if callable(disable_adapter):
        try:
            disable_adapter_ctx = disable_adapter()
        except Exception:
            disable_adapter_ctx = nullcontext()

    max_batch_chunk = int(os.getenv("KD_TEACHER_CHUNK", "4"))
    max_batch_chunk = max(1, max_batch_chunk)

    teacher_logits_full = None
    with disable_adapter_ctx:
        with torch.no_grad():
            try:
                logits_chunks = []
                for start in range(0, scaffold_ids_teacher.size(0), max_batch_chunk):
                    end = min(scaffold_ids_teacher.size(0), start + max_batch_chunk)
                    ids_chunk = scaffold_ids_teacher[start:end]
                    gold_chunk = gold_ids_teacher_trim[start:end]
                    _, _, logits_chunk = teacher_llm.loss_with_text_prompt(
                        ids_chunk,
                        gold_chunk,
                        return_logits=True,
                        compute_loss=False,
                    )
                    logits_chunks.append(logits_chunk)
                teacher_logits_full = torch.cat(logits_chunks, dim=0)
            except RuntimeError as exc:
                print("[WARN] KD teacher forward failed; retrying per-example:", exc)
                teacher_logits_full = None
                logits_chunks = []
                fallback_failed = False
                for row in range(scaffold_ids_teacher.size(0)):
                    ids_row = scaffold_ids_teacher[row : row + 1]
                    gold_row = gold_ids_teacher_trim[row : row + 1]
                    try:
                        _, _, logits_row = teacher_llm.loss_with_text_prompt(
                            ids_row,
                            gold_row,
                            return_logits=True,
                            compute_loss=False,
                        )
                        logits_chunks.append(logits_row)
                    except RuntimeError as inner_exc:
                        print("[WARN] KD teacher per-example fallback failed on row; attempting CPU fallback:", inner_exc)
                        fallback_failed = True
                        break
                if not fallback_failed and logits_chunks:
                    teacher_logits_full = torch.cat(logits_chunks, dim=0)
                if teacher_logits_full is None:
                    cpu_success = False
                    try:
                        teacher_llm.model.to("cpu")
                        ids_cpu = scaffold_ids_teacher.to("cpu")
                        gold_cpu = gold_ids_teacher_trim.to("cpu")
                        _, _, logits_cpu = teacher_llm.loss_with_text_prompt(
                            ids_cpu,
                            gold_cpu,
                            return_logits=True,
                            compute_loss=False,
                        )
                        teacher_logits_full = logits_cpu.to(student_device)
                        cpu_success = True
                        print("[WARN] KD teacher CPU fallback succeeded")
                    except RuntimeError as cpu_exc:
                        print("[WARN] KD teacher CPU fallback failed; skipping KD for batch:", cpu_exc)
                        teacher_logits_full = None
                    finally:
                        try:
                            teacher_llm.model.to(teacher_device)
                        except RuntimeError as restore_exc:
                            print("[WARN] KD teacher restore failed; leaving on current device:", restore_exc)
                    if not cpu_success:
                        return torch.zeros((), device=student_device)

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

        chunk = max(1, int(os.getenv("KD_STATE_CHUNK", "4")))
        teacher_states = None
        try:
            hidden_chunks = None
            for start in range(0, teacher_inputs.size(0), chunk):
                end = min(teacher_inputs.size(0), start + chunk)
                inputs_chunk = teacher_inputs[start:end]
                mask_chunk = teacher_mask[start:end]
                out_chunk = model(
                    input_ids=inputs_chunk,
                    attention_mask=mask_chunk,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                if hidden_chunks is None:
                    hidden_chunks = [[] for _ in out_chunk.hidden_states]
                for idx_layer, tensor_layer in enumerate(out_chunk.hidden_states):
                    hidden_chunks[idx_layer].append(tensor_layer)
            if hidden_chunks is not None:
                teacher_states = tuple(torch.cat(chunks, dim=0) for chunks in hidden_chunks)
        except RuntimeError as exc:
            print("[WARN] KD state teacher chunked forward failed; retrying per-example:", exc)
            teacher_states = None

        if teacher_states is None:
            fallback_failed = False
            hidden_chunks = None
            for row in range(teacher_inputs.size(0)):
                try:
                    out_row = model(
                        input_ids=teacher_inputs[row : row + 1],
                        attention_mask=teacher_mask[row : row + 1],
                        output_hidden_states=True,
                        use_cache=False,
                        return_dict=True,
                    )
                    if hidden_chunks is None:
                        hidden_chunks = [[] for _ in out_row.hidden_states]
                    for idx_layer, tensor_layer in enumerate(out_row.hidden_states):
                        hidden_chunks[idx_layer].append(tensor_layer)
                except RuntimeError as inner_exc:
                    print("[WARN] KD state teacher per-example fallback failed; attempting CPU run:", inner_exc)
                    fallback_failed = True
                    break
            if not fallback_failed and hidden_chunks is not None:
                teacher_states = tuple(torch.cat(chunks, dim=0) for chunks in hidden_chunks)

        if teacher_states is None:
            try:
                original_device = next(model.parameters()).device
            except StopIteration:
                original_device = device
            try:
                model.to("cpu")
                teacher_inputs_cpu = teacher_inputs.to("cpu")
                teacher_mask_cpu = teacher_mask.to("cpu")
                out_cpu = model(
                    input_ids=teacher_inputs_cpu,
                    attention_mask=teacher_mask_cpu,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                teacher_states = tuple(t.to(device) for t in out_cpu.hidden_states)
                print("[WARN] KD state CPU fallback succeeded")
            except RuntimeError as cpu_exc:
                print("[WARN] KD state CPU fallback failed; skipping state KD for batch:", cpu_exc)
                teacher_states = None
            finally:
                try:
                    model.to(original_device)
                except Exception:
                    pass

        if teacher_states is None:
            return torch.zeros((), device=device)

        teacher_states = tuple(t.to(device) for t in teacher_states)

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
