import torch
import torch.nn.functional as F
from typing import Optional, List


@torch.no_grad()
def _teacher_step_logits_text(llm, scaffold_ids: torch.Tensor, gold_ids: torch.Tensor, t: int) -> torch.Tensor:
    """
    Teacher (text-prompted) logits at step t (predict token t of the answer),
    conditioned on [scaffold_ids, gold_ids[:t]] as input, predicting gold_ids[t].
    """
    device = next(llm.model.parameters()).device
    ids_t = torch.cat([scaffold_ids, gold_ids[:, :t+1]], dim=1).to(device)
    attn_mask = torch.ones_like(ids_t, dtype=torch.long, device=device)
    out = llm.model(input_ids=ids_t, attention_mask=attn_mask, use_cache=False, return_dict=True)
    return out.logits[:, -1, :]  # [B, V]


def _compose_student_inputs_from_prefix(
    llm,
    prefix_embeds: torch.Tensor,
    gold_ids: torch.Tensor,
    t: int,
    anchor_ids: Optional[List[int]],
    append_bos_after_prefix: Optional[bool],
) -> torch.Tensor:
    """
    Compose student inputs: [prefix] + optional [anchor] + optional [BOS] + teacher-forced gold[:t]
    Returns inputs_embeds for the LLM.
    """
    device = next(llm.model.parameters()).device
    emb_dtype = llm.input_embed.weight.dtype if hasattr(llm.input_embed, "weight") else None
    if emb_dtype is not None:
        prefix_embeds = prefix_embeds.to(device, dtype=emb_dtype)
    else:
        prefix_embeds = prefix_embeds.to(device)

    B = prefix_embeds.size(0)
    parts = [prefix_embeds]

    # optional anchor
    if anchor_ids and len(anchor_ids) > 0:
        anc = torch.tensor(anchor_ids, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)
        parts.append(llm.input_embed(anc))

    # BOS policy: None => auto (only if NO anchor)
    if append_bos_after_prefix is None:
        append_bos_after_prefix = not (anchor_ids and len(anchor_ids) > 0)
    if append_bos_after_prefix:
        bos_id = getattr(llm.tokenizer, "bos_token_id", None)
        if bos_id is not None:
            bos = torch.full((B, 1), int(bos_id), dtype=torch.long, device=device)
            parts.append(llm.input_embed(bos))

    # teacher-forced gold prefix up to t (exclusive of the to-be-predicted token)
    if t >= 0:
        tf = gold_ids[:, :t+1].to(device)
        parts.append(llm.input_embed(tf))

    return torch.cat(parts, dim=1)


def k_token_ce_from_prefix(
    llm,
    prefix_embeds: torch.Tensor,
    gold_ids: torch.Tensor,
    K: int = 4,
    anchor_ids: Optional[List[int]] = None,
    append_bos_after_prefix: Optional[bool] = None,
) -> torch.Tensor:
    """
    Auxiliary CE over the first K steps (teacher-forced).
    Average CE across t=0..K-1 (skip if sequence shorter).
    """
    device = next(llm.model.parameters()).device
    A = gold_ids.size(1)
    total = 0.0
    steps = 0

    for t in range(min(K, A)):
        inputs_embeds = _compose_student_inputs_from_prefix(
            llm, prefix_embeds, gold_ids, t, anchor_ids, append_bos_after_prefix
        )
        attn_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long, device=device)
        out = llm.model(inputs_embeds=inputs_embeds, attention_mask=attn_mask, use_cache=False, return_dict=True)
        logits = out.logits[:, -1, :]  # predict gold_ids[:, t]
        total = total + F.cross_entropy(logits.float(), gold_ids[:, t].to(device))
        steps += 1

    if steps == 0:
        return torch.zeros((), device=device)
    return total / float(steps)


def kd_first_k_prefix_vs_text(
    student_llm,
    teacher_llm,
    prefix_embeds: torch.Tensor,
    scaffold_ids: torch.Tensor,
    gold_ids: torch.Tensor,
    K: int = 4,
    tau: float = 1.0,
    anchor_ids: Optional[List[int]] = None,
    append_bos_after_prefix: Optional[bool] = None,
) -> torch.Tensor:
    """
    KL(student||teacher) over first K steps.
    Teacher: text prompt (scaffold_ids).
    Student: latent prefix (+ optional anchor/BOS) with teacher-forced gold[:t].
    """
    device = next(student_llm.model.parameters()).device
    total = 0.0
    steps = 0

    for t in range(min(K, gold_ids.size(1))):
        with torch.no_grad():
            T_logits = _teacher_step_logits_text(teacher_llm, scaffold_ids, gold_ids, t)
            T = F.softmax(T_logits / tau, dim=-1)

        inputs_embeds = _compose_student_inputs_from_prefix(
            student_llm, prefix_embeds, gold_ids, t, anchor_ids, append_bos_after_prefix
        )
        attn_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long, device=device)
        out = student_llm.model(inputs_embeds=inputs_embeds, attention_mask=attn_mask, use_cache=False, return_dict=True)
        S_logits = out.logits[:, -1, :]
        S_log = F.log_softmax(S_logits / tau, dim=-1)

        total = total + F.kl_div(S_log, T, reduction="batchmean") * (tau * tau)
        steps += 1

    if steps == 0:
        return torch.zeros((), device=device)
    return total / float(steps)
