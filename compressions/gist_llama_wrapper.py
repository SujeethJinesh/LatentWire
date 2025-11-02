"""
Wrapper for Llama 3.1 8B that supports Gist attention masking.

This is the CRITICAL piece - we need to inject attention_mask_gist into the model.

Their approach (from gist_llama.py lines 536-542):
```python
attention_mask_gist_float = torch.full_like(
    attention_mask, torch.tensor(torch.finfo(attention_mask.dtype).min)
)
attention_mask_gist_float = attention_mask_gist_float.masked_fill(
    attention_mask_gist.bool(), 0.0
)
attention_mask = attention_mask + attention_mask_gist_float
```

We'll use monkey-patching to inject this without modifying transformers source.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast


class GistLlamaWrapper(nn.Module):
    """
    Wrapper around Llama that adds gist attention masking support.

    This is minimal - the official repo modifies the model source directly.
    For ASAP reproduction, we monkey-patch the forward() method.
    """

    def __init__(self, base_model: AutoModelForCausalLM):
        super().__init__()
        self.model = base_model
        self.config = base_model.config

        # Store original forward method
        self._original_model_forward = base_model.model.forward

        # Replace with our gist-aware version
        base_model.model.forward = self._gist_model_forward

    def _gist_model_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attention_mask_gist: Optional[torch.Tensor] = None,  # NEW: gist mask
    ):
        """
        Modified forward that accepts attention_mask_gist.

        This combines the standard causal attention_mask with gist masking.
        """
        # If gist mask provided, combine it with standard mask
        if attention_mask_gist is not None:
            # Convert gist mask to float
            # Their implementation (gist_llama.py lines 536-542):
            if attention_mask is None:
                # Create default causal mask
                batch_size, seq_length = input_ids.shape if input_ids is not None else inputs_embeds.shape[:2]
                attention_mask = torch.ones(
                    (batch_size, seq_length),
                    dtype=torch.long,
                    device=input_ids.device if input_ids is not None else inputs_embeds.device
                )

            # Prepare decoder attention mask
            # This is normally done inside model.forward(), but we need to intercept
            # The model's _prepare_decoder_attention_mask creates a causal mask
            # We need to apply it first, then add gist mask

            # Get the prepared causal mask from the model
            # (This is a bit hacky but works without modifying transformers source)
            combined_attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, input_ids.shape if input_ids is not None else inputs_embeds.shape[:2],
                inputs_embeds if inputs_embeds is not None else self.model.embed_tokens(input_ids),
                past_key_values[0][0].shape[2] if past_key_values is not None else 0
            )

            # Now add gist mask (their approach from lines 536-542)
            # attention_mask_gist is [batch, 1, seq, seq] with 1=attend, 0=don't attend
            # We need to convert to their format where 0=attend, -inf=don't
            attention_mask_gist_float = torch.full_like(
                combined_attention_mask,
                torch.finfo(combined_attention_mask.dtype).min
            )
            attention_mask_gist_float = attention_mask_gist_float.masked_fill(
                attention_mask_gist.bool(), 0.0
            )
            combined_attention_mask = combined_attention_mask + attention_mask_gist_float

            # Call original forward with combined mask
            return self._original_model_forward(
                input_ids=input_ids,
                attention_mask=combined_attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            # No gist mask, use original forward
            return self._original_model_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        """
        Prepare causal attention mask.
        Copied from LlamaModel implementation.
        """
        # Create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = self._make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = self._expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            expanded_attn_mask = expanded_attn_mask.to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    @staticmethod
    def _make_causal_mask(input_shape, dtype, device, past_key_values_length=0):
        """Make causal mask for self-attention."""
        bsz, tgt_len = input_shape
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)

        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    @staticmethod
    def _expand_mask(mask, dtype, tgt_len=None):
        """Expand attention mask from [bsz, seq_len] to [bsz, 1, tgt_seq_len, src_seq_len]."""
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attention_mask_gist: Optional[torch.Tensor] = None,  # NEW: gist mask
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass with optional gist attention masking.

        Args:
            attention_mask_gist: [batch, 1, seq, seq] gist attention mask
                Values: 1 = attend, 0 = don't attend
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward through model (with gist masking if provided)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            attention_mask_gist=attention_mask_gist,  # Pass gist mask
        )

        hidden_states = outputs[0]
        logits = self.model.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def save_pretrained(self, *args, **kwargs):
        """Delegate to base model."""
        return self.model.save_pretrained(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        """Load and wrap a pretrained model."""
        base_model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        return cls(base_model)


# ==============================================================================
# Usage Example
# ==============================================================================

if __name__ == "__main__":
    from transformers import AutoTokenizer
    from gist_minimal import make_gist_mask

    # Load model and tokenizer
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Add <GIST> token
    tokenizer.add_special_tokens({"additional_special_tokens": ["<GIST>"]})
    gist_token_id = tokenizer.additional_special_tokens_ids[-1]

    # Load wrapped model
    model = GistLlamaWrapper.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0"
    )

    # Resize for new token
    model.model.resize_token_embeddings(len(tokenizer))

    # Test gist masking
    text = "Instruction: Say hello\n<GIST>\nOutput: Hello!"
    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")

    # Create gist mask
    gist_mask = make_gist_mask(inputs['input_ids'], gist_token_id)

    # Forward with gist masking
    outputs = model(
        input_ids=inputs['input_ids'],
        attention_mask_gist=gist_mask,
    )

    print(f"Successfully tested gist masking!")
    print(f"Logits shape: {outputs.logits.shape}")
    print(f"Gist mask shape: {gist_mask.shape}")
