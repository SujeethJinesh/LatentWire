# -*- coding: utf-8 -*-
"""Common utilities for LatentWire.

This module provides utilities for:
- Chat template formatting
- Text truncation
- Token handling
- Data preprocessing
"""

import re
from typing import List, Dict, Any, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


def apply_chat_template(
    messages: List[Dict[str, str]],
    tokenizer: Any,
    add_generation_prompt: bool = True
) -> str:
    """Apply chat template to messages.

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        tokenizer: Tokenizer with chat_template support
        add_generation_prompt: Whether to add generation prompt at the end

    Returns:
        Formatted chat string
    """
    if hasattr(tokenizer, 'apply_chat_template'):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt
        )
    else:
        # Fallback for tokenizers without chat template
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                formatted += f"System: {content}\n"
            elif role == "user":
                formatted += f"User: {content}\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}\n"

        if add_generation_prompt:
            formatted += "Assistant: "

        return formatted


def truncate_text(
    text: str,
    max_length: int,
    tokenizer: Optional[Any] = None,
    truncation_side: str = "right"
) -> str:
    """Truncate text to maximum length.

    Args:
        text: Input text to truncate
        max_length: Maximum length in tokens (if tokenizer provided) or characters
        tokenizer: Optional tokenizer for token-level truncation
        truncation_side: "left" or "right" truncation

    Returns:
        Truncated text
    """
    if tokenizer is not None:
        # Token-level truncation
        tokens = tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) <= max_length:
            return text

        if truncation_side == "left":
            tokens = tokens[-max_length:]
        else:  # right
            tokens = tokens[:max_length]

        return tokenizer.decode(tokens, skip_special_tokens=True)
    else:
        # Character-level truncation
        if len(text) <= max_length:
            return text

        if truncation_side == "left":
            return text[-max_length:]
        else:  # right
            return text[:max_length]


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def extract_answer_span(text: str, answer_prefix: str = "Answer:") -> Optional[str]:
    """Extract answer span from text.

    Args:
        text: Full text containing answer
        answer_prefix: Prefix marking the answer

    Returns:
        Extracted answer or None if not found
    """
    if answer_prefix in text:
        # Find the answer prefix
        idx = text.find(answer_prefix)
        # Extract everything after the prefix
        answer = text[idx + len(answer_prefix):].strip()

        # Stop at next question or double newline
        for delimiter in ["\n\n", "Question:", "Q:"]:
            if delimiter in answer:
                answer = answer[:answer.find(delimiter)].strip()

        return answer

    return None


def format_squad_example(
    context: str,
    question: str,
    answer: Optional[str] = None,
    include_answer_prefix: bool = True
) -> str:
    """Format a SQuAD example for training/evaluation.

    Args:
        context: Context paragraph
        question: Question about the context
        answer: Optional answer text
        include_answer_prefix: Whether to include "Answer:" prefix

    Returns:
        Formatted example string
    """
    formatted = f"Context: {context.strip()}\n\n"
    formatted += f"Question: {question.strip()}"

    if include_answer_prefix:
        formatted += "\n\nAnswer:"
        if answer is not None:
            formatted += f" {answer.strip()}"
    elif answer is not None:
        formatted += f"\n\n{answer.strip()}"

    return formatted


def get_special_token_ids(tokenizer: Any) -> Dict[str, Optional[int]]:
    """Get special token IDs from tokenizer.

    Args:
        tokenizer: Tokenizer instance

    Returns:
        Dictionary of special token names to IDs
    """
    special_tokens = {}

    # Common special tokens
    token_attrs = [
        ("bos_token_id", "bos_token"),
        ("eos_token_id", "eos_token"),
        ("pad_token_id", "pad_token"),
        ("unk_token_id", "unk_token"),
        ("sep_token_id", "sep_token"),
        ("cls_token_id", "cls_token"),
        ("mask_token_id", "mask_token"),
    ]

    for id_attr, token_attr in token_attrs:
        if hasattr(tokenizer, id_attr):
            special_tokens[id_attr] = getattr(tokenizer, id_attr)
        elif hasattr(tokenizer, token_attr):
            token = getattr(tokenizer, token_attr)
            if token is not None:
                special_tokens[id_attr] = tokenizer.convert_tokens_to_ids(token)
            else:
                special_tokens[id_attr] = None
        else:
            special_tokens[id_attr] = None

    return special_tokens


def create_attention_mask(
    input_ids: Any,
    pad_token_id: Optional[int] = None
) -> Any:
    """Create attention mask from input IDs.

    Args:
        input_ids: Input token IDs tensor
        pad_token_id: ID of padding token

    Returns:
        Attention mask tensor
    """
    if pad_token_id is None:
        # No padding, all tokens are attended to
        import torch
        return torch.ones_like(input_ids)
    else:
        # Mask padding tokens
        import torch
        return (input_ids != pad_token_id).long()


def batch_encode_texts(
    texts: List[str],
    tokenizer: Any,
    max_length: int = 512,
    padding: Union[bool, str] = True,
    truncation: bool = True,
    return_tensors: str = "pt"
) -> Dict[str, Any]:
    """Batch encode multiple texts.

    Args:
        texts: List of text strings
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        padding: Padding strategy
        truncation: Whether to truncate
        return_tensors: Format for returned tensors

    Returns:
        Dictionary with input_ids, attention_mask, etc.
    """
    return tokenizer(
        texts,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors=return_tensors
    )


def decode_tokens(
    token_ids: Any,
    tokenizer: Any,
    skip_special_tokens: bool = True,
    clean_up_tokenization_spaces: bool = True
) -> Union[str, List[str]]:
    """Decode token IDs to text.

    Args:
        token_ids: Token IDs (tensor or list)
        tokenizer: Tokenizer instance
        skip_special_tokens: Whether to skip special tokens
        clean_up_tokenization_spaces: Whether to clean up spaces

    Returns:
        Decoded text string(s)
    """
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=clean_up_tokenization_spaces
    )