"""KVWire v1 serialization (measured bytes on wire)."""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .int4 import pack_int4, unpack_int4, int4_byte_length

MAGIC = b"KVWIRE01"
HEADER_STRUCT = struct.Struct("<8sI")


@dataclass
class KVWireConfig:
    wire_index_dtype: str = "uint16"  # uint16 or uint32
    wire_scale_dtype: str = "fp16"    # fp16 or bf16
    wire_quant_mode: str = "int8"     # int8 or int4
    wire_scale_granularity: str = "per_block"  # per_tensor, per_head, per_block
    wire_include_headers: bool = True
    wire_version: str = "kvwire_v1"
    wire_compression: str = "none"


def _as_numpy(x: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    if isinstance(x, np.ndarray):
        arr = x
    else:
        try:
            import torch

            if torch.is_tensor(x):
                arr = x.detach().cpu().numpy()
            else:
                arr = np.asarray(x)
        except Exception:
            arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def _dtype_from_str(name: str) -> np.dtype:
    name = (name or "").lower()
    if name in ("uint16", "u16"):
        return np.uint16
    if name in ("uint32", "u32"):
        return np.uint32
    raise ValueError(f"Unsupported index dtype: {name}")


def _scale_dtype_from_str(name: str) -> np.dtype:
    name = (name or "").lower()
    if name in ("fp16", "float16"):
        return np.float16
    if name in ("bf16", "bfloat16"):
        # Use uint16 container for bf16 bytes.
        return np.uint16
    raise ValueError(f"Unsupported scale dtype: {name}")


def _float_to_bf16(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    bits = x.view(np.uint32)
    bf16 = (bits >> 16).astype(np.uint16)
    return bf16


def _bf16_to_float(x: np.ndarray) -> np.ndarray:
    bits = (x.astype(np.uint32) << 16)
    return bits.view(np.float32)


def _qmax(num_bits: int) -> int:
    return (2 ** (num_bits - 1)) - 1


def _quantize_tensor(x: np.ndarray, num_bits: int, granularity: str) -> Tuple[np.ndarray, np.ndarray]:
    if x.size == 0:
        return x.astype(np.int8), np.zeros((1,), dtype=np.float32)
    granularity = (granularity or "per_block").lower()
    qmax = _qmax(num_bits)
    x_fp32 = x.astype(np.float32)
    if granularity == "per_block":
        # Scale per (B,H,T)
        reduce_axes = (-1,)
        scale = np.max(np.abs(x_fp32), axis=reduce_axes, keepdims=True)
    elif granularity == "per_head":
        # Scale per (B,H)
        reduce_axes = (-1, -2)
        scale = np.max(np.abs(x_fp32), axis=reduce_axes, keepdims=True)
    elif granularity in ("per_tensor", "per_layer"):
        scale = np.max(np.abs(x_fp32))
        scale = np.array(scale, dtype=np.float32).reshape((1, 1, 1, 1))
    else:
        raise ValueError(f"Unsupported scale granularity: {granularity}")
    scale = np.clip(scale / float(qmax), 1e-6, None)
    q = np.round(x_fp32 / scale)
    q = np.clip(q, -qmax, qmax).astype(np.int8)
    return q, scale.astype(np.float32)


def _dequantize_tensor(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (q.astype(np.float32) * scale.astype(np.float32)).astype(np.float32)


def _serialize_scales(scale_k: np.ndarray, scale_v: np.ndarray, scale_dtype: str) -> Tuple[bytes, Dict[str, Any]]:
    dtype = _scale_dtype_from_str(scale_dtype)
    if scale_dtype.lower() in ("bf16", "bfloat16"):
        k_data = _float_to_bf16(scale_k)
        v_data = _float_to_bf16(scale_v)
    else:
        k_data = scale_k.astype(dtype)
        v_data = scale_v.astype(dtype)
    scales = np.concatenate([k_data.reshape(-1), v_data.reshape(-1)])
    meta = {
        "scale_dtype": scale_dtype,
        "scale_k_shape": list(scale_k.shape),
        "scale_v_shape": list(scale_v.shape),
    }
    return scales.tobytes(), meta


def _deserialize_scales(blob: bytes, meta: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    scale_dtype = meta.get("scale_dtype", "fp16")
    dtype = _scale_dtype_from_str(scale_dtype)
    raw = np.frombuffer(blob, dtype=dtype)
    k_shape = tuple(meta.get("scale_k_shape") or [])
    v_shape = tuple(meta.get("scale_v_shape") or [])
    k_size = int(np.prod(k_shape)) if k_shape else 0
    k_raw = raw[:k_size]
    v_raw = raw[k_size:]
    if scale_dtype.lower() in ("bf16", "bfloat16"):
        k = _bf16_to_float(k_raw).reshape(k_shape)
        v = _bf16_to_float(v_raw).reshape(v_shape)
    else:
        k = k_raw.astype(np.float32).reshape(k_shape)
        v = v_raw.astype(np.float32).reshape(v_shape)
    return k, v


def _build_header(meta: Dict[str, Any]) -> bytes:
    payload = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    return HEADER_STRUCT.pack(MAGIC, len(payload)) + payload


def _parse_header(blob: bytes) -> Tuple[Dict[str, Any], int]:
    if len(blob) < HEADER_STRUCT.size:
        raise ValueError("Blob too small for KVWire header")
    magic, length = HEADER_STRUCT.unpack_from(blob, 0)
    if magic != MAGIC:
        raise ValueError("Invalid KVWire magic")
    start = HEADER_STRUCT.size
    end = start + length
    meta = json.loads(blob[start:end].decode("utf-8")) if length else {}
    return meta, end


def pack(payload: Dict[str, Any], cfg: KVWireConfig) -> bytes:
    blob, _ = pack_with_breakdown(payload, cfg)
    return blob


def pack_with_breakdown(payload: Dict[str, Any], cfg: KVWireConfig) -> Tuple[bytes, Dict[str, Any]]:
    if payload is None:
        raise ValueError("payload is required")
    if cfg.wire_compression not in ("none", "", None):
        raise NotImplementedError("KVWire v1 compression is not implemented")
    k = _as_numpy(payload.get("k"))
    v = _as_numpy(payload.get("v"))
    if k.shape != v.shape:
        raise ValueError("K/V shape mismatch")
    if k.ndim != 4:
        raise ValueError("Expected K/V shape [B, H, T, D]")
    indices = payload.get("indices")
    if indices is None:
        indices = np.arange(k.shape[2], dtype=_dtype_from_str(cfg.wire_index_dtype))
    else:
        indices = _as_numpy(indices, dtype=_dtype_from_str(cfg.wire_index_dtype)).reshape(-1)
        if indices.size != k.shape[2]:
            raise ValueError("indices length must match token dimension")
        if indices.size > 1:
            order = np.argsort(indices, kind="stable")
            if not np.all(order == np.arange(indices.size)):
                indices = indices[order]
                k = k[:, :, order, :]
                v = v[:, :, order, :]

    quant_mode = (cfg.wire_quant_mode or "int8").lower()
    if quant_mode not in ("int8", "int4"):
        raise ValueError(f"Unsupported wire_quant_mode: {quant_mode}")

    num_bits = 8 if quant_mode == "int8" else 4
    qk, scale_k = _quantize_tensor(k, num_bits=num_bits, granularity=cfg.wire_scale_granularity)
    qv, scale_v = _quantize_tensor(v, num_bits=num_bits, granularity=cfg.wire_scale_granularity)

    if quant_mode == "int8":
        k_bytes = qk.astype(np.int8).tobytes()
        v_bytes = qv.astype(np.int8).tobytes()
    else:
        k_bytes = pack_int4(qk)
        v_bytes = pack_int4(qv)

    scale_blob, scale_meta = _serialize_scales(scale_k, scale_v, cfg.wire_scale_dtype)

    index_blob = indices.tobytes()
    meta = {
        "version": cfg.wire_version,
        "quant_mode": quant_mode,
        "index_dtype": cfg.wire_index_dtype,
        "scale_dtype": cfg.wire_scale_dtype,
        "scale_granularity": cfg.wire_scale_granularity,
        "k_shape": list(k.shape),
        "v_shape": list(v.shape),
        "indices_len": int(indices.size),
        "sections": {
            "indices": len(index_blob),
            "k_quant": len(k_bytes),
            "v_quant": len(v_bytes),
            "scales": len(scale_blob),
        },
        "scale_meta": scale_meta,
    }
    header = _build_header(meta) if cfg.wire_include_headers else b""
    blob = header + index_blob + k_bytes + v_bytes + scale_blob
    breakdown = {
        "header_bytes": len(header),
        "index_bytes": len(index_blob),
        "payload_bytes": len(k_bytes) + len(v_bytes),
        "scale_bytes": len(scale_blob),
        "total_bytes": len(blob),
    }
    return blob, breakdown


def unpack(blob: bytes, cfg: Optional[KVWireConfig] = None) -> Dict[str, Any]:
    if not blob:
        raise ValueError("Empty KVWire blob")
    cfg = cfg or KVWireConfig()
    meta, offset = _parse_header(blob)
    sections = meta.get("sections", {})
    idx_len = int(sections.get("indices", 0))
    k_len = int(sections.get("k_quant", 0))
    v_len = int(sections.get("v_quant", 0))
    s_len = int(sections.get("scales", 0))

    index_blob = blob[offset:offset + idx_len]
    offset += idx_len
    k_blob = blob[offset:offset + k_len]
    offset += k_len
    v_blob = blob[offset:offset + v_len]
    offset += v_len
    scale_blob = blob[offset:offset + s_len]

    idx_dtype = _dtype_from_str(meta.get("index_dtype", cfg.wire_index_dtype))
    indices = np.frombuffer(index_blob, dtype=idx_dtype)

    k_shape = tuple(meta.get("k_shape") or [])
    v_shape = tuple(meta.get("v_shape") or [])

    quant_mode = meta.get("quant_mode", cfg.wire_quant_mode)
    if quant_mode == "int8":
        k_quant = np.frombuffer(k_blob, dtype=np.int8).reshape(k_shape)
        v_quant = np.frombuffer(v_blob, dtype=np.int8).reshape(v_shape)
    elif quant_mode == "int4":
        k_quant = unpack_int4(k_blob, k_shape)
        v_quant = unpack_int4(v_blob, v_shape)
    else:
        raise ValueError(f"Unsupported quant_mode in blob: {quant_mode}")

    scale_meta = meta.get("scale_meta", {})
    scale_k, scale_v = _deserialize_scales(scale_blob, scale_meta)
    k = _dequantize_tensor(k_quant, scale_k)
    v = _dequantize_tensor(v_quant, scale_v)

    return {
        "meta": meta,
        "indices": indices,
        "k_quant": k_quant,
        "v_quant": v_quant,
        "scale_k": scale_k,
        "scale_v": scale_v,
        "k": k,
        "v": v,
    }


def measure_bytes(payload: Dict[str, Any], cfg: KVWireConfig) -> Dict[str, Any]:
    _, breakdown = pack_with_breakdown(payload, cfg)
    return breakdown


def estimate_payload_bytes(shape: Tuple[int, ...], quant_mode: str) -> int:
    if not shape:
        return 0
    num_elems = int(np.prod(shape))
    quant_mode = (quant_mode or "int8").lower()
    if quant_mode == "int8":
        return num_elems
    if quant_mode == "int4":
        return int4_byte_length(num_elems)
    raise ValueError(f"Unsupported quant mode: {quant_mode}")
