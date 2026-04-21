# Toy K/V Slot Mixed-Precision Bridge

This toy separates key transport from value transport.
Keys are responsible for route retrieval, values are responsible for answer content,
and an incoherent basis rotation probes whether low-bit quantization is sensitive to
coordinate alignment before reconstruction.

| Method | Basis | Route acc | Answer acc | K MSE | V MSE | Bytes estimate | Route help | Answer help |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| uniform_low_bit | none | 0.8281 | 0.8672 | 0.4586 | 0.5463 | 49152.0 | 0.0000 | 0.0000 |
| key_protected_value_low | none | 0.8672 | 0.9219 | 0.2004 | 0.5463 | 50694.0 | 0.0391 | 0.0547 |
| value_protected_key_low | none | 0.8281 | 0.8672 | 0.4586 | 0.1572 | 50182.0 | 0.0000 | 0.0000 |
| mixed_kv_precision | none | 0.8672 | 0.9219 | 0.2004 | 0.1572 | 51724.0 | 0.0391 | 0.0547 |
| incoherent_basis_rotation | hadamard | 0.8594 | 0.8906 | 0.1582 | 0.1578 | 49152.0 | 0.0312 | 0.0234 |

Interpretation:

The key-protected variant should primarily help route recovery, the value-protected
variant should primarily help answer reconstruction, the mixed K/V allocation should
offer the best combined tradeoff, and the basis rotation should change the quantization
geometry without changing the underlying task semantics.
