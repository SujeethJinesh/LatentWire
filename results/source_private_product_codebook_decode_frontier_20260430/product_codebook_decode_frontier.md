# Source-Private Product-Codebook Decode Frontier

- pass gate: `True`
- rows: `9`
- pass rows: `8`
- remaps with pass: `[101, 103, 107]`
- max cached receiver p50 ms: `0.0257`
- max request-public table p50 ms: `0.4942`
- max resident table p50 ms: `0.02000`
- min cached speedup vs prior recorded: `371.893x`
- max prediction mismatch count: `0`
- max table prediction mismatch count: `0`

## Rows

| Remap | Budget | N | Functional pass | Prior p50 ms | Source packet kernel p50 | Request table p50 | Resident table p50 | Cached vector p50 | Batch p50 | Speedup vs prior | Mismatches | Pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | 2 | 256 | `True` | 10.285 | 0.0669 | 0.3599 | 0.01687 | 0.0228 | 0.00190 | 451.263x | 0/0 | `True` |
| 101 | 4 | 256 | `True` | 9.659 | 0.1711 | 0.4204 | 0.01846 | 0.0242 | 0.00194 | 399.685x | 0/0 | `True` |
| 101 | 6 | 256 | `True` | 9.944 | 0.1290 | 0.4942 | 0.01988 | 0.0257 | 0.00203 | 387.423x | 0/0 | `True` |
| 103 | 2 | 256 | `True` | 9.039 | 0.0734 | 0.3547 | 0.01675 | 0.0226 | 0.00198 | 400.257x | 0/0 | `True` |
| 103 | 4 | 256 | `True` | 9.415 | 0.1429 | 0.4117 | 0.01856 | 0.0242 | 0.00215 | 389.268x | 0/0 | `True` |
| 103 | 6 | 256 | `True` | 9.499 | 0.1316 | 0.4915 | 0.01975 | 0.0255 | 0.00214 | 371.893x | 0/0 | `True` |
| 107 | 2 | 256 | `False` | 8.313 | 0.0709 | 0.3519 | 0.01675 | 0.0221 | 0.00211 | 375.708x | 0/0 | `False` |
| 107 | 4 | 256 | `True` | 9.005 | 0.1758 | 0.4203 | 0.01838 | 0.0238 | 0.00210 | 378.478x | 0/0 | `True` |
| 107 | 6 | 256 | `True` | 9.759 | 0.1479 | 0.4911 | 0.02000 | 0.0255 | 0.00193 | 382.077x | 0/0 | `True` |

## Interpretation

The product-codebook packet gate failed the strict systems rule because the prior metric timed source packet construction and repeated target-side candidate feature hashing inside every row. This frontier isolates the receiver-side lookup that a real target would run after it already has public prompt/candidate state T, and adds a direct PQ distance-table path so the systems row matches product-quantization practice rather than full-vector reconstruction. A pass here does not claim end-to-end model inference speedup; it shows that the learned byte packet can be decoded as a low-latency table lookup once target side information is cached.

## Pass Rule

For every remapped codebook there must be at least one functionally passing product-codebook row whose cached and table-lookup target-side decoders exactly match the canonical decoder, whose request-public table decode has p50 <2 ms and p95 <5 ms, and whose resident lookup kernel has p50 <0.25 ms. Cold receiver decode, source packet construction, and public table construction are reported separately.
