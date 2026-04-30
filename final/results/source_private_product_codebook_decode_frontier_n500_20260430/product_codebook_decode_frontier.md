# Source-Private Product-Codebook Decode Frontier

- pass gate: `True`
- rows: `3`
- pass rows: `3`
- remaps with pass: `[101, 103, 107]`
- max cached receiver p50 ms: `0.0212`
- max request-public table p50 ms: `0.3694`
- max resident table p50 ms: `0.01767`
- min cached speedup vs prior recorded: `17.377x`
- max prediction mismatch count: `0`
- max table prediction mismatch count: `0`

## Rows

| Remap | Budget | N | Functional pass | Prior p50 ms | Source packet kernel p50 | Request table p50 | Resident table p50 | Cached vector p50 | Batch p50 | Speedup vs prior | Mismatches | Pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | 4 | 500 | `True` | 0.360 | 0.2023 | 0.3670 | 0.01671 | 0.0207 | 0.00168 | 17.400x | 0/0 | `True` |
| 103 | 4 | 500 | `True` | 0.378 | 0.1335 | 0.3635 | 0.01654 | 0.0208 | 0.00162 | 18.185x | 0/0 | `True` |
| 107 | 4 | 500 | `True` | 0.368 | 0.1599 | 0.3694 | 0.01767 | 0.0212 | 0.00157 | 17.377x | 0/0 | `True` |

## Interpretation

The product-codebook packet gate failed the strict systems rule because the prior metric timed source packet construction and repeated target-side candidate feature hashing inside every row. This frontier isolates the receiver-side lookup that a real target would run after it already has public prompt/candidate state T, and adds a direct PQ distance-table path so the systems row matches product-quantization practice rather than full-vector reconstruction. A pass here does not claim end-to-end model inference speedup; it shows that the learned byte packet can be decoded as a low-latency table lookup once target side information is cached.

## Pass Rule

For every remapped codebook there must be at least one functionally passing product-codebook row whose cached and table-lookup target-side decoders exactly match the canonical decoder, whose request-public table decode has p50 <2 ms and p95 <5 ms, and whose resident lookup kernel has p50 <0.25 ms. Cold receiver decode, source packet construction, and public table construction are reported separately.
