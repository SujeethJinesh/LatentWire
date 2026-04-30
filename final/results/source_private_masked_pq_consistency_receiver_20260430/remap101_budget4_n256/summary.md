# Source-Private Masked-PQ Consistency Receiver

- examples: `256`
- source packet pass: `False`
- pass gate vs deterministic L2: `False`
- learned matched accuracy: `0.250`
- deterministic L2 matched accuracy: `0.582`
- target-only accuracy: `0.250`
- best learned control: `zero_source` at `0.250`
- learned minus target: `0.000`
- learned minus deterministic L2: `-0.332`

| Condition | Learned acc | L2 acc | Learned bytes | Learned p50 ms | L2 p50 ms |
|---|---:|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 0.00 | 0.6597 | 0.0007 |
| matched_product_codebook | 0.250 | 0.582 | 4.00 | 5.7290 | 0.0642 |
| zero_source | 0.250 | 0.250 | 0.00 | 0.6529 | 0.0007 |
| label_shuffled_ridge | 0.250 | 0.211 | 4.00 | 5.4087 | 0.0647 |
| constrained_shuffled_source | 0.250 | 0.113 | 4.00 | 5.2955 | 0.0650 |
| answer_masked_source | 0.250 | 0.254 | 4.00 | 5.9582 | 0.0651 |
| permuted_codes | 0.250 | 0.199 | 4.00 | 5.5288 | 0.0646 |
| wrong_codebook_packet | 0.250 | 0.273 | 4.00 | 5.4429 | 0.0651 |
| random_same_byte | 0.250 | 0.266 | 4.00 | 0.7566 | 0.0612 |
| structured_json_same_byte | 0.250 | 0.262 | 4.00 | 5.4919 | 0.0639 |
| structured_free_text_same_byte | 0.250 | 0.242 | 4.00 | 5.5317 | 0.0649 |
| target_derived_sidecar | 0.250 | 0.266 | 4.00 | 0.7671 | 0.0610 |

Pass rule: source_packet_pass requires learned matched PQ receiver to beat target-only by >=0.15 with all corrupt controls within target+0.05. pass_gate additionally requires learned matched accuracy to beat deterministic PQ L2 by >=0.03 at the same byte budget.
