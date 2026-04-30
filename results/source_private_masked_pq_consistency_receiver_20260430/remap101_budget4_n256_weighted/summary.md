# Source-Private Masked-PQ Consistency Receiver

- examples: `256`
- source packet pass: `True`
- pass gate vs deterministic L2: `False`
- learned matched accuracy: `0.582`
- deterministic L2 matched accuracy: `0.582`
- target-only accuracy: `0.250`
- best learned control: `wrong_codebook_packet` at `0.273`
- learned minus target: `0.332`
- learned minus deterministic L2: `0.000`

| Condition | Learned acc | L2 acc | Learned bytes | Learned p50 ms | L2 p50 ms |
|---|---:|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 0.00 | 0.7161 | 0.0008 |
| matched_product_codebook | 0.582 | 0.582 | 4.00 | 6.5619 | 0.0669 |
| zero_source | 0.250 | 0.250 | 0.00 | 0.7237 | 0.0008 |
| label_shuffled_ridge | 0.211 | 0.211 | 4.00 | 6.5726 | 0.0687 |
| constrained_shuffled_source | 0.113 | 0.113 | 4.00 | 6.2506 | 0.0667 |
| answer_masked_source | 0.254 | 0.254 | 4.00 | 6.7031 | 0.0673 |
| permuted_codes | 0.199 | 0.199 | 4.00 | 6.2477 | 0.0668 |
| wrong_codebook_packet | 0.273 | 0.273 | 4.00 | 6.1750 | 0.0666 |
| random_same_byte | 0.266 | 0.266 | 4.00 | 0.8237 | 0.0617 |
| structured_json_same_byte | 0.262 | 0.262 | 4.00 | 6.0073 | 0.0659 |
| structured_free_text_same_byte | 0.242 | 0.242 | 4.00 | 6.1193 | 0.0670 |
| target_derived_sidecar | 0.266 | 0.266 | 4.00 | 0.8380 | 0.0631 |

Pass rule: source_packet_pass requires learned matched PQ receiver to beat target-only by >=0.15 with all corrupt controls within target+0.05. pass_gate additionally requires learned matched accuracy to beat deterministic PQ L2 by >=0.03 at the same byte budget.
