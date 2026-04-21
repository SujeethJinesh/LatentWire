# Toy Hub Dictionary Bridge

Deterministic ablation for hub-and-spoke shared dictionaries versus quadratic pairwise bridges.

| Method | Examples | Accuracy | MSE | Atom recovery | Hub residual | Pairwise residual | Heldout frac | Pair seen frac | Adapters | Params | Bytes | Compute | Help | Harm | MSE help | MSE harm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| monolithic_bridge | 240 | 0.2542 | 1.0795 | 0.1958 | 1.0413 | 1.0795 | 0.3708 | 1.0000 | 1 | 420.0000 | 1680.0000 | 400.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| pairwise_bridges | 240 | 0.6792 | 0.4595 | 0.6542 | 0.4413 | 0.4595 | 0.3708 | 0.6292 | 20 | 8400.0000 | 33600.0000 | 400.0000 | 0.4250 | 0.0000 | 0.6292 | 0.0000 |
| hub_shared_dictionary | 240 | 1.0000 | 0.0199 | 1.0000 | 0.0179 | 0.0199 | 0.3708 | 1.0000 | 12 | 5240.0000 | 20980.0000 | 1000.0000 | 0.7458 | 0.0000 | 1.0000 | 0.0000 |
| held_out_family_transfer | 89 | 1.0000 | 0.0206 | 1.0000 | 0.0181 | 0.0206 | 1.0000 | 1.0000 | 12 | 5240.0000 | 20980.0000 | 1000.0000 | 0.8652 | 0.0000 | 1.0000 | 0.0000 |
| random_hub | 240 | 0.1875 | 1.9371 | 0.1000 | 1.8474 | 1.9371 | 0.3708 | 0.0000 | 12 | 5240.0000 | 20960.0000 | 800.0000 | 0.1125 | 0.1792 | 0.0583 | 0.9417 |
| oracle | 240 | 1.0000 | 0.0000 | 1.0000 | 0.0019 | 0.0000 | 0.3708 | 1.0000 | 0 | 0.0000 | 0.0000 | 20.0000 | 0.7458 | 0.0000 | 1.0000 | 0.0000 |

## Failure Tags

- `monolithic_bridge`: atom_confusion=0, heldout_pair_missing=0, high_hub_residual=109, high_pairwise_residual=16, lost_monolithic_correct=0, ok_correct=61, other=54
- `pairwise_bridges`: atom_confusion=0, heldout_pair_missing=77, high_hub_residual=0, high_pairwise_residual=0, lost_monolithic_correct=0, ok_correct=163, other=0
- `hub_shared_dictionary`: atom_confusion=0, heldout_pair_missing=0, high_hub_residual=0, high_pairwise_residual=0, lost_monolithic_correct=0, ok_correct=240, other=0
- `held_out_family_transfer`: atom_confusion=0, heldout_pair_missing=0, high_hub_residual=0, high_pairwise_residual=0, lost_monolithic_correct=0, ok_correct=89, other=0
- `random_hub`: atom_confusion=0, heldout_pair_missing=78, high_hub_residual=43, high_pairwise_residual=11, lost_monolithic_correct=34, ok_correct=45, other=29
- `oracle`: atom_confusion=0, heldout_pair_missing=0, high_hub_residual=0, high_pairwise_residual=0, lost_monolithic_correct=0, ok_correct=240, other=0

## Scaling

- Pairwise trained adapters: 20 of 30 ordered pairs.
- Hub adapters: 12 encoder/decoder adapters plus shared atom dictionary.

## Interpretation

Pairwise bridges fit seen ordered family pairs but cannot natively transfer through the held-out family. The hub dictionary pays one encoder and decoder per family, exposes atom recovery and hub residuals, and tests whether shared route atoms can replace quadratic bridge growth.

## Sources Consulted

- https://arxiv.org/abs/2602.15382
- https://arxiv.org/abs/2410.06981
- https://arxiv.org/abs/2511.03945
- https://arxiv.org/abs/2604.09360
