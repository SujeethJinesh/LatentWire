# Toy Symmetry Bridge

- seed: `0`
- dim: `16`
- train examples: `64`
- test examples: `64`

| Scenario | Method | Train MSE | Test MSE | Train Cos | Test Cos | Train R@1 | Test R@1 | Perm Acc | Perm Exact |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| identity | identity | 0.0015 | 0.0015 | 0.9996 | 0.9996 | 1.0000 | 1.0000 | - | - |
| identity | permutation_only | 0.0015 | 0.0015 | 0.9996 | 0.9996 | 1.0000 | 1.0000 | - | - |
| identity | orthogonal_procrustes | 0.0014 | 0.0017 | 0.9996 | 0.9995 | 1.0000 | 1.0000 | - | - |
| identity | permutation_plus_procrustes | 0.0014 | 0.0017 | 0.9996 | 0.9995 | 1.0000 | 1.0000 | - | - |
| identity | ridge_stitch | 0.0012 | 0.0020 | 0.9997 | 0.9994 | 1.0000 | 1.0000 | - | - |
| permutation | identity | 2.4453 | 2.2290 | 0.3624 | 0.3598 | 0.0156 | 0.0000 | - | - |
| permutation | permutation_only | 0.0014 | 0.0014 | 0.9996 | 0.9996 | 1.0000 | 1.0000 | 1.0000 | true |
| permutation | orthogonal_procrustes | 0.0012 | 0.0016 | 0.9997 | 0.9995 | 1.0000 | 1.0000 | - | - |
| permutation | permutation_plus_procrustes | 0.0012 | 0.0016 | 0.9997 | 0.9995 | 1.0000 | 1.0000 | 1.0000 | true |
| permutation | ridge_stitch | 0.0010 | 0.0019 | 0.9997 | 0.9995 | 1.0000 | 1.0000 | - | - |
| orthogonal_rotation | identity | 2.2623 | 2.4162 | 0.3587 | 0.3650 | 0.0000 | 0.0156 | - | - |
| orthogonal_rotation | permutation_only | 1.8269 | 2.0193 | 0.4731 | 0.4429 | 0.0938 | 0.0156 | - | - |
| orthogonal_rotation | orthogonal_procrustes | 0.0014 | 0.0019 | 0.9996 | 0.9995 | 1.0000 | 1.0000 | - | - |
| orthogonal_rotation | permutation_plus_procrustes | 0.0014 | 0.0019 | 0.9996 | 0.9995 | 1.0000 | 1.0000 | - | - |
| orthogonal_rotation | ridge_stitch | 0.0011 | 0.0022 | 0.9997 | 0.9994 | 1.0000 | 1.0000 | - | - |
| permutation_rotation | identity | 2.6164 | 2.4316 | 0.2623 | 0.3279 | 0.0156 | 0.0156 | - | - |
| permutation_rotation | permutation_only | 2.3454 | 2.1204 | 0.3090 | 0.3739 | 0.0000 | 0.0312 | 0.0000 | false |
| permutation_rotation | orthogonal_procrustes | 0.0013 | 0.0018 | 0.9996 | 0.9995 | 1.0000 | 1.0000 | - | - |
| permutation_rotation | permutation_plus_procrustes | 0.0013 | 0.0018 | 0.9996 | 0.9995 | 1.0000 | 1.0000 | 0.0000 | false |
| permutation_rotation | ridge_stitch | 0.0011 | 0.0021 | 0.9997 | 0.9994 | 1.0000 | 1.0000 | - | - |
| nonlinear_noise | identity | 1.7513 | 1.6794 | 0.1493 | 0.2206 | 0.0156 | 0.0156 | - | - |
| nonlinear_noise | permutation_only | 1.6374 | 1.6634 | 0.2138 | 0.2119 | 0.0156 | 0.0312 | - | - |
| nonlinear_noise | orthogonal_procrustes | 0.3835 | 0.4544 | 0.9273 | 0.9094 | 1.0000 | 1.0000 | - | - |
| nonlinear_noise | permutation_plus_procrustes | 0.3835 | 0.4544 | 0.9273 | 0.9094 | 1.0000 | 1.0000 | - | - |
| nonlinear_noise | ridge_stitch | 0.0336 | 0.0760 | 0.9700 | 0.9377 | 1.0000 | 1.0000 | - | - |

## Best by Scenario
- identity: best test MSE = `identity` (0.0015), best test recall = `identity` (1.0000)
- permutation: best test MSE = `permutation_only` (0.0014), best test recall = `permutation_only` (1.0000)
- orthogonal_rotation: best test MSE = `orthogonal_procrustes` (0.0019), best test recall = `orthogonal_procrustes` (1.0000)
- permutation_rotation: best test MSE = `orthogonal_procrustes` (0.0018), best test recall = `orthogonal_procrustes` (1.0000)
- nonlinear_noise: best test MSE = `ridge_stitch` (0.0760), best test recall = `orthogonal_procrustes` (1.0000)
